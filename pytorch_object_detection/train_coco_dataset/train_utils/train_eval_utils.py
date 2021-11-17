import math
import sys
import time
import json

import torch
from pycocotools.cocoeval import COCOeval

import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco91to80 = data_loader.dataset.coco91to80
    coco80to91 = dict([(str(v), k) for k, v in coco91to80.items()])
    results = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            per_image_boxes = output["boxes"]
            # 对于coco_eval, 需要的每个box的数据格式为[x_min, y_min, w, h]
            # 而我们预测的box格式是[x_min, y_min, x_max, y_max]，所以需要转下格式
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]
            per_image_classes = output["labels"]
            per_image_scores = output["scores"]

            # 遍历每个目标的信息
            for object_score, object_class, object_box in zip(
                    per_image_scores, per_image_classes, per_image_boxes):
                object_score = float(object_score)
                # 要将类别信息还原回coco91中
                coco80_class = int(object_class)
                coco91_class = int(coco80to91[str(coco80_class)])
                # We recommend rounding coordinates to the nearest tenth of a pixel
                # to reduce resulting JSON file size.
                object_box = [round(b, 2) for b in object_box.tolist()]

                res = {"image_id": img_id,
                       "category_id": coco91_class,
                       "bbox": object_box,
                       "score": round(object_score, 3)}
                results.append(res)

        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    all_results = utils.all_gather(results)

    if utils.is_main_process():
        # 将所有进程上的数据合并到一个list当中
        results = []
        for res in all_results:
            results.extend(res)

        # write predict results into json file
        json_str = json.dumps(results, indent=4)
        with open('predict_tmp.json', 'w') as json_file:
            json_file.write(json_str)

        # accumulate predictions from all images
        coco_true = data_loader.dataset.coco
        coco_pre = coco_true.loadRes('predict_tmp.json')

        coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        coco_info = coco_evaluator.stats.tolist()  # numpy to list
    else:
        coco_info = None

    return coco_info
