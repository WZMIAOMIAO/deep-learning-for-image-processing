"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
import torchvision
from tqdm import tqdm
import numpy as np
from pycocotools.cocoeval import COCOeval

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import CocoDetection
from backbone import vgg


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = './coco80_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    category_index = json.load(json_file)
    json_file.close()

    coco_root = parser_data.data_path

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = CocoDetection(coco_root, "val", data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除feature中最后的maxpool层
    backbone.out_channels = 512

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    # num_classes equal 80 + background classes
    model = FasterRCNN(backbone=backbone,
                       num_classes=parser_data.num_classes + 1,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    # print(model)

    model.to(device)

    # evaluate on the val dataset
    cpu_device = torch.device("cpu")
    coco91to80 = val_dataset.coco91to80
    coco80to91 = dict([(str(v), k) for k, v in coco91to80.items()])
    results = []

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

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

    # accumulate predictions from all images
    # write predict results into json file
    json_str = json.dumps(results, indent=4)
    with open('predict_tmp.json', 'w') as json_file:
        json_file.write(json_str)

    # accumulate predictions from all images
    coco_true = val_dataset.coco
    coco_pre = coco_true.loadRes('predict_tmp.json')

    coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_evaluator)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_evaluator, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[str(i + 1)], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='80', help='number of classes')

    # 数据集的根目录(coco2017根目录)
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights', default='./save_weights/model.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
