import argparse
import datetime
import pickle

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset, init_distributed_mode, torch_distributed_zero_first


def main(opt, hyp):
    # 初始化各进程
    init_distributed_mode(opt)

    if opt.rank in [-1, 0]:
        print(opt)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=opt.name)

    device = torch.device(opt.device)
    if "cuda" not in device.type:
        raise EnvironmentError("not find GPU device for training.")

    # 使用DDP后会对每个device上的gradients取均值，所以需要放大学习率
    hyp["lr0"] *= max(1., opt.world_size * opt.batch_size / 64)

    wdir = "weights" + os.sep  # weights dir
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    # accumulate n times before optimizer update (bs 64)
    accumulate = max(round(64 / (opt.world_size * opt.batch_size)), 1)
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size
        if opt.rank in [-1, 0]:  # 只在第一个进程中显示打印信息
            print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    random.seed(0)  # 设置随机种子
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    if opt.rank in [-1, 0]:
        # Remove previous results
        for f in glob.glob(results_file) + glob.glob("tmp.pk"):
            os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)

    start_epoch = 0
    best_map = 0.0
    # 如果指定了预训练权重，则载入预训练权重
    if weights.endswith(".pt"):
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items()
                             if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        if opt.rank in [-1, 0]:
            # load results
            if ckpt.get("training_results") is not None:
                with open(results_file, "w") as file:
                    file.write(ckpt["training_results"])  # write results.txt

        # epochs
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        # 总共训练3x2=6个parameters
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)
    else:
        # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
        # 若要训练全部权重，删除以下代码
        darknet_end_layer = 74  # only yolov3spp cfg
        # Freeze darknet53 layers
        # 总共训练21x3+3x2=69个parameters
        for idx in range(darknet_end_layer + 1):  # [0, 74]
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # SyncBatchNorm
    # 如果只训练最后的predictor(其中不含bn层)，SyncBatchNorm没有作用
    if opt.freeze_layers is False:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
    model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(opt.rank):
        train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                            augment=True,
                                            hyp=hyp,  # augmentation hyperparameters
                                            rect=opt.rect,  # rectangular training
                                            cache_images=opt.cache_images,
                                            single_cls=opt.single_cls,
                                            rank=opt.rank)
        # 验证集的图像尺寸指定为img_size(512)
        val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                          hyp=hyp,
                                          cache_images=opt.cache_images,
                                          single_cls=opt.single_cls,
                                          rank=opt.rank)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if opt.rank in [-1, 0]:
        print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=nw,
        pin_memory=True, collate_fn=train_dataset.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=val_sampler, num_workers=nw,
        pin_memory=True, collate_fn=val_dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    # start training
    # caching val_data when you have plenty of memory(RAM)
    with torch_distributed_zero_first(opt.rank):
        if os.path.exists("tmp.pk") is False:
            coco = get_coco_api_from_dataset(val_dataset)
            with open("tmp.pk", "wb") as f:
                pickle.dump(coco, f)
        else:
            with open("tmp.pk", "rb") as f:
                coco = pickle.load(f)

    if opt.rank in [-1, 0]:
        print("starting traning for %g epochs..." % epochs)
        print('Using %g dataloader workers' % nw)

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_data_loader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True)
        # update scheduler
        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:
            # evaluate on the test dataset
            result_info = train_util.evaluate(model, val_data_loader,
                                              coco=coco, device=device)

            # only first process in DDP process to record info and save weights
            if opt.rank in [-1, 0]:
                coco_mAP = result_info[0]
                voc_mAP = result_info[1]
                coco_mAR = result_info[8]

                # write into tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                            "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                    for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # write into txt
                with open(results_file, "a") as f:
                    # 记录coco的12个指标加上训练总损失和lr
                    result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

                # update best mAP(IoU=0.50:0.95)
                if coco_mAP > best_map:
                    best_map = coco_mAP

                if opt.savebest is False:
                    # save weights every epoch
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
                else:
                    # only save best weights
                    if best_map == coco_mAP:
                        with open(results_file, 'r') as f:
                            save_files = {
                                'model': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'training_results': f.read(),
                                'epoch': epoch,
                                'best_map': best_map}
                            torch.save(save_files, best.format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if opt.rank in [-1, 0]:
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    main(opt, hyp)
