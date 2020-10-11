import argparse

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils.datasets import *
from utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils.coco_utils import get_coco_api_from_dataset

wdir = "weights" + os.sep  # weights dir
last = wdir + "last.pt"
best = wdir + "best.pt"
results_file = "results.txt"

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.00001,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
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
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    # init_seeds()  # 初始化随机种子，保证结果可复现
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # remove previous results
    for f in glob.glob("*_batch*.jpg") + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)

    # optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if ".bias" in k:
            pg2 += [v]  # biases (bn biases and conv2d biases)
        elif "Conv2d.weight" in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else, (bn weight)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"])
    else:
        optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    if weights.endswith(".pt"):
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

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

    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * 0.99 + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # Plot lr schedule
    # y = []
    # for _ in range(epochs+20):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # model.yolo_layers = model.module.yolo_layers

    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    train_dataset = LoadImageAndLabels(train_path, imgsz_train, batch_size,
                                       augment=True,
                                       hyp=hyp,  # augmentation hyperparameters
                                       rect=opt.rect,  # rectangular training
                                       cache_images=opt.cache_images,
                                       single_cls=opt.single_cls)

    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = LoadImageAndLabels(test_path, imgsz_test, batch_size,
                                     hyp=hyp,
                                     rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                     cache_images=opt.cache_images,
                                     single_cls=opt.single_cls)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 计算每个类别的目标个数，并计算每个类别的比重
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # start training
    nb = len(train_dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    # caching val_data when you have plenty of memory(RAM)
    print("caching val_data for evaluation.")
    coco = get_coco_api_from_dataset(val_dataset)
    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    for epoch in range(start_epoch, epochs):
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               batch_size=batch_size,
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True)
        # update scheduler
        scheduler.step()

        # evaluate on the test dataset
        result_info = train_util.evaluate(model, val_datasetloader,
                                          coco=coco, device=device)

        # write into tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                    "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                tb_writer.add_scalar(tag, x, epoch)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./weights/yolov3spp-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    opt = parser.parse_args()
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    print(opt)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
