import torch
import torchvision
import transforms
from network_files.faster_rcnn_framework import FasterRCNN
from network_files.rpn_function import AnchorsGenerator
from backbone.mobilenetv2_model import MobileNetV2
from torchvision.ops import misc
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils
import os


def create_model(num_classes):
    backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = os.getcwd()
    # load train data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=utils.collate_fn)

    # load validation data set
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=utils.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=21)
    print(model)

    model.to(device)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    num_epochs = 5
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, print_freq=50)

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device)

    torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 冻结backbone部分底层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)
    num_epochs = 20
    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, print_freq=50, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device)

        # save weights
        if epoch > 10:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)
    # print(predictions)


if __name__ == "__main__":
    main()
