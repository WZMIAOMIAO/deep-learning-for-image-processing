import os
import sys
import json
import argparse
import math

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from tqdm import tqdm

from my_dataset import MyDataSet
from model import GoogLeNet
from utils import read_split_data, train_one_epoch, evaluate,get_params_groups


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(args)
    if os.path.exists("./logs/weights") is False:
        os.makedirs("./logs/weights")
    tb_writer = SummaryWriter('logs')

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    num_classes = args.num_classes
    # model = GoogLeNet(num_classes=num_classes, aux_logits=False, init_weights=True)


    # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
    # 官方的模型中使用了bn层以及改了一些参数，不能混用
    import torchvision
    model = torchvision.models.googlenet(num_classes=6)
    model_dict = model.state_dict()
    # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
    pretrain_model = torch.load("./googlenet.pth")
    del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
                "aux2.fc2.weight", "aux2.fc2.bias",
                "fc.weight", "fc.bias"]
    pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    # add graph
    init_img = torch.zeros((1, 3, 224, 224))
    tb_writer.add_graph(model, init_img)
    loss_function = nn.CrossEntropyLoss()

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=args.wd)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):

        # train
        train_loss= train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./logs/weights/best_model.pth")
            best_acc = val_acc

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./resize640")
    parser.add_argument('--model-name', default='googlenet', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='/content/gdrive/MyDrive/deep-learning-for-image-processing/model_data/vgg16-397923af.pth',
                        help='initial weights path')
    # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
