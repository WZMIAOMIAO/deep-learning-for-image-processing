import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils

results_file = "/home/chaoc/Desktop/deep-learning-for-image-processing/results_file.txt"
def create_model(num_classes):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
        
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

device = torch.device("cuda")
print(device.type)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

data_set = "/home/chaoc/Desktop/deep-learning-for-image-processing/data_set"
# check voc root
if os.path.exists(data_set) is False:
    raise FileNotFoundError("data_set dose not in path:'{}'.".format(data_set))

# load train data set
# VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
train_dataset = VOCDataSet(data_transform["train"], 0)

# 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using %g dataloader workers' % nw)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)

# load validation data set
# VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
val_dataset = VOCDataSet(data_transform["val"], 1)
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=val_dataset.collate_fn)

# create model num_classes equal background + 20 classes
model = create_model(7)
# print(model)

model.to(device)

# define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.33)

train_loss = []
learning_rate = []
val_map = []

for epoch in range(0, 200):
    # train for one epoch, printing every 10 iterations
    mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                            device=device, epoch=epoch,
                                            print_freq=50, warmup=True)
    train_loss.append(mean_loss.item())
    learning_rate.append(lr)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    coco_info = utils.evaluate(model, val_data_set_loader, device=device)

    # write into txt
    with open(results_file, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")

    val_map.append(coco_info[1])  # pascal mAP

    # save weights
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    torch.save(save_files, "/home/chaoc/Desktop/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/save_weights/resNetFpn-model-{}.pth".format(epoch))

    print("ok{epoch}")

# plot loss and lr curve
if len(train_loss) != 0 and len(learning_rate) != 0:
    from plot_curve import plot_loss_and_lr
    plot_loss_and_lr(train_loss, learning_rate)

# plot mAP curve
if len(val_map) != 0:
    from plot_curve import plot_map
    plot_map(val_map)