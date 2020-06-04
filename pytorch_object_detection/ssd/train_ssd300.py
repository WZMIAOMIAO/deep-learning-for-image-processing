from src.ssd_model import SSD300, Backbone
from torch import nn
import torch
import transform
from my_dataset import VOC2012DataSet
import os
import train_utils.train_eval_utils as utils


def create_model(num_classes=21):
    backbone = Backbone(pretrain_path=None)
    model = SSD300(backbone=backbone, num_classes=num_classes)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train": transform.Compose([transform.SSDCropping(),
                                    transform.Resize(),
                                    transform.ColorJitter(),
                                    transform.ToTensor(),
                                    transform.RandomHorizontalFlip(),
                                    transform.Normalization(),
                                    transform.AssignGTtoDefaultBox()]),
        "val": transform.Compose([transform.Resize(),
                                  transform.ToTensor(),
                                  transform.Normalization()])
    }

    voc_path = "../"
    train_dataset = VOC2012DataSet(voc_path, data_transform['train'], True)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=2,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=utils.collate_fn)

    val_dataset = VOC2012DataSet(voc_path, data_transform['val'], False)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.collate_fn)

    model = create_model(num_classes=21)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    for epoch in range(10):
        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=1, warmup=True)

        lr_scheduler.step()

        utils.evaluate(model, val_data_loader, device=device)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/ssd300-{}.pth".format(epoch))

    # inputs = torch.rand(size=(2, 3, 300, 300))
    # output = model(inputs)
    # print(output)


if __name__ == '__main__':
    main()
