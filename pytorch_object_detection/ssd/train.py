from src.ssd_model import SSD300, Backbone
from src.res50_backbone import resnet50
from torch import nn
import torch


def create_model(num_classes=21):
    backbone = Backbone(pretrain_path=None)
    model = SSD300(backbone=backbone, num_classes=num_classes)
    return model


def main():
    model = create_model(num_classes=21)
    inputs = torch.rand(size=(2, 3, 300, 300))
    output = model(inputs)
    print(output)


if __name__ == '__main__':
    main()
