"""
refer to:
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
import os
import math
import argparse

from absl import logging
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.models.resnet import resnet34 as create_model
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate

logging.set_verbosity(logging.FATAL)


def export_onnx(model, onnx_filename, onnx_bs):
    model.eval()
    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    opset_version = 13

    print(f"Export ONNX file: {onnx_filename}")
    dummy_input = torch.randn(onnx_bs, 3, 224, 224).cuda()
    torch.onnx.export(model,
                      dummy_input,
                      onnx_filename,
                      verbose=False,
                      opset_version=opset_version,
                      enable_onnx_checker=False,
                      input_names=["input"],
                      output_names=["output"])


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (images, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(images.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cuda()


def main(args):
    quant_modules.initialize()
    assert torch.cuda.is_available(), "only support GPU!"

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
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

    # ########################## #
    # Post Training Quantization #
    # ########################## #
    # We will use histogram based calibration for activations and the default max calibration for weights.
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model = create_model(num_classes=args.num_classes)
    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.cuda()

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, val_loader, num_batches=1000 // batch_size)
        compute_amax(model, method="percentile", percentile=99.99)
        # validate
        evaluate(model=model, data_loader=val_loader, epoch=0)

    torch.save(model.state_dict(), "quant_model_calibrated.pth")

    if args.qat:
        # ########################### #
        # Quantization Aware Training #
        # ########################### #
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
        # Scheduler(half of a cosine period)
        lf = lambda x: (math.cos(x * math.pi / 2 / args.epochs)) * (1 - args.lrf) + args.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        for epoch in range(args.epochs):
            # train
            train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, epoch=epoch)

            scheduler.step()

            # validate
            evaluate(model=model, data_loader=val_loader, epoch=epoch)

    export_onnx(model, args.onnx_filename, args.onnx_bs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 训练好的权重路径
    parser.add_argument('--weights', type=str, default='./resNet(flower).pth',
                        help='trained weights path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--onnx-filename', default='resnet34.onnx', help='save onnx model filename')
    parser.add_argument('--onnx-bs', default=1, help='save onnx model batch size')
    parser.add_argument('--qat', type=bool, default=True, help='whether use quantization aware training')

    opt = parser.parse_args()

    main(opt)
