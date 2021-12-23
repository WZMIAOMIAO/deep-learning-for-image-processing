"""
该脚本能够把验证集中预测错误的图片挑选出来，并记录在record.txt中
"""
import os
import json
import argparse
import sys

import torch
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import read_split_data


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model.eval()
    with torch.no_grad():
        with open("record.txt", "w") as f:
            # validate
            data_loader = tqdm(val_loader, file=sys.stdout)
            for step, data in enumerate(data_loader):
                images, labels = data
                pred = model(images.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                contrast = torch.eq(pred_classes, labels.to(device)).tolist()
                labels = labels.tolist()
                pred_classes = pred_classes.tolist()
                for i, flag in enumerate(contrast):
                    if flag is False:
                        file_name = val_images_path[batch_size * step + i]
                        true_label = class_indict[str(labels[i])]
                        false_label = class_indict[str(pred_classes[i])]
                        f.write(f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default='./weights/model-19.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
