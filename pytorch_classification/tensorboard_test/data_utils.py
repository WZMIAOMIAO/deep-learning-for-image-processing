import os
import json
import pickle
import random

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def plot_class_preds(net,
                     images_dir: str,
                     transform,
                     num_plot: int = 5,
                     device="cpu"):
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    label_path = os.path.join(images_dir, "label.txt")
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r')
    # {"0": "daisy"}
    flower_class = json.load(json_file)
    # {"daisy": "0"}
    class_indices = dict((v, k) for k, v in flower_class.items())

    # reading label.txt file
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert len(split_info) == 2, "label format error, expect file_name and class_name"
                image_name, class_name = split_info
                image_path = os.path.join(images_dir, image_name)
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)
        images.append(img)
        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)
    for i in range(num_imgs):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(1, num_imgs, i+1, xticks=[], yticks=[])

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        plt.imshow(npimg.astype('uint8'))

        title = "{}, {:.2f}%\n(label: {})".format(
            flower_class[str(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            flower_class[str(labels[i])]  # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

    return fig


