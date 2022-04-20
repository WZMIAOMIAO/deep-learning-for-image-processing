import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
root = "/home/wz/my_github/data_set/flower_data/flower_photos"  # 数据集所在根目录


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images, labels = data


if __name__ == '__main__':
    main()
