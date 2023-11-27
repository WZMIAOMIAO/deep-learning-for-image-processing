import os
import logging
from datetime import datetime

from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image


def create_logger(name: str = "train logger"):
    logger_fmt = logging.Formatter("%(asctime)s %(filename)s[%(lineno)d] %(levelname)s:%(message)s")
    logger = logging.Logger(name=name)

    # 将日志打印至终端
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logger_fmt)
    logger.addHandler(stream_handler)

    # 将日志输出至文档留存, 命名为时间戳
    now_str = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
    file_handler = logging.FileHandler(f"train_log_{now_str}.txt")
    file_handler.setFormatter(logger_fmt)
    logger.addHandler(file_handler)

    return logger


def record_args(logger: logging.Logger, args):
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")


def make_dirs(dirs: str):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_gen_imgs(imgs: torch.Tensor, save_num: int = 10, save_path: str = "gen_img.jpg"):
    b, c, h, w = imgs.shape
    save_num = min(b, save_num)

    imgs = imgs[:save_num]
    img_list = imgs.chunk(chunks=save_num, dim=0)
    imgs = torch.concat(img_list, dim=3)

    imgs.mul_(0.5).add_(0.5)

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_image(imgs, save_path)


def combine_imgs(img_path: str):
    img_list = []
    img_path_list = [i for i in os.listdir(img_path) if i.startswith("gen_img")]
    img_path_list = sorted(img_path_list, key=lambda x: int(x[:-4].split("_")[-1]))
    for i in img_path_list:
        img = Image.open(os.path.join(img_path, i))
        img_list.append(np.array(img))

    img = np.concatenate(img_list, axis=0)
    img = Image.fromarray(img)
    img.save(os.path.join(img_path, "combine_img.jpg"))


if __name__ == '__main__':
    combine_imgs("gen_imgs")
