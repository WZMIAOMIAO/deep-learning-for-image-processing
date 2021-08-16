import random

import torch
import torchvision.transforms as t
from torchvision.transforms import functional as F

from src import dboxes300_coco, calc_iou_tensor, Encoder


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image).contiguous()
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            # bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


# This function is from https://github.com/chauhan-utk/ssd.DomainAdaptation.
class SSDCropping(object):
    """
    根据原文，对图像进行裁剪,该方法应放在ToTensor前
    Cropping for SSD, according to original paper
    Choose between following 3 conditions:
    1. Preserve the original image
    2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    3. Random crop
    Reference to https://github.com/chauhan-utk/src.DomainAdaptation
    """
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )
        self.dboxes = dboxes300_coco()

    def __call__(self, image, target):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:  # 不做随机裁剪处理
                return image, target

            htot, wtot = target['height_width']

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2:  # 保证宽高比例在0.5-2之间
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                # boxes的坐标是在0-1之间的
                bboxes = target["boxes"]
                ious = calc_iou_tensor(bboxes, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                # all(): Returns True if all elements in the tensor are True, False otherwise.
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                # 查找所有的gt box的中心点有没有在采样patch中的
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                # 如果所有的gt box的中心点都不在采样的patch中，则重新找
                if not masks.any():
                    continue

                # 修改采样patch中的所有gt box的坐标（防止出现越界的情况）
                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                # 虑除不在采样patch中的gt box
                bboxes = bboxes[masks, :]
                # 获取在采样patch中的gt box的标签
                labels = target['labels']
                labels = labels[masks]

                # 裁剪patch
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image.crop((left_idx, top_idx, right_idx, bottom_idx))

                # 调整裁剪后的bboxes坐标信息
                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                # 更新crop后的gt box坐标信息以及标签信息
                target['boxes'] = bboxes
                target['labels'] = labels

                return image, target


class Resize(object):
    """对图像进行resize处理,该方法应放在ToTensor前"""
    def __init__(self, size=(300, 300)):
        self.resize = t.Resize(size)

    def __call__(self, image, target):
        image = self.resize(image)
        return image, target


class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.trans(image)
        return image, target


class Normalization(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = t.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target


class AssignGTtoDefaultBox(object):
    """将DefaultBox与GT进行匹配"""
    def __init__(self):
        self.default_box = dboxes300_coco()
        self.encoder = Encoder(self.default_box)

    def __call__(self, image, target):
        boxes = target['boxes']
        labels = target["labels"]
        # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, target
