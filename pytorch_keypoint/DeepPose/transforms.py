import math
import random
from typing import Tuple

import cv2
import torch
import numpy as np

from wflw_horizontal_flip_indices import wflw_flip_indices_dict


def adjust_box(xmin: int, ymin: int, xmax: int, ymax: int, fixed_size: Tuple[int, int]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    w = xmax - xmin
    h = ymax - ymin

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def affine_points_np(keypoint: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Args:
        keypoint [k, 2]
        m [2, 3]
    """
    ones = np.ones((keypoint.shape[0], 1), dtype=np.float32)
    keypoint = np.concatenate([keypoint, ones], axis=1)  # [k, 3]
    new_keypoint = np.matmul(keypoint, m.T)
    return new_keypoint


def affine_points_torch(keypoint: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Args:
        keypoint [n, k, 2]
        m [n, 2, 3]
    """
    dtype = keypoint.dtype
    device = keypoint.device

    n, k, _ = keypoint.shape
    ones = torch.ones(size=(n, k, 1), dtype=dtype, device=device)
    keypoint = torch.concat([keypoint, ones], dim=2)  # [n, k, 3]
    new_keypoint = torch.matmul(keypoint, m.transpose(1, 2))
    return new_keypoint


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, image: np.ndarray, target):
        image = cv2.resize(image, dsize=(self.w, self.h), fx=0, fy=0,
                           interpolation=cv2.INTER_LINEAR)

        return image, target


class ToTensor(object):
    """将opencv图像转为Tensor, HWC2CHW, 并缩放数值至0~1"""
    def __call__(self, image, target):
        image = torch.from_numpy(image).permute((2, 0, 1))
        image = image.to(torch.float32) / 255.

        if "ori_keypoint" in target and "keypoint" in target:
            target["ori_keypoint"] = torch.from_numpy(target["ori_keypoint"])
            target["keypoint"] = torch.from_numpy(target["keypoint"])
        target["m_inv"] = torch.from_numpy(target["m_inv"])
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).reshape((3, 1, 1))
        self.std = torch.as_tensor(std, dtype=torch.float32).reshape((3, 1, 1))

    def __call__(self, image: torch.Tensor, target: dict):
        image.sub_(self.mean).div_(self.std)

        if "keypoint" in target:
            _, h, w = image.shape
            keypoint = target["keypoint"]
            keypoint[:, 0] /= w
            keypoint[:, 1] /= h
            target["keypoint"] = keypoint
        return image, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转"""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.wflw_flip_ids = list(wflw_flip_indices_dict.values())

    def __call__(self, image: np.ndarray, target: dict):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))

            # [k, 2]
            if "keypoint" in target:
                _, w, _ = image.shape
                keypoint: torch.Tensor = target["keypoint"]
                keypoint = keypoint[self.wflw_flip_ids]
                keypoint[:, 0] = w - keypoint[:, 0]
                target["keypoint"] = keypoint

        return image, target


class AffineTransform(object):
    """shift+scale+rotation"""
    def __init__(self,
                 scale_factor: Tuple[float, float] = (0.65, 1.35),
                 scale_prob: float = 1.,
                 rotate: int = 45,
                 rotate_prob: float = 0.6,
                 shift_factor: float = 0.15,
                 shift_prob: float = 0.3,
                 fixed_size: Tuple[int, int] = (256, 256)):
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate = rotate
        self.rotate_prob = rotate_prob
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.fixed_size = fixed_size  # (h, w)

    def __call__(self, img: np.ndarray, target: dict):
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin

        if random.random() < self.shift_prob:
            shift_w_factor = random.uniform(-self.shift_factor, self.shift_factor)
            shift_h_factor = random.uniform(-self.shift_factor, self.shift_factor)
            src_xmin -= int(src_w * shift_w_factor)
            src_xmax -= int(src_w * shift_w_factor)
            src_ymin -= int(src_h * shift_h_factor)
            src_ymax -= int(src_h * shift_h_factor)

        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2], dtype=np.float32)
        src_p2 = src_center + np.array([0, -src_h / 2], dtype=np.float32)  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0], dtype=np.float32)   # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2], dtype=np.float32)
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0], dtype=np.float32)  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2], dtype=np.float32)  # right middle

        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_factor)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2], dtype=np.float32)  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0], dtype=np.float32)   # right middle

        if random.random() < self.rotate_prob:
            angle = random.randint(-self.rotate, self.rotate)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle),
                                            -src_h / 2 * math.cos(angle)], dtype=np.float32)
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle),
                                            src_w / 2 * math.sin(angle)], dtype=np.float32)

        src = np.stack([src_center, src_p2, src_p3])
        dst = np.stack([dst_center, dst_p2, dst_p3])

        m = cv2.getAffineTransform(src, dst).astype(np.float32)  # 计算正向仿射变换矩阵
        m_inv = cv2.getAffineTransform(dst, src).astype(np.float32)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        warp_img = cv2.warpAffine(src=img,
                                  M=m,
                                  dsize=tuple(self.fixed_size[::-1]),  # [w, h]
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0),
                                  flags=cv2.INTER_LINEAR)

        if "keypoint" in target:
            keypoint = target["keypoint"]
            keypoint = affine_points_np(keypoint, m)
            target["keypoint"] = keypoint

        # from utils import draw_keypoints
        # keypoint[:, 0] /= self.fixed_size[1]
        # keypoint[:, 1] /= self.fixed_size[0]
        # draw_keypoints(warp_img, keypoint, "affine.jpg", 2, is_rel=True)

        target["m"] = m
        target["m_inv"] = m_inv
        return warp_img, target
