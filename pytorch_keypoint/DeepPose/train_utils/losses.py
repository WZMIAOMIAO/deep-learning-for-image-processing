import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.l1_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class SmoothL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.smooth_l1_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.mse_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class WingLoss(nn.Module):
    """refer https://github.com/TropComplique/wing-loss/blob/master/loss.py
    """
    def __init__(self, w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w * (1.0 - math.log(1.0 + w / epsilon))

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                wh_tensor: torch.Tensor,
                mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            wh_tensor [1, 1, 2]
            label [N, K, 2]
            mask [N, K]
        """
        delta = (pred - label).abs() * wh_tensor  # rel to abs
        losses = torch.where(condition=self.w > delta,
                             input=self.w * torch.log(1.0 + delta / self.epsilon),
                             other=delta - self.C)
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class SoftWingLoss(nn.Module):
    """refer mmpose/models/losses/regression_loss.py
    """
    def __init__(self, omega1: float = 2.0, omega2: float = 20.0, epsilon: float = 0.5) -> None:
        super().__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        self.B = omega1 - omega2 * math.log(1.0 + omega1 / epsilon)

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                wh_tensor: torch.Tensor,
                mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            wh_tensor [1, 1, 2]
            mask [N, K]
        """
        delta = (pred - label).abs() * wh_tensor  # rel to abs
        losses = torch.where(condition=delta < self.omega1,
                             input=delta,
                             other=self.omega2 * torch.log(1.0 + delta / self.epsilon) + self.B)
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        loss = torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)
        return loss
