from typing import List

import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.img_shape = img_shape  # [C, H, W]
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            *self._create_clock(latent_dim + num_classes, 128, normlize=False),
            *self._create_clock(128, 256),
            *self._create_clock(256, 512),
            *self._create_clock(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]
        x = torch.concat([noise, self.label_emb(labels)], dim=1)
        x = self.model(x)
        x = x.reshape(b, *self.img_shape)  # [B, C, H, W]
        return x

    @staticmethod
    def _create_clock(in_feat: int, out_feat: int, normlize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normlize:
            layers.append(nn.BatchNorm1d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, img_shape: List[int]):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = imgs.flatten(start_dim=1)
        x = torch.concat([x, self.label_emb(labels)], dim=1)
        validity = self.model(x)

        return validity
