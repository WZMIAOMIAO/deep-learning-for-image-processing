from typing import List

import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.first_channels = 256
        self.img_shape = img_shape  # [C, H, W]

        self.conv_blocks = nn.Sequential(
            # 1x1 -> 4x4
            *self._create_block(latent_dim, self.first_channels, stride=1),

            # 4x4 -> 8x8
            *self._create_block(self.first_channels, self.first_channels // 2),

            # 8x8 -> 16x16
            *self._create_block(self.first_channels // 2, self.first_channels // 4),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.first_channels // 4, self.img_shape[0],
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    @staticmethod
    def _create_block(in_feat: int, out_feat: int, stride: int = 2):
        padding = 1 if stride == 2 else 0
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(noise)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape: List[int]):
        super().__init__()
        self.first_channels = 64
        self.img_shape = img_shape  # [C, H, W]
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            *self._create_clock(self.img_shape[0], self.first_channels, normlize=False),

            # 16x16 -> 8x8
            *self._create_clock(self.first_channels, self.first_channels * 2),

            # 8x8 -> 4x4
            *self._create_clock(self.first_channels * 2, self.first_channels * 4)
        )

        self.adv_layer = nn.Sequential(
            nn.Conv2d(self.first_channels * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    @staticmethod
    def _create_clock(in_feat: int, out_feat: int, normlize=True):
        layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=not normlize)]
        if normlize:
            layers.append(nn.BatchNorm2d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.model(imgs)
        x = self.adv_layer(x)
        validity = torch.flatten(x, start_dim=1)

        return validity
