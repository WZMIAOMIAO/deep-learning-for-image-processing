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
        self.img_shape = img_shape  # [C, H, W]

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 2X
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 4X
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 8X
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 16X
            nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(noise)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape: List[int]):
        super().__init__()

        self.img_shape = img_shape  # [C, H, W]
        self.model = nn.Sequential(
            *self._create_clock(self.img_shape[0], 64, normlize=False),
            *self._create_clock(64, 128),
            *self._create_clock(128, 256),
            *self._create_clock(256, 512)
        )

        self.adv_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
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
