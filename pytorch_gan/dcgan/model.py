from typing import List

import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.img_shape = img_shape  # [C, H, W]
        self.init_size = [img_shape[1] // 4, img_shape[2] // 4]
        self.layer1 = nn.Linear(latent_dim, self.init_size[0] * self.init_size[1] * 128)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 2X
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 4X
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]
        x = self.layer1(noise)
        x = x.reshape([b, 128, self.init_size[0], self.init_size[1]])  # [B, C, H, W]
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape: List[int]):
        super().__init__()

        self.img_shape = img_shape  # [C, H, W]
        self.model = nn.Sequential(
            *self._create_clock(self.img_shape[0], 16, normlize=False),
            *self._create_clock(16, 32),
            *self._create_clock(32, 64),
            *self._create_clock(64, 128)
        )

        # The height and width of downsampled image
        ds_size = [self.img_shape[1] // 2 ** 4, self.img_shape[2] // 2 ** 4]
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size[0] * ds_size[1], 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    @staticmethod
    def _create_clock(in_feat: int, out_feat: int, normlize=True):
        layers = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=not normlize)]
        if normlize:
            layers.append(nn.BatchNorm2d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout2d(0.25))
        return layers

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.model(imgs)
        x = torch.flatten(x, start_dim=1)
        validity = self.adv_layer(x)

        return validity
