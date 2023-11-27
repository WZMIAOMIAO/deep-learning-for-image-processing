from typing import List

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.img_shape = img_shape  # [C, H, W]
        self.first_feature_channels = 512
        self.init_size = [img_shape[1] // 2 ** 3, img_shape[2] // 2 ** 3]

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.layer1 = nn.Linear(latent_dim + num_classes,
                                self.init_size[0] * self.init_size[1] * self.first_feature_channels)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 2X
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(384, 192, kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 4X
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(192, 96, kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 8X
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(96, img_shape[0], kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]

        x = torch.concat([noise, self.label_emb(labels)], dim=1)
        x = self.layer1(x)
        x = x.reshape([b, self.first_feature_channels, self.init_size[0], self.init_size[1]])  # [B, C, H, W]
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape: List[int], num_classes: int):
        super().__init__()
        self.img_shape = img_shape  # [C, H, W]
        self.model = nn.Sequential(
            *self._create_clock(self.img_shape[0], 64, normlize=False),
            # *self._create_clock(32, 64, stride=1),
            *self._create_clock(64, 128),
            # *self._create_clock(128, 256, stride=1),
            *self._create_clock(128, 256),
            *self._create_clock(256, 512, stride=1),
        )

        # The height and width of downsampled image
        ds_size = [self.img_shape[1] // 2 ** 3, self.img_shape[2] // 2 ** 3]
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size[0] * ds_size[1], 1),
            nn.Sigmoid()
        )

        self.aux_layer = nn.Sequential(
            nn.Linear(512 * ds_size[0] * ds_size[1], num_classes),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def _create_clock(in_feat: int, out_feat: int, stride: int = 2, normlize=True):
        layers = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=not normlize)]
        if normlize:
            layers.append(nn.BatchNorm2d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.model(imgs)
        x = torch.flatten(x, start_dim=1)
        validity = self.adv_layer(x)

        label = self.aux_layer(x)

        return validity, label
