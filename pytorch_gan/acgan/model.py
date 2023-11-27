from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int, img_shape: List[int]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape  # [C, H, W]
        self.first_feature_channels = 512
        self.init_size = [img_shape[1] // 2 ** 3, img_shape[2] // 2 ** 3]

        self.layer1 = nn.Linear(latent_dim + num_classes,
                                self.init_size[0] * self.init_size[1] * self.first_feature_channels)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.first_feature_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # 2X
            *self._create_clock(in_feat=self.first_feature_channels),

            # 4X
            *self._create_clock(in_feat=self.first_feature_channels // 2),

            # 8X
            *self._create_clock(in_feat=self.first_feature_channels // 4),

            nn.Conv2d(self.first_feature_channels // 8, img_shape[0], kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.apply(weights_init)

    @staticmethod
    def _create_clock(in_feat: int):
        out_feat = in_feat // 2
        layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_feat),
                  nn.LeakyReLU(0.2, inplace=True)]
        return layers

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]

        onehot_labels = F.one_hot(labels, num_classes=self.num_classes).to(dtype=noise.dtype, device=noise.device)
        x = torch.concat([noise, onehot_labels], dim=1)
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
            *self._create_clock(64, 128, drop_rate=0.4),
            *self._create_clock(128, 256, drop_rate=0.4),
            *self._create_clock(256, 512, stride=1, drop_rate=0.4),
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

        self.apply(weights_init)

    @staticmethod
    def _create_clock(in_feat: int, out_feat: int, stride: int = 2, normlize=True, drop_rate: float = 0.):
        layers = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=not normlize)]
        if normlize:
            layers.append(nn.BatchNorm2d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        return layers

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.model(imgs)
        x = torch.flatten(x, start_dim=1)
        validity = self.adv_layer(x)

        label = self.aux_layer(x)

        return validity, label
