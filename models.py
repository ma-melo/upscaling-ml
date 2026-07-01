"""
Arquiteturas de super-resolução.

SRCNN -> baseline simples (Dong et al., 2016): 3 camadas conv.
         Espera receber a imagem LR já upscalada via bicubic.
"""

import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EDSRBaseline(nn.Module):
    def __init__(self, num_features=64, num_blocks=16, scale=4):
        super().__init__()
        self.scale = scale
        self.num_features = num_features

        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        self.tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = self._make_upsampler(scale)

        self.out = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def _make_upsampler(self, scale):
        if scale == 4:
            return nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.num_features, self.num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            return nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            raise NotImplementedError(f"Scale {scale} not supported")

    def forward(self, x):
        x = self.head(x)

        res = x
        x = self.body(x)
        x = self.tail(x)
        x = x + res

        x = self.upsample(x)
        x = self.out(x)

        return x