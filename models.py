"""
Arquiteturas de super-resolução.

SRCNN -> baseline simples (Dong et al., 2016): 3 camadas conv.
         Espera receber a imagem LR já upscalada via bicubic.
"""

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class SRCNN(nn.Module):
    """SRCNN original (Dong et al., 2016).

    Arquitetura: Conv(9) → ReLU → Conv(1) → ReLU → Conv(5)
    Entrada: imagem LR já upscalada via bicúbica
    """
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


class SRCNNAugmented(nn.Module):
    """SRCNN com arquitetura idêntica, mas treinada com data augmentation robusta.

    A diferença está no dataset (SRDatasetAugmented) usado durante treino.
    Isso aumenta a robustez sem mudar a arquitetura.
    """
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


class SRCNNMultiScale(nn.Module):
    """SRCNN que treina em múltiplas escalas (2x, 3x, 4x).

    Mantém a mesma arquitetura, mas durante treino recebe:
    - imagens upscaladas em diferentes escalas (2x, 3x, 4x)
    - criando um modelo mais generalista
    """
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


class SRCNNAugmentedMultiScale(nn.Module):
    """SRCNN com data augmentation robusta E múltiplas escalas.

    Combina o melhor dos dois mundos:
    - Dataset com augmentações diversas (blur, ruído, contraste)
    - Treino em escalas 2x, 3x, 4x
    - Modelo mais robusto e generalista
    """
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
    
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return (self.RDB3(self.RDB2(self.RDB1(x)))) * 0.2 + x

class ESRGANGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.HRconv(fea)))

class VGGDiscriminator(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 2, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1), nn.LeakyReLU(0.2, True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(nf * 4, 256), nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg19.features.children())[:35]).eval()
        for param in self.features.parameters(): 
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)