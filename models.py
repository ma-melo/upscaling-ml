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


# Espaço reservado para a próxima fase do trabalho:
#
# class EDSRBaseline(nn.Module):
#     """EDSR sem BatchNorm, 16 blocos residuais, 64 filtros (Lim et al., 2017)."""
#     ...