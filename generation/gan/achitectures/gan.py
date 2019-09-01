import torch
import torch.nn as nn


# class SpectralNormalization(nn.Module):
#     def __init__(self, nno):
#         """
#         Apply spectral normalization to nn.Module object
#         Args:
#             nno: nn.Module object
#
#         Returns:
#         """
#         super(SpectralNormalization, self).__init__()
#         self.nno = nn.utils.spectral_norm(nno)
#
#     def forward(self, x):
#         return self.nno(x)


class Discriminator(nn.Module):
    def __init__(self, nc=4, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 4, 9, 1, 0, bias=False),
            nn.Conv2d(4, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, nc=1, oc=4, ngf=64):
        super(Generator, self).__init__()
        self.generation_sequence = nn.Sequential(
            nn.Conv2d(nc, ngf * 8, 5, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, ngf * 4, 5, padding=3, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 2, 5, padding=3, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.Conv2d(ngf * 2, ngf, 3, padding=3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 32, 5, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 5, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, oc, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.generation_sequence(x)
