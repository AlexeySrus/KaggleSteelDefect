import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc=4, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 4, 9, 1, 0, bias=False),
            nn.Conv2d(4, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, nc=1, oc=4):
        super(Generator, self).__init__()
        self.generation_sequence = nn.Sequential(
            nn.Conv2d(nc, 4, 5),
            nn.ReLU(),
            nn.Conv2d(4, 12, 5),
            nn.ReLU(),
            nn.Conv2d(12, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=5),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, padding=5),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, padding=3),
            nn.ReLU(),
            nn.Conv2d(8, oc, 5, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.generation_sequence(x)
