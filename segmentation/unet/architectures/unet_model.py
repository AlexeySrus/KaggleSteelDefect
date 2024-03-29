# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class MultiUNet(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super(MultiUNet, self).__init__()
        self.unet1 = UNet(n_channels, 1)
        self.unet2 = UNet(n_channels, 1)
        self.unet3 = UNet(n_channels, 1)
        self.unet4 = UNet(n_channels, 1)

    def forward(self, x):
        y1 = self.unet1(x)
        y2 = self.unet2(x)
        y3 = self.unet3(x)
        y4 = self.unet4(x)

        return torch.cat((y1, y2, y3, y4), dim=1)