import torch
import torch.nn as nn
from torch.nn import *
from generation.gan.utils.tensor_utils import crop_batch_by_center


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)


class BasicBlock(nn.Module):
    """Basic block without padding from ResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.InstanceNorm2d(planes)
        # self.bn1 = nn.LocalResponseNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.InstanceNorm2d(planes)
        # self.bn2 = nn.LocalResponseNorm(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += crop_batch_by_center(residual, out.shape)
        out = self.relu(out)

        return out


class OneImageNetModel(nn.Module):
    def __init__(self, output_filters=3,
                 residuals=True, n=1, activation=nn.ReLU()):
        super(OneImageNetModel, self).__init__()

        self.conv1 = nn.Conv2d(
            1, output_filters, kernel_size=5, bias=False, stride=1
        )

        self.conv_list = []
        for i in range(n):
            if residuals:
                self.conv_list.append(
                    BasicBlock(output_filters, output_filters)
                )
            else:
                self.conv_list.append(
                    nn.Conv2d(
                        output_filters, output_filters,
                        kernel_size=5, bias=False
                    )
                )
                self.conv_list.append(activation)
        self.conv_list = nn.Sequential(*tuple(self.conv_list))

        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv_list(x)
        return x


class VideoImprovingNet(nn.Module):
    def __init__(self, n=1,
                 residuals=True, input_images=6, filters_per_model=32):
        super(VideoImprovingNet, self).__init__()

        self.input_images_count = input_images
        self.filters_per_model = filters_per_model

        self.per_image_models = nn.ModuleList([
            OneImageNetModel(
                output_filters=filters_per_model,
                residuals=residuals,
                n=n
            )
            for i in range(input_images)
        ])

        self.conv1 = nn.Conv2d(
            filters_per_model * input_images,
            32,
            kernel_size=5,
            padding=5,
            padding_mode='reflex',
            bias=False
        )

        self.conv2 = nn.Conv2d(
            32,
            16,
            kernel_size=5,
            padding=5,
            padding_mode='reflex',
            bias=False
        )

        self.conv_out = nn.Conv2d(
            16,
            1,
            kernel_size=1,
            bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, *input):
        assert len(input) == len(self.per_image_models)
        base_models_outputs = []
        for i, input_image in enumerate(input):
            base_models_outputs.append(
                self.per_image_models[i](input_image)
            )

        x = torch.cat(tuple(base_models_outputs), dim=1)

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv_out(x)

        # print(x.shape)

        return x

    def _inference(self, x):
        x = torch.cat(
            (self.per_image_models[0](x),) * self.input_images_count,
            dim=1
        )

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv_out(x)

        return torch.sigmoid(x)

    def inference(self, x):
        return self.forward(*tuple([
                               x
                           ] * self.input_images_count))