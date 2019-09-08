import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.nn as nn

# from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.densenet import DenseNet

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
# from pretrainedmodels.models.torchvision_models import pretrained_settings

from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}

# Encoder implementation

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(1, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(1, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SENetEncoder(SENet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False

        del self.last_linear
        del self.avg_pool

    def forward(self, x):
        for module in self.layer0[:-1]:
            x = module(x)

        x0 = x
        x = self.layer0[-1](x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


senet_encoders = {
    'senet154': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['senet154'],
        'out_shapes': (2048, 1024, 512, 256, 128),
        'params': {
            'block': SEBottleneck,
            'dropout_p': 0.2,
            'groups': 64,
            'layers': [3, 8, 36, 3],
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet50': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 4, 6, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet101': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 4, 23, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnet152': {
        'encoder': SENetEncoder,
        'pretrained_settings': pretrained_settings['se_resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNetBottleneck,
            'layers': [3, 8, 36, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 1,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnext50_32x4d': {
        'encoder': SENetEncoder,
        #         'pretrained_settings': pretrained_settings['se_resnext50_32x4d'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNeXtBottleneck,
            'layers': [3, 4, 6, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 32,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },

    'se_resnext101_32x4d': {
        'encoder': SENetEncoder,
        #         'pretrained_settings': pretrained_settings['se_resnext101_32x4d'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': SEResNeXtBottleneck,
            'layers': [3, 4, 23, 3],
            'downsample_kernel_size': 1,
            'downsample_padding': 0,
            'dropout_p': None,
            'groups': 32,
            'inplanes': 64,
            'input_3x3': False,
            'num_classes': 1000,
            'reduction': 16
        },
    },
}


class ResNetEncoder(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


resnet_encoders = {
    'resnet18': {
        'encoder': ResNetEncoder,
        #         'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'resnet34': {
        'encoder': ResNetEncoder,
        #         'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet50': {
        'encoder': ResNetEncoder,
        #         'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet101': {
        'encoder': ResNetEncoder,
        #         'pretrained_settings': pretrained_settings['resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'resnet152': {
        'encoder': ResNetEncoder,
        #         'pretrained_settings': pretrained_settings['resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
}


class DenseNetEncoder(DenseNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.classifier
        self.initialize()

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x0 = x

        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x, x1 = self._transition(x, self.features.transition1)

        x = self.features.denseblock2(x)
        x, x2 = self._transition(x, self.features.transition2)

        x = self.features.denseblock3(x)
        x, x3 = self._transition(x, self.features.transition3)

        x = self.features.denseblock4(x)
        x4 = self.features.norm5(x)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        state_dict.pop('classifier.bias')
        state_dict.pop('classifier.weight')

        super().load_state_dict(state_dict)


densenet_encoders = {
    'densenet121': {
        'encoder': DenseNetEncoder,
        #         'pretrained_settings': pretrained_settings['densenet121'],
        'out_shapes': (1024, 1024, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
        }
    },

    'densenet169': {
        'encoder': DenseNetEncoder,
        #         'pretrained_settings': pretrained_settings['densenet169'],
        'out_shapes': (1664, 1280, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 32, 32),
        }
    },

    'densenet201': {
        'encoder': DenseNetEncoder,
        #         'pretrained_settings': pretrained_settings['densenet201'],
        'out_shapes': (1920, 1792, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 48, 32),
        }
    },

    'densenet161': {
        'encoder': DenseNetEncoder,
        #         'pretrained_settings': pretrained_settings['densenet161'],
        'out_shapes': (2208, 2112, 768, 384, 96),
        'params': {
            'num_init_features': 96,
            'growth_rate': 48,
            'block_config': (6, 12, 36, 24),
        }
    },

}

encoders = {}
encoders.update(resnet_encoders)
# encoders.update(dpn_encoders)
# encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)


# encoders.update(inception_encoders)


def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained='imagenet'):
    settings = encoders[encoder_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError(
            'Avaliable pretrained options {}'.format(settings.keys()))

    formatted_settings = {}
    formatted_settings['input_space'] = settings[pretrained].get('input_space')
    formatted_settings['input_range'] = settings[pretrained].get('input_range')
    formatted_settings['mean'] = settings[pretrained].get('mean')
    formatted_settings['std'] = settings[pretrained].get('std')
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained='imagenet'):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x


# Decoder implementation

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
def _upsample(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class PyramidStage(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = _upsample(x, size=(h, w))
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.stages = nn.ModuleList([
            PyramidStage(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [stage(x) for stage in self.stages] + [x]
        x = torch.cat(xs, dim=1)
        return x


class AUXModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return x


class PSPDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            downsample_factor=8,
            use_batchnorm=True,
            psp_out_channels=512,
            final_channels=21,
            aux_output=False,
            dropout=0.2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.out_channels = self._get(encoder_channels)
        self.aux_output = aux_output
        self.dropout_factor = dropout

        self.psp = PSPModule(
            self.out_channels,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = Conv2dReLU(
            self.out_channels * 2,
            psp_out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        if self.dropout_factor:
            self.dropout = nn.Dropout2d(p=dropout)

        self.final_conv = nn.Conv2d(psp_out_channels, final_channels,
                                    kernel_size=(3, 3), padding=1)

        if self.aux_output:
            self.aux = AUXModule(self.out_channels, final_channels)

        self.initialize()

    def _get(self, xs):
        if self.downsample_factor == 4:
            return xs[3]
        elif self.downsample_factor == 8:
            return xs[2]
        elif self.downsample_factor == 16:
            return xs[1]
        else:
            raise ValueError('Downsample factor should bi in [4, 8, 16], got {}'
                             .format(self.downsample_factor))

    def forward(self, x):

        features = self._get(x)
        x = self.psp(features)
        x = self.conv(x)
        if self.dropout_factor:
            x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(
            x,
            scale_factor=self.downsample_factor,
            mode='bilinear',
            align_corners=True
        )

        if self.training and self.aux_output:
            aux = self.aux(features)
            x = [x, aux]

        return x


class PSPNet(EncoderDecoder):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        psp_in_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
            to construct PSP module on it.
        psp_out_channels: number of filters in PSP block.
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        psp_aux_output: if ``True`` add auxiliary classification output for encoder training
        psp_dropout: spatial dropout rate between 0 and 1.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
    Returns:
        ``torch.nn.Module``: **PSPNet**
    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            psp_in_factor=8,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_aux_output=False,
            classes=21,
            dropout=0.2,
            activation='softmax',
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = PSPDecoder(
            encoder_channels=encoder.out_shapes,
            downsample_factor=psp_in_factor,
            psp_out_channels=psp_out_channels,
            final_channels=classes,
            dropout=dropout,
            aux_output=psp_aux_output,
            use_batchnorm=psp_use_batchnorm,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'psp-{}'.format(encoder_name)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x


class Unet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
        attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
    Returns:
        ``torch.nn.Module``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            attention_type=None
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)


class DefSENet(nn.Module):
    def __init__(self, input_channels=1, classes=1, **kwargs):
        super(DefSENet, self).__init__()
        self.model = Unet(
            'se_resnext50_32x4d',
            classes=classes,
            encoder_weights=None,
            activation='sigmoid'
        )

    def forward(self, x):
        result = self.model(x)
        # print('AAAAAAAAAAAAAAAA', result.shape)
        return result
