import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

from misc.utils import *


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual * 0), 1)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False,
                 dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# No projection: identity shortcut
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)



        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet, self).__init__()



        factor = 1

        strides = [1, 2, 1]
        filt_sizes = [64, 128, 256, 512, 1024][:len(layers)]
        self.series_blocks, self.ds = [], []
        self.parallel_blocks, self.parallel_ds = [], []
        self.layer_config = layers
        self.inplanes1 = int(64 * factor)
        self.inplanes2 = int(64 * factor)

        self.de_pred = nn.Sequential(Conv2d(filt_sizes[-1] * 4, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        #initialize_weights(self.modules())

        self.conv1 = nn.Conv2d(3, self.inplanes1, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer_for_block(block, filt_size, num_blocks, stride=stride)
            self.series_blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        self.series_blocks = nn.ModuleList(self.series_blocks)
        self.ds = nn.ModuleList(self.ds)

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer_for_parallel_block(block, filt_size, num_blocks, stride=stride)
            self.parallel_blocks.append(nn.ModuleList(blocks))
            self.parallel_ds.append(ds)
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
        self.parallel_ds = nn.ModuleList(self.parallel_ds)





    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer_for_block(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes1 != planes * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []

        layers.append(block(self.inplanes1, planes, stride, downsample))

        self.inplanes1 = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes1, planes))

        return layers, downsample


    def _make_layer_for_parallel_block(self, block, planes, blocks, stride=1):


        downsample = None

        if stride != 1 or self.inplanes2 != planes * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []

        layers.append(block(self.inplanes2, planes, stride, downsample))

        self.inplanes2 = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes2, planes))

        return layers, downsample


    def forward(self, x, policy=None):

        t = 0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if policy is not None:

            for segment, num_blocks in enumerate(self.layer_config):

                for b in range(num_blocks):
                    action = policy[:, t].contiguous()

                    action_mask = action.float().view(-1, 1, 1, 1)

                    residual = self.ds[segment](x) if b == 0 else x
                    output = self.series_blocks[segment][b](x)

                    residual_ = self.parallel_ds[segment](x) if b == 0 else x

                    output_ = self.parallel_blocks[segment][b](x)

                    f1 = F.relu(residual + output)
                    f2 = F.relu(residual_ + output_)

                    x = f1 * (1 - action_mask) + f2 * action_mask
                    t += 1

        else:
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 else x
                    output = self.series_blocks[segment][b](x)
                    x = F.relu(residual + output)
                    t += 1

        x = self.de_pred(x)

        x = F.upsample(x, scale_factor=8)

        return x


def resnet26(num_class=10, blocks=Bottleneck):
    return ResNet(blocks, [3,4,6], num_class)