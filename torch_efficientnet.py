#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# NeurIPS Research Test
# Working Directory :
# Redesign from the pytorch_test_2024
###########################################################################
_description = '''\
====================================================
torch_efficient.py : Based on torch module
                    Written by 
====================================================
Example : python torch_efficient.py 
'''
#=============================================================
# Definitions
#=============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt
#%matplotlib inline

# utils
import numpy as np
import time
import copy

import my_debug as DBG
###########################################################################

###########################################################################
# Swish activation function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

    def _check(self, _inch=3, _inputsize=224):
        x = torch.randn(_inch, _inch, _inputsize, _inputsize)
        model = Swish()
        output = model(x)

        print(self.__class__.__name__, 'output size:', output.size())

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

    def _check(self, _inch=3, _inputsize=56, _outputsize=17):
        x = torch.randn(_inch, _inputsize, _outputsize, _outputsize)
        model = SEBlock(x.size(1))
        output = model(x)

        print(self.__class__.__name__, 'output size:', output.size())

class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * MBConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

class SepConv(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x
# ------------------------------------------------------------
# Main EfficientNet Class
# ------------------------------------------------------------
class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        # efficient net
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)
        self.stage3 = self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)
        self.stage4 = self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)
        self.stage5 = self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)
        self.stage6 = self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)
        self.stage7 = self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)
        self.stage8 = self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

# ------------------------------------------------------------
# Interface of EfficientNet model
# ------------------------------------------------------------
class EfficientNet_model:
    def __init__(self,
            architecture=7,
            in_channels =3,
            in_height   =224,
            in_width    =224,
            num_classes =10
            ):
        self.efficientnet_ID    = architecture
        self.in_channels        = in_channels
        self.in_height          = in_height
        self.in_width           = in_width
        self.num_classes        = num_classes
        self.efficientnet_list  =[]
        self.efficientnet_list.append(self.efficientnet_b0)
        self.efficientnet_list.append(self.efficientnet_b1)
        self.efficientnet_list.append(self.efficientnet_b2)
        self.efficientnet_list.append(self.efficientnet_b3)
        self.efficientnet_list.append(self.efficientnet_b4)
        self.efficientnet_list.append(self.efficientnet_b5)
        self.efficientnet_list.append(self.efficientnet_b6)
        self.efficientnet_list.append(self.efficientnet_b7)


    def efficientnet_b0(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)
    def efficientnet_b1(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, dropout=0.2, se_scale=4)
    def efficientnet_b2(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., dropout=0.3, se_scale=4)
    def efficientnet_b3(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, dropout=0.3, se_scale=4)
    def efficientnet_b4(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, dropout=0.4, se_scale=4)
    def efficientnet_b5(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, dropout=0.4, se_scale=4)
    def efficientnet_b6(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, dropout=0.5, se_scale=4)
    def efficientnet_b7(self):
        return EfficientNet(num_classes=self.num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, dropout=0.5, se_scale=4)

    def get_efficientnet_model(self):
        return_model = self.efficientnet_list[self.efficientnet_ID]

        print("efficientnet_id : ", self.efficientnet_ID)
        return return_model()

#=============================================================
# Test Processing
#=============================================================
import argparse
import textwrap
# ------------------------------------------------------------
# Parsing the Argument and service Function
# ------------------------------------------------------------
def _ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='test pytorch_inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-eb', '--efficinetnet_id',
                        help="[0] Index of Efficient Net. This framework provide from b0 to b7: Default is b0",
                        type=int, default=0)
    parser.add_argument('-ec', '--efficinetnet_num_classes',
                        help="[10] number of classes for efficient net",
                        type=int, default=10)
    parser.add_argument('-t', '--training',
                        help="[0] test [(1)] training",
                        type=int, default=1)
    args = parser.parse_args()
    args.training = True if args.training == 1 else False
    return args


if __name__ == "__main__":
    args    = _ArgumentParse(_description)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _samples        = 1
    _in_channels    = 3
    _in_height      = 224
    _in_width       = 224
    EF    = EfficientNet_model(architecture=args.efficinetnet_id,
                               in_channels=_in_channels, in_height=_in_height, in_width=_in_width)
    model = EF.get_efficientnet_model().to(device)
    x = torch.randn(_samples, _in_channels, _in_height, _in_width).to(device)
    try:
        output = model(x)
        print('output size:', output.size())
    except Exception as e:
        print("Error Occurs!!!!")
        print(e)
        exit()

    try:
        # input size =[num_samples, channels, width, height]
        summary(model, input_size = (1, _in_channels, 456, 456), device=device.type)
    except Exception as e:
        print(e)

    print("=============================================================")
    print("Process Finished!!")
    print("=============================================================")

