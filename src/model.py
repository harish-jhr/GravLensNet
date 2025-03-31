import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from torchsummary import summary

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, choice=1):
        super(BasicConvBlock, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn_1', nn.BatchNorm2d(out_channels)),
            ('relu_1', nn.ReLU()),
            ('conv_2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(out_channels))
        ]))

        # Shortcut connection
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            if choice == 1:  # Identity mapping with padding
                padding = out_channels // 4
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, padding, padding, 0, 0)))
            elif choice == 2:  # 1x1 Convolutional projection
                self.shortcut = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                    ('bn', nn.BatchNorm2d(out_channels))
                ]))

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Modify input conv layer to accept (1,150,150) images
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 1 input channel
        self.bn0 = nn.BatchNorm2d(16)

        # Residual blocks
        self.block1 = self.build_layer(block_type, 16, num_blocks[0], starting_stride=1)
        self.block2 = self.build_layer(block_type, 32, num_blocks[1], starting_stride=2)
        self.block3 = self.build_layer(block_type, 64, num_blocks[2], starting_stride=2)

        # Global Average Pooling + Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 3)  # Output: 3 classes (no, sphere, vort)

    def build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        stride_list = [starting_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in stride_list:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# Function to instantiate ResNet-20 with 20 layers
def ResNet20():
    return ResNet(block_type=BasicConvBlock, num_blocks=[5, 5, 5])  # 6n+2 -> 20 layers
