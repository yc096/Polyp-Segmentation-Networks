# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 21:15 
# @Author : yc096
# @File : ContextFeatureExtraction.py
import torch
import torch.nn as nn


class ContextFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1 * 3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1 * 5, dilation=5)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1 * 7, dilation=7)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.bn(out)
        return out
