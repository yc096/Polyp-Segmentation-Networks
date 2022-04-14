# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 20:43 
# @Author : yc096
# @File : ChannelAttention.py
import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_out_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(in_channels=in_out_channels, out_channels=in_out_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_out_channels // reduction_ratio, out_channels=in_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        return self.sigmoid(avg + max)