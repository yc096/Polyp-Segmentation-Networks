# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 20:43 
# @Author : yc096
# @File : SpatialAttention.py
import torch
import torch.nn as nn
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)  # Returns a namedtuple (values, indices)
        out = torch.cat([avg, max], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out