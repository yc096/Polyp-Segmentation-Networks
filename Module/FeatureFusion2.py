# _*_ coding: utf-8 _*_
# @Time : 2022/4/16 9:35 
# @Author : yc096
# @File : FeatureFusion2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSpatialAttention(nn.Module):  # Total params: 498
    def __init__(self):
        super(MultiSpatialAttention, self).__init__()
        #
        self.convl2l = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=False)
        self.convl2m = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=False)
        self.convl2s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=False)
        self.convm2l = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.convm2m = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.convm2s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.convs2l = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.convs2m = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.convs2s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        #
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample2x = nn.AvgPool2d(kernel_size=4, stride=4)
        self.upsample2x = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        x_l, x_m, x_s = x

        x_l_avg = torch.mean(x_l, dim=1, keepdim=True)
        x_l_max, _ = torch.max(x_l, dim=1, keepdim=True)  # Returns a namedtuple (values, indices)
        x_l = torch.cat([x_l_avg, x_l_max], dim=1)

        x_m_avg = torch.mean(x_m, dim=1, keepdim=True)
        x_m_max, _ = torch.max(x_m, dim=1, keepdim=True)
        x_m = torch.cat([x_m_avg, x_m_max], dim=1)

        x_s_avg = torch.mean(x_s, dim=1, keepdim=True)
        x_s_max, _ = torch.max(x_s, dim=1, keepdim=True)
        x_s = torch.cat([x_s_avg, x_s_max], dim=1)

        # large scale
        y_l2l = self.convl2l(x_l)
        y_m2l = self.upsample(self.convm2l(x_m))
        y_s2l = self.upsample2x((self.convs2l(x_s)))
        y_l = y_l2l + y_m2l + y_s2l

        # mid scale
        y_l2m = self.convl2m(self.downsample(x_l))
        y_m2m = self.convm2m(x_m)
        y_s2m = self.upsample(self.convs2m(x_s))
        y_m = y_l2m + y_m2m + y_s2m

        # small scale
        y_l2s = self.convl2s(self.downsample2x(x_l))
        y_m2s = self.convm2s(self.downsample(x_m))
        y_s2s = self.convs2s(x_s)
        y_s = y_l2s + y_m2s + y_s2s

        return y_l.sigmoid(), y_m.sigmoid(), y_s.sigmoid()


class FeatureFusion2(nn.Module):
    def __init__(self):
        super(FeatureFusion2, self).__init__()
        self.MSA = MultiSpatialAttention()
        self.small_up1 = UpsampleBlock(in_channels=128, out_channels=64)
        self.small_up2 = UpsampleBlock(in_channels=64, out_channels=32)
        self.small_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.small_conv_bn = nn.BatchNorm2d(num_features=32)
        self.mid_up1 = UpsampleBlock(in_channels=64, out_channels=32)
        self.mid_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.mid_conv_bn = nn.BatchNorm2d(num_features=32)
        self.large_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.large_con_bn = nn.BatchNorm2d(num_features=32)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.randn([16, 32, 176, 176]), torch.randn([16, 64, 88, 88]), torch.randn([16, 128, 44, 44])  # 测试使用, 训练时删除
        x_l, x_m, x_s = x
        x_l_score, x_m_score, x_s_score = self.MSA(x)
        x_s = torch.mul(x_s, x_s_score)
        x_s = self.activation(self.small_conv_bn(self.small_conv(self.small_up2(self.small_up1(x_s)))))
        x_m = torch.mul(x_m, x_m_score)
        x_m = self.activation(self.mid_conv_bn(self.mid_conv(self.mid_up1(x_m))))
        x_l = self.activation(self.large_con_bn(self.large_conv(x_l)))
        out = x_l + x_m + x_s
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


model = FeatureFusion2()
from torchstat import stat

stat(model, (1, 1, 1))
