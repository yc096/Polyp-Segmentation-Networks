# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 23:27 
# @Author : yc096
# @File : Model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, alpha_in=0.5, alpha_out=0.5):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, alpha_in, alpha_out)
        self.bn_h = nn.BatchNorm2d(num_features=out_channels - int(out_channels * alpha_out))
        self.bn_l = nn.BatchNorm2d(num_features=out_channels * alpha_out)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h, x_l = self.bn_h(x_h), self.bn_l(x_l)
        return x_h, x_l


class OctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, alpha_in=0.5, alpha_out=0.5):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, alpha_in, alpha_out)
        self.bn_h = nn.BatchNorm2d(num_features=out_channels - int(out_channels * alpha_out))
        self.bn_l = nn.BatchNorm2d(num_features=out_channels * alpha_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h, x_l = self.bn_h(x_h), self.bn_l(x_l)
        x_h, x_l = self.activation(x_h), self.activation(x_l)
        return x_h, x_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, alpha_in=0.5, alpha_out=0.5):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride  # 设置为2时会将输入缩减一半尺寸,不影响卷积步距.
        '''
        in_channels * alpha_in = 低通的输入通道
        out_channels * alpha_out = 低通的输出通道
        in_channels - (in_channels * alpha_in) = 高通的输入通道
        out_channels - (out_channels * alpha_out) = 高通的输出通道
        '''
        self.convl2l = nn.Conv2d(in_channels=int(in_channels * alpha_in), out_channels=int(out_channels * alpha_out),
                                 kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.convl2h = nn.Conv2d(in_channels=int(in_channels * alpha_in), out_channels=out_channels - int(out_channels * alpha_out),
                                 kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.convh2h = nn.Conv2d(in_channels=in_channels - int(in_channels * alpha_in), out_channels=out_channels - int(out_channels * alpha_out),
                                 kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.convh2l = nn.Conv2d(in_channels=in_channels - int(in_channels * alpha_in), out_channels=int(out_channels * alpha_out),
                                 kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = torch.randn([16, 32, 352, 352]), torch.randn([16, 32, 176, 176])  # 仅为获取参数量,训练时删除此行代码
        x_h, x_l = x
        if self.stride == 2:
            x_h, x_l = self.downsample(x_h), self.downsample(x_l)
        x_h2h = self.convh2h(x_h)
        x_h2l = self.convh2l(self.downsample(x_h))
        x_l2l = self.convl2l(x_l)
        x_l2h = self.upsample(self.convl2h(x_l))
        x_h = x_h2h + x_l2h
        x_l = x_l2l + x_h2l
        return x_h, x_l
