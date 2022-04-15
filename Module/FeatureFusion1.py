# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 21:31 
# @Author : yc096
# @File : FeatureFusion1.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion1(nn.Module):
    def __init__(self, decoder_in_channels, encoder_in_channels):
        super(FeatureFusion1, self).__init__()
        self.channel_attention = ChannelAttention(in_out_channels=encoder_in_channels)
        self.spatial_attention = SpatialAttention()
        #
        self.decoder_conv = nn.Conv2d(in_channels=decoder_in_channels, out_channels=encoder_in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder_conv_bn = nn.BatchNorm2d(num_features=encoder_in_channels)
        #
        self.encoder_conv = nn.Conv2d(in_channels=encoder_in_channels, out_channels=encoder_in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.encoder_conv_bn = nn.BatchNorm2d(num_features=encoder_in_channels)
        #

    def forward(self, decoder_feature, encoder_feature):
        # decoder
        decoder = self.decoder_conv_bn(self.decoder_conv(decoder_feature))
        decoder = F.interpolate(decoder, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_spatial_score = self.spatial_attention(decoder)

        # encoder
        encoder = self.encoder_conv_bn(self.encoder_conv(encoder_feature))
        encoder = torch.mul(encoder, decoder_spatial_score)

        # feature fusion
        out = encoder + decoder
        out = torch.mul(self.channel_attention(out), out)
        return out


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

# FeatureFusion1(decoder_in_channels=128,encoder_in_channels=128)Total params: 37,475
# FeatureFusion1(decoder_in_channels=128,encoder_in_channels=64) Total params: 13,667
# FeatureFusion1(decoder_in_channels=64,encoder_in_channels=64) Total params: 9,571
# FeatureFusion1(decoder_in_channels=64,encoder_in_channels=32) Total params: 3,555
# FeatureFusion1(decoder_in_channels=32,encoder_in_channels=32) Total params: 2,531