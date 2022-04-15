# _*_ coding: utf-8 _*_
# @Time : 2022/4/15 11:45
# @Author : yc096
# @File : Model1_FeatureFusion1.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


def channel_shuffle(x, g_number):
    batchsize, num_channels, height, width = x.data.shape
    assert (num_channels % g_number == 0)
    channels_per_group = num_channels // g_number
    x = x.reshape(batchsize, g_number, channels_per_group, height * width)  # (batch,group,channel_per_group,H*W)
    x = x.permute(0, 2, 1, 3).contiguous()  # (batch,channel_per_group,group,H*W)
    x = x.reshape(batchsize, -1, height, width)  # (batch,channel,H,W)
    return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        left = self.maxpool(x)
        right = self.conv(x)
        out = torch.cat((left, right), dim=1)
        out = self.relu(self.bn(out))
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

    def forward(self, decoder_feature=None, encoder_feature=None):
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



class FCU(nn.Module):
    def __init__(self, in_out_channels, kernel_size=3, dilation=1, dropout_prob=0):
        super(FCU, self).__init__()
        inter_channels = in_out_channels // 2
        padding = ((kernel_size - 1) // 2) * dilation
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_prob)
        # left branch
        self.left_conv1 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=False)
        self.left_conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=False)
        self.left_conv3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=False)
        self.left_conv4 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=False)
        self.left_bn1 = nn.BatchNorm2d(num_features=inter_channels)
        self.left_bn2 = nn.BatchNorm2d(num_features=inter_channels)
        # right branch
        self.right_conv1 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=False)
        self.right_conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=False)
        self.right_conv3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=False)
        self.right_conv4 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=False)
        self.right_bn1 = nn.BatchNorm2d(num_features=inter_channels)
        self.right_bn2 = nn.BatchNorm2d(num_features=inter_channels)

    def forward(self, x):
        residual = x
        x1, x2 = channel_split(x)
        left = self.relu(self.left_conv1(x1))
        left = self.relu(self.left_bn1(self.left_conv2(left)))
        left = self.relu(self.left_conv3(left))
        left = self.relu(self.left_bn2(self.left_conv4(left)))
        if self.drop.p != 0:
            left = self.drop(left)
        right = self.relu(self.right_conv1(x2))
        right = self.relu(self.right_bn1(self.right_conv2(right)))
        right = self.relu(self.right_conv3(right))
        right = self.relu(self.right_bn2(self.right_conv4(right)))
        if self.drop.p != 0:
            right = self.drop(right)
        out = torch.cat([left, right], dim=1)
        out = residual + out
        out = self.relu(out)
        out = channel_shuffle(out, g_number=2)
        return out



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Stage 1
        self.downsample1 = DownsampleBlock(in_channels=3, out_channels=32)
        self.conv1 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0.03)
        self.conv2 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0.03)
        self.conv3 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0.03)
        self.conv4 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0.03)
        self.conv5 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0.03)
        # Stage 2
        self.downsample2 = DownsampleBlock(in_channels=32, out_channels=64)
        self.conv6 = FCU(in_out_channels=64, kernel_size=5, dilation=1, dropout_prob=0.03)
        self.conv7 = FCU(in_out_channels=64, kernel_size=5, dilation=1, dropout_prob=0.03)
        self.conv8 = FCU(in_out_channels=64, kernel_size=5, dilation=1, dropout_prob=0.03)
        self.conv9 = FCU(in_out_channels=64, kernel_size=5, dilation=1, dropout_prob=0.03)
        # Stage 3
        self.downsample3 = DownsampleBlock(in_channels=64, out_channels=128)
        self.conv10 = FCU(in_out_channels=128, kernel_size=3, dilation=2, dropout_prob=0.03)
        self.conv11 = FCU(in_out_channels=128, kernel_size=3, dilation=5, dropout_prob=0.03)
        self.conv12 = FCU(in_out_channels=128, kernel_size=3, dilation=9, dropout_prob=0.03)
        #decoder 1
        self.upsample1 = UpsampleBlock(in_channels=128,out_channels=64)
        self.ff1 = FeatureFusion1(decoder_in_channels=128,encoder_in_channels=64)
        self.conv13 = FCU(in_out_channels=64,kernel_size=5,dilation=1,dropout_prob=0)
        self.conv14 = FCU(in_out_channels=64, kernel_size=5, dilation=1, dropout_prob=0)
        #decoder 2
        self.upsample2 = UpsampleBlock(in_channels=64,out_channels=32)
        self.ff2 = FeatureFusion1(decoder_in_channels=64,encoder_in_channels=32)
        self.conv15 = FCU(in_out_channels=32,kernel_size=3,dilation=1,dropout_prob=0)
        self.conv16 = FCU(in_out_channels=32, kernel_size=3, dilation=1, dropout_prob=0)
        self.conv_pred =  nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        # encoder 1
        encoder1 = self.downsample1(x)
        encoder1 = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(encoder1)))))
        # encoder 2
        encoder2 = self.downsample2(encoder1)
        encoder2 = self.conv9(self.conv8(self.conv7(self.conv6(encoder2))))
        # encoder 3
        encoder3 = self.downsample3(encoder2)
        encoder3 = self.conv12(self.conv11(self.conv10(encoder3)))
        # decoder 1
        decoder1 = self.upsample1(encoder3)
        decoder1 = self.ff1(decoder1,encoder2)
        decoder1 = self.conv14(self.conv13(decoder1))
        # decoder 2
        decoder2 = self.upsample2(decoder1)
        decoder2 = self.ff2(decoder2,encoder1)
        decoder2 = self.conv16(self.conv15(decoder2))
        # finnal pred
        out = self.conv_pred(decoder2)
        return out
