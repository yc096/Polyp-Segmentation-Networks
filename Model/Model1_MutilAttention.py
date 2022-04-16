# _*_ coding: utf-8 _*_
# @Time : 2022/4/9 10:57 
# @Author : yc096
# @File : Model1.py
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
        # x = torch.randn([16, 32, 176, 176]), torch.randn([16, 64, 88, 88]), torch.randn([16, 128, 44, 44])  # 测试使用, 训练时删除
        x_l, x_m, x_s = x
        x_l_score, x_m_score, x_s_score = self.MSA(x)
        x_s = torch.mul(x_s, x_s_score)
        x_s = self.activation(self.small_conv_bn(self.small_conv(self.small_up2(self.small_up1(x_s)))))
        x_m = torch.mul(x_m, x_m_score)
        x_m = self.activation(self.mid_conv_bn(self.mid_conv(self.mid_up1(x_m))))
        x_l = self.activation(self.large_con_bn(self.large_conv(x_l)))
        out = x_l + x_m + x_s
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


class Model1(nn.Module):  # Total params: 679,394
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

        # MultiSpatialAttention
        self.ff = FeatureFusion2()
        # final pred
        self.conv_pred = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

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
        # MultiSpatialAttention
        ff = self.ff((encoder1,encoder2,encoder3))
        # finnal pred
        out = self.conv_pred(ff)
        return out
