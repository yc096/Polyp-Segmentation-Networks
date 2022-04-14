# _*_ coding: utf-8 _*_
# @Time : 2022/4/14 20:54 
# @Author : yc096
# @File : Functional.py
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