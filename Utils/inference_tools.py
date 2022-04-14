# _*_ coding: utf-8 _*_
# @Time : 2022/4/9 22:16
# @Author : yc096
# @File : inference_tools.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time


class inference_tools():
    def __init__(self):
        self.demo_tensor = torch.randn([1, 1, 352, 352])

    def tensorGrayImageShow(self, tensor, window_name=None):
        grayimage = self.tensor2GrayImage(tensor)
        self.ndarrayImageShow(grayimage, window_name)

    def tensorRGBImageShow(self, tensor, window_name=None):
        grayimage = self.tensor2RGBImage(tensor)
        self.ndarrayImageShow(grayimage, window_name)

    def tensor2GrayImage(self, tensor):
        # tensor.shape为 -> [1,1,H,W]
        grayimage = np.squeeze(tensor.cpu().numpy(), axis=(0, 1))  # [B,C,H,W] ->[H,W]
        return grayimage

    def tensor2RGBImage(self, tensor):
        # tensor.shape -> [1,3,H,W]
        rgbimage = np.squeeze(tensor.cpu().numpy(), axis=0)  # [B,C,H,W] ->[C,H,W]
        rgbimage = np.transpose(rgbimage, [1, 2, 0])
        return rgbimage

    def ndarrayImageShow(self, image, window_name=None):
        """
        显示一副图像,输入图像符合opencv格式,且颜色通道为RGB.
        图像类型为np.uint8时,值范围[0,255].
        图像类型为np.float32时,值范围[0,1].
        对于彩色图像[H,W,3]
        对于灰度图像[H,W]
        """
        if window_name == None:
            window_name = str(time.time_ns())
        else:
            window_name = str(window_name)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        if image.ndim == 2:
            cv2.imshow(window_name, image)  # 灰度图
        else:
            cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

