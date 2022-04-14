# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 10:02 
# @Author : yc096
# @File : train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.loss import structure_loss,BinaryDiceLoss
from Utils.Trainer import Trainer

if __name__ == '__main__':
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 构建模型
    from Model.ENet import ENet
    model = ENet().to(device)
    # 构建损失函数
    criterion = structure_loss()
    # 构建优化器 #1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    # 构建动态学习率
    lr_updater = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    # 开始训练
    trainer = Trainer(model,optimizer,criterion,lr_updater,device)
    trainer.train()