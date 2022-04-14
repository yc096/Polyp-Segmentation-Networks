import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import PolypDatasets
from Utils.loss import structure_loss, structure_loss_PraNet
from Model.PraNet import PraNet
from Utils.metrics import metrics
from torch.profiler import profile, record_function, ProfilerActivity

#temp
import numpy as np
class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred.cpu().numpy(), gt.cpu().numpy())
        self.prediction.append(score)


    def cal(self, input, target):
        smooth = 1e-5
        input = input > 0.5
        target_ = target > 0.5
        intersection = (input & target_).sum()
        union = (input | target_).sum()

        return (intersection + smooth) / (union + smooth)
    def show(self):
        return np.mean(self.prediction)

class cal_dice(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred.cpu().numpy(), gt.cpu().numpy())
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def show(self):
        return np.mean(self.prediction)

class IoU():
    """
    计算每一个类的交并比并且返回mIoU.
    混淆矩阵每行代表了实际、每列代表了预测.
    """
    def __init__(self, num_classes):
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes
        self.conf.fill(0)

    def reset(self):
        self.conf.fill(0)

    def add_batch(self, pred, target):
        """
        计算预测图和标签图的混淆矩阵.不明白的可以参考:https://zhuanlan.zhihu.com/p/359002353
        pred:[B,K,H,W]  --->    B个样本,每个样本K个类别预测图.
        target:[B,H,W]  --->    B个样本.
        """
        # 将预测和标签转为ndarray
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        assert np.ndim(pred) == 4, '参数pred仅接收4维如[B,C,H,W]的预测输出,但是接收维度:{}'.format(pred.shape)
        assert pred.shape[0] == target.shape[0], '预测输出的Batch_Size:{}和标签的Batch_Size:{}不一致.'.format(pred.shape[0],target.shape[0])
        assert (target.max()<self.num_classes) and (target.min()>= 0 ),'标签值应当在0-(k-1),k为类别数.'

        x = pred + self.num_classes*target
        bincount_2d = np.bincount(
            x.astype(np.int64).flatten(),minlength=self.num_classes**2
        )
        hist = bincount_2d.reshape((self.num_classes,self.num_classes))
        self.conf += hist
        pass

    def show(self):
        #展示结果
        conf_matrix = self.conf
        TP = np.diag(conf_matrix)
        FP = np.sum(conf_matrix,0) - TP
        FN = np.sum(conf_matrix,1) - TP
        iou = TP/(FP+FN+TP) #每一行对应label中的一类
        miou = np.nanmean(iou)
        return iou,miou

#temp
if __name__ == '__main__':
    # 权重路径
    pth_path = r'D:\息肉分割记录\EPSNet_6datasets_OnlyEncoder_PAN_featurefusion\20220411-18_11_23_138.pth'
    # 选择设
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 构建模型
    from Model.EPSNet_onlyEncoder_PAN_fusionfeature import EPSNet
    model = EPSNet().to(device).eval()
    model.load_state_dict(torch.load(pth_path))

    # 构建损失函数
    criterion = structure_loss()
    # 性能指标
    metrics = metrics()
    # 获得数据迭代器
    # # train_loader, Kvasir_loader, CVC_ClinicDB_loader, CVC_ColonDB_loader, CVC_300_loader, ETIS_loader,EndoTect_2020_loader
    dataloader = PolypDatasets.get_data_loader(BATCH_SIZE=1, NUM_WORERS=8)
    Kvasir_loader = dataloader[1]
    CVC_ClinicDB_loader = dataloader[2]
    CVC_ColonDB_loader = dataloader[3]
    CVC_300_loader = dataloader[4]
    ETIS_loader = dataloader[5]
    EndoTect_2020_loader = dataloader[6]

    warming_input = torch.randn([10, 3, 352, 352]).to(device)
    warming_out = model(warming_input)
    torch.cuda.synchronize()
    print('暖机完成')

    #
    myiou = cal_iou()
    mydice = cal_dice()

    #
    iou1 = IoU(num_classes=2)

    total_time = 0
    for inputs, masks in Kvasir_loader:
        metrics.reset()
        inputs = inputs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            # ----------------------
            torch.cuda.synchronize()
            inference_time = time.perf_counter()
            outputs = model(inputs)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - inference_time
            # ----------------------
            loss = criterion(outputs, masks)
            metrics.add_batch(torch.sigmoid(outputs[-1].detach()), masks.detach(), threshold=0.5)
            myiou.update(torch.sigmoid(outputs[-1].detach()), masks.detach())
            mydice.update(torch.sigmoid(outputs[-1].detach()), masks.detach())
            iou1.add_batch(outputs.detach().sigmoid(),masks.detach())
            # print(metrics.getMetrics())
            # print(myiou.show())
            # print(mydice.show())
            total_time += inference_time
            # print(metrics.getMetrics())
            # print(inference_time)
        # break   #只获取一张图像作测试
    print(metrics.getMetrics())
    print(myiou.show())
    print(mydice.show())
    print('Total Inference Time:{} Dataset Length:{} Avg Inference:{}'.format(total_time,len(CVC_ColonDB_loader),total_time/len(CVC_ColonDB_loader)))
