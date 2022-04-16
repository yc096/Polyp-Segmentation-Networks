import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import PolypDatasets
from Utils.loss import structure_loss, structure_loss_PraNet
from Utils.metrics import metrics

# temp
import numpy as np


class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def reset(self):
        self.prediction = []
    def update(self, pred, gt):
        batch = pred.shape[0]
        for index in range(0, batch):
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
    def reset(self):
        self.prediction = []
    def update(self, pred, gt):
        batch = pred.shape[0]
        for index in range(0, batch):
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



def getTime():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        print('!cuda不可用,此时记录的时间会有偏差!')
    return time.perf_counter()


if __name__ == '__main__':
    # 权重路径
    pth_path = r'C:\WorkSpace\Polyp-Segmentation-Networks\Checkpoint\20220416-12_48_56_130.pth'
    # 选择设
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 构建模型
    from Model.Model1_MutilAttention import Model1

    model = Model1().to(device)
    # model.load_state_dict(torch.load(pth_path))  # GPU
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))  # CPU
    # 构建损失函数
    criterion = structure_loss()
    # 性能指标
    metrics = metrics()
    # 获得数据迭代器
    test_data_loader = PolypDatasets.get_test_data_loader(BATCH_SIZE=1)
    # 暖机
    warming_input = torch.randn([10, 3, 352, 352]).to(device)
    warming_out = model(warming_input)
    #
    myiou = cal_iou()
    mydice = cal_dice()


    for i, data_loader in enumerate(test_data_loader):
        model.eval()
        metrics.reset()
        myiou.reset()
        mydice.reset()
        epoch_loss = 0.0
        epoch_time = 0.0
        for inputs, masks in data_loader:
            print(data_loader.dataset.dataset_name)
            inputs = inputs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                # ----------------------
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.perf_counter() - inference_time
                # ----------------------
                loss = criterion(outputs, masks)
                epoch_loss += (loss.item())
                epoch_time += inference_time
                metrics.add_batch(pred=outputs.detach().sigmoid(), mask=masks.detach(), threshold=0.5)
                myiou.update(outputs.detach().sigmoid(), masks.detach())
                mydice.update(outputs.detach().sigmoid(), masks.detach())
        print('123')
        print(metrics.show())
        print(myiou.show())
        print(mydice.show())
        print('Total Inference Time:{} Dataset Length:{} Avg Inference:{}')
        break
