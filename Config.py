# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 9:20 
# @Author : yc096
# @File : Config.py
import os

# -----Project Setting-----#
PROJECT_ROOT = r'C:\WorkSpace\Polyp-Segmentation-Framework'
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'Data')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'CheckPoint')

# -----Train Config-----#
START_EPOCH = 1
MAX_EPOCH = 150
TEST_PER_EPOCH = 1  # 0表示不测试
SAVE_MODEL_PER_EPOCH = 1  # 训练几轮保存一次模型参数
TRAIN_SIZE_H, TRAIN_SIZE_W = 352, 352  # 训练时网络所需的尺寸

# -----Dataloader Config-----#
DATASETS_NAME_TRAIN = ['Kvasir-CVC-ClinicDB']
DATASETS_NAME_TEST = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300', 'EndoTect-2020']
BATCH_SIZE = 16
NUM_WORERS = 8
