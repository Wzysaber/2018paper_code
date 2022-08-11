import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置采用的GPU序号

import scipy.io as io
import imageio

import configs.configs as cfg
import torch.optim as optim

from data.HSI_data import HSI_data as fun_data
from data.get_train_test_set import get_train_test_set as fun_get_set

from model.MSCNN import MSCNN
from model.SSUN import SSUN
from model.LSTM import lstm

from tool.train import train as fun_train
from tool.test import test as fun_test
from matplotlib import pyplot as plt

import show
import warnings

warnings.filterwarnings("ignore")


def main():
    # (1)基本参数
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']  # 导入优化模型的相应参数
    cfg_test = cfg.test

    # （2）导入数据
    data_sets = fun_get_set(cfg_data)

    train_data = fun_data(data_sets, cfg_data['train_data'])
    test_data = fun_data(data_sets, cfg_data['test_data'])
    no_gt_data = fun_data(data_sets, cfg_data['no_gt_data'])

    # （3）训练的相关配置
    device = torch.device("cuda:2")

    # 加载模型
    model = SSUN().to(device)

    # 损失函数
    loss_fun = nn.CrossEntropyLoss()

    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'],
    #                       momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    # optimizer = optim.RMSprop(model.parameters(), lr=cfg_optim['lr'],
    #                           momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    optimizer = optim.Adam(params=model.parameters(), lr=cfg_optim['lr'],
                           betas=cfg_optim['betas'], eps=1e-8, weight_decay=cfg_optim['weight_decay'])

    # 训练
    fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)

    # 测试
    pred_train_label = fun_test(train_data, data_sets['ori_gt'], model, device, cfg_test)
    pred_test_label = fun_test(test_data, data_sets['ori_gt'], model, device, cfg_test)
    pred_no_gt_label = fun_test(no_gt_data, data_sets['ori_gt'], model, device, cfg_test)

    predict_label = torch.cat([pred_train_label, pred_test_label], dim=0)

    # 直接显示相应的图像
    HSI = show.Predict_Label2Img(predict_label)
    plt.imshow(HSI)
    plt.show()

    save_folder = cfg_test['save_folder']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)  # 用于创建目录

    io.savemat(save_folder + '/classification_label.mat', {'predict_label_CNN1D': predict_label})  # 将测试数据保存在mat文件中


if __name__ == '__main__':
    main()
