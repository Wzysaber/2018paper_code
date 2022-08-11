import torch
import time
import datetime
import math
import os

from data.HSI_data import batch_collate as collate_fn
from torch.utils.data import DataLoader


# from model.LSTM import lstm
# from model.MSCNN import MSCNN


# 学习概调整策略
def adjust_lr(lr_init, lr_gamma, optimizer, epoch, step_index):
    if epoch < 1:
        lr = 0.0001 * lr_init
    else:
        lr = lr_init * lr_gamma ** ((epoch - 1) // step_index)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# 训练
def train(train_data, model, loss_fun, optimizer, device, cfg):
    '''
        调用时：fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)
        batch_data = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True)

        cfg_train = cfg.train['train_model']
         train_data = fun_data(data_sets, cfg_data['train_data'])
    '''
    # (1)基础参数配置

    num_workers = cfg['workers_num']  # 导入同时工作的线程数
    gpu_num = cfg['gpu_num']  # 几个GPU工作

    save_folder = cfg['save_folder']  # './weights/'
    save_name = cfg['save_name']  # 'model_CNN1D'

    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']  # 步进为60
    lr_adjust = cfg['lr_adjust']  # 设置为Ture

    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    if gpu_num > 1 and cfg['gpu_train']:
        # 采用多卡GPU服务器
        model = torch.nn.DataParallel(model).to(device)
        # 使用样例
        # model = model.cuda()
        # device_ids = [0, 1, 2, 3]  # id为0和1的两块显卡
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.to(device)

    # (2)加载模型开始训练

    model.train()

    # 是否采用上次训练的模型
    if cfg['reuse_model']:

        print('loading model')

        checkpoint = torch.load(cfg['reuse_file'], map_location=device)  # 用来加载torch.save() 保存的模型文件
        start_epoch = checkpoint['epoch']

        model_dict = model.state_dict()  # state_dict其实就是一个字典，该自点中包含了模型各层和其参数tensor的对应关系。

        pretrained_dict = {k: v for k, v in checkpoint['model'].item() if k in model_dict}  # 再用预训练模型参数更新model_dict
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)  # 装载已经训练好的模型
    else:
        start_epoch = 0

    batch_num = math.ceil(len(train_data) / batch_size)  # 向上取整，返回的是共有多少个训练次数的数目
    print('Start training!')

    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()  # 记录初始时间

        batch_data = DataLoader(train_data, batch_size, shuffle=True, \
                                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

        # 判断学习律是否采用步进的形式
        if lr_adjust:
            lr = adjust_lr(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        for batch_idx, batch_sample in enumerate(batch_data):  # 遍历加载的数据集

            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()

            # (1)导入图片和标签
            if len(batch_sample) > 3:
                img, target, indices, img_pca = batch_sample
                img_pca = img_pca.to(device)
            else:
                img, target, indices = batch_sample

            img = img.to(device)
            target = target.to(device)

            # (2)前向传播
            prediction_SSUN = model(img, img_pca)[0]
            prediction_lstm = model(img, img_pca)[1]
            prediction_MSCNN = model(img, img_pca)[2]

            # (3)计算损失
            loss1 = loss_fun(prediction_SSUN, target.long())  # 这里target应该是标签值
            loss2 = loss_fun(prediction_lstm, target.long())  # 这里target应该是标签值
            loss3 = loss_fun(prediction_MSCNN, target.long())  # 这里target应该是标签值

            loss = loss1 + loss2 + loss3

            # （4）优化器，反向传播
            optimizer.zero_grad()  # 将梯度归零
            loss.backward()  # 反向传播计算得到每个参数的梯度值
            optimizer.step()  # 通过梯度下降执行一步参数更新

            batch_time1 = time.time()
            batch_time = batch_time1 - batch_time0  # 在一个迭代中所花费的时间

            # estimated time of Arrival
            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)

            epoch_loss += loss.item()  # item()返回的是一个浮点型的数据

            predict_label = prediction_SSUN.detach().softmax(dim=-1).argmax(dim=1,
                                                                            keepdim=True)  # 返回一个新的从当前图中分离的Variable
            # 返回的 Variable 不会梯度更新。
            # 不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()  # 预测正确的数量之和
            label_num += len(target)

        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0  # 在一个epoch中所花费的时间
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        # 将相应的数据进行打印
        print('Epoch: {}/{} || lr: {} || loss: {} || Train acc: {:.2f}% || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss / batch_num, 100 * predict_correct / label_num,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))
                      )
              )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # 递归创建目录

    # 存储最终的模型
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))
