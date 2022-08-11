import torch
import torch.nn as nn
from torchvision import transforms


# 将103通道的图降维成为3通道的图
def extract_pc(image, pc=3):
    '''
    :function:123
    :param image:
    :param pc:
    :return:
    '''
    channel, height, width = image.shape
    data = image.contiguous().reshape(channel, height * width)  # 存在contiguous函数，在改变data的值后
    data_c = data - data.mean(dim=1).unsqueeze(1)
    # 计算一个矩阵或一批矩阵 input 的奇异值分解
    u, s, v = torch.svd(data_c.matmul(data_c.T))  # data_c矩阵乘以data_c的转置
    sorted_data, indices = s.sort(descending=True)  # 将s中的数按降序进行排列
    image_pc = u[:, indices[0:pc]].T.matmul(data)
    return image_pc.reshape(pc, height, width)


# (x - mean(x))/std(x) normalize to mean: 0, std: 1
# 将数据进行归一化处理

def std_norm(image):
    image = image.permute(1, 2, 0).numpy()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
    ])

    return trans(image)


# (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel
def one_zero_norm(image):
    channel, height, width = image.shape
    data = image.view(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1)) / (data_max.unsqueeze(1) - data_min.unsqueeze(1))  # 在第二个维度上插入一个维度

    return data.view(channel, height, width)


# input tensor image size with CxHxW
# -1 + 2 * (x - min(x))/(max(x) - min(x))  normalize to (-1, 1) for each channel
# 同样对数据进行归一化处理,范围在（-1，1）
def pos_neg_norm(image):
    channel, height, width = image.shape
    data = image.view(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = -1 + 2 * (data - data_min.unsqueeze(1)) / (data_max.unsqueeze(1) - data_min.unsqueeze(1))

    return data.view(channel, height, width)


# function：construct sample,切分得到patch,储存每个patch的坐标值
# input: image：torch.size(103, 610, 340)
#         window_size：27
# output：pad_image, batch_image_indices
def construct_sample(image, window_size=27):
    # 先输入照片的通道数等数据的指标
    channel, height, width = image.shape

    half_window = int(window_size // 2)  # 13
    # 使用输入边界的复制值来填充
    pad = nn.ReplicationPad2d(half_window)  # 上下左右伸展13单位值，就是26
    # uses (padding_left, padding_right,padding_top, padding_bottom)

    pad_image = pad(image.unsqueeze(0)).squeeze(0)  # torch.Size([103, 636, 366])

    # 用数组存储切分得到的patch的坐标
    # torch.Size([207400, 4])
    batch_image_indices = torch.zeros((height * width, 4), dtype=torch.long)

    t = 0
    for h in range(height):
        for w in range(width):
            batch_image_indices[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_image, batch_image_indices


# 这里的gt是相应的标签值，背景图是0，相应的标志物为1到9
# 将gt的标签进行变化，使背景为-1，相应的标志物为0到9
def label_transform(gt):
    '''
        function：tensor label to 0-n for training
        input: gt
        output：gt
        tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        -> tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    '''
    label = torch.unique(gt)  # 返回标签label中的不同值
    gt_new = torch.zeros_like(gt)  # zeros_like(a)的目的是构建一个与a同维度的数组，并初始化所有变量为零。
    # zeros,则需要代入参数

    for each in range(len(label)):  # 长度为10
        indices = torch.where(gt == label[each])

        if label[0] == 0:
            gt_new[indices] = each - 1
        else:
            gt_new[indices] = each

    # tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    # labeL_new = torch.unique(gt_new)

    return gt_new


# 将标签值进行还原
def label_inverse_transform(predict_result, gt):
    label_origin = torch.unique(gt)
    label_predict = torch.unique(predict_result)

    predict_result_origin = torch.zeros_like(predict_result)
    for each in range(len(label_predict)):
        indices = torch.where(predict_result == label_predict[each])  # 此时他会返回等式相等时的坐标
        if len(label_predict) != len(label_origin):
            predict_result_origin[indices] = label_origin[each + 1]
        else:
            predict_result_origin[indices] = label_origin[each]

    return predict_result_origin


def select_sample(gt, ntr):
    '''
        function: 用img_gt的标签信息划分样本
        input: gt -> torch.Size(610, 340)；  ntr -> train_set_num 30
        output：data_sample = {'train_indices': train_indices, 'train_num': train_num,
                   'test_indices': test_indices, 'test_num': test_num,
                   'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0) }
    '''
    gt_vector = gt.reshape(-1, 1).squeeze(1)  # 使用reshape函数来对其进行重组, reshape(1,-1)转化成1行
    # torch.Size([207400])

    label = torch.unique(gt)

    first_time = True

    for each in range(len(label)):  # each 0~9
        indices_vector = torch.where(gt_vector == label[each])  # 返回1位的索引，也就标签值的具体位置
        # 将相应的标签进行遍历
        indices = torch.where(gt == label[each])  # 返回2维的索引,比如gt中-1的二维坐标

        # print(indices)
        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        # 相应的背景值为 -1
        if label[each] == -1:
            no_gt_indices = torch.cat([
                indices_vector.unsqueeze(1),
                indices_row.unsqueeze(1),
                indices_column.unsqueeze(1)],
                dim=1
            )
            no_gt_num = torch.tensor(len(indices_vector))

        # 其他标签 0-8
        else:
            class_num = torch.tensor(len(indices_vector))
            # each循环得到class_num：6631->18649->2099->3064->1345->5029->1330->3682->947
            # 在不同标签下得到的长度值

            # 得到选择的数量  ntr = train_set_num 30
            # if ntr < 1:  # 表现为百分数
            #     ntr0 = int(ntr * class_num)
            # else:
            #     ntr0 = ntr
            # # 最小值也得选10
            # if ntr0 < 10:
            #     sel_num = 10
            # elif ntr0 > class_num // 2:
            #     sel_num = class_num // 2
            # else:
            #     sel_num = ntr0

            train_num_array = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50]
            sel_num = train_num_array[each-1]

            sel_num = torch.tensor(sel_num)  # tensor(30)

            # 将标签进行打乱
            rand_indices0 = torch.randperm(class_num)  # torch.randperm 给定参数n，返回一个从0到n-1的随机整数排列
            rand_indices = indices_vector[rand_indices0]

            # 划分训练集train,测试集test
            # 划分打乱后的随机整数排列
            tr_ind0 = rand_indices0[0:sel_num]  # torch.Size([30])
            te_ind0 = rand_indices0[sel_num:]  # 将剩下的数据用作测试集

            # 划分随机整数排列对应的gt
            tr_ind = rand_indices[0:sel_num]  # torch.Size([30])
            te_ind = rand_indices[sel_num:]

            # 训练集train: 索引+坐标
            sel_tr_ind = torch.cat([
                tr_ind.unsqueeze(1),
                indices_row[tr_ind0].unsqueeze(1),
                indices_column[tr_ind0].unsqueeze(1)],
                dim=1
            )  # torch.Size([30, 3])

            # 测试集test
            sel_te_ind = torch.cat([
                te_ind.unsqueeze(1),
                indices_row[te_ind0].unsqueeze(1),
                indices_column[te_ind0].unsqueeze(1)],
                dim=1
            )  # torch.Size([6601, 3])

            if first_time:
                first_time = False

                train_indices = sel_tr_ind
                train_num = sel_num.unsqueeze(0)

                test_indices = sel_te_ind
                test_num = (class_num - sel_num).unsqueeze(0)

            else:
                train_indices = torch.cat([train_indices, sel_tr_ind], dim=0)
                train_num = torch.cat([train_num, sel_num.unsqueeze(0)])

                test_indices = torch.cat([test_indices, sel_te_ind], dim=0)
                test_num = torch.cat([test_num, (class_num - sel_num).unsqueeze(0)])

    # 训练集
    rand_tr_ind = torch.randperm(train_num.sum())
    train_indices = train_indices[rand_tr_ind,]
    # 测试集
    rand_te_ind = torch.randperm(test_num.sum())  # torch.Size([42506])
    test_indices = test_indices[rand_te_ind,]  # torch.Size([42506, 3])
    # 背景
    rand_no_gt_ind = torch.randperm(no_gt_num.sum())  # torch.Size([164624])
    no_gt_indices = no_gt_indices[rand_no_gt_ind,]  # torch.Size([164624, 3])

    # 将6种数据参数进行保存
    data_sample = {'train_indices': train_indices, 'train_num': train_num,
                   'test_indices': test_indices, 'test_num': test_num,
                   'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0)
                   }

    return data_sample



