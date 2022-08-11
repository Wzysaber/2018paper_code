import torch
import torch.utils.data as data



class HSI_data(data.Dataset):
    def __init__(self, data_sample, cfg):
        self.phase = cfg['phase']

        # img:pad_img
        # img_indices:每个patch的坐标合集
        self.img = data_sample['pad_img']
        self.img_indices = data_sample['pad_img_indices']
        self.gt = data_sample['img_gt']

        self.pca = 'img_pca_pad' in data_sample  # 判断'img_pca_pad'是否在data_sample中
        # 是的话返回 Ture，不是返回Falus

        if self.pca:
            self.img_pca = data_sample['img_pca_pad']
        # data_indices:用img_gt的标签信息划分得到的样本

        if self.phase == 'train':
            self.data_indices = data_sample['train_indices']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_indices']
        elif self.phase == 'no_gt':
            self.data_indices = data_sample['no_gt_indices']

    def __len__(self):
        return len(self.data_indices)

    # 将其中的函数根据相应的下标进行索引
    # 该方法支持从 0 到 len(self)的索引
    # data_indices=: torch.Size([270, 3])
    # img_indices=: torch.Size([207400, 4])
    def __getitem__(self, idx):

        index = self.data_indices[idx]
        img_index = self.img_indices[index[0]]  # img_index 坐标

        # 从pad_img中根据坐标截取样本
        img = self.img[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        label = self.gt[index[1], index[2]]

        if self.pca:
            img_pca = self.img_pca[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]

            # 存在pca的话，则从相应的图像中进行截取
            return img, label, index, img_pca
        else:
            return img, label, index


# 用来处理不同情况下的输入的dataset的封装
def batch_collate(batch):
    images = []
    labels = []
    indices = []
    images_pca = []

    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
        indices.append(sample[2])

        if len(sample) > 3:
            images_pca.append(sample[3])

    # stack将相应的数组进行连接
    if len(images_pca) > 0:
        return torch.stack(images, 0), torch.stack(labels), \
               torch.stack(indices), torch.stack(images_pca, 0)
    else:
        return torch.stack(images, 0), torch.stack(labels), \
               torch.stack(indices)
