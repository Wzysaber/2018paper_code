import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.io import loadmat
import spectral as spy

# 预先进行相应数据集的定义
datasets_type = 'HSI Data Sets'

# data_root = '/home/students/master/2022/wangzy/PyCharm-Remote/datasets/PaviaU.mat'
# image_name = 'paviaU'
# gt_name = 'paviaU_gt'
# paviaU的特征分为了9类
# torch([32, 103, 610, 340])

# data_root = '/home/students/master/2022/wangzy/PyCharm-Remote/datasets/Urban.mat'
# image_name = 'Urban'
# gt_name = 'R'
# Urban的特征分为了9类

data_root = '/home/students/master/2022/wangzy/PyCharm-Remote/datasets/Indian_pines.mat'
image_name = 'indian_pines'
gt_name = 'R'
# indian_pines的特征分为了16类
# torch([32, 220, 145, 145])

# data_root = '/home/students/master/2022/wangzy/PyCharm-Remote/datasets/KSC.mat'
# image_name = 'KSC'
# gt_name = 'KSC_gt'
# indian_pines的特征分为了13类
# torch([32, 176, 512, 614])


# 其他相应的参数进行配置

phase = ['train', 'test', 'no_gt']
pca_num = 5
train_set_num = 64
patch_size = 24

# 构建data字典，将所有的数据放在data中来进行调用
data = dict(
    data_path=data_root,
    image_name=image_name,
    gt_name=gt_name,
    train_set_num=train_set_num,
    patch_size=patch_size,
    pca=pca_num,
    train_data=dict(
        phase=phase[0]
    ),
    test_data=dict(
        phase=phase[1]
    ),
    no_gt_data=dict(
        phase=phase[2]
    )
)

# 建立相应模型的预参数

model = dict(
    in_fea_num=1,
    out_fea_num=9,
)

# 训练模型的预参数

lr = 1e-3

train = dict(
    # 优化器的相应数据
    optimizer=dict(
        typename='SGD',
        lr=lr,
        betas=(0.9, 0.999),
        momentum=0.9,  # 动量
        weight_decay=1e-4  # 权重衰减
    ),

    train_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=16,
        epoch=500,
        batch_size=64,
        # 学习率的相应参数
        lr=lr,
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=460,
        save_folder='./weights/',
        save_name='model_CNN1D',
        reuse_model=False,
        reuse_file='./weights/model_CNN1D_Final.pth'
    )
)

test = dict(
    batch_size=1000,
    gpu_train=True,
    gpu_num=1,
    workers_num=16,
    model_weights='./weights/model_CNN1D_Final.pth',
    save_folder='./result'
)


def main():
    data = loadmat(data_root)
    print(data.keys())


if __name__ == '__main__':
    main()
