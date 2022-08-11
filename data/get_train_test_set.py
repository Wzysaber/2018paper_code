import torch
import scipy.io as io
import data_preprocess as pre_fun


def get_train_test_set(cfg):
    '''
        function： (1)划分数据集train,test
                   (2)加载数据集，转化为tensor，label transform，
                   (3)切分patch，储存每个patch的坐标值，
                   (4)由gt划分样本，最终得到data_sample

        input: cfg,也就是在config中编辑相应的数据

        output：输出由gt进行划分的data_sample

            # dict_keys(['train_indices', 'train_num', 'test_indices', 'test_num',
            # 'no_gt_indices', 'no_gt_num', 'pad_img', 'pad_img_indices', 'img_gt', 'ori_gt'])
    '''

    # 从cfg中导入设定好的参数
    data_path = cfg['data_path']  # 导入存放的地址
    image_name = cfg['image_name']  # paviaU
    # 这个其实就是103通道的整个图像的信息

    gt_name = cfg['gt_name']  # 'paviaU_gt'
    # 这个是地图的特征信息，同时具有标签值
    # [0,0,1,1,1,4,4,4,]这种的标签图

    train_set_num = cfg['train_set_num']  # 30,每一次数据集训练的次数
    patch_size = cfg['patch_size']  # 27,用于切分图像的尺寸

    # 加载数据高光谱的数据集
    # 加载的数据变化形式，先进行.astype('float32')再进行.from_numpy(img)，就转化为torch.Size的格式
    data = io.loadmat(data_path)  # 从相应的文件夹导入

    img = data[image_name].astype('float32')  # .astype转换数组的数据类型  (610, 340, 103) [w,h,c]
    gt = data[gt_name].astype('float32')  # 转换成float32  (610, 340) ,这个数据从数据库中导入，只有一个数据

    img = torch.from_numpy(img)  # 转tensor   # torch.Size(610, 340, 103)
    gt = torch.from_numpy(gt)  # torch.Size(610, 340)

    img = img.permute(2, 0, 1)  # 变换tensor的维度,把channel放到第一维CxHxW  # torch.Size(103, 610, 340)
    img = pre_fun.std_norm(img)  # 归一化，torch.Size(103, 610, 340) 将数据分布在（0，1）之间

    # label transform  0~9 -> -1~8
    # 将标签值进行转换，应该在mat文件中，对不同的物体的label值就做好了定义
    img_gt = pre_fun.label_transform(gt)  # torch.size(610, 340)

    # construct_sample：切分patch，储存每个patch的坐标值
    # img_pad的值为([103, 636, 366]),
    # img_pad_indices的值为([207400, 4])
    img_pad, img_pad_indices = pre_fun.construct_sample(img, patch_size)

    # (1)select_sample：用img_gt的标签信息划分样本
    # (2)得到的data_sample = {'train_indices': train_indices, 'train_num': train_num,
    #                    'test_indices': test_indices, 'test_num': test_num,
    #                    'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0)
    #                    }
    data_sample = pre_fun.select_sample(img_gt, train_set_num)

    # data_sample再添加几项数据
    data_sample['pad_img'] = img_pad
    data_sample['pad_img_indices'] = img_pad_indices
    data_sample['img_gt'] = img_gt  # 转化后的特征标签图的数据
    data_sample['ori_gt'] = gt  # 原始特征标签图的数据

    # print('data_sample.keys()',data_sample.keys())
    # dict_keys(['train_indices', 'train_num', 'test_indices', 'test_num',
    # 'no_gt_indices', 'no_gt_num', 'pad_img', 'pad_img_indices', 'img_gt', 'ori_gt'])

    # 在预处理中cfg['pca']=1，在我的理解里就是是否执行的标志位
    # 属于是buff叠满了
    # 将图像进行扩展，归0化和，标准差
    if cfg['pca'] > 0:
        img_pca = pre_fun.extract_pc(img, cfg['pca'])
        img_pca = pre_fun.one_zero_norm(img_pca)
        img_pca = pre_fun.std_norm(img_pca)

        img_pca_pad, _ = pre_fun.construct_sample(img_pca, patch_size)

        data_sample['img_pca_pad'] = img_pca_pad

    return data_sample
