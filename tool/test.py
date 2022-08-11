import torch

from data.HSI_data import batch_collate as collate_fn
from torch.utils.data import DataLoader

import data.data_preprocess as pre_fun


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())  # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据
    model_keys = set(model.state_dict().keys())  # state_dict其实就是一个字典，该自点中包含了模型各层和其参数tensor的对应关系。
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'  # 警告位，当有用值小于0则发出警告

    return True


def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x  # 直接赋给一个变量，然后再像一般函数那样调用
    # eg: f(x)
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(test_data, origin_gt, model, device, cfg):
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']

    batch_size = cfg['batch_size']

    model = load_model(model, cfg['model_weights'], device)  # 加载模型，文件的格式是pth
    model.eval()
    model = model.to(device)

    # gpu_num  环境
    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)  # # 采用多卡GPU服务器
    else:
        model = model.to(device)

    batch_data = DataLoader(test_data, batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True)

    # 数据预定义
    predict_correct = 0
    label_num = 0
    predict_label = []

    for batch_idx, batch_sample in enumerate(batch_data):

        if len(batch_sample) > 3:
            img, target, indices, img_pca = batch_sample
            img_pca = img_pca.to(device)
        else:
            img, target, indices = batch_sample

        img = img.to(device)

        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        # 反向传播时不会自动求导了，大大节约了显存

        with torch.no_grad():
            prediction = model(img, img_pca)[0]

        label = prediction.softmax(dim=-1).cpu().argmax(dim=1, keepdim=True)

        if target.sum() > 0:
            predict_correct += label.eq(target.view_as(label)).sum().item()
            label_num += len(target)

        label = pre_fun.label_inverse_transform(label, origin_gt.long())
        predict_label.append(torch.cat([indices, label], dim=1))

    predict_label = torch.cat(predict_label, dim=0)

    if label_num > 0:
        print('OA {:.2f}%'.format(100 * predict_correct / label_num))

    return predict_label
