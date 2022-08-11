# 纯数学操作来进行相应的变化

import numpy as np


def fun_norm(X, M='AllNorm'):
    if X.dtype != float:
        X = X.astype(np.float32)

    if type(M) != str: #如果不是字符型还要将其转化
        M = str(M)

    if M.lower() == 'allnorm':  #字符串中的大写字母转化为小写字母
        X_min_value = X.min()
        X_max_value = X.max()
        if X_min_value == X_max_value:
            return X
        else:
            X0 = X - X_min_value
            Xd = X_max_value - X_min_value + 0.001

            return 0.1 + (1 - 0.9) * (X0 / Xd)

    elif M.lower() == 'rownorm':
        X_min_row = X.min(axis=1)  #axis =0 沿行方向， axis =1 沿列的方向
        X_max_row = X.max(axis=1)
        X0 = X - np.tile(X_min_row, (X.shape[1], 1)).T  # np.tile按照各个方向复制
        Xd = X_max_row - X_min_row + 0.001
        return 0.1 + (1 - 0.9) * X0 * np.tile(1./Xd, (X.shape[1], 1)).T

    elif M.lower() == 'colnorm':
        X_min_col = X.min(axis=0)
        X_max_col = X.max(axis=0)
        X0 = X - np.tile(X_min_col, (X.shape[0], 1)).T
        Xd = X_max_col - X_min_col + 0.001
        return 0.1 + (1 - 0.9) * X0 * np.tile(1. / Xd, (X.shape[0], 1)).T

    elif M.lower() == '2norm':
        X_2norm = np.linalg.norm(X, axis=1)  # 求行向量的范数
        return X * (1. / np.tile(X_2norm, (X.shape[1], 1))).T

    elif M.lower() == 'stdnorm':
        X_mean = X.mean(axis=0)
        X0 = X - np.tile(X_mean, (X.shape[0], 1))
        X_std = X.std(axis=0) + 0.001  # axis=0计算每一列的标准差
        return X * (1. / np.tile(X_std, (X.shape[0], 1)))

    else:
        return X
