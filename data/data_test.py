import scipy.io as sio
import numpy as np
import configs.configs as cfg

def normalize(x, k):
    if k == 1:
        mu = np.mean(x, 0)
        x_norm = x - mu
        sigma = np.std(x_norm, 0)
        x_norm = x_norm / sigma
        return x_norm
    elif type == 2:
        minx = np.min(x, 0)
        maxx = np.max(x, 0)
        x_norm = x - minx
        x_norm = x_norm / (maxx - minx)
        return x_norm


def HyperFunctions(timestep=4, s1s2=2):
    cfg_data = cfg.data
    data_path = cfg_data['data_path']
    data = sio.loadmat(data_path)
    x = data['indian_pines']
    y = data['R']

    train_num_array = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50]
    train_num_array = np.array(train_num_array).astype('int')
    [row, col, n_feature] = x.shape
    x = x.reshape(row * col, n_feature)
    y = y.reshape(row * col, 1)
    # 16
    n_class = y.max()
    # 55
    nb_features = int(n_feature / timestep)
    # 1765
    train_num_all = sum(train_num_array)
    # (21025, 4, 55)
    x = normalize(x, 1)

    x_reshape = np.zeros((x.shape[0], timestep, nb_features))
    if s1s2 == 2:

        for j in range(0, timestep):
            x_reshape[:, j, :] = x[:, j:j + (nb_features - 1) * timestep + 1:timestep]
    else:
        for j in range(0, timestep):
            x_reshape[:, j, :] = x[:, j * nb_features:(j + 1) * nb_features]

    x_data_all = x_reshape

    randomarray = list()

    for i in range(1, n_class + 1):
        index = np.where(y == i)[0]
        n_data = index.shape[0]
        randomarray.append(np.random.permutation(n_data))

    flag1 = 0
    flag2 = 0
    # (1765, 4, 55)
    x_train = np.zeros((train_num_all, timestep, nb_features))

    x_test = np.zeros((sum(y > 0)[0] - train_num_all, timestep, nb_features))

    for i in range(1, n_class + 1):
        index = np.where(y == i)[0]
        # 46
        n_data = index.shape[0]
        # 30
        train_num = train_num_array[i - 1]
        randomx = randomarray[i - 1]
        if s1s2 == 2:

            for j in range(0, timestep):
                x_train[flag1:flag1 + train_num, j, :] = x[index[randomx[0:train_num]],
                                                         j:j + (nb_features - 1) * timestep + 1:timestep]
                x_test[flag2:flag2 + n_data - train_num, j, :] = x[index[randomx[train_num:n_data]],
                                                                 j:j + (nb_features - 1) * timestep + 1:timestep]

        else:
            for j in range(0, timestep):
                x_train[flag1:flag1 + train_num, j, :] = x[index[randomx[0:train_num]],
                                                         j * nb_features:(j + 1) * nb_features]
                x_test[flag2:flag2 + n_data - train_num, j, :] = x[index[randomx[train_num:n_data]],
                                                                 j * nb_features:(j + 1) * nb_features]

        flag1 = flag1 + train_num
        flag2 = flag2 + n_data - train_num
    # (1765, 4, 55) (8484, 4, 55)
    return x_data_all.astype('float32')


def main():
    out1 = HyperFunctions()
    print(out1.shape)



if __name__ == '__main__':
    main()
