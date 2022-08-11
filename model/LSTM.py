import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM 将相应的数据进行学习记忆


class lstm(nn.Module):
    def __init__(self, band_num=4, chose_model=1):
        super(lstm, self).__init__()
        self.band = band_num
        self.chose_model = chose_model
        self.lstm_model = nn.LSTM(
            input_size=55,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        # h0 = torch.randn(2, 3, 6)
        # c0 = torch.randn(2, 3, 6)  在调用
        self.outlayer = nn.Sequential(
            nn.Linear(128, 50),
            nn.Linear(50, 16)
        )

    def forward(self, x):
        b, c, h_size, w_size = x.shape
        input = x[:, :, h_size // 2, w_size // 2]
        input = input.reshape(b, c)

        nb_features = int(c // self.band)
        input_reshape = torch.zeros((x.shape[0], self.band, nb_features)).type_as(x)

        if self.chose_model == 1:
            for j in range(0, self.band):
                input_reshape[:, j, :] = input[:, j:j + (nb_features - 1) * self.band + 1:self.band]
        else:
            for j in range(0, self.band):
                input_reshape[:, j, :] = input[:, j * nb_features:(j + 1) * nb_features]

        out, (h0, c0) = self.lstm_model(input_reshape)
        out = out[:, -1, :]  # torch.Size([32, 128])

        out_org = out
        out_finnal = self.outlayer(out)
        return out_org, out_finnal


def main():
    net = lstm(4, 1)
    tmp = torch.randn(128, 220, 24, 24)

    out = net(tmp)[1]
    print(out.shape)


if __name__ == '__main__':
    main()
