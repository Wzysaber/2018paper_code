import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTM import lstm
from model.MSCNN import MSCNN


class SSUN(nn.Module):
    def __init__(self):
        super(SSUN, self).__init__()
        self.spatial = lstm()
        self.spectal = MSCNN()

        self.outlayer = nn.Linear(6176, 16)

    def forward(self, img, img_gt):
        out_spatial = self.spatial(img)[0]
        out_spectal = self.spectal(img_gt)[0]

        out = torch.cat([
            out_spatial, out_spectal
        ], dim=1)

        out1 = self.outlayer(out)
        out2 = self.spatial(img)[1]
        out3 = self.spectal(img_gt)[1]

        return out1, out2, out3


def main():
    net = SSUN()
    img = torch.randn(64, 220, 24, 24)
    img_gt = torch.randn(64, 5, 24, 24)

    out = net(img, img_gt)[1]
    print(out.shape)


if __name__ == '__main__':
    main()
