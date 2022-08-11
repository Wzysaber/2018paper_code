import torch
import torch.nn as nn
import torch.nn.functional as F


# MSCNN，将patch的图片进行相应的带入，经过卷积池化来进行相应的各个部分的全连接
# 图片是经过了PCA处理降维了，只具有5个通道数

class MSCNN(nn.Module):
    def __init__(self):
        super(MSCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.outlayer = nn.Linear(6048, 16)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.pool1(out1)

        out2 = self.conv2(out1)
        out2 = self.pool2(out2)

        out3 = self.conv3(out2)
        out3 = self.pool3(out3)

        batchsz1 = out1.size(0)
        batchsz2 = out2.size(0)
        batchsz3 = out3.size(0)
        out1 = out1.view(batchsz1, -1)
        out2 = out2.view(batchsz2, -1)
        out3 = out3.view(batchsz3, -1)

        out = torch.cat([
            out1, out2, out3
        ], dim=1)

        # print(out1.shape, out2.shape, out3.shape)

        out_org = out
        out_finnal = self.outlayer(out)

        return out_org, out_finnal


def main():
    net = MSCNN()
    tmp = torch.randn(64, 5, 24, 24)
    out1 = net(tmp)[1]

    print(out1.shape)


if __name__ == '__main__':
    main()
