import torch
import torch.nn as nn


def Predict_Label2Img(predict_label):
    # predict_label torch.Size([207400, 4])
    # predict_img (610, 340)
    # indian_pines=([145, 145])  21025
    # KSC = ([512, 614])
    predict_img = torch.zeros([145, 145])
    num = predict_label.shape[0]  # 207400

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = predict_label[i][3]
        predict_img[x][y] = l

    return predict_img


if __name__ == '__main__':
    predict_label = torch.ones([21025, 4])
    predict_img = Predict_Label2Img(predict_label)
    print('over')
