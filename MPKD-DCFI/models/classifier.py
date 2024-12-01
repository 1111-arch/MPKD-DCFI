from __future__ import print_function

import torch.nn as nn
import torchvision.models as models
import torch

#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    # def __init__(self, dim_in, n_label=20):
    #     super(LinearClassifier, self).__init__()
    #
    #     self.net = nn.Linear(dim_in, n_label)
    #     self.n_label = n_label
    #     self.n = False

    # def set_n_to_True(self):
    #     for i in range(self.n_label):
    #         if i % 2 == 0:
    #             with torch.no_grad():
    #                 self.net.bias[i] += 10
    #     self.net.bias = nn.Parameter(self.net.bias)

    # def set_n_to_True(self):
    #     for i in range(0, self.n_label):
    #         if i<10:
    #             if i % 3 == 0:
    #                 with torch.no_grad():
    #                     self.net.bias[i] += 10
    #         else:
    #             if i % 2 == 0:
    #                  with torch.no_grad():
    #                     self.net.bias[i] += 10
    #
    #     self.net.bias = nn.Parameter(self.net.bias)
    def __init__(self, dim_in, n_label=20):
        super(LinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 256),  # 增加隐藏层
            nn.BatchNorm1d(256),  # 批量归一化
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(256, n_label),
        )
        self.n_label = n_label
        self.n = False
        #
        # self.net = nn.Sequential(
        #     nn.Linear(dim_in, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, n_label),
        # )


    def set_n_to_True(self):
            for i in range(self.n_label):
                if i % 2 == 0:
                    with torch.no_grad():
                        self.net.bias[i] += 10
            self.net.bias = nn.Parameter(self.net.bias)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    # def __init__(self, dim_in, n_label=10, p=0.1):
    #     super(NonLinearClassifier, self).__init__()
    #
    #     self.net = nn.Sequential(
    #         nn.Linear(dim_in, 200),
    #         nn.Dropout(p=p),
    #         nn.BatchNorm1d(200),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(200, n_label),
    #     )



    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 256),  # 增加隐藏层的神经元数量
            nn.BatchNorm1d(256),     # 批量归一化
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 使用 LeakyReLU
            nn.Dropout(p=p),
            nn.Linear(256, 128),     # 添加额外的隐藏层
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=p),
            nn.Linear(128, n_label),
        )




    def forward(self, x):
        return self.net(x)
