import numpy as np
import torch
import torch.nn as nn


def normalize_vector(v):  # 向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Dropout(nn.Module):
    def __init__(self, p=0.5):  # 在较深层网络中，dropout取值0.5的效果最好，在浅层网络中，dropout取值0.2较好
        super(Dropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
        self.kp = 1 - p

    def forward(self, x):
        if self.training:
            # 生成mask矩阵。
            # torch.rand_like：生成和x相同尺寸的张量，取值在[0,1)之间均匀分布。
            mask = (torch.rand_like(x) < self.kp)
            # 先用mask矩阵对x矩阵进行处理，再除以1 - p（保留概率），即上述所说的反向DropOut操作，不需要在测试集上再缩放。
            return x * mask / self.kp
        else:
            return x


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            # 式子算起来有些许区别。
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x
