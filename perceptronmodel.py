import numpy as np
from activatefunction import sign


class Perceptron:
    def __init__(self, X, y, learning_rate=0.01, n_iters=1000):
        self.X = X
        self.y = y
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = 0  # 权重
        self.bias = 0  # 偏置

    def train(self):
        for i in range(self.n_iters):
            n_error = 0
            for j in range(self.X.shape[0]):
                sign(n_error)
