import numpy as np
from activatefunction import sign


class Perceptron:  # 暂未完成
    def __init__(self, x, y, learning_rate=0.1, n_iters=1000):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置

    def train(self, error_threshold=0):
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            n_error = 0
            for idx, x_i in enumerate(self.x):
                y_output = sign(np.dot(x_i.T, self.weights) + self.bias)
                if y_output != self.y[idx]:
                    self.weights += self.y[idx] * x_i * self.lr
                    self.bias += self.y[idx] * self.lr
                    n_error += 1
            if n_error <= error_threshold:  # 停止条件
                # 此处是以设定的错误预测阈值的数量来作为停止条件
                # 用权重变化量作为停止条件效果会更好
                break

    def predict(self, x):
        y_neuron = np.dot(x.T, self.weights) + self.bias
        y_predict = sign(y_neuron)
        return y_predict
