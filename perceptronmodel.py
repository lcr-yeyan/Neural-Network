import numpy as np
from activatefunction import sign


class Perceptron:   # 基础感知机模型
    def __init__(self, x, y, learning_rate=0.1, n_iters=500):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置

    def train(self, error_threshold=0):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            n_error = 0
            for idx, x_i in enumerate(self.x):
                y_output = sign(np.dot(x_i.T, self.weights) + self.bias)  # 计算初始偏差
                if y_output != self.y[idx]:  # 特征判定
                    self.weights += self.y[idx] * x_i * self.lr
                    self.bias += self.y[idx] * self.lr
                    n_error += 1
            if n_error <= error_threshold:  # 停止条件
                # 此处是以设定的错误预测阈值的数量来作为停止条件
                # 用权重变化量、学习步长骤减作为停止条件效果会更好
                break

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict


class SGDPerceptron:   # 基础感知机模型，SGD优化
    def __init__(self, x, y, learning_rate=0.1, n_iters=500, batch_size=20):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置
        self.batch_size = batch_size

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i + self.batch_size] for i in range(0, self.x.shape[0], self.batch_size)]
        return batches

    def train(self):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            for batch in self.batches():
                for x_b, y_b in batch:
                    y_output = sign(np.dot(x_b, self.weights) + self.bias)  # 计算初始偏差
                    if y_output != y_b:  # 特征判定
                        self.weights += y_b * x_b * self.lr
                        self.bias += y_b * self.lr

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict


class PocketPerceptron:   # 基础感知机模型，基于SGD优化，使用口袋算法(暂未完成)
    def __init__(self, x, y, learning_rate=0.1, n_iters=500, batch_size=20):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置
        self.batch_size = batch_size

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i + self.batch_size] for i in range(0, self.x.shape[0], self.batch_size)]
        return batches

    def train(self):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            for batch in self.batches():
                for x_b, y_b in batch:
                    y_output = sign(np.dot(x_b, self.weights) + self.bias)  # 计算初始偏差
                    if y_output != y_b:  # 特征判定
                        self.weights += y_b * x_b * self.lr
                        self.bias += y_b * self.lr

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict


class KernelPerceptron:   # 基础感知机模型，基于SGD优化，使用核感知机算法(暂未完成)
    def __init__(self, x, y, learning_rate=0.1, n_iters=500, batch_size=20):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置
        self.batch_size = batch_size

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i + self.batch_size] for i in range(0, self.x.shape[0], self.batch_size)]
        return batches

    def train(self):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            for batch in self.batches():
                for x_b, y_b in batch:
                    y_output = sign(np.dot(x_b, self.weights) + self.bias)  # 计算初始偏差
                    if y_output != y_b:  # 特征判定
                        self.weights += y_b * x_b * self.lr
                        self.bias += y_b * self.lr

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict


class VotePerceptron:   # 基础感知机模型，基于SGD优化，使用表决感知机算法(暂未完成)
    def __init__(self, x, y, learning_rate=0.1, n_iters=500, batch_size=20):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置
        self.batch_size = batch_size

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i + self.batch_size] for i in range(0, self.x.shape[0], self.batch_size)]
        return batches

    def train(self):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            for batch in self.batches():
                for x_b, y_b in batch:
                    y_output = sign(np.dot(x_b, self.weights) + self.bias)  # 计算初始偏差
                    if y_output != y_b:  # 特征判定
                        self.weights += y_b * x_b * self.lr
                        self.bias += y_b * self.lr

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict


class BeePerceptron:   # 基础感知机模型，基于SGD优化，使用蜂群感知机算法(暂未完成)
    def __init__(self, x, y, learning_rate=0.1, n_iters=500, batch_size=20):
        self.x = x  # 训练样本
        self.y = y  # 训练标签
        self.lr = learning_rate  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = 0  # 偏置
        self.batch_size = batch_size

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i + self.batch_size] for i in range(0, self.x.shape[0], self.batch_size)]
        return batches

    def train(self):  # 模型的训练函数
        n_samples, n_features = self.x.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            for batch in self.batches():
                for x_b, y_b in batch:
                    y_output = sign(np.dot(x_b, self.weights) + self.bias)  # 计算初始偏差
                    if y_output != y_b:  # 特征判定
                        self.weights += y_b * x_b * self.lr
                        self.bias += y_b * self.lr

    def predict(self, x):  # 模型的执行函数
        y_predict = sign(np.dot(x, self.weights) + self.bias)
        return y_predict

