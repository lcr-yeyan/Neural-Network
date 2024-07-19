import numpy as np
from usefulfunction import normalize_vector


# y为实际输出，d为期望输出，alpha为学习速率
def Hebb(alpha, x, y):  # 无监督学习
    delta = alpha * np.dot(x.T, y) / int(x.shape[0])
    return delta


def delta_rule(alpha, d, x, y, f_delta):  # 有监督学习，f_delta为在y值上转移函数的导数
    delta = alpha * np.dot(x.T, d - y) / int(x.shape[0]) * f_delta
    return delta


def lms(alpha, d, x, f_y):  # 有监督学习，f_y为在y值上转移函数的值
    delta = alpha * np.dot(x.T, d - f_y) / int(x.shape[0])
    return delta


def perceptron_rule(alpha, d, x, f_y):  # 有监督学习，f_y是阈值型函数在y值上的值
    delta = alpha * np.dot(x.T, d - f_y) / int(x.shape[0])
    return delta


def correlation_rule(alpha, x, d):  # 有监督学习
    delta = alpha * np.dot(x.T, d) / int(x.shape[0])
    return delta


def random_rule():  # 随机型神经网络学习规则
    return


def in_rule(alpha, x, w):  # 内星学习规则
    delta = alpha * (x - w)
    return delta


def out_rule(alpha, y, w):  # 外星学习规则
    delta = alpha * (y - w)
    return delta


def winner_takes_all(X, num_neurons, learning_rate, num_iterations):  # from Wenxinyiyan
    weights = np.random.rand(num_neurons, X.shape[1])
    for _ in range(num_iterations):
        for input_vector in X:
            # 归一化输入向量
            input_vector_normalized = normalize_vector(input_vector)

            # 计算每个神经元的激活值（这里使用点积作为激活函数）
            activations = np.dot(weights, input_vector_normalized)

            # 找到激活值最高的神经元（胜者）
            winner_index = np.argmax(activations)

            # 网络输出：可以是胜者神经元的权重，或者是胜者神经元的激活值
            # 这里我们选择胜者神经元的权重作为输出（也可以根据实际情况选择）
            output = weights[winner_index]

            # 更新胜者神经元的权重
            weights[winner_index] += learning_rate * (input_vector_normalized - weights[winner_index])
            # 输出当前胜者的信息
            print(f"Winner index: {winner_index}, Activation: {activations[winner_index]}, Output: {output}")
    return weights

