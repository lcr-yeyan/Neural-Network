import numpy as np


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


def random_rule():  # 水平暂不足
    return


def in_rule(alpha, x, w):  # 内星学习规则
    delta = alpha * (x - w)
    return delta


def out_rule(alpha, y, w):  # 外星学习规则
    delta = alpha * (y - w)
    return delta


def winner_takes_all_rule():  # 水平暂不足
    return

