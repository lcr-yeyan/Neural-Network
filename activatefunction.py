import math


def sigmoid(k, x):
    return 1 / (1 + math.exp(-x * k))


def d_sigmoid(k, x):
    return sigmoid(k, x) * (1 - sigmoid(k, x))


def relu(x):
    return 0 if x <= 0 else x


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def d_tanh(x):
    return 1 - tanh(x) ** 2


def softplus(x):
    return math.log10(1 + math.exp(x))


def elu(alpha, x):
    return x if x > 0 else alpha * (math.exp(x) - 1)


def lrelu(alpha, x):
    return x if x > 0 else alpha * x


def step(input, threshold):
    return 1 if input > threshold else 0
