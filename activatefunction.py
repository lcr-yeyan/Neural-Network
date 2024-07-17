import math


def sigmoid(k, x):
    return 1 / (1 + math.exp(-x * k))


def relu(x):
    return 0 if x <= 0 else x


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def softplus(x):
    return math.log10(1 + math.exp(x))


def elu(alpha, x):
    return x if x > 0 else alpha * (math.exp(x) - 1)