import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return 0 if x <= 0 else x


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


