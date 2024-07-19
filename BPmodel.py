import numpy as np
import activatefunction as acfun


class Neuron:
    def __init__(self, weight, bias):
        self.bias = bias
        self.weight = np.array(weight)

    def activate(self, input):
        return acfun.relu(np.dot(input, self.weight) + self.bias)
