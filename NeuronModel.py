import time
from activatefunction import step
import numpy as np


class Neuron:
    def __init__(self, bias=0.0, activation_function=lambda x: x):
        self.inputs = []
        self.bias = bias
        self.act_fun = activation_function

    def add_connection(self, weight, in_neuron):  # 添加某个神经元的输入连接
        self.inputs.append((weight, in_neuron))

    def activate(self):  # 激活函数
        sum_weighted = sum(weight * source.activate() for weight, source in self.inputs) + self.bias
        return self.act_fun(sum_weighted)


class Neuron:
    def __init__(self, weights, bias, activation='sigmoid'):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.activation_functions = {
            'sigmoid': self.sigmoid,
            'relu': self.relu
        }
        if activation not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation_functions[self.activation](weighted_sum)


class NeuronLayer:
    def __init__(self, num_neurons, input_size, activation_function='sigmoid'):
        self.neurons = []
        for _ in range(num_neurons):
            # 随机初始化权重和偏置（这里简单使用随机小数，实际应用中可能需要更复杂的初始化策略）
            weights = np.random.rand(input_size)
            bias = np.random.rand(1)[0]
            self.neurons.append(Neuron(weights, bias, activation_function))

    def forward(self, inputs):
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return outputs


class MPNeuron:
    def __init__(self, weights=None, threshold=0):
        if weights is None:
            weights = []
        self.weights = weights
        self.threshold = threshold
        self.in_connection = []

    def add_connection(self, weight, in_neuron):  # 添加某个神经元的输入连接
        self.in_connection.append((weight, in_neuron))

    def activate(self):
        sum_weighted = sum(weight * source.activate() for weight, source in self.in_connection)
        return step(sum_weighted, self.threshold)  # 激活函数为阶跃函数


class DelayMPNeuron:  # 功能实现，但不够强大
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.in_connection = {}

    def add_connection(self, weight, delay, in_neuron):  # 添加某个神经元的输入连接
        self.in_connection[in_neuron] = (weight, delay)

    def activate(self, inputs_value, t):  # inputs_value为当前结点输入神经元的值字典，键为神经元，值为对应的输入, t为时间量
        sum_weighted = 0
        for inputs_neuron, params in self.in_connection.items():
            weight, delay = params
            sum_weighted += inputs_value[inputs_neuron] * weight * (t - delay)
        return step(sum_weighted, self.threshold)  # 激活函数为阶跃函数


class RefractoryMPNeuron:  # 并未实现功能
    def __init__(self, weights=None, threshold=1.0, refractory_period=0.5):
        self.threshold = threshold  # 激活阈值
        self.current_potential = 0  # 当前电位
        self.refractory_period = refractory_period  # 不应期长度
        self.last_fired = None  # 上次激活的时间标记
        self.inputs = []
        self.weights = weights

    def add_connection(self, weight, source_neuron):  # 添加某个神经元的输入连接
        self.inputs.append((weight, source_neuron))

    def stimulate(self, stimulus):
        current_time = time.time()
        if self.last_fired is not None and (current_time - self.last_fired) < self.refractory_period:
            print("The neuron is in a refractory period")
            return

        self.current_potential += stimulus
        if self.current_potential >= self.threshold:
            self.fire()

    def fire(self):
        print("Neuron fired")
        self.current_potential = 0
        self.last_fired = time.time()
