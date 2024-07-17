import time


class Neuron:
    def __init__(self, bias=0.0, activation_function=lambda x: x):
        self.inputs = []
        self.bias = bias
        self.act_fun = activation_function

    def add_connection(self, weight, source_neuron):  # 添加某个神经元的输入连接
        self.inputs.append((weight, source_neuron))

    def activate(self):  # 激活函数
        sum_weighted = sum(weight * source.activate() for weight, source in self.inputs) + self.bias
        return self.act_fun(sum_weighted)


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
        return 1 if sum_weighted > self.threshold else 0  # 激活函数为阶跃函数


class DelayMPNeuron:  # 并未实现功能
    def __init__(self, weights=None, delay_length=1, threshold=0):
        if weights is None:
            weights = []
        self.weights = weights
        self.threshold = threshold
        self.delay_length = delay_length  # 延迟长度，即记忆的时间步长数
        self.memory = [0.0] * delay_length  # 存储最近几个时间步长的加权输入和
        self.current_index = 0  # 当前时间步长在记忆中的索引
        self.in_connection = []

    def add_connection(self, weight, in_neuron):  # 添加某个神经元的输入连接
        self.in_connection.append((weight, in_neuron))

    def activate(self):
        # 更新记忆
        self.memory[self.current_index] = sum(weight * source.activate() for weight, source in self.in_connection)
        self.current_index = (self.current_index + 1) % self.delay_length  # 循环更新索引
        end_index = self.current_index
        if end_index < self.delay_length:
            # 如果当前索引小于延迟长度，窗口跨越了数组的开始
            start_index = 0
        else:
            # 否则，窗口完全在数组内
            start_index = end_index - self.delay_length
        weights_input = sum(self.memory[i] for i in range(start_index, end_index))
        return 1 if weights_input > self.threshold else 0  # 转移函数为阶跃函数


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

