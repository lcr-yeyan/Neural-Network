import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        # bidirectional为双向RNN启动参数
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x ：[batch_size, sequence_length, input_size]，即批量大小、序列长度、输入数据维度
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        # h0：[num_layers * num_directions, batch_size, hidden_size]，即层数乘以方向数、批量大小、隐藏层大小
        out, h_n = self.rnn(x, h_0)
        # h_n形状与h_0相同
        out = self.fc(out[:, -1, :])
        # out：[batch size，sequence_length，hidden_size]
        return out
