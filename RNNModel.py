import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, h_n = self.rnn(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

