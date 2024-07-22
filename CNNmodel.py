import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self, n_feature):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(5 * 5 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_feature)

    def forward(self, x):
        # 假设输入图像尺寸为32*32*3
        x = F.relu(self.conv1(x))  # (输入高度 - 卷积核高度 + 2*填充) / 步长 + 1，步长默认为1
        # 32*32*3->28*28*16
        x = F.max_pool2d(x, 2)  # 池化层窗口大小为2x2，步长默认为2，(输入高度 / 步长)，(输入宽度 / 步长)
        # 28*28*16->14*14*16
        x = F.relu(self.conv2(x))
        # 14*14*16->10*10*32
        x = F.max_pool2d(x, 2)
        # 10*10*32->5*5*32

        x = x.view(-1, 5 * 5 * 32)  # 将tensor展平

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


