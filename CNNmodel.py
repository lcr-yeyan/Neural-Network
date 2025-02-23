import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):  # 引入dropout防止过拟合方法
    def __init__(self, n_feature, dropout_rate=0.5):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(5 * 5 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_feature)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

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
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class Net(nn.Module):
    def __init__(self, c):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, c)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        # res = torch.reshape(conv3_out, (conv3_out.size(0), 64*3*3))
        out = self.dense(res)
        return out


class Net112(nn.Module):
    def __init__(self):
        super(Net112, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x112x112
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x28x28
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x14x14
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),  # fc4 64*14*14 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out


class Net224(nn.Module):
    def __init__(self):
        super(Net224, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x224x224
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x112x112
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x112x112
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x56x56
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x28x28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),  # 96x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)  # 96x14x14
        )

        self.dense = nn.Sequential(
            nn.Linear(96 * 14 * 14, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


class Net96(nn.Module):
    def __init__(self):
        super(Net96, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x96x96
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x48x48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x48x48
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x24x24
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x24x24
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x12x12
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),  # 96x12x12
            nn.ReLU(),
            nn.MaxPool2d(2)  # 96x6x6
        )

        self.dense = nn.Sequential(
            nn.Linear(96 * 6 * 6, 128),  # fc4 96*6*6 -> 128
            nn.ReLU(),
            nn.Linear(128, 4)  # fc5 128->4
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


class Net64x48(nn.Module):
    def __init__(self):
        super(Net64x48, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x64x48
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x32x24
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x16x12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x16x12
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x8x6
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 8 * 6, 128),  # fc4 64*8*6 -> 128
            nn.ReLU(),
            nn.Linear(128, 2)  # fc5 128->2
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


class AlexNet(nn.Module):
    def __init__(self, dropout_rate=0.5, n_feature=1000):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_feature),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x