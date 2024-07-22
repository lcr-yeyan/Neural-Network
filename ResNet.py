import torch.nn as nn
import torch.nn.functional as F


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    # 定义空网络结构
    blk = []
    for i in range(num_residuals):
        # 第2,3,4个Inception块的第一个残差模块连接1x1卷积层
        if i == 0 and not first_block:
            blk.append(BasicResNet(input_channels, num_channels,
                                   use_1x1conv=True, strides=2))
        else:
            blk.append(BasicResNet(num_channels, num_channels))
    return blk


class ResNet1(nn.Module):
    def __init__(self, c):
        super(ResNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1)  # 64x14x14
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1)  # 64x7x7
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, c)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out) + conv1_out
        conv2_out = self.pool(self.relu(conv2_out))
        conv3_out = self.conv3(conv2_out) + conv2_out
        conv3_out = self.pool(self.relu(conv3_out))
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        features = self.dense[0](res)
        features_relu = self.dense[1](features)
        out = self.dense[2](features_relu)

        if self.training:
            return out
        else:
            return out, features


class BasicResNet(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super(BasicResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
