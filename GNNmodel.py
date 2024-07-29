import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SimpleGCN, self).__init__()
        # 第一层GCN
        self.conv1 = GCNConv(num_features, hidden_channels)
        # 第二层GCN，直接映射到类别数
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层图卷积 + ReLU激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层图卷积
        x = self.conv2(x, edge_index)

        # 通常，对于节点分类任务，我们不需要对节点特征进行求和或平均等聚合
        return F.log_softmax(x, dim=1)
