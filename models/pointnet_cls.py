import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):   # k=40 分类类别数目
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3  # xyz
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)  # 计算对数概率
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        '''
            NLLLoss的输入是一个对数概率向量和一个目标标签.它不会计算对数概率.
            适合网络的最后一层是log＿softmax.
            损失函数 nn.CrossEntropyLoss()与NLLLoss()相同，唯一的不同是它去做softmax.
        '''
        loss = F.nll_loss(pred, target)  # 分类损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)  # 特征变换正则化损失

        # 总损失
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
