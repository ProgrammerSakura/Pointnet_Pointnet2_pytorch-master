import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# Tnet模型
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()

        # mlp
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # channel 3 或 6
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # fc
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 9=3*3->3*3矩阵
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # x-> B N C
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 相当于 最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # 展平

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # 9

        # iden生成单位变换矩阵
        # repeat 表示复制，from_numpy表示转换为tensor类型数据。
        # view 表示改变维度。
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # 仿射变换
        x = x.view(-1, 3, 3)  # view展平为batchsize*3*3的张量
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):  # k=64表示输入的点云特征为64维度
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)  # 返回一个k*k的变换矩阵
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # 3*3 T-net
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat   # 判断是分类还是分割
        self.feature_transform = feature_transform
        if self.feature_transform:   # 特征变换矩阵
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()  # B 点云个数 D 特征个数(3 xyz 或 6 (xyz坐标 + 法向量)) N (一个物体所取点的个数)
        trans = self.stn(x)
        x = x.transpose(2, 1)  # 交换维度->(B N D) 即 n*3
        if D > 3:
            feature = x[:, :, 3:]  # 取特征
            x = x[:, :, :3]  # 坐标 xyz
        '''
            对输入的点云进行输入转换（input transform） 
            input transform：计算两个tensor的矩阵乘法
            bmm是两个三维张量相乘，两个输入tensor维度是(BxNxD)和(BxDxD）， 
            第一维b代表batch size.输出为（BxNxD）
        '''
        x = torch.bmm(x, trans)  # 点云乘以变换矩阵
        # 特征拼接，恢复原状
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)  # B D N
        x = F.relu(self.bn1(self.conv1(x)))  # MLP

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)  # B N D
            x = torch.bmm(x, trans_feat)  # NxD * DxD->NxD
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # 得到局部特征
        # MLP(64, 128, 1024)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # 最大池化得到全局特征
        x = x.view(-1, 1024)  # 展平
        if self.global_feat:  # global_feat=True则返回全局特征
            return x, trans, trans_feat
        else:  # 否则返回全局特征与局部特征的拼接做分割任务。
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 对特征矩阵正则化
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]  # torch．eye(n，m＝None，out＝None)返回一个2维张量，对角线位置全1，其它位置全0
    if trans.is_cuda:
        I = I.cuda()
    # 正则化损失函数
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
