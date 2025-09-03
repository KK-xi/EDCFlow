import torch.nn as nn
from torch import nn, einsum
from einops import rearrange
import torch

class SE(nn.Module):
    def __init__(self, features, M=2, r=16):
        super(SE, self).__init__()
        d = int(features / r)
        self.M = M
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(d, features, kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, feats):
        b, c, h, w = feats.shape

        feats_U = feats.reshape(b, -1, h, w)
        feats_S = self.gap(feats_U)
        attention_vectors = self.fc(feats_S)
        attention_vectors = self.sigmoid(attention_vectors)
        attention_vectors = attention_vectors.view(b, c, 1, 1)

        feats_V = feats * attention_vectors

        return feats_V



class MSE(nn.Module):
    def __init__(self, features, M=3, r=16):
        super(MSE, self).__init__()
        d = int(M*features / r)
        self.M = M
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(M*features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, feats):
        b, m, c, h, w = feats.shape

        feats_U = feats.reshape(b, -1, h, w)
        feats_S = self.gap(feats_U)
        fea_z = self.fc(feats_S)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.view(b, m, c, 1, 1)

        feats_V = feats * attention_vectors

        return feats_V
