import torch.nn.functional as F
from torch import nn


class SE_Block(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int=16, activation=nn.ReLU()):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            activation,
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        
        return x * y.expand_as(x)
