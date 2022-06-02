from SEBlock import SE_Block
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_se_block: bool=True, stride: int=1):
        super().__init__()
        self.add_se_block = add_se_block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.se_block = SE_Block(in_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.add_se_block:
            x = self.se_block(x)   
      
        return x

class SkipConn(nn.Module):
    def __init__(self, in_channels: int, desired_channels: int):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=desired_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(desired_channels)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool=False, stride: int=1):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2)
            self.skip = SkipConn(in_channels=in_channels, desired_channels=out_channels)
        else:   
            self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
            self.skip = nn.Identity()
        
        
    def forward(self, x):
        
        out = self.conv(x)
        y = self.skip(x) + out
        y = F.relu(y)
        
        return y
