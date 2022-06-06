from SEBlock import SE_Block
import torch.nn.functional as F
from torch import nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, add_se_block: bool=True, stride: int=1):
        super().__init__()
        self.add_se_block = add_se_block
        out_channels = mid_channels * 4
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1,
                               bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        
        if self.add_se_block:
            self.se_block = SE_Block(in_channels=out_channels)
        else:
            self.se_block = nn.Sequential()
        
    def forward(self):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.se_block(x)  
        
        return x

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
        
        if self.add_se_block:
            self.se_block = SE_Block(in_channels=out_channels)
        else:
            self.se_block = nn.Sequential()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.se_block(x)   
      
        return x

class SkipConn(nn.Module):
    def __init__(self, in_channels: int, desired_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=desired_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(desired_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, resnet_number: int, downsample: bool=False):
        super().__init__()
        self.downsample = downsample
        self.resnet_number = resnet_number
        
        if self.downsample:
            
            if self.resnet_number < 50:
                self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2)
                self.skip = SkipConn(in_channels=in_channels, desired_channels=out_channels)
            else: 
                self.conv = BottleneckBlock(in_channels=in_channels, mid_channels=out_channels, stride=2)
                self.skip = SkipConn(in_channels=in_channels, desired_channels=out_channels)
                
        else:   
            
            if self.resnet_number < 50:
                self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
                self.skip = nn.Identity()
            else:
                self.conv = BottleneckBlock(in_channels=in_channels, mid_channels=out_channels)
                self.skip = nn.Identity()
        
    def forward(self, x):
        
        out = self.conv(x)
        y = self.skip(x) + out
        y = F.relu(y)
        
        return y


class ResNet_body(nn.Module):
    def __init__(self, resnet_number: int):
        super().__init__()
        # resnet number can be 18, 34
        self.resnet_number = resnet_number
        number_blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3],
                         50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        channels = {'small': [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256,
                              256, 512, 512, 512],
                    'big': [64, 64, 256, 64, 256, 128, 512, 128, 512, 256,
                            1024, 256, 1024, 512, 2048, 512]}
        # save list with number of the blocks
        range_num = number_blocks[self.resnet_number]
        
        if self.resnet_number < 50:
            using_channels = channels['small']
        else:
            using_channels = channels['big']
        
        convblock2_1 = ResBlock(in_channels=using_channels[0], out_channels=using_channels[1], resnet_number=self.resnet_number)
        convblocks2 = nn.ModuleList([ResBlock(in_channels=using_channels[2], out_channels=using_channels[3],
                                              resnet_number=self.resnet_number) for i in range(range_num[0] - 1)])
        convblock2_n = nn.Sequential(*convblocks2)
        
        convblock3_1 = ResBlock(in_channels=using_channels[4], out_channels=using_channels[5], downsample=True, resnet_number=self.resnet_number)
        convblocks3 = nn.ModuleList([ResBlock(in_channels=using_channels[6], out_channels=using_channels[7],
                                              resnet_number=self.resnet_number) for i in range(range_num[1] - 1)])
        convblock3_n = nn.Sequential(*convblocks3)

        convblock4_1 = ResBlock(in_channels=using_channels[8], out_channels=using_channels[9], downsample=True, resnet_number=self.resnet_number)
        convblocks4 = nn.ModuleList([ResBlock(in_channels=using_channels[10], out_channels=using_channels[11], 
                                              resnet_number=self.resnet_number) for i in range(range_num[2] - 1)])
        convblock4_n = nn.Sequential(*convblocks4)

        convblock5_1 = ResBlock(in_channels=using_channels[12], out_channels=using_channels[13], downsample=True, resnet_number=self.resnet_number)
        convblocks5 = nn.ModuleList([ResBlock(in_channels=using_channels[14], out_channels=using_channels[15], 
                                              resnet_number=self.resnet_number) for i in range(range_num[3] - 1)])
        convblock5_n = nn.Sequential(*convblocks5)
        
        self.block = nn.Sequential(convblock2_1, convblock2_n,
                                   convblock3_1, convblock3_n,
                                   convblock4_1, convblock4_n,
                                   convblock5_1, convblock5_n)
    def forward(self, x):
        
        x = self.block(x)
        
        return x


class ResNetNN(nn.Module):
    def __init__(self, num_classes, resnet_number):
        super().__init__()
        self.num_classes = num_classes
        self.resnet_number = resnet_number
        sizes = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
        out_size = sizes[self.resnet_number]
        
        self.network = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     ResNet_body(self.resnet_number),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(),
                                     nn.Linear(out_size, self.num_classes))
        
    def forward(self, x):
        x = self.network(x)
        
        return x
