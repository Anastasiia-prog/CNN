from SEBlock import SE_Block
import torch.nn.functional as F
from torch import nn

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, add_se_block: bool=True, stride: int=1, groups: int=1, 
                 downsample: torch.nn.modules.container.Sequential=nn.Sequential(),
                 base_width=64, bwidth: int=4):
        '''
        if groups is more than 1, we need to change the mid_channels and out_channels and then ResNet will be ResNext
        
        :param: bwidth: the whole number of th channels across all groups
        '''
        super().__init__() 
        mid_channels = int(out_channels * (base_width / 64))
        
        if groups > 1:
            self.expansion = 2
            mid_channels = groups * bwidth
            out_channels = groups * bwidth
            
        self.add_se_block = add_se_block
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1,
                               bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1,
                               bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(num_features=mid_channels)
        
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*self.expansion)
        
        if self.add_se_block:
            self.se_block = SE_Block(in_channels=out_channels*self.expansion)
        else:
            self.se_block = nn.Sequential()
            
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se_block(out)  
        
        out = out + self.downsample(x)
        out = F.relu(out)
        
        return out

class ConvBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, add_se_block: bool=True, stride: int=1, 
                 downsample: torch.nn.modules.container.Sequential=nn.Sequential(),
                 base_width=64):
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
  
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se_block(out) 
        
        out = out + self.downsample(x)
        out = F.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, block: object, layers: list, num_classes: int, start_channels: int, width_per_group=64):
        '''
        start_channels - how many channels in source images (for example for RGB images = 3)
        '''
        super().__init__()
        
        self.in_channels = 64
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(in_channels=start_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block=block, channels=64, blocks=layers[0])
        self.layer2 = self.make_layer(block=block, channels=128, blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block=block, channels=256, blocks=layers[2], stride=2)
        self.layer4 = self.make_layer(block=block, channels=512, blocks=layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flat = nn.Flatten()
        
    def make_layer(self, block, channels, blocks, stride=1):
        downsample = nn.Sequential()

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels * block.expansion, stride=stride, kernel_size=1),
                nn.BatchNorm2d(num_features=channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels=self.in_channels, out_channels=channels, stride=stride, downsample=downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels=self.in_channels, out_channels=channels,
                                base_width=self.base_width))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        
        return x

def ResNetNN(resnet_number: int, start_channels: int=3,  num_classes :int=10):
    resnets = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
               101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    
    layers = resnets[resnet_number]
    
    if resnet_number < 50:
        block = ConvBlock
    else:
        block = BottleneckBlock
    
    return ResBlock(block=block, layers=layers, num_classes=num_classes, start_channels=start_channels)
