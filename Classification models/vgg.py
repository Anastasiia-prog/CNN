from torch import nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_convs=2):
        super().__init__()
        self.num_of_convs = num_of_convs
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if self.num_of_convs == 3:
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        if self.num_of_convs == 3:
            x = self.conv3(x)
            x = F.relu(x)
        x = self.pool1(x)
        
        return x

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = VGGBlock(in_channels=3, out_channels=64)
        self.convblock2 = VGGBlock(in_channels=64, out_channels=128)
        self.convblock3 = VGGBlock(in_channels=128, out_channels=256, num_of_convs=3)
        self.convblock4 = VGGBlock(in_channels=256, out_channels=512, num_of_convs=3)
        self.convblock5 = VGGBlock(in_channels=512, out_channels=512, num_of_convs=3)
        
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, 10)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        
        x = self.flat1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
