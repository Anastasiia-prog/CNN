from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(kernel_size=2)
        
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(512*3*3, 4096)
        self.drop1 = nn.Dropout()
        
        self.drop2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        
        x = F.relu(x)
        x = self.maxp1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxp2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxp3(x)
        print(x.shape)
        x = self.flat1(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        return x
