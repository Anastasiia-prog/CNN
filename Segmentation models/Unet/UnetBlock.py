import torch
from torch import nn
import torchvision
from SEBlock import SE_Block
import torch.nn.functional as F


class ConvBlock2x(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation=nn.ReLU(), padding=1):
        super().__init__(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding),
                         nn.BatchNorm2d(out_channels),
                         activation,
                         nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding),
                         nn.BatchNorm2d(out_channels),
                         activation)


class Encoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, block=ConvBlock2x):
        super().__init__(nn.MaxPool2d(2),
                          block(in_channels=in_channels, 
                                     out_channels=out_channels))

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, block=ConvBlock2x):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=2, stride=2)
        self.conv_block = block(in_channels=out_channels*2, out_channels=out_channels)
        
    
    def forward(self, x_down, x_up):
        x_up = self.upsample(x_up)
        
        b_down, c_down, h_down, w_down = x_down.shape
        b_up, c_up, h_up, w_up = x_up.shape
        if (h_up > h_down) or (w_up > w_down):
            raise ValueError("Up tensor must be smaller than down tensor")
        offset = ((h_down - h_up) // 2, (w_down - w_up) // 2)
        x_down_cropped = x_down[:, :, offset[0]:offset[0] + h_up, offset[1]:offset[1] + w_up]
        
        x = torch.cat((x_down_cropped, x_up), axis=1)
        result = self.conv_block(x)
        return result


class Unet(nn.Module):
    def __init__(self, num_classes=151, num_scales=4, base_filters=64, block=ConvBlock2x):
        '''
        num_scales: number of U-Net blocks, which changes the images sizes
        base_filters: channels number for the first level
        '''
        
        super().__init__()
        self.input_convolutions = ConvBlock2x(in_channels=3, out_channels=base_filters)
        
        layers = []
        filters = base_filters
        for i in range(num_scales):
            layers.append(Encoder(in_channels=filters, out_channels=filters*2, block=block))
            filters *= 2
        self.down_layers = torch.nn.Sequential(*layers)
        
        layers = []
        for i in range(num_scales):
            layers.append(Decoder(in_channels=filters, out_channels=filters//2, block=block))
            filters //= 2
        self.up_layers = torch.nn.Sequential(*layers)
        
        self.output_convolution = torch.nn.Conv2d(in_channels=filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
 
        down_results = [self.input_convolutions(x)]
        
        for layer in self.down_layers:
            down_results.append(layer(down_results[-1]))
        x = down_results[-1]
        
        for i, layer in enumerate(self.up_layers):
            x = layer(down_results[-2 - i], x)
        x = self.output_convolution(x)
        
        return x
