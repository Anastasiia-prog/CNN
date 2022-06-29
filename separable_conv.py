from torch import nn


class SeparableConv2d(nn.Module):
    ''' depthwise + pointwise
        depthwise: 
        groups=in_channels, out_channels = kernel_size * in_channels '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        return out
