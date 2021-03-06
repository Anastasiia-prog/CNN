from SEBlock import SE_Block
from torch import nn
import torch
import math


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, groups=1, activation=nn.SiLU(), post_activation=True):
        super().__init__(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False, groups=groups),
                         nn.BatchNorm2d(num_features=out_channels),
                         activation if post_activation else nn.Identity())

class Inverted_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, padding, reduction_ratio=4, stride=1,
                 survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob
        
        expanded_channels = expansion_factor * in_channels
        self.skip_connection = (in_channels == out_channels) and (stride == 1) 
        # Inverted Bottleneck
        expand = nn.Identity() if (expansion_factor == 1) else ConvBlock(in_channels=in_channels,
                                                                         out_channels=expanded_channels,
                                                                         kernel_size=1,
                                                                         padding=0)
        
        self.mvblock = nn.Sequential(expand,
                                     ConvBlock(in_channels=expanded_channels, out_channels=expanded_channels,
                                               kernel_size=kernel_size, stride=stride, 
                                               groups=expanded_channels, padding=padding),            # Depthwise
                                     SE_Block(in_channels=expanded_channels, reduction_ratio=reduction_ratio,
                                              activation=nn.SiLU()),                                   # Squeeze and excitation
                                     ConvBlock(in_channels=expanded_channels, out_channels=out_channels,
                                               kernel_size=1,  post_activation=False, padding=0)) # Pointwise
        
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        batch_size = len(x)
        random_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        bit_mask = self.survival_prob < random_tensor

        x = x.div(1 - self.survival_prob)
        x = x * bit_mask
        
        return x
        
    def forward(self, x):
        out = self.mvblock(x)
        
        if self.skip_connection:
            out = x + self.stochastic_depth(out)
                
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, start_channels=3,
                 channels_factor=1., depth_factor=1.,
                 base_depths=[1, 2, 2, 3, 3, 4, 1],
                 base_channels = [(32, 16), (16, 24), (24, 40),
                                  (40, 80), (80, 112), (112, 192),
                                  (192, 320), (320, 1280)],
                 survival_probs = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171],
                 kernel_sizes = [3, 3, 5, 3, 5, 5, 3],
                 strides = [1, 2, 2, 2, 1, 2, 1]):
        
        '''
        param: base_channels: the number of channels
        param: channels_factor: coefficient for changing the base channels
        param: base_depths: the number of repeats by default
        param: depth_factor: coefficient for changing the base repeats for every layer
        param: survival_probs: probability for stochastic depth
        '''
        
        super().__init__()
        scaled_channels = [(self.scale_channels(channels[0], channels_factor),
                            self.scale_channels(channels[1], channels_factor)) 
                            for channels in base_channels]
        
        scaled_depths = [math.ceil(depth_factor*d) for d in base_depths]

        self.stem = ConvBlock(in_channels=start_channels, out_channels=scaled_channels[0][0], stride=2, padding=1, kernel_size=3)

        stages = []
        for i in range(7):
            expansion_factor = 1 if (i == 0) else 6
            reduction_ratio = 4 if (i == 0) else 24
            stage = self.create_stage(*scaled_channels[i], scaled_depths[i],
                                      expansion_factor=expansion_factor, kernel_size=kernel_sizes[i], 
                                      stride=strides[i], reduction_ratio=reduction_ratio, 
                                      survival_prob=survival_probs[i])
            stages.append(stage)
            
        self.stages = nn.Sequential(*stages)
        
        pre_classifier = ConvBlock(*scaled_channels[-1], kernel_size=1, padding=0)

        self.classifier = nn.Sequential(pre_classifier,
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Dropout(),
                                        nn.Flatten(),
                                        nn.Linear(scaled_channels[-1][1], num_classes))
        
    def create_stage(self, in_channels, out_channels, num_layers, expansion_factor, 
                     kernel_size=3, stride=1, reduction_ratio=24, survival_prob=0):

        padding = kernel_size // 2

        layers = []
        
        for i in range(num_layers):
            current_channels = in_channels if i == 0 else out_channels
            stride = stride if i == 0 else 1
            layer = Inverted_bottleneck(in_channels=current_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=padding, stride=stride, reduction_ratio=reduction_ratio, 
                                        survival_prob=survival_prob, expansion_factor=expansion_factor)
            layers.append(layer)
            
        layers = nn.Sequential(*layers)
        
        return layers

    def scale_channels(self, channels, channels_factor):
        
        channels *= channels_factor
        modified_channels = (int(channels + 4) // 8) * 8
        modified_channels = max(8, modified_channels)
        # check that round didn't down by more than 10 %
        if modified_channels < 0.9 * channels:
            modified_channels += 8
            
        return modified_channels

    def forward(self, x):
        
        x = self.stem(x)
        x = self.stages(x)
        x = self.classifier(x)
        
        return x
