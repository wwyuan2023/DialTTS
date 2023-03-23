
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearGLU(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super(LinearGLU, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features*2, bias=bias)
        self.glu = nn.GLU(-1)
        
        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain("sigmoid")
        )
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        return self.glu(self.linear(x))


class Conv1dGLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv1dGLU, self).__init__()
        
        if padding is None:
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.conv = nn.Conv1d(
            in_channels, out_channels*2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.glu = nn.GLU(1)
        
        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain("sigmoid")
        )
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)
    
    def forward(self, x, masked_fill=None):
        # x: (B, C, T)
        # masked_fill: (B, 1, T)
        x = self.glu(self.conv(x))
        if masked_fill is not None:
            x = x.masked_fill(masked_fill, 0.)
        return x

