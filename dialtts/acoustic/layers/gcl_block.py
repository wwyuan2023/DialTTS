
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import Conv1dGLU


class GCLBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hiddend_size,
        kernel_size,
        padding=None,
        groups=1,
        bias=True,
        dropout_rate=0.,
    ):
        super(GCLBlock, self).__init__()
        
        assert kernel_size % 2 == 1
        if padding is None: padding = int(kernel_size / 2)
        
        self.convs = nn.Sequential(
            nn.Conv1d(input_size, hiddend_size, kernel_size, groups=groups, padding=padding, bias=bias),
            nn.Dropout(dropout_rate, True),
            nn.ReLU(True),
            Conv1dGLU(hiddend_size, input_size, 1)
        )
        self.norm = nn.LayerNorm(input_size)
    
    def forward(self, x, masked_fill=None):
        # x: (B, C, T)
        # masked_fill: (B, 1, T)
        x = x + self.convs(x)
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        if masked_fill is not None:
            x = x.masked_fill(masked_fill, 0.)
        return x


