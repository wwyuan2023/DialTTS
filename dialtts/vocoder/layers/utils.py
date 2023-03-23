# -*- coding: utf-8 -*-

# Copyright 2020 Alibaba Cloud
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    
    __constants__ = ['num_channels', 'eps', 'affine']
    num_channels: int
    eps: float
    affine: bool
    
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(ChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # input: (B,C,...)
        var, mean = torch.var_mean(input, dim=[1], unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            input_shape = input.shape
            input_view = [ 1, self.num_channels, *[1 for _ in input_shape[2:]] ]
            input = input * self.weight.view(*input_view) + self.bias.view(*input_view)
        
        return input

    def extra_repr(self):
        return '{num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class GTU(torch.nn.Module):
    
    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super(GTU, self).__init__()
        self.dim = dim

    def forward(self, input):
        xa, xb = torch.split(input, input.size(self.dim)//2, dim=self.dim)
        return torch.tanh(xa) * torch.sigmoid(xb)

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)
        

class Swish(torch.nn.Module):
    
    __constants__ = ['num_parameters']
    num_parameters: int
    
    def __init__(self, num_parameters=1, init=1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init))
    
    def forward(self, x):
        return x * torch.sigmoid(self.weight * x)
    
    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)



