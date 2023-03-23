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
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
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


class MaskedGroupNorm(nn.Module):
    
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(MaskedGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input, mask=None):
        # input: (B,C,...)
        # mask: (B,1,...), bool
        if mask is None:
            return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        
        # GroupNorm with mask
        input_shape = input.size()
        B, C = input_shape[0], input_shape[1]
        G = self.num_groups
        assert C % G == 0
        
        masked_fill = ~mask
        input = input.masked_fill(masked_fill, 0.)
        input = input.view(B, G, -1)
        
        ns = torch.sum(mask, dim=-1, keepdim=True).float() * C / G
        mean = torch.sum(input, dim=-1, keepdim=True) / ns
        var = torch.sum(input**2, dim=-1, keepdim=True) / ns - (mean**2)
        
        input = (input - mean) / torch.sqrt(var + self.eps)
        input = input.view(input_shape)
        
        if self.affine:
            input_shape = input.shape
            input_view = [ 1, self.num_channels, *[1 for _ in input_shape[2:]] ]
            input = input * self.weight.view(*input_view) + self.bias.view(*input_view)
        
        return input

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class GTU(nn.Module):
    
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
        

class Swish(nn.Module):
    
    __constants__ = ['num_parameters']
    num_parameters: int
    
    def __init__(self, num_parameters=1, init=1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init))
    
    def forward(self, x):
        return x * torch.sigmoid(self.weight * x)
    
    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class CondLayerNorm(nn.Module):
    
    __constants__ = ['input_size', 'embed_size', 'eps']
    input_size: int
    embed_size: int
    eps: float
    
    def __init__(
        self,
        input_size,
        embed_size,
        eps=1e-5,
    ):
        super(CondLayerNorm, self).__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.eps = eps
        self.scale_layer = nn.Linear(embed_size, input_size, bias=True)
        self.bias_layer = nn.Linear(embed_size, input_size, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        u = 1.0/float(self.embed_size)
        nn.init.uniform_(self.scale_layer.weight, 0, u)
        nn.init.uniform_(self.bias_layer.weight, -u, u)
        nn.init.zeros_(self.scale_layer.bias)
        nn.init.zeros_(self.bias_layer.bias)
    
    def forward(self, x, s):
        # input: (B,T,C)
        # s: speaker embeded, (B,C)
        y = F.layer_norm(x, (self.input_size,), eps=self.eps)
        
        scale = self.scale_layer(s)
        bias = self.bias_layer(s)
        y = y * scale.unsqueeze(1) + bias.unsqueeze(1) # (B,T,C)

        return y
    
    def extra_repr(self):
        return '{input_size}, embed_size={embed_size}, ' \
            'eps={eps}'.format(**self.__dict__)
    
