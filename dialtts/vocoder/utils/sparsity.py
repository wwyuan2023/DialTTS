# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""GRU Sparsity class."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class GRUHiddenSparsity(nn.Module):
    """Structured parsity on GRU hidden units of LPCNet, Sparse method reference to below, https://github.com/mozilla/LPCNet/blob/107cedbdddfa4cf911a8ae7952a4a5b13081b787/training_tf2/lpcnet.py#L72
    
    Args:
        module (nn.Module): GRU module.
        densities (list): quantity of parameters to prune.
        
    """
    def __init__(
        self,
        module,
        blocks=(4,8),
        densities=(0.1, 0.1, 0.2),
    ):
        super(GRUHiddenSparsity, self).__init__()
        assert len(densities) == 3, f"Length of densities must be equal to 3"
        
        name = "weight_hh_l0"
        weight = getattr(module, name)
        hidden_size = weight.shape[1]
        assert hidden_size % 8 == 0, f"Hidden size of GRU must be an integer multiple of 8"
        
        module = prune.identity(module, name)
        
        self.name = name
        self.blocks = tuple(blocks)
        self.densities = tuple([min(1.0, d) for d in densities])
    
    def __call__(self, module, steps, start_steps=40000, end_steps=100000, interval=1000):

        if not(self.training) or not(start_steps <= steps <= end_steps) or ((steps - start_steps) % interval != 0):
            return
        
        orig = getattr(module, self.name + "_orig")
        mask = getattr(module, self.name + "_mask")
        weight = orig.detach() * mask # (3*N, N)
        N = weight.shape[1]
        weights = torch.split(weight, N, dim=0)
        diags = torch.diag_embed(torch.ones(N, device=weight.device)) # (N,N)
        
        blocks = self.blocks
        densities = self.densities
        new_masks = [None] * 3
        
        for k in range(len(densities)):
            density, A = densities[k], weights[k]
            if steps < end_steps:
                r = 1. - (steps - start_steps) / (end_steps - start_steps)
                density = 1. - (1. - densities[k]) * (1. - r**3)
            A = A - torch.diag_embed(torch.diag(A))
            L = A.reshape(N//blocks[1], blocks[1], N//blocks[0], blocks[0])
            S = torch.sum(L**2, -1)
            S = torch.sum(S, 1)
            SS, _ = torch.sort(S.view(-1))
            thresh = SS[round(SS.shape[0]*(1.-density))]
            new_mask = (S>=thresh).float()
            new_mask = torch.repeat_interleave(new_mask, blocks[1], dim=0)
            new_mask = torch.repeat_interleave(new_mask, blocks[0], dim=1)
            new_mask = torch.clamp(new_mask + diags, max=1.)
            new_masks[k] = new_mask
            #print(f"/////// k={k}, density={density} ", torch.sum(new_mask) / (N*N))
        
        mask.data = torch.cat(new_masks, dim=0).data
        #print(f"/////// total density={density} ", torch.sum(mask) / (N*N))
        return
    
    def extra_repr(self) -> str:
        return 'blocks={}, densities={}'.format(self.blocks, self.densities)


class GRUInputSparsity(nn.Module):
    """Structured parsity on GRU input units of LPCNet, Sparse method reference to below, https://github.com/mozilla/LPCNet/blob/ef368da47c2a05282eefe6b8689402f4bc18de08/training_tf2/lpcnet.py#L130
    
    Args:
        module (nn.Module): GRU module.
        densities (list): quantity of parameters to prune.
        
    """
    def __init__(
        self,
        module,
        input_size,
        blocks=(4,8),
        densities=(0.5, 0.5, 1.0),
    ):
        super(GRUInputSparsity, self).__init__()
        assert len(densities) == 3, f"Length of densities must be equal to 3"
        
        name = "weight_ih_l0"
        weight = getattr(module, name)
        input_size = min(weight.shape[1], input_size)
        assert weight.shape[0] % 8 == 0 and input_size % 4 == 0, \
            f"The shape of weight must be an integer multiple of 8x4"

        module = prune.identity(module, name)
        
        self.name = name
        self.input_size = input_size
        self.blocks = tuple(blocks)
        self.densities = tuple([min(1.0, d) for d in densities])
    
    def __call__(self, module, steps, start_steps=40000, end_steps=100000, interval=1000):

        if not(self.training) or not(start_steps <= steps <= end_steps) or ((steps - start_steps) % interval != 0):
            return
        
        orig = getattr(module, self.name + "_orig")
        mask = getattr(module, self.name + "_mask")
        weight = orig.detach() * mask # (3*N, M)
        N, M, M2 = weight.shape[0] // 3, weight.shape[1], self.input_size
        weights = torch.split(weight, N, dim=0)
        ones = torch.ones((N,M-M2), device=weight.device) if M > M2 else None
        
        blocks = self.blocks
        densities = self.densities
        new_masks = [None] * 3
        
        for k in range(len(densities)):
            density, A = densities[k], weights[k]
            if steps < end_steps:
                r = 1. - (steps - start_steps) / (end_steps - start_steps)
                density = 1. - (1. - densities[k]) * (1. - r**3)
            A2 = A[:, -M2:]
            L = A2.reshape(N//blocks[1], blocks[1], M2//blocks[0], blocks[0])
            S = torch.sum(L**2, -1)
            S = torch.sum(S, 1)
            SS, _ = torch.sort(S.view(-1))
            thresh = SS[round(SS.shape[0]*(1.-density))]
            new_mask = (S>=thresh).float()
            new_mask = torch.repeat_interleave(new_mask, blocks[1], dim=0)
            new_mask = torch.repeat_interleave(new_mask, blocks[0], dim=1)
            new_masks[k] = new_mask if ones is None else torch.cat((ones, new_mask), dim=1)
            #print(f"/////// k={k}, density={density} ", torch.sum(new_mask) / (N*N))
        
        mask.data = torch.cat(new_masks, dim=0).data
        #print(f"/////// total density={density} ", torch.sum(mask) / (N*N))
        return
    
    def extra_repr(self) -> str:
        return 'input_size={}, blocks={}, densities={}'.format(self.input_size, self.blocks, self.densities)

