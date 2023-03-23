# -*- coding: utf-8 -*-

"""warm-up and decay learning rate scheduler in: https://arxiv.org/pdf/1706.03762.pdf
"""

import torch
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, lr_floor=1e-9, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.lr_floor = lr_floor
        super(WarmupDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        step = max(self.last_epoch, 1)
        gamma = self.warmup_steps**0.5 * min(step * self.warmup_steps**-1.5, step**-0.5)
        return [max(self.lr_floor, base_lr*gamma) for base_lr in self.base_lrs]



