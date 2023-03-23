# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.2


class PatchNet(nn.Module):
    def __init__(
        self,
        in_channels=48,
        out_channels=1,
        num_layers=4,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_channels=256,
        use_weight_norm=True,
    ):
        super(PatchNet, self).__init__()
        
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation > 0
        
        F = in_channels
        s0 = int(F ** (1.0 / float(num_layers)))
        s1 = stride
        k0 = s0 * 2 + 1
        k1 = kernel_size
        d0 = 1
        d1 = dilation
        cc = conv_channels
        
        convs = [
            nn.Conv2d(1, cc, (k0,k1), stride=(s0,s1), padding=0),
            nn.LeakyReLU(LRELU_SLOPE, True),
        ]
        F = int((F - k0) / s0 + 1)
        for i in range(num_layers - 2):
            convs += [
                nn.Conv2d(cc, cc, (k0,k1), stride=(s0,s1), dilation=(d0,d1), padding=0),
                nn.LeakyReLU(LRELU_SLOPE, True),
            ]
            F = int((F - k0) / s0 + 1)
        convs += [
            nn.Conv2d(cc, 1, (F,1), stride=(1,1), padding=0),
        ]
        
        self.convs = nn.Sequential(*convs)
        
        # reset parameters
        self.reset_parameters()
        
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)
                )
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        
        self.apply(_remove_weight_norm)
    
    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
        
        self.apply(_apply_weight_norm)
    
    def forward(self, x):
        # x: (B, C, T)
        x = self.convs(x.unsqueeze(1)) # (B, 1, C, T) -> (B, 1, 1, T')
        return x.squeeze_(1).squeeze_(2) # (B, T')


class MultiPatchNet(nn.Module):
    def __init__(
        self,
        in_channels=48,
        out_channels=1,
        num_layers=[4, 4, 4, 4],
        kernel_size=[5, 5, 5, 5],
        stride=[1, 1, 1, 1],
        dilation=[1, 2, 4, 8],
        conv_channels=[256, 256, 256, 256],
        use_weight_norm=[True, True, True, True],
    ):
        super(MultiPatchNet, self).__init__()
        
        self.nets = nn.ModuleList([
            PatchNet(
                in_channels, out_channels,
                num_layers=num_layers[i],
                kernel_size=kernel_size[i],
                stride=stride[i],
                dilation=dilation[i],
                conv_channels=conv_channels[i],
                use_weight_norm=use_weight_norm[i]
            ) for i in range(len(num_layers))
        ])
    
    def forward(self, x):
        outs = []
        for d in self.nets:
            outs.append(d(x))
        return outs
    

if __name__ == "__main__":
    
    x = torch.randn(1, 48, 100)
    f = MultiPatchNet(
        in_channels=48,
        out_channels=1,
        num_layers=[4, 4, 4, 4],
        kernel_size=[5, 5, 5, 5],
        stride=[1, 1, 1, 1],
        dilation=[1, 2, 4, 8],
        conv_channels=[256, 256, 256, 256],
        use_weight_norm=[True, True, True, True]
    )
    ys = f(x)
    print(f, x.size())
    for y in ys:
        print(y.size())
    print("Done!")






