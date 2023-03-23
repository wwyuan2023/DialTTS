# -*- coding: utf-8 -*-

"""HiFi-GAN Vocoder Module"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dialtts.vocoder.layers import GTU


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 3, 5),
    ):
        super(ResBlock1, self).__init__()
        
        assert kernel_size % 2 == 1
        assert channels % 16 == 0
        
        paddings = [ int((kernel_size - 1) * d / 2) for d in dilations ]
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size, stride=1,
                dilation=dilation, padding=padding
            ) for dilation, padding in zip(dilations, paddings)
        ])
        
        padding = int((kernel_size - 1) / 2)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size, stride=1,
                dilation=1, padding=padding
            ) for _ in dilations
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.fill_(0.0)
        
        self.apply(_reset_parameters)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE, False)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE, True)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 3, 5),
    ):
        super(ResBlock2, self).__init__()
        
        assert kernel_size % 2 == 1
        assert channels % 16 == 0
        
        paddings = [ int((kernel_size - 1) * d / 2) for d in dilations ]
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size, stride=1,
                dilation=dilation, padding=padding
            ) for dilation, padding in zip(dilations, paddings)
        ])
        
        padding = int((kernel_size - 1) / 2)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size, stride=1,
                dilation=1, padding=padding
            ) for _ in dilations
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.fill_(0.0)
        
        self.apply(_reset_parameters)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE, False)
            xt = c1(xt)
            xt = F.tanh(xt)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock3(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 3, 5),
    ):
        super(ResBlock3, self).__init__()
        
        assert kernel_size % 2 == 1
        assert channels % 16 == 0
        
        paddings = [ int((kernel_size - 1) * d / 2) for d in dilations ]
        cc = (max(1, channels//16) * 16) // 2
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, cc*2,
                kernel_size, stride=1,
                dilation=dilation, padding=padding
            ) for dilation, padding in zip(dilations, paddings)
        ])
        
        padding = int((kernel_size - 1) / 2)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                cc, channels,
                kernel_size, stride=1,
                dilation=1, padding=padding
            ) for _ in dilations
        ])
        
        self.gtu = GTU(dim=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.fill_(0.0)
        
        self.apply(_reset_parameters)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE, False)
            xt = c1(xt)
            xt = self.gtu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator Module"""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        resblock_class="ResBlock3",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
        upsample_initial_channels=512,
        upsample_scales=[2, 4, 6],
        use_weight_norm=True,
    ):
        super(HiFiGANGenerator, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_scales = upsample_scales
        self.upsample_factor = np.prod(upsample_scales)
        
        resblock = globals()[resblock_class]
        min_conv_channel = out_channels * 16
        
        # check hyper parameters is valid
        assert upsample_initial_channels >= np.prod(upsample_scales)
        assert upsample_initial_channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        
        # define initial layers
        self.pre_convs = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, 1, padding=int(kernel_size/2)),
            nn.ELU(inplace=True),
            nn.Conv1d(in_channels, upsample_initial_channels, 1)
        )
        
        # define upsampling network
        self.upsample_transpose = nn.ModuleList()
        self.upsample_interpole = nn.ModuleList()
        for i in range(len(upsample_scales)):
            upsample_scale = upsample_scales[i]
            in_c = max(min_conv_channel, upsample_initial_channels//(2**i))
            out_c = max(min_conv_channel, upsample_initial_channels//(2**(i+1)))
            self.upsample_transpose.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_c, out_c,
                        upsample_scale*2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                    )
                )
            )
            self.upsample_interpole.append(
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE, False),
                    nn.Upsample(scale_factor=upsample_scale, mode='nearest'),
                    nn.Conv1d(
                        in_c, out_c,
                        upsample_scale * 2 + 1,
                        stride=1,
                        padding=upsample_scale,
                    )
                )
            )
        
        # define resblocks
        self.resblock_convs = nn.ModuleList()
        for i in range(len(upsample_scales)):
            c = max(min_conv_channel, upsample_initial_channels//(2**(i+1)))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblock_convs.append(resblock(c, k, d))
        
        # define final layers
        self.post_convs = nn.Sequential(
            nn.ELU(inplace=False),
            nn.Conv1d(out_c, out_c, kernel_size, padding=int(kernel_size/2)),
            nn.ELU(inplace=True),
            nn.Conv1d(out_c, out_channels, 1),
            nn.Tanh(),
        )
        
        # init weight
        self.reset_parameters()
        
        # apply weight normalization
        if use_weight_norm:
            self.apply_weight_norm()
        
        # alias infer
        self.inference = self.infer
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
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
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)
        
        self.apply(_apply_weight_norm)
    
    def forward(self, c, steps=0):
        # c: (B, C, T)
        x = self.pre_convs(c)
        xc = x
        for i in range(len(self.upsample_scales)):
            x = F.leaky_relu(x, LRELU_SLOPE, True)
            x = self.upsample_transpose[i](x)
            xc = self.upsample_interpole[i](xc)
            x = x + xc
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblock_convs[i*self.num_kernels+j](x)
            x = xs / float(self.num_kernels)
        x = self.post_convs(x)
        return x
    
    @torch.no_grad()
    def infer(self, c):
        return self.forward(c) # (B,c,t)


if __name__ == "__main__":

    x = torch.randn(1, 48, 8)
    f = HiFiGANGenerator(48, 2, upsample_initial_channels=256)
    print(f)
    y = f(x)
    print(y.size())
    