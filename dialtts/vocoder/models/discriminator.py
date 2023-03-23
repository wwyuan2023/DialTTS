# -*- coding: utf-8 -*-

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, Conv2d, LeakyReLU
from torch.nn.utils import weight_norm, spectral_norm

try:
    from pytorch_wavelets import DWT1DForward 
except:
    warnings.warn("You must install `pytorch_wavelets` first."
          "Reference: https://pytorch-wavelets.readthedocs.io/en/latest/readme.html#installation", UserWarning)


LRELU_SLOPE = 0.2


def _stft(x, fft_size, hop_size, win_size, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, fft_size//2+1, T//hop_size+1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))


class PerioDiscriminator(nn.Module):
    """Resolution Period Discriminator module"""
    def __init__(
        self,
        period,
        use_weight_norm=True,
    ):
        super().__init__()
        self.period = period
        fnorm = weight_norm if use_weight_norm else spectral_norm
        self.convs1 = nn.ModuleList([
            fnorm(Conv2d(  1,  64, (5,1), stride=(2,1), padding=(2,0))),
            fnorm(Conv2d( 64, 128, (5,1), stride=(2,1), padding=(2,0))),
            fnorm(Conv2d(128, 256, (5,1), stride=(2,1), padding=(2,0))),
        ])
        self.convs2 = nn.ModuleList([
            fnorm(Conv1d(2, 1, 1)),
            fnorm(Conv1d(4, 1, 1)),
            fnorm(Conv1d(8, 1, 1)),
        ])
        self.convs3 = nn.ModuleList([
            fnorm(Conv2d(1,  64, 1)),
            fnorm(Conv2d(1, 128, 1)),
            fnorm(Conv2d(1, 256, 1)),
        ])
        self.convs4 = nn.Sequential(
            fnorm(Conv2d(256, 256, (3,1), stride=(1,1))), LeakyReLU(LRELU_SLOPE, True),
            fnorm(Conv2d(256,  64, (3,1), stride=(1,1))), LeakyReLU(LRELU_SLOPE, True),
            fnorm(Conv2d( 64,   1, (3,1), stride=(1,1))),
        )
        self.dwt = DWT1DForward(J=1, wave='db1', mode='reflect')
    
    def _reshape_period(self, x):
        B, C, T = x.size()
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad))
        x = x.view(B, C, -1, self.period)
        return x
    
    def forward(self, x):
        wavelet = [x]
        x = self._reshape_period(x)
        for i, (f1,f2,f3) in enumerate(zip(self.convs1, self.convs2, self.convs3)):
            x = f1(x) # (B,C,H,W) -> (B,C',H//2,W)
            x = F.leaky_relu(x, LRELU_SLOPE, False)
            nw = []
            for w in wavelet:
                cA, cD = self.dwt(w)
                nw.append(cA)
                nw.append(cD[0])
            wavelet = nw
            xw = torch.cat(nw, dim=1)
            xw = f2(xw)
            xw = self._reshape_period(xw)
            xw = f3(xw)
            x += xw
        
        x = self.convs4(x)
        return x.view(x.size(0), 1, -1)


class MultiPerioDiscriminator(nn.Module):
    """Multi-Resolution Period Discriminator module"""
    def __init__(
        self,
        periods=[48,32,24,16,8],
        use_weight_norm=True,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            self.discriminators += [PerioDiscriminator(period, use_weight_norm)]
        
    def forward(self, x):
        outs = []
        for d in self.discriminators:
            outs.append(d(x))
        return outs


class WaveDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        layers=10,
        conv_channels=64,
        use_weight_norm=True,
    ):
        super().__init__()
        
        fnorm = weight_norm if use_weight_norm else spectral_norm
        convs = [
            fnorm(Conv1d(in_channels, conv_channels, 1, padding=0, dilation=1)),
            LeakyReLU(LRELU_SLOPE, True), 
        ]
        for i in range(layers - 2):
            convs += [
                fnorm(Conv1d(conv_channels, conv_channels, kernel_size, padding=0, dilation=i+2)),
                LeakyReLU(LRELU_SLOPE, True), 
            ]
        convs += [ 
            fnorm(Conv1d(conv_channels, 1, 1, padding=0, dilation=1))
        ]
        self.convs = nn.Sequential(*convs)
        
        if not use_weight_norm:
            self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, Conv1d) or isinstance(m, Conv2d):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)
                )
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
        
    def forward(self, x):
        return self.convs(x).squeeze(1)


class MultiWaveDiscriminator(nn.Module):
    def __init__(
        self,
        num_dwt=5,
        kernel_size=3,
        layers=10,
        conv_channels=32,
        use_weight_norm=True,
    ):
        super().__init__()
        self.num_dwt = num_dwt
        self.discriminators = nn.ModuleList([
            WaveDiscriminator(
                2**i,
                kernel_size,
                layers,
                conv_channels,
                use_weight_norm=use_weight_norm
            ) for i in range(num_dwt)
        ])
        self.dwt = DWT1DForward(J=1, wave='db1', mode='reflect')

    def forward(self, x):
        wavelet = [x]
        outs = []
        for i, d in enumerate(self.discriminators):
            outs.append(d(x))
            if i == self.num_dwt: break
            nw = []
            for w in wavelet:
                cA, cD = self.dwt(w) # w:(B,c,T), cA/cD[0]:(B,c,ceil(T/2))
                nw.append(cA)
                nw.append(cD[0])
            wavelet = nw
            x = torch.cat(nw, dim=1) # xw:(B,c*2,ceil(T/2))
        return outs


class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_size=1024,
        window="hann_window",
        num_layers=4,
        kernel_size=3,
        stride=1,
        conv_channels=256,
        use_weight_norm=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', getattr(torch, window)(win_size), persistent=True)
        
        fnorm = weight_norm if use_weight_norm else spectral_norm
        F = fft_size//2 + 1
        s0 = int(F ** (1.0 / float(num_layers)))
        s1 = stride
        k0 = s0 * 2 + 1
        k1 = kernel_size
        cc = conv_channels
        
        convs = [
            fnorm(Conv2d(1, cc, (k0,k1), stride=(s0,s1), padding=0)),
            LeakyReLU(LRELU_SLOPE, True),
        ]
        F = int((F - k0) / s0 + 1)
        for i in range(num_layers - 2):
            convs += [
                fnorm(Conv2d(cc, cc, (k0,k1), stride=(s0,s1), padding=0)),
                LeakyReLU(LRELU_SLOPE, True),
            ]
            F = int((F - k0) / s0 + 1)
        convs += [
            fnorm(Conv2d(cc, 1, (F,1), stride=(1,1), padding=0)),
        ]
        
        self.convs = nn.Sequential(*convs)
        
        if not use_weight_norm:
            self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, Conv1d) or isinstance(m, Conv2d):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)
                )
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
        
    def forward(self, x):
        # x: (B, 1, t)
        x = _stft(x.squeeze(1), self.fft_size, self.hop_size, self.win_size, self.window)
        x = x.unsqueeze(1) # (B, 1, F, T)
        x = self.convs(x) # (B, 1, F, T) -> (B, 1, 1, T')
        return x.squeeze_(1).squeeze_(2) # (B, T')


class MultiSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        window="hann_window",
        num_layers=[4, 4, 4],
        kernel_sizes=[3, 3, 3],
        conv_channels=[256, 256, 256],
        use_weight_norm=True,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                fft_size=fft_size,
                hop_size=hop_size,
                win_size=win_size,
                window=window,
                num_layers=num_layer,
                kernel_size=kernel_size,
                conv_channels=conv_channel,
                use_weight_norm=use_weight_norm
            ) for fft_size, hop_size, win_size, num_layer, kernel_size, conv_channel in \
                zip(fft_sizes, hop_sizes, win_sizes, num_layers, kernel_sizes, conv_channels)
        ])
    
    def forward(self, x):
        outs = []
        for d in self.discriminators:
            outs.append(d(x))
        return outs


class Discriminators1(MultiWaveDiscriminator):
    def __init__(
        self,
        num_dwt=5,
        kernel_size=3,
        layers=10,
        conv_channels=32,
        use_weight_norm=True,
    ):
        super(Discriminators1, self).__init__(
            num_dwt=num_dwt,
            kernel_size=kernel_size,
            layers=layers,
            conv_channels=conv_channels,
            use_weight_norm=use_weight_norm,
        )
        
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).

        Returns:
            Tensor: List of output tensor (B, T')

        """
        return super(Discriminators1, self).forward(x)


class Discriminators2(MultiSTFTDiscriminator):
    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        window="hann_window",
        num_layers=[4, 4, 4],
        kernel_sizes=[3, 3, 3],
        conv_channels=[256, 256, 256],
        use_weight_norm=True,
    ):
        super(Discriminators2, self).__init__(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_sizes=win_sizes,
            window=window,
            num_layers=num_layers,
            kernel_sizes=kernel_sizes,
            conv_channels=conv_channels,
            use_weight_norm=use_weight_norm,
        )
        
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).

        Returns:
            Tensor: List of output tensor (B, T')

        """
        return super(Discriminators2, self).forward(x)

class Discriminators3(nn.Module):
    def __init__(
        self,
        multi_wave_discriminator_params={
            "num_dwt": 5,
            "kernel_size": 5,
            "layers": 10,
            "conv_channels": 64,
            "use_weight_norm": True,
        },
        multi_stft_discriminator_params={
            "fft_sizes": [64, 128, 256, 512, 1024, 2048, 4096],
            "hop_sizes": [16, 32, 64, 128, 256, 512, 1024],
            "win_sizes": [64, 128, 256, 512, 1024, 2048, 4096],
            "num_layers": [4, 5, 6, 7, 8, 9, 10],
            "kernel_sizes": [5, 5, 5, 5, 5, 5, 3],
            "conv_channels": [64, 64, 64, 64, 64, 64, 64],
            "use_weight_norm": True,
        },
    ):
        super().__init__()
        self.mwd = MultiWaveDiscriminator(**multi_wave_discriminator_params)
        self.mfd = MultiSTFTDiscriminator(**multi_stft_discriminator_params)
        
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).

        Returns:
            Tensor: List of output tensor (B, T')

        """
        return self.mwd(x) + self.mfd(x)

class Discriminators4(nn.Module):
    def __init__(
        self,
        multi_period_discriminator_params={
            "periods": [48, 32, 24, 16, 8],
            "use_weight_norm": True,
        },
        multi_wave_discriminator_params={
            "num_dwt": 5,
            "kernel_size": 5,
            "layers": 10,
            "conv_channels": 64,
            "use_weight_norm": True,
        },
    ):
        super().__init__()
        self.mpd = MultiPerioDiscriminator(**multi_period_discriminator_params)
        self.mwd = MultiWaveDiscriminator(**multi_wave_discriminator_params)
        
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).

        Returns:
            Tensor: List of output tensor (B, T')

        """
        return self.mpd(x) + self.mwd(x)


if __name__ == "__main__":
    
    d = Discriminators3(
            multi_wave_discriminator_params={
                "num_dwt": 3,
                "kernel_size": 5,
                "layers": 10,
                "conv_channels": 48,
                "use_weight_norm": True,
            },
            multi_stft_discriminator_params={
                "fft_sizes": [128, 256, 512, 1024],
                "hop_sizes": [32, 64, 128, 256],
                "win_sizes": [128, 256, 512, 1024],
                "num_layers": [5, 6, 7, 8],
                "kernel_sizes": [5, 5, 5, 5],
                "conv_channels": [48, 48, 48, 48],
                "use_weight_norm": True,
            },
        )
    print(d)
    x = torch.randn(1, 1, 8000)
    y = d(x)
    print(x.size(), len(y))
    for i in y:
        print(i.size())
    
    