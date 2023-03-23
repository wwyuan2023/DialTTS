# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_size, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, fft_size//2+1, #frames).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(x_mag), torch.log(y_mag))


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size, shift_size, win_size, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_size = win_size
        self.register_buffer('window', getattr(torch, window)(win_size), persistent=False)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Magnitude of predicted signal (B, F, T).
            y (Tensor): Magnitude of groundtruth signal (B, F, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_size, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_size, self.window)
        sc_loss = self.spectral_convergence_loss(x, y)
        mag_loss = self.log_stft_magnitude_loss(x, y)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_sizes=[600, 1200, 240],
        window="hann_window"
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_sizes (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal, (B, T).
            y (Tensor): Groundtruth signal, (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
