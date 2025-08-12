"""
Implementation of Multi-Scale Spectral Loss as described in DDSP, 
which is originally suggested in NSF (Wang et al., 2019)
"""

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from xspeech.dataio.mask import make_pad_mask 

class STFTLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, fft_size=1024, shift_size=160, win_length=640, eps=1e-7, low_slice=False):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.low_slice = low_slice
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(eps)
        self.spec = torchaudio.transforms.Spectrogram(n_fft = self.fft_size, 
                                                      hop_length = self.shift_size, 
                                                      win_length = self.win_length)

    def forward(self, x_pred, x_true, lengths, clip_min, clip_type):
        """
        forward
        """
        lengths = lengths // self.shift_size + 1
        total_num = lengths.sum().item()
        mask = make_pad_mask(lengths)
        mask = mask.unsqueeze(1)
        x_mag = self.spec(x_pred)
        y_mag = self.spec(x_true)
        x_mag = x_mag.masked_fill(mask, 0.0)
        y_mag = y_mag.masked_fill(mask, 0.0)


        if clip_type == "clamp":
            x_mag = torch.clamp(x_mag, clip_min)
            y_mag = torch.clamp(y_mag, clip_min)
        else:
            x_mag = torch.where(x_mag < clip_min, torch.FloatTensor([0.000001]).cuda(), x_mag)
            y_mag = torch.where(y_mag < clip_min, torch.FloatTensor([0.000001]).cuda(), y_mag)

        #x_mag = torch.sqrt(torch.clamp(x_mag, 1e-7))
        #y_mag = torch.sqrt(torch.clamp(y_mag, 1e-7))

        if self.low_slice:
            x_mag = x_mag[:, :self.fft_size // 4, :]
            y_mag = y_mag[:, :self.fft_size // 4, :]

        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag) / total_num

        return sc_loss, mag_loss

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""
    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p = "fro") / torch.norm(y_mag, p = "fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""
    def __init__(self, clamp_log_eps=1e-6):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()
        self.clamp_eps = clamp_log_eps

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(self._amp_to_db(y_mag, self.clamp_eps), self._amp_to_db(x_mag, self.clamp_eps), reduction='sum')

    def _amp_to_db(self, x, eps=1e-6):
        """ calculate log-magnitude spectrum"""
        return torch.log(torch.clamp(x, min = eps))


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[4096, 2048, 1024],
                 hop_sizes=[720, 480, 240], # 15ms, 10ms, 5ms
                 win_lengths=[2400, 1200, 480],
                 window="hann_window", clamp_log_eps=1e-6, low_slice=False):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        #win_lengths = [int(x * 4) for x in hop_sizes]
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, clamp_log_eps, low_slice)]

    def forward(self, x, y, lengths, clip_min=1e-7, clip_type="clamp"):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y, lengths, clip_min, clip_type)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

class MagLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, low_slice=False):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MagLoss, self).__init__()
        
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y, clip_min=1e-7):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """

        sc_loss = self.spectral_convergence_loss(x, y)
        mag_loss = self.log_stft_magnitude_loss(x, y)
        
        #fft_bins = x.shape[1]
        #sc_low_loss = self.spectral_convergence_loss(x[:, :fft_bins // 6,:], y[:, :fft_bins // 6, :])
        #mag_low_loss = self.log_stft_magnitude_loss(x[:,:fft_bins//6], y[:, :fft_bins//6, :])

        sc_loss = sc_loss
        msg_loss = mag_loss

        return sc_loss, mag_loss


class Delta():
    def __init__(self, k):
        super(Delta, self).__init__()
        self.k = k

    def forward(self, x, delta_type):
        """ 
        Args:
            x (Tensor): Predicted signal (B, F, T).

        Returns:
            delta k
        """
        B, F, T = x.shape
        if delta_type == "time":
            out_delta = x[:, :, :T-k] - x[:, :, k:T]            
        elif delta_type == "freq":
            out_delta = x[:, :F-k, :] - x[:, k:F, :]
        
        return out_delta

class DeltaSTFTLoss(torch.nn.Module):
    """Delta STFT loss module."""

    def __init__(self, k=1, fft_size=4096, hop_size=1024, win_length=4096, window="hann_window"):
        """Initialize Delta STFT loss module.

        Args:
            k (int): delta order.
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(DeltaSTFTLoss, self).__init__()
        self.delta = Delta(k)
        self.k = k
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.spec = torchaudio.transforms.Spectrogram(n_fft = fft_size, 
                                                      hop_length = hop_size, 
                                                      win_length = win_length)

    def forward(self, x, y, lengths, clip_min=1e-7, clip_type="clamp"):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            delta k loss
        """
        lengths = lengths // self.hop_size + 1
        t_lengths = lengths - self.k
        mask = make_pad_mask(lengths)
        mask = mask.unsqueeze(1)
        t_mask = make_pad_mask(t_lengths)
        t_mask = t_mask.unsqueeze(1)

        x_mag = self.spec(x)
        y_mag = self.spec(y)
        assert x_mag.shape[1] == self.fft_size // 2 + 1
        if clip_type == "clamp":
            x_mag = torch.clamp(x_mag, clip_min)
            y_mag = torch.clamp(y_mag, clip_min)
        else:
            x_mag = torch.where(x_mag < clip_min, torch.FloatTensor([0.000001]).cuda(), x_mag)
            y_mag = torch.where(y_mag < clip_min, torch.FloatTensor([0.000001]).cuda(), y_mag)

        x_time_delta = self.delta(x_mag, delta_type="time")
        y_time_delta = self.delta(y_mag, delta_type="time")
        x_freq_delta = self.delta(x_mag, delta_type="freq")
        y_freq_delta = self.delta(y_mag, delta_type="freq")
    
        x_time_delta = x_time_delta.masked_fill(t_mask, 0.0)
        y_time_delta = y_time_delta.masked_fill(t_mask, 0.0)
        x_freq_delta = x_freq_delta.masked_fill(mask, 0.0)
        y_freq_delta = y_freq_delta.masked_fill(mask, 0.0)

        t_loss1 = F.l1_loss(x_time_delta, y_time_delta, reduction='none')
        t_loss1 = t_loss1.mean()
        t_loss2 = F.mse_loss(x_time_delta, y_time_delta, reduction='none')
        t_loss2 = t_loss2.mean()
        t_loss = t_loss1 + t_loss2

        f_loss1 = F.l1_loss(x_freq_delta, y_freq_delta, reduction='none')
        f_loss1 = f_loss1.mean()
        f_loss2 = F.mse_loss(x_freq_delta, y_freq_delta, reduction='none')
        f_loss2 = f_loss2.mean()
        f_loss = f_loss1 + f_loss2

        return t_loss, f_loss

