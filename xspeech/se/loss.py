import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from xspeech.dataio.mask import make_pad_mask 

EPS = 1e-8

def wmp_loss(logit, target, lengths, weight = 1.0, beta = 1.0):
    #mel_w = mel_scale(257, 16000)
    #mel_w = mel_w.to(logit.device)
    if target.dtype == torch.complex64:
        N, C, H, W = target.size() 
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        target_pad = torch.view_as_real(target)
        target_pad = torch.reshape(target_pad, [N, C, H, W*2]).permute(2, 0, 1, 3)   
        target_pad = torch.reshape(target_pad, [H, N, C*W*2])
        target_pad = target_pad.masked_fill(target_mask, 0.0)
        target_pad = torch.reshape(target_pad.permute(1, 0, 2), [N, H, C*W, 2])
        logit = torch.reshape(logit.permute(2, 0, 3, 1), [H, N, -1])
        logit_pad = logit.masked_fill(target_mask, 0.0)
        logit_pad = torch.reshape(logit_pad.permute(1, 0, 2), [N, H, -1, 2])
        ## mag ##
        target_mag = torch.sqrt(target_pad[...,0]**2 + target_pad[...,1]**2 + EPS)
        logit_mag = torch.sqrt(logit_pad[...,0]**2 + logit_pad[...,1]**2 + EPS)
        dif_mag = logit_mag - target_mag
        ## phase ##
        target_theta = torch.atan2(target_pad[...,1], target_pad[...,0]+EPS)
        logit_theta = torch.atan2(logit_pad[...,1], logit_pad[...,0]+EPS)
        dif_theta = 2 * target_mag * torch.sin((logit_theta - target_theta)/2)

        #loss = torch.sum(dif_mag**2 + weight * dif_theta**2) 
        loss = torch.sum(weight * dif_theta**2) 
        #loss = torch.sum(weight * mel_w * dif_theta**2) 
        return beta * loss / total_frame / 2
    else:
        assert logit.dim() == target.dim()
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        dim = target.dim()
        if dim == 4:
            N, C, H, W = target.size()
            target = target.permute(2, 0, 1, 3)
            target = target.reshape(H, N, C*W)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.permute(2, 0, 1, 3)
            logit = logit.reshape(H, N, C*W)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)
        elif dim == 2:
            T, _ = target.size()
            assert T % len(lengths) == 0
            H = T // len(lengths)
            N = len(lengths)
            target = target.reshape(H, N, -1)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.reshape(H, N, -1)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)  
        #return torch.sum(mel_w * (logit_pad - target_pad)**2) / total_frame
        return F.mse_loss(logit_pad, target_pad, reduction='sum') / total_frame

def mse_loss(logit, target, lengths, alpha = 1.0):
    if target.dtype == torch.complex64:
        N, C, H, W = target.size() 
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        target_pad = torch.view_as_real(target)
        target_pad = torch.reshape(target_pad, [N, C, H, W*2]).permute(2, 0, 1, 3)   
        target_pad = torch.reshape(target_pad, [H, N, C*W*2])
        target_pad = target_pad.masked_fill(target_mask, 0.0)
        target_pad = torch.reshape(target_pad.permute(1, 0, 2), [N, H, C*W, 2])
        logit = torch.reshape(logit.permute(2, 0, 3, 1), [H, N, -1])
        logit_pad = logit.masked_fill(target_mask, 0.0)
        logit_pad = torch.reshape(logit_pad.permute(1, 0, 2), [N, H, -1, 2])
        r_loss = F.mse_loss(target_pad[...,0], logit_pad[...,0], reduction='sum')
        i_loss = F.mse_loss(target_pad[...,1], logit_pad[...,1], reduction='sum')
        return alpha * (r_loss+i_loss) / total_frame / 2
    else:
        assert logit.dim() == target.dim()
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        dim = target.dim()
        
        if dim == 4:
            N, C, H, W = target.size()
            target = target.permute(2, 0, 1, 3)
            target = target.reshape(H, N, C*W)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.permute(2, 0, 1, 3)
            logit = logit.reshape(H, N, C*W)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)
        elif dim == 2:
            T, _ = target.size()
            assert T % len(lengths) == 0
            H = T // len(lengths)
            N = len(lengths)
            target = target.reshape(H, N, -1)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.reshape(H, N, -1)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)  
        return F.mse_loss(logit_pad, target_pad, reduction='sum') / total_frame

def mean_norm(s1, s2, lengths):
    total_num = lengths.unsqueeze(-1).unsqueeze(-1) * 257
    total = torch.sum(s1 * s2, dim=1, keepdim=True).sum(dim=-1, keepdim=True)
    return total / total_num

def simse_loss(logit, target, complex_target, lengths, alpha = 1.0):
    if target.dtype == torch.complex64:
        N, C, H, W = target.size() 
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        target_pad = torch.view_as_real(target)
        target_pad = torch.reshape(target_pad, [N, C, H, W*2]).permute(2, 0, 1, 3)   
        target_pad = torch.reshape(target_pad, [H, N, C*W*2])
        target_pad = target_pad.masked_fill(target_mask, 0.0)
        target_pad = torch.reshape(target_pad.permute(1, 0, 2), [N, H, C*W, 2])
        logit = torch.reshape(logit.permute(2, 0, 3, 1), [H, N, -1])
        logit_pad = logit.masked_fill(target_mask, 0.0)
        logit_pad = torch.reshape(logit_pad.permute(1, 0, 2), [N, H, -1, 2])
        r_loss = F.mse_loss(target_pad[...,0], logit_pad[...,0], reduction='sum')
        i_loss = F.mse_loss(target_pad[...,1], logit_pad[...,1], reduction='sum')
        return alpha * (r_loss+i_loss) / total_frame / 2
    else:
        assert logit.dim() == target.dim()
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        dim = target.dim()
        if dim == 4:
            N, C, H, W = target.size()
            target = target.permute(2, 0, 1, 3)
            target = target.reshape(H, N, C*W)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.permute(2, 0, 1, 3)
            logit = logit.reshape(H, N, C*W)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)

            ctarget_pad = torch.view_as_real(complex_target)
            ctarget_pad = torch.reshape(ctarget_pad, [N, C, H, W*2]).permute(2, 0, 1, 3)   
            ctarget_pad = torch.reshape(ctarget_pad, [H, N, C*W*2])
            ctarget_pad = ctarget_pad.masked_fill(target_mask, 0.0)
            ctarget_pad = torch.reshape(ctarget_pad.permute(1, 0, 2), [N, H, C*W, 2])
            ctarget_pad = torch.view_as_complex(ctarget_pad)
            s_h = mean_norm(target_pad, logit_pad, lengths) * ctarget_pad / (mean_norm(target_pad, target_pad, lengths) + EPS)
            noise = s_h.abs() - logit_pad
            loss = mean_norm(noise, noise, lengths) / (mean_norm(s_h.abs(), s_h.abs(), lengths) + EPS)
        return torch.mean(loss)

def weight_mp_loss(logit, target, ww, lengths, weight = 1.0, beta = 1.0):
    if target.dtype == torch.complex64:
        N, C, H, W = target.size() 
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        target_pad = torch.view_as_real(target)
        target_pad = torch.reshape(target_pad, [N, C, H, W*2]).permute(2, 0, 1, 3)   
        target_pad = torch.reshape(target_pad, [H, N, C*W*2])
        target_pad = target_pad.masked_fill(target_mask, 0.0)
        target_pad = torch.reshape(target_pad.permute(1, 0, 2), [N, H, C*W, 2])
        logit = torch.reshape(logit.permute(2, 0, 3, 1), [H, N, -1])
        logit_pad = logit.masked_fill(target_mask, 0.0)
        logit_pad = torch.reshape(logit_pad.permute(1, 0, 2), [N, H, -1, 2])
        ## mag ##
        target_mag = torch.sqrt(target_pad[...,0]**2 + target_pad[...,1]**2 + EPS)
        logit_mag = torch.sqrt(logit_pad[...,0]**2 + logit_pad[...,1]**2 + EPS)
        dif_mag = logit_mag - target_mag
        ## phase ##
        target_theta = torch.atan2(target_pad[...,1], target_pad[...,0]+EPS)
        logit_theta = torch.atan2(logit_pad[...,1], logit_pad[...,0]+EPS)
        dif_theta = 2 * target_mag * torch.sin((logit_theta - target_theta)/2)

        loss = torch.sum(weight * ww * dif_theta**2) 
        #loss = torch.sum(weight * mel_w * dif_theta**2) 
        return beta * loss / total_frame / 2
    else:
        assert logit.dim() == target.dim()
        total_frame = lengths.sum().item()
        target_mask = make_pad_mask(lengths)
        target_mask.t_()
        target_mask = target_mask.unsqueeze(-1)
        dim = target.dim()
        if dim == 4:
            N, C, H, W = target.size()
            target = target.permute(2, 0, 1, 3)
            target = target.reshape(H, N, C*W)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.permute(2, 0, 1, 3)
            logit = logit.reshape(H, N, C*W)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)
        elif dim == 2:
            T, _ = target.size()
            assert T % len(lengths) == 0
            H = T // len(lengths)
            N = len(lengths)
            target = target.reshape(H, N, -1)
            target_pad = target.masked_fill(target_mask, 0.0)
            target_pad = target_pad.permute(1, 0, 2)
            logit = logit.reshape(H, N, -1)
            logit_pad = logit.masked_fill(target_mask, 0.0)
            logit_pad = logit_pad.permute(1, 0, 2)  
        return torch.sum(ww * (logit_pad - target_pad)**2) / total_frame
        #return F.mse_loss(logit_pad, target_pad, reduction='sum') / total_frame

def mel_scale(num_fft_bins, sample_freq):
    window_length_padded = (num_fft_bins - 1) * 2
    fft_bin_width = sample_freq / window_length_padded
    sum = 0
    out = torch.zeros(num_fft_bins, dtype=torch.float32)
    for n in range(1, num_fft_bins):
        #freq = abs(20 - n - 0.5) * fft_bin_width
        freq = (n + 0.5) * fft_bin_width
        scale = 1127 / (700 + freq)
        #scale = 1127 / (500 + freq)
        out[n] = scale
        sum += scale

    out = out / sum * num_fft_bins
    return out

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)

def si_snr(estimated, original):
    # [B, C, T]
    estimated_len = estimated.shape[-1]
    original_len = original.shape[-1]
    if estimated_len < original_len:
        original = original[:, :, :estimated_len] 
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - target
    snr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return snr.squeeze_(dim=-1)

def SI_SDR(reference, estimation):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)。

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References:
        SDR- Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference**2, axis=-1, keepdims=True)

    optimal_scaling = (
        np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    )

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection**2, axis=-1) / np.sum(noise**2, axis=-1)
    return 10 * np.log10(ratio)

'''
if __name__ == '__main__':
    lengths = torch.tensor([1,2], dtype=torch.int32)
    a = torch.tensor([[[[2],[1]],[[2],[3]]],[[[1],[4]],[[2],[2]]]], dtype=torch.float32)
    b = torch.tensor([[[[2+1j],[1+1j]],[[2+1j],[3+1j]]],[[[1+1j],[4+1j]],[[2+1j],[2+1j]]]])
    loss = mse_loss(a, b, lengths)
    print(loss)
'''
