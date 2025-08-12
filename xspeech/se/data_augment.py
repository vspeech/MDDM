# -*- coding: utf-8 -*-
"""
    utf-8
"""
import random
import numpy as np
import torch
from xspeech.dataio.util import *

def compute_noise(signal, noise, snr):
    weight = 10 ** (float(snr) / 10)
    signal_square = signal * signal
    sum_signal = float(signal_square.sum())
    noise_square = noise * noise
    sum_noise = float(noise_square.sum())
    col1 = (sum_noise + 1e-6) * weight
    col0 = (sum_signal / col1) ** 0.5
    return noise * col0

def apply_one_fir(signal, firs, M, seq_len, channels, sample_num):
    C = channels
    N = ceil2i(M) * 2 # fft num
    L = N - (M - 1) # fft shift
    length = 0
    padded_length = 0
    frames = 0
    frame_offsets = []
    padded_seq_lengths = []
    seq_frames = []
    for i in range(sample_num):
        T = seq_len[i]
        nL = int((T + L - 1) / L)
        Tp = nL * L + M - 1
        for j in range(nL):
            frame_offsets.append(padded_length + j * L)
        length += T
        padded_length += Tp
        frames += nL
        padded_seq_lengths.append(Tp)
        seq_frames.append(nL)
    # padding
    x_p = torch.zeros((padded_length, 1), dtype=torch.float32)
    src = []
    dst = []
    block_height = []
    offset_x = 0
    offset_xp = 0
    for i in range(sample_num):
        T = int(seq_len[i])
        Tp = padded_seq_lengths[i]
        src.append(offset_x)
        dst.append(offset_xp + M - 1)
        block_height.append(T)
        offset_x += T
        offset_xp += Tp
    max_block_height = max(block_height)
    src = torch.tensor(src, dtype=torch.int32)
    dst = torch.tensor(dst, dtype=torch.int32)
    block_height = torch.tensor(block_height, dtype=torch.int32)
    T = int(seq_len[0])
    x_p[M-1:T+M-1, :] = signal.reshape(-1, 1)
    ## reorder
    combined_height = frames + channels * sample_num
    f_height = channels * sample_num
    x_f_all = torch.zeros((combined_height, N), dtype=torch.float32)
    for i in range(len(frame_offsets) + 1):
        if i < len(frame_offsets):
            off = frame_offsets[i]
            x_f_all[i, :] = x_p[off:off+N,:].squeeze()
        else:
            x_f_all[i, :] = firs
    ## conv1d = (fft ==> multiply ==> ifft ==> copy back)
    out = torch.zeros((C * length, 1), dtype=torch.float32)
    XF = torch.fft.fft(x_f_all) # [combined_height, N], complex
    X = XF[:frames] # [frames, N], complex
    F = XF[frames:] # [channels * sample_num, N], complex
    offset_x = 0
    Ys = []
    for i in range(sample_num):
        nL = seq_frames[i]
        Xi = X[offset_x:offset_x+nL] # [nL, N], complex
        Fi = F[i*C:i*C+C] # [C, N], complex
        Xi = Xi.unsqueeze(0) # [1, nL, N], complex
        Fi = Fi.unsqueeze(1) # [C, 1, N], complex
        Yi = Xi * Fi # [C, nL, N], complex
        Ys.append(Yi)
        offset_x += nL
    Y = torch.cat(Ys, dim=1) # [C, frames, N], complex
    Y_ifft = torch.fft.ifft(Y) # [C, frames, N], complex
    Y_real = Y_ifft.real # [C, frames, N], float
    Y_slice = Y_real[:, :, M-1:] # [C, frames, L], float
    offset_x = 0
    offset_y = 0
    for i in range(sample_num):
        T = int(seq_len[i])
        nL = seq_frames[i]
        Yi = Y_slice[:, offset_x:offset_x+nL, :] # [C, nL, L], float
        Yi = torch.reshape(Yi, (C, -1)) # [C, nL * L], float
        Yi = Yi[:, :T] # [C, T], float
        Yi = torch.flatten(Yi).unsqueeze(-1) # [C*T, 1], float
        out[offset_y:offset_y+T*C] = Yi
        offset_x += nL
        offset_y += T*C
    return out

'''
def apply_fir(signal, firs, M, seq_len, channels, sample_num, rir_flag, device):
    C = channels
    N = ceil2i(M) * 2 # fft num
    L = N - (M - 1) # fft shift
    length = 0
    padded_length = 0
    frames = 0
    frame_offsets = []
    padded_seq_lengths = []
    seq_frames = []
    for i in range(sample_num):
        T = seq_len[i]
        nL = int((T + L - 1) / L)
        Tp = nL * L + M - 1
        for j in range(nL):
            frame_offsets.append(padded_length + j * L)
        length += T
        padded_length += Tp
        frames += nL
        padded_seq_lengths.append(Tp)
        seq_frames.append(nL)
    # padding
    x_p = torch.zeros((padded_length, 1), dtype=torch.float32, device=device)
    src = []
    dst = []
    block_height = []
    offset_x = 0
    offset_xp = 0
    for i in range(sample_num):
        T = int(seq_len[i])
        Tp = padded_seq_lengths[i]
        src.append(offset_x)
        dst.append(offset_xp + M - 1)
        block_height.append(T)
        offset_x += T
        offset_xp += Tp
    max_block_height = max(block_height)
    src = torch.tensor(src, dtype=torch.int32, device=device)
    dst = torch.tensor(dst, dtype=torch.int32, device=device)
    block_height = torch.tensor(block_height, dtype=torch.int32, device=device)
    DP.copy_from(x_p, signal, dst, src, block_height, sample_num, max_block_height)
    # reorder
    combined_height = frames + channels * sample_num
    f_height = channels * sample_num
    x_f_all = torch.zeros((combined_height, N), dtype=torch.float32, device=device)
    frame_offsets = torch.tensor(frame_offsets, dtype=torch.int32, device=device)
    DP.fft_input_reorder_fir(x_p, frame_offsets, 1, N, L, N, x_f_all, firs, f_height, frames)
    # conv1d = (fft ==> multiply ==> ifft ==> copy back)
    out = torch.zeros((C * length, 1), dtype=torch.float32, device=device)
    XF = torch.fft.fft(x_f_all) # [combined_height, N], complex
    X = XF[:frames] # [frames, N], complex
    F = XF[frames:] # [channels * sample_num, N], complex
    offset_x = 0
    Ys = []
    for i in range(sample_num):
        nL = seq_frames[i]
        Xi = X[offset_x:offset_x+nL] # [nL, N], complex
        Fi = F[i*C:i*C+C] # [C, N], complex
        Xi = Xi.unsqueeze(0) # [1, nL, N], complex
        Fi = Fi.unsqueeze(1) # [C, 1, N], complex
        Yi = Xi * Fi # [C, nL, N], complex
        Ys.append(Yi)
        offset_x += nL
    Y = torch.cat(Ys, dim=1) # [C, frames, N], complex
    Y_ifft = torch.fft.ifft(Y) # [C, frames, N], complex
    Y_real = Y_ifft.real # [C, frames, N], float
    Y_slice = Y_real[:, :, M-1:] # [C, frames, L], float
    offset_x = 0
    offset_y = 0
    for i in range(sample_num):
        T = int(seq_len[i])
        nL = seq_frames[i]
        Yi = Y_slice[:, offset_x:offset_x+nL, :] # [C, nL, L], float
        Yi = torch.reshape(Yi, (C, -1)) # [C, nL * L], float
        Yi = Yi[:, :T] # [C, T], float
        if rir_flag == 0:
            Yi = signal[offset_y:offset_y+T*C]
        Yi = torch.flatten(Yi).unsqueeze(-1) # [C*T, 1], float
        out[offset_y:offset_y+T*C] = Yi
        offset_x += nL
        offset_y += T*C
    return out

def aug(speech, speech_length, rir_lists, noise_lists, configs):
    device = speech.device
    seq_len = speech_length.cpu().numpy()
    rir_nums = len(rir_lists)
    noise_nums = len(noise_lists)
    batch_size = configs['dataset_conf']['batch_conf']['batch_size']
    prob1 = configs['dataset_conf']['rir_conf']['prob']
    prob2 = configs['dataset_conf']['noise_conf']['prob']
    zoom = configs['dataset_conf']['rir_conf']['zoom']
    min_ampl = configs['dataset_conf']['rir_conf']['min_ampl']
    max_ampl = configs['dataset_conf']['rir_conf']['max_ampl']
    min_snr = configs['dataset_conf']['noise_conf']['min_snr']
    max_snr = configs['dataset_conf']['noise_conf']['max_snr']

    rir_len1 = []
    rir_len2 = []
    offset1 = []
    offset2 = []
    firs1 = []
    firs2 = []
    rir_flag = []
    pos1 = 0
    pos2 = 0
    for i in range(batch_size):
        rir_prob = float(torch.rand(1))
        if rir_prob < prob1:
            rir_flag.append(1)
        else:
            rir_flag.append(0)
        rand_idx_rir = random.randint(0, rir_nums-1)
        rir_list = rir_lists[rand_idx_rir].split()
        rir = rir_list[1].strip()
        assert(int(rir_list[-1]) % 4 == 0)
        one_len = int(rir_list[-1]) // 4
        offset1.append(pos1)
        offset2.append(pos2)
        rir_len1.append(one_len)
        rir_len2.append(2400)
        pos1 += one_len
        pos2 += 2400
        rir_seq = np.fromfile(rir, dtype=np.float32, count=one_len)
        rir_seq_truth = np.fromfile(rir, dtype=np.float32, count=2400)
        firs1.extend(rir_seq)
        firs2.extend(rir_seq_truth)

    firs1 = torch.tensor(firs1, device=device)
    firs2 = torch.tensor(firs2, device=device)
    M1 = int(max(rir_len1))
    M2 = int(max(rir_len2))
    N1 = ceil2i(M1) * 2
    N2 = ceil2i(M2) * 2
    signal_firs1 = torch.zeros((batch_size, N1), dtype=torch.float32, device=device)
    signal_firs2 = torch.zeros((batch_size, N2), dtype=torch.float32, device=device)
    for i in range(batch_size):
        signal_firs1[i:(i + 1), :rir_len1[i]] = firs1[offset1[i]:offset1[i]+rir_len1[i]].unsqueeze(0)
        signal_firs2[i:(i + 1), :rir_len2[i]] = firs2[offset2[i]:offset2[i]+rir_len2[i]].unsqueeze(0)

    speech_rir = apply_fir(speech.permute(1,0), signal_firs1, M1, seq_len, 1, batch_size, rir_flag, device)
    speech_rir_truth = apply_fir(speech.permute(1,0), signal_firs2, M2, seq_len, 1, batch_size, rir_flag, device)
    speech_rir = speech_rir.permute(1,0)
    speech_rir_truth = speech_rir_truth.permute(1,0)

    offset = 0
    max_len = int(seq_len.max())
    pad_speech_noisy = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    pad_speech_rir_truth = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    for i in range(batch_size):
        length = seq_len[i]
        ## zoom
        if zoom:
            src_max = float(speech_rir_truth[:, offset:offset+length].max()) + 1e-4
            tar_max = random.uniform(min_ampl, max_ampl)
            zoom_ratio = tar_max / src_max
            speech_rir_truth[:, offset:offset+length] *= zoom_ratio
            speech_rir[:, offset:offset+length] *= zoom_ratio
        pad_speech_rir_truth[i, :length] = speech_rir_truth[:, offset:offset+length]

        noise_prob = float(torch.rand(1))
        if noise_prob < prob2 or rir_flag[i] == 0:
            rand_idx = random.randint(0, noise_nums-1)
            noise_list = noise_lists[rand_idx].split()
            noise_file = noise_list[1].strip()
            noise_offset = int(noise_list[2])
            assert(int(noise_list[3]) % 2 == 0)
            noise_len = int(noise_list[3]) // 2
            noise = np.fromfile(noise_file, dtype=np.int16, count=noise_len, offset=noise_offset)
            noise = torch.tensor(noise, dtype=torch.float32, device=device)
            noise = noise.unsqueeze(0)
            if noise_len > length:
                start = random.randint(0, noise_len - length - 1)
                noise_new = noise[:, start:start+length]
            else:
                extra_len = length - noise_len
                extra_noise = noise[:, :extra_len]
                noise_new = torch.cat((noise, extra_noise), 1) 
            snr = random.randint(min_snr, max_snr)
            noise_new = compute_noise(speech_rir, noise_new, snr)
            pad_speech_noisy[i, :length] = speech_rir[:, offset:offset+length] + noise_new
        else:
            pad_speech_noisy[i, :length] = speech_rir[:, offset:offset+length]
        offset += length

    return pad_speech_noisy, pad_speech_rir_truth
'''
