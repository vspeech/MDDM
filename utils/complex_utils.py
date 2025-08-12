import torch
import os
import re
import sys
from torch.nn.utils.rnn import pad_sequence
from xspeech.dataio.mask import make_pad_mask

eps = 1e-6

def padding_est(logit, lengths):
    total_frame = lengths.sum().item()
    target_mask = make_pad_mask(lengths)
    target_mask.t_()
    target_mask = target_mask.unsqueeze(-1)
    N, C, H, W = logit.size()
    logit = logit.permute(2, 0, 1, 3)
    logit = logit.reshape(H, N, C*W)
    logit_pad = logit.masked_fill(target_mask, 0.0)
    logit_pad = torch.reshape(logit_pad, [H, N, C, W]).permute(1, 2, 0, 3)
    #logit_pad = logit_pad.reshape(B*C, W, T)
    return logit_pad 

def cplx_mag_crm(in1, in2, min_snr, max_snr, min_u, max_u):
    a = in1.abs()
    b = in2.abs()
    a = a ** 2
    b = b ** 2
    s = (max_snr - min_snr) / (max_u - min_u)
    pt = torch.div(a, b + 1e-6).clamp(1e-6)
    snr = 10 * pt.log10()
    tmp = (abs(min_snr) * min_u + abs(max_snr) * max_u) / (max_snr - min_snr)
    u0 = torch.full(snr.size(), tmp).cuda()
    u = u0 - snr / s
    mask = snr > max_snr
    u.masked_fill_(mask, min_u)
    mask = snr < min_snr
    u.masked_fill_(mask, max_u)
    u = u.clamp(0.0) 
    return torch.div(pt, pt + u).clamp(0.0, 1.0)

def cplx_mag_irm(in1, in2):
    a = in1.abs()
    b = in2.abs()
    return torch.div(a, b + 1e-6).clamp(0.0, 1.0)

def cplx_cirm(in1, in2):
    mask = in1 / (in2 + eps)
    mag = mask.abs()
    mag_gain = torch.clamp(mag, 0.0, 1.0) / (mag + eps)
    mask = torch.view_as_real(mask) * mag_gain.unsqueeze(-1)
    mask = torch.view_as_complex(mask)
    return mask

def cplx_mul_irm(in1, in2, lengths):
    assert(in1.dim() == in2.dim())
    if torch.cuda.is_available():
        rank = 0
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    
    padding_out = torch.mul(in1, in2)
    #padding_out = in2
    
    N, C, H, W = padding_out.size()
    repad_width = C * W * 2
    sample_num = len(lengths)
    padding_out = padding_out.permute(2, 0, 1, 3)
    padding_out_real = torch.view_as_real(padding_out)
    padding_out_real = torch.reshape(padding_out_real, [-1, repad_width])
    nt = torch.sum(lengths).item()
    output = torch.zeros((nt, repad_width), dtype=torch.float32, device=device)
    aec.reduce_to_dense(output, padding_out_real, lengths, sample_num)
    output = torch.reshape(output, [-1, W, 2])
    return output

def pad_to_dense(input, lengths):
    if torch.cuda.is_available():
        rank = 0
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    
    N, C, H, W = input.size()
    repad_width = C * W * 2
    #repad_width = C * W
    sample_num = len(lengths)

    input = input.permute(2, 0, 1, 3)
    padding_out_real = torch.view_as_real(input)
    #padding_out_real = input
    padding_out_real = torch.reshape(padding_out_real, [-1, repad_width])
    nt = torch.sum(lengths).item()
    output = torch.zeros((nt, repad_width), dtype=torch.float32, device=device)
    aec.reduce_to_dense(output, padding_out_real, lengths, sample_num)
    output = torch.reshape(output, [-1, W, 2])
    return output

def pad_to_dense_cpu(input, lengths):
    if torch.cuda.is_available():
        rank = 0
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    
    N, C, H, W = input.size()
    repad_width = C * W * 2
    #repad_width = C * W
    sample_num = len(lengths)
    T = torch.max(lengths).item()

    padding_out_real = torch.view_as_real(input)
    #padding_out_real = input
    padding_out_real = torch.reshape(padding_out_real, [-1, repad_width])
    nt = torch.sum(lengths).item()
    output = torch.zeros((nt, repad_width), dtype=torch.float32, device=device)
    offset1 = 0
    offset2 = 0
    for i in range(sample_num):
        each_len = lengths[i].item()
        output[offset1 : offset1 + each_len, :] = padding_out_real[offset2 : offset2 + each_len, :]
        offset1 += each_len
        offset2 += T
    #print("mean = ", (padding_out_real-output).mean())
    output = torch.reshape(output, [-1, W, 2])
    return output

def dense_to_pad_cpu(input, lengths, channels):
    assert input.dim() == 2
    T = torch.max(lengths).item()  #one channel lenght
    
    if torch.cuda.is_available():
        rank = 0
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    
    sample_num = len(lengths)
    output = torch.zeros((sample_num * T * channels, input.size(1)), dtype=torch.float32, device=device)
    offset1 = 0
    offset2 = 0
    for i in range(sample_num):
        for c in range(channels):
            each_len = lengths[i]
            output[offset2 : offset2 + each_len, :] = input[offset1 : offset1 + each_len, :]
            offset1 += each_len
            offset2 += T

    return output   # N C T F

def read_filter(file):
    fi = open(file, 'r')
    out = []
    sub_lst = []
    for line in fi.readlines():
        line = re.split(r"[ ]+", line.strip()) 
        if len(line) == 1:
            out.append(sub_lst)
            sub_lst = []
        elif len(line) == 4:
            sub_lst.extend([float(line[i].split(',')[0]) for i in range(len(line))])
        elif len(line) == 3:
            sub_lst.extend([float(line[i].split(',')[0]) for i in range(len(line))])
        else:
            continue

    return out

def subbandcompose(input, lengths, channels):
    if torch.cuda.is_available():
        rank = 0
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    
    assert input.dim() == 3
    in_height, in_width, num = input.size()
    window = in_width - 1
    filter_len = window * 6
    band_num = window * 2

    subband_filter_lst = read_filter('xspeech/dataio/subband_filter.lst')
    subband_filter_128 = subband_filter_lst[1]
    
    sample_num = len(lengths)
    mask = lengths * channels
    out_len = mask * window
    out_height = in_height * window
    output = torch.zeros((out_height, 1), dtype=torch.float32, device=device)
    input_complex = torch.view_as_complex(input)
    
    ifft_out = torch.fft.irfft(input_complex, norm="forward")
    
    wide_band_in = torch.zeros((in_height, filter_len), dtype=torch.float32, device=device)

    for i in range(0,3):
        wide_band_in[:,i*band_num:(i+1)*band_num] = ifft_out[:,:]

    filter = torch.tensor(subband_filter_128, dtype=torch.float32, device=device)
    filter = filter.unsqueeze(0) 
    wide_band_in = wide_band_in * filter
    wide_band_in = wide_band_in * window

    #print("band = ", wide_band_in[:100, :])
    aec.wind_subband_composition(output, wide_band_in, mask, channels, window, 6, sample_num)
    
    return output / 32768.0, out_len

def subbandcompose_cpu(input, lengths, channels):
    #if torch.cuda.is_available():
    #    rank = 0
    #    device = torch.device('cuda:{:d}'.format(rank))
    #else:
    device = torch.device('cpu')
    
    assert input.dim() == 3
    assert channels == 1
    in_height, in_width, num = input.size()
    window = in_width - 1
    filter_len = window * 6
    band_num = window * 2

    subband_filter_lst = read_filter('xspeech/dataio/subband_filter.lst')
    subband_filter_128 = subband_filter_lst[1]
   
    lengths = lengths.to(device)
    sample_num = len(lengths)
    mask = lengths * channels
    out_len = mask * window
    out_height = in_height * window
    output = torch.zeros((out_height, 1), dtype=torch.float32, device=device)
    #output = torch.zeros((in_height, window), dtype=torch.float32, device=device)
    input_complex = torch.view_as_complex(input)
    
    ifft_out = torch.fft.irfft(input_complex, norm="forward")
    
    wide_band_in = torch.zeros((in_height, filter_len), dtype=torch.float32, device=device)

    for i in range(0,3):
        wide_band_in[:,i*band_num:(i+1)*band_num] = ifft_out[:,:]

    filter = torch.tensor(subband_filter_128, dtype=torch.float32, device=device)
    filter = filter.unsqueeze(0) 
    wide_band_in = wide_band_in * filter
    wide_band_in = wide_band_in * window
    wide_band_in = torch.reshape(wide_band_in, [-1, 1])

    wide_band_in = torch.reshape(wide_band_in, [in_height, 6, window]).permute(1, 0, 2)
    input = torch.zeros((6, in_height-5, window), dtype=torch.float32, device=device)
    for i in range(6):
        input[i, :, :] = wide_band_in[i, 6-i-1:in_height-i, :]
    output = torch.sum(input, 0)
    for i in range(in_height-5, in_height):
        for j in range(window):
            val = 0
            for k in range(6):
                if i + k >= in_height:
                    break
                #tmp = wide_band_in[(i - k) * filter_len + k * window + j, :]
                tmp = wide_band_in[(i + k) * filter_len + (6 - k - 1) * window + j, :]
                val += tmp.item()
            output[i, j] = val
    output = torch.reshape(output, [-1, 1]) 
    
    aec.subband_composition_st(output, wide_band_in, mask, channels, window, 6, sample_num)
    #aec.subband_composition(output, wide_band_in, mask, in_offset, out_offset, channels, window, 6, sample_num)
    
    return output / 32768.0, out_len

def pad2batch(inputs, length, nchannels=2):
    # input: [T, 1] or [T]
    assert ( inputs.dim() < 3 )
    if inputs.dim() == 2:
        inputs = torch.squeeze(inputs)
    tmp_output = []
    pos = 0
    for idx in range(len(length)):
        seq_len = length[idx]
        tmp_output.append(
              torch.transpose(inputs[pos:pos+seq_len].reshape([nchannels,-1]), 0, 1)
        )
        pos += seq_len
    outputs = pad_sequence(tmp_output, batch_first=True)
    outputs = outputs.permute(0,2,1).contiguous()
    return outputs

'''
if __name__ == '__main__':
    a = torch.randint(0, 100, (3, 2))
    b = torch.randint(0, 100, (3, 2))
    print(a)
    print(b)
    d = cplx_mag_crm(a, b, -15, 10, 1, 10)
    print(d)
'''
