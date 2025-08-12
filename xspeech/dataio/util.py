# -*- coding: utf-8 -*-

import os
import sys
import random
import struct
import numpy as np
import torch

def write_pcm(cur_sent, cur_sent_len, channels, sample_idx, out_dir, prefix):
    data = np.reshape(cur_sent, (channels, cur_sent_len))
    data = data.astype(np.int16)
    pos = 0
    for i in range(channels):
        datastr = struct.pack('<%dh' % cur_sent_len, *(data[i]))
        with open(os.path.join(out_dir, '%s_sample%d_c%d.pcm' % (prefix, sample_idx, i)), 'wb') as f:
            f.write(datastr)
        pos += cur_sent_len
    return

def write_label(sample_type, raw_label, sample_idx):
    with open(os.path.join('out_pcm', '%d.lab.txt' % sample_idx), 'w') as f:
        lab_str = ' '.join([str(item[0]) for item in raw_label])
        f.write('sample_type: %d\n' % sample_type)
        f.write('raw_label: %s\n' % lab_str)
    return

def dummy_apply_fir(one_sent, fir):
    out = torch.zeros
    one_sent = torch.tensor(one_sent).float().to(device)
    one_sent = one_sent.squeeze().unsqueeze(0)
    multi_channel_sent = torch.tile(one_sent, (2, 1))
    multi_channel_sent = torch.flatten(multi_channel_sent)
    multi_channel_sent = multi_channel_sent.unsqueeze(-1)
    return multi_channel_sent

def ceil2i(x):
    if x <= 0:
        return 0
    out = 1
    while out < x:
        out = out * 2
    return out

def ratio_str_to_prob(ratio_str):
    '''
        '4:2:1' => [4/7, 6/7, 7/7]
    '''
    ratios = [int(item) for item in ratio_str.split(':')]
    ratios_acc = [ratios[0]]
    for i in range(1, len(ratios)):
        ratios_acc.append(ratios[i] + ratios_acc[i-1])
    ratios_prob = []
    for i in range(len(ratios_acc)):
        ratios_prob.append(ratios_acc[i] / sum(ratios))
    return ratios_prob

def rand_num_from_ratio(ratios_prob):
    '''
        ratios_prob: [4/7, 6/7, 7/7]
        rand_prob:
            in [0, 4/7), return 1
            in [4/7, 6/7), return 2
            in [6/7, 7/7], return 3
    '''
    rand_prob = random.random()
    num = 0
    while num < len(ratios_prob):
        if rand_prob < ratios_prob[num]:
            num += 1
            break
        num += 1
    return num

def random_int(ratio_str, sample_num):
    ratios = [int(item) for item in ratio_str.split(':')]
    assert len(ratios) == 2
    headers = torch.zeros(sample_num, dtype=torch.int32, device=ratio_str.device)
    for i in range(sample_num):
        headers[i] = random.randint(ratios[0], ratios[1])
    return headers
