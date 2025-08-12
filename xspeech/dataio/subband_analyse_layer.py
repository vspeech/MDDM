# -*- coding: utf-8 -*-

import random
import re
import numpy as np


import torch
from xspeech.utils import complex_utils
from torch.utils.cpp_extension import load

from xspeech.dataio.util import *

aec = load(name='aec', verbose=False,
                build_directory='ninja_build',
                sources=['extensions/aec/aec.cpp',
                         'extensions/aec/aec_ops.cu'])

fft_fclp = load(name='fft_fclp', verbose=False,
          build_directory='ninja_build',
          sources=['extensions/fft_fclp/fft_fclp.cpp',
                   'extensions/fft_fclp/fft_fclp_ops.cu'])

class SubbandAnalyseLayer():
    def __init__(self, data_process_cfg, rank):
        self.cfg = data_process_cfg
        self.device = torch.device("cuda:%d" % (rank))
        self.init()

    def init(self):
        self.window = self.cfg['window']
        self.repad = self.cfg['repad']
        self.outdim = self.cfg['outdim']

    def forward(self, noisy_sent, noisy_sent_len, channels):
        subband_filter_lst = self.read_filter('xspeech/dataio/subband_filter.lst')
        subband_filter_128 = subband_filter_lst[1]
        sample_num = len(noisy_sent_len)
        
        padded_inputs = complex_utils.pad2batch(noisy_sent, noisy_sent_len, channels)
        B, C, one_max_len = padded_inputs.shape
        assert B == sample_num

        cut_len = torch.zeros((sample_num), dtype=torch.int32, device=self.device)
        fft_fclp.fft_compute_out_len(cut_len, noisy_sent_len, channels, self.window, self.window)
        nt = torch.sum(cut_len).item()
        nct = nt * channels

        offsets = torch.zeros((nct), dtype=torch.int32, device=self.device)
        fft_fclp.fft_compute_offsets(offsets, noisy_sent_len, cut_len, channels, self.window)

        reordered_in = torch.zeros((nct, self.window), dtype=torch.float32, device=self.device)
        fft_fclp.fft_input_reorder(reordered_in, noisy_sent, offsets, channels, self.window, self.window, self.window)

        filter_len = self.window * 6
        band_num = self.window * 2
        filter = torch.tensor(subband_filter_128, dtype=torch.float32, device=self.device)
        out_len = cut_len * channels
        wide_band_in = torch.zeros((nct, filter_len), dtype=torch.float32, device=self.device)
        aec.wind_to_wide_band(wide_band_in, reordered_in, out_len, channels, 6, sample_num);
        filter = filter.unsqueeze(0)
        wide_band_in = wide_band_in * filter
        fft_in = torch.zeros((nct, band_num), dtype=torch.float32, device=self.device)
        fft_in = wide_band_in[:, 0:band_num]
        for i in range(1,3):
            fft_in += wide_band_in[:,i*band_num:(i+1)*band_num]

        fft_out = torch.fft.rfft(fft_in)

        if self.repad:
            fft_out_real = torch.view_as_real(fft_out)

            fft_width = self.outdim 
            repad_width = channels * fft_width
            fft_out_real = torch.reshape(fft_out_real, [-1, repad_width])

            T = torch.max(cut_len).item()
            repad_out = torch.zeros((T * sample_num, repad_width), dtype=torch.float32, device=self.device)
            repad_out_mask = torch.zeros((T * sample_num), dtype=torch.int32, device=self.device)

            fft_fclp.append_to_output(repad_out, repad_out_mask, fft_out_real, cut_len, sample_num)
            repad_out = torch.reshape(repad_out, [-1, sample_num, channels, int(fft_width / 2), 2]).permute(1, 2, 0, 3, 4)
            repad_out = torch.view_as_complex(repad_out)

            return repad_out, cut_len, padded_inputs

        return fft_out, cut_len, padded_inputs

    def read_filter(self, file):
        fi = open(file, 'r')
        out = []
        sub_lst = []
        for line in fi.readlines():
            line = re.split(r"[ ]+", line.strip())
            #line = line.strip().split(' ')
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
