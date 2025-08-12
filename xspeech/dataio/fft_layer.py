# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
from torch.utils.cpp_extension import load

from xspeech.dataio.util import *

fft_fclp = load(name='fft_fclp', verbose=True,
                build_directory='ninja_build',
                sources=['extensions/fft_fclp/fft_fclp.cpp',
                         'extensions/fft_fclp/fft_fclp_ops.cu'])

class FFTLayer():
    def __init__(self, data_process_cfg, rank):
        self.cfg = data_process_cfg
        self.device = torch.device("cuda:%d" % (rank))
        self.init()

    def init(self):
        self.window = self.cfg['window']
        self.shift = self.cfg['shift']
        self.nfft = self.cfg['nfft']
        self.use_hamming = self.cfg['use_hamming']
        self.repad = self.cfg['repad']

    def forward(self, noisy_sent, noisy_sent_len, channels):
        #noisy_sent, noisy_sent_len = x

        sample_num = len(noisy_sent_len)
        fft_len = torch.zeros((sample_num), dtype=torch.int32, device=self.device)
        fft_fclp.fft_compute_out_len(fft_len, noisy_sent_len, channels, self.window, self.shift)
        nt = torch.sum(fft_len).item()
        nct = nt * channels

        offsets = torch.zeros((nct), dtype=torch.int32, device=self.device)
        fft_fclp.fft_compute_offsets(offsets, noisy_sent_len, fft_len, channels, self.shift)

        reordered_in = torch.zeros((nct, self.nfft), dtype=torch.float32, device=self.device)
        fft_fclp.fft_input_reorder(reordered_in, noisy_sent, offsets, channels, self.window, self.shift, self.nfft)

        if self.use_hamming:
            hamming_win = torch.hamming_window(self.nfft, periodic=False, device=self.device)
            hamming_win = hamming_win.unsqueeze(0)
            reordered_in = reordered_in * hamming_win
        
        fft_out = torch.fft.rfft(reordered_in)

        if self.repad:
            fft_out_real = torch.view_as_real(fft_out)

            fft_width = 2 * (int(self.nfft / 2) + 1)
            repad_width = channels * fft_width
            fft_out_real = torch.reshape(fft_out_real, [-1, repad_width])

            T = torch.max(fft_len).item()
            repad_out = torch.zeros((T * sample_num, repad_width), dtype=torch.float32, device=self.device)
            repad_out_mask = torch.zeros((T * sample_num), dtype=torch.int32, device=self.device)

            fft_fclp.append_to_output(repad_out, repad_out_mask, fft_out_real, fft_len, sample_num)
            repad_out = torch.reshape(repad_out, [-1, sample_num, channels, int(fft_width / 2), 2]).permute(1, 2, 0, 3, 4)
            repad_out = torch.view_as_complex(repad_out)

            return repad_out, fft_len

        return fft_out, fft_len
