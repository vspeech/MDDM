# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
from torch.utils.cpp_extension import load
from xspeech.utils import complex_utils
from xspeech.dataio.mask import make_pad_mask
from xspeech.dataio.util import *

fft_fclp = load(name='fft_fclp', verbose=True,
                build_directory='ninja_build',
                sources=['extensions/fft_fclp/fft_fclp.cpp',
                         'extensions/fft_fclp/fft_fclp_ops.cu'])

class STFTLayer():
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
        sample_num = len(noisy_sent_len)
        in_size = self.nfft // 2 + 1
        padded_inputs = complex_utils.pad2batch(noisy_sent, noisy_sent_len, channels)
        B, C, one_max_len = padded_inputs.shape
        
        noise_spec = torch.stft(padded_inputs.view(B*C,one_max_len), self.nfft, self.shift, self.window, window=torch.hamming_window(512, periodic=False, device=self.device), center=False, normalized=False, return_complex=True)
        noise_spec = torch.reshape(noise_spec, [B, C, in_size, -1]).permute(0, 1, 3, 2)
    
        fft_len = torch.zeros((sample_num), dtype=torch.int32, device=self.device)
        fft_fclp.fft_compute_out_len(fft_len, noisy_sent_len, channels, self.window, self.shift)
        input_len = (fft_len - 1) * self.shift + self.window
        with torch.no_grad():
            mask = make_pad_mask(fft_len)
            mask.t_()
            mask = mask.unsqueeze(-1)

        N, C, H, W = noise_spec.size()
        noise_spec_real = torch.view_as_real(noise_spec).permute(2, 0, 1, 3, 4)        
        noise_spec_real = noise_spec_real.reshape(H, N, C*W*2)
        repad_out = noise_spec_real.masked_fill(mask, 0.0)
        repad_out = torch.reshape(repad_out, [H, N, C, W, 2]).permute(1, 2, 0, 3, 4)
        repad_out = torch.view_as_complex(repad_out)
        assert H == torch.max(fft_len).item()
        padded_inputs_mask = self._input_organize(padded_inputs, input_len)

        return repad_out, fft_len, padded_inputs_mask

    def _input_organize(self, input, lengths):
        sample_num = len(lengths)
        out = torch.zeros_like(input, dtype=torch.float32, device=self.device)
        for i in range(sample_num):
            new_len = lengths[i].item()
            out[i, :, :new_len] = input[i, :, :new_len]

        return out
        
