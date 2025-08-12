# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import argparse
import numpy as np
import logging
import torch
import torchaudio
import torch.nn.functional as F

sys.path.append(os.getcwd())
from concurrent import futures

from dora.log import LogProgress
from mydemucs.apply import apply_model
from mydemucs import distrib
from mydemucs.hdemucs import HDemucs
from mydemucs.utils import DummyPoolExecutor

from xspeech.se.nets import se 
from xspeech.dataio.stream_reader import StreamDataReader
from xspeech.dataio.util import *

#torch.backends.cudnn.enabled = False
torch.backends.mkldnn.enabled=False
#torch.set_num_threads(4)

logger = logging.getLogger(__name__)

def debug(out_dir, est, clean, noisy, lengths, sample_num, prefix):
    #out_dir = 'out'
    #noisy_np = noisy.cpu().detach().numpy() 
    #clean_np = clean.cpu().detach().numpy() 
    est_np = est
    noisy_np = noisy 
    clean_np = clean
    pos = 0
    for i in range(sample_num):
        cur_sent_len = lengths[i].item()
        #cur_input = noisy_np[:cur_sent_len]
        #write_pcm(cur_input, cur_sent_len, 1, out_dir, prefix+"_noisy") 
        #cur_input1 = clean_np[:cur_sent_len]
        #write_pcm(cur_input1, cur_sent_len, 1, out_dir, prefix+"_clean") 
        cur_input2 = est_np[:cur_sent_len]
        write_pcm(cur_input2, cur_sent_len, 1, out_dir, prefix+"_est") 
        pos += cur_sent_len
    return

def trans_ddp_ckpt(state_dict):
    out = {}
    for k, v in state_dict.items():
        #new_key = k.replace('module.net.', '')
        new_key = k.replace('net.', '')
        out[new_key] = v
    return out

def renew_model(configs, ckpt):
    kw = {}
    for k, v in configs['model']['conf'].items():
        kw[k] = v
    extra = {'sources': [1], 'audio_channels': 1, 'samplerate': 48000}
    net = HDemucs(**extra, **kw)
    ddp_ckpt = torch.load(ckpt, map_location='cpu')
    #ddp_ckpt = torch.load(ckpt)
    new_state_dict = trans_ddp_ckpt(ddp_ckpt)
    net.load_state_dict(new_state_dict)
    return net

def inference(model, test_lst, out_score, out_dir, num_workers, num_prints, shifts, split, overlap, samplerate):
    out = open(out_score, 'w')
    idx = 0
    with open(test_lst, 'r') as f:
        test_set = f.readlines()

    eval_device = 'cpu'
    indexes = range(distrib.rank, len(test_set), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=num_prints,
                          name='Eval')

    pool = futures.ProcessPoolExecutor if num_workers else DummyPoolExecutor
    with pool(num_workers) as pool:
        for index in indexes:
            track = test_set[index].strip()
            name = track.split('/')[-1]
            dir1 = track.split('/')[-2]
            dir2 = track.split('/')[-3]
            mix, _ = torchaudio.load(track)
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix * (1 << 15) 
            mix = mix.cuda()
            #meant = mix.mean(dim=1, keepdim=True)  # mono mixture
            #stdt = mix.std(dim=1, keepdim=True)
            #mix = (mix - meant) / stdt
            estimates = apply_model(model, mix[None],
                                    shifts=shifts, split=split,
                                    overlap=overlap)
            estimates = estimates.squeeze(1).squeeze(1)
            #estimates = estimates * stdt + meant
            estimates = estimates.to(eval_device)
            out_path = out_dir + "/" + dir2 + "/" + dir1
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            path = out_path + "/" + "vo_" + name
            torchaudio.save(path, estimates / 32768, sample_rate=samplerate, encoding='PCM_S', bits_per_sample=16)
    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='network inference')
    parser.add_argument('--train_conf')
    parser.add_argument('--mdl_ckpt')
    parser.add_argument('--test_lst')
    parser.add_argument('--shifts')
    parser.add_argument('--split')
    parser.add_argument('--overlap')
    parser.add_argument('--sr')
    parser.add_argument('--out_score')
    parser.add_argument('--out_dir')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    with open(args.train_conf, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    #reader = StreamDataReader(args.chunk_frm_num, args.chunk_frm_shift, args.win_size, args.win_step, args.nfft)

    model = renew_model(configs, args.mdl_ckpt)
    model.eval()
    model.cuda()

    inference(model, args.test_lst, args.out_score, args.out_dir, 2, 4, int(args.shifts), True, float(args.overlap), int(args.sr))
