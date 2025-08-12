#
# @file:     se_model.py
# @author:   xunan
# @created:  2024-01-05 14:58
# @modified: 2024-01-05 14:58
# @brief: 
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from xspeech.utils import complex_utils
from xspeech.se.nets import se
from typing import Dict, List, Optional, Tuple
from xspeech.module.masked_modules import ISTFT, SubbandCompose
from xspeech.se.loss import mse_loss, si_snr, wmp_loss, SI_SDR
from xspeech.utils.checkpoint import load_checkpoint, load_trained_modules
from xspeech.utils.file_utils import read_lists 
from mydemucs.hdemucs import HDemucs
from xspeech.dataio.util import *
from xspeech.dataio.mask import make_pad_mask
from mydemucs.stft_loss_torchaudio import MultiResolutionSTFTLoss, DeltaSTFTLoss
#from pystoi.stoi import stoi
#from pesq import pesq

class SEModel(nn.Module):
    idx0 = 0
    idx1 = 0
    EPS = 1e-8
    def __init__(self, net, configs):
        super(SEModel, self).__init__()
        self.net = net
        self.configs = configs
        self.alpha = configs['loss']['alpha']
        self.beta = configs['loss']['beta']
        self.gamma = configs['loss']['gamma']
        if configs['loss']['stft_loss']:
            self.stft_loss = MultiResolutionSTFTLoss(fft_sizes = [4096, 2048, 1024],
                                                     hop_sizes = [720, 480, 240],
                                                     win_lengths = [2400, 1200, 480])
        if configs['loss']['delta_loss']:
            self.delta_loss = DeltaSTFTLoss(configs['loss']['k'], 4096, 1024, 4096)

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        speech = batch['pcm'].to(device)
        speech_truth = batch['pcm_truth'].to(device)
        speech_len = batch['pcm_lengths'].to(device)
        estimate, speech_len = self.net(speech.unsqueeze(1), speech_len)
        estimate = estimate.squeeze(1).squeeze(1)
        total_num = speech_len.sum().item()
        target_mask = make_pad_mask(speech_len)
        target_pad = speech_truth.masked_fill(target_mask, 0.0)
        logit_pad = estimate.masked_fill(target_mask, 0.0)
        assert estimate.shape == speech_truth.shape
        if self.configs['loss']['type'] == 'mse':
            loss = F.mse_loss(logit_pad, target_pad, reduction='sum') / total_num
            #loss = loss.mean()
        elif self.configs['loss']['type'] == 'l1':
            loss = F.l1_loss(logit_pad, target_pad, reduction='sum') / total_num
            #loss = loss.mean()
        elif self.configs['loss']['type'] == 'l12':
            loss1 = F.l1_loss(logit_pad, target_pad, reduction='sum') / total_num
            #loss1 = loss1.mean()
            loss2 = F.mse_loss(logit_pad, target_pad, reduction='sum') / total_num
            #loss2 = loss2.mean()
            if self.configs['loss']['stft_loss'] and self.configs['loss']['delta_loss']:
                loss_sc, loss_mag = self.stft_loss(logit_pad, target_pad, speech_len)
                loss_t, loss_f = self.delta_loss(logit_pad, target_pad, speech_len)
                loss = loss1 + self.gamma * loss2 + self.alpha * (loss_sc + loss_mag) + self.beta * (loss_t + loss_f) 
                return {"loss": loss, "loss1": loss1, "loss2": loss2, "loss_sc": loss_sc, "loss_mag": loss_mag, "loss_t": loss_t, "loss_f": loss_f}
            elif self.configs['loss']['stft_loss']:
                loss_sc, loss_mag = self.stft_loss(logit_pad, target_pad, speech_len)
                loss = loss1 + self.gamma * loss2 + self.alpha * (loss_sc + loss_mag)
                return {"loss": loss, "loss1": loss1, "loss2": loss2, "loss_sc": loss_sc, "loss_mag": loss_mag}
            elif self.configs['loss']['delta_loss']:
                loss_t, loss_f = self.delta_loss(logit_pad, target_pad, speech_len)
                loss = loss1 + self.gamma * loss2 + self.beta * (loss_t + loss_f)
                return {"loss": loss, "loss1": loss1, "loss2": loss2, "loss_t": loss_t, "loss_f": loss_f}
            else:
                loss = loss1 + self.gamma * loss2
        return {"loss": loss, "loss1": loss1, "loss2": self.gamma * loss2}

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def debug(self, noisy, clean, lengths, sample_num, prefix):
        out_dir = 'out'
        noisy_np = noisy.cpu().detach().numpy() 
        clean_np = clean.cpu().detach().numpy() 
        pos = 0
        for i in range(sample_num):
            cur_sent_len = lengths[i].item()
            #cur_input = noisy_np[i, pos:pos+cur_sent_len]
            cur_input = noisy_np[i, :cur_sent_len]
            write_pcm(cur_input, cur_sent_len, 1, self.idx0, out_dir, prefix+"noisy") 
            self.idx0 += 1
            #cur_input1 = clean_np[i, pos:pos+cur_sent_len]
            cur_input1 = clean_np[i, :cur_sent_len]
            write_pcm(cur_input1, cur_sent_len, 1, self.idx1, out_dir, prefix+"clean") 
            self.idx1 += 1
            pos += cur_sent_len
        return

def init_se_model(configs):
    mdl_type = configs['model']['type']
    mdl_nnet = configs['model']['nnet']
    if mdl_type == 'model-se':
        if mdl_nnet == 'resunet':
            #net = se.UNetResComplex_100Mb(channels=configs["model"]["channels"])
            net = se.UNetRes_20Mb(channels=configs["model"]["channels"])
        elif mdl_nnet == 'resunet_gate':
            net = se.UNetRes_F(channels=configs["model"]["channels"])
        elif mdl_nnet == 'h-unet':
            kw = {}
            for k, v in configs['model']['conf'].items():
                kw[k] = v
            extra = {'sources': [1], 'audio_channels': 1, 'samplerate': 48000}
            #net = HUFormer(**extra, **kw)
            #net = HDUFormer(**extra, **kw)
            net = HDemucs(**extra, **kw)
            #net = HTDemucs(**extra, **kw)
        elif mdl_nnet == 'conv-lstm':
            net = se.ConvLstmNnet(configs['model'])
        else:
            assert False, 'illegal nnet type: {}'.format(mdl_nnet)
        
        model = SEModel(net, configs)
    else:
        assert False, 'illegal model type: {}'.format(mdl_type)

    if 'pretrain' in configs:
        infos = load_checkpoint(model, configs['pretrain'])
    else:
        infos = {}
    configs["init_infos"] = infos

    return model, configs
    
