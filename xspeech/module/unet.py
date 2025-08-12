import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import init
from torch.nn import Linear
from xspeech.dataio.mask import make_pad_mask
from xspeech.module.light_gru_cell import LightGruCell
from xspeech.module.masked_modules import MaskedBatchNorm1d, MaskedConv2d, LightGru

class SubLightGru(nn.Module):
    def __init__(self, input_size, freq_size, hidden_size, out_size, norm_type=None):
        super(SubLightGru, self).__init__()
        
        self.gru = LightGru(input_size, hidden_size, out_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.transform = nn.Linear(hidden_size, out_size, bias=False)
        if not norm_type is None:
            if norm_type.upper() == 'BN':
                self.norm = MaskedBatchNorm1d(out_size*freq_size)
 
    def forward(self, inputs, lengths):
        assert inputs.dim() == 4
        B,C,T,D = inputs.size()          
        assert C == self.input_size
        # to [T,B,D,C]
        re_inputs = torch.reshape(inputs.permute(2,0,3,1), [T,B*D,C])
        packed_inputs = self.gru(re_inputs, lengths)
        # to [B,C,T,D]
        packed_inputs = torch.reshape(packed_inputs, [T,B,D,-1]).permute(1,3,0,2).contiguous()
        if hasattr(self, 'norm'):
            # bn need B,C,T,D
            packed_inputs = self.norm(packed_inputs, lengths)
        return packed_inputs

class SubRNN(nn.Module):
    def __init__(self, input_size, freq_size, hidden_size, out_size, rnn_type='lstm', norm_type=None):
        super(SubRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.transform = nn.Linear(hidden_size, out_size, bias=False)
        if not norm_type is None:
            if norm_type.upper() == 'BN':
                self.norm = MaskedBatchNorm1d(out_size*freq_size)
 
    def forward(self, inputs, lengths):
        assert inputs.dim() == 4
        B,C,T,D = inputs.size()          
        assert C == self.input_size
        # to [T,B,D,C]
        re_inputs = torch.reshape(inputs.permute(2,0,3,1), [T,B*D,C])
        re_lengths = lengths.view([B,1]).repeat(1,D).reshape([-1]).cpu()

        packed_inputs = pack_padded_sequence(re_inputs, re_lengths, enforce_sorted=False, batch_first=False)
        packed_inputs = self.rnn(packed_inputs)[0]
        packed_inputs,_ = pad_packed_sequence(packed_inputs, batch_first=False)
        packed_inputs = self.transform(packed_inputs)
        # to [B,C*D,T]
        packed_inputs = torch.reshape(packed_inputs, [T,B,D,-1]).permute(1,3,0,2).contiguous()
        if hasattr(self, 'norm'):
            # bn need B,C,T,D
            packed_inputs = self.norm(packed_inputs, lengths)
        return packed_inputs

class Encoder(nn.Module):
    def __init__(self, in_size, in_channel, out_channel, kernel_size, stride, causal=True, nonlinear='relu6'):
        super(Encoder, self).__init__()
        self.pad = nn.ConstantPad2d(padding=[kernel_size[1]//2,kernel_size[1]//2,kernel_size[0]-1,0],value=0)
        self.conv = MaskedConv2d(in_channel, out_channel, kernel_size=[3,3], stride=stride)
        self.bn = MaskedBatchNorm1d(out_channel*in_size//2)
        self.kernel_size = kernel_size    
        self.nonlinear = nn.ReLU6() if nonlinear=='relu6' else None    
 
    def forward(self, inputs, lengths):
        inputs = self.pad(inputs)
        inputs, lengths = self.conv(inputs, lengths+self.kernel_size[0]-1)
        inputs = self.bn(inputs, lengths)
        if not self.nonlinear is None:
            inputs =  self.nonlinear(inputs)
        return inputs, lengths 

class Decoder(nn.Module):
    def __init__(self, in_size, in_channel, out_channel, kernel_size, stride, causal=True, nonlinear='relu6'):
        super(Decoder, self).__init__()
        self.pad = nn.ConstantPad2d(padding=[kernel_size[1]//2,kernel_size[1]//2,kernel_size[0]-1,0],value=0)
        self.conv = MaskedConv2d(in_channel, out_channel*2, kernel_size=kernel_size, stride=stride)
        self.bn = MaskedBatchNorm1d(out_channel*in_size*2)
        self.nonlinear = nn.ReLU6() if nonlinear=='relu6' else None    
    
    def forward(self, inputs, sc, lengths):
        # inputs: [B,C,T,D]
        inputs = torch.cat([inputs, sc], 1)
        inputs = self.pad(inputs)
        inputs, lengths = self.conv(inputs, lengths+2)
        B,C,T,D = inputs.shape
        left, right = torch.chunk(inputs, 2, 1) 
        inputs = torch.stack([left, right],-1).reshape([B,C//2,T,-1])
        inputs = self.bn(inputs, lengths)
        if not self.nonlinear is None:
            inputs =  self.nonlinear(inputs)
        return inputs, lengths 

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, nonlinear='relu6'):
        super(ConvBlock, self).__init__()
        self.pad = nn.ConstantPad2d(padding=[kernel_size[1]//2,kernel_size[1]//2,kernel_size[0]-1,0],value=0)
        self.conv1 = MaskedConv2d(in_channel, out_channel, kernel_size=[3,3], stride=stride)
        #self.bn = MaskedBatchNorm1d(out_channel*in_size//2)
        self.kernel_size = kernel_size    
        self.conv2 = MaskedConv2d(out_channel, out_channel, kernel_size=[3,3], stride=stride)
        self.nonlinear = nn.ReLU6() if nonlinear=='relu6' else None    
        self.shortcut = MaskedConv2d(in_channel, out_channel, kernel_size=1, bias=True)         

    def forward(self, inputs, lengths):
        sc, lengths = self.shortcut(inputs, lengths)
        inputs = self.pad(inputs)
        inputs, lengths = self.conv1(inputs, lengths+self.kernel_size[0]-1)
        #inputs = self.bn(inputs, lengths)
        if not self.nonlinear is None:
            inputs =  self.nonlinear(inputs)
        inputs = self.pad(inputs)
        inputs, lengths = self.conv2(inputs, lengths+self.kernel_size[0]-1)
        if not self.nonlinear is None:
            inputs = self.nonlinear(inputs)
        out = inputs + sc
        return out, lengths 
