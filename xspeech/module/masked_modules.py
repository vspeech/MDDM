#
# @file:     padded_modules.py
# @author:   xunan
# @created:  2024-04-12 21:16
# @modified: 2024-04-12 21:16
# @brief: 
#

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from xspeech.dataio.mask import make_pad_mask

class MaskedGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(MaskedGroupNorm, self).__init__(
            num_groups,
            num_channels,
            eps,
            affine
        )
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert num_channels % num_groups == 0
        self.groups = num_channels // num_groups

    def forward(self, input, lengths):
        dim = input.dim()
        if dim == 3:
            N, C, T = input.size()
            input = input.unsqueeze(2)
        N, C, W, T = input.size()
        with torch.no_grad():
            mask = make_pad_mask(lengths)
            mask = mask.unsqueeze(1).unsqueeze(1)
            lengths = lengths.reshape(N, -1, 1, 1, 1)
            counts = lengths * W * self.groups
        
        x = input.masked_fill(mask, 0.0)
        x = x.view(-1, self.num_groups, self.groups, x.size(2), x.size(3))
        gn_mean = x.sum(dim=(2,3,4), keepdim=True) / counts
        gn_var = (x ** 2).sum(dim=(2,3,4), keepdim=True) / counts - gn_mean ** 2
        x_norm = (x - gn_mean) / torch.sqrt(gn_var + self.eps)
        x_norm = x_norm.view(-1, self.num_channels, x.size(3), x.size(4))
        #group_inputs = torch.split(x, split_size_or_sections=self.groups, dim=1)
        #results = []
        #for g_input in group_inputs:  
        #    gn_mean = g_input.sum(dim=(1, 2, 3), keepdim=True) / counts 
        #    gn_var = (g_input ** 2).sum(dim=(1, 2, 3), keepdim=True) / counts - gn_mean ** 2
        #    gn_result = (g_input - gn_mean) / torch.sqrt(gn_var + self.eps)
        #    results.append(gn_result)
        #x = torch.cat(results, dim=1)  # 按照通道维进行合并
        x_norm = x_norm.reshape(N, C, -1).permute(0, 2, 1)
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        
        x_norm = x_norm.reshape(N, W, -1, C).permute(0, 3, 1, 2)
        x_norm = x_norm.masked_fill(mask, 0.0)
        if dim == 3:
            assert W == 1
            x_norm = x_norm.reshape(N, C, T)
        return x_norm

class MaskedBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, is_update_running_stats=True, eps=1e-6, momentum=1e-3,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )
        self.register_buffer('counter', torch.tensor(0.0, dtype=torch.float32))
        self.is_update_running_stats = is_update_running_stats

    def forward(self, input, lengths):
        with torch.no_grad():
            mask = make_pad_mask(lengths)
            mask.t_()
            mask = mask.unsqueeze(-1)
            counts = lengths.sum()
        
        dim = input.dim()
        if dim == 4:
            N, C, H, W = input.size()
            input = input.permute(2, 0, 1, 3)
            input = input.reshape(H, N, C*W)


        x = input.masked_fill(mask, 0.0)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        # calculate running estimates
        if self.training and self.is_update_running_stats:
            mean = x.sum([0, 1]) / counts
            var = (x ** 2).sum([0, 1]) / counts - mean ** 2
            with torch.no_grad():
                '''
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * counts / (counts - 1)\
                    + (1 - exponential_average_factor) * self.running_var
                '''
                self.running_mean = mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = var * counts / (counts - 1) + (1 - exponential_average_factor) * self.running_var
                self.counter = 1.0 + (1 - exponential_average_factor) * self.counter
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            x = x * self.weight + self.bias

        if self.training:
            x.masked_fill_(mask, 0.0)

        if dim == 4:
            x = x.reshape(H, N, C, W).permute(1, 2, 0, 3)

        return x

class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, is_update_running_stats=True, eps=1e-6, momentum=1e-3,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm2d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )
        self.register_buffer('counter', torch.tensor(0.0, dtype=torch.float32))
        self.is_update_running_stats = is_update_running_stats

    def forward(self, input, lengths):
        B, C, W, T = input.shape
        with torch.no_grad():
            mask = make_pad_mask(lengths)
            mask = mask.unsqueeze(1).unsqueeze(1)
            counts = lengths.sum() * W
        
        dim = input.dim()
        assert dim == 4
        x = input.masked_fill(mask, 0.0)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        # calculate running estimates
        if self.training and self.is_update_running_stats:
            mean = x.sum([0, 2, 3]) / counts
            var = (x ** 2).sum([0, 2, 3]) / counts - mean ** 2
            with torch.no_grad():
                #self.running_mean = exponential_average_factor * mean\
                #    + (1 - exponential_average_factor) * self.running_mean
                ## Update running_var with unbiased var
                #self.running_var = exponential_average_factor * var * counts / (counts - 1)\
                #    + (1 - exponential_average_factor) * self.running_var
                self.running_mean = mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = var * counts / (counts - 1) + (1 - exponential_average_factor) * self.running_var
                self.counter = 1.0 + (1 - exponential_average_factor) * self.counter
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
            x = x * self.weight + self.bias
            x = x.view(B, W, -1, C).permute(0, 3, 1, 2)

        if self.training:
            x = x.masked_fill(mask, 0.0)

        return x

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', is_1d=False):
        super(MaskedConv2d, self).__init__()
        self.is_1d = is_1d
        if is_1d:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias, padding_mode
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias, padding_mode
            )
        
    def forward(self, input, lengths):
        output = self.conv(input)
        
        with torch.no_grad():
            dim = 0
            if not self.is_1d:
                dim = 1
            kernel = self.conv.kernel_size[dim]
            padding = self.conv.padding[dim]
            stride = self.conv.stride[dim]
            dilation = self.conv.dilation[dim]
            lengths = (lengths - dilation * (kernel - 1) - 1 + 2 * padding) // stride + 1
            input_mask = make_pad_mask(lengths)
            if self.is_1d:
                input_mask = input_mask.unsqueeze(1)
            else:
                input_mask = input_mask.unsqueeze(1).unsqueeze(1)

        output = output.masked_fill(input_mask, 0)

        return output, lengths

class MaskedConvTr2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', is_1d_tr=False):
        super(MaskedConvTr2d, self).__init__()
        self.is_1d_tr = is_1d_tr
        if is_1d_tr:
            self.conv = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride)
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride)
        
    def forward(self, input, lengths):
        output = self.conv(input)
        
        with torch.no_grad():
            dim = 0
            if not self.is_1d_tr:
                dim = 1
            kernel = self.conv.kernel_size[dim]
            padding = self.conv.padding[dim]
            stride = self.conv.stride[dim]
            lengths = (lengths - 1) * stride - 2 * padding + kernel
            input_mask = make_pad_mask(lengths)
            if self.is_1d_tr:
                input_mask = input_mask.unsqueeze(1)
            else:
                input_mask = input_mask.unsqueeze(1).unsqueeze(1)

        output.masked_fill_(input_mask, 0)

        return output, lengths

class MaskedGateConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', houyi_init=False, act=nn.ReLU6()):
        super(MaskedGateConv2d, self).__init__()
        self.activation = act
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        
        self.sigmoid = nn.Sigmoid()
        if houyi_init:
            self.reset_parameters()

    def reset_parameters(self):
        mean = 0.0
        stddev = 0.02
        for weight in self.parameters():
            weight.data.normal_(mean, stddev)
    
    def gated(self, mask):
        return self.sigmoid(mask)
 
    #def forward(self, input, lengths):
    def forward(self, input):
        x = self.conv(input)
        mask = self.mask_conv(input)
        #assert lengths1.sum().item() == mask_lengths.sum().item()
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        '''
        with torch.no_grad():
            kernel = self.conv.kernel_size[0]
            padding = self.conv.padding[0]
            stride = self.conv.stride[0]
            lengths = (lengths - kernel + 2 * padding) // stride + 1
            input_mask = make_pad_mask(lengths)
            input_mask = input_mask.unsqueeze(1).unsqueeze(-1)

        x.masked_fill_(input_mask, 0)
        '''
        return x
        #return x, lengths1

class MaskedAudioConv(nn.Module):
    def __init__(self, group_num, fbank_dim, start_bin, end_bin, filter_num, filter_size, stride, padding, bias=True, houyi_init=False):
        super(MaskedAudioConv, self).__init__()
        self.conv_start_bin = [int(item) for item in start_bin.split(':')]
        self.conv_end_bin = [int(item) for item in end_bin.split(':')]
        assert len(self.conv_start_bin) == group_num
        assert len(self.conv_end_bin) == group_num
        assert len(filter_size) == 2 and len(stride) == 2 and len(padding) == 2
        self.group_num = group_num
        self.fbank_dim = fbank_dim
        self.convs = nn.ModuleList()
        for i in range(group_num):
            one_conv = nn.Conv2d(1, filter_num, filter_size, stride, padding, bias=bias)
            self.convs.append(one_conv)

    def forward(self, input, lengths):
        '''
            input: (B, T, 80)
        '''
        assert input.dim() == 3
        assert input.size(2) == self.fbank_dim
        out = []
        for i in range(self.group_num):
            cur_conv = self.convs[i]
            cur_start = self.conv_start_bin[i]
            cur_end = self.conv_end_bin[i] + self.convs[i].kernel_size[1]
            cur_input = input[:, None, :, cur_start:cur_end]
            cur_out = cur_conv(cur_input)
            out.append(cur_out)
        out = torch.cat(out, dim=1) # [B, C, T, W]
        out, _ = torch.max(out, dim=-1) # [B, C, T]
        out = out.transpose(1, 2)

        with torch.no_grad():
            kernel = self.convs[0].kernel_size[0]
            padding = self.convs[0].padding[0]
            stride = self.convs[0].stride[0]
            lengths = (lengths - kernel + 2 * padding) // stride + 1
            input_mask = make_pad_mask(lengths)
            input_mask = input_mask.unsqueeze(-1)

        out.masked_fill_(input_mask, 0)
            
        return out, lengths

class SubRNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, rnn_type='lstm', sub=False):
        super(SubRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=False, bidirectional=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.transform = nn.Linear(2 * hidden_size, out_size, bias=False)
        self.sub = sub

    def forward(self, inputs, lengths):
        assert inputs.dim() == 4
        B,C,T,D = inputs.size()   
        if self.sub:
            assert C == self.input_size, (C, self.input_size)
            # to [T,B,D,C]
            re_inputs = torch.reshape(inputs.permute(2,0,3,1), [T,B*D,C])
            re_lengths = lengths.view([B,1]).repeat(1,D).reshape([-1]).cpu()
        else:
            assert D*C == self.input_size, (D*C, self.input_size)
            re_inputs = torch.reshape(inputs.permute(2,0,3,1), [T,B,D*C])
            re_lengths = lengths.cpu()

        packed_inputs = pack_padded_sequence(re_inputs, re_lengths, enforce_sorted=False, batch_first=False)
        packed_inputs = self.rnn(packed_inputs)[0]
        packed_inputs,_ = pad_packed_sequence(packed_inputs, batch_first=False)
        packed_inputs = self.transform(packed_inputs)
        # to [B,C,D,T]
        packed_inputs = torch.reshape(packed_inputs, [T,B,D,-1]).permute(1,3,0,2).contiguous()
        return packed_inputs

class LSTMP(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, houyi_init=False):
        super(LSTMP, self).__init__()
        self.hidden_size = hidden_size
        self.proj_size = proj_size

        self.wx = Linear(input_size, hidden_size * 4)
        self.wr = Linear(proj_size, hidden_size * 4, bias=False)
        self.wo = Linear(hidden_size, proj_size, bias=False)
        self.wc = torch.nn.Parameter(torch.Tensor(1, hidden_size * 3))

        if houyi_init:
            self.reset_parameters()

    def reset_parameters(self):
        mean = 0.0
        stddev = 0.02
        for weight in self.parameters():
            weight.data.normal_(mean, stddev)
        
    def forward(self, input, lengths):
        '''
            input: (T,N,C)
        '''
        T, N = input.size(0), input.size(1)

        out_seq = []

        wc_i = self.wc[:, :self.hidden_size]
        wc_f = self.wc[:, self.hidden_size:self.hidden_size*2]
        wc_o = self.wc[:, self.hidden_size*2:self.hidden_size*3]

        c_prev = torch.zeros(N, self.hidden_size)
        r_prev = torch.zeros(N, self.proj_size)
        if self.training:
            c_prev = c_prev.cuda()
            r_prev = r_prev.cuda()

        for t in range(T):
            gcec_o = self.wx(input[t,:,:]) + self.wr(r_prev) # [N, hidden_size * 4]
            i_o, f_o, o_o, c_o = gcec_o.chunk(4, -1) # [N, hidden_size]

            in_gate = F.sigmoid(i_o + wc_i * c_prev) # [N, hidden_size]
            forget_gate = F.sigmoid(f_o + wc_f * c_prev) # [N, hidden_size]

            c_cur = forget_gate * c_prev + in_gate * F.tanh(c_o) # [N, hidden_size]
            out_gate = F.sigmoid(o_o + wc_o * c_cur) # [N, hidden_size]

            o_t = out_gate * F.tanh(c_cur) # [N, hidden_size]
            r_t = self.wo(o_t) # [N, proj_size]

            c_prev = c_cur
            r_prev = r_t
            out_seq.append(r_t)

        output = torch.stack(out_seq, dim=0) # [T, N, proj_size]
        
        with torch.no_grad():
            mask = make_pad_mask(lengths)
            mask = mask.permute(1, 0).unsqueeze(-1) # [T, N, 1]

        output.masked_fill_(mask, 0)

        return output

class SkipLayer(nn.Module):
    def __init__(self, step, type='nctw'):
        super(SkipLayer, self).__init__()
        self.step = step
        self.type = type
        self.rand_start = 0

    def forward(self, input, lengths=None, label=None):
        #self.rand_start = (self.rand_start + 1) % self.step
        if self.type == 'nctw':
            out = input[:, :, self.rand_start::self.step, :]
        elif self.type == 'tnc':
            out = input[self.rand_start::self.step, :, :]
        else:
            assert False, 'invalid skip type: %s' % self.type

        if lengths is not None:
            lengths = (lengths - self.rand_start + self.step - 1) // self.step
    
        if label is not None:
            assert (label.dim() == 2)
            sample_num = input.size(0)
            new_label = label.reshape(-1, sample_num, label.size(-1)) # [T, N, 1]
            new_label = new_label[self.rand_start::self.step, :, :]
            new_label = new_label.reshape(-1, label.size(-1))
            
        if label is None:
            if lengths is None:
                return out
            else:
                return out, lengths
        else:
            if lengths is None:
                return out, new_label
            else:
                return out, lengths, new_label
 
class ISTFT(nn.Module):
    def __init__(self, nfft, shift, window):
        super(ISTFT, self).__init__()

        self.nfft = nfft
        self.shift = shift
        self.window = window

    def forward(self, input, lengths):
        '''
            input: (B, C, T, F)
        '''
        B, C, T, F = input.size() 
        assert T == torch.max(lengths).item()
        out_len = (T - 1) * self.shift + self.window
        device = input.device
        input = torch.reshape(input, [-1, T, F]).permute(0, 2, 1)
        output = torch.istft(input, self.nfft, self.shift, self.window, window=torch.hamming_window(self.window, periodic=False).to(device), center=False, normalized=False, onesided=True, length=out_len, return_complex=False)
        output = torch.reshape(output, [B, C, -1])
        return output

class SubbandCompose(nn.Module):
    def __init__(self, window):
        super(SubbandCompose, self).__init__()

        self.window = window
        self.filter_len = window * 6
        self.band_num = window * 2        

    def forward(self, input, lengths):
        '''
            input: (B, C, T, F)   
        '''
        B, C, T, F = input.size()
        device = input.device
        assert T == torch.max(lengths).item()
        subband_filter_lst = self._read_filter('xspeech/dataio/subband_filter.lst')
        subband_filter_128 = subband_filter_lst[1]
        sample_num = len(lengths)
        #out_len = lengths * channels * window
        input = torch.reshape(input, [-1, F])
        ifft_out = torch.fft.irfft(input, norm="forward")
        total_frame = input.size(0)
        assert input.size(0) % sample_num == 0
        in_height = input.size(0) // sample_num
        assert in_height > 5
        wide_band_in = torch.zeros(total_frame, self.filter_len).to(device)       
        for i in range(0,3):
            wide_band_in[:,i*self.band_num:(i+1)*self.band_num] = ifft_out[:,:]

        filter = torch.tensor(subband_filter_128).to(device)
        filter = filter.unsqueeze(0)
        wide_band_in = wide_band_in * filter
        wide_band_in = wide_band_in * self.window
        true_len = in_height - 5
        output = torch.zeros(true_len*sample_num, self.window).to(device)
        
        for i in range(sample_num):
            one_band_in_ori = torch.reshape(wide_band_in[i*in_height:(i+1)*in_height, :], [-1, 1])
            one_band_in = torch.reshape(wide_band_in[i*in_height:(i+1)*in_height, :], [-1, 6, self.window]).permute(1, 0, 2)
            pre_output = torch.zeros((6, in_height-5, self.window)).to(device)
            for k in range(6):
                pre_output[k, :, :] = one_band_in[k, 6-k-1:in_height-k, :]
            output[i*true_len:(i+1)*true_len] = torch.sum(pre_output, 0)
            #output[i*in_height:(i+1)*in_height-5, :] = torch.sum(pre_output, 0)
        #    for p in range(in_height-5, in_height):
        #        for j in range(self.window):
        #            val = 0
        #            for k in range(6):
        #                if p + k >= in_height:
        #                    break
        #                tmp = one_band_in_ori[(p + k) * self.filter_len + (6 - k - 1) * self.window + j, :]
        #                val += tmp.item()
        #            output[i * in_height + p, j] = val
        output = torch.reshape(output, [B, C, -1]) 
        return output / 32768.0

    def _read_filter(self, file):
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

'''
if __name__ == '__main__':
    x = torch.rand(2, 4, 3, 4)
    lengths = torch.tensor([4, 4], dtype=torch.int32)
    num_groups = 2
    #gn = nn.GroupNorm(num_groups=num_groups, num_channels=x.shape[1], eps=0, affine=True)
    #out = gn(x)
    #my_gn = MaskedGroupNorm(num_groups=num_groups, num_channels=x.shape[1], eps=0, affine=True)
    #my_out = my_gn(x, lengths)
    
    bn = nn.BatchNorm2d(num_features=4, eps=1e-6, momentum=1e-3)
    out = bn(x)
    my_bn = MaskedBatchNorm2d(num_features=4, eps=1e-6, momentum=1e-3)
    my_out = my_bn(x, lengths)
    print(out)
    print(my_out)
    print((out - my_out).sum())
'''
