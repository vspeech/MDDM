# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import librosa
import logging
import json
import random
import tarfile
import numpy as np
import scipy.signal as ss
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from xspeech.se.data_augment import apply_one_fir
from xspeech.dataio.util import *

torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def compute_noise(signal, noise, snr):
    weight = 10 ** (float(snr) / 10)
    signal_square = signal * signal
    sum_signal = float(signal_square.sum())
    noise_square = noise * noise
    sum_noise = float(noise_square.sum())
    col1 = sum_noise * weight + 1e-6
    #col1 = (sum_noise + 1e-6) * weight
    col0 = (sum_signal / col1) ** 0.5
    return noise * col0

def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        #assert 'key' in obj
        assert 'wav' in obj
        #key = obj['key']
        wav_file = obj['wav']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(wav=waveform,
                           txt=wav_file,
                           sample_rate=sample_rate)
            if 'test_wav_truth' in obj:
                wav_truth_file = obj['test_wav_truth']
                waveform_truth, _ = torchaudio.load(wav_truth_file)
                waveform_truth = waveform_truth * (1 << 15)
                waveform_noisy = waveform * (1 << 15)
                example['wave_truth'] = waveform_truth
                example['wave_noisy'] = waveform_noisy
                example['test_wav_truth'] = waveform_truth
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=10240,
           min_length=10,
           frame_length=1536,
           frame_shift=480):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert sample['sample_rate'] == 48000
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        #num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        num_durs = sample['wav'].size(1) / sample['sample_rate']
        sample_len = sample['wav'].size(1)
        max_num_length = int(max_length * sample['sample_rate'])
        if num_durs < min_length:
            continue
        if num_durs > max_length:
            #st = random.randint(0, sample_len - max_num_length)
            #sample['wav'] = sample['wav'][:, st:st+max_num_length]
            sample['wav'] = sample['wav'][:, :max_num_length]
        #elif sample_len < max_num_length - 2 * sample['sample_rate']:
        #    copy_num = int(max_num_length // sample_len)
        #    if copy_num == 0:
        #        copy_num += 1
        #    sample_tmp = sample['wav'].expand(copy_num, sample_len)
        #    sample_tmp = sample_tmp.reshape(-1, copy_num * sample_len)
        #    res_len = max_num_length - copy_num * sample_len
        #    st_len = sample_len - res_len
        #    sample_tail = sample['wav'][:, st_len:]
        #    sample['wav'] = torch.cat((sample_tmp, sample_tail), dim=1)
        #else:
        #    tensor_zero = torch.zeros([1, max_num_length - sample_len])
        #    sample['wav'] = torch.cat((sample['wav'], tensor_zero), dim=1)
        
        #length = sample['wav'].size(1)
        #num_frames = (length - frame_length) // frame_shift + 1
        #frame_pad = int(np.ceil(num_frames/64)) * 64 - num_frames
        #num_frames += frame_pad
        #wav_pad = (num_frames - 1) * frame_shift + frame_length - length
        #if wav_pad >= 0:
        #    sample['wav'] = F.pad(sample['wav'], (0, wav_pad))
        #else:
        #    sample['wav'] = sample['wav'][:, :wav_pad]
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample

def reverb_rir(data,
               rir_list_file="",
               prob=0.5,
               zoom=True,
               min_ampl=500,
               max_ampl=26000,
               rir_lists=[]):
    rir_nums = len(rir_lists)
    for sample in data:
        assert 'wav' in sample
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # conv rir
        seq_len = np.array([waveform.size(1)])
        #frames = waveform.numpy()
        #orig_frames_shape = frames.shape
        rir_prob = float(torch.rand(1))
        rir_flag = 0
        if rir_prob < prob:
            rand_idx_rir = random.randint(0, rir_nums-1)
            rir_list = rir_lists[rand_idx_rir].split()
            rir = rir_list[1].strip()
            assert(int(rir_list[-1]) % 4 == 0)
            one_len = int(rir_list[-1]) // 4
            rir_offset = int(rir_list[2])
            rir_seq = np.fromfile(rir, dtype=np.float32, count=one_len, offset=rir_offset)
            rir_seq_truth = np.fromfile(rir, dtype=np.float32, count=2400, offset=rir_offset)
            rir_seq = torch.tensor(rir_seq)
            rir_seq_truth = torch.tensor(rir_seq_truth)
            N1 = ceil2i(one_len) * 2
            N2 = ceil2i(2400) * 2
            firs1 = torch.zeros((1, N1), dtype=torch.float32)
            firs2 = torch.zeros((1, N2), dtype=torch.float32)
            firs1[:, :one_len] = rir_seq
            firs2[:, :2400] = rir_seq_truth
            #print(sample['txt'])
            wave_rir = apply_one_fir(waveform, firs1, one_len, seq_len, 1, 1)
            wave_truth = apply_one_fir(waveform, firs2, 2400, seq_len, 1, 1)
            wave_rir = wave_rir.reshape(1, -1)
            wave_truth = wave_truth.reshape(1, -1)
            rir_flag = 1
        else:
            wave_rir = waveform
            wave_truth = waveform
        
        if zoom:
            src_max = float(wave_rir.max()) + 1e-4
            tar_max = random.uniform(min_ampl, max_ampl)
            zoom_ratio = tar_max / src_max
            wave_truth *= zoom_ratio
            wave_rir *= zoom_ratio

        sample['wave_rir'] = wave_rir
        sample['wave_truth'] = wave_truth
        sample['rir_flag'] = rir_flag
        yield sample

def add_noise(data,
              noise_list_file="",
              prob=0.5,
              min_snr=0,
              max_snr=25,
              noise_lists=[],
              min_ampl=1500,
              max_ampl=26000,
              thre=32768.0,
              step=10,
              idx=0):
    noise_nums = len(noise_lists)
    random.shuffle(noise_lists)
    for sample in data:
        assert 'wave_rir' in sample
        assert 'wave_truth' in sample
        waveform = sample['wave_rir']
        waveform_truth = sample['wave_truth']
        rir_flag = sample['rir_flag']
        length = waveform.shape[1] 
        ## add noise pcm 48k
        noise_prob = float(torch.rand(1))
        if noise_prob < prob or rir_flag == 0:
            if idx + step >= noise_nums:
                end = noise_nums - 1
            else:
                end = idx + step
            rand_idx = random.randint(idx, end)
            noise_list = noise_lists[rand_idx].split()
            if end == noise_nums - 1:
                idx = 0
            else:
                idx += step + 1
            #logging.info("IDX: {}".format(rand_idx))
            #noise_list = noise_lists[rand_idx].split()
            noise_file = noise_list[1].strip()
            if len(noise_list) == 4:
                noise_offset = int(noise_list[2])
                assert(int(noise_list[3]) % 2 == 0)
                noise_len = int(noise_list[3]) // 2
            else:
                assert len(noise_list) == 3
                noise_tot_len = int(noise_list[2]) // 2
                if noise_tot_len - 1 < length:
                    noise_offset = 0
                    noise_len = noise_tot_len
                else:
                    noise_offset = random.randint(0, noise_tot_len-length-1)
                    noise_offset = noise_offset * 2
                    noise_len = length
            noise = np.fromfile(noise_file, dtype=np.int16, count=noise_len, offset=noise_offset)
            noise = torch.tensor(noise, dtype=torch.float32)
            noise = noise.unsqueeze(0)
            if noise_len > length:
                start = random.randint(0, noise_len - length - 1)
                noise_new = noise[:, start:start+length]
            else:
                extra_len = length - noise_len
                extra_noise = noise[:, :extra_len]
                noise_new = torch.cat((noise, extra_noise), 1) 
            snr = random.randint(min_snr, max_snr)
            noise_new = compute_noise(waveform, noise_new, snr)
            wave_noisy = waveform + noise_new
        else:
            wave_noisy = waveform
       
        src_max = float(wave_noisy.max()) + 1e-6
        tar_max = random.uniform(min_ampl, max_ampl)
        zoom_ratio = tar_max / src_max
        waveform_truth *= zoom_ratio
        wave_noisy *= zoom_ratio
        #scale0 = torch.max(torch.abs(waveform_truth)).item()
        #scale = torch.max(torch.abs(wave_noisy)).item()
        #if scale0 > thre:
        #    waveform_truth = torch.div(waveform_truth, scale0 / thre)
        #if scale > thre:
        #    wave_noisy = torch.div(wave_noisy, scale / thre)

        sample['wave_noisy'] = wave_noisy
        sample['wave_truth'] = waveform_truth
        #sample['wave_noisy'] = wave_noisy / (1 << 15)
        #sample['wave_truth'] = waveform_truth / (1 << 15)
        yield sample

def compute_fft(data,
                frame_length=25,
                frame_shift=10,
                nfft=25):
    """ Extract fft

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wave_truth' in sample
        assert 'wave_noisy' in sample
        sample_rate = sample['sample_rate']
        wave_truth = sample['wave_truth']
        wave_noisy = sample['wave_noisy']

        ## Only keep feat 
        mat_truth = torch.stft(wave_truth.squeeze(0), nfft, frame_shift, frame_length, window=torch.hamming_window(frame_length, periodic=False), center=False, normalized=False, return_complex=True)
        mat_noisy = torch.stft(wave_noisy.squeeze(0), nfft, frame_shift, frame_length, window=torch.hamming_window(frame_length, periodic=False), center=False, normalized=False, return_complex=True)
        mat_truth = mat_truth.permute(1,0)
        mat_noisy = mat_noisy.permute(1,0)
        sample['feat_truth'] = mat_truth
        sample['feat_noisy'] = mat_noisy
        yield sample

def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        sample['feat'] = mat
        yield sample


def compute_mfcc(data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        sample['feat'] = mat
        yield sample


def compute_log_mel_spectrogram(data,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
        if padding > 0:
            waveform = F.pad(waveform, (0, padding))
        window = torch.hann_window(n_fft)
        stft = torch.stft(waveform,
                          n_fft,
                          hop_length,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs()**2

        filters = torch.from_numpy(
            librosa.filters.mel(sr=sample_rate,
                                n_fft=n_fft,
                                n_mels=num_mel_bins))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        sample['feat'] = log_spec.transpose(0, 1)
        yield sample

def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def spec_trim(data, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            if 'feat_noisy' not in sample:
                buf.sort(key=lambda x: x['wave_noisy'].size(1))
            else:
                buf.sort(key=lambda x: x['feat_noisy'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    if 'feat_noisy' not in sample:
        buf.sort(key=lambda x: x['wave_noisy'].size(1))
    else:
        buf.sort(key=lambda x: x['feat_noisy'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat}]]

        Returns:
            Iterable[Tuple(keys, feats, feats lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat_noisy'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat_noisy'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat_noisy'] for i in order]
        sorted_feats_truth = [sample[i]['feat_truth'] for i in order]
        #sorted_keys = [sample[i]['key'] for i in order]
        sorted_wavs = [sample[i]['wave_noisy'].squeeze(0) for i in order]
        sorted_wavs_truth = [sample[i]['wave_truth'].squeeze(0) for i in order]
        wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                                   dtype=torch.int32)

        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padded_feats_truth = pad_sequence(sorted_feats_truth,
                                    batch_first=True,
                                    padding_value=0)
        padded_wavs = pad_sequence(sorted_wavs,
                                   batch_first=True,
                                   padding_value=0)
        padded_wavs_truth = pad_sequence(sorted_wavs_truth,
                                   batch_first=True,
                                   padding_value=0)
        
        if 'test_wav_truth' in sample[0]:
            yield {
                "feats": padded_feats,
                "feats_truth": padded_feats_truth,
                "feats_lengths": feats_lengths,
                "pcm": padded_wavs,
                "pcm_truth": padded_wavs_truth,
                "pcm_lengths": wav_lengths,
                "test_wav_truth": padded_wavs_truth,
            }
        else:
            yield {
                "feats": padded_feats,
                "feats_truth": padded_feats_truth,
                "feats_lengths": feats_lengths,
                "pcm": padded_wavs,
                "pcm_truth": padded_wavs_truth,
                "pcm_lengths": wav_lengths,
            }

def padding_pcm(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat}]]

        Returns:
            Iterable[Tuple(keys, feats, feats lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        pcm_length = torch.tensor([x['wave_noisy'].size(1) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(pcm_length, descending=True)
        sorted_wavs = [sample[i]['wave_noisy'].squeeze(0) for i in order]
        sorted_wavs_truth = [sample[i]['wave_truth'].squeeze(0) for i in order]
        wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                                   dtype=torch.int32)

        padded_wavs = pad_sequence(sorted_wavs,
                                   batch_first=True,
                                   padding_value=0)
        padded_wavs_truth = pad_sequence(sorted_wavs_truth,
                                   batch_first=True,
                                   padding_value=0)
        
        #meant = padded_wavs.mean(dim=1, keepdim=True)
        #stdt = padded_wavs.std(dim=1, keepdim=True)
        #padded_wavs = (padded_wavs - meant) / (1e-5 + stdt)
        #padded_wavs_truth = (padded_wavs_truth - meant) / (1e-5 + stdt)

        if 'test_wav_truth' in sample[0]:
            yield {
                "pcm": padded_wavs,
                "pcm_truth": padded_wavs_truth,
                "pcm_lengths": wav_lengths,
                "test_wav_truth": padded_wavs_truth,
            }
        else:
            yield {
                "pcm": padded_wavs,
                "pcm_truth": padded_wavs_truth,
                "pcm_lengths": wav_lengths,
            }
