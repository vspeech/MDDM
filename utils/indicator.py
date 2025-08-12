#
# @file:     indicator.py
# @author:   zousaisai@baidu.com
# @created:  2021-06-03 15:48
# @modified: 2021-06-03 15:48
# @brief: 
#

import glog
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_digits, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_digits)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        glog.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_digits):
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + ']'


class TensorAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, ten):
        self.name = name
        self.avg = torch.zeros_like(ten)
        self.sum = torch.zeros_like(ten)
        self.count = torch.zeros_like(ten)

    def reset(self):
        self.avg.zero_()
        self.sum.zero_()
        self.count.zero_()

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '\n\t {name} {avg}'
        return fmtstr.format(**self.__dict__)

