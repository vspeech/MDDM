"""Positionwise feed forward layer definition."""
  
import torch
import torch.nn as nn
import torch.nn.functional as F
from xspeech.module.masked_modules import MaskedLinear

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = MaskedLinear(d_model, d_ff)
        self.w_2 = MaskedLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        return self.w_2(self.dropout(F.relu(self.w_1(x, lengths))), lengths)
