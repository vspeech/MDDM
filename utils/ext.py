import os
import sys
import glob

import torch
from torch.utils.cpp_extension import load

DP = load(name='data_perturb', verbose=True,
          build_directory='ninja_build',
          sources=['extensions/data_perturb/data_perturb.cpp',
                   'extensions/data_perturb/data_perturb_ops.cu'])

