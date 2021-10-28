import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math

from typing import Optional, Tuple, Union, List, Dict
from linear_module import LinearProj
from activation_module import Activation




class FFN(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        """
        Feed-forward network.
        """

        self.proj_up = LinearProj(config['proj_up'])
        self.activation = Activation(config['activation'])
        self.proj_down = LinearProj(config['proj_down'])

    def forward(self, x):
        x = self.proj_up(x)
        x = self.activation(x)
        x = self.proj_down(x)
        return x




