import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        """
        This rotary embedding (RoPE) implementation is taken 100% from lucidrains. Explanation forthcoming.
        :param dim: number of features to rotate
        """
        super().__init__()

        # Create a frequency for every second dimension, start with f = 1.0, divide by 10 to get the next, and repeat
        # For example, if dim = 8, inv_freq = [1.0, 0.1, 0.01, 0.001]
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq).cuda()  # count up from 0 to (max_seq_len - 1)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)  # multiply t with inv_freq, shape (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # repeat freqs once, shape (max_seq_len, dim)
        return rearrange(emb, 'n d -> () () n d')


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)  # split the features into two
    x1, x2 = x.unbind(dim=-2)  # separate them
    return torch.cat((-x2, x1), dim=-1)  # rotate one of them (multiplying by negative 1), return the concatenation


def apply_rotary_pos_emb(x, frequencies):
    num_features = frequencies.shape[-1]  # The number of features we wish to rotate
    x_rotate = x[..., :num_features]  # Features to rotate
    x_orig = x[..., num_features:]  # Features to keep, as is

    seq_len = x_rotate.shape[-2]  # Length of the input
    frequencies = frequencies[:, :, -seq_len:]  # Take the frequencies we need (just up to seq_len)
    x_rotate = (x_rotate * frequencies.cos()) + (rotate_half(x_rotate) * frequencies.sin())  # Apply rotation

    x = torch.cat([x_rotate, x_orig], dim=-1)  # Piece back together
    return x
