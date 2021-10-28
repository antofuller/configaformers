import torch.nn.functional as F
import torch
from torch import nn
from typing import List

# This module is dedicated to Norm Macdonald


# Implementations from https://github.com/lucidrains/x-transformer

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        _norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / _norm.clamp(min=self.eps) * self.g


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def get_norm(norm_type: str, dim: int):
    # TODO: Batch norm may involve rearranging
    norm_type = norm_type.lower()  # Make lowercase
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim)

    if norm_type == 'rms_norm':
        return RMSNorm(dim)

    if norm_type == 'scale_norm':
        return ScaleNorm(dim)

    else:
        print(f"Norm: {norm_type} not available.")

