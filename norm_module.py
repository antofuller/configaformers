import torch.nn.functional as F
import torch
from torch import nn
from typing import List
from utils import set_default


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


def init_norm(_key, _config, _dim):
    if _key not in _config:
        norm_bool = False
        norm_function = False
    else:
        assert type(_config[_key]) == str, f"{_config[_key]} is type {type(_config[_key])}, but should be a string!"
        norm_bool = True
        norm_function = get_norm(norm_type=_config[_key], dim=_dim)

    return norm_bool, norm_function


def get_norm(norm_type: str, dim: int):
    # TODO: Batch norm may involve rearranging
    norm_type = norm_type.lower()  # Make lowercase
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim)

    elif norm_type == 'rms_norm':
        return RMSNorm(dim)

    elif norm_type == 'scale_norm':
        return ScaleNorm(dim)

    else:
        print(f"Norm: {norm_type} not available.")


class Norm(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        # Checking input_dim settings
        assert 'input_dim' in config, f"Norm module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside norm module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']
        self.output_dim = self.input_dim

        # Configuring input_norm and output_norm
        norm_name = set_default(_key='norm_type', _dict=config, _default='layer_norm')
        self.norm = get_norm(norm_type=config[norm_name], dim=self.input_dim)

        # Configuring names
        self.input_name = set_default(_key='input_name', _dict=config, _default='x')
        self.output_name = set_default(_key='output_name', _dict=config, _default='x')

    def forward(self, _data):
        _data[self.output_name] = self.norm(_data[self.input_name])
        return _data