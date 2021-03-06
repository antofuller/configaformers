import torch
from torch import nn
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
                 _streams,
                 ):
        super().__init__()
        """
        Norm module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        input_shape = _streams[self.input_name]

        # Configuring norm
        norm_name = set_default(_look='norm_type', _dict=config, _default='layer_norm')
        self.norm = get_norm(norm_type=norm_name, dim=self.input_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, input_shape],
                                             ],

                                  'outputs': [[self.output_name, input_shape],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = self.norm(_data[self.input_name])
        return _data


class ScaleAlongDimension(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Learned scale used in as a weighted residual, or for scaling mha heads (see NormFormer)
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.dim_to_scale = set_default(_look='dim_to_scale', _dict=config, _default=2, _type=int)
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_shape = _streams[self.input_name]
        assert self.dim_to_scale > 0, f'dim_to_scale must be greater than 0!'
        assert self.dim_to_scale <= len(self.input_shape), f'dim_to_scale must less than or equal to the number of ' \
                                                           f'input dimensions!'
        num_params = self.input_shape[self.dim_to_scale]

        # Initialize gate to 1
        self.scale = nn.Parameter(torch.ones(num_params), requires_grad=True)

        # Built einsum input strings
        self.einsum_in_1 = 'abcdef'  # max of 6 dims
        self.einsum_in_1 = self.einsum_in_1[:len(self.input_shape)]
        self.einsum_in_2 = self.einsum_in_1[self.dim_to_scale]

        print(f"{self.einsum_in_1},{self.einsum_in_2}->{self.einsum_in_1}")

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, self.input_shape],
                                             ],

                                  'outputs': [[self.output_name, self.input_shape],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = torch.einsum(f'{self.einsum_in_1},{self.einsum_in_2}->{self.einsum_in_1}', _data[self.input_name], self.scale)
        return _data