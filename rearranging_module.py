import torch.nn.functional as F
import torch
from torch import nn, einsum
from norm_module import get_norm, init_norm
from einops import rearrange, repeat, reduce
from utils import set_default


class MakeHeads(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Rearrange tensor module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]

        # Checking attention head settings (if num_heads is not given, default to 1)
        self.num_heads = set_default(_look='num_heads', _dict=config, _default=1, _type=int)
        assert self.input_dim % self.num_heads == 0, "num_heads must divide evenly into input_dim!"
        self.head_dim = int(self.input_dim / self.num_heads)
        self.scale = self.head_dim ** -0.5

        self.output_dim = self.head_dim

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', self.num_heads, len_input, self.head_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name],
                                            'batch length (num_heads head_dim) -> batch num_heads length head_dim',
                                            num_heads=self.num_heads)

        return _data
