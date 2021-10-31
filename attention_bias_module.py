import torch.nn.functional as F
import torch
from torch import nn, einsum
from norm_module import get_norm, init_norm
from einops import rearrange, repeat, reduce
from utils import set_default


class AttentionBias(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Module for biasing the attention dots matrix
        """
        # Configure input(s) and output(s)
        self.input_name_attn_dots = set_default(_look='input_name_attn_dots', _dict=config, _default='attn_dots')
        self.input_name_attn_bias = set_default(_look='input_name_attn_bias', _dict=config, _default='attn_bias')
        self.output_name = set_default(_look='output_name', _dict=config, _default='attn_dots')

        # attn_dots will be of shape (bsz, num_heads, length_queries, length_keys)
        self.num_heads = _streams[self.input_name_attn_dots][1]
        len_queries = _streams[self.input_name_attn_dots][-2]
        len_keys = _streams[self._attn_dots][-1]

        num_heads_bias = _streams[self.input_name_attn_bias][1]
        assert num_heads_bias == self.num_heads, f'num_heads in the attention bias ({num_heads_bias}) must be ' \
                                                 f'equal to num_heads in the attention dots ({self.num_heads})!'

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_attn_dots, ['BSZ', self.num_heads, len_queries, len_keys]],
                                             [self.input_name_attn_bias, ['BSZ', self.num_heads, len_queries, len_keys]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', self.num_heads, len_queries, len_keys]],
                                              ]
                                  }

    def forward(self, _data):
        b, h, i, j = _data[self.input_name_attn_dots].shape

        if _data[self.input_name_attn_bias].shape[0] != b:
            _data[self.input_name_attn_bias] = _data[self.input_name_attn_bias].repeat(b, 1, 1, 1)

        # trim the bias tensor such that it matches the shape of attn_dots
        _data[self.input_name_attn_bias] = _data[self.input_name_attn_bias][:, :, :i, :j]

        _data[self.input_name_attn_dots] = _data[self.input_name_attn_dots] + _data[self.input_name_attn_bias]

        return _data