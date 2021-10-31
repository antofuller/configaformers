import torch.nn.functional as F
import torch
from torch import nn, einsum
import math
from norm_module import get_norm, init_norm
from einops import rearrange, repeat, reduce
from utils import set_default


class AttentionOffset(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Module for offsetting the attention dots matrix
        """
        # Configure input(s) and output(s)
        self.input_name_attn_dots = set_default(_look='input_name_attn_dots', _dict=config, _default='attn_dots')
        self.input_name_attn_offset = set_default(_look='input_name_attn_offset', _dict=config, _default='attn_bias')
        self.output_name = set_default(_look='output_name', _dict=config, _default='attn_dots')

        # attn_dots will be of shape (bsz, num_heads, length_queries, length_keys)
        self.num_heads = _streams[self.input_name_attn_dots][1]
        len_queries = _streams[self.input_name_attn_dots][-2]
        len_keys = _streams[self.input_name_attn_dots][-1]

        num_heads_bias = _streams[self.input_name_attn_bias][1]
        assert num_heads_bias == self.num_heads, f'num_heads in the attention bias ({num_heads_bias}) must be ' \
                                                 f'equal to num_heads in the attention dots ({self.num_heads})!'

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_attn_dots, ['BSZ', self.num_heads, len_queries, len_keys]],
                                             [self.input_name_attn_offset, ['BSZ', self.num_heads, len_queries, len_keys]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', self.num_heads, len_queries, len_keys]],
                                              ]
                                  }

    def forward(self, _data):
        b, h, i, j = _data[self.input_name_attn_dots].shape

        if _data[self.input_name_attn_offset].shape[0] != b:
            _data[self.input_name_attn_offset] = _data[self.input_name_attn_offset].repeat(b, 1, 1, 1)

        # trim the bias tensor such that it matches the shape of attn_dots
        _data[self.input_name_attn_offset] = _data[self.input_name_attn_offset][:, :, :i, :j]

        _data[self.input_name_attn_dots] = _data[self.input_name_attn_dots] + _data[self.input_name_attn_offset]

        return _data


def _get_slopes(heads):
    # This implementation is taken 100% from lucidrains
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def build_attention_offset(func,
                           query_inputs,
                           context_inputs):
    stacked_rows = []
    for i in query_inputs:
        row = []
        for j in context_inputs:
            offset = func(i, j)
            row.append(offset)
        stacked_rows.append(row)

    return torch.Tensor(stacked_rows).view(1, 1, len(query_inputs), len(context_inputs))


def get_alibi(num_heads,
              max_length=2048,
              slopes=None,
              mask='causal',
              mask_precision='full',
              ):
    if mask == 'causal':
        if mask_precision == 'half':
            dummy_tensor = torch.Tensor([1.0]).half()
            mask_value = max_neg_value(dummy_tensor)
        else:
            dummy_tensor = torch.Tensor([1.0]).float()
            mask_value = max_neg_value(dummy_tensor)

        def my_func(x1, x2):
            if x2 > x1:
                return mask_value
            else:
                return -abs(x1 - x2)
    else:
        def my_func(x1, x2):
            return -abs(x1 - x2)

    offset_template = build_attention_offset(func=my_func,
                                             query_inputs=range(max_length),
                                             context_inputs=range(max_length))

    if slopes is None:
        slopes = _get_slopes(heads=num_heads)

    offset_list = []
    for n in range(num_heads):
        offset_list.append(offset_template*slopes[n])

    offsets = torch.cat(offset_list, dim=0)
    return offsets

