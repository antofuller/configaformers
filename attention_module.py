import torch.nn.functional as F
import torch
from torch import nn, einsum
from utils import set_default


def get_attention_function(attn_type: str):
    attn_type = attn_type.lower()  # Make lowercase
    if attn_type == 'softmax':
        return F.softmax

    elif attn_type == 'relu':
        return F.relu

    elif attn_type == 'tanh':
        return F.tanh

    else:
        print(f"Attention type: {attn_type} not available.")


class MHADots(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        MHA (Multi-Head Attention) Dots (dot-products) module
        """
        # Configure input(s) and output(s)
        self.input_name_queries = set_default(_look='input_name_queries', _dict=config, _default='x')
        self.input_name_keys = set_default(_look='input_name_keys', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='attn_dots')

        self.input_dim_queries = _streams[self.input_name_queries][-1]
        self.input_dim_keys = _streams[self.input_name_keys][-1]
        len_queries = _streams[self.input_name_queries][-2]
        len_keys = _streams[self.input_name_keys][-2]

        self.num_heads = _streams[self.input_name_queries][1]
        num_heads_keys = _streams[self.input_name_keys][1]
        batch = _streams[self.input_name_queries][0]

        assert self.input_dim_queries == self.input_dim_keys, f'Queries dim ({self.input_dim_queries}) must equal' \
                                                              f' keys dim ({self.input_dim_keys})'

        assert self.num_heads == num_heads_keys, f'Queries num_heads ({self.num_heads}) must equal' \
                                                              f' keys num_heads ({num_heads_keys})'

        self.input_dim = self.input_dim_queries
        self.output_dim = self.input_dim  # Set output_dim equal to input_dim for now (this isn't really correct)
        self.scale = self.input_dim_queries ** -0.5

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_queries, [batch, self.num_heads, len_queries, self.input_dim_queries]],
                                             [self.input_name_keys, [batch, self.num_heads, len_keys, self.input_dim_keys]],
                                             ],

                                  'outputs': [[self.output_name, [batch, self.num_heads, len_queries, len_keys]],
                                              ]
                                  }

    def forward(self, _data):
        # Attention operates on a set, so it must receive inputs of shape (bsz, num_heads, length, head_dim)

        # Perform attention operation, and insert into _data
        _data[self.output_name] = einsum('b h i d, b h j d -> b h i j',
                                         _data[self.input_name_queries], _data[self.input_name_keys]) * self.scale

        return _data


class MHAWeightedSum(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        MHA (Multi-Head Attention) Weighted Summation module
        """
        # Configure input(s) and output(s)
        self.input_name_values = set_default(_look='input_name_values', _dict=config, _default='x')
        if 'input_name_attn_scores' in config.keys():
            # Configure module to use pre-calculated attention scores, no need for dots
            self.input_name_scores = set_default(_look='input_name_attn_scores', _dict=config, _default='attn_scores')
            attn_matrix_shape = _streams[self.input_name_scores]
            attn_matrix_name = self.input_name_scores
            self.use_old_scores = True
        else:
            self.input_name_dots = set_default(_look='input_name_attn_dots', _dict=config, _default='attn_dots')
            attn_matrix_shape = _streams[self.input_name_dots]
            attn_matrix_name = self.input_name_dots
            self.attention_type = set_default(_look='attn_function', _dict=config, _default='softmax')
            self.attn_function = get_attention_function(attn_type=self.attention_type)
            self.use_old_scores = False

        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.output_name_attn_scores = set_default(_look='output_name_attn_scores', _dict=config,
                                                   _default=False, _type=None)

        self.input_dim_values = _streams[self.input_name_values][-1]
        self.num_heads = attn_matrix_shape[1]  # For dots, this will be num_heads i.e. (dots.shape[1])
        len_queries = attn_matrix_shape[-2]
        len_keys = attn_matrix_shape[-1]
        batch = _streams[self.input_name_values][0]

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_values, [batch, self.num_heads, len_keys, self.input_dim_values]],
                                             [attn_matrix_name, [batch, self.num_heads, len_queries, len_keys]],
                                             ],

                                  'outputs': [[self.output_name, [batch, self.num_heads, len_queries, self.input_dim_values]],
                                              ]
                                  }

        if self.output_name_attn_scores:
            self.streams_in_module['outputs'].append([self.output_name_attn_scores, [batch, self.num_heads, len_queries, len_keys]])

    def forward(self, _data):
        if self.use_old_scores:
            # If we are given pre-calculated attention scores, then use them here
            attention_scores = _data[self.input_name_scores]
        else:
            # Make attention map out of attention dots (usually just a softmax over the last dimension)
            attention_scores = self.attn_function(_data[self.input_name_dots], dim=-1)

        # Return attention_scores if configured (these can be re-used in deeper layers, see LazyFormer)
        if self.output_name_attn_scores:
            _data[self.output_name_attn_scores] = attention_scores

        # Weighted sum of value heads based on attn_map
        _data[self.output_name] = einsum('b h i j, b h j d -> b h i d', attention_scores, _data[self.input_name_values])
        return _data