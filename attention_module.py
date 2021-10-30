import torch.nn.functional as F
import torch
from torch import nn, einsum
from norm_module import get_norm, init_norm
from einops import rearrange, repeat, reduce
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

        self.input_dim_queries = _streams[self.input_name_queries]
        self.input_dim_keys = _streams[self.input_name_keys]
        assert self.input_dim_queries == self.input_dim_keys, f'Queries dim ({self.input_dim_queries}) must equal' \
                                                              f' keys dim ({self.input_dim_keys})'
        self.input_dim = self.input_dim_queries
        self.output_dim = self.input_dim  # Set output_dim equal to input_dim for now (this isn't really correct)

        # Checking attention head settings (if num_heads is not given, default to 1)
        self.num_heads = set_default(_look='num_heads', _dict=config, _default=1, _type=int)
        assert self.input_dim % self.num_heads == 0, "num_heads must divide evenly into input_dim!"
        self.head_dim = int(self.input_dim / self.num_heads)
        self.scale = self.head_dim ** -0.5

        # Checking norm_query_heads settings
        self.norm_query_heads_bool, self.norm_query_heads = init_norm(_key='norm_query_heads',
                                                                      _config=config,
                                                                      _dim=self.head_dim)
        # Checking key_value_heads settings
        self.norm_key_heads_bool, self.norm_key_heads = init_norm(_key='norm_key_heads',
                                                                  _config=config,
                                                                  _dim=self.head_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_queries, ['BSZ', 'LEN', self.input_dim_queries]],
                                             [self.input_name_keys, ['BSZ', 'CTX', self.input_dim_keys]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', self.num_heads, 'LEN', 'CTX']],
                                              ]
                                  }

    def forward(self, _data):
        # Attention operates on a set, so it must receive inputs of shape (bsz, seq_len, dim)
        # We first reshape the tensors into (bsz, num_heads, length, head_dim), then we (optionally) normalize the head
        # features. This is not the same as normalizing the query or key features first, then reshaping.

        # Prepare queries
        queries = rearrange(_data[self.input_name_queries],
                            'batch length_queries (num_heads head_dim) -> batch num_heads length_queries head_dim',
                            num_heads=self.num_heads)
        if self.norm_query_heads_bool:
            queries = self.norm_query_heads(queries)

        # Prepare keys
        keys = rearrange(_data[self.input_name_keys],
                         'batch length_keys (num_heads head_dim) -> batch num_heads length_keys head_dim',
                         num_heads=self.num_heads)
        if self.norm_key_heads_bool:
            keys = self.norm_key_heads(keys)

        # Perform attention operation, and insert into _data
        _data[self.output_name] = einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale

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
        self.input_name_dots = set_default(_look='input_name_dots', _dict=config, _default='attn_dots')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.output_name_attn_scores = set_default(_look='output_name_attention_scores', _dict=config,
                                                   _default=False, _type=None)

        self.input_dim_values = _streams[self.input_name_values]
        self.output_dim = self.input_dim_values
        self.num_heads = _streams[self.input_name_dots]  # For dots, this will be num_heads i.e. (dots.shape[1])

        # Checking attention head settings (if num_heads is not given, default to 1)
        assert self.input_dim_values % self.num_heads == 0, "num_heads must divide evenly into input_dim!"
        self.head_dim = int(self.input_dim_values / self.num_heads)

        # Checking norm_value_heads settings
        self.norm_value_heads_bool, self.norm_value_heads = init_norm(_key='norm_value_heads',
                                                                      _config=config,
                                                                      _dim=self.head_dim)

        self.attention_type = set_default(_look='attn_function', _dict=config, _default='softmax')
        self.attn_function = get_attention_function(attn_type=self.attention_type)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_values, ['BSZ', 'CTX', self.input_dim_values]],
                                             [self.input_name_values, ['BSZ', self.num_heads, 'LEN', 'CTX']],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', 'LEN', self.output_dim]],
                                              ]
                                  }

        if self.output_name_attn_scores:
            self.streams_in_module['outputs'].append([self.output_name_attn_scores, self.num_heads, 'heads'])

    def forward(self, _data):
        # Attention operates on a set, so it must receive inputs of shape (bsz, set_length, dim)
        # We first reshape the values into (bsz, num_heads, length, head_dim), then we (optionally) normalize the head
        # features. This is not the same as normalizing the value features first, then reshaping.

        # Prepare values
        values = rearrange(_data[self.input_name_values],
                           'batch length_values (num_heads head_dim) -> batch num_heads length_values head_dim',
                           num_heads=self.num_heads)
        if self.norm_value_heads_bool:
            values = self.norm_value_heads(values)

        # Make attention map out of attention dots (usually just a softmax over the last dimension)
        attention_scores = self.attn_function(_data[self.input_name_dots], dim=-1)

        # Return attention_scores if configured (these can be re-used in deeper layers, see LazyFormer)
        if self.output_name_attn_scores:
            _data[self.output_name_attn_scores] = attention_scores

        # Weighted sum of value heads based on attn_map
        _data[self.output_name] = einsum('b h i j, b h j d -> b h i d', attention_scores, values)

        # Merge the heads back together so we have the same number of features as our input
        # The length is the length/number of queries (or the i in the above einsum)
        _data[self.output_name] = rearrange(_data[self.output_name],
                                            'batch num_heads length head_dim -> batch length (num_heads head_dim)')
        return _data