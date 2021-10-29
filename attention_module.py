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
                 ):
        super().__init__()
        """
        MHA (Multi-Head Attention) Dots (dot-products)
        """

        # Checking input_dim settings
        assert 'input_dim' in config, f"MHADots module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside MHADots module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']
        self.output_dim = self.input_dim  # Set output_dim equal to input_dim for now (this isn't really correct)

        # Checking attention head settings (if num_heads is not given, default to 1)
        self.num_heads = set_default(_key='num_heads', _dict=config, _default=1, _type=int)
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

        self.input_name_queries = set_default(_key='input_name_queries', _dict=config, _default='queries')
        self.input_name_keys = set_default(_key='input_name_keys', _dict=config, _default='keys')
        self.output_name = set_default(_key='output_name', _dict=config, _default='attn_dots')

    def forward(self, _data):
        # Attention operates on a set, so it must receive inputs of shape (bsz, seq_len, dim)
        # We first reshape the tensors into (bsz, num_heads, length, head_dim), then we (optionally) normalize the head
        # features. This is not the same as normalizing the query or key features first, then reshaping.

        # Prepare queries
        queries = rearrange(_data[self.input_name_query],
                            'batch length_queries (num_heads head_dim) -> batch num_heads length_queries head_dim',
                            num_heads=self.num_heads)
        if self.norm_query_heads_bool:
            queries = self.norm_query_heads(queries)

        # Prepare keys
        keys = rearrange(_data[self.input_name_key],
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
                 ):
        super().__init__()
        """
        MHA (Multi-Head Attention) Weighted Summation
        """

        # Checking input_dim settings
        assert 'input_dim' in config, f"MHAWeightedSum module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside MHAWeightedSum module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']
        self.output_dim = self.input_dim  # Set output_dim equal to input_dim (this is not configurable)

        # Checking attention head settings (if num_heads is not given, default to 1)
        self.num_heads = set_default(_key='num_heads', _dict=config, _default=1, _type=int)
        assert self.input_dim % self.num_heads == 0, "num_heads must divide evenly into input_dim!"
        self.head_dim = int(self.input_dim / self.num_heads)

        # Checking norm_value_heads settings
        self.norm_value_heads_bool, self.norm_value_heads = init_norm(_key='norm_value_heads',
                                                                      _config=config,
                                                                      _dim=self.head_dim)

        # Configure input/output names
        self.input_name_values = set_default(_key='input_name_values', _dict=config, _default='values')
        self.input_name_dots = set_default(_key='input_name_dots', _dict=config, _default='attn_dots')
        self.output_name = set_default(_key='output_name', _dict=config, _default='x')
        self.output_name_attn_scores = set_default(_key='output_name_attention_scores', _dict=config,
                                                   _default=False, _type=None)

        self.attention_type = set_default(_key='attn_function', _dict=config, _default='softmax')
        self.attn_function = get_attention_function(attn_type=self.attention_type)

    def forward(self, _data):
        # Attention operates on a set, so it must receive inputs of shape (bsz, seq_len, dim)
        # We first reshape the values into (bsz, num_heads, length, head_dim), then we (optionally) normalize the head
        # features. This is not the same as normalizing the value features first, then reshaping.

        # Prepare values
        values = rearrange(_data[self.input_name_value],
                           'batch length_values (num_heads head_dim) -> batch num_heads length_values head_dim',
                           num_heads=self.num_heads)
        if self.norm_value_heads_bool:
            values = self.norm_value_heads(values)

        # Make attention map out of attention dots (usually just a softmax over the last dimension)
        attention_scores = self.attn_function(_data[self.input_name_dots], dim=-1)
        if self.output_name_attn_scores:
            _data[self.output_name_attn_scores] = attention_scores

        # Weighted sum of value heads based on attn_map
        _data[self.output_name] = einsum('b h i j, b h j d -> b h i d', attention_scores, values)

        # Merge the heads back together so we have the same number of features as our input
        # The length is the length/number of queries (or the i in the above einsum)
        _data[self.output_name] = rearrange(_data[self.output_name],
                                            'batch num_heads length head_dim -> batch length (num_heads head_dim)')
        return _data