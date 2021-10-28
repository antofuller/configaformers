import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math

from typing import Optional, Tuple, Union, List, Dict
from linear_module import LinearProj
from activation_module import Activation


def get_block(_type):
    _type = _type.lower()  # Make lowercase
    if _type == "activation":
        return Activation

    elif _type == "linear":
        return LinearProj

    else:
        raise "Layer type does not match any available types."


class Block(nn.Module):
    def __init__(self,
                 block_config,
                 ):
        super().__init__()
        # Type checking
        assert type(block_config) == list, f"Block's config should be a list, it was given a {type(block_config)}"
        for module_config in block_config:
            assert type(module_config) == dict, f"Block's config should be a list of dicts, it was given a" \
                                                f" {type(module_config)}, inside the list."

        # Build block by iterating over modules
        self.module_list = nn.ModuleList([])
        for i_mod, module_config in enumerate(block_config):
            assert 'type' in module_config.keys(), f'Module not given a type'
            assert type('type') == str,  f"Module's type needs to be a string."

            block = get_block(_type=module_config['type'])
            block = block(module_config)

            # Check dimensions between modules
            input_dim = block.input_dim
            current_dim = block.output_dim

            if i_mod != 0:
                assert current_dim == input_dim, f'The output dim of the previous block (index: {i_mod-1})' \
                                                 f' is {current_dim}, but the input dim of this block (index: {i_mod})' \
                                                 f' is {input_dim}. They must match. '

            self.module_list.append(block)

    def forward(self, x):

        for i_mod, _module in enumerate(self.module_list):
            x = _module(x)

        return x



