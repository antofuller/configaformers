import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math
from typing import Optional, Tuple, Union, List, Dict
from block_builder import Block


class ConfigaModel(nn.Module):
    def __init__(self,
                 model_config,
                 ):
        super().__init__()
        # Type checking
        assert type(model_config['blocks']) == list, f"ConfigaModel's config should be a list, it was given a {type(model_config)}"

        x_shape = ['B', 'L_x', model_config['input_dim']]
        streams = {'x': x_shape}
        self.block_list = nn.ModuleList([])
        for i_block, block_config in enumerate(model_config['blocks']):
            block = Block(block_config=block_config, input_streams=streams)
            output_streams = block.streams
            streams = output_streams
            self.block_list.append(block)

    def forward(self, _data):

        for i_block, _block in enumerate(self.block_list):
            _data = _block(_data)

        return _data




