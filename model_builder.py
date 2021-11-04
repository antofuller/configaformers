import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math
from typing import Optional, Tuple, Union, List, Dict
from block_builder import Block
from RoPE_module import get_rope


class ConfigaFormer(nn.Module):
    def __init__(self,
                 model_config,
                 input_streams,
                 ):
        super().__init__()
        # Type checking
        assert type(model_config['blocks']) == list, f"ConfigaFormer's config should be a list, it was given a {type(model_config)}"

        # create rope embeds
        self.rope_dict = get_rope(model_config['blocks'])

        streams = input_streams
        self.block_list = nn.ModuleList([])
        for i_block, _block in enumerate(model_config['blocks']):
            block_config = _block['config']
            N = _block['repeat']
            print(f"Block #{i_block+1}, {N}x")

            for n in range(N):
                if n == 0:
                    _print = True
                else:
                    _print = False

                block = Block(block_config=block_config, input_streams=streams, print_streams=_print)
                output_streams = block.streams
                streams = output_streams
                self.block_list.append(block)
            print("\n")

    def forward(self, _data):
        # if rope embeds exist, then add frequency embeddings to _data
        if self.rope_dict != {}:
            # Get input device
            for _input in _data:
                device = _data[_input].device
                break

            for rope_key in self.rope_dict.keys():
                _data[rope_key] = self.rope_dict[rope_key].to(device)

        for i_block, _block in enumerate(self.block_list):
            _data = _block(_data)

        return _data




