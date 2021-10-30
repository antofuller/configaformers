import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math
from typing import Optional, Tuple, Union, List, Dict
from attention_module import MHADots, MHAWeightedSum
from linear_module import LinearProj
from activation_module import Activation
from norm_module import Norm
from stream_module import MakeStream, MergeStreams
from embedding_module import Embedding


def get_module(module_type):
    module_type = module_type.lower()  # Make lowercase
    if module_type == "activation":
        return Activation

    elif module_type == "linear":
        return LinearProj

    elif module_type == "mha_dots":
        return MHADots

    elif module_type == "mha_sum":
        return MHAWeightedSum

    elif module_type == "norm":
        return Norm

    elif module_type == "make_stream":
        return MakeStream

    elif module_type == "merge_streams":
        return MergeStreams

    elif module_type == "embedding":
        return Embedding

    else:
        raise "Layer type does not match any available types."


class Block(nn.Module):
    def __init__(self,
                 block_config,
                 ):
        super().__init__()
        # Type checking
        assert type(block_config['modules']) == list, f"Block's config should be a list, it was given a {type(block_config)}"
        for module_config in block_config['modules']:
            assert type(module_config) == dict, f"Block's config should be a list of dicts, it was given a" \
                                                f" {type(module_config)}, inside the list."

        streams = {'x': block_config['input_dim']}
        self.module_list = nn.ModuleList([])
        for i_mod, module_config in enumerate(block_config['modules']):
            assert 'type' in module_config.keys(), f'Module not given a type'
            assert type('type') == str,  f"Module's type needs to be a string."

            # if 'input_key' not in module_config.keys():
            #     module_config['input_key'] = 'x'  # Defaults to receive x as input to the module

            _module = get_module(module_type=module_config['type'])
            _module = _module(module_config, _streams=streams)

            # Update and print used streams
            module_inputs = _module.streams_in_module['inputs']
            module_outputs = _module.streams_in_module['outputs']

            string_to_print = f"{module_config['type']}-> Input(s):"
            for mod_input in module_inputs:
                string_to_add = f" {mod_input[0]} ({mod_input[1]} {mod_input[2]}) -"
                string_to_print += string_to_add

            string_to_print += f" Output(s):"
            for mod_output in module_outputs:
                string_to_add = f" {mod_output[0]} ({mod_output[1]} {mod_output[2]}) -"
                string_to_print += string_to_add

                # Update streams
                streams[mod_output[0]] = mod_output[1]

            print(string_to_print)

            self.module_list.append(_module)

    def forward(self, _data):

        for i_mod, _module in enumerate(self.module_list):
            _data = _module(_data)

        return _data




