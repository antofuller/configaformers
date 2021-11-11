import torch
from torch import nn
from attention_module import MHADots, MHAWeightedSum
from linear_module import LinearProj
from activation_module import Activation
from norm_module import Norm, ScaleAlongDimension
from stream_module import MakeStream, MergeStreams, CutSequence
from embedding_module import Embedding
from attention_offset_module import AttentionOffset
from rope_module import RoPE
from rearranging_module import MakeHeads, MergeHeads, ShiftSequence, DownSampleSequence, UpSampleSequence
from dropout_module import Dropout


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

    elif module_type == 'attention_offset':
        return AttentionOffset

    elif module_type == 'rope':
        return RoPE

    elif module_type == 'make_heads':
        return MakeHeads

    elif module_type == 'merge_heads':
        return MergeHeads

    elif module_type == 'shift':
        return ShiftSequence

    elif module_type == 'down_sample':
        return DownSampleSequence

    elif module_type == 'up_sample':
        return UpSampleSequence

    elif module_type == 'dropout':
        return Dropout

    elif module_type == 'cut_sequence':
        return CutSequence

    elif module_type == 'scale_dim':
        return ScaleAlongDimension

    else:
        raise "Layer type does not match any available types."


def process_shape(_shape):
    _string = "("  # start string
    num_dims = len(_shape)
    for i in range(num_dims):
        _string += f"{_shape[i]}, "
    return _string[:-2] + ")"


class Block(nn.Module):
    def __init__(self,
                 block_config,
                 input_streams,
                 print_streams,
                 ):
        super().__init__()
        # Type checking
        assert type(block_config) == list, f"Block's config should be a list, it was given a {type(block_config)}"
        for module_config in block_config:
            assert type(module_config) == dict, f"Block's config should be a list of dicts, it was given a" \
                                                f" {type(module_config)}, inside the list."

        self.streams = input_streams
        self.module_list = nn.ModuleList([])
        for i_mod, module_config in enumerate(block_config):
            assert 'type' in module_config.keys(), f'Module not given a type'
            assert type('type') == str,  f"Module's type needs to be a string."

            _module = get_module(module_type=module_config['type'])
            _module = _module(module_config, _streams=self.streams)

            # Update and print used streams
            module_inputs = _module.streams_in_module['inputs']
            module_outputs = _module.streams_in_module['outputs']

            string_to_print = f"{module_config['type']} -> Input(s):"
            for mod_input in module_inputs:
                _shape = mod_input[1]
                _name = mod_input[0]

                string_to_add = f" {_name} {process_shape(_shape)},"
                string_to_print += string_to_add

            string_to_print = string_to_print[:-1] + f" - Output(s):"
            for mod_output in module_outputs:
                _shape = mod_output[1]
                _name = mod_output[0]

                string_to_add = f" {_name} {process_shape(_shape)},"
                string_to_print += string_to_add

                # Update stream
                self.streams[_name] = _shape

            string_to_print = string_to_print[:-1]
            if print_streams:
                print(string_to_print)

            self.module_list.append(_module)

    def forward(self, _data):

        for i_mod, _module in enumerate(self.module_list):
            _data = _module(_data)

        return _data




