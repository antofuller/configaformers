import torch.nn.functional as F
import torch
from torch import nn
from typing import List
from norm_module import init_norm
from utils import set_default


def get_non_linearity(non_lin: str):
    non_lin = non_lin.lower()  # Make lowercase
    if non_lin == 'none':
        return nn.Identity()

    if non_lin == 'gelu':
        return nn.GELU()

    if non_lin == 'relu':
        return nn.ReLU()

    if non_lin == 'sigmoid':
        return nn.Sigmoid()

    if non_lin == 'tanh':
        return nn.Tanh()

    else:
        print(f"Non-linearity not available.")


class Activation(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Activation module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name]
        len_input = _streams[self.input_name][-2]

        # Checking input_norm settings
        self.input_norm_bool, self.input_norm = init_norm(_key='input_norm',
                                                          _config=config,
                                                          _dim=self.input_dim)

        # Checking activation settings
        if 'activation_function' not in config:
            config['activation_function'] = 'gelu'  # Defaults to gelu

        if type(config['activation_function']) is list:
            # Gated linear unit(s) (GLUs) will be configured
            self.glu_bool = True
            self.num_projections = len(config['activation_function'])

            assert self.num_projections <= 4, f"The max number of allowable activation gates is 4," \
                                              f" you entered {len(config['activation_function'])}."

            assert self.input_dim % self.num_projections == 0, f"The number of activation functions needs to" \
                                                               f" divide evenly into input_dim."

            self.output_dim = int(self.input_dim / self.num_projections)

            # Insert activation functions into a ModuleList
            self.gating_act_funcs = nn.ModuleList([])
            for act_string in config['activation_function']:
                assert type(act_string) == str, f'Activation function list needs to be strings,' \
                                                f' it was given {type(act_string)}.'

                self.gating_act_funcs.append(get_non_linearity(non_lin=act_string))

        elif type(config['activation_function']) is str:
            # No GLUs
            self.glu_bool = False
            self.act_func = get_non_linearity(non_lin=config['activation_function'])
            self.output_dim = self.input_dim

        else:
            print(f"activation_function must be a string or a list of strings.")

        # Checking output_norm settings
        self.output_norm_bool, self.output_norm = init_norm(_key='output_norm',
                                                            _config=config,
                                                            _dim=self.output_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input, self.output_dim]],
                                              ]
                                  }

    def _split_and_multiply(self, _x):
        # Split into chunks of equal shape along the feature/last dimension
        _x = _x.chunk(self.num_projections, dim=-1)

        # First, apply the configured non-linearity to each chunk, then multiply the chunks by each other (element-wise)
        if self.num_projections == 2:
            _x = self.gating_act_funcs[0](_x[0]) * self.gating_act_funcs[1](_x[1])

        elif self.num_projections == 3:
            _x = self.gating_act_funcs[0](_x[0]) * self.gating_act_funcs[1](_x[1]) * self.gating_act_funcs[2](_x[2])

        elif self.num_projections == 4:
            _x = self.gating_act_funcs[0](_x[0]) * self.gating_act_funcs[1](_x[1]) * self.gating_act_funcs[2](_x[2])\
                 * self.gating_act_funcs[3](_x[3])

        else:
            raise "self.num_projections out of range, inside of forward pass"

        return _x

    def forward(self, _data):
        if self.input_norm_bool:
            _data[self.output_name] = self.input_norm(_data[self.input_name])
        elif self.output_name != self.input_name:
            _data[self.output_name] = _data[self.input_name].clone()

        if self.glu_bool:
            _data[self.output_name] = self._split_and_multiply(_data[self.output_name])
        else:
            _data[self.output_name] = self.act_func(_data[self.output_name])

        if self.output_norm_bool:
            _data[self.output_name] = self.output_norm(_data[self.output_name])

        return _data


