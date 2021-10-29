import torch.nn.functional as F
import torch
from torch import nn
from typing import List
from norm_module import get_norm


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
                 ):
        super().__init__()

        # Checking input_dim settings
        assert 'input_dim' in config, f"Activation was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside Activation, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']

        # Checking input_norm settings
        if 'input_norm' not in config:
            self.input_norm_bool = False  # Defaults to False

        else:
            if type(config['input_norm']) == str:
                self.input_norm_bool = True
                self.input_norm = get_norm(norm_type=config['input_norm'], dim=self.input_dim)
            else:
                self.input_norm_bool = False

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
        if 'output_norm' not in config:
            self.output_norm_bool = False  # Defaults to False

        else:
            if type(config['output_norm']) == str:
                self.output_norm_bool = True
                self.output_norm = get_norm(norm_type=config['output_norm'], dim=self.output_dim)
            else:
                self.output_norm_bool = False

        self.input_name = config['input_name']

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
            _data[self.input_name] = self.input_norm(_data[self.input_name])

        if self.glu_bool:
            _data[self.input_name] = self._split_and_multiply(_data[self.input_name])
        else:
            _data[self.input_name] = self.act_func(_data[self.input_name])

        if self.output_norm_bool:
            _data[self.input_name] = self.output_norm(_data[self.input_name])

        return _data


