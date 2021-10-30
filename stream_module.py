import torch.nn.functional as F
import torch
from torch import nn
from norm_module import init_norm
from utils import set_default


class MakeStream(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()

        # Configuring names
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        assert 'output_name' in config.keys(), f"MakeStream must be given an output_name!"
        assert type(config['output_name']) == str, f"In MakeStream, output_name must be a string," \
                                                   f" but it is a {type(config['output_name'])}!"
        self.output_name = config['output_name']

        # Checking input_dim settings
        assert 'input_dim' in config, f"MakeStream module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside MakeStream module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']
        self.output_dim = self.input_dim

    def forward(self, _data):
        _data[self.output_name] = _data[self.input_name].clone()
        return _data


class MergeStreams(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()

        # Configuring names
        self.input_name_1 = set_default(_look='input_name_1', _dict=config, _default='x')
        self.input_name_2 = set_default(_look='input_name_2', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.merge_name = set_default(_look='merge_type', _dict=config, _default='add')

        # Checking input_dim settings
        assert 'input_dim' in config, f"MergeStreams module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside MergeStreams module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']

        if (self.merge_name == 'add') or (self.merge_name == 'mult'):
            self.output_dim = self.input_dim
        elif self.merge_name == 'cat':
            self.output_dim = int(2*self.input_dim)
        else:
            print(f'{self.merge_name} did not match any options.')

    def forward(self, _data):
        if self.merge_name == 'add':
            _data[self.output_name] = _data[self.input_name_1] + _data[self.input_name_2]
        elif self.merge_name == 'mult':
            _data[self.output_name] = _data[self.input_name_1] * _data[self.input_name_2]
        elif self.merge_name == 'cat':
            _data[self.output_name] = torch.cat([_data[self.input_name_1], _data[self.input_name_2]], dim=-1)
        else:
            print(f'{self.merge_name} did not match any options.')
        return _data
