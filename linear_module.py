import torch.nn.functional as F
import torch
from torch import nn
from norm_module import init_norm
from utils import set_default


class LinearProj(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        # Checking input_dim settings
        assert 'input_dim' in config, f"Linear module was not given input_dim, it is needed!"
        assert type(config['input_dim']) == int, f"Inside linear module, input_dim is a {type(config['input_dim'])}," \
                                                 f" it needs to be an integer!"
        self.input_dim = config['input_dim']

        # Checking output_dim settings
        assert 'output_dim' in config, f"Linear module was not given output_dim, it is needed!"
        assert type(config['output_dim']) == int, f"Inside linear module, output_dim is a" \
                                                  f" {type(config['output_dim'])}, it needs to be an integer!"
        self.output_dim = config['output_dim']

        # Configuring input_norm and output_norm
        self.input_norm_bool, self.input_norm = init_norm(_key='input_norm',
                                                          _config=config,
                                                          _dim=self.input_dim)
        self.output_norm_bool, self.output_norm = init_norm(_key='output_norm',
                                                            _config=config,
                                                            _dim=self.output_dim)

        self.proj = nn.Linear(self.input_dim, self.output_dim)

        # Configuring names
        self.input_name = set_default(_key='input_name', _dict=config, _default='x')
        self.output_name = set_default(_key='output_name', _dict=config, _default='x')

    def forward(self, _data):
        if self.input_norm_bool:
            _data[self.output_name] = self.input_norm(_data[self.input_name])
        elif self.output_name != self.output_name:
            _data[self.output_name] = _data[self.input_name]

        _data[self.output_name] = self.proj(_data[self.output_name])

        if self.output_norm_bool:
            _data[self.output_name] = self.output_norm(_data[self.output_name])

        return _data
