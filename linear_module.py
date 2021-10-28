import torch.nn.functional as F
import torch
from torch import nn
from norm_module import get_norm


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

        # Checking input_norm settings
        if 'input_norm' not in config:
            self.input_norm_bool = False  # Defaults to False

        else:
            if type(config['input_norm']) == str:
                self.input_norm_bool = True
                self.input_norm = get_norm(norm_type=config['input_norm'], dim=self.input_dim)
            else:
                self.input_norm_bool = False

        # Checking output_norm settings
        if 'output_norm' not in config:
            self.output_norm_bool = False  # Defaults to False

        else:
            if type(config['output_norm']) == str:
                self.output_norm_bool = True
                self.output_norm = get_norm(norm_type=config['output_norm'], dim=self.output_dim)
            else:
                self.output_norm_bool = False

        self.proj = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, _x):
        if self.input_norm_bool:
            _x = self.input_norm(_x)

        _x = self.proj(_x)

        if self.output_norm_bool:
            _x = self.output_norm(_x)

        return _x
