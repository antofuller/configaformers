import torch.nn.functional as F
import torch
from torch import nn
from utils import set_default


class LinearProj(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Linear module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_shape = _streams[self.input_name]
        self.input_dim = self.input_shape[-1]
        if 'output_dim' in config:
            assert type(config['output_dim']) == int, f"Inside linear module, output_dim is a" \
                                                  f" {type(config['output_dim'])}, it needs to be an integer!"
            self.output_dim = config['output_dim']
        else:
            self.output_dim = self.input_dim

        # Set output shape as the same as input shape, other than the features dimension
        self.output_shape = self.input_shape.copy()
        self.output_shape[-1] = self.output_dim

        # Init linear projection
        self.proj = nn.Linear(self.input_dim, self.output_dim)

        if 'init_bias' in config.keys():
            self.proj.bias.data.fill_(config['init_bias'])
            print(f'Bias initialized to {config["init_bias"]}')

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, self.input_shape],
                                             ],

                                  'outputs': [[self.output_name, self.output_shape],
                                              ]
                                  }

    def forward(self, _data):
        if self.output_name != self.input_name:
            _data[self.output_name] = _data[self.input_name].clone()

        _data[self.output_name] = self.proj(_data[self.output_name])

        return _data
