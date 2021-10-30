import torch.nn.functional as F
import torch
from torch import nn
from norm_module import init_norm
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

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]
        if 'output_dim' in config:
            assert type(config['output_dim']) == int, f"Inside linear module, output_dim is a" \
                                                  f" {type(config['output_dim'])}, it needs to be an integer!"
            self.output_dim = config['output_dim']
        else:
            self.output_dim = self.input_dim

        # Configuring input_norm and output_norm
        self.input_norm_bool, self.input_norm = init_norm(_key='input_norm',
                                                          _config=config,
                                                          _dim=self.input_dim)
        self.output_norm_bool, self.output_norm = init_norm(_key='output_norm',
                                                            _config=config,
                                                            _dim=self.output_dim)

        self.proj = nn.Linear(self.input_dim, self.output_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        if self.input_norm_bool:
            _data[self.output_name] = self.input_norm(_data[self.input_name])
        elif self.output_name != self.input_name:
            _data[self.output_name] = _data[self.input_name].clone()

        _data[self.output_name] = self.proj(_data[self.output_name])

        if self.output_norm_bool:
            _data[self.output_name] = self.output_norm(_data[self.output_name])

        return _data
