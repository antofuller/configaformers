import torch
from torch import nn
from utils import set_default


class Dropout(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Norm module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]
        self.output_dim = self.input_dim

        # Configuring norm
        assert 'prob' in config.keys(), f"Dropout module must be given prob (probability of dropout)!"
        assert type(config['prob']) == float, f"Dropout prob must be a float, but was given a {type(config['prob'])}!"
        assert 1 > config['prob'] >= 0, f'Dropout prob was given {config["prob"]} but should be less than 1, and ' \
                                        f'greater than or equal to 0.'
        self.drop = nn.Dropout(config['prob'])
        batch = _streams[self.input_name][0]

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, [batch, len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, [batch, len_input, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = self.drop(_data[self.input_name])
        return _data