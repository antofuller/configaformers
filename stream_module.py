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
        """
        Make data stream
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]

        assert 'output_name' in config.keys(), f"When making a stream, 'output_name' must be given!"
        self.output_name = config['output_name']
        self.output_dim = self.input_dim  # dims must match, since we are just making a copy

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = _data[self.input_name].clone()
        return _data


class MergeStreams(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Merge data streams - can only merge 1 streams right now
        """
        # Configure input(s) and output(s)
        self.input_name_1 = set_default(_look='input_name_1', _dict=config, _default='x')
        self.input_name_2 = set_default(_look='input_name_2', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.merge_name = set_default(_look='merge_type', _dict=config, _default='add')

        self.input_dim_1 = _streams[self.input_name_1][-1]
        len_input_1 = _streams[self.input_name_1][-2]
        self.input_dim_2 = _streams[self.input_name_2][-1]
        len_input_2 = _streams[self.input_name_2][-2]

        assert len_input_1 == len_input_2, f"Merging streams must have the same length!"

        if (self.merge_name == 'add') or (self.merge_name == 'mult'):
            assert self.input_dim_1 == self.input_dim_2, f"When merging streams with 'add' or 'mult', the two input" \
                                                         f" streams must have the same number of features."
            self.output_dim = self.input_dim_1
        elif self.merge_name == 'cat':
            self.output_dim = self.input_dim_1 + self.input_dim_2
        else:
            print(f'{self.merge_name} did not match any options.')

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_1, ['BSZ', len_input_1, self.input_dim_1]],
                                             [self.input_name_2, ['BSZ', len_input_2, self.input_dim_2]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input_1, self.output_dim]],
                                              ]
                                  }

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


class CutSequence(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Cut data stream
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.start = set_default(_look='start', _dict=config, _default=0)
        self.end = set_default(_look='end', _dict=config, _default=1)

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]

        if (type(self.start) == int) and (type(self.end) == int):
            len_output = self.end - self.start
        elif self.start == 0:
            len_output = self.end
        else:
            len_output = f"{self.end} - {self.start}"

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_output, self.input_dim]],
                                              ]
                                  }

    def forward(self, _data):
        if type(self.start) == int:
            start_idx = self.start
        else:
            start_idx = _data[self.start]

        if type(self.end) == int:
            end_idx = self.end
        else:
            end_idx = _data[self.end]

        _data[self.output_name] = _data[self.input_name][:, start_idx:end_idx, :]
        return _data