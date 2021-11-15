import torch
from torch import nn
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
        self.input_shape = _streams[self.input_name]

        assert 'output_name' in config.keys(), f"When making a stream, 'output_name' must be given!"
        self.output_name = config['output_name']

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, self.input_shape],
                                             ],

                                  'outputs': [[self.output_name, self.input_shape],
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
        Merge data streams via element-wise add, subtract, or multiply
        """
        # Configure input(s) and output(s)
        self.input_name_1 = set_default(_look='input_name_1', _dict=config, _default='x')
        self.input_name_2 = set_default(_look='input_name_2', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.merge_name = set_default(_look='merge_type', _dict=config, _default='add')

        self.input_shape_1 = _streams[self.input_name_1]
        self.input_shape_2 = _streams[self.input_name_2]

        assert (self.merge_name == 'add') or (self.merge_name == 'multiply') or (self.merge_name == 'subtract'), \
            f"Merge stream operations available are: 'add', 'multiply', and 'subtract'!"

        if len(self.input_shape_1) < len(self.input_shape_2):
            self.output_shape = self.input_shape_2
        else:
            self.output_shape = self.input_shape_1

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name_1, self.input_shape_1],
                                             [self.input_name_2, self.input_shape_2],
                                             ],

                                  'outputs': [[self.output_name, self.output_shape],
                                              ]
                                  }

    def forward(self, _data):
        if self.merge_name == 'add':
            _data[self.output_name] = _data[self.input_name_1] + _data[self.input_name_2]
        elif self.merge_name == 'subtract':
            _data[self.output_name] = _data[self.input_name_1] - _data[self.input_name_2]
        elif self.merge_name == 'multiply':
            _data[self.output_name] = _data[self.input_name_1] * _data[self.input_name_2]
        else:
            print(f'{self.merge_name} did not match any options.')
        return _data


class CutStream(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        IN TESTING
        Cut data stream
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        assert 'start' in config.keys(), f"Cut_stream must be given a starting index!"
        assert 'end' in config.keys(), f"Cut_stream must be given an ending index!"
        assert 'cut_dim' in config.keys(), f"Cut_stream must be given a dimension which is cut!"

        self.start = config['start']
        self.end = config['end']
        self.cut_dim = config['cut_dim']

        self.input_shape = _streams[self.input_name]

        if (type(self.start) == int) and (type(self.end) == int):
            cut_dim_output = self.end - self.start
        elif self.start == 0:
            cut_dim_output = self.end
        else:
            cut_dim_output = f"{self.end} - {self.start}"

        self.output_shape = self.input_shape.clone()
        self.output_shape[self.cut_dim] = cut_dim_output

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, self.input_shape],
                                             ],

                                  'outputs': [[self.output_name, self.output_shape],
                                              ]
                                  }

    def forward(self, _data):
        if type(self.start) == int:
            start_idx = self.start
        else:
            start_idx = _data['input_sizes'][self.start]

        if type(self.end) == int:
            end_idx = self.end
        else:
            end_idx = _data['input_sizes'][self.end]

        if self.cut_dim == 0:
            _data[self.output_name] = _data[self.input_name][start_idx:end_idx, ...]

        elif self.cut_dim == 1:
            _data[self.output_name] = _data[self.input_name][:, start_idx:end_idx, ...]

        elif self.cut_dim == 2:
            _data[self.output_name] = _data[self.input_name][:, :, start_idx:end_idx, ...]

        elif self.cut_dim == 3:
            _data[self.output_name] = _data[self.input_name][:, :, :, start_idx:end_idx]

        else:
            print('cut_stream only supports up to 4 dimensional data')

        return _data