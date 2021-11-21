import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
from utils import set_default, exists


class MakeHeads(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Make heads by rearranging
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]

        # Checking attention head settings (if num_heads is not given, default to 1)
        self.num_heads = set_default(_look='num_heads', _dict=config, _default=1, _type=int)
        assert self.input_dim % self.num_heads == 0, "num_heads must divide evenly into input_dim!"
        self.head_dim = int(self.input_dim / self.num_heads)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, [_streams[self.input_name][0], len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, [_streams[self.input_name][0], self.num_heads, len_input, self.head_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name],
                                            'batch length (num_heads head_dim) -> batch num_heads length head_dim',
                                            num_heads=self.num_heads)

        return _data


class MergeHeads(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Merge heads by rearranging
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        self.num_heads = _streams[self.input_name][1]
        len_input = _streams[self.input_name][-2]

        self.output_dim = int(self.num_heads*self.input_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, [_streams[self.input_name][0], self.num_heads, len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, [_streams[self.input_name][0], len_input, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name],
                                            'batch num_heads length head_dim -> batch length (num_heads head_dim)',
                                            num_heads=self.num_heads)

        return _data


class ShiftSequence(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Shifting representations along the sequence dimension, using a concatenation
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]

        # Count up the total number of features concatenated in order to set the output dim
        assert 'instructions' in config.keys(), f"SequenceShift (shift) must be given instructions."
        self.instructions = config['instructions']
        total_features = 0
        for pos_ins in self.instructions:
            start_idx, end_idx = pos_ins['features']
            total_features += (end_idx - start_idx)

        self.output_dim = total_features

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input, self.output_dim]],
                                              ]
                                  }

    @staticmethod
    def shift(t, amount, mask=None):
        # Only tested on tensors of shape (bsz, seq_len, dim)
        if amount == 0:  # If the amount of time-steps is 0, then we just return the input tensor
            return t

        if exists(mask):  # Set masked values to zero
            t = t.masked_fill(~mask[..., None], 0.0)

        # This pad operator shifts the features in the sequence (or time) dimension by the amount given, and fills in
        # the start or end of the sequence with zeros
        return F.pad(t, (0, 0, -amount, amount), value=0.0)  # changed negative signs here, testing it now

    def forward(self, _data):
        splitted = []
        for pos_ins in self.instructions:
            start_idx, end_idx = pos_ins['features']
            shift_amt = pos_ins['relative_pos']  # Number of sequence positions to shift by

            chunk = _data[self.input_name][..., start_idx:end_idx]  # Select features and remove them from input tensor
            chunk = self.shift(chunk, shift_amt)  # Perform the shift operation
            splitted.append(chunk)  # Store them in a list

        # Piece the slices back together (with some of the chunks shifted)
        _data[self.output_name] = torch.cat(splitted, dim=-1)

        return _data


class DownSampleSequence(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Down-sampling representations along the sequence dimension
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        assert 'stride' in config.keys(), f"Down-sample module needs to be given a stride!"
        assert type(config['stride']) == int, f"Down-sample module needs to be given a stride that is an integer, " \
                                              f" it was given a {type(config['stride'])}!"

        self.stride = config['stride']
        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]
        len_output = 'L_DS'

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_output, self.input_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = _data[self.input_name][:, 0::self.stride, :]
        return _data


class UpSampleSequence(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Up-sampling representations along the sequence dimension
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        input_shape = _streams[self.input_name]

        assert len(input_shape) == 3, f"Input shape to up-sample must be (BSZ, length, dim)!"

        assert 'num_chunks' in config.keys(), f"Up-sample module needs to be given num_chunks!"
        assert type(config['num_chunks']) == int, f"Up-sample module needs to be given num_chunks that is an integer, " \
                                              f" it was given a {type(config['num_chunks'])}!"

        self.num_chunks = config['num_chunks']
        self.input_dim = _streams[self.input_name][-1]
        len_input = _streams[self.input_name][-2]
        len_output = 'L_US'

        assert (self.input_dim % self.num_chunks) == 0, f'When up-sampling, input_dim ({self.input_dim}) needs to be ' \
                                                        f'evenly divisible by num_chunks ({self.num_chunks})!'
        self.output_dim = int(self.input_dim/self.num_chunks)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_output, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _x = rearrange(_data[self.input_name], 'b l d -> b l () d')
        _x = torch.cat(_x.chunk(self.num_chunks, dim=-1), dim=2)
        _data[self.output_name] = rearrange(_x, 'b l c d -> b (l c) d')

        return _data


class PackBatch(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        WORK IN PROGRESS
        Pack batch by rearranging
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')
        self.group_size = set_default(_look='group_size', _dict=config, _default=10, _type=int)

        input_dim = _streams[self.input_name][-1]
        input_length = _streams[self.input_name][-2]
        output_dim = input_dim
        output_length = int(input_length/self.group_size)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', input_length, input_dim]],
                                             ],

                                  'outputs': [[self.output_name, [f'{self.group_size}BSZ', output_length, output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name], 'b (l g) d -> (b l) g d', g=self.group_size)
        return _data


class UnPackBatch(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        WORK IN PROGRESS (only works with BSZ=1)
        UnPack batch by rearranging
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        input_dim = _streams[self.input_name][-1]
        input_length = _streams[self.input_name][-2]
        output_dim = input_dim

        input_batch_multiplier = int(_streams[self.input_name][0].replace('BSZ', ''))
        output_length = input_length * input_batch_multiplier

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, [_streams[self.input_name][0], input_length, input_dim]],
                                             ],

                                  'outputs': [[self.output_name, [f'BSZ', output_length, output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name], '(b l) g d -> b (l g) d', b=1)
        return _data