import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
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
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', self.num_heads, len_input, self.head_dim]],
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
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', self.num_heads, len_input, self.input_dim]],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', len_input, self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = rearrange(_data[self.input_name],
                                            'batch num_heads length head_dim -> batch length (num_heads head_dim)',
                                            num_heads=self.num_heads)

        return _data


class SequenceShift(nn.Module):
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
        for ins in self.instructions:
            total_features += self.instructions[ins]

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
            start_idx, end_idx = self.instructions[pos_ins]['features']
            shift_amt = self.instructions[pos_ins]['relative_pos']  # Number of sequence positions to shift by

            chunk = _data[self.input_name][..., start_idx:end_idx]  # Select features and remove them from input tensor
            chunk = self.shift(chunk, shift_amt)  # Perform the shift operation
            splitted.append(chunk)  # Store them in a list

        # Piece the slices back together (with some of the chunks shifted)
        _data[self.output_name] = torch.cat(splitted, dim=-1)

        return _data