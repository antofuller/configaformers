import torch
from torch import nn, einsum
from einops import rearrange
from utils import set_default


def get_rope(config):
    rope_dict = {}
    for i_block, _block in enumerate(config):
        block_config = _block['config']
        for i_mod, module_config in enumerate(block_config):
            if module_config['type'] == 'rope':
                assert 'rotate_dim' in module_config.keys(), f"RoPE module must be given rotate_dim"
                rotate_dim = module_config['rotate_dim']
                max_length = set_default(_look='max_length', _dict=module_config, _default=2048, _type=int)

                # Create RoPE frequencies
                inv_freq = 1. / (10000 ** (torch.arange(0, rotate_dim, 2).float() / rotate_dim))
                t = torch.arange(max_length).type_as(inv_freq)  # count up from 0 to (max_seq_len - 1)
                freqs = torch.einsum('i , j -> i j', t, inv_freq)  # multiply t with inv_freq, shape (max_seq_len, dim/2)
                RoPE_emb = torch.cat((freqs, freqs), dim=-1)  # repeat freqs once, shape (max_seq_len, dim)
                rope_dict[f'rope_{rotate_dim}'] = RoPE_emb

    return rope_dict


class RoPE(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Rotary Position Embedding (RoPE)
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='x')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        self.input_dim = _streams[self.input_name][-1]
        input_shape = _streams[self.input_name_queries]  # will be the same as output shape

        self.rotate_dim = config['rotate_dim']
        max_length = set_default(_look='max_length', _dict=config, _default=2048, _type=int)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, input_shape],
                                             [f'rope_{self.rotate_dim}', [max_length, self.rotate_dim]]
                                             ],

                                  'outputs': [[self.output_name, input_shape],
                                              ]
                                  }

    @staticmethod
    def rotate_half(x):
        x = rearrange(x, '... (j d) -> ... j d', j=2)  # split the features into two
        x1, x2 = x.unbind(dim=-2)  # separate them
        return torch.cat((-x2, x1), dim=-1)  # rotate one of them (multiplying by negative 1), return the concatenation

    def apply_rotary_pos_emb(self, x, frequencies):
        num_features = frequencies.shape[-1]  # The number of features we wish to rotate
        x_rotate = x[..., :num_features]  # Features to rotate
        x_orig = x[..., num_features:]  # Features to keep, as is

        seq_len = x_rotate.shape[-2]  # Length of the input
        frequencies = frequencies[:, :, -seq_len:]  # Take the frequencies we need (just up to seq_len)
        x_rotate = (x_rotate * frequencies.cos()) + (self.rotate_half(x_rotate) * frequencies.sin())  # Apply rotation

        x = torch.cat([x_rotate, x_orig], dim=-1)  # Piece back together
        return x

    def forward(self, _data):
        _data[self.output_name] = self.apply_rotary_pos_emb(x=_data[self.input_name],
                                                            frequencies=_data[f'rope_{self.rotate_dim}'])
        return _data



