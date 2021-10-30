import torch.nn.functional as F
import torch
from torch import nn
from norm_module import init_norm
from utils import set_default


class Embedding(nn.Module):
    def __init__(self,
                 config,
                 _streams,
                 ):
        super().__init__()
        """
        Embedding module
        """
        # Configure input(s) and output(s)
        self.input_name = set_default(_look='input_name', _dict=config, _default='emb_ids')
        self.output_name = set_default(_look='output_name', _dict=config, _default='x')

        # In order to make an embedding module, we need the number of classes (in NLP this would be the vocab size)
        # and the embedding dimension (output_dim)
        assert 'num_classes' in config.keys(), f"Embedding module must be given num_classes"
        assert type(config['num_classes']) == int, f"Inside embedding module, num_classes is a" \
                                              f" {type(config['output_dim'])}, it needs to be an integer!"
        self.num_classes = config['num_classes']

        assert 'output_dim' in config.keys(), f"Embedding module must be given an output_dim"
        assert type(config['output_dim']) == int, f"Inside embedding module, output_dim is a" \
                                              f" {type(config['output_dim'])}, it needs to be an integer!"
        self.output_dim = config['output_dim']

        self.embedding = nn.Embedding(self.num_classes, self.output_dim)

        # Prepare streams info
        self.streams_in_module = {'inputs': [[self.input_name, ['BSZ', 'LEN']],
                                             ],

                                  'outputs': [[self.output_name, ['BSZ', 'LEN', self.output_dim]],
                                              ]
                                  }

    def forward(self, _data):
        _data[self.output_name] = self.embedding(_data[self.input_name])
        return _data
