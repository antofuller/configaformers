import torch.nn as nn
from building_blocks import FFN, Attention
from positional_and_masking_utils import RotaryEmbedding
import torch
import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


"""
*** Basic implementation to test building blocks ***
"""


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def build_layer(layer_config, _dim_model):
    if layer_config["type"] == "Attention":
        default_args = get_default_args(Attention)

        input_dict = {}
        for key in default_args:
            if key == "type":
                continue
            if key not in layer_config.keys():
                input_dict[key] = default_args[key]
            else:
                input_dict[key] = layer_config[key]

        print(input_dict)
        return Attention(dim=_dim_model,
                         **input_dict,
                         )


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Config
        self.dim_model = config['dim_model']
        self.vocab_size = config['vocab_size']
        self.num_layers = len(config['layers'])
        self.dim_rope = config['dim_rope']

        self.layers = nn.ModuleList([])
        for layer_id in range(self.num_layers):
            self.layers.append(nn.ModuleList([
                build_layer(layer_config=config['layers'][layer_id], _dim_model=self.dim_model),
            ]))
        # for layer_id in range(self.num_layers):
        #     self.layers.append(nn.ModuleList([
        #         Attention(dim=dim,
        #                   attn_dim=dim,
        #                   num_heads=heads,
        #                   ),
        #         FFN(dim,
        #             ff_mult=4),
        #     ]))

        # Input utils
        self.token_emb = nn.Embedding(self.vocab_size, self.dim_model)
        self.rotary_pos_emb = RotaryEmbedding(self.dim_rope)

        self.logits_input_norm = nn.LayerNorm(self.dim_model)
        self.to_logits = nn.Linear(self.dim_model, self.vocab_size)

    def forward(self, seq_ids):
        bsz = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        x = self.token_emb(seq_ids).view(bsz, seq_len, self.dim_model)
        rotary_pos_emb = self.rotary_pos_emb(seq_len)

        for self_attn, self_ff in self.layers:
            x, _, _ = self_attn(x=x, rotary_pos_emb=rotary_pos_emb)
            x = self_ff(x=x)

        x = self.logits_input_norm(x)
        logits = self.to_logits(x).view(bsz, seq_len, self.vocab_size)
        return logits