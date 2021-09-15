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


def add_default_config(layer_config):
    if layer_config["type"] == "Attention":
        default_args = get_default_args(Attention)

        input_dict = {}
        for key in {**layer_config, **default_args}.keys():
            if key not in layer_config.keys():
                input_dict[key] = default_args[key]
            else:
                input_dict[key] = layer_config[key]

        return input_dict

    if layer_config["type"] == "FFN":
        default_args = get_default_args(FFN)

        input_dict = {}
        for key in {**layer_config, **default_args}.keys():
            if key not in layer_config.keys():
                input_dict[key] = default_args[key]
            else:
                input_dict[key] = layer_config[key]

        return input_dict


# def build_layer(layer_config, _dim_model):
#     if layer_config["type"] == "Attention":
#         exclude_keys = ['type']
#         input_dict = {k: layer_config[k] for k in set(list(layer_config.keys())) - set(exclude_keys)}
#
#         return Attention(dim=_dim_model,
#                          **input_dict)
#
#     if layer_config["type"] == "FFN":
#         exclude_keys = ['type']
#         input_dict = {k: layer_config[k] for k in set(list(layer_config.keys())) - set(exclude_keys)}
#
#         return FFN(dim=_dim_model,
#                    **input_dict)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Config
        self.config = config
        self.dim_model = config['dim_model']
        self.vocab_size = config['vocab_size']
        self.num_layers = len(config['layers'])
        self.dim_rope = config['dim_rope']

        for layer_id in range(self.num_layers):
            config['layers'][layer_id] = add_default_config(config['layers'][layer_id])

        self.layers = nn.ModuleList([])
        for layer_id in range(self.num_layers):
            layer_config = config['layers'][layer_id]
            if layer_config["type"] == "Attention":
                exclude_keys = ['type']
                input_dict = {k: layer_config[k] for k in set(list(layer_config.keys())) - set(exclude_keys)}

                self.layers.append(nn.ModuleList([
                    Attention(dim=self.dim_model,
                              **input_dict),
                ]))

            elif layer_config["type"] == "FFN":
                exclude_keys = ['type']
                input_dict = {k: layer_config[k] for k in set(list(layer_config.keys())) - set(exclude_keys)}

                self.layers.append(nn.ModuleList([
                    FFN(dim=self.dim_model,
                        **input_dict),
                ]))
            else:
                print(f"Layer type does not match any available types.")

        # Input utils
        self.token_emb = nn.Embedding(self.vocab_size, self.dim_model)
        self.rotary_pos_emb = RotaryEmbedding(self.dim_rope)

        self.logits_input_norm = nn.LayerNorm(self.dim_model)
        self.to_logits = nn.Linear(self.dim_model, self.vocab_size)

    def forward(self, seq_ids):
        bsz = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        x = self.token_emb(seq_ids).view(bsz, seq_len, self.dim_model)
        rotary_pos_emb_init = self.rotary_pos_emb(seq_len)
        attn_map = None
        dots = None

        for layer_id, layer_func in enumerate(self.layers):
            layer_config = self.config['layers'][layer_id]

            if layer_config['type'] == "Attention":
                if layer_config['rotate_qk_bool'] or layer_config['rotate_v_bool']:
                    rotary_pos_emb = rotary_pos_emb_init
                else:
                    rotary_pos_emb = None

                x, attn_map, dots = layer_func(x=x,
                                               previous_attn_map=attn_map,
                                               previous_attn_dots=dots,
                                               rotary_pos_emb=rotary_pos_emb,
                                               )

            elif layer_config['type'] == "FFN":
                x = layer_func(x=x)

        x = self.logits_input_norm(x)
        logits = self.to_logits(x).view(bsz, seq_len, self.vocab_size)
        return logits