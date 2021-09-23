import torch.nn as nn
from building_blocks import FFN, Attention, Classifier
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


def get_block(_type):
    if _type == "Attention":
        return Attention

    elif _type == "FFN":
        return FFN

    else:
        raise "Layer type does not match any available types."


def add_default_config(layer_config):
    block = get_block(_type=layer_config["type"])

    default_args = get_default_args(block)

    input_dict = {}
    for key in {**layer_config, **default_args}.keys():
        if key not in layer_config.keys():
            input_dict[key] = default_args[key]
        else:
            input_dict[key] = layer_config[key]

    return input_dict


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Model config
        self.config = config
        self.dim_model = config['dim_model']
        self.vocab_size = config['vocab_size']
        self.num_layers = len(config['layers'])

        if 'input_emb_size' not in config:
            self.input_emb_size = self.dim_model
            self.config['input_emb_size'] = self.input_emb_size
        else:
            self.input_emb_size = config['input_emb_size']

        for layer_id in range(self.num_layers):
            config['layers'][layer_id] = add_default_config(config['layers'][layer_id])

        self.layers = nn.ModuleList([])
        for layer_id in range(self.num_layers):
            layer_config = config['layers'][layer_id]
            block = get_block(_type=layer_config["type"])

            exclude_keys = ['type']
            input_dict = {k: layer_config[k] for k in set(list(layer_config.keys())) - set(exclude_keys)}

            self.layers.append(nn.ModuleList([
                block(dim=self.dim_model,
                      **input_dict),
            ]))

        self.classifiers = nn.ModuleList([])
        for classifier_idx in range(len(config['classifiers'])):
            self.classifiers.append(nn.ModuleList([
                Classifier(dim=self.dim_model,
                           num_classes=config['classifiers'][classifier_idx]),
            ]))

        # Input utils
        self.token_emb = nn.Embedding(self.vocab_size, self.input_emb_size)

        if self.input_emb_size != self.dim_model:
            self.proj_input = nn.Linear(self.input_emb_size, self.dim_model)

    def forward(self, seq_ids):
        bsz = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        x = self.token_emb(seq_ids).view(bsz, seq_len, self.input_emb_size)
        if self.input_emb_size != self.dim_model:
            # Only needed if the token embedding size is different from dim_model
            x = self.proj_input(x).view(bsz, seq_len, self.dim_model)

        attn_map = None  # No attention has occurred yet
        dots = None  # No attention has occurred yet

        for layer_id, layer_func in enumerate(self.layers):
            layer_config = self.config['layers'][layer_id]

            if layer_config['type'] == "Attention":
                x, attn_map, dots = layer_func[0](x=x,
                                                  previous_attn_map=attn_map,
                                                  previous_attn_dots=dots,
                                                  )

            elif layer_config['type'] == "FFN":
                x = layer_func[0](x=x)

        if len(self.config['classifiers']) > 0:
            # If we have classifiers, use them here
            output_list = []
            for classifier_fn in self.classifiers:
                output_list.append(classifier_fn[0](x))

            return output_list

        else:
            # If the model has no classifiers, output the final hidden states
            return x
