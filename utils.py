import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math


def set_default(_key,
                _dict,
                _default,
                _type=str,
                ):
    if _key in _dict.keys():
        out = _dict[_key]
    else:
        out = _default

    if _type:
        assert type(out) == _type, f"{out} is type {type(out)}, but should be type {_type}"
    return out