import torch
import math


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def set_default(_look,
                _dict,
                _default,
                _type=str,
                ):
    if _look in _dict.keys():
        out = _dict[_look]
    else:
        out = _default

    if _type:
        assert type(out) == _type, f"{out} is type {type(out)}, but should be type {_type}"
    return out


def _get_slopes(heads):
    # This implementation is taken 100% from lucidrains
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]


def build_attention_offset(func,
                           query_inputs,
                           context_inputs):
    stacked_rows = []
    for i in query_inputs:
        row = []
        for j in context_inputs:
            offset = func(i, j)
            row.append(offset)
        stacked_rows.append(row)

    return torch.Tensor(stacked_rows).view(1, 1, len(query_inputs), len(context_inputs))


def get_alibi(num_heads,
              max_length=2048,
              slopes=None,
              mask='causal',
              mask_precision='full',
              ):
    if mask == 'causal':
        if mask_precision == 'half':
            dummy_tensor = torch.Tensor([1.0]).half()
            mask_value = max_neg_value(dummy_tensor)
        else:
            dummy_tensor = torch.Tensor([1.0]).float()
            mask_value = max_neg_value(dummy_tensor)

        def my_func(x1, x2):
            if x2 > x1:
                return mask_value
            else:
                return -abs(x1 - x2)
    else:
        def my_func(x1, x2):
            return -abs(x1 - x2)

    offset_template = build_attention_offset(func=my_func,
                                             query_inputs=range(max_length),
                                             context_inputs=range(max_length))

    if slopes is None:
        slopes = _get_slopes(heads=num_heads)

    offset_list = []
    for n in range(num_heads):
        offset_list.append(offset_template*slopes[n])

    return torch.cat(offset_list, dim=1)

