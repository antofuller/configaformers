
def exists(val):
    return val is not None


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


x = TokenShift_k(x)
a = torch.zeros_like(x)
a[:, ::k, :] = Attention(ln_1(x[:, ::k, :]))
x = x + a
m = torch.zeros_like(x)
m[:, ::k, :] = FFN(ln_2(x[:, ::k, :]))
x = x + m