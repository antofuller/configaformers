
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
