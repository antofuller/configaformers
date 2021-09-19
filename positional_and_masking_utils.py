import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


patch_typeguard()


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


"""
Positional utils
"""


class AttentionBiasMask(nn.Module):
    def __init__(self, config_heads, num_heads, max_length, mask_precision="full"):
        super().__init__()
        self.num_heads = num_heads
        self.config_heads = config_heads
        assert num_heads == len(
            self.config_heads), f"You configured {len(self.config_heads)} attention head biases/masks, but there are {num_heads} heads."

        # Build bidirectional linear biases template
        template = []
        for i in range(max_length):
            template.append(torch.LongTensor([x for x in range(0 - i, max_length - i)]).view(1, -1).abs())
        template = -torch.cat(template, dim=0)

        """
        If max_length is 5, it would look like: 
        [ 0, -1, -2, -3, -4, -5],
        [-1,  0, -1, -2, -3, -4],
        [-2, -1,  0, -1, -2, -3],
        [-3, -2, -1,  0, -1, -2],
        [-4, -3, -2, -1,  0, -1],
        [-5, -4, -3, -2, -1,  0]
        _back (from below) refers to the biases in the bottom-left triangle, and _fwd refers to the upper-right triangle
        """

        # Build fwd and back masks that will be used to manipulate the template
        fwd_mask = torch.ones_like(template).bool()
        fwd_mask = ~fwd_mask.tril()

        """
        fwd_mask would look like this:
        [False,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True],
        [False, False, False, False,  True,  True],
        [False, False, False, False, False,  True],
        [False, False, False, False, False, False]
        """

        back_mask = torch.ones_like(template).bool()
        back_mask = ~back_mask.triu()

        """
        back_mask would look like this:
        [False, False, False, False, False, False],
        [ True, False, False, False, False, False],
        [ True,  True, False, False, False, False],
        [ True,  True,  True, False, False, False],
        [ True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True, False]
        """

        if (mask_precision == "full") or (mask_precision == "Full"):
            dummy_tensor = torch.Tensor([1.0]).float()
            mask_value = max_neg_value(dummy_tensor)
            template = template.float()
        elif (mask_precision == "half") or (mask_precision == "Half"):
            dummy_tensor = torch.Tensor([1.0]).half()
            mask_value = max_neg_value(dummy_tensor)
            template = template.half()
        else:
            print(f"{mask_precision} must be 'full' or 'half'")

        slopes = self._get_slopes(heads=num_heads)  # Get head slopes

        all_heads = []
        for i_head, head_config in enumerate(self.config_heads):
            _back, _fwd = head_config  # The bias/mask setting looking forward, or back (at each token)

            head_bias = template.clone()

            # First, adjust _back biases based on this head_config
            if (_back == "none") or (_back == "None"):
                # Zero out all backward biases
                head_bias = head_bias.triu()
            elif type(_back) == float:
                # Add an offset to all backward biases
                back_offset = back_mask * _back
                head_bias += back_offset
            elif (_back == "linear") or (_back == "Linear"):
                pass  # Keep standard/linear backward biases

            # Now, adjust _fwd biases based on this head_config
            if (_fwd == "none") or (_fwd == "None"):
                # Zero out all forward biases
                head_bias = head_bias.tril()
            elif type(_fwd) == float:
                # Add an offset to all forward biases
                fwd_offset = fwd_mask * _fwd
                head_bias += fwd_offset
            elif (_fwd == "linear") or (_fwd == "Linear"):
                pass  # Keep standard/linear forward biases

            head_bias *= slopes[i_head]  # Apply slope *before* masking

            if (_fwd == "mask") or (_fwd == "Mask"):
                # Mask all forward positions (causal mask)
                head_bias.masked_fill_(fwd_mask, mask_value)
            if (_back == "mask") or (_back == "Mask"):
                # Mask all backward positions (anti-causal mask)
                head_bias.masked_fill_(back_mask, mask_value)

            if ((_fwd == "mask") or (_fwd == "Mask")) and ((_back == "mask") or (_back == "Mask")):
                print(f"YOU ARE MASKING OUT BOTH FORWARD AND BACKWARD DIRECTIONS!!!")

            all_heads.append(head_bias.view(1, max_length, max_length))

        all_heads = torch.cat(all_heads, dim=0)
        self.attention_bias_mask = all_heads.cuda()

    @staticmethod
    def _get_slopes(heads):
        # This implementation is taken 100% from lucidrains
        # Explanation forthcoming
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                                                           :heads - closest_power_of_2]

    def forward(self, qk_dots):
        b, h, i, j = qk_dots.shape

        bias = repeat(self.attention_bias_mask, 'h i j -> b h i j', b=b)  # repeat over the batch dimension
        bias = bias[:, :, :i, :j].view(qk_dots.shape)  # trim the bias tensor such that it matches the shape of qk_dots

        return qk_dots + bias  # this adds them together, creating the relative positional bias


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        """
        This rotary embedding (RoPE) implementation is taken 100% from lucidrains. Explanation forthcoming.
        :param dim: number of features to rotate
        """
        super().__init__()

        # Create a frequency for every second dimension, start with f = 1.0, divide by 10 to get the next, and repeat
        # For example, if dim = 8, inv_freq = [1.0, 0.1, 0.01, 0.001]
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq).cuda()  # count up from 0 to (max_seq_len - 1)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)  # multiply t with inv_freq, shape (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # repeat freqs once, shape (max_seq_len, dim)
        return rearrange(emb, 'n d -> () () n d')


@typechecked
def rotate_half(x: TensorType["batch", "num_heads", "length", "dim"]) \
        -> TensorType["batch", "num_heads", "length", "dim"]:

    x = rearrange(x, '... (j d) -> ... j d', j=2)  # split the features into two
    x1, x2 = x.unbind(dim=-2)  # separate them
    return torch.cat((-x2, x1), dim=-1)  # rotate one of them (multiplying by negative 1), return the concatenation


@typechecked
def apply_rotary_pos_emb(x: TensorType["batch", "num_heads", "length", "dim"],
                         frequencies: TensorType[1, 1, "max_length", "rope_dim"]) \
        -> TensorType["batch", "num_heads", "length", "dim"]:

    num_features = frequencies.shape[-1]  # The number of features we wish to rotate
    x_rotate = x[..., :num_features]  # Features to rotate
    x_orig = x[..., num_features:]  # Features to keep, as is

    seq_len = x_rotate.shape[-2]  # Length of the input
    frequencies = frequencies[:, :, -seq_len:]  # Take the frequencies we need (just up to seq_len)
    x_rotate = (x_rotate * frequencies.cos()) + (rotate_half(x_rotate) * frequencies.sin())  # Apply rotation

    x = torch.cat([x_rotate, x_orig], dim=-1)  # Piece back together
    return x


# def old_apply_rotary_pos_emb(t, freqs):
#     # This implementation is taken 100% from lucidrains
#     # Explanation forthcoming
#     seq_len = t.shape[-2]
#     freqs = freqs[:, :, -seq_len:]
#     return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


"""
Masking utils
"""


# class AttentionMask(nn.Module):
#     def __init__(self,
#                  max_length,
#                  mask_type,
#                  ):
#
#         super().__init__()
#
#         all_mask_types = ["decoder"]
#
#         assert mask_type in all_mask_types, f"mask_type must be one of {all_mask_types}"
#
#         self.mask_type = mask_type
#
#         if self.mask_type == "decoder":
#             # Create a causal mask with zeros in the lower triangle, and negative infinity in the upper triangle
#             self.mask_base = torch.ones(max_length, max_length).triu(diagonal=1)
#             dummy_tensor = torch.Tensor([1.0])
#             mask_value = max_neg_value(dummy_tensor)
#             self.mask_base = self.mask_base * mask_value
#
#
#
#
#     def forward(self, qk_dots):
#         b, h, i, j = qk_dots.shape
#
#         bias = repeat(self.bias, 'h i j -> b h i j', b=b)  # repeat over the batch dimension
#         bias = bias[:, :, :i, :j].view(qk_dots.shape)  # trim the bias tensor such that it matches the shape of qk_dots
#
#         return qk_dots + bias  # this adds them together, creating the relative positional bias