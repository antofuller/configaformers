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


class Alibi(nn.Module):
    def __init__(self, heads, max_length):
        super().__init__()
        """
        This builds the matrix seen on the right side of Figure 3 - https://arxiv.org/pdf/2108.12409.pdf . It is a 
        relative position bias (since it biases the attention pattern based on the relative positions of the inputs). 
        Right now, I only have an encoder implementation of this matrix (which isn't mentioned in the paper). The causal
        version will be added shortly; it's actually just the lower triangle portion.

        From the paper, Alibi seems to perform on par with RoPE, but it can generalize to longer sequences not seen
        during training. RoPE cannot do that. 
        """

        self.heads = heads

        rows = []
        for i in range(max_length):  # build the full Alibi relative position bias matrix
            rows.append(torch.LongTensor([x for x in range(0 - i, max_length - i)]).view(1, -1).abs())

        # This implementation alternates between upper triangular and lower triangular biases. Using the full matrix
        # doesn't seem to work as well - likely since the forward and backward biases would be identical (i.e. attention
        # wouldn't be able to tell the difference between a token X spots after or X spots before). However, the lead
        # author of Alibi claims that using different biases on forward vs backward positions may work.

        lower_tri_rows = -torch.cat(rows, 0).tril()
        upper_tri_rows = -torch.cat(rows, 0).triu()

        lower_tri_rows = rearrange(lower_tri_rows, 'i j -> () i j')
        upper_tri_rows = rearrange(upper_tri_rows, 'i j -> () i j')
        slopes = self._get_slopes(heads=int(heads / 2))

        all_rows = []
        for h_ in range(int(heads / 2)):
            all_rows.append(lower_tri_rows * slopes[h_])
            all_rows.append(upper_tri_rows * slopes[h_])

        # The resultant bias applies the Alibi position bias looking forward to half of the heads, and backwards to the
        # other half. Since for each head, only 1 direction contains positional information, you should probably use
        # RoPE along with Alibi, to give the opposite direction some positional information.

        self.bias = torch.cat(all_rows, dim=0).cuda()  # shape (heads, max_length, max_length)

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

        bias = repeat(self.bias, 'h i j -> b h i j', b=b)  # repeat over the batch dimension
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
                         frequencies: TensorType[1, 1, "max_length", "dim"]) \
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