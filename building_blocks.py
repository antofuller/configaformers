import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import math


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Alibi(nn.Module):
    def __init__(self, heads, max_length):
        super().__init__()
        """
        This builds the matrix seen on the right side of Figure 3 - https://arxiv.org/pdf/2108.12409.pdf . It is a 
        relative position bias (since it biases the attention pattern based on the relative positions of the inputs). 
        Right now, I only have an encoder implementation of this matrix (which isn't mentioned in the paper). The causal
        version will be added shortly; it's actually just the lower triangle portion.
        
        From the paper, Alibi seems to perform on par with RoPE, but it can generalize to longer sequences not seen
        during training. RoPE cannot do that. EleutherAI discord experiments/rumors claim that using Alibi and RoPE
        slightly improves performance; my experiments would agree. 
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
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        b, h, i, j = qk_dots.shape

        bias = repeat(self.bias, 'h i j -> b h i j', b=b)  # repeat over the batch dimension
        bias = bias[:, :, :i, :j].view(qk_dots.shape)  # trim the bias tensor such that it matches the shape of qk_dots

        return qk_dots + bias  # this adds them together, creating the relative positional bias


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """
        This rotary embedding (RoPE) implementation is taken 100% from lucidrains. Explanation forthcoming.
        :param dim: number of features to rotate
        """

        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq).cuda()
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> () () n d')


def rotate_half(x):
    # This implementation is taken 100% from lucidrains
    # Explanation forthcoming
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    # This implementation is taken 100% from lucidrains
    # Explanation forthcoming
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


"""
Feed-forward networks (FFNs), aka multi-layer perceptrons (MLPs), receive each individual token representation and
perform some computation on them. Typically they consist of a linear projection to a larger dimension, are passed
through an activation function (aka non-linearity), and are then linearly projected back down into the input size.

In transformers, FFNs are 1 of the 2 main building blocks (along with attention). Each token representation is fed
into the FFN in a batch. Thus, the weights in the FFN are shared across the sequence dimension (the first token will use
the exact same FFN as a middle token). Crucially, there is no information exchange across the sequence, unlike attention
or convolution operators that can "combine" information from other representations in the sequence. 

Without FFNs, transformers don't work well: https://arxiv.org/abs/2103.03404
"""


class VanillaFFN(nn.Module):
    def __init__(self,
                 dim,
                 ff_mult=4,
                 dropout=0.0,
                 pre_norm_bool=True,
                 post_norm_bool=False,
                 ):
        """
        This is the "vanilla", or standard FFN used in transformer blocks. We use a GELU activation function because
        that is most common, and the exact choice of activation function should not matter that much. Please see
        https://arxiv.org/abs/2102.11972 - page 8.

        :param dim: Input and output dimension size
        :param ff_mult: Hidden layer dimension size multiplier
        :param dropout: Features to dropout (between 0 and 1)
        :param pre_norm_bool: Apply layer normalization before the FFN
        :param post_norm_bool: Apply layer normalization after the FFN
        """
        super().__init__()

        # config
        inner_dim = int(dim * ff_mult)
        self.pre_norm_bool = pre_norm_bool
        self.post_norm_bool = post_norm_bool

        # functions
        if self.pre_norm_bool:
            self.pre_norm = nn.LayerNorm(dim)

        if self.post_norm_bool:
            self.post_norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # project to more features
            nn.GELU(),  # activation function
            nn.Dropout(dropout),  # set some features to 0
            nn.Linear(inner_dim, dim)  # project back down
        )

    def forward(self, x):
        residual = x  # store input

        if self.pre_norm_bool:
            x = self.pre_norm(x)  # normalize the representations before the FFN

        x = self.net(x)  # send through FFN
        x = x + residual  # add the layer's input to create a residual/skip connection

        if self.post_norm_bool:
            x = self.post_norm(x)  # normalize the representations after the residual

        return x


class GLUVariantFFN(nn.Module):
    def __init__(self,
                 dim,
                 ff_mult,
                 num_projections=2,
                 num_gelu=1,
                 dropout=0.0,
                 pre_norm_bool=True,
                 post_norm_bool=False,
                 ):
        """
        Gated Linear Unit (GLU) variants for feedforward networks. See: https://arxiv.org/abs/2002.05202

        Examples:
        GEGLU ---> default config
        Bilinear ---> num_gelu=0, remainder are default
        Trilinear ---> num_projections=3, num_gelu=0, remainder are default

        *WARNING*: Increasing num_projections will increase the parameter count of your model. To match the param count
        of a VanillaFFN with ff_mult=4, use ff_mult=2.667 if num_projections=2, ff_mult=2 if num_projections=3, or
        ff_mult=1.6 if num_projections=4

        :param dim: Input and output dimension size
        :param ff_mult: Hidden layer dimension size multiplier
        :param num_projections: Number of input projections which are multiplied by each other, element-wise
        :param num_gelu: Number of projections to send through a GELU
        :param dropout: Features to dropout (between 0 and 1)
        :param pre_norm_bool: Apply layer normalization before the FFN
        :param post_norm_bool: Apply layer normalization after the FFN
        """
        super().__init__()

        # config
        inner_dim = int(ff_mult*dim)
        assert 4 >= num_projections >= 2, "num_projections must be 2, 3, or 4"
        assert num_projections >= num_gelu >= 0, "num_gelu must be >= 0, and <= num_projections"
        assert inner_dim % num_projections == 0, "num_projections must divide evenly into inner_dim"

        self.dim = dim
        self.num_projections = num_projections
        self.num_gelu = num_gelu
        self.pre_norm_bool = pre_norm_bool
        self.post_norm_bool = post_norm_bool

        # functions
        if self.pre_norm_bool:
            self.pre_norm = nn.LayerNorm(dim)

        if self.post_norm_bool:
            self.post_norm = nn.LayerNorm(dim)

        self.proj_up = nn.Linear(dim, inner_dim * num_projections)
        self.dropout = nn.Dropout(dropout)
        self.proj_down = nn.Linear(inner_dim, dim)

    def forward(self, x):
        residual = x  # store input

        if self.pre_norm_bool:
            x = self.pre_norm(x)  # normalize the representations before the FFN

        # linearly project up to inner_dim * num_projections features, then split into chunks of equal shape
        chunks = self.proj_up(x).chunk(self.num_projections, dim=-1)

        # apply GELU(s) to the required number of chunks
        if self.num_gelu > 0:
            chunks = [chunk if _idx >= self.num_gelu else F.gelu(chunk) for _idx, chunk in enumerate(chunks)]

        # multiply the chunks by each other, element-wise
        if self.num_projections == 2:
            x = chunks[0] * chunks[1]
        elif self.num_projections == 3:
            x = chunks[0] * chunks[1] * chunks[2]
        elif self.num_projections == 4:
            x = chunks[0] * chunks[1] * chunks[2] * chunks[3]
        else:
            raise "self.num_projections out of range, inside of forward pass"

        x = self.dropout(x)
        x = self.proj_down(x)  # project back down
        x = x + residual  # add the layer's input to create a residual/skip connection

        if self.post_norm_bool:
            x = self.post_norm(x)  # normalize the representations after the residual

        return x


"""
Attention is the second main building block used in transformers. Simply, attention allows for inputs to "see", or "take
into account" other inputs in the sequence. (insert link to explainer blogs)

The attention mechanism is expensive because every element in the sequence must be compared with every other element.
This leads to the computational cost growing with the sequence length *squared*. There have been dozens of attention 
variants that lower the computational cost of attention, but thus far, none provide similar performance. 

According to lucidrains, the routing transformer and pooling transformers are the most promising. But unless you are
working with very long sequences (i.e. in the thousands of tokens) - stick with the vanilla attention mechanism.
"""


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        use_previous_attention=False,
    ):
        super().__init__()
        """
        Explanation forthcoming
        :param dim:
        :param heads:
        :param use_previous_attention:
        """

        assert dim % heads == 0
        dim_head = int(dim / heads)
        self.scale = dim_head ** -0.5
        self.num_heads = heads

        qk_dim = v_dim = dim_head * heads

        if use_previous_attention:
            # If we use the attention pattern from the last attention layer, we don't need queries and keys
            self.to_v = nn.Linear(dim, v_dim, bias=False)

        else:
            # Standard attention layer that will calculate the attention pattern from queries and keys
            self.to_q = nn.Linear(dim, qk_dim, bias=False)
            self.to_k = nn.Linear(dim, qk_dim, bias=False)
            self.to_v = nn.Linear(dim, v_dim, bias=False)

        self.attn_fn = F.softmax
        self.to_out = nn.Linear(v_dim, dim)

    def forward(self, x,
                positional_bias_fn=None,
                previous_attn=None,
                rotary_pos_emb=None):

        residual = x  # store input

        if exists(previous_attn):
            # Only v is needed, since we will re-use a previous attention pattern
            v_input = x
            v = self.to_v(v_input)  # create values via a linear projection
            attn = previous_attn  # set the current attention pattern to the previous pattern

        else:
            # For self-attention, the qkv inputs all start from x (this layer's input)
            q_input = x
            k_input = x
            v_input = x

            q = self.to_q(q_input)  # create queries via a linear projection
            k = self.to_k(k_input)  # create keys via a linear projection
            v = self.to_v(v_input)  # create values via a linear projection

            # Along the feature dimension, rearrange the tensor into heads (hence the name multi-headed attention)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

            if exists(rotary_pos_emb):
                # Implementation is taken 100% from lucidrains.
                # We want to rotate the q, k, and v to incorporate positional info. If we only want to rotate a portion
                # of the features, then we must slice the tensors, rotate the slice, then piece them back together with
                # the un-rotated portion of the tensor

                l = rotary_pos_emb.shape[-1]  # the number of features we wish to rotate
                (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))  # slicing qkv
                ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))  # rotating the slices
                q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))  # piece back together

            # Perform a dot product between the queries and keys, along the feature dimension. The resultant tensor,
            # dots, is a measure of similarity between the features
            # TBD: Explain einsum notation, a bit
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            if exists(positional_bias_fn):
                dots = positional_bias_fn(dots)  # apply Alibi (relative positional bias) to the attention pattern

            attn = self.attn_fn(dots, dim=-1)  # take the softmax over the length of the sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # grab the value heads based on the attention pattern
        out = rearrange(out, 'b h n d -> b n (h d)')  # merge the heads back together so we have the same number of
        # features as our input
        out = self.to_out(out)  # send through a final linear projection
        out = out + residual  # add the layer's input to create a residual/skip connection

        return out, attn  # return the output, and the attn pattern (in case we want to re-use it later)

