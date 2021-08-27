import torch.nn.functional as F
import torch
from torch import nn, einsum
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from einops import rearrange, repeat, reduce

patch_typeguard()  # use before @typechecked


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        When used in an FFN, GEGLU replaces the first linear layer. This implementation of GEGLU first sends the input
        through a linear layer with twice the output size of the standard linear layer in an FFN. It then splits this
        output into two equal chunks. These chunks will now each have the size of dim_out. One of the chunks (named gate)
        will be sent through a GELU non-linearity, and then multiplied (element wise) by the other chunk that was not
        sent through the non-linearity. The "GLU" in GEGLU stands for "gated linear unit", since it acts as a gate that
        decides how much of each feature to "let through". The "GE" prefix refers to the type of non-linearity used, in
        this case a GELU.
        :param dim_in: number of features feed into the FFN
        :param dim_out: number of inner dimension features of the FFN
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # the chunk operation splits the projection into 2 chunks, along the last dimension (features)
        x, gate = self.proj(x).chunk(2, dim=-1)
        return gate * F.gelu(x)


class Bilinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        When used in an FFN, this replaces the first linear layer. It first sends the input through a linear layer
        with twice the output size of the standard linear layer in an FFN. It then splits this output into two equal
        chunks. These chunks will now each have the size of dim_out. Finally, the two chunks are multiplied by each
        other, element-wise. According to https://arxiv.org/abs/2002.05202 , this technique is competitive with GEGLU,
        but crucially, it omits the non-linearity. There is no explicit activation function!
        :param dim_in: number of features feed into the FFN
        :param dim_out: number of inner dimension features of the FFN
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # the chunk operation splits the projection into 2 chunks, along the last dimension (features)
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        return x1 * x2


class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_type="standard",
                 ff_mult=4,
                 dropout=0.):
        """
        This is a feed-forward network (FFN). Each input (hidden state or embedding) will be ran through the exact same
        network (weight sharing), in transformers this means each token (word, or sub-word) will be processed through
        the same FFN in a batch process. Unlike the attention or convolution mechanisms, the input cannot "see" the rest
        of the sequence in this operation.
        :param dim: The dimension of the FFN (aka the number of features); for transformers they are almost always equal
        to the dimension of the model (d_model).
        :param ffn_type: GEGLU and bilinear techniques have 1/3 more parameters than the standard option.
        :param ff_mult: Transformers typically use an FFN inner size that is 4 times larger than the input dimension.
        Apparently, a value of 4 optimizes run-time on current hardware (GPUs).
        :param dropout: When training a model for 1 epoch only, dropout likely isn't needed, but it's an option.
        """
        # The GELU non-linearity (aka activation function) is standard for an FFN. The choice of non-linearities don't
        # have much of an impact on final performance anyway. See page 8 (https://arxiv.org/pdf/2102.11972.pdf)
        super().__init__()

        inner_dim = int(dim * ff_mult)
        if ffn_type == "GEGLU":
            self.net = nn.Sequential(
                GEGLU(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim)
            )
        elif ffn_type == "standard":
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim)
            )
        elif ffn_type == "bilinear":
            self.net = nn.Sequential(
                Bilinear(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim)
            )
        else:
            print(f'ffn_type does not match any available options')

        # despite many varieties of applying layer norms in transformers, pre-norm still seems like the best option
        self.input_norm = nn.LayerNorm(dim)

    @typechecked
    def forward(self, x: TensorType["batch", "sequence", "features"])\
            -> TensorType["batch", "sequence", "features"]:

        # A residual (skip) connection, and a pre-FFN layer normalization are standard choices for transformers
        # Why? Because using them helps training
        residual = x
        x = self.input_norm(x)

        ff_out = self.net(x)
        return residual + ff_out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        causal=True,
        mask=None,
        dropout=0.,
    ):
        """
        TBD
        :param dim:
        :param heads:
        :param causal:
        :param mask:
        :param dropout:
        """
        super().__init__()
        assert dim % heads == 0
        dim_head = int(dim / heads)
        self.scale = dim_head ** -0.5
        self.num_heads = heads
        self.causal = causal
        self.mask = mask

        qk_dim = v_dim = dim_head * heads

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, v_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_fn = F.softmax
        self.to_out = nn.Linear(v_dim, dim)

    @typechecked
    def forward(self, x: TensorType["batch", "sequence", "features"],
                mask=None) \
            -> TensorType["batch", "sequence", "features"]:

        device = x.device
        q_input = x
        k_input = x
        v_input = x

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)