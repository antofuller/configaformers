import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ff_mult=4,
                 dropout=0.):
        """
        This is a feed-forward network (FFN). Each input (hidden state or embedding) will be ran through the exact same
        network (weight sharing), in transformers this means each token (word, or sub-word) will be processed through
        the same FFN in a batch process. Unlike the attention or convolution mechanisms, the input cannot "see" the rest
        of the sequence in this operation.
        :param dim: The dimension of the FFN (aka the number of features); for transformers they are almost always equal
        to the dimension of the model (d_model).
        :param ff_mult: Transformers typically use an FFN inner size that is 4 times larger than the input dimension.
        Apparently, a value of 4 optimizes run-time on current hardware (GPUs).
        :param dropout: When training a model for 1 epoch only, dropout likely isn't needed, but it's an option.
        """
        # The GELU non-linearity is standard for an FFN. The choice of non-linearities don't have much of an impact on
        # final performance anyway. See page 8 (https://arxiv.org/pdf/2102.11972.pdf)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )
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

