import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from positional_and_masking_utils import apply_rotary_pos_emb, AttentionBiasMask
import math
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Optional, Tuple, Union, List, Dict
from positional_and_masking_utils import RotaryEmbedding


patch_typeguard()


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class ReluSquared(nn.Module):
    # Not used yet
    def forward(self, _x):
        return F.relu(_x) ** 2


@typechecked
def shift(t: TensorType["batch", "length", "dim"],  # The tensor that will be shifted
          amount: int,  # The amount of time-steps to shift by
          mask: Optional[TensorType["batch", "length"]] = None,  # Token mask
          ) \
        -> TensorType["batch", "length", "dim"]:

    if amount == 0:  # If the amount of time-steps is 0, then we just return the input tensor
        return t

    if exists(mask):  # Set masked values to zero
        t = t.masked_fill(~mask[..., None], 0.0)

    # This pad operator shifts the features in the sequence (or time) dimension by the amount given, and fills in the
    # start or end of the sequence with zeros
    return F.pad(t, (0, 0, amount, -amount), value=0.0)


class ShiftTokens(nn.Module):
    def __init__(self, config,
                 dim,
                 shift_type,
                 shift_act,
                 ):
        super().__init__()
        self.config = config
        self.shift_type = shift_type
        self.shift_act = shift_act
        sum_features = sum([x['features'] for x in config])  # Add up the number of features to ensure the sum is equal
        # to the total number of features

        assert sum_features == dim, f"Features add up to {sum_features} but dim is {dim}"

    @typechecked
    def forward(self, _x: TensorType["batch", "length", "dim"],
                mask=None,
                ) \
            -> TensorType["batch", "length", "dim"]:

        splitted = []
        feature_position = 0  # Keep track of feature position during for loop
        for idx in range(len(self.config)):
            features_amt = self.config[idx]['features']  # Number of features to grab, for this chunk
            shift_amt = self.config[idx]['shift']  # Number of sequence positions to shift by

            start_idx = feature_position
            end_idx = start_idx + features_amt

            chunk = _x[:, :, start_idx:end_idx]  # Select features and remove them from the input tensor
            chunk = shift(chunk, shift_amt, mask)  # Perform the shift operation

            if self.shift_act == "sigmoid":
                chunk = torch.sigmoid(chunk)

            splitted.append(chunk)  # Store them in a list
            feature_position += features_amt  # Update the feature position

        if self.shift_type == "slice":
            # Piece the slices back together (with some of the chunks shifted)
            _x = torch.cat(splitted, dim=-1)

        elif self.shift_type == "add":
            # Piece the slices back together, then add it to the input _x
            _x = torch.cat(splitted, dim=-1) + _x

        elif self.shift_type == "mult":
            # Piece the slices back together, then multiply it to the input _x
            _x = torch.cat(splitted, dim=-1) * _x

        else:
            print(f"shift_type: {self.shift_type} is not available")

        return _x


class Classifier(nn.Module):
    def __init__(
        self,
        dim: int,  # Input dimension size (typically it is d_model)
        ff_mult: Union[int, float] = 4,  # Hidden layer dimension size multiplier
        dropout: float = 0.0,  # Features to dropout (between 0 and 1)
        num_classes: int = 2,  # Number of classes
    ):
        super().__init__()
        """
        num_classes is the number of features to output, for language modeling, num_classes will be equal to the vocab
        size - where we have 1 class per token. For binary classification (like ELECTRA) we can use num_classes = 2. 
        For knowledge distillation, or contrastive learning, you can output an embedding via num_classes=768, 1024, etc.
        
        Right now, only a vanilla FFN is available. We use an FFN before the final linear output to limit the harm each
        classifier does to the backbone model, since the final layers of a network will specialize to the (pre)training 
        task, and as a result, not be as general.
        """

        inner_dim = int(dim * ff_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

        self.to_logits = nn.Linear(dim, num_classes)
        self.norm = nn.LayerNorm(dim)

    @typechecked
    def forward(self,
                x: TensorType["batch", "length", "dim"],
                ) \
            -> TensorType["batch", "length", "num_classes"]:

        residual = x  # Store input
        x = self.norm(x)  # Input norm
        x = self.net(x)  # 1-layer MLP

        x = self.norm(x + residual)  # Skip connection and norm
        x = self.to_logits(x)  # Linearly project to the desired output size
        return x


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


class FFN(nn.Module):
    def __init__(self,
                 dim: int,  # Input and output dimension size (typically it is d_model)
                 ff_mult: Union[int, float],  # Hidden layer dimension size multiplier
                 num_projections: int = 0,  # Number of projections which are multiplied by each other, element-wise
                 num_gelu: int = 0,  # Number of projections to send through a GELU
                 dropout: float = 0.0,  # Features to dropout (between 0 and 1)
                 pre_norm_bool: bool = True,  # Apply layer normalization before the FFN
                 post_norm_bool: bool = False,  # Apply layer normalization after the FFN
                 token_shift_config: Optional[List[Dict]] = None,  # Config for token shifting
                 inner_token_shift_config: Optional[List[Dict]] = None,  # Config for token shifting the inner features
                 output_gate: Optional[Tuple[str, str]] = None,  # If, and how, to gate the block's output
                 add_residual: bool = True,  # Add skip connection
                 ):
        super().__init__()
        """
        This is the feedforward network (FFN) block. We use a GELU activation function because that is most common, and
        the exact choice of activation function should not matter that much. Please see https://arxiv.org/abs/2102.11972
        - page 8.
        
        It can be configured as Gated Linear Unit (GLU) variants for feedforward networks. 
        See: https://arxiv.org/abs/2002.05202

        Examples:
        GEGLU ---> num_projections=2, num_gelu=1
        Bilinear ---> num_projections=2, num_gelu=0
        Trilinear ---> num_projections=3, num_gelu=0
        RWKV_ChannelMix ---> num_projections=2, num_gelu=1, output_gate= ("sigmoid", "on_residual")
        From: https://github.com/BlinkDL/RWKV-LM (except gelu instead of mish activation)

        *WARNING*: Increasing num_projections will increase the parameter count of your model. To match the param count
        of a vanilla FFN with ff_mult=4, use ff_mult=2.667 if num_projections=2, ff_mult=2 if num_projections=3, or
        ff_mult=1.6 if num_projections=4
        """

        # Config
        inner_dim = int(ff_mult*dim)
        assert 4 >= num_projections, "num_projections must be less than or equal to 4"
        assert num_projections >= num_gelu >= 0, "num_gelu must be >= 0, and <= num_projections"
        if num_projections != 0:
            assert inner_dim % num_projections == 0, "num_projections must divide evenly into inner_dim"

        self.dim = dim
        self.num_projections = num_projections
        self.num_gelu = num_gelu
        self.pre_norm_bool = pre_norm_bool
        self.post_norm_bool = post_norm_bool
        self.output_gate = output_gate
        self.token_shift_config = token_shift_config
        self.inner_token_shift_config = inner_token_shift_config
        self.add_residual = add_residual

        # Functions
        if self.pre_norm_bool:
            self.pre_norm = nn.LayerNorm(dim)

        if self.post_norm_bool:
            self.post_norm = nn.LayerNorm(dim)

        if self.num_projections == 0:
            self.proj_up = nn.Linear(dim, inner_dim)
        else:
            self.proj_up = nn.Linear(dim, inner_dim * num_projections)

        if self.output_gate:
            self.final_gate = nn.Linear(dim, dim)

        if self.token_shift_config:
            self.shift_tokens = ShiftTokens(config=token_shift_config, dim=dim)

        if self.inner_token_shift_config:
            self.shift_tokens_inner = ShiftTokens(config=inner_token_shift_config, dim=inner_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_down = nn.Linear(inner_dim, dim)

    @typechecked
    def _split_and_multiply(self, _x: TensorType["batch", "length", "in_dim"]) \
            -> TensorType["batch", "length", "out_dim"]:

        # Split into chunks of equal shape along the feature/last dimension
        _x = _x.chunk(self.num_projections, dim=-1)

        if self.num_gelu > 0:
            # Loop through every chunk, if the chunk index is less than self.num_gelu, then apply a GELU
            # This will result in GELU(s) being applied to self.num_gelu chunks
            _x = [F.gelu(chunk) if _idx < self.num_gelu else chunk for _idx, chunk in enumerate(_x)]

        # Multiply the chunks by each other, element-wise
        if self.num_projections == 2:
            _x = _x[0] * _x[1]
        elif self.num_projections == 3:
            _x = _x[0] * _x[1] * _x[2]
        elif self.num_projections == 4:
            _x = _x[0] * _x[1] * _x[2] * _x[3]
        else:
            raise "self.num_projections out of range, inside of forward pass"

        return _x

    @typechecked
    def _apply_output_gate(self, _x: TensorType["batch", "length", "dim"],
                           _residual: TensorType["batch", "length", "dim"],
                           ) \
            -> TensorType["batch", "length", "dim"]:

        if self.output_gate[1] == "on_residual":
            gate = self.final_gate(_residual)  # Linearly project the layer's input (aka residual)

        elif self.output_gate[1] == "not_on_residual":
            gate = self.final_gate(_x)  # Linearly project the hidden state

        else:
            raise "The second element of self.output_gate needs to be 'on_residual' or 'not_on_residual'"

        if self.output_gate[0] == "none":
            pass  # Don't apply any activation function to the gate

        elif self.output_gate[0] == "gelu":
            gate = F.gelu(gate)

        elif self.output_gate[0] == "sigmoid":
            gate = torch.sigmoid(gate)

        else:
            raise "The first element of self.output_gate needs to be 'none', 'gelu', or 'sigmoid'"

        _x = gate * _x  # Take the sigmoid of r, and multiply it by x, to gate it

        return _x

    @typechecked
    def forward(self, x: TensorType["batch", "length", "dim"]) \
            -> TensorType["batch", "length", "dim"]:

        residual = x  # Store input

        if self.pre_norm_bool:
            x = self.pre_norm(x)  # Normalize the representations before the FFN

        if self.token_shift_config:
            x = self.shift_tokens(x)  # Shift neighboring token representations

        x = self.proj_up(x)  # Linearly project up to inner_dim, or inner_dim * num_projections, features

        if self.num_projections == 0:
            x = F.gelu(x)
        else:
            x = self._split_and_multiply(x)  # Split up and multiply, element-wise, the intermediate representations

        if self.inner_token_shift_config:
            x = self.shift_tokens_inner(x)  # Shift neighboring token representations

        x = self.dropout(x)  # Set some features to zero
        x = self.proj_down(x)  # Project back down

        if self.output_gate:
            x = self._apply_output_gate(_x=x, _residual=residual)

        if self.post_norm_bool:
            x = self.post_norm(x)  # Normalize the representations

        if self.add_residual:
            x = x + residual  # Add the layer's input to create a residual/skip connection

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
    def __init__(self,
                 dim: int,  # Input and output dimension size (typically it is d_model)
                 num_heads: int,  # Number of attention heads
                 bias_mask_config: List[List],  # Attention biasing and/or masking config
                 dim_attn: Optional[int] = None,  # Dimension size of attention (typically it is equal to dim)
                 previous_attention_bool: bool = False,  # Whether or not to re-use the last attention map
                 residual_attention_bool: bool = False,  # Whether or not to use an attention skip connection
                 pre_norm_bool: bool = True,  # Apply layer normalization before attention
                 post_norm_bool: bool = False,  # Apply layer normalization after attention
                 rotate_qk_bool: bool = True,  # Apply a rotation to queries and keys, before their dot product
                 rotate_v_bool: bool = True,  # Apply a rotation to values
                 dim_rope: Optional[int] = None,  # Number of features to rotate
                 token_shift_config: Optional[List[Dict]] = None,  # Config for token shifting
                 shift_type: str = "slice",  # Token shift type, one of slice, add, or mult
                 shift_act: Optional[str] = "none",  # Apply an activation to the shifted features, sigmoid or none
                 output_gate: Optional[Tuple[str, str]] = None,  # If, and how, to gate the block's output
                 add_residual: bool = True,  # Add skip connection
                 ):
        super().__init__()
        """
        Standard attention function, with a few features. Lazy attention (set previous_attention_bool=True) allows us
        to skip calculating a new attention map, and re-use the last attention map: https://arxiv.org/abs/2102.12702 .
        When not using lazy attention, we can use residual attention (https://arxiv.org/abs/2012.11747) by giving this
        module previous_attn_dots, which are the dots from the last attention layer.
        """

        # Config
        dim_head = int(dim_attn / num_heads)
        assert dim_attn % num_heads == 0, "The attention dimension size (dim_attn) must divide evenly into num_heads"

        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.previous_attention_bool = previous_attention_bool
        self.residual_attention_bool = residual_attention_bool
        self.pre_norm_bool = pre_norm_bool
        self.post_norm_bool = post_norm_bool

        if dim_attn:
            self.dim_attn = dim_attn
        else:
            # If dim_attn is not given, just use dim_model
            self.dim_attn = dim

        self.rotate_qk_bool = rotate_qk_bool
        self.rotate_v_bool = rotate_v_bool
        self.token_shift_config = token_shift_config
        self.output_gate = output_gate
        self.add_residual = add_residual

        if self.rotate_qk_bool or self.rotate_v_bool:
            if dim_rope:
                self.dim_rope = dim_rope
            else:
                # If qk or v needs to be rotated, but no dim_rope is given, use a quarter of the attention head size
                self.dim_rope = int(dim_head/4)
        else:
            self.dim_rope = None

        # Functions
        if self.previous_attention_bool:
            # If we use the attention pattern from the last attention layer, we don't need queries and keys
            self.to_v = nn.Linear(dim, dim_attn, bias=False)

        else:
            # Standard attention layer that will calculate the attention pattern from queries and keys
            self.to_q = nn.Linear(dim, dim_attn, bias=False)
            self.to_k = nn.Linear(dim, dim_attn, bias=False)
            self.to_v = nn.Linear(dim, dim_attn, bias=False)

        if self.pre_norm_bool:
            self.pre_norm = nn.LayerNorm(dim)

        if self.post_norm_bool:
            self.post_norm = nn.LayerNorm(dim)

        if self.token_shift_config:
            self.shift_tokens = ShiftTokens(config=token_shift_config,
                                            dim=dim,
                                            shift_type=shift_type,
                                            shift_act=shift_act)

        if self.output_gate:
            self.final_gate = nn.Linear(dim, dim)

        if self.rotate_qk_bool or self.rotate_v_bool:
            self.rotary_pos_emb = RotaryEmbedding(self.dim_rope)

        self.attn_fn = F.softmax
        self.to_out = nn.Linear(dim_attn, dim)

        self.biasing_and_masking = AttentionBiasMask(config_heads=bias_mask_config,
                                                     num_heads=num_heads,
                                                     max_length=2048,
                                                     )

    @typechecked
    def _calculate_attention_map(self,
                                 _q: TensorType["batch", "length_queries", "dim"],
                                 _k: TensorType["batch", "length_keys", "dim"],
                                 _rope: Optional[TensorType[1, 1, "max_length", "rope_dim"]],
                                 ) \
            -> TensorType["batch", "num_heads", "length_queries", "length_keys"]:

        _q = self.to_q(_q)  # create queries via a linear projection
        _k = self.to_k(_k)  # create keys via a linear projection

        # For q, and k, rearrange the features into heads (hence the name multi-headed attention)
        _q = rearrange(_q, 'batch length_queries (num_heads head_dim) -> batch num_heads length_queries head_dim',
                       num_heads=self.num_heads)

        _k = rearrange(_k, 'batch length_keys (num_heads head_dim) -> batch num_heads length_keys head_dim',
                       num_heads=self.num_heads)

        if self.rotate_qk_bool:
            assert exists(_rope), "Layer must be given RoPE (rotary embeddings) if rotate_qk_bool is True"
            # Apply a rotation to queries and keys which encodes positional information via their dot product
            # Resulting dot products will be higher (more similar) the closer they are to each other in the sequence

            _q = apply_rotary_pos_emb(x=_q, frequencies=_rope)
            _k = apply_rotary_pos_emb(x=_k, frequencies=_rope)

        # Perform a dot product between the queries and keys, along the feature dimension. The resultant tensor,
        # dots, is a measure of similarity between the features, for each head
        # TBD: Explain einsum notation, a bit
        _dots = einsum('b h i d, b h j d -> b h i j', _q, _k) * self.scale

        return _dots

    @typechecked
    def _weighted_sum(self,
                      _v: TensorType["batch", "length_keys", "dim"],
                      _attn_map: TensorType["batch", "num_heads", "length_queries", "length_keys"],
                      _rope: Optional[TensorType[1, 1, "max_length", "rope_dim"]],
                      ) \
            -> TensorType["batch", "num_heads", "length_queries", "head_dim"]:

        _v = self.to_v(_v)  # create values via a linear projection

        # For q, k, and v, rearrange the features into heads (hence the name multi-headed attention)
        _v = rearrange(_v, 'batch length (num_heads head_dim) -> batch num_heads length head_dim',
                       num_heads=self.num_heads)

        if self.rotate_v_bool:
            assert exists(_rope), "Layer must be given RoPE (rotary embeddings) if rotate_v_bool is True"
            _v = apply_rotary_pos_emb(x=_v, frequencies=_rope)

        _v = einsum('b h i j, b h j d -> b h i d', _attn_map, _v)  # Weighted sum of value heads based on attn_map

        return _v

    @typechecked
    def _apply_output_gate(self, _x: TensorType["batch", "length", "dim"],
                           _residual: TensorType["batch", "length", "dim"],
                           ) \
            -> TensorType["batch", "length", "dim"]:

        if self.output_gate[1] == "on_residual":
            gate = self.final_gate(_residual)  # Linearly project the layer's input (aka residual)

        elif self.output_gate[1] == "not_on_residual":
            gate = self.final_gate(_x)  # Linearly project the hidden state

        else:
            raise "The second element of self.output_gate needs to be 'on_residual' or 'not_on_residual'"

        if self.output_gate[0] == "none":
            pass  # Don't apply any activation function to the gate

        elif self.output_gate[0] == "gelu":
            gate = F.gelu(gate)

        elif self.output_gate[0] == "sigmoid":
            gate = torch.sigmoid(gate)

        else:
            raise "The first element of self.output_gate needs to be 'none', 'gelu', or 'sigmoid'"

        _x = gate * _x  # Take the sigmoid of r, and multiply it by x, to gate it

        return _x

    @typechecked
    def forward(self,
                x: TensorType["batch", "length_queries", "dim"],  # Layer input
                context: Optional[TensorType["batch", "length_keys", "dim"]] = None,  # Keys/values or memory
                previous_attn_map: Optional[TensorType["batch", "num_heads", "length_queries", "length_keys"]] = None,
                previous_attn_dots: Optional[TensorType["batch", "num_heads", "length_queries", "length_keys"]] = None,
                ) \
            -> Tuple[
                TensorType["batch", "length_queries", "dim"],  # Layer output (hidden states)
                Optional[TensorType["batch", "num_heads", "length_queries", "length_keys"]],
                Optional[TensorType["batch", "num_heads", "length_queries", "length_keys"]],
            ]:

        residual = x  # Store input

        if self.dim_rope:
            rotary_pos_emb = self.rotary_pos_emb(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.pre_norm_bool:
            x = self.pre_norm(x)  # Normalize the representations before attention

        if self.token_shift_config:
            x = self.shift_tokens(x)  # Shift neighboring token representations

        if exists(previous_attn_map) and self.previous_attention_bool:  # Re-use the last attention map
            if exists(context):
                v_input = context  # Used for cross-attention
            else:
                v_input = x  # Used for self-attention

            attn_map = previous_attn_map  # Set the current attention map to the last map (includes last mask)
            dots = None  # We did not calculate any attention dots

        elif not self.previous_attention_bool:  # Calculate attention map from queries and keys
            if exists(context):
                # For cross-attention, the layer's input are queries, and the keys/values come from the context
                q_input = x
                k_input = context
                v_input = context
            else:
                # For self-attention, the qkv inputs all start from x (this layer's input)
                q_input = x
                k_input = x
                v_input = x

            dots = self._calculate_attention_map(_q=q_input, _k=k_input, _rope=rotary_pos_emb)

            if self.residual_attention_bool:
                dots = dots + previous_attn_dots  # Add attention dots residual connection

            dots = self.biasing_and_masking(dots)  # Apply biasing and/or masking

            attn_map = self.attn_fn(dots, dim=-1)  # Take the softmax over the length of the sequence (keys/values)

        else:
            raise "If self.previous_attention_bool is True, previous_attn_map needs to be given"

        x = self._weighted_sum(_v=v_input, _attn_map=attn_map, _rope=rotary_pos_emb)

        # Merge the heads back together so we have the same number of features as our input
        x = rearrange(x, 'batch num_heads length_queries attn_dim -> batch length_queries (num_heads attn_dim)')

        x = self.to_out(x)  # Send through a final linear projection

        if self.output_gate:
            x = self._apply_output_gate(_x=x, _residual=residual)

        if self.post_norm_bool:
            x = self.post_norm(x)  # Normalize the representations

        if self.add_residual:
            x = x + residual  # Add the layer's input to create a residual/skip connection

        return x, attn_map, dots  # Return the output, attention map, and the dots (in case we need them later)
