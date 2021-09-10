import torch.nn as nn
from building_blocks import FFN, NewAttention
from positional_and_masking_utils import RotaryEmbedding
import torch
from einops import rearrange, repeat, reduce


"""
*** Basic implementation to test building blocks ***
"""


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, vocab_sz):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                NewAttention(dim=dim,
                             attn_dim=dim,
                             num_heads=heads,
                             ),
                FFN(dim,
                    ),
            ]))

        # Config
        self.dim = dim
        self.vocab_size = vocab_sz
        self.num_layers = depth
        self.heads = heads
        dim_head = dim // heads

        # Input utils
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.rotary_pos_emb = RotaryEmbedding(int(dim_head/4))

        self.logits_input_norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(self.dim, self.vocab_size)

        self.mask_base = torch.ones(512, 512).triu(diagonal=1)
        dummy_tensor = torch.Tensor([1.0])
        mask_value = max_neg_value(dummy_tensor)
        self.mask_base = (self.mask_base * mask_value).cuda()
        print(self.mask_base)

    def forward(self, seq_ids):
        bsz = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        x = self.token_emb(seq_ids).view(bsz, seq_len, self.dim)
        rotary_pos_emb = self.rotary_pos_emb(seq_len)

        mask = self.mask_base[:seq_len, :seq_len]
        mask = repeat(mask, 'i j -> b h i j', b=bsz, h=self.heads)

        for self_attn, self_ff in self.layers:
            x, _, _ = self_attn(x=x, mask=mask, rotary_pos_emb=rotary_pos_emb)
            x = self_ff(x=x)

        x = self.logits_input_norm(x)
        logits = self.to_logits(x).view(bsz, seq_len, self.vocab_size)
        return logits