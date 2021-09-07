import torch.nn as nn
from building_blocks import FFN, Attention
from positional_and_masking_utils import RotaryEmbedding

"""
*** Basic implementation to test building blocks ***
"""


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, vocab_sz):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim,
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

    def forward(self, seq_ids):
        bsz = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        x = self.token_emb(seq_ids).view(bsz, seq_len, self.dim)
        rotary_pos_emb = self.rotary_pos_emb(seq_len)

        for self_attn, self_ff in self.layers:
            x = self_attn(x=x, rotary_pos_emb=rotary_pos_emb)
            x = self_ff(x=x)

        x = self.logits_input_norm(x)
        logits = self.to_logits(x).view(bsz, seq_len, self.vocab_size)
        return logits