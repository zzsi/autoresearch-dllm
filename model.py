from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DLMConfig:
    sequence_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ffn_mult: int = 4
    dropout: float = 0.0


class DenoiserBlock(nn.Module):
    def __init__(self, config: DLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.n_embd)
        hidden = config.ffn_mult * config.n_embd
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
        )

    def forward(self, x):
        a = self.norm1(x)
        a, _ = self.attn(a, a, a, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class MaskedDLM(nn.Module):
    def __init__(self, config: DLMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.sequence_len, config.n_embd)
        self.blocks = nn.ModuleList([DenoiserBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        assert seqlen <= self.config.sequence_len
        pos = torch.arange(seqlen, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
