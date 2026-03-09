from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DLMConfig:
    sequence_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ffn_mult: int = 4
    dropout: float = 0.0
    softcap: float = 15.0


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
        logits = self.lm_head(x)
        if self.config.softcap > 0:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        return logits


# ---------------------------------------------------------------------------
# Modern DLM — LLaDA-style architecture
# RMSNorm, RoPE, SwiGLU, bidirectional attention, softcap, Mitchell init
# ---------------------------------------------------------------------------

@dataclass
class ModernDLMConfig:
    sequence_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ffn_mult: float = 8 / 3  # SwiGLU hidden ratio (param-matched to 4x GELU MLP)
    rope_theta: float = 10000.0
    softcap: float = 30.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, :, None, :]  # (1, T, 1, D/2)
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def _apply_rotary_emb(x, cos, sin):
    """x: (B, T, H, D), cos/sin: (1, T, 1, D/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x2 * sin + x1 * cos], dim=-1)


class BidirectionalAttention(nn.Module):
    def __init__(self, config: ModernDLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        # (B, T, H, D) -> (B, H, T, D) for SDPA
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModernDLMConfig):
        super().__init__()
        hidden = int(config.ffn_mult * config.n_embd)
        hidden = ((hidden + 63) // 64) * 64  # round to 64 for efficiency
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ModernDenoiserBlock(nn.Module):
    def __init__(self, config: ModernDLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = BidirectionalAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class ModernDLM(nn.Module):
    """LLaDA-style masked diffusion language model.

    Time-free bidirectional transformer with RMSNorm, RoPE, SwiGLU,
    weight tying, softcap, and Mitchell initialization.
    """

    def __init__(self, config: ModernDLMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.rotary = RotaryEmbedding(
            config.n_embd // config.n_head,
            max_seq_len=config.sequence_len,
            theta=config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            [ModernDenoiserBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

    @torch.no_grad()
    def init_weights(self):
        """Mitchell-style init (LLaDA): std=1/√d, per-layer scaling 1/√(2*layer)."""
        std = 1.0 / (self.config.n_embd ** 0.5)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        for layer_idx, block in enumerate(self.blocks):
            layer_std = std / ((2 * (layer_idx + 1)) ** 0.5)
            nn.init.normal_(block.attn.q_proj.weight, mean=0.0, std=layer_std)
            nn.init.normal_(block.attn.k_proj.weight, mean=0.0, std=layer_std)
            nn.init.normal_(block.attn.v_proj.weight, mean=0.0, std=layer_std)
            nn.init.zeros_(block.attn.o_proj.weight)
            nn.init.normal_(block.ffn.gate_proj.weight, mean=0.0, std=layer_std)
            nn.init.normal_(block.ffn.up_proj.weight, mean=0.0, std=layer_std)
            nn.init.zeros_(block.ffn.down_proj.weight)

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        x = self.token_embed(tokens)
        cos, sin = self.rotary(seqlen)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.config.softcap > 0:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        return logits
