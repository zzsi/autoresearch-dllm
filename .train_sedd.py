"""
Faithful SEDD (Score Entropy Discrete Diffusion) replication.
Absorbing graph, loglinear noise, score entropy loss, AdaLN, PC sampling.

Reference: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

Usage: CUDA_VISIBLE_DEVICES=1 uv run .train_sedd.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, make_token_dataloader

TIME_BUDGET = 43200  # 12 hours

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH = 12
N_EMBD = 768
N_HEAD = 12
COND_DIM = 128
MLP_RATIO = 4
DROPOUT = 0.1
USE_ORIGINAL_ARCH = True  # True = match original SEDD arch exactly (6.4% better loss)

TOTAL_BATCH_SIZE = 2 ** 15
DEVICE_BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.999)
WARMUP_STEPS = 2500
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999

SAMPLING_EPS = 1e-3

EVAL_ITERS = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sedd_small")

# ---------------------------------------------------------------------------
# LogLinear Noise Schedule
# ---------------------------------------------------------------------------

class LogLinearNoise:
    """LogLinear noise: sigma(t) = -log(1 - (1-eps)*t), so mask_prob = 1 - e^(-sigma) ≈ (1-eps)*t."""
    def __init__(self, eps=1e-3):
        self.eps = eps

    def __call__(self, t):
        """Returns (sigma, dsigma) given t in [eps, 1]."""
        sigma = -torch.log1p(-(1 - self.eps) * t)
        dsigma = (1 - self.eps) / (1 - (1 - self.eps) * t)
        return sigma, dsigma


# ---------------------------------------------------------------------------
# Absorbing Graph
# ---------------------------------------------------------------------------

class AbsorbingGraph:
    """
    Absorbing state discrete diffusion graph.
    Vocab tokens: 0..D-1, absorbing (mask) token: D.
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size  # D (without mask)
        self.dim = vocab_size + 1     # D+1 (with mask token at index D)
        self.mask_id = vocab_size     # absorbing state index

    def sample_transition(self, x, sigma):
        """Forward process: mask tokens with probability 1 - e^(-sigma)."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand_like(x, dtype=torch.float32) < move_chance
        return torch.where(move_indices, self.mask_id, x)

    def score_entropy(self, log_score, sigma, x_t, x_0):
        """
        Score entropy loss for absorbing graph.
        Only contributes at masked positions (x_t == mask_id).
        log_score: (B, L, D+1) — model output (log score ratios)
        """
        rel_ind = (x_t == self.mask_id)  # (B, L)

        # esigm1 = exp(sigma) - 1, computed carefully for small sigma
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # ratio = 1 / (exp(sigma) - 1) at masked positions
        ratio = 1.0 / esigm1.expand_as(x_t)[rel_ind]  # (N_masked,)
        other_ind = x_0[rel_ind]  # true clean tokens at masked positions (N_masked,)

        # Gather log_score at true token positions
        log_score_masked = log_score[rel_ind]  # (N_masked, D+1)
        neg_term = ratio * torch.gather(log_score_masked, -1, other_ind.unsqueeze(-1)).squeeze(-1)

        # Positive term: sum of exp(score) over non-mask tokens
        pos_term = log_score_masked[:, :-1].exp().sum(dim=-1)

        # Constant: ratio * (log(ratio) - 1)
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros_like(x_t, dtype=torch.float32)
        entropy[rel_ind] = pos_term - neg_term + const
        return entropy

    def reverse_rate(self, x, score):
        """Compute reverse transition rate from score."""
        # transp_rate for absorbing: -one_hot(x) with special case for mask
        # For masked positions: rate to any non-mask token j = score[j]
        # For non-masked positions: rate to mask = -1, self = 1 (but we only care about masked)
        # Simplified: reverse_rate[j] = score[j] * transp_rate[j]
        D = self.dim
        transp_rate = -F.one_hot(x, num_classes=D).float()
        # At mask positions, add 1 to all entries
        mask_pos = (x == self.mask_id)
        transp_rate[mask_pos] += 1.0

        normalized_rate = transp_rate * score
        # Fix diagonal
        normalized_rate.scatter_(-1, x.unsqueeze(-1), torch.zeros_like(normalized_rate[..., :1]))
        normalized_rate.scatter_(-1, x.unsqueeze(-1), -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def staggered_score(self, score, dsigma):
        """Compute p_{sigma-dsigma}(z) / p_sigma(x) ≈ e^{-dsigma*E} * score."""
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp().unsqueeze(-1)
        score[..., -1] += extra_const
        return score

    def transp_transition(self, x, sigma):
        """Transposed transition matrix row for position x."""
        D = self.dim
        edge = (-sigma).unsqueeze(-1).exp() * F.one_hot(x, num_classes=D).float()
        # For mask positions: add (1 - e^(-sigma))
        mask_pos = (x == self.mask_id)
        correction = torch.where(mask_pos, 1 - (-sigma).exp(), torch.zeros_like(sigma))
        edge += correction.unsqueeze(-1)
        return edge


# ---------------------------------------------------------------------------
# SEDD Model — two variants:
#   USE_ORIGINAL_ARCH=True  → match original SEDD repo exactly
#   USE_ORIGINAL_ARCH=False → our simplified version (RMSNorm, no bias, etc.)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class OrigLayerNorm(nn.Module):
    """Original SEDD LayerNorm: fp32 cast, weight-only (no bias)."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0, rope_on_v=True):
        super().__init__()
        self.rope_on_v = rope_on_v
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def _apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, cond_dim, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


def _adaln_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SEDDAttention(nn.Module):
    def __init__(self, n_embd, n_head, rope_on_v=True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.rope_on_v = rope_on_v
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        if self.rope_on_v:
            v = _apply_rotary_emb(v, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SEDDBlock(nn.Module):
    def __init__(self, n_embd, n_head, cond_dim, mlp_ratio=4, original_arch=False, dropout=0.0):
        super().__init__()
        self.original_arch = original_arch
        Norm = OrigLayerNorm if original_arch else RMSNorm
        self.norm1 = Norm(n_embd)
        self.attn = SEDDAttention(n_embd, n_head, rope_on_v=not original_arch)
        self.norm2 = Norm(n_embd)
        hidden = mlp_ratio * n_embd
        mlp_bias = original_arch
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=mlp_bias),
            nn.GELU(approximate="tanh") if original_arch else nn.GELU(),
            nn.Linear(hidden, n_embd, bias=mlp_bias),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * n_embd, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, cos, sin, c):
        mods = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        h = _adaln_modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.dropout1(self.attn(h, cos, sin))
        h = _adaln_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.dropout2(self.mlp(h))
        return x


class SEDDModel(nn.Module):
    """SEDD model: DiT-style with AdaLN, RoPE, bidirectional attention."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, cond_dim, mlp_ratio, seq_len,
                 original_arch=False, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = vocab_size + 1  # +1 for absorbing token
        self.n_embd = n_embd
        self.original_arch = original_arch
        Norm = OrigLayerNorm if original_arch else RMSNorm

        if original_arch:
            # Original: raw Parameter with kaiming_uniform init
            self.token_embed_weight = nn.Parameter(torch.empty(self.dim, n_embd))
            nn.init.kaiming_uniform_(self.token_embed_weight, a=math.sqrt(5))
        else:
            self.token_embed = nn.Embedding(self.dim, n_embd)

        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary = RotaryEmbedding(
            n_embd // n_head, max_seq_len=seq_len,
            rope_on_v=not original_arch,
        )

        self.blocks = nn.ModuleList([
            SEDDBlock(n_embd, n_head, cond_dim, mlp_ratio, original_arch=original_arch, dropout=dropout)
            for _ in range(n_layer)
        ])

        self.norm_final = Norm(n_embd)
        self.final_adaLN = nn.Linear(cond_dim, 2 * n_embd, bias=True)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)
        self.output_proj = nn.Linear(n_embd, self.dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def _embed_tokens(self, indices):
        if self.original_arch:
            return self.token_embed_weight[indices]
        return self.token_embed(indices)

    def forward(self, indices, sigma):
        """
        indices: (B, L) int — token indices (0..vocab_size for real, vocab_size for mask)
        sigma: (B,) float — noise level
        Returns: (B, L, D+1) log-scores
        """
        x = self._embed_tokens(indices)
        c = F.silu(self.sigma_map(sigma))

        cos, sin = self.rotary(indices.size(1))
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, cos, sin, c)

            shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=2)
            x = _adaln_modulate(self.norm_final(x), shift, scale)
            x = self.output_proj(x)

        # Scale by sigma (SEDD's numerical stabilization for absorbing)
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log()
        x = x - esigm1_log[:, None, None] - math.log(self.dim - 1)

        # Zero out self-scores (score of current token should be 0)
        x = torch.scatter(x, -1, indices.unsqueeze(-1), torch.zeros_like(x[..., :1]))

        return x

    def get_score(self, indices, sigma):
        """Return exp(log_score) for sampling."""
        return self.forward(indices, sigma).exp()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.data, 1 - self.decay)

    def apply(self, model):
        """Temporarily apply EMA weights."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])
        del self.backup


# ---------------------------------------------------------------------------
# PC Sampler (Euler predictor + denoiser)
# ---------------------------------------------------------------------------

def _sample_categorical(probs):
    """Sample from categorical distribution, handling edge cases."""
    probs = probs.clamp(min=0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])


@torch.no_grad()
def generate_sedd(model, graph, noise, seq_len, steps=128, device="cuda"):
    """PC sampler with Analytic predictor + denoiser."""
    B = 1
    eps = SAMPLING_EPS

    # Start from all-mask
    x = torch.full((B, seq_len), graph.mask_id, dtype=torch.long, device=device)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    for i in range(steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        sigma_cur = noise(torch.tensor([t], device=device))[0]
        sigma_next = noise(torch.tensor([t_next], device=device))[0]
        dsigma = sigma_cur - sigma_next

        sigma_batch = sigma_cur.expand(B)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_score = model(x, sigma_batch)
        score = log_score.float().exp()

        # Analytic predictor
        stag_score = graph.staggered_score(score, dsigma)
        transp_trans = graph.transp_transition(x, dsigma)
        probs = stag_score * transp_trans
        x = _sample_categorical(probs)

    # Final denoising step
    t_final = timesteps[-1]
    sigma_final = noise(torch.tensor([t_final], device=device))[0]
    sigma_batch = sigma_final.expand(B)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        log_score = model(x, sigma_batch)
    score = log_score.float().exp()

    stag_score = graph.staggered_score(score, sigma_final)
    transp_trans = graph.transp_transition(x, sigma_final)
    probs = stag_score * transp_trans
    # Truncate absorbing state for final output
    probs = probs[..., :-1]
    x = _sample_categorical(probs)

    return x


# ---------------------------------------------------------------------------
# Mid-training generation
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL = 5000  # generate samples every N steps
CKPT_INTERVAL = 10000   # save checkpoint every N steps

@torch.no_grad()
def mid_train_generate(raw_model, ema, graph, noise_schedule, tokenizer, device, n=3, steps=128, seq_len=256):
    was_training = raw_model.training
    ema.apply(raw_model)
    raw_model.eval()
    print()
    for i in range(n):
        output = generate_sedd(raw_model, graph, noise_schedule, seq_len, steps=steps, device=device)
        output = output.clamp(0, tokenizer.get_vocab_size() - 1)
        text = tokenizer.decode(output[0].tolist())
        print(f"  [sample {i}] {text[:200]}")
    ema.restore(raw_model)
    if was_training:
        raw_model.train()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()  # 8192

graph = AbsorbingGraph(vocab_size)  # mask_id = 8192
noise_schedule = LogLinearNoise(eps=SAMPLING_EPS)

raw_model = SEDDModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=DEPTH,
    cond_dim=COND_DIM,
    mlp_ratio=MLP_RATIO,
    seq_len=MAX_SEQ_LEN,
    original_arch=USE_ORIGINAL_ARCH,
    dropout=DROPOUT,
).to(device)

ema = EMA(raw_model, decay=EMA_DECAY)
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

# ---------------------------------------------------------------------------
# Resume from checkpoint if available
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
start_step = 0
prev_training_time = 0.0

if os.path.exists(ckpt_path):
    print(f"Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if "ema_state" in ckpt:
        ema.shadow = ckpt["ema_state"]
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("num_steps", 0)
    prev_training_time = ckpt.get("training_seconds", 0.0)
    prev_loss = ckpt.get("val_loss", "N/A")
    print(f"  Resumed at step {start_step}, prev training time: {prev_training_time:.0f}s, prev val_loss: {prev_loss}")
    del ckpt

model = torch.compile(raw_model, dynamic=False, mode="max-autotune-no-cudagraphs")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

train_loader = make_token_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x_batch, epoch = next(train_loader)

num_params = sum(p.numel() for p in raw_model.parameters())
print(f"SEDD Training")
print(f"Vocab size: {vocab_size:,} (+1 absorbing = {graph.dim})")
print(f"Num params: {num_params / 1e6:.1f}M")
print(f"Config: d{DEPTH} w{N_EMBD} h{N_HEAD} cond_dim={COND_DIM} original_arch={USE_ORIGINAL_ARCH}")
print(f"Time budget: {TIME_BUDGET}s ({TIME_BUDGET/3600:.0f}h)")
print(f"LR: {LR}, WD: {WEIGHT_DECAY}, betas: {BETAS}")
print(f"Grad accum steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def save_checkpoint(raw_model, ema, optimizer, step, total_training_time, val_loss=None):
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "ema_state": ema.shadow,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "num_steps": step,
        "total_tokens": step * TOTAL_BATCH_SIZE,
        "training_seconds": total_training_time,
        "config": dict(
            vocab_size=vocab_size, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=DEPTH, cond_dim=COND_DIM, mlp_ratio=MLP_RATIO,
            seq_len=MAX_SEQ_LEN, original_arch=USE_ORIGINAL_ARCH,
        ),
    }, ckpt_path)

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = start_step

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        # Sample timestep
        t = (1 - SAMPLING_EPS) * torch.rand(DEVICE_BATCH_SIZE, device=device) + SAMPLING_EPS
        sigma, dsigma = noise_schedule(t)

        # Forward process: mask tokens
        x_noisy = graph.sample_transition(x_batch, sigma[:, None])

        # Model forward
        torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_score = model(x_noisy, sigma)

        # Score entropy loss
        loss_per_token = graph.score_entropy(log_score.float(), sigma[:, None], x_noisy, x_batch)
        # Weight by dsigma (noise rate)
        loss = (dsigma[:, None] * loss_per_token).sum(dim=-1).mean()

        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x_batch, epoch = next(train_loader)

    # LR schedule: warmup only (SEDD uses warmup + constant)
    lr_mult = min(1.0, (step + 1) / WARMUP_STEPS) if WARMUP_STEPS > 0 else 1.0
    for group in optimizer.param_groups:
        group["lr"] = LR * lr_mult

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.update(raw_model)

    train_loss_f = train_loss.item()
    if (not torch.isfinite(train_loss)) or train_loss_f > 1e6:
        print(f"\nFAIL at step {step}: loss={train_loss_f}")
        raise SystemExit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > start_step + 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step - start_step + 1))
    elapsed = total_training_time + prev_training_time
    progress = min(elapsed / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - elapsed)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.4f} | "
        f"lr: {LR * lr_mult:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == start_step:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    # Periodic generation
    if step > start_step and step % SAMPLE_INTERVAL == 0:
        mid_train_generate(raw_model, ema, graph, noise_schedule, tokenizer, device)

    # Periodic checkpoint
    if step > start_step and step % CKPT_INTERVAL == 0:
        gc.enable()
        save_checkpoint(raw_model, ema, optimizer, step, elapsed)
        print(f"\n  [checkpoint saved at step {step}]")
        gc.disable()

    step += 1
    elapsed_check = total_training_time + prev_training_time
    if step > start_step + 10 and elapsed_check >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE
total_elapsed = total_training_time + prev_training_time

# ---------------------------------------------------------------------------
# Eval: score entropy on val set
# ---------------------------------------------------------------------------

gc.enable()
val_loader = make_token_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")

# Apply EMA for eval
ema.apply(raw_model)
model_eval = raw_model  # use uncompiled for eval
model_eval.eval()

val_losses = []
with torch.no_grad():
    for _ in range(EVAL_ITERS):
        vx, _ = next(val_loader)
        t = (1 - SAMPLING_EPS) * torch.rand(DEVICE_BATCH_SIZE, device=device) + SAMPLING_EPS
        sigma, dsigma = noise_schedule(t)
        vx_noisy = graph.sample_transition(vx, sigma[:, None])
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_score = model_eval(vx_noisy, sigma)
        loss_per_token = graph.score_entropy(log_score.float(), sigma[:, None], vx_noisy, vx)
        vloss = (dsigma[:, None] * loss_per_token).sum(dim=-1).mean()
        val_losses.append(vloss.item())
val_loss = sum(val_losses) / len(val_losses)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss (score_entropy): {val_loss:.4f}")
print(f"training_seconds:         {total_elapsed:.1f}")
print(f"total_seconds:            {t_end - t_start:.1f}")
print(f"peak_vram_mb:             {peak_vram_mb:.1f}")
print(f"total_tokens_M:           {total_tokens / 1e6:.1f}")
print(f"num_steps:                {step}")
print(f"num_params_M:             {num_params / 1e6:.1f}")

# ---------------------------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------------------------

save_checkpoint(raw_model, ema, optimizer, step, total_elapsed, val_loss=val_loss)
print(f"Checkpoint saved to {ckpt_path}")

# ---------------------------------------------------------------------------
# Generate samples
# ---------------------------------------------------------------------------

gc.enable()
print(f"\nGenerating samples with Analytic PC sampler (128 steps)...")

gen_seq_len = 256
num_samples = 5

for i in range(num_samples):
    output = generate_sedd(raw_model, graph, noise_schedule, gen_seq_len, steps=128, device=device)
    # Clamp to valid vocab range
    output = output.clamp(0, vocab_size - 1)
    text = tokenizer.decode(output[0].tolist())
    print(f"  [{i}] {text[:200]}")

# Restore non-EMA weights
ema.restore(raw_model)

# Also try with more steps
print(f"\nGenerating with 256 steps...")
ema.apply(raw_model)
for i in range(3):
    output = generate_sedd(raw_model, graph, noise_schedule, gen_seq_len, steps=256, device=device)
    output = output.clamp(0, vocab_size - 1)
    text = tokenizer.decode(output[0].tolist())
    print(f"  [{i}] {text[:200]}")
ema.restore(raw_model)

print("\nDone.")
