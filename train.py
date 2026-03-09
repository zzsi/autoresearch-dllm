"""
Autoresearch-DLLM pretraining script.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time

import torch
import torch.nn.functional as F

from duel_eval import evaluate_duel_bpb
from model import DLMConfig, MaskedDLM
from policies import build_policy
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_token_dataloader

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 8
N_EMBD = 512
N_HEAD = 8
FFN_MULT = 4

# Optimization
TOTAL_BATCH_SIZE = 2 ** 18
DEVICE_BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.1
GRAD_CLIP_NORM = 1.0

# DLM specifics
MASK_RATIO = 0.25
POLICY_NAME = "left_to_right"
REVEAL_PER_STEP = 8

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
mask_token_id = tokenizer.get_mask_token_id()

config = DLMConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    ffn_mult=FFN_MULT,
    dropout=0.0,
)
model = MaskedDLM(config).to(device)
model = torch.compile(model, dynamic=False)
policy = build_policy(POLICY_NAME)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

train_loader = make_token_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, epoch = next(train_loader)

num_params = sum(p.numel() for p in model.parameters())
print(f"Vocab size: {vocab_size:,}")
print(f"Policy: {policy.name}, reveal_per_step={REVEAL_PER_STEP}")
print(f"Num params: {num_params / 1e6:.1f}M")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def make_masked_batch(clean_tokens, mask_id, mask_ratio):
    rand = torch.rand_like(clean_tokens, dtype=torch.float32)
    masked_positions = rand < mask_ratio
    x_masked = clean_tokens.clone()
    x_masked[masked_positions] = mask_id
    return x_masked, masked_positions


t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        x_masked, masked_pos = make_masked_batch(x, mask_token_id, MASK_RATIO)
        with autocast_ctx:
            logits = model(x_masked)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                reduction="none",
            ).view_as(x)
            denom = masked_pos.sum().clamp_min(1)
            loss = (loss_flat * masked_pos).sum() / denom

        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LR * lrm

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if (not torch.isfinite(train_loss)) or train_loss_f > 1e4:
        print("FAIL")
        raise SystemExit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE

model.eval()
with autocast_ctx:
    val_bpb_duel = evaluate_duel_bpb(
        model=model,
        tokenizer=tokenizer,
        batch_size=DEVICE_BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        policy=policy,
        reveal_per_step=REVEAL_PER_STEP,
    )

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb_duel:     {val_bpb_duel:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
