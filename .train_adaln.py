"""
Test AdaLN timestep conditioning on our existing DLLM recipe.
Same loss (CE + 1/t weighting), same architecture, but AdaLN replaces t_proj.

Usage: CUDA_VISIBLE_DEVICES=1 uv run .train_adaln.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import ModernDLMConfig, ModernDLM
from prepare import MAX_SEQ_LEN, Tokenizer, make_token_dataloader

TIME_BUDGET = 1800

# ---------------------------------------------------------------------------
# Hyperparameters — same as our 1800s best but with AdaLN
# ---------------------------------------------------------------------------

DEPTH = 8
N_EMBD = 768
N_HEAD = 12
FFN_MULT = 8 / 3

TOTAL_BATCH_SIZE = 2 ** 15
DEVICE_BATCH_SIZE = 8
LR = 2.2e-3
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.99)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.0
GRAD_CLIP_NORM = 1.0

# DLM specifics
T_LO = 0.05
T_HI = 0.99
CLAMP_MIN = 0.3
SOFTCAP = 20.0

EVAL_ITERS = 200
SAMPLE_INTERVAL = 2000

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "dllm_adaln")

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

config = ModernDLMConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    ffn_mult=FFN_MULT,
    mask_token_id=mask_token_id,
    qk_norm=True,
    softcap=SOFTCAP,
    adaln=True,
    cond_dim=128,
)
raw_model = ModernDLM(config).to(device)
raw_model.init_weights()
model = torch.compile(raw_model, dynamic=False, mode="max-autotune-no-cudagraphs")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
for group in optimizer.param_groups:
    group["base_lr"] = group["lr"]

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

train_loader = make_token_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, epoch = next(train_loader)

num_params = sum(p.numel() for p in model.parameters())
print(f"Vocab size: {vocab_size:,}")
print(f"Num params: {num_params / 1e6:.1f}M")
print(f"Config: d{DEPTH} w{N_EMBD} h{N_HEAD} adaln=True cond_dim={config.cond_dim}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Grad accum steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_masked_batch(clean_tokens, mask_id):
    bsz = clean_tokens.size(0)
    t = torch.rand(bsz, 1, device=clean_tokens.device) * (T_HI - T_LO) + T_LO
    rand = torch.rand_like(clean_tokens, dtype=torch.float32)
    masked_positions = rand < t
    x_masked = clean_tokens.clone()
    x_masked[masked_positions] = mask_id
    return x_masked, masked_positions, t

# ---------------------------------------------------------------------------
# Mid-training generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mid_train_generate(model, tokenizer, val_loader, autocast_ctx):
    was_training = model.training
    model.eval()
    val_tokens, _ = next(val_loader)
    prompt_ids = val_tokens[:, :32]
    prompt_text = tokenizer.decode(prompt_ids[0].tolist())

    mask_id = tokenizer.get_mask_token_id()
    seq_len = 128
    z = torch.full((1, seq_len), mask_id, dtype=torch.long, device=prompt_ids.device)
    z[0, :32] = prompt_ids[0]

    n_masked = seq_len - 32
    steps = 32
    reveal_per_step = max(1, n_masked // steps)
    for _ in range(steps):
        mask_index = (z == mask_id)
        if not mask_index.any():
            break
        with autocast_ctx:
            logits = model(z)
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask_index, x0, z)
        conf = torch.where(mask_index, x0_conf, torch.tensor(-float("inf"), device=z.device))
        k = min(reveal_per_step, int(mask_index.sum().item()))
        if k > 0:
            _, top_idx = torch.topk(conf[0], k=k)
            z[0, top_idx] = x0[0, top_idx]

    output_text = tokenizer.decode(z[0].tolist())
    print(f"\n  [sample] prompt: {prompt_text[:80]}...")
    print(f"           output: {output_text[:150]}...")
    if was_training:
        model.train()

sample_val_loader = make_token_dataloader(tokenizer, 1, MAX_SEQ_LEN, "val")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        x_masked, masked_pos, t = make_masked_batch(x, mask_token_id)
        # Pass t to model for AdaLN conditioning
        t_squeezed = t.squeeze(-1)  # (B,)
        torch.compiler.cudagraph_mark_step_begin()
        with autocast_ctx:
            logits = model(x_masked, t=t_squeezed)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                reduction="none",
            ).view_as(x)
            loss = (loss_flat * masked_pos.float() / t.clamp(min=CLAMP_MIN)).sum() / masked_pos.sum().clamp_min(1)

        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", LR)
        group["lr"] = base_lr * lrm

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

    if step > 0 and step % SAMPLE_INTERVAL == 0:
        _mid_train_generate(raw_model, tokenizer, sample_val_loader, autocast_ctx)

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

val_loader = make_token_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
model.eval()
val_losses = []
with torch.no_grad():
    for _ in range(EVAL_ITERS):
        vx, _ = next(val_loader)
        vx_masked, v_masked_pos, vt = make_masked_batch(vx, mask_token_id)
        vt_squeezed = vt.squeeze(-1)
        with autocast_ctx:
            vlogits = model(vx_masked, t=vt_squeezed)
            vloss_flat = F.cross_entropy(
                vlogits.view(-1, vlogits.size(-1)),
                vx.view(-1),
                reduction="none",
            ).view_as(vx)
            vloss = (vloss_flat * v_masked_pos.float() / vt.clamp(min=CLAMP_MIN)).sum() / v_masked_pos.sum().clamp_min(1)
        val_losses.append(vloss.item())
val_loss = sum(val_losses) / len(val_losses)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"adaln:            True")

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
torch.save({
    "model_state_dict": raw_model.state_dict(),
    "config": asdict(config),
    "val_loss": val_loss,
    "num_steps": step,
    "total_tokens": total_tokens,
}, ckpt_path)
print(f"Checkpoint saved to {ckpt_path}")

# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------

gc.enable()
raw_model.eval()

gen_seq_len = 256
prompt_len = 32
num_prompted = 5
num_unconditional = 3
results = []
sample_id = 0

val_loader2 = make_token_dataloader(tokenizer, 1, MAX_SEQ_LEN, "val")

print(f"\nGenerating {num_prompted} prompted + {num_unconditional} unconditional samples...")
for i in range(num_prompted):
    val_tokens, _ = next(val_loader2)
    prompt_ids = val_tokens[:, :prompt_len]
    prompt_text = tokenizer.decode(prompt_ids[0].tolist())

    mask_id = tokenizer.get_mask_token_id()
    z = torch.full((1, gen_seq_len), mask_id, dtype=torch.long, device=device)
    z[0, :prompt_len] = prompt_ids[0]

    n_masked = gen_seq_len - prompt_len
    steps = 64
    reveal_per_step = max(1, n_masked // steps)
    with torch.no_grad():
        for _ in range(steps):
            mask_index = (z == mask_id)
            if not mask_index.any():
                break
            with autocast_ctx:
                logits = raw_model(z)
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index, x0, z)
            conf = torch.where(mask_index, x0_conf, torch.tensor(-float("inf"), device=z.device))
            k = min(reveal_per_step, int(mask_index.sum().item()))
            if k > 0:
                _, top_idx = torch.topk(conf[0], k=k)
                z[0, top_idx] = x0[0, top_idx]

    generated_text = tokenizer.decode(z[0].tolist())
    results.append({"id": sample_id, "type": "prompted", "prompt": prompt_text, "generated": generated_text})
    print(f"  [{sample_id}] prompt: {prompt_text[:60]}...")
    print(f"       output: {generated_text[:120]}...")
    sample_id += 1

for i in range(num_unconditional):
    z = torch.full((1, gen_seq_len), mask_id, dtype=torch.long, device=device)
    steps = 64
    reveal_per_step = max(1, gen_seq_len // steps)
    with torch.no_grad():
        for _ in range(steps):
            mask_index = (z == mask_id)
            if not mask_index.any():
                break
            with autocast_ctx:
                logits = raw_model(z)
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index, x0, z)
            conf = torch.where(mask_index, x0_conf, torch.tensor(-float("inf"), device=z.device))
            k = min(reveal_per_step, int(mask_index.sum().item()))
            if k > 0:
                _, top_idx = torch.topk(conf[0], k=k)
                z[0, top_idx] = x0[0, top_idx]

    generated_text = tokenizer.decode(z[0].tolist())
    results.append({"id": sample_id, "type": "unconditional", "prompt": "", "generated": generated_text})
    print(f"  [{sample_id}] unconditional: {generated_text[:120]}...")
    sample_id += 1

out_path = os.path.join(OUTPUT_DIR, "generations.jsonl")
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Saved {len(results)} generations to {out_path}")
