"""
Generate SEDD samples matching GPT reference format (prompted + unconditional).
Uses the same val prompts as the GPT reference.

Usage: CUDA_VISIBLE_DEVICES=1 uv run .gen_sedd_samples.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, make_token_dataloader

CHECKPOINT = "outputs/sedd_small/checkpoint.pt"
OUTPUT_PATH = "outputs/sedd_reference.jsonl"
NUM_PROMPTED = 10
NUM_UNCONDITIONAL = 5
PROMPT_LEN = 32
GEN_SEQ_LEN = 256
SAMPLING_STEPS = 256
SAMPLING_EPS = 1e-3

# ---------------------------------------------------------------------------
# Import model classes from train script (exec the class definitions only)
# ---------------------------------------------------------------------------

code = open(".train_sedd.py").read()
start = code.index("# LogLinear Noise")
end = code.index("# ---------------------------------------------------------------------------\n# Mid-training generation")
exec(code[start:end])

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

graph = AbsorbingGraph(vocab_size)
noise_schedule = LogLinearNoise(eps=SAMPLING_EPS)

# Load checkpoint
ckpt = torch.load(CHECKPOINT, map_location=device)
cfg = ckpt["config"]
model = SEDDModel(
    vocab_size=cfg["vocab_size"],
    n_embd=cfg["n_embd"],
    n_head=cfg["n_head"],
    n_layer=cfg["n_layer"],
    cond_dim=cfg["cond_dim"],
    mlp_ratio=cfg["mlp_ratio"],
    seq_len=cfg["seq_len"],
    original_arch=cfg.get("original_arch", False),
    dropout=0.0,  # no dropout for inference
).to(device)

# Load EMA weights
if "ema_state" in ckpt:
    for name, p in model.named_parameters():
        if name in ckpt["ema_state"]:
            p.data.copy_(ckpt["ema_state"][name])
else:
    model.load_state_dict(ckpt["model_state_dict"])

model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")
print(f"  Steps: {ckpt.get('num_steps')}, Val loss: {ckpt.get('val_loss', 'N/A')}")
print(f"  Config: d{cfg['n_layer']} w{cfg['n_embd']} h{cfg['n_head']}")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

val_loader = make_token_dataloader(tokenizer, 1, MAX_SEQ_LEN, "val")
results = []
sample_id = 0

print(f"\nGenerating {NUM_PROMPTED} prompted samples (prompt={PROMPT_LEN}, len={GEN_SEQ_LEN}, steps={SAMPLING_STEPS})...")

for i in range(NUM_PROMPTED):
    val_tokens, _ = next(val_loader)
    prompt_ids = val_tokens[0, :PROMPT_LEN]
    prompt_text = tokenizer.decode(prompt_ids.tolist())

    # Start from all-mask, fill in prompt
    mask_id = graph.mask_id
    x = torch.full((1, GEN_SEQ_LEN), mask_id, dtype=torch.long, device=device)
    x[0, :PROMPT_LEN] = prompt_ids

    # PC sampler with analytic predictor
    timesteps = torch.linspace(1, SAMPLING_EPS, SAMPLING_STEPS + 1, device=device)
    with torch.no_grad():
        for si in range(SAMPLING_STEPS):
            t = timesteps[si]
            t_next = timesteps[si + 1]
            sigma_cur = noise_schedule(torch.tensor([t], device=device))[0]
            sigma_next = noise_schedule(torch.tensor([t_next], device=device))[0]
            dsigma = sigma_cur - sigma_next
            sigma_batch = sigma_cur.expand(1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_score = model(x, sigma_batch)
            score = log_score.float().exp()
            stag_score = graph.staggered_score(score, dsigma)
            transp_trans = graph.transp_transition(x, dsigma)
            probs = stag_score * transp_trans

            # Don't resample prompt positions
            prompt_mask = torch.zeros_like(x, dtype=torch.bool)
            prompt_mask[0, :PROMPT_LEN] = True
            sampled = _sample_categorical(probs)
            x = torch.where(prompt_mask, x, sampled)

        # Final denoising
        t_final = timesteps[-1]
        sigma_final = noise_schedule(torch.tensor([t_final], device=device))[0]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_score = model(x, sigma_final.expand(1))
        score = log_score.float().exp()
        stag_score = graph.staggered_score(score, sigma_final)
        transp_trans = graph.transp_transition(x, sigma_final)
        probs = stag_score * transp_trans
        probs = probs[..., :-1]  # remove absorbing state
        sampled = _sample_categorical(probs)
        x = torch.where(prompt_mask, x, sampled)

    x = x.clamp(0, vocab_size - 1)
    generated_text = tokenizer.decode(x[0].tolist())
    continuation = generated_text[len(prompt_text):]

    results.append({
        "id": sample_id, "type": "prompted", "prompt": prompt_text,
        "generated": continuation, "num_tokens": GEN_SEQ_LEN,
    })
    print(f"  [{sample_id}] prompt: {prompt_text[:60]}...")
    print(f"       output: {continuation[:120]}...")
    sample_id += 1

print(f"\nGenerating {NUM_UNCONDITIONAL} unconditional samples...")

for i in range(NUM_UNCONDITIONAL):
    x = torch.full((1, GEN_SEQ_LEN), graph.mask_id, dtype=torch.long, device=device)

    timesteps = torch.linspace(1, SAMPLING_EPS, SAMPLING_STEPS + 1, device=device)
    with torch.no_grad():
        for si in range(SAMPLING_STEPS):
            t = timesteps[si]
            t_next = timesteps[si + 1]
            sigma_cur = noise_schedule(torch.tensor([t], device=device))[0]
            sigma_next = noise_schedule(torch.tensor([t_next], device=device))[0]
            dsigma = sigma_cur - sigma_next

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_score = model(x, sigma_cur.expand(1))
            score = log_score.float().exp()
            stag_score = graph.staggered_score(score, dsigma)
            transp_trans = graph.transp_transition(x, dsigma)
            probs = stag_score * transp_trans
            x = _sample_categorical(probs)

        # Final denoising
        t_final = timesteps[-1]
        sigma_final = noise_schedule(torch.tensor([t_final], device=device))[0]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_score = model(x, sigma_final.expand(1))
        score = log_score.float().exp()
        stag_score = graph.staggered_score(score, sigma_final)
        transp_trans = graph.transp_transition(x, sigma_final)
        probs = stag_score * transp_trans
        probs = probs[..., :-1]
        x = _sample_categorical(probs)

    x = x.clamp(0, vocab_size - 1)
    generated_text = tokenizer.decode(x[0].tolist())

    results.append({
        "id": sample_id, "type": "unconditional", "prompt": "",
        "generated": generated_text, "num_tokens": GEN_SEQ_LEN,
    })
    print(f"  [{sample_id}] {generated_text[:150]}...")
    sample_id += 1

with open(OUTPUT_PATH, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved {len(results)} samples to {OUTPUT_PATH}")
