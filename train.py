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
from model import ModernDLMConfig, ModernDLM
from policies import build_policy
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_token_dataloader

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 5
N_EMBD = 512
N_HEAD = 8
FFN_MULT = 8 / 3  # SwiGLU param-matched to 4x GELU MLP

# Optimization
TOTAL_BATCH_SIZE = 2 ** 15
DEVICE_BATCH_SIZE = 64
ADAMW_LR = 2.0e-3
MUON_LR = 0.02
WEIGHT_DECAY = 0.05
MUON_WEIGHT_DECAY = 0.15
BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.0
GRAD_CLIP_NORM = 1.0

# DLM specifics
MASK_RATIO = "random"  # "random" = LLaDA-style t~U[eps,1], or float for fixed ratio
POLICY_NAME = "confidence_first"
REVEAL_PER_STEP = 1

# ---------------------------------------------------------------------------
# MuonAdamW optimizer (from autoresearch, adapted for DLM)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group.get('eps', 1e-8))
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group.get("beta2", 0.85))
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group.get("ns_steps", 5), red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


def setup_muon_adamw(model, adamw_lr, muon_lr, adamw_wd, muon_wd, betas):
    """Set up MuonAdamW: Muon for 2D attention/FFN weights, AdamW for rest."""
    adamw_params = []
    matrix_params_by_shape = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D weight matrices in attention and FFN use Muon
        if p.ndim == 2 and any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                      'gate_proj', 'up_proj', 'down_proj']):
            key = p.shape
            if key not in matrix_params_by_shape:
                matrix_params_by_shape[key] = []
            matrix_params_by_shape[key].append(p)
        else:
            adamw_params.append(p)

    param_groups = [
        dict(kind='adamw', params=adamw_params, lr=adamw_lr, betas=betas, weight_decay=adamw_wd),
    ]
    for shape in sorted(matrix_params_by_shape.keys()):
        param_groups.append(dict(
            kind='muon', params=matrix_params_by_shape[shape], lr=muon_lr,
            momentum=0.95, ns_steps=5, beta2=0.85, weight_decay=muon_wd,
        ))

    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer

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
    softcap=20.0,
)
model = ModernDLM(config).to(device)
model.init_weights()
model = torch.compile(model, dynamic=False, mode="max-autotune")
policy = build_policy(POLICY_NAME)
optimizer = setup_muon_adamw(model, ADAMW_LR, MUON_LR, WEIGHT_DECAY, MUON_WEIGHT_DECAY, BETAS)

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
    bsz = clean_tokens.size(0)
    if mask_ratio == "random":
        t_lo, t_hi = 0.05, 0.99
        t = torch.rand(bsz, 1, device=clean_tokens.device) * (t_hi - t_lo) + t_lo
        rand = torch.rand_like(clean_tokens, dtype=torch.float32)
        masked_positions = rand < t
    else:
        rand = torch.rand_like(clean_tokens, dtype=torch.float32)
        masked_positions = rand < mask_ratio
        t = None
    x_masked = clean_tokens.clone()
    x_masked[masked_positions] = mask_id
    return x_masked, masked_positions, t


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        x_masked, masked_pos, t = make_masked_batch(x, mask_token_id, MASK_RATIO)
        with autocast_ctx:
            logits = model(x_masked)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                reduction="none",
            ).view_as(x)
            if t is not None:
                # Capped 1/t weighting: CE / max(t, 0.3), capping at ~3.3x
                loss = (loss_flat * masked_pos.float() / t.clamp(min=0.3)).sum() / (x.size(0) * x.size(1))
            else:
                denom = masked_pos.sum().clamp_min(1)
                loss = (loss_flat * masked_pos).sum() / denom

        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum

    # No grad clip for Muon (implicit normalization), but clip AdamW params
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
