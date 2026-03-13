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

import duel_eval as _duel_eval
from duel_eval import evaluate_duel_bpb
from model import ModernDLMConfig, ModernDLM
from policies import build_policy
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_token_dataloader
_duel_eval.EVAL_TOKENS = 10 * 16 * MAX_SEQ_LEN  # override: ~327K tokens for tractable eval

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 6
N_EMBD = 512
N_HEAD = 8
FFN_MULT = 8 / 3  # SwiGLU param-matched to 4x GELU MLP

# Optimization
TOTAL_BATCH_SIZE = 2 ** 15
DEVICE_BATCH_SIZE = 16
LR = 1.2e-3
WEIGHT_DECAY = 0.15
BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.0
GRAD_CLIP_NORM = 1.0

# Optimizer choice:
# - "adamw": default behavior
# - "fused_adamw": AdamW fused CUDA kernels
# - "muon_adamw": Muon on matrix params + AdamW on embedding/scalar params
OPTIMIZER_NAME = "adamw"

# Muon + AdamW knobs (used only with OPTIMIZER_NAME="muon_adamw")
MATRIX_LR = 0.02
EMBEDDING_LR = 1.0
UNEMBEDDING_LR = 0.004
SCALAR_LR = 0.5
MUON_MOMENTUM = 0.95
MUON_BETA2 = 0.95
MUON_NS_STEPS = 5

# DLM specifics
MASK_RATIO = "random"  # "random" = LLaDA-style t~U[eps,1], or float for fixed ratio
POLICY_NAME = "confidence_first"
REVEAL_PER_STEP = 4

# ---------------------------------------------------------------------------
# Optimizer implementations
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for matrix params, AdamW for non-matrix params."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def _step_adamw(self, group):
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq / bias2).sqrt().add_(eps)
            if wd != 0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-(lr / bias1))

    @torch.no_grad()
    def _step_muon(self, group):
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        shape = params[0].shape
        for p in params:
            if p.shape != shape:
                raise ValueError("Muon group expects same-shaped params")

        state = self.state[params[0]]
        if (not state) or state.get("num_params") != len(params):
            state["num_params"] = len(params)
            state["momentum_buffer"] = torch.zeros(
                len(params), *shape, dtype=params[0].dtype, device=params[0].device
            )
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=params[0].dtype, device=params[0].device
            )

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack([p.detach() for p in params])
        momentum_buffer = state["momentum_buffer"]
        second_momentum_buffer = state["second_momentum_buffer"]

        momentum = group["momentum"]
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp(momentum_buffer, momentum)

        x = g.bfloat16()
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        coeffs = POLAR_EXPRESS_COEFFS[: group["ns_steps"]]
        if g.size(-2) > g.size(-1):
            for a, b, c in coeffs:
                a_mat = x.mT @ x
                b_mat = b * a_mat + c * (a_mat @ a_mat)
                x = a * x + x @ b_mat
        else:
            for a, b, c in coeffs:
                a_mat = x @ x.mT
                b_mat = b * a_mat + c * (a_mat @ a_mat)
                x = a * x + b_mat @ x
        g = x.to(stacked_params.dtype)

        beta2 = group["beta2"]
        if beta2 is not None:
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
            red_dim_size = g.size(red_dim)
            v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
            v_norm = v_norm_sq.sqrt()
            second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
            scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
            v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
            final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
            g = g * final_scale.to(g.dtype)

        lr = group["lr"]
        wd = group["weight_decay"]
        if wd != 0:
            mask = (g * stacked_params) >= 0
            stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
        else:
            stacked_params.sub_(lr * g)

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer group kind: {group['kind']}")


def _split_params_for_muon(model):
    seen = set()
    embedding_params = []
    unembedding_params = []
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith("token_embed"):
            embedding_params.append(p)
        elif name.startswith("lm_head"):
            unembedding_params.append(p)
        elif p.ndim >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    return embedding_params, unembedding_params, matrix_params, scalar_params


def build_optimizer(model):
    if OPTIMIZER_NAME == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
        return opt

    if OPTIMIZER_NAME == "fused_adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
        return opt

    if OPTIMIZER_NAME == "muon_adamw":
        embedding_params, unembedding_params, matrix_params, scalar_params = _split_params_for_muon(model)
        param_groups = []
        if unembedding_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=unembedding_params,
                    lr=UNEMBEDDING_LR,
                    base_lr=UNEMBEDDING_LR,
                    betas=BETAS,
                    eps=ADAM_EPS,
                    weight_decay=0.0,
                )
            )
        if embedding_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=embedding_params,
                    lr=EMBEDDING_LR,
                    base_lr=EMBEDDING_LR,
                    betas=BETAS,
                    eps=ADAM_EPS,
                    weight_decay=0.0,
                )
            )
        if scalar_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=scalar_params,
                    lr=SCALAR_LR,
                    base_lr=SCALAR_LR,
                    betas=BETAS,
                    eps=ADAM_EPS,
                    weight_decay=0.0,
                )
            )
        for shape in sorted({tuple(p.shape) for p in matrix_params}):
            group_params = [p for p in matrix_params if tuple(p.shape) == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=MATRIX_LR,
                    base_lr=MATRIX_LR,
                    momentum=MUON_MOMENTUM,
                    beta2=MUON_BETA2,
                    weight_decay=WEIGHT_DECAY,
                    ns_steps=MUON_NS_STEPS,
                )
            )
        return MuonAdamW(param_groups)

    raise ValueError(f"Unknown OPTIMIZER_NAME={OPTIMIZER_NAME!r}")

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
    softcap=25.0,
)
model = ModernDLM(config).to(device)
model.init_weights()
model = torch.compile(model, dynamic=False, mode="max-autotune")
policy = build_policy(POLICY_NAME)
optimizer = build_optimizer(model)

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
print(f"Optimizer: {OPTIMIZER_NAME}")


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
        t_lo, t_hi = 1e-3, 1.0
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
                # 1/t weighting (matches dllm repo LinearAlphaScheduler)
                loss = (loss_flat * masked_pos.float() / t).sum() / masked_pos.sum().clamp_min(1)
            else:
                denom = masked_pos.sum().clamp_min(1)
                loss = (loss_flat * masked_pos).sum() / denom

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

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE

EVAL_BATCH_SIZE = DEVICE_BATCH_SIZE  # match training batch to avoid recompilation

model.eval()
with autocast_ctx:
    val_bpb_duel = evaluate_duel_bpb(
        model=model,
        tokenizer=tokenizer,
        batch_size=EVAL_BATCH_SIZE,
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
