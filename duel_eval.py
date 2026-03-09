import math

import torch
import torch.nn.functional as F

from prepare import EVAL_TOKENS, get_token_bytes, make_token_dataloader


@torch.no_grad()
def duel_logprob_batch(model, x, mask_token_id, policy, reveal_per_step):
    z = torch.full_like(x, mask_token_id)
    masked = torch.ones_like(x, dtype=torch.bool)
    total_logprob = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)

    while masked.any():
        logits = model(z)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        selected = policy.select_positions(masked, probs, reveal_per_step)
        for b in range(x.size(0)):
            pos = selected[b]
            valid = pos[pos >= 0]
            if valid.numel() == 0:
                continue
            tgt = x[b, valid]
            total_logprob[b] += log_probs[b, valid, tgt].sum().float()
            z[b, valid] = tgt
            masked[b, valid] = False

    return total_logprob


@torch.no_grad()
def evaluate_duel_bpb(model, tokenizer, batch_size, seq_len, policy, reveal_per_step):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_token_dataloader(tokenizer, batch_size, seq_len, "val")
    steps = max(1, EVAL_TOKENS // (batch_size * seq_len))
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, _ = next(val_loader)
        logprob = duel_logprob_batch(model, x, tokenizer.get_mask_token_id(), policy, reveal_per_step)
        nbytes = token_bytes[x]
        total_nats += (-logprob).sum().item()
        total_bytes += nbytes.sum().item()

    return total_nats / (math.log(2) * total_bytes)
