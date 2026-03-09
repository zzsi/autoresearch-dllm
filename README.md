# autoresearch-dllm

Small, single-GPU autonomous research loop for deterministic diffusion language models (DLMs), inspired by Andrej Karpathy's `autoresearch`.

The key difference from autoregressive GPT:

- model: masked denoiser (bidirectional transformer)
- generation/evaluation: deterministic unmasking policy
- metric: exact DUEL validation bits per byte (`val_bpb_duel`)

## Repository structure

- `prepare.py` — fixed constants, data prep, tokenizer, dataloaders
- `train.py` — the file to iterate on in experiments
- `model.py` — tiny masked DLM backbone
- `policies.py` — deterministic unmasking policies
- `duel_eval.py` — exact DUEL likelihood and BPB evaluator
- `program.md` — baseline autonomous loop instructions

## Quick start

```bash
# install dependencies
uv sync

# download shards + train tokenizer
uv run prepare.py

# run one 5-minute experiment
uv run train.py
```

## Main metric

`val_bpb_duel` is computed from exact log-likelihood under the configured deterministic unmasking process:

- same held-out validation shard every run
- same policy + tie-break every run
- same wall-clock training budget every run

Lower is better.
