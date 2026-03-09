# autoresearch-dllm

This is the diffusion-language-model version of `autoresearch`.

## Setup

1. Create a branch `autoresearch-dllm/<tag>` for a fresh run.
2. Read `README.md`, `prepare.py`, `train.py`, `model.py`, `policies.py`, `duel_eval.py`.
3. Verify data + tokenizer exist under `~/.cache/autoresearch_dllm/`; otherwise run:
   - `uv run prepare.py`
4. Initialize `results.tsv` with:

```tsv
commit	val_bpb_duel	memory_gb	status	description
```

## Constraints

- Modify only `train.py` during experiment iterations.
- Do not modify `prepare.py`, `policies.py`, or `duel_eval.py` while benchmarking a single leaderboard family.
- Keep deterministic unmask policy fixed for comparability.

## Goal

Minimize `val_bpb_duel` under fixed `TIME_BUDGET` (5 minutes).

## Run command

```bash
uv run train.py > run.log 2>&1
```

Extract metrics:

```bash
grep "^val_bpb_duel:\|^peak_vram_mb:" run.log
```

## Loop

1. Edit `train.py` for one idea.
2. Commit.
3. Run.
4. Log results in `results.tsv`.
5. Keep only improvements; revert regressions.
