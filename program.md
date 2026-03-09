# autoresearch-dllm

This is the diffusion-language-model version of `autoresearch`. You are an autonomous researcher running experiments on a masked diffusion language model. Your job is to minimize `val_bpb_duel` — the exact DUEL validation bits per byte under a deterministic unmasking policy.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch-dllm/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-dllm/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader. Do not modify.
   - `train.py` — the main file you modify. Training loop, hyperparameters, optimizer.
   - `model.py` — model architectures. You may modify this too.
   - `policies.py` — deterministic unmasking policies. Do not modify within a single leaderboard family.
   - `duel_eval.py` — exact DUEL likelihood evaluator. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch_dllm/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a **single GPU**. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — training loop, hyperparameters, optimizer, loss function, masking strategy, batch size, model size, etc.
- Modify `model.py` — model architecture, layers, activations, initialization, etc. Two architectures are available: `MaskedDLM` (basic) and `ModernDLM` (LLaDA-style with RMSNorm, RoPE, SwiGLU).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `duel_eval.py`. The DUEL evaluator is the ground truth metric.
- Modify `policies.py` within a single leaderboard family. The deterministic unmasking policy must stay fixed for comparability.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the lowest `val_bpb_duel`.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the masking strategy, the loss weighting. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful `val_bpb_duel` gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 `val_bpb_duel` improvement that adds 20 lines of hacky code? Probably not worth it. A 0.01 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Key DLM concepts

This is not a GPT. The model is a **masked denoiser** (bidirectional transformer). Key differences from autoregressive `autoresearch`:

- **Training**: randomly mask tokens, predict the originals at masked positions. The masking ratio and loss weighting strategy are important knobs.
- **Evaluation**: DUEL exact likelihood — start fully masked, iteratively reveal tokens using a deterministic policy, accumulate log-probabilities. This is the metric you are optimizing.
- **Model**: bidirectional attention (no causal mask). Can use RoPE or learned positions. `model.py` has two variants: `MaskedDLM` (basic) and `ModernDLM` (LLaDA-style).

**Important research directions** (but use your own judgment):
- LLaDA-style random mask ratio `t ~ U[0,1]` with `1/t` loss weighting (proper ELBO bound)
- Switching from `MaskedDLM` to `ModernDLM` (RMSNorm, RoPE, SwiGLU)
- Note: `ModernDLM` has softcap built into `forward()`, so remove any external softcap in `train.py` if you switch to it.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb_duel:     19.261930
training_seconds: 300.4
total_seconds:    316.8
peak_vram_mb:     8830.4
total_tokens_M:   171.2
num_steps:        653
num_params_M:     29.7
depth:            8
```

**Important**: the script uses `\r` (carriage return) for in-progress logging, so the log file has very few actual newlines. To extract metrics from the log file, use:

```bash
tr '\r' '\n' < run.log | grep "^val_bpb_duel:\|^peak_vram_mb:"
```

Do NOT use plain `grep` on `run.log` — it will fail because the training lines are `\r`-separated.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb_duel	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val_bpb_duel` achieved (e.g. 19.261930) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.6 — divide `peak_vram_mb` by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb_duel	memory_gb	status	description
a1b2c3d	19.261930	8.6	keep	baseline
b2c3d4e	15.430000	8.8	keep	random mask ratio + 1/t weighting
c3d4e5f	16.100000	8.6	discard	switch to GELU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-dllm/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `train.py` (and/or `model.py`) with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `tr '\r' '\n' < run.log | grep "^val_bpb_duel:\|^peak_vram_mb:"`
6. If the grep output is empty, the run crashed. Run `tail -c 5000 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
7. Record the results in the tsv
8. If `val_bpb_duel` improved (lower), you "advance" the branch, keeping the git commit
9. If `val_bpb_duel` is equal or worse, you `git reset --hard` back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes, explore different masking strategies, try different optimizers. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
