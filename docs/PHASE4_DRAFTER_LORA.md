# Phase 4: a LoRA on the drafter (SDAR-1.7B), for rd1 / rd2

Branch `phase4-drafter-lora`, cut from `phase3-redraft-8gpu` (which contains everything on `phase2-drafter-swap`).
Source of the design: the *SDAR Drafter LoRA Runbook* page (12 Oct 2026). Until now only the judge (the 128M LoRA on
the 30B) was ever trained. Phase 4 trains the drafter too, so that the drafter writes what the 30B writes. The judge,
its LoRA and its 0.95 commit rule stay frozen.

No result exists yet. Every speed or time number in this file is the runbook's projection, not a measurement.

## 1. What is new, and where it lives

| file | what it is |
|---|---|
| `code/drafter_lora/lora.py` | LoRA r 32 / alpha 64 on q, k, v, o, gate, up, down of all 28 layers (no `peft`; the repo's `adapter.LoRALinear`). 34,865,152 trainable parameters, asserted. `adapter_off` = the original drafter q_0 (bit-identical to the base). `merge_into_base` / `save_merged` write the full HF directory that `run_da_decode.py --drafter` loads |
| `code/drafter_lora/data.py` | corpus waves A + B, the 3,000-item stratified held-out split, the training states (40% first-draft, 60% re-draft), the two-stream layout and mask, the one-pass plan |
| `code/drafter_lora/forward.py` | the two-stream forward (explicit 4D mask through SDPA) and the loss |
| `code/drafter_lora/train_drafter_lora.py` | **the trainer** (torchrun, 8 GPUs, manual all-reduce) |
| `code/drafter_lora/replay.py` | s8 first-draft replay of a corpus item and the re-draft probe (the 25% check, Gate 1, smoke parity) |
| `code/drafter_lora/eval_agree.py` | **Gate 1** + the smoke parity check; dumps the trained drafts for Gate 2 |
| `code/drafter_lora/judge_probe.py` | **Gate 2**: the frozen judge's commit share and precision, stored vs trained drafts |
| `code/drafter_lora/compare_gates.py` | **Gates 3 and 4**: new rd1 / rd2 decodes vs Phase 3 (and vanilla), paired by item |
| `code/drafter_lora/smoke_check.py` | the smoke test's verdict |
| `code/drafter_lora/check_merged.py` | the merged drafter loads with `AutoModelForCausalLM` and `run_da_decode.py --drafter` takes it |
| `code/drafter_lora/selfcheck.py` | the real SDAR-1.7B on real corpus items: count, two-stream == remote forward, KL 0, one step, replay parity, merge round trip (`p4.sh setup` runs it on GPU 0) |
| `code/drafter_lora/hf_progress.py` | downloads with one-line progress (GB, files, MB/s, ETA), retries, size check; used by `fetch_p4.py` and by `setup_p2.sh` for the two models |
| `code/drafter_lora/fetch_p4.py` | corpus + Phase 3 + vanilla records from Hugging Face, with the runbook's count checks |
| `code/drafter_lora/backup_p4.py` | allowlisted upload to `$P4_HF_REPO` after a secret scan |
| `code/drafter_lora/p4.sh` | one command per stage: `setup smoke train gate1 gate2 gate3 gate4 backup` |
| `code/draftalign/{drafter,da_decode,run_da_decode}.py` | the decode-time ban: `--redraft-ban-rejected` (cell suffix `_ban`). Off by default; with it off, records, cell names and config hashes are unchanged |
| `code/tests/test_drafter_lora.py`, `code/tests/test_da_decode_redraft.py` | CPU tests (24 new) |
| `.env.example` | the keys the scripts read; copy to `.env` (git-ignored, mode 600) |

## 2. Run order (on the 8x RTX PRO 6000 box, from the repository root)

```bash
cp .env.example .env && chmod 600 .env        # fill in HF_TOKEN, WANDB_*, (GITHUB_TOKEN for pushes)
bash code/drafter_lora/p4.sh setup   # setup_p3_8gpu.sh (16/16 checks) + fetch_p4.py + CPU tests + sm_120 check
bash code/drafter_lora/p4.sh smoke   # STOP unless: 34,865,152 params, KL 0 at step 1, loss falls, parity >= 95%, pass <= 6 h
bash code/drafter_lora/p4.sh train   # one pass; stops itself at 25% if agreement gains < +1.0 point
bash code/drafter_lora/p4.sh gate1   # first-draft agreement >= +3.0 points, retention >= 97% (+ merge fidelity check)
bash code/drafter_lora/p4.sh gate2   # judge commit precision within 0.5 points
bash code/drafter_lora/p4.sh gate3   # rd1 + ban on items[k::32], k = 0..7: >= 5% fewer judge passes, CI covers 0, loops not up
bash code/drafter_lora/p4.sh gate4   # rd1 and rd2 + ban, all 2,250 items
nohup /venv/p2/bin/python code/drafter_lora/backup_p4.py --loop 30 > logs/phase4/backup.log 2>&1 &
```

Each stage exits 3 at its gate's failure and never starts the next stage itself. `touch logs/phase4/STOP` stops before
the next run. Judge retraining (runbook section 8) is not scripted; it needs approval first.

The prompt to paste into Claude Code on the rented box is `docs/PHASE4_BOX_PROMPT.md`.

### 2a. How training proceeds (what you will see)

1. **Start** (`[data]` line): both indexes load (331,818 items) and 3,000 are held out (saved to
   `holdout_pids.json`). The pass is 328,818 items in ceil(328,818 / 256) = 1,285 optimiser steps, with warmup 26
   steps and saves at every 10%. Then `[model] ... trainable 34,865,152 (attention 12,845,056 + MLP 22,020,096)`
   prints; any other count stops the run.
2. **Step-0 agreement** (`[agree step0]`): all 8 GPUs replay the s8 first draft on the held-out items with the
   adapter off. This is the untrained baseline for the 25% check. It takes minutes; the time is printed.
3. **Steps.** Each step deals its 256 items to the 8 ranks, balanced by length. Each rank packs its items into
   micro-batches of at most `--micro-tokens` (16,384) padded positions. A micro-batch runs one two-stream forward
   with gradients (q_theta), one without (adapter off, q_0), and a backward. Then gradients are all-reduced, clipped
   at 1.0, and AdamW steps. One line per step:
   `[step 120/1285] loss 1.83 ce 1.82 kl 4.1e-03 kl_share 0.47 acc 0.58 (orig 0.47) ret 0.985 gnorm 0.61 lr 8.00e-05 41.2 it/s 16500 tok/s`
   * `ce` falls; `acc` (the trained drafter's argmax == the 30B's token on hidden slots) climbs above `orig` (the
     original drafter on the same slots).
   * `kl` is exactly 0 at step 1 (printed as `[step 1] KL sum 0.000e+00`) and stays small.
   * `kl_share` is the share of hidden slots the original already gets right (≈ `orig`).
   * `ret` is the share of those the trained drafter still gets right. It should stay near 1.
   * `it/s` and `tok/s` are measured items and corpus tokens per second.
4. **Step 20, projection** (`[projection] x s/step x 1285 steps = y h`, also `speed.json`). The run stops itself if
   y exceeds `--max-hours` (6).
5. **Every 10%**: `ckpt/step_NNNNNN/` (adapter + optimizer), `ckpt/latest`.
6. **25%, the check** (`[check 25%] agreement ... vs step 0 ...: +x.xx points`). The run stops with exit 3 and
   `STOP.json` unless x >= +1.00.
7. **End**: `final_adapter/`, `merged/` (bf16 full HF directory; `[merged] ... run_da_decode.py --drafter ...`),
   and `DONE.json`.

Logs: `logs/phase4/train/main.log` (stdout) and `train.rank0.jsonl` (one JSON row per step), mirrored to W&B
(group `phase4-drafter-lora`; project `$DA_WANDB_PROJECT`, else `$WANDB_PROJECT`; `ENABLE_WANDB=0` turns it off).
If the process dies, re-run `p4.sh train`: `--resume` restarts from `ckpt/latest`, so at most 10% of the pass is
redone. Training states are a pure function of (seed, item, block), so a resumed run sees the same data.

The trainer alone:

```bash
torchrun --standalone --nproc_per_node=8 code/drafter_lora/train_drafter_lora.py \
  --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
  --out logs/phase4/train/main --resume
```

Defaults are the runbook's: `--lora-r 32 --alpha 64 --kl-weight 0.1 --p-first 0.4 --lr 8e-5 --betas 0.9,0.95
--warmup-frac 0.02 --min-lr-frac 0.1 --clip 1.0 --items-per-step 256 --holdout 3000 --check-at 0.25
--check-min-delta 0.01 --max-hours 6`. Outputs are in `logs/phase4/train/main/`: `config.json`, `model_info.json`,
`holdout_pids.json`, `train.rank0.jsonl`, `speed.json`, `check_25pct.json`, `ckpt/step_NNNNNN/` (every 10%, `latest`),
`final_adapter/`, `merged/`, and `STOP.json` when a check stopped the run.

## 3. The loss, as implemented

H = the valid [MASK] slots of the micro-batch. T_i = the 30B's token. q_0 = the same forward with the adapter off,
under no_grad.

    L = (1/|H|) sum_{i in H} -log q_theta,i(T_i)  +  0.1 (1/|H|) sum_{i in H : argmax q_0,i = T_i} KL(q_0,i || q_theta,i)

Logits are fp32 at the loss rows only, and the [MASK] logit is -inf in every distribution. The KL is computed as
`F.kl_div(log q_theta, log q_0, log_target=True)` summed over the vocabulary. The [MASK] column is zeroed in both
log-distributions first, because exp(-inf)·(-inf − -inf) is NaN while its true contribution is 0. Each micro-batch
is normalised by its own |H|. The step averages its micro-batches over all ranks. `lora_B` starts at zero, so the KL
is exactly 0 at step 1; the trainer logs it and the smoke check asserts it.

## 4. Choices the runbook left open, and how the code resolves them

1. **No `peft`.** The forward bypasses `model.forward`, so a self-contained wrapper is simpler and needs no version
   pin against transformers 4.54.1. It is the same `LoRALinear` the judge uses: fp32 A/B, cast to bf16 inside the
   matmul.
2. **The two-stream forward is not `train_fast.py`'s.** The judge's two-stream trainer computes the prefix under
   no_grad. That is exact there only because the judge's adapter is flag-gated off on prefix rows. The drafter's
   LoRA is ungated: it changes the prefix's keys and values at inference too. So the prefix here is a *clean stream*
   with gradients: [clean prompt + blocks | every usable block in its training state]. Clean rows are block-causal;
   a noisy block sees the clean blocks before it plus itself; its position ids are its clean positions. A test pins
   it: two-stream output equals one clean-prefix block-causal forward per block.
3. **Items are keyed by `pid`.** Both waves number items from idx 0, and `StateSampler` / `group_states` key on
   (split, idx). `judge_probe.py` renumbers the held-out entries 0..n-1 before building its samplers.
4. **Held-out split.** Proportional per-domain quotas (largest remainder), numpy `default_rng(0)` over each domain's
   sorted pids, saved to `holdout_pids.json`. Every evaluation reads that file back.
5. **The replay drafts post-EOS slots.** The stored SGLang drafts fill every slot except the prompt tail. The replay
   does the same, so the parity check compares like with like. Agreement counts the corpus' `valid` slots only.
6. **Post-EOS slots in training states hold the true tokens**, exactly as `StateSampler` does (the runbook's
   instruction). This shows the drafter where the answer ended in its last block. `--post-eos mask` is the
   alternative, off (section 5).
7. **The 25% check's baseline** is the same in-memory model at step 0, with the adapter off. That is bit-identical
   to the stock drafter, and it runs through the same replay code on the same held-out items.
8. **Merge.** Gates 3–4 load a bf16-merged drafter. Merging a small delta into bf16 weights can round part of it
   away, so Gate 1 also compares adapter vs merged agreement on 300 held-out items.
9. **Reference records** download to `logs/phase4/ref/{phase3,phase2}/` (git-ignored, under `safe_out`), not `ref/`.
   The vanilla bundle `assets/ref_stock_30B/` is flat and also holds pooled (`_p48desf`) and off-split cells, so
   Gate 4 reads exactly `*_B32_t95_stock.<set>.shard*.jsonl` (checked against the Hub listing on 12 Oct).
10. **Micro-batches** are packed by padded size (`--micro-tokens 16384` positions, B × Lmax). `--micro-tokens` is the
    knob to change if the smoke test shows memory headroom or pressure.
11. **A first-draft state never reveals every valid slot.** k is drawn from {0, 4, ..., 28} as the runbook says, then
    capped at (valid slots − 1), so a short last block keeps at least one loss row; a block without a stored draft
    (`has_draft` False) gets k = 0. The runbook does not cover this edge case.
12. **Where the code differs from the runbook's box prompt, and why.** The runbook's "Build" step (trainer, ban, gate
    scripts) is already done on this branch, so the box clones `phase4-drafter-lora` instead of cutting it. Keys live
    in `/workspace/.env`, as the runbook says; `p4.sh` also reads a repo-root `.env` (git-ignored) for local use. W&B
    logging is an optional extra. The corpus is 2.3 GB on the Hub, not 11 GB. `peft` is not used: the runbook allows
    `adapter.py` instead, and `lora.py` uses its `LoRALinear`.
13. **The merged drafter is checked as the runbook asks** (`check_merged.py`, run by `p4.sh train`): it must load
    with `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, local_files_only=True)`, and
    `run_da_decode.py --drafter <dir>` must resolve it (cell `..._rd1_ban_dr<hash>`).

## 5. Settled by the runbook (12 Oct)

1. **Held-out set:** 3,000 items stratified by domain, numpy `default_rng(0)`, as the runbook says. The 128M judge
   trained on the same waves, so Gate 2's absolute precision is optimistic. The comparison is paired, so the delta
   holds.
2. **Post-EOS slots** hold the truth, as `StateSampler` does; the runbook says to treat them that way.
   `--post-eos mask` stays off.
3. **Backups** go to the HF dataset `Arushhh/phase4-drafter-lora` (created private). W&B, if enabled, uses
   `$WANDB_PROJECT`.
