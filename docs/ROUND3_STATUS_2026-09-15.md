# Round 3 — live status and every link (2026-09-15)

Updated automatically while the run is live; the last section says when.

## What is running

| item | value |
|---|---|
| run | `r3_A` — judge LoRA, fresh from stock `JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc3…`, drafter `JetLM/SDAR-1.7B-Chat-b32` @ `2c83c05d…`, **block 32** |
| launched | 2026-09-15 02:38 IST (21:08 UTC 14 Sep), all 8 GPUs, `torchrun --nproc_per_node=8`, grad-accum 4 = **32 states/step** (recipe-exact) |
| schedule | **2,175 steps = 3 epochs** of the 23,558-block index; lr 2e-5, warmup 50, cosine to 0.5; LoRA r32/α64 on q,k,v,o + draft flag; `pool_Ms 128`; holdout 150 items, eval every 200, checkpoint every 100 |
| measured | 12.9 s/step; stopped 1560 @08:16 IST, resumed from step001500; latest: `[step 2000/2175] loss 1.9457 ce 1.5408 cert 0.2025 acc 0.667 top1 0.739 uniq 50.9 M None lr 1.02e-05 115 tok/s` |
| W&B | run https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_A · project https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign |
| launcher | `code/draftalign/launch_r3_train.sh` (exact command inside) |

## The data (option B, 2026-09-14 21:45 IST)

| pool | blocks | share | items | how a decode was admitted |
|---|---:|---:|---:|---|
| code — `nvidia/OpenCodeInstruct` (`opencode_pool:0:3200`) | 8,297 | 35.2% | 2,344 | passes the row's unit tests (74% of decodes) |
| maths — `openai/gsm8k` train (`gsm8k_pool:0:2400`) | 8,400 | 35.7% | 1,410 of 2,253 correct | final answer correct (94%) |
| IF — verifiable IF pool (`if_pool:0:2000`) | 6,861 | 29.1% | 690 | strict pass on **every** constraint (35.5% of decodes) |
| **total** | **23,558** | | 4,444 | index sha256 `1c2ce419…` |

Decode = the training-free blind rule (`--draft s8 --judge blind --tau-acc-q .50`), stock 30B + stock 1.7B.
Blocks = `build_train_set.py --only-correct --max-seq 1200 --exclude-frozen`. Mix = `code/draftalign/build_r3_mix.py`.

## Where everything is (public Hugging Face)

**Processed dataset: 267 MB (253 files) under `processed/` — 23,558 training blocks (31 MB as `trainset_r3/`: index + per-pool `items.tar`) plus every decode record and per-block trace (236 MB). Whole folder incl. sources, docs, env: 355 MB.** Checkpoints add ~321 MB each (adapter 107 MB + optimizer state 214 MB).

Repo: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts
Folder: `Nemotron-data-subset-training-diffusion-moe/` →
https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-data-subset-training-diffusion-moe

| what | path in the folder |
|---|---|
| **checkpoints** (adapter.safetensors + adapter_config.json + trainer_state.pt, one dir per 100 steps) | `checkpoints/r3_A/stepNNNNNN/` |
| training index + per-pool blocks | `processed/trainset_r3/` (`index.jsonl`, `mix_manifest.json`, `<pool>/index.jsonl`, `<pool>/items.tar`) |
| decode records + traces | `processed/datagen/<cell>/` (`*.jsonl`, `traces.tar`) |
| selected source rows | `unprocessed/` (+ `SOURCES.md`) |
| recipes, docs, code as run, env freezes, `code_changes.diff` | `strategy/` |
| probe results (after training) | `processed/probes_r3/r3_A/` (`summary.jsonl`, `best.json`) |
| live status table | `README.md` in the folder |

Checkpoint links are listed in the table at the bottom of this file as they land.

## Resume on another machine

```bash
hf download Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
  --include "Nemotron-data-subset-training-diffusion-moe/processed/trainset_r3/*" \
  --include "Nemotron-split-moe-train/r3_A/checkpoints/stepNNNNNN/*" --local-dir .
# unpack <pool>/items.tar next to each <pool>/index.jsonl, then:
torchrun --standalone --nproc_per_node=8 code/draftalign/train.py --index <trainset_r3>/index.jsonl \
  --out logs/draftalign/train/r3_A --resume <stepNNNNNN dir> --steps 2175 --grad-accum 4 --pool-Ms 128 \
  --no-train-gates --draft-flag --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 --max-seq 1200 \
  --holdout-items 150 --eval-every 200 --save-every 100 --seed 1 --keep-checkpoints 0
```

Evaluate a checkpoint exactly as the deck does (setting 3/4/5): `run_da_decode.py --adapter <stepNNNNNN> --judge full --draft s8 [--pool 32|48 --pool-rule des --pool-flush]`.

## Checkpoints pushed

<!-- CKPT_TABLE_START -->
| step | HF link | pushed |
|---|---|---|
| 100 | [step000100](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000100) · sha256 `5c5bd74b30004c4a…` | yes |
| 200 | [step000200](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000200) · sha256 `9e226518d4ca2540…` | yes |
| 300 | [step000300](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000300) · sha256 `04f2d976f55be009…` | yes |
| 400 | [step000400](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000400) · sha256 `c11ff34a77519752…` | yes |
| 500 | [step000500](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000500) · sha256 `01e843e527621310…` | yes |
| 600 | [step000600](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000600) · sha256 `c28019115309a6c8…` | yes |
| 700 | [step000700](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000700) · sha256 `8f3b73a4c67818a3…` | yes |
| 800 | [step000800](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000800) · sha256 `8d878bc0b1f795a3…` | yes |
| 900 | [step000900](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step000900) · sha256 `a56e760c6f953712…` | yes |
| 1000 | [step001000](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001000) · sha256 `f1e7682e5773cf3e…` | yes |
| 1100 | [step001100](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001100) · sha256 `3aed50321f541bec…` | yes |
| 1200 | [step001200](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001200) · sha256 `2d3944ac52895257…` | yes |
| 1300 | [step001300](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001300) · sha256 `f5a11d8a29b7b9b4…` | yes |
| 1400 | [step001400](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001400) · sha256 `2e99a87bccaf2dc4…` | yes |
| 1500 | [step001500](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001500) · sha256 `8905cf3deebb9666…` | yes |
| 1600 | [step001600](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001600) · sha256 `5ebfb3497f59c18c…` | yes |
| 1700 | [step001700](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001700) · sha256 `c6aef1c9682b2b7d…` | yes |
| 1800 | [step001800](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001800) · sha256 `a36ca97e78e486f0…` | yes |
| 1900 | [step001900](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step001900) · sha256 `68785f3512e4e031…` | yes |
| 2000 | [step002000](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/step002000) · sha256 `d4e694e5f9be1981…` | yes |
<!-- CKPT_TABLE_END -->

## r3_B — calibrated retrain from epoch 1 (launched 11:48 IST)

- **11:44 IST:** r3_A stopped for good at step 2,000 (user decision). Three epochs of the r3_A loss left confident overrides only ~78–81% precise (code ~77%) and flat, while confident confirms degraded from 99.1% to 96.7%.
- **11:48 IST:** `r3_B` launched from `r3_A/step000700` (sha `8f3b73a4…`) with a fresh optimizer and lr schedule, 710 steps (= 1 epoch, so it ends at epoch 2). The data, split, LoRA, 32 states/step and seed 1 are unchanged.
- **Loss (new flags; r3_A's loss is bit-identical when they are off):**
  - `--beta 0.5` (was 2.0);
  - `--calib 1.0`, BCE(top-1 prob of the MASK-suppressed softmax, [token == target]);
  - `--override-weight 2.0` and `--code-override-weight 3.0` on slots whose draft token is wrong.
- **Holdout eval** runs every 100 steps and logs `draft_pass1_ovr_conf_precision` for checkpoint selection.
- **Checkpoints** are pushed every 100 steps: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_B
- **W&B:** https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_B
- **Calibration logging (added ~12:55 IST by resuming from step 300; training unchanged):** `--calib-log --calib-eval-states 64 --calib-eval-extra 16` (`code/draftalign/calib_metrics.py`).
  - Every 10 steps: each loss term and its cumulative mean, a confidence histogram with per-bin accuracy (ECE), and CE/acc/conf per slot kind (mask / confirm / override).
  - Every 100 steps, on the holdout: masked-position equal-mass ECE and top-label NLL vs mask ratio t (bootstrap CIs), reliability diagrams at t 0.1/0.5/0.9, and joint calibration (P(all committed tokens correct) vs product of confidences, by tokens committed per pass).
  - Outputs: `logs/draftalign/train/r3_B/calib/` and W&B `calib/*`, `calib_plots/*`.
- **Post-training eval:** `code/draftalign/eval_after_r3B.sh`. It pushes checkpoints right away, then runs judge + DES48 on the full HumanEval, MBPP, MATH-500, GPQA, GSM8K-500 and AIME sets. After that come the missing stock + DES48/32 baselines on AIME/GPQA, shortest first. Cutoff 17:30 IST; summary in `probes_r3/r3_B/eval_summary.md`.

## Run history, diagnosis and report (11:29 IST)

**Report:** https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg (method, data split, curves, evals, plateau probe).
**Split-train backup (Anupam, public):** https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A (`checkpoints/`, `train_logs/`, `evals/`, `analysis/`). Arushhh backups are stopped.

Timeline (IST, 15 Sep):
- **08:16:** stopped at step 1560 (user decision, holdout gains flat). Eval wave run on step 700 (≈ epoch 1) and step 1400 (≈ epoch 2) for HumanEval-164, MBPP-257, GSM8K500[:100] and MATH500[:50]; step-1500 evals dropped. Code scoring verified identical to earlier rounds (0 mismatches over 842 items).
- **~09:50:** evals paused, plateau probe run on 7 checkpoints, training resumed from `step001500` with the **unchanged recipe** (user decision: same strategy for uniformity, analyse after more steps).
- **11:29:** at step ~1940/2175. Server shuts down at 12:20 IST. No final-checkpoint eval is scheduled, because the user asked not to stop training.

Evals, paired against stock (ep1 → ep2):

| set | stock | judge | judge+DES48 | stock+DES48 |
|---|---:|---|---|---:|
| HumanEval | 78.0 | 68.9 → 69.5 | 67.1 → 69.5 | 78.7 |
| MBPP | 65.4 | 54.1 → 57.6 | 56.8 → 61.9 | 63.8 |
| GSM8K-100 | 93 | 90 → 88 | 90 → 90 | |
| MATH-50 | 76 | 68 → 66 | 72 → 68 (DES48) | |

Plateau diagnosis:
- Train accuracy slope from step 700 to 1560 is +0.06±0.16 pts per 100 steps. Train and holdout are both flat, so this is not overfitting.
- The certainty term keeps falling and the overconfidence gap grows (11.9 → 13.4 pts).
- Confirm commits at τ 0.95 rise from 64.9% to 69.8% of slots, while their precision drops from 99.1% to 96.7%. Override precision is about 81%.
- Code losses come from parallel commits that are each confident but jointly broken; 30–60% of the lost MBPP programs don't parse. Stock+DES keeps accuracy, so the judge's commit/override rule is the cause.
- Next lever (not applied): `--tau-override 1.0/0.99` plus a confirm-τ sweep with DES48, then a retrain with lower β and penalties on confident wrong commits.

## Probe results

<!-- PROBES_START -->
(not yet run — the wave starts after training)
<!-- PROBES_END -->

_Last update: 2026-09-15 13:55 UTC_

## Final status (19:24 IST, 15 Sep 2026)

All evals complete. Coverage (✓ = full set scored):

| model | HumanEval | MBPP | GSM8K-500 | MATH-500 | GPQA | AIME |
|---|---|---|---|---|---|---|
| stock | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| stock + DES48 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| stock + DES32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| blind accept | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **r3_B + DES48 (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **r3_B judge, no pool (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| round-2 judge + DES48 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r3_A epoch 1 (step 700) + DES48 | ✓ | ✓ | 100/500 (subset only) | 50/500 (subset only) | — | — |
| r3_A epoch 2 (step 1400) + DES48 | ✓ | ✓ | 100/500 (subset only) | 50/500 (subset only) | — | — |
| r3_A last (step 2000) + DES48 | ✓ | ✓ | — | — | — | — |

Headline, accuracy (Δ vs stock, speedup):

| set | stock | stock + DES48 | r3_B + DES48 (ours) | r3_B judge, no pool (ours) | round-2 judge + DES48 | r3_A epoch 2 + DES48 |
|---|---:|---|---|---|---|---|
| HumanEval | 78.0 | 78.7 (+0.6, 1.19×) | 75.0 (-3.0, 1.50×) | 75.0 (-3.0, 1.22×) | 62.2 (-15.9, 1.84×) | 69.5 (-8.5, 1.49×) |
| MBPP | 65.4 | 63.8 (-1.6, 1.16×) | 61.1 (-4.3, 1.38×) | 62.3 (-3.1, 1.15×) | 53.7 (-11.7, 1.25×) | 61.9 (-3.5, 1.49×) |
| GSM8K-500 | 90.2 | 91.2 (+1.0, 1.26×) | 90.8 (+0.6, 1.54×) | 90.6 (+0.4, 1.20×) | 87.8 (-2.4, 1.74×) | 90.0 (-2.0, 1.74×) n=100 |
| MATH-500 | 72.4 | 70.6 (-1.8, 1.22×) | 69.2 (-3.2, 1.38×) | 67.0 (-5.4, 1.11×) | 69.2 (-3.2, 1.55×) | 68.0 (-8.0, 1.55×) n=50 |
| GPQA | 37.4 | 35.9 (-1.5, 1.10×) | 28.8 (-8.6, 1.24×) | 37.9 (+0.5, 1.05×) | 27.8 (-9.6, 1.23×) | — |
| AIME | 6.7 | 6.7 (+0.0, 1.19×) | 7.8 (+1.1, 1.38×) | 7.8 (+1.1, 1.11×) | 8.9 (+2.2, 1.48×) | — |

**Artifacts:** results-only page (rows = models, ICLR abstract) https://claude.ai/artifact/BD1EFygV8vfBcnLKs7Zd4V · full report https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg
**GitHub:** branch `round3-2026-09-14` — results `docs/results/r3_B/` (README, eval_summary.md/json, ABSTRACT_ICLR2027.md, r3_results.html, calib/), model index `docs/MODELS_AND_RESULTS_2026-09-15.md`, this status file.
**Hugging Face (public, Anupam-Rawat-IITB):** https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train — `r3_B/checkpoints` (ours, final `step000710`), `r3_A/checkpoints`, `round2_t2b_union_flag/step003057`, `reference_evals/`, `r3_B/evals/`, `r3_B/analysis/`, `r3_A/evals/`.
