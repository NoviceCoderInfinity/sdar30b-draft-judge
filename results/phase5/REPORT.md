# Phase 5 — judge retraining on fresh wave-C data: FINAL REPORT

**24 Oct 2026, 05:55 IST. Instance 57349095 (TEXAS: 8× RTX PRO 6000 Blackwell, 2× EPYC 9555). Wrap-up before operator removes the server.**

## Verdict — FAIL (128M Phase-4 judge retained)

Phase 5 retrained **only** the flag-gated 128M LoRA judge on fresh, unseen wave-C states, aiming for ≥5% fewer judge passes/block vs Phase 4 (7.68 rd1) at accuracy within noise. **Every** retrained checkpoint degenerates in live HF decode into non-terminating repetition loops; accuracy collapses far outside noise (ifeval541 56%→~8%). On the original success criterion (<2% improvement = kill) this is a **KILL/FAIL**. The 128M Phase-4 judge is unaffected and is reconfirmed healthy on this box (table below). **Keep 128M (Phase 4) as the production judge.**

Full root cause: `logs/phase5/FAILURE_ANALYSIS.md` (and `logs/phase5/eval_diag/FINDINGS.md`).

## 1. Diag bisect — the retrained judge is broken at every dose
`logs/phase5/eval_diag`, ifeval541 first 16 items, exact rd1 flags, plain HF harness (`run_da_decode.py`), 2 GPUs/adapter.

| adapter | tokens | acc | looping | stop@EOS | truncated | gen→EOS |
|---|---|---|---|---|---|---|
| start_128m (Phase 4) | 128.00M | **56%** | 1/16 | 89% | 11% | 343 |
| rec tok_136M | +8M | 9% | 10/11 | 14% | 86% | 892 |
| rec final 147.61M | +19.6M | 7% | 14/15 | 14% | 86% | 891 |
| rec_main_ep2 final 155.66M | +27.66M | 8% | 11/12 | 12% | 88% | 910 |

Breakage is full-blown at the FIRST checkpoint (+8M tokens) and flat thereafter → systematic in the training setup, not over-training. No retrained checkpoint is salvageable.

## 2. Root cause (summary)
A **train/decode engine mismatch** introduced by Phase 5 recording the judge states in **SGLang** for the first time, while **training and eval stay in the HF/torch harness** (the standing rule). SGLang↔HF disagree ~10% on judge argmax (step-0 = 0.8978 for the *identical* 128M adapter) and ~40% on tokens (CP3 vanilla 0.596). Fitting the HF judge to SGLang-trajectory states is off-distribution → the retrained judge **over-commits on its own HF trajectory**, corrupts the committed context, and the frozen base never reaches EOS → repetition loops / truncation. The recorded-state training **construction itself is correct** (target = vanilla T not recorded argmax; committed slots never loss rows; redrafts are inputs not targets; flag only on draft-filled open slots; post-EOS convention is the long-standing one 128M also used). The held-out probe and step-0 are computed on the same SGLang-recorded states and are blind to live-decode non-termination; the only visible fingerprint was the step-0 gap growing 0.8978 → 0.7256.

## 3. Same-box Phase-4 judge rd1 (the production judge, this box) — HEALTHY
Plain HF harness, rd1 + `--redraft-ban-rejected --loop-guard 12`, adapter = `logs/phase5/judge/start_128m` (128M, never merged), 8-way sharded, full sets. Token-identity column compares `gen_ids` against the Phase-4-box rd1 records (`logs/phase4/decode/rd1`).

| set | n | acc (this box) | acc (P4 box) | pass/blk (this) | pass/blk (P4) | median s | trunc | stop@EOS | tokens identical vs P4 |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k500 | 500 | 0.910 | 0.914 | 5.75 | 5.77 | 12.78 | 0% | 100% | 495/500 |
| math500 | 500 | 0.722 | 0.716 | 5.27 | 5.23 | 28.67 | 6% | 94% | 395/500 |
| humaneval | 164 | 0.768 | 0.768 | 4.66 | 4.67 | 6.48 | 1% | 99% | 164/164 |
| mbpp | 257 | 0.704 | 0.704 | 7.12 | 7.12 | 5.67 | 0% | 100% | 256/257 |

acc within ±0.6 pt, passes/block within ±0.04, near-perfect token identity → **the harness and the 128M judge reproduce the Phase-4 production numbers on this box.** Only the Phase-5 retrained judge is broken. (gpqa, aime, ifeval541 rd1 on this box were NOT reached before the stop.)

## 4. Vanilla multiplier — PARTIAL (`logs/phase5/vanilla/multiplier.txt`)
This-box `--stock` vanilla vs Phase-4-box vanilla, paired by (split, idx). chain128 was stopped by PID at 05:46 IST mid-vanilla; only 2 of 7 sets completed.

| set | n | tok identical | sec_this | sec_p4 | ratio | gpu_active ratio |
|---|---|---|---|---|---|---|
| ifeval541 | 28 | 28/28 | 34.72 | 32.95 | 1.015 | 0.997 |
| math500 | 23 | 23/23 | 33.87 | 33.42 | 1.013 | 1.005 |
| **POOLED** | 51 | — | — | — | **1.013** | **1.001** |

This box is ~1.3% slower (wall) / ~0.1% (gpu-active) than the Phase-4 box, 100% token-identical on 51 paired items → the two boxes are directly comparable (cross-box multiplier ≈ 1.01). PARTIAL: 2 of 7 sets.

## 5. Timeline (IST, 23–24 Oct 2026)
- **23 Oct ~10:09** box t0 (prior box 57321867); **14:31** TEXAS takeover (57349095).
- Generation (prior box, 250,790 prompts) → **drafts** (Phase-4 merged drafter, SGLang) → **pack** → **CP3 parity** → **index** (wave-C, REPEAT_SCIENCE=0, fresh/unseen).
- **CP5 recording** (SGLang `judge_sglang.py --mode record`, rd1 + redraft-ban + loop-guard 12, 128M judge): folders `record` (15,999), `record2` (disjoint), `record3`, `record_run1` — **191 states.npz** total; filtered training index = **47,011 items** (recorded ~47.8k − decontam/dup exclusions − no-ops).
- **Training** (HF `train_c1b_rec.py`, 8-GPU torchrun, `--continue-from` 128M): 128.00M → 136M → 144M → **147.61M** (epoch 1, `rec_main/final`, ~03:01) → epoch-2 resume → 152M → **155.66M** (`rec_main_ep2/final`, 04:05 via `--stop-at`).
- **04:15–04:31** rd1 eval of `rec_main_ep2/final` → **degenerate loops** on ifeval541 → stopped by PID.
- **04:31–05:05** `diag` bisect (4 adapters) + CPU root-cause → FAIL confirmed; `FAILURE_ANALYSIS.md` written.
- **04:39–05:46** `chain128` same-box Phase-4 rd1 eval (gsm8k500/math500/humaneval/mbpp done) + partial vanilla; stopped by PID at 05:46 for server removal.

## 6. Checkpoints (all have adapter.safetensors + adapter_config.json + trainer_state.json + trainer_state.pt)
| path | tokens |
|---|---|
| `logs/phase5/judge/start_128m` | 128.00M (Phase-4 start) |
| `logs/phase5/judge/rec_main/tok_000136000000` | 136.00M |
| `logs/phase5/judge/rec_main/tok_000144000000` | 144.01M |
| `logs/phase5/judge/rec_main/roll_00012094` | rolling (step 12094) |
| `logs/phase5/judge/rec_main/final` | 147.61M (epoch 1 end) |
| `logs/phase5/judge/rec_main_ep2/tok_000152000000` | 152.02M |
| `logs/phase5/judge/rec_main_ep2/final` | 155.66M (epoch 2 end) |

## 7. Restore / reproduce
See **`docs/PHASE5_RESUME.md`** (clone at pinned commit → `hf download Arushhh/phase5-loop-guard` → `sha256sum -c logs/phase5/MANIFEST.sha256` → resume via `train_cp6.sh` or `--continue-from`; eval via plain HF `run_da_decode.py`). Code: GitHub `phase5-box-s1` (+ `phase5-logs`). Data/states/checkpoints: HF dataset `Arushhh/phase5-loop-guard`.

## 8. Lesson for a valid retrain
Record training states in the **same harness** that trains/evals (no cross-engine gap), or require SGLang↔HF per-decision agreement ≫ 0.9 before training on SGLang states; prefer **on-policy / DAgger** over off-policy imitation of a teacher-forced predecessor; and gate training checkpoints on a few **live autoregressive** decodes (EOS/termination + accuracy) — the teacher-forced probe cannot see this failure.
