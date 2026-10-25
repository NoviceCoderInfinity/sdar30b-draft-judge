# r3_B results (live)

> Naming: r3_A / r3_B are the **Nemotron-split** judges; the earlier GSM8K/MATH judge (`round2_t2b_union_flag/step003057`, row `prev_judge_des48` in `eval_summary.json`) is the **standard-split** judge. Comparison page: https://claude.ai/artifact/N1Teqns5a1sJR5vGZvwVhL

Updated 2026-09-15 19:24 IST. Checkpoint: `r3_B/step000710` = epoch 2 of the calibrated recipe (from r3_A step 700, β 0.5 + calibration BCE + override weights 2/3). Weights: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_B/checkpoints

**Results-only page (rows = models, + ICLR 2027 abstract): https://claude.ai/artifact/BD1EFygV8vfBcnLKs7Zd4V** · full report (method, curves, calibration): https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg

- `ABSTRACT_ICLR2027.md` — the draft ICLR 2027 abstract (storyline + numbers, regenerated from the eval files)
- `eval_summary.md` — full-set results, every row paired against stock and stock + DES48 (95% bootstrap CI)
- `calib/` — training-time calibration evals (ECE/NLL vs mask ratio, reliability grids, joint calibration, trends)
- `r3_A_eval_summary.md` — the previous recipe's epoch-1/2 evals (subsets for GSM8K/MATH)

Status of r3_B judge + DES48 per set: humaneval complete, mbpp complete, gsm8k500 complete, math500 complete, gpqa complete, aime complete
Status of r3_B judge (no DES, queued last; may be cut at the 17:30 IST cutoff): humaneval complete, mbpp complete, gsm8k500 complete, math500 complete, gpqa complete, aime complete

| set | r3_B judge+DES48 acc | Δ vs stock [CI] | Δ vs stock+DES48 [CI] | speedup vs stock (exp/token) | r3_B judge (no DES) acc · Δ vs stock · speedup | r3_A epoch 2 judge+DES48 Δ vs stock |
|---|---:|---|---|---:|---|---|
| HumanEval | 75.0 (complete) | -3.0 [-8.5, +1.8] (n=164) | -3.7 [-9.1, +1.2] (n=164) | 1.50× | 75.0 (complete, n=164) · -3.0 [-7.9, +1.8] (n=164) · 1.22× | -8.5 [-15.9, -1.8] (n=164) |
| MBPP (sanitized) | 61.1 (complete) | -4.3 [-10.1, +1.9] (n=257) | -2.7 [-8.6, +2.7] (n=257) | 1.38× | 62.3 (complete, n=257) · -3.1 [-9.3, +2.7] (n=257) · 1.15× | -3.5 [-9.7, +3.1] (n=257) |
| GSM8K-500 | 90.8 (complete) | +0.6 [-1.8, +3.0] (n=500) | -0.4 [-2.4, +1.6] (n=500) | 1.54× | 90.6 (complete, n=500) · +0.4 [-1.6, +2.4] (n=500) · 1.20× | -2.0 [-8.0, +3.0] (n=100) |
| MATH-500 | 69.2 (complete) | -3.2 [-6.6, +0.2] (n=500) | -1.4 [-4.8, +2.0] (n=500) | 1.38× | 67.0 (complete, n=500) · -5.4 [-8.8, -2.0] (n=500) · 1.11× | -8.0 [-16.0, -2.0] (n=50) |
| GPQA-Diamond | 28.8 (complete) | -8.6 [-15.7, -1.5] (n=198) | -7.1 [-13.6, -0.5] (n=198) | 1.24× | 37.9 (complete, n=198) · +0.5 [-7.1, +7.6] (n=198) · 1.05× | — |
| AIME | 7.8 (complete) | +1.1 [-2.2, +5.6] (n=90) | +1.1 [-2.2, +5.6] (n=90) | 1.38× | 7.8 (complete, n=90) · +1.1 [-4.4, +6.7] (n=90) · 1.11× | — |
