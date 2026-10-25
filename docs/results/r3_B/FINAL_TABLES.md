# Round-3 final results — per-dataset tables (19:51 IST, 15 Sep 2026)

Rows = decode methods; every cell is on the full benchmark and paired against the same questions. Δ = accuracy change in points with a 2,000-resample bootstrap 95% CI; passes/block = (forwards − prefill − flush)/blocks; experts/token = unique expert-layer loads per generated token (30B model only; the 1.7B drafter is not charged); speedup = stock experts/token ÷ the method's. Judge = r3_B `step000710` (calibrated loss, epoch 2), decoded with `--judge full --draft s8`; DES = `--pool N --pool-rule des --pool-flush`. Source: `eval_summary.json` in this folder (regenerate with `code/draftalign/eval_summary_r3B.py --step step000710`). Weights and per-question records: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train

### HumanEval (164)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 78.0 | reference | — | 8.26 | 982 | 1.00× | complete (164/164) |
| stock + DES48 | 78.7 | +0.6 [-2.4, +4.3] | reference | 8.05 | 823 | 1.19× | complete (164/164) |
| stock + DES32 | 73.8 | -4.3 [-9.8, +1.2] | -4.9 [-10.4, +0.0] | 8.45 | 620 | 1.58× | complete (164/164) |
| judge (ours, r3_B, no pool) | 75.0 | -3.0 [-7.9, +1.8] | -3.7 [-8.5, +1.2] | 6.11 | 806 | 1.22× | complete (164/164) |
| judge + DES48 (ours, r3_B) | 75.0 | -3.0 [-8.5, +1.8] | -3.7 [-9.1, +1.2] | 5.94 | 657 | 1.50× | complete (164/164) |

### MBPP (257, sanitized)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 65.4 | reference | — | 11.93 | 1714 | 1.00× | complete (257/257) |
| stock + DES48 | 63.8 | -1.6 [-5.8, +3.1] | reference | 12.09 | 1482 | 1.16× | complete (257/257) |
| stock + DES32 | 65.0 | -0.4 [-5.8, +5.4] | +1.2 [-4.7, +7.0] | 12.30 | 1130 | 1.52× | complete (257/257) |
| judge (ours, r3_B, no pool) | 62.3 | -3.1 [-9.3, +2.7] | -1.6 [-7.4, +4.3] | 9.67 | 1492 | 1.15× | complete (257/257) |
| judge + DES48 (ours, r3_B) | 61.1 | -4.3 [-10.1, +1.9] | -2.7 [-8.6, +2.7] | 9.71 | 1239 | 1.38× | complete (257/257) |

### GSM8K-500 (500)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 90.2 | reference | — | 10.06 | 1164 | 1.00× | complete (500/500) |
| stock + DES48 | 91.2 | +1.0 [-0.8, +2.8] | reference | 9.96 | 920 | 1.26× | complete (500/500) |
| stock + DES32 | 89.0 | -1.2 [-3.6, +1.2] | -2.2 [-4.2, +0.0] | 10.46 | 676 | 1.72× | complete (500/500) |
| judge (ours, r3_B, no pool) | 90.6 | +0.4 [-1.6, +2.4] | -0.6 [-2.6, +1.4] | 7.81 | 969 | 1.20× | complete (500/500) |
| judge + DES48 (ours, r3_B) | 90.8 | +0.6 [-1.8, +3.0] | -0.4 [-2.4, +1.6] | 7.83 | 755 | 1.54× | complete (500/500) |

### MATH-500 (500)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 72.4 | reference | — | 8.52 | 854 | 1.00× | complete (500/500) |
| stock + DES48 | 70.6 | -1.8 [-4.8, +1.4] | reference | 8.58 | 700 | 1.22× | complete (500/500) |
| stock + DES32 | 65.2 | -7.2 [-10.6, -3.6] | -5.4 [-9.2, -1.6] | 9.21 | 524 | 1.63× | complete (500/500) |
| judge (ours, r3_B, no pool) | 67.0 | -5.4 [-8.8, -2.0] | -3.6 [-7.2, +0.0] | 7.28 | 769 | 1.11× | complete (500/500) |
| judge + DES48 (ours, r3_B) | 69.2 | -3.2 [-6.6, +0.2] | -1.4 [-4.8, +2.0] | 7.37 | 620 | 1.38× | complete (500/500) |

### GPQA-Diamond (198)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 37.4 | reference | — | 16.07 | 1381 | 1.00× | complete (198/198) |
| stock + DES48 | 35.9 | -1.5 [-9.1, +6.1] | reference | 16.72 | 1256 | 1.10× | complete (198/198) |
| stock + DES32 | 30.3 | -7.1 [-14.6, +0.0] | -5.6 [-13.1, +2.0] | 17.03 | 921 | 1.50× | complete (198/198) |
| judge (ours, r3_B, no pool) | 37.9 | +0.5 [-7.1, +7.6] | +2.0 [-5.1, +8.6] | 14.68 | 1310 | 1.05× | complete (198/198) |
| judge + DES48 (ours, r3_B) | 28.8 | -8.6 [-15.7, -1.5] | -7.1 [-13.6, -0.5] | 14.45 | 1112 | 1.24× | complete (198/198) |

### AIME (90)

| method | accuracy % | Δ vs stock [95% CI] | Δ vs stock + DES48 [95% CI] | passes / block | experts / token | speedup vs stock | status |
|---|---:|---|---|---:|---:|---:|---|
| stock (baseline) | 6.7 | reference | — | 10.77 | 998 | 1.00× | complete (90/90) |
| stock + DES48 | 6.7 | +0.0 [-3.3, +3.3] | reference | 10.79 | 836 | 1.19× | complete (90/90) |
| stock + DES32 | 7.8 | +1.1 [-3.3, +5.6] | +1.1 [-2.2, +5.6] | 11.53 | 627 | 1.59× | complete (90/90) |
| judge (ours, r3_B, no pool) | 7.8 | +1.1 [-4.4, +6.7] | +1.1 [-3.3, +5.6] | 9.17 | 896 | 1.11× | complete (90/90) |
| judge + DES48 (ours, r3_B) | 7.8 | +1.1 [-2.2, +5.6] | +1.1 [-2.2, +5.6] | 9.12 | 725 | 1.38× | complete (90/90) |
