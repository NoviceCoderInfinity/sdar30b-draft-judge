# Round-3 downstream eval summary (r3_A)

Generated 2026-09-15 19:20 IST (2026-09-15T13:50:02Z). Every Δ is paired on the shared idx; stock rows are on the dataset universe below. Metric definitions: AGENTS.md §8.

Coverage: {'complete': 39, 'partial': 0, 'unscored': 0, 'pending': 10, 'dropped': 11, 'scheduled': 0, 'n/a': 0}

## step000700 (epoch 1)

### HumanEval — universe n=164 (first 164 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 164 / 164 | 78.0 | ref | 8.26 | 982.3 | 1.000x | 17.1 |
| stock + DES32 | complete | 164 / 164 | 73.8 | -4.3 [-9.8, +1.2] (n=164) | 8.45 | 620.2 | 1.584x | 7.1 |
| stock + DES48 | complete | 164 / 164 | 78.7 | +0.6 [-2.4, +4.3] (n=164) | 8.05 | 822.9 | 1.194x | 9.0 |
| judge | complete | 164 / 164 | 68.9 | -9.1 [-15.2, -3.7] (n=164) | 6.20 | 829.7 | 1.184x | 7.5 |
| judge + DES32 | complete | 164 / 164 | 69.5 | -8.5 [-15.9, -1.8] (n=164) | 6.39 | 523.9 | 1.875x | 5.5 |
| judge + DES48 | complete | 164 / 164 | 67.1 | -11.0 [-17.7, -4.3] (n=164) | 6.37 | 707.4 | 1.389x | 7.2 |

### MBPP — universe n=257 (first 257 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 257 / 257 | 65.4 | ref | 11.93 | 1714.0 | 1.000x | 15.1 |
| stock + DES32 | complete | 257 / 257 | 65.0 | -0.4 [-5.8, +5.4] (n=257) | 12.30 | 1129.6 | 1.517x | 10.9 |
| stock + DES48 | complete | 257 / 257 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | 12.09 | 1482.3 | 1.156x | 13.8 |
| judge | complete | 257 / 257 | 54.1 | -11.3 [-17.5, -5.1] (n=257) | 9.76 | 1532.7 | 1.118x | 7.3 |
| judge + DES32 | complete | 257 / 257 | 56.0 | -9.3 [-15.6, -2.7] (n=257) | 10.09 | 993.2 | 1.726x | 5.2 |
| judge + DES48 | complete | 257 / 257 | 56.8 | -8.6 [-15.2, -2.3] (n=257) | 9.84 | 1265.9 | 1.354x | 6.7 |

### GSM8K-500 (first 100) — universe n=100 (idx observed in round-3 records so far)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 100 / 100 | 93.0 | ref | 9.99 | 1178.8 | 1.000x | 17.3 |
| stock + DES32 | complete | 100 / 100 | 91.0 | -2.0 [-8.0, +4.0] (n=100) | 10.53 | 692.4 | 1.703x | 20.9 |
| stock + DES48 | complete | 100 / 100 | 92.0 | -1.0 [-6.0, +3.0] (n=100) | 9.94 | 930.3 | 1.267x | 27.0 |
| judge | complete | 100 / 100 | 90.0 | -3.0 [-7.0, +1.0] (n=100) | 7.80 | 985.8 | 1.196x | 15.2 |
| judge + DES32 | complete | 100 / 100 | 88.0 | -5.0 [-11.0, +0.0] (n=100) | 8.46 | 570.7 | 2.066x | 10.0 |
| judge + DES48 | complete | 100 / 100 | 90.0 | -3.0 [-8.0, +2.0] (n=100) | 7.76 | 761.1 | 1.549x | 13.0 |

### MATH-500 (first 50) — universe n=50 (first 50 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 50 / 50 | 76.0 | ref | 8.41 | 849.2 | 1.000x | 71.0 |
| stock + DES32 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +4.0] (n=50) | 9.32 | 531.2 | 1.599x | 50.9 |
| stock + DES48 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +2.0] (n=50) | 8.69 | 718.7 | 1.182x | 63.8 |
| judge | complete | 50 / 50 | 68.0 | -8.0 [-18.0, +2.0] (n=50) | 7.88 | 835.4 | 1.016x | 36.3 |
| judge + DES32 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +4.0] (n=50) | 8.18 | 475.6 | 1.786x | 25.8 |
| judge + DES48 | complete | 50 / 50 | 72.0 | -4.0 [-16.0, +8.0] (n=50) | 7.47 | 629.5 | 1.349x | 32.5 |

## step001400 (epoch 2)

### HumanEval — universe n=164 (first 164 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 164 / 164 | 78.0 | ref | 8.26 | 982.3 | 1.000x | 17.1 |
| stock + DES32 | complete | 164 / 164 | 73.8 | -4.3 [-9.8, +1.2] (n=164) | 8.45 | 620.2 | 1.584x | 7.1 |
| stock + DES48 | complete | 164 / 164 | 78.7 | +0.6 [-2.4, +4.3] (n=164) | 8.05 | 822.9 | 1.194x | 9.0 |
| judge | complete | 164 / 164 | 69.5 | -8.5 [-14.6, -2.4] (n=164) | 5.76 | 793.1 | 1.239x | 6.7 |
| judge + DES32 | complete | 164 / 164 | 68.9 | -9.1 [-16.5, -2.4] (n=164) | 5.91 | 492.6 | 1.994x | 5.4 |
| judge + DES48 | complete | 164 / 164 | 69.5 | -8.5 [-15.9, -1.8] (n=164) | 5.71 | 659.3 | 1.490x | 6.3 |

### MBPP — universe n=257 (first 257 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 257 / 257 | 65.4 | ref | 11.93 | 1714.0 | 1.000x | 15.1 |
| stock + DES32 | complete | 257 / 257 | 65.0 | -0.4 [-5.8, +5.4] (n=257) | 12.30 | 1129.6 | 1.517x | 10.9 |
| stock + DES48 | complete | 257 / 257 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | 12.09 | 1482.3 | 1.156x | 13.8 |
| judge | complete | 257 / 257 | 57.6 | -7.8 [-14.4, -1.6] (n=257) | 8.86 | 1402.7 | 1.222x | 6.8 |
| judge + DES32 | complete | 257 / 257 | 54.5 | -10.9 [-17.5, -4.3] (n=257) | 9.05 | 888.5 | 1.929x | 4.8 |
| judge + DES48 | complete | 257 / 257 | 61.9 | -3.5 [-9.7, +3.1] (n=257) | 8.78 | 1150.5 | 1.490x | 6.0 |

### GSM8K-500 (first 100) — universe n=100 (idx observed in round-3 records so far)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 100 / 100 | 93.0 | ref | 9.99 | 1178.8 | 1.000x | 17.3 |
| stock + DES32 | complete | 100 / 100 | 91.0 | -2.0 [-8.0, +4.0] (n=100) | 10.53 | 692.4 | 1.703x | 20.9 |
| stock + DES48 | complete | 100 / 100 | 92.0 | -1.0 [-6.0, +3.0] (n=100) | 9.94 | 930.3 | 1.267x | 27.0 |
| judge | complete | 100 / 100 | 88.0 | -5.0 [-10.0, +0.0] (n=100) | 6.93 | 896.2 | 1.315x | 14.4 |
| judge + DES32 | complete | 100 / 100 | 87.0 | -6.0 [-12.0, +0.0] (n=100) | 7.52 | 515.7 | 2.286x | 9.0 |
| judge + DES48 | complete | 100 / 100 | 90.0 | -3.0 [-8.0, +2.0] (n=100) | 6.89 | 686.9 | 1.716x | 11.8 |

### MATH-500 (first 50) — universe n=50 (first 50 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 50 / 50 | 76.0 | ref | 8.41 | 849.2 | 1.000x | 71.0 |
| stock + DES32 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +4.0] (n=50) | 9.32 | 531.2 | 1.599x | 50.9 |
| stock + DES48 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +2.0] (n=50) | 8.69 | 718.7 | 1.182x | 63.8 |
| judge | complete | 50 / 50 | 66.0 | -10.0 [-22.0, +2.0] (n=50) | 6.72 | 720.1 | 1.179x | 35.2 |
| judge + DES32 | complete | 50 / 50 | 60.0 | -16.0 [-28.0, -6.0] (n=50) | 7.45 | 434.4 | 1.955x | 25.6 |
| judge + DES48 | complete | 50 / 50 | 68.0 | -8.0 [-16.0, -2.0] (n=50) | 6.37 | 547.0 | 1.552x | 31.1 |

## step001500 (dropped)

### HumanEval — universe n=164 (first 164 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 164 / 164 | 78.0 | ref | 8.26 | 982.3 | 1.000x | 17.1 |
| stock + DES32 | complete | 164 / 164 | 73.8 | -4.3 [-9.8, +1.2] (n=164) | 8.45 | 620.2 | 1.584x | 7.1 |
| stock + DES48 | complete | 164 / 164 | 78.7 | +0.6 [-2.4, +4.3] (n=164) | 8.05 | 822.9 | 1.194x | 9.0 |
| judge | complete | 164 / 164 | 70.7 | -7.3 [-14.0, -0.6] (n=164) | 5.73 | 788.0 | 1.247x | 6.6 |
| judge + DES32 | DROPPED (82/164 ran) | 82 / 82 | 70.7 | -7.3 [-15.9, +1.2] (n=82) | 5.25 | 455.5 | 2.102x | 5.0 |
| judge + DES48 | DROPPED | 0 / 0 | — | — | — | — | — | — |

### MBPP — universe n=257 (first 257 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 257 / 257 | 65.4 | ref | 11.93 | 1714.0 | 1.000x | 15.1 |
| stock + DES32 | complete | 257 / 257 | 65.0 | -0.4 [-5.8, +5.4] (n=257) | 12.30 | 1129.6 | 1.517x | 10.9 |
| stock + DES48 | complete | 257 / 257 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | 12.09 | 1482.3 | 1.156x | 13.8 |
| judge | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | DROPPED | 0 / 0 | — | — | — | — | — | — |

### GSM8K-500 (first 100) — universe n=100 (idx observed in round-3 records so far)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 100 / 100 | 93.0 | ref | 9.99 | 1178.8 | 1.000x | 17.3 |
| stock + DES32 | complete | 100 / 100 | 91.0 | -2.0 [-8.0, +4.0] (n=100) | 10.53 | 692.4 | 1.703x | 20.9 |
| stock + DES48 | complete | 100 / 100 | 92.0 | -1.0 [-6.0, +3.0] (n=100) | 9.94 | 930.3 | 1.267x | 27.0 |
| judge | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | DROPPED | 0 / 0 | — | — | — | — | — | — |

### MATH-500 (first 50) — universe n=50 (first 50 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 50 / 50 | 76.0 | ref | 8.41 | 849.2 | 1.000x | 71.0 |
| stock + DES32 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +4.0] (n=50) | 9.32 | 531.2 | 1.599x | 50.9 |
| stock + DES48 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +2.0] (n=50) | 8.69 | 718.7 | 1.182x | 63.8 |
| judge | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | DROPPED | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | DROPPED | 0 / 0 | — | — | — | — | — | — |

## step002000 (step 2,000)

### HumanEval — universe n=164 (first 164 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 164 / 164 | 78.0 | ref | 8.26 | 982.3 | 1.000x | 17.1 |
| stock + DES32 | complete | 164 / 164 | 73.8 | -4.3 [-9.8, +1.2] (n=164) | 8.45 | 620.2 | 1.584x | 7.1 |
| stock + DES48 | complete | 164 / 164 | 78.7 | +0.6 [-2.4, +4.3] (n=164) | 8.05 | 822.9 | 1.194x | 9.0 |
| judge | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | complete | 164 / 164 | 70.1 | -7.9 [-14.6, -1.2] (n=164) | 5.38 | 614.3 | 1.599x | 6.3 |

### MBPP — universe n=257 (first 257 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 257 / 257 | 65.4 | ref | 11.93 | 1714.0 | 1.000x | 15.1 |
| stock + DES32 | complete | 257 / 257 | 65.0 | -0.4 [-5.8, +5.4] (n=257) | 12.30 | 1129.6 | 1.517x | 10.9 |
| stock + DES48 | complete | 257 / 257 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | 12.09 | 1482.3 | 1.156x | 13.8 |
| judge | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | complete | 257 / 257 | 55.6 | -9.7 [-16.0, -3.5] (n=257) | 8.30 | 1087.1 | 1.577x | 5.9 |

### GSM8K-500 (first 100) — universe n=100 (idx observed in round-3 records so far)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 100 / 100 | 93.0 | ref | 9.99 | 1178.8 | 1.000x | 17.3 |
| stock + DES32 | complete | 100 / 100 | 91.0 | -2.0 [-8.0, +4.0] (n=100) | 10.53 | 692.4 | 1.703x | 20.9 |
| stock + DES48 | complete | 100 / 100 | 92.0 | -1.0 [-6.0, +3.0] (n=100) | 9.94 | 930.3 | 1.267x | 27.0 |
| judge | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |

### MATH-500 (first 50) — universe n=50 (first 50 idx of the reference set)

| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |
|---|---|---:|---:|---|---:|---:|---:|---:|
| stock | complete | 50 / 50 | 76.0 | ref | 8.41 | 849.2 | 1.000x | 71.0 |
| stock + DES32 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +4.0] (n=50) | 9.32 | 531.2 | 1.599x | 50.9 |
| stock + DES48 | complete | 50 / 50 | 68.0 | -8.0 [-20.0, +2.0] (n=50) | 8.69 | 718.7 | 1.182x | 63.8 |
| judge | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES32 | PENDING | 0 / 0 | — | — | — | — | — | — |
| judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |

## Epoch direction (same arm, paired on shared items)

| dataset | arm | epoch 2 − epoch 1 Δacc [CI] | e2-only right / e1-only right | exp/token e1 → e2 | final (step 1900) − epoch 2 Δacc |
|---|---|---|---:|---|---|
| HumanEval | judge | +0.6 [-4.9, +6.1] (n=164) | 12 / 11 | 829.7 → 793.1 | — |
| HumanEval | judge + DES32 | -0.6 [-6.7, +6.1] (n=164) | 13 / 14 | 523.9 → 492.6 | — |
| HumanEval | judge + DES48 | +2.4 [-3.7, +9.1] (n=164) | 17 / 13 | 707.4 → 659.3 | — |
| MBPP | judge | +3.5 [-1.2, +8.6] (n=257) | 27 / 18 | 1532.7 → 1402.7 | — |
| MBPP | judge + DES32 | -1.6 [-7.4, +3.9] (n=257) | 25 / 29 | 993.2 → 888.5 | — |
| MBPP | judge + DES48 | +5.1 [-0.0, +10.5] (n=257) | 30 / 17 | 1265.9 → 1150.5 | — |
| GSM8K-500 (first 100) | judge | -2.0 [-7.0, +3.0] (n=100) | 2 / 4 | 985.8 → 896.2 | — |
| GSM8K-500 (first 100) | judge + DES32 | -1.0 [-7.0, +5.0] (n=100) | 5 / 6 | 570.7 → 515.7 | — |
| GSM8K-500 (first 100) | judge + DES48 | +0.0 [-4.0, +4.0] (n=100) | 2 / 2 | 761.1 → 686.9 | — |
| MATH-500 (first 50) | judge | -2.0 [-14.0, +10.0] (n=50) | 4 / 5 | 835.4 → 720.1 | — |
| MATH-500 (first 50) | judge + DES32 | -8.0 [-22.0, +6.0] (n=50) | 5 / 9 | 475.6 → 434.4 | — |
| MATH-500 (first 50) | judge + DES48 | -4.0 [-16.0, +8.0] (n=50) | 4 / 6 | 629.5 → 547.0 | — |

## Previous round — team checkpoint T2b step 3057 (`_ad734af656`), documentation only

| dataset | arm | n | acc % | Δacc vs stock [CI] | exp/token | speedup (exp) |
|---|---|---:|---:|---|---:|---:|
| HumanEval | judge | 164 | 64.6 | -13.4 [-20.7, -6.7] (n=164) | 610.4 | 1.609x |
| HumanEval | judge + DES32 | — | no cell | — | — | — |
| HumanEval | judge + DES48 | 164 | 62.2 | -15.9 [-22.6, -9.1] (n=164) | 533.6 | 1.841x |
| MBPP | judge | 257 | 51.4 | -14.0 [-20.6, -7.8] (n=257) | 1588.2 | 1.079x |
| MBPP | judge + DES32 | 257 | 55.6 | -9.7 [-16.0, -3.5] (n=257) | 1006.5 | 1.703x |
| MBPP | judge + DES48 | 257 | 53.7 | -11.7 [-18.3, -5.1] (n=257) | 1366.9 | 1.254x |
| GSM8K-500 (first 100) | judge | 100 | 90.0 | -3.0 [-9.0, +2.0] (n=100) | 885.6 | 1.331x |
| GSM8K-500 (first 100) | judge + DES32 | 100 | 92.0 | -1.0 [-5.0, +3.0] (n=100) | 504.6 | 2.336x |
| GSM8K-500 (first 100) | judge + DES48 | 100 | 91.0 | -2.0 [-6.0, +2.0] (n=100) | 706.7 | 1.668x |
| MATH-500 (first 50) | judge | 50 | 64.0 | -12.0 [-22.0, -2.0] (n=50) | 686.8 | 1.236x |
| MATH-500 (first 50) | judge + DES32 | 50 | 62.0 | -14.0 [-26.0, -2.0] (n=50) | 403.8 | 2.103x |
| MATH-500 (first 50) | judge + DES48 | 50 | 74.0 | -2.0 [-10.0, +6.0] (n=50) | 535.7 | 1.585x |
