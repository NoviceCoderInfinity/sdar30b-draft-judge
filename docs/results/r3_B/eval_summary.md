# r3_B eval summary — `logs/draftalign/probes_r3/r3_B/step000710` (full sets)

Generated 2026-09-15 19:24 IST. Δ = paired on shared items, 95% bootstrap CI. exp/token = 30B expert loads per token (drafter not charged); speedup = stock exp/token ÷ arm exp/token on the shared items.

## HumanEval (n = 164)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 164 / 164 | 78.0 | ref | — | 8.26 | 982.3 | 1.00× |
| stock + DES32 | complete | 164 / 164 | 73.8 | -4.3 [-9.8, +1.2] (n=164) | -4.9 [-10.4, +0.0] (n=164) | 8.45 | 620.2 | 1.58× |
| stock + DES48 | complete | 164 / 164 | 78.7 | +0.6 [-2.4, +4.3] (n=164) | ref | 8.05 | 822.9 | 1.19× |
| blind accept (no training) | complete | 164 / 164 | 75.6 | -2.4 [-6.1, +1.2] (n=164) | -3.0 [-7.9, +1.8] (n=164) | 6.33 | 782.5 | 1.26× |
| round-2 judge + DES48 (T2b 3057) | complete | 164 / 164 | 62.2 | -15.9 [-22.6, -9.1] (n=164) | -16.5 [-23.2, -10.4] (n=164) | 5.15 | 533.6 | 1.84× |
| r3_A epoch 1 judge + DES48 | complete | 164 / 164 | 67.1 | -11.0 [-17.7, -4.3] (n=164) | -11.6 [-18.3, -5.5] (n=164) | 6.37 | 707.4 | 1.39× |
| r3_A epoch 2 judge + DES48 | complete | 164 / 164 | 69.5 | -8.5 [-15.9, -1.8] (n=164) | -9.1 [-15.9, -3.0] (n=164) | 5.71 | 659.3 | 1.49× |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | complete | 164 / 164 | 70.1 | -7.9 [-14.6, -1.2] (n=164) | -8.5 [-15.2, -2.4] (n=164) | 5.38 | 614.3 | 1.60× |
| r3_B epoch 2 judge | complete | 164 / 164 | 75.0 | -3.0 [-7.9, +1.8] (n=164) | -3.7 [-8.5, +1.2] (n=164) | 6.11 | 805.8 | 1.22× |
| r3_B epoch 2 judge + DES48 | complete | 164 / 164 | 75.0 | -3.0 [-8.5, +1.8] (n=164) | -3.7 [-9.1, +1.2] (n=164) | 5.94 | 657.0 | 1.50× |

## MBPP (sanitized) (n = 257)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 257 / 257 | 65.4 | ref | — | 11.93 | 1714.0 | 1.00× |
| stock + DES32 | complete | 257 / 257 | 65.0 | -0.4 [-5.8, +5.4] (n=257) | +1.2 [-4.7, +7.0] (n=257) | 12.30 | 1129.6 | 1.52× |
| stock + DES48 | complete | 257 / 257 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | ref | 12.09 | 1482.3 | 1.16× |
| blind accept (no training) | complete | 257 / 257 | 60.3 | -5.1 [-8.6, -1.2] (n=257) | -3.5 [-8.6, +1.6] (n=257) | 10.20 | 1523.9 | 1.12× |
| round-2 judge + DES48 (T2b 3057) | complete | 257 / 257 | 53.7 | -11.7 [-18.3, -5.1] (n=257) | -10.1 [-16.3, -4.3] (n=257) | 10.77 | 1366.9 | 1.25× |
| r3_A epoch 1 judge + DES48 | complete | 257 / 257 | 56.8 | -8.6 [-15.2, -2.3] (n=257) | -7.0 [-13.6, -1.2] (n=257) | 9.84 | 1265.9 | 1.35× |
| r3_A epoch 2 judge + DES48 | complete | 257 / 257 | 61.9 | -3.5 [-9.7, +3.1] (n=257) | -1.9 [-8.2, +4.3] (n=257) | 8.78 | 1150.5 | 1.49× |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | complete | 257 / 257 | 55.6 | -9.7 [-16.0, -3.5] (n=257) | -8.2 [-14.4, -1.9] (n=257) | 8.30 | 1087.1 | 1.58× |
| r3_B epoch 2 judge | complete | 257 / 257 | 62.3 | -3.1 [-9.3, +2.7] (n=257) | -1.6 [-7.4, +4.3] (n=257) | 9.67 | 1492.0 | 1.15× |
| r3_B epoch 2 judge + DES48 | complete | 257 / 257 | 61.1 | -4.3 [-10.1, +1.9] (n=257) | -2.7 [-8.6, +2.7] (n=257) | 9.71 | 1239.3 | 1.38× |

## GSM8K-500 (n = 500)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 500 / 500 | 90.2 | ref | — | 10.06 | 1164.2 | 1.00× |
| stock + DES32 | complete | 500 / 500 | 89.0 | -1.2 [-3.6, +1.2] (n=500) | -2.2 [-4.2, +0.0] (n=500) | 10.46 | 676.2 | 1.72× |
| stock + DES48 | complete | 500 / 500 | 91.2 | +1.0 [-0.8, +2.8] (n=500) | ref | 9.96 | 920.5 | 1.26× |
| blind accept (no training) | complete | 500 / 500 | 90.4 | +0.2 [-1.2, +1.6] (n=500) | -0.8 [-2.8, +1.2] (n=500) | 8.49 | 1016.8 | 1.14× |
| round-2 judge + DES48 (T2b 3057) | complete | 500 / 500 | 87.8 | -2.4 [-4.8, +0.0] (n=500) | -3.4 [-6.0, -1.0] (n=500) | 6.76 | 667.7 | 1.74× |
| r3_A epoch 1 judge + DES48 | PARTIAL 100/500 scored, 100 decoded | 100 / 100 | 90.0 | -2.0 [-8.0, +3.0] (n=100) | -2.0 [-8.0, +3.0] (n=100) | 7.76 | 761.1 | 1.57× |
| r3_A epoch 2 judge + DES48 | PARTIAL 100/500 scored, 100 decoded | 100 / 100 | 90.0 | -2.0 [-8.0, +3.0] (n=100) | -2.0 [-7.0, +2.0] (n=100) | 6.89 | 686.9 | 1.74× |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_B epoch 2 judge | complete | 500 / 500 | 90.6 | +0.4 [-1.6, +2.4] (n=500) | -0.6 [-2.6, +1.4] (n=500) | 7.81 | 968.6 | 1.20× |
| r3_B epoch 2 judge + DES48 | complete | 500 / 500 | 90.8 | +0.6 [-1.8, +3.0] (n=500) | -0.4 [-2.4, +1.6] (n=500) | 7.83 | 755.5 | 1.54× |

## MATH-500 (n = 500)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 500 / 500 | 72.4 | ref | — | 8.52 | 853.5 | 1.00× |
| stock + DES32 | complete | 500 / 500 | 65.2 | -7.2 [-10.6, -3.6] (n=500) | -5.4 [-9.2, -1.6] (n=500) | 9.21 | 524.0 | 1.63× |
| stock + DES48 | complete | 500 / 500 | 70.6 | -1.8 [-4.8, +1.4] (n=500) | ref | 8.58 | 700.4 | 1.22× |
| blind accept (no training) | complete | 500 / 500 | 70.8 | -1.6 [-4.6, +1.4] (n=500) | +0.2 [-3.2, +3.6] (n=500) | 7.40 | 764.0 | 1.12× |
| round-2 judge + DES48 (T2b 3057) | complete | 500 / 500 | 69.2 | -3.2 [-6.8, +0.2] (n=500) | -1.4 [-4.6, +2.0] (n=500) | 6.39 | 549.5 | 1.55× |
| r3_A epoch 1 judge + DES48 | PARTIAL 50/500 scored, 50 decoded | 50 / 50 | 72.0 | -4.0 [-16.0, +8.0] (n=50) | +4.0 [-6.0, +16.0] (n=50) | 7.47 | 629.5 | 1.35× |
| r3_A epoch 2 judge + DES48 | PARTIAL 50/500 scored, 50 decoded | 50 / 50 | 68.0 | -8.0 [-16.0, -2.0] (n=50) | +0.0 [-10.0, +10.0] (n=50) | 6.37 | 547.0 | 1.55× |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_B epoch 2 judge | complete | 500 / 500 | 67.0 | -5.4 [-8.8, -2.0] (n=500) | -3.6 [-7.2, +0.0] (n=500) | 7.28 | 769.1 | 1.11× |
| r3_B epoch 2 judge + DES48 | complete | 500 / 500 | 69.2 | -3.2 [-6.6, +0.2] (n=500) | -1.4 [-4.8, +2.0] (n=500) | 7.37 | 620.1 | 1.38× |

## GPQA-Diamond (n = 198)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 198 / 198 | 37.4 | ref | — | 16.07 | 1381.1 | 1.00× |
| stock + DES32 | complete | 198 / 198 | 30.3 | -7.1 [-14.6, +0.0] (n=198) | -5.6 [-13.1, +2.0] (n=198) | 17.03 | 921.1 | 1.50× |
| stock + DES48 | complete | 198 / 198 | 35.9 | -1.5 [-9.1, +6.1] (n=198) | ref | 16.72 | 1256.5 | 1.10× |
| blind accept (no training) | complete | 198 / 198 | 35.4 | -2.0 [-8.1, +4.0] (n=198) | -0.5 [-8.1, +7.1] (n=198) | 15.48 | 1348.4 | 1.02× |
| round-2 judge + DES48 (T2b 3057) | complete | 198 / 198 | 27.8 | -9.6 [-17.2, -2.5] (n=198) | -8.1 [-15.7, -1.0] (n=198) | 14.60 | 1120.0 | 1.23× |
| r3_A epoch 1 judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_A epoch 2 judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_B epoch 2 judge | complete | 198 / 198 | 37.9 | +0.5 [-7.1, +7.6] (n=198) | +2.0 [-5.1, +8.6] (n=198) | 14.68 | 1310.3 | 1.05× |
| r3_B epoch 2 judge + DES48 | complete | 198 / 198 | 28.8 | -8.6 [-15.7, -1.5] (n=198) | -7.1 [-13.6, -0.5] (n=198) | 14.45 | 1112.5 | 1.24× |

## AIME (n = 90)

| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |
|---|---|---:|---:|---|---|---:|---:|---:|
| stock | complete | 90 / 90 | 6.7 | ref | — | 10.77 | 997.7 | 1.00× |
| stock + DES32 | complete | 90 / 90 | 7.8 | +1.1 [-3.3, +5.6] (n=90) | +1.1 [-2.2, +5.6] (n=90) | 11.53 | 626.9 | 1.59× |
| stock + DES48 | complete | 90 / 90 | 6.7 | +0.0 [-3.3, +3.3] (n=90) | ref | 10.79 | 836.3 | 1.19× |
| blind accept (no training) | complete | 90 / 90 | 5.6 | -1.1 [-5.6, +2.2] (n=90) | -1.1 [-3.3, +0.0] (n=90) | 9.75 | 927.3 | 1.08× |
| round-2 judge + DES48 (T2b 3057) | complete | 90 / 90 | 8.9 | +2.2 [-2.2, +6.7] (n=90) | +2.2 [-2.2, +6.7] (n=90) | 8.35 | 672.9 | 1.48× |
| r3_A epoch 1 judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_A epoch 2 judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_A last ckpt (step 2000, 2.82 ep) judge + DES48 | PENDING | 0 / 0 | — | — | — | — | — | — |
| r3_B epoch 2 judge | complete | 90 / 90 | 7.8 | +1.1 [-4.4, +6.7] (n=90) | +1.1 [-3.3, +5.6] (n=90) | 9.17 | 896.2 | 1.11× |
| r3_B epoch 2 judge + DES48 | complete | 90 / 90 | 7.8 | +1.1 [-2.2, +5.6] (n=90) | +1.1 [-2.2, +5.6] (n=90) | 9.12 | 725.2 | 1.38× |

