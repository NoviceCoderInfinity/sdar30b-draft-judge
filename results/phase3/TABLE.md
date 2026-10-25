# Phase 3: re-draft between judge passes

SDAR-1.7B drafter + Phase 1's 128M judge, block 32, threshold 0.95, s8 first draft, no expert pool. rd1 / rd2 re-draft the [MASK] slots after every judge pass / every 2 passes, from committed tokens only.

Setup checks: `{"stock_items": 16, "stock_identical": 16, "judge_items": 16, "judge_with_reference": 16, "judge_identical": 16, "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}`

This box: NVIDIA RTX PRO 6000 Blackwell Server Edition. Tokens per second compares only the setups decoded here (judge, judge_sample, rd1, rd2); stock and today's judge times come from other machines and are not shown.

## Table 1

| Benchmark | Measure | Stock 30B (earlier records) | SDAR-1.7B + judge (Phase 1 records) | SDAR-1.7B + judge (this box) | judge_sample | re-draft every pass | re-draft every 2 passes |
|---|---|---:|---:|---:|---:|---:|---:|
| HumanEval (n = 164) | Accuracy, % (Δ vs stock) | 78.0 | 74.4 (−3.7) | 74.4 (−3.7) | — | 75.6 (−2.4) | **79.9 (+1.8)** |
|  | Passes / block (× fewer than stock) | 8.26 | 5.91 (1.40×) | 5.91 (1.40×) | — | **4.58 (1.80×)** | 4.65 (1.78×) |
|  | Tokens / s (ms / token), this box | other machine | other machine | 7.23 (138.3 ms) | — | **8.61 (116.2 ms)** | 8.58 (116.5 ms) |
| MBPP (n = 257) | Accuracy, % (Δ vs stock) | 65.4 | 66.1 (+0.8) | 66.1 (+0.8) | — | **71.2 (+5.8)** | 69.3 (+3.9) |
|  | Passes / block (× fewer than stock) | 11.93 | 8.75 (1.36×) | 8.75 (1.36×) | — | **7.17 (1.66×)** | 7.30 (1.63×) |
|  | Tokens / s (ms / token), this box | other machine | other machine | 4.70 (212.6 ms) | — | 5.03 (199.0 ms) | **5.12 (195.3 ms)** |
| GSM8K-500 (n = 500) | Accuracy, % (Δ vs stock) | 90.2 | 91.6 (+1.4) | 91.6 (+1.4) | **100.0 (+8.3)** | 91.6 (+1.4) | 92.0 (+1.8) |
|  | Passes / block (× fewer than stock) | 10.06 | 7.12 (1.41×) | 7.12 (1.41×) | 7.67 (1.31×) | **5.93 (1.70×)** | 6.03 (1.67×) |
|  | Tokens / s (ms / token), this box | other machine | other machine | **6.91 (144.7 ms)** | 6.25 (159.9 ms) | 6.83 (146.3 ms) | 6.85 (146.0 ms) |
| IFEval-541 (n = 541) | Accuracy, % (Δ vs stock) | 52.7 | **51.0 (−1.7)** | — | 41.7 (−16.7) | — | — |
|  | Passes / block (× fewer than stock) | 20.23 | **15.57 (1.30×)** | — | 17.23 (1.17×) | — | — |
|  | Tokens / s (ms / token), this box | other machine | other machine | — | 4.21 (237.4 ms) | — | — |
| MATH-500 (n = 500) | Accuracy, % (Δ vs stock) | 72.4 | 71.6 (−0.8) | — | **75.0 (0.0)** | — | — |
|  | Passes / block (× fewer than stock) | 8.52 | 6.38 (1.34×) | — | **6.29 (1.36×)** | — | — |
|  | Tokens / s (ms / token), this box | other machine | other machine | — | 8.93 (111.9 ms) | — | — |
| GPQA-Diamond (n = 198) | Accuracy, % (Δ vs stock) | 37.4 | **36.9 (−0.5)** | — | 16.7 (−8.3) | — | — |
|  | Passes / block (× fewer than stock) | 16.07 | 12.00 (1.34×) | — | **10.41 (1.54×)** | — | — |
|  | Tokens / s (ms / token), this box | other machine | other machine | — | 6.18 (161.8 ms) | — | — |
| AIME (n = 90) | Accuracy, % (Δ vs stock) | 6.7 | 5.6 (−1.1) | — | **8.3 (+8.3)** | — | — |
|  | Passes / block (× fewer than stock) | 10.77 | 7.87 (1.37×) | — | **6.90 (1.56×)** | — | — |
|  | Tokens / s (ms / token), this box | other machine | other machine | — | 8.58 (116.5 ms) | — | — |
| All benchmarks (1 of 7 with every setup finished) (n = 500) | Accuracy, % (Δ vs stock) | 90.2 | 91.6 (+1.4 [−0.6, +3.2]) | 91.6 (+1.4 [−0.6, +3.2]) | **100.0 (+8.3 [0.0, +25.0])** | 91.6 (+1.4 [−0.6, +3.4]) | 92.0 (+1.8 [0.0, +3.8]) |
|  | Passes / block (× fewer than stock) | 10.06 | 7.12 (1.41×) | 7.12 (1.41×) | 7.67 (1.31×) | **5.93 (1.70×)** | 6.03 (1.67×) |
|  | Tokens / s (ms / token), this box | other machine | other machine | **6.91 (144.7 ms)** | 6.25 (159.9 ms) | 6.83 (146.3 ms) | 6.85 (146.0 ms) |

Bold: the best of the judge setups in that row (stock is the reference and never bold). Accuracy deltas on single benchmarks are point estimates; the all-benchmarks row adds the 95% CI.

## Table 2: against today's judge (Phase 1 records)

| Benchmark | Setup | Δ accuracy vs today's judge [95% CI] | Passes / block (today → setup) | Expert loads / token | Drafter forwards / token | Charged loads / token | Forced passes / block | Re-draft rounds / block | Re-drafted slots / block | Re-draft = judge's token | Re-draft = rejected draft | Copy rate, draft slots | Copy rate, mask slots |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | judge | 0.0 [0.0, 0.0] | 5.91 → 5.91 | 739 → 739 | 0.359 → 0.359 | 870 → 870 | 2.22 → 2.22 | 0.00 | 0.00 | — | — | 2.91% | 6.67% |
| HumanEval | rd1 | +1.2 [−4.9, +7.3] | 5.91 → 4.58 | 739 → 578 | 0.359 → 0.695 | 870 → 832 | 2.22 → 1.01 | 3.06 | 31.49 | 19.1% | 45.0% | 3.74% | — |
| HumanEval | rd2 | +5.5 [0.0, +11.0] | 5.91 → 4.65 | 739 → 627 | 0.359 → 0.511 | 870 → 814 | 2.22 → 1.14 | 1.42 | 11.92 | 30.7% | 20.7% | 3.62% | 11.52% |
| MBPP | judge | 0.0 [0.0, 0.0] | 8.75 → 8.75 | 1285 → 1285 | 0.443 → 0.443 | 1447 → 1447 | 3.33 → 3.33 | 0.00 | 0.00 | — | — | 4.21% | 9.91% |
| MBPP | rd1 | +5.1 [0.0, +10.1] | 8.75 → 7.17 | 1285 → 1062 | 0.443 → 1.323 | 1447 → 1545 | 3.33 → 1.80 | 5.62 | 62.61 | 13.6% | 45.6% | 4.89% | — |
| MBPP | rd2 | +3.1 [−1.6, +7.8] | 8.75 → 7.30 | 1285 → 1122 | 0.443 → 0.824 | 1447 → 1422 | 3.33 → 1.93 | 2.69 | 25.61 | 24.4% | 20.0% | 4.86% | 17.52% |
| GSM8K-500 | judge | 0.0 [0.0, 0.0] | 7.12 → 7.12 | 889 → 889 | 0.332 → 0.332 | 1010 → 1010 | 2.29 → 2.29 | 0.00 | 0.00 | — | — | 8.42% | 17.74% |
| GSM8K-500 | judge_sample | 0.0 [0.0, 0.0] | 7.12 → 7.67 | 889 → 936 | 0.332 → 0.337 | 1010 → 1059 | 2.29 → 2.77 | 0.00 | 0.00 | — | — | 8.07% | 15.78% |
| GSM8K-500 | rd1 | 0.0 [−1.4, +1.4] | 7.12 → 5.93 | 889 → 755 | 0.332 → 0.767 | 1010 → 1034 | 2.29 → 1.19 | 4.34 | 40.06 | 27.9% | 24.9% | 9.31% | — |
| GSM8K-500 | rd2 | +0.4 [−1.2, +1.8] | 7.12 → 6.03 | 889 → 770 | 0.332 → 0.545 | 1010 → 969 | 2.29 → 1.30 | 2.08 | 19.55 | 35.2% | 15.9% | 8.60% | 22.51% |
| IFEval-541 | judge_sample | 0.0 [0.0, 0.0] | 15.57 → 17.23 | 1230 → 1386 | 0.301 → 0.305 | 1340 → 1497 | 9.67 → 10.38 | 0.00 | 0.00 | — | — | 8.74% | 26.47% |
| MATH-500 | judge_sample | 0.0 [0.0, 0.0] | 6.38 → 6.29 | 688 → 673 | 0.297 → 0.296 | 796 → 781 | 2.02 → 2.06 | 0.00 | 0.00 | — | — | 5.35% | 14.50% |
| GPQA-Diamond | judge_sample | 0.0 [0.0, 0.0] | 12.00 → 10.41 | 1090 → 974 | 0.296 → 0.292 | 1198 → 1080 | 5.76 → 4.59 | 0.00 | 0.00 | — | — | 5.71% | 23.39% |
| AIME | judge_sample | 0.0 [0.0, 0.0] | 7.87 → 6.90 | 789 → 708 | 0.291 → 0.291 | 896 → 814 | 2.83 → 2.33 | 0.00 | 0.00 | — | — | 5.96% | 26.00% |
| All benchmarks (1 of 7 with every setup finished) | judge | 0.0 [0.0, 0.0] | 7.12 → 7.12 | 889 → 889 | 0.332 → 0.332 | 1010 → 1010 | 2.29 → 2.29 | 0.00 | 0.00 | — | — | 8.42% | 17.74% |
| All benchmarks (1 of 7 with every setup finished) | judge_sample | 0.0 [0.0, 0.0] | 7.12 → 7.67 | 889 → 936 | 0.332 → 0.337 | 1010 → 1059 | 2.29 → 2.77 | 0.00 | 0.00 | — | — | 8.07% | 15.78% |
| All benchmarks (1 of 7 with every setup finished) | rd1 | 0.0 [−1.4, +1.4] | 7.12 → 5.93 | 889 → 755 | 0.332 → 0.767 | 1010 → 1034 | 2.29 → 1.19 | 4.34 | 40.06 | 27.9% | 24.9% | 9.31% | — |
| All benchmarks (1 of 7 with every setup finished) | rd2 | +0.4 [−1.2, +1.8] | 7.12 → 6.03 | 889 → 770 | 0.332 → 0.545 | 1010 → 969 | 2.29 → 1.30 | 2.08 | 19.55 | 35.2% | 15.9% | 8.60% | 22.51% |

Re-draft = judge's token: share of re-drafted tokens equal to the judge's top choice at the pass before (the drafter switched to what the judge wanted). Copy rate: share of open slots whose judge argmax equals the next slot's token (the untrained next-token copy); compare rd1 with rd2, not with zero.

