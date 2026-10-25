# Phase 3 re-draft on the 8-GPU machine: MEASURED

Generated 2026-10-08T00:12:03Z from the raw per-item records. Every number is MEASURED; nothing is extrapolated or adjusted.

Machine: 8x NVIDIA RTX PRO 6000 Blackwell Server Edition, Intel Xeon Gold 6430 (vast.ai, gpu_frac 1.0). rd1 / rd2: `--drafter d32 --draft s8 --judge full --adapter <128M> --judge-diag --full-redraft-every 1|2`, block 32, threshold 0.95, no expert pool, batch 1, 8 shards (one process per GPU).

**This machine vs the Phase 1 machine on identical work** (setup check, base pipeline, 16 HumanEval items, 16/16 token-identical): 233.9 s here vs 107.6 s in the Phase 1 records = **2.174x the time** (prompt prefill 1.845x). Speedups against Phase 1 base records therefore compare two machines of different speed; they are listed only as raw ratios, not as the method's speedup.

`--judge-diag` cost, measured: 209.6 us per judge pass (max over 8 GPUs); its share of each run's decode time is in the table.

## Table 1: rd1 and rd2, every item decoded on this machine

| Benchmark | Setup | Items | Accuracy % | Δ acc vs Phase 1 base, pts [95% CI] | Passes / block | 30B expert loads / token | Experts / 30B forward | Experts / judge pass | Drafter fwd / token | Tokens / s | ms / token | Total generation time | Wall (8 GPUs) | judge-diag share |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IFEval-541 | base, Phase 1 records | 541 | 51.0 | reference | 15.57 | 1230 | 2209 | 2198 | 0.30 | 8.93 (Phase 1 machine) | 112.0 | 5.77 h | · | · |
| IFEval-541 | rd1 | 541 | 46.6 | −4.4 [−8.1, −0.4] | 11.31 | 894 | 2179 | 2170 | 1.25 | 4.73 | 211.5 | 13.17 h | 105.3 min | 0.037% |
| IFEval-541 | rd2 | 541 | 48.1 | −3.0 [−6.5, +0.7] | 12.23 | 974 | 2204 | 2195 | 0.82 | 4.65 | 214.9 | 12.22 h | 101.5 min | 0.040% |
| MATH-500 | base, Phase 1 records | 500 | 71.6 | reference | 6.38 | 688 | 2805 | 2781 | 0.30 | 15.73 (Phase 1 machine) | 63.6 | 4.59 h | · | · |
| MATH-500 | rd1 | 500 | 70.8 | −0.8 [−4.0, +2.2] | 5.24 | 571 | 2768 | 2736 | 0.65 | 7.91 | 126.4 | 9.72 h | 79.9 min | 0.028% |
| MATH-500 | rd2 | 500 | 70.2 | −1.4 [−4.2, +1.4] | 5.46 | 598 | 2785 | 2754 | 0.48 | 7.63 | 131.1 | 9.54 h | 84.4 min | 0.029% |
| GPQA-Diamond | base, Phase 1 records | 198 | 36.9 | reference | 12.00 | 1090 | 2540 | 2518 | 0.30 | 10.00 (Phase 1 machine) | 100.0 | 2.96 h | · | · |
| GPQA-Diamond | rd1 | 198 | 37.4 | +0.5 [−6.6, +8.1] | 9.21 | 855 | 2532 | 2503 | 1.06 | 5.21 | 192.0 | 5.86 h | 49.3 min | 0.033% |
| GPQA-Diamond | rd2 | 198 | 37.9 | +1.0 [−5.6, +7.6] | 9.56 | 884 | 2535 | 2508 | 0.69 | 5.22 | 191.6 | 5.67 h | 48.3 min | 0.034% |
| AIME | base, Phase 1 records | 90 | 5.6 | reference | 7.87 | 789 | 2738 | 2717 | 0.29 | 14.42 (Phase 1 machine) | 69.4 | 1.33 h | · | · |
| AIME | rd1 | 90 | 5.6 | 0.0 [−3.3, +3.3] | 6.38 | 646 | 2704 | 2674 | 0.74 | 6.98 | 143.2 | 2.89 h | 24.5 min | 0.030% |
| AIME | rd2 | 90 | 5.6 | 0.0 [−4.4, +4.4] | 6.53 | 663 | 2714 | 2687 | 0.52 | 7.21 | 138.7 | 2.73 h | 23.8 min | 0.032% |
| HumanEval | base, Phase 1 records | 164 | 74.4 | reference | 5.91 | 739 | 2600 | 2485 | 0.36 | 14.59 (Phase 1 machine) | 68.5 | 0.33 h | · | · |
| HumanEval | rd1 | 164 | 75.6 | +1.2 [−4.9, +7.3] | 4.58 | 578 | 2622 | 2507 | 0.70 | 7.77 | 128.7 | 0.78 h | 8.2 min | 0.028% |
| HumanEval | rd2 | 164 | 79.9 | +5.5 [0.0, +11.0] | 4.65 | 627 | 2632 | 2493 | 0.51 | 7.48 | 133.7 | 0.63 h | 7.0 min | 0.030% |
| MBPP | base, Phase 1 records | 257 | 66.1 | reference | 8.75 | 1285 | 2566 | 2451 | 0.44 | 8.54 (Phase 1 machine) | 117.2 | 0.45 h | · | · |
| MBPP | rd1 | 257 | 71.2 | +5.1 [0.0, +10.1] | 7.17 | 1062 | 2533 | 2393 | 1.32 | 4.18 | 239.1 | 0.95 h | 8.9 min | 0.031% |
| MBPP | rd2 | 257 | 69.3 | +3.1 [−1.6, +7.8] | 7.30 | 1122 | 2583 | 2447 | 0.82 | 4.18 | 239.4 | 0.90 h | 8.1 min | 0.032% |
| GSM8K-500 | base, Phase 1 records | 500 | 91.6 | reference | 7.12 | 889 | 2910 | 2872 | 0.33 | 12.58 (Phase 1 machine) | 79.5 | 1.88 h | · | · |
| GSM8K-500 | rd1 | 500 | 91.6 | 0.0 [−1.4, +1.4] | 5.93 | 755 | 2896 | 2848 | 0.77 | 6.10 | 163.9 | 3.94 h | 31.4 min | 0.028% |
| GSM8K-500 | rd2 | 500 | 92.0 | +0.4 [−1.2, +1.8] | 6.03 | 770 | 2900 | 2855 | 0.55 | 6.18 | 161.8 | 3.88 h | 31.3 min | 0.029% |

## Table 2: raw ratios against the base pipeline (same items)

| Benchmark | Setup | Paired items | Actual speedup, tok/s | Actual speedup, total time | GPU-only speedup, tok/s | GPU-only speedup, GPU time | H (30B prefill) | H drafter prefill | H 1st half | H 2nd half | Base reference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IFEval-541 | rd1 | 541 | 1.192x | 0.986x | 1.173x | 0.970x | 1.007 | · | · | · | base on THIS machine (same-machine speedup) |
| IFEval-541 | rd1 | 541 | 0.529x | 0.438x | · | · | 2.193 | 2.218 | 2.187 | 2.200 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| IFEval-541 | rd2 | 541 | 1.173x | 1.063x | 1.154x | 1.045x | 0.992 | · | · | · | base on THIS machine (same-machine speedup) |
| IFEval-541 | rd2 | 541 | 0.521x | 0.472x | · | · | 2.162 | 2.186 | 2.156 | 2.168 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| MATH-500 | rd1 | 500 | 1.081x | 1.015x | 1.090x | 1.023x | 1.024 | · | · | · | base on THIS machine (same-machine speedup) |
| MATH-500 | rd1 | 500 | 0.503x | 0.472x | · | · | 2.135 | 2.150 | 2.140 | 2.131 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| MATH-500 | rd2 | 500 | 1.042x | 1.033x | 1.088x | 1.079x | 1.052 | · | · | · | base on THIS machine (same-machine speedup) |
| MATH-500 | rd2 | 500 | 0.485x | 0.481x | · | · | 2.192 | 2.203 | 2.195 | 2.188 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| GPQA-Diamond | rd1 | 198 | 1.156x | 1.119x | 1.111x | 1.075x | 0.982 | · | · | · | base on THIS machine (same-machine speedup) |
| GPQA-Diamond | rd1 | 198 | 0.521x | 0.504x | · | · | 2.074 | 1.943 | 2.057 | 2.092 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| GPQA-Diamond | rd2 | 198 | 1.158x | 1.158x | 1.138x | 1.138x | 0.993 | · | · | · | base on THIS machine (same-machine speedup) |
| GPQA-Diamond | rd2 | 198 | 0.522x | 0.522x | · | · | 2.098 | 1.964 | 2.082 | 2.116 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| AIME | rd1 | 90 | 1.109x | 1.051x | 1.092x | 1.035x | 1.003 | · | · | · | base on THIS machine (same-machine speedup) |
| AIME | rd1 | 90 | 0.485x | 0.459x | · | · | 2.271 | 2.239 | 2.327 | 2.211 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| AIME | rd2 | 90 | 1.145x | 1.112x | 1.116x | 1.084x | 0.982 | · | · | · | base on THIS machine (same-machine speedup) |
| AIME | rd2 | 90 | 0.500x | 0.486x | · | · | 2.223 | 2.202 | 2.277 | 2.166 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| HumanEval | rd1 | 164 | 1.190x | 0.955x | 1.158x | 0.930x | 0.987 | · | · | · | base on THIS machine (same-machine speedup) |
| HumanEval | rd1 | 164 | 0.532x | 0.427x | · | · | 2.101 | 2.106 | 2.125 | 2.077 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| HumanEval | rd2 | 164 | 1.146x | 1.178x | 1.121x | 1.153x | 0.985 | · | · | · | base on THIS machine (same-machine speedup) |
| HumanEval | rd2 | 164 | 0.513x | 0.527x | · | · | 2.098 | 2.104 | 2.126 | 2.071 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| MBPP | rd1 | 257 | 1.095x | 1.059x | 1.067x | 1.032x | 0.993 | · | · | · | base on THIS machine (same-machine speedup) |
| MBPP | rd1 | 257 | 0.490x | 0.474x | · | · | 2.132 | 2.133 | 2.150 | 2.114 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| MBPP | rd2 | 257 | 1.093x | 1.121x | 1.080x | 1.107x | 0.996 | · | · | · | base on THIS machine (same-machine speedup) |
| MBPP | rd2 | 257 | 0.489x | 0.502x | · | · | 2.138 | 2.144 | 2.157 | 2.118 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| GSM8K-500 | rd1 | 500 | 1.093x | 1.075x | 1.074x | 1.056x | 0.996 | · | · | · | base on THIS machine (same-machine speedup) |
| GSM8K-500 | rd1 | 500 | 0.485x | 0.477x | · | · | 2.193 | 2.223 | 2.182 | 2.204 | Phase 1 records (OTHER machine: raw cross-machine ratio) |
| GSM8K-500 | rd2 | 500 | 1.107x | 1.093x | 1.095x | 1.081x | 0.996 | · | · | · | base on THIS machine (same-machine speedup) |
| GSM8K-500 | rd2 | 500 | 0.491x | 0.485x | · | · | 2.192 | 2.225 | 2.181 | 2.203 | Phase 1 records (OTHER machine: raw cross-machine ratio) |

## Re-drafting vs swapping the drafter (Phase 2)

rd1 and rd2 keep the base drafter (SDAR-1.7B-Chat-b32) and add re-drafting; Phase 2 swapped the drafter instead. Speedups are over the SDAR-1.7B + judge base: rd1 / rd2 measured on this machine against the base run here; CoDA / Qwen3 from the Phase 2 page (Phase 1 machine; computed by that page's method from expert loads). Accuracy, passes per block and expert loads per token do not depend on the machine.

| Benchmark | CoDA-1.7B speedup | Qwen3-0.6B speedup | rd1 speedup (actual / GPU-only) | rd2 speedup (actual / GPU-only) | Expert loads / token: base / CoDA / Qwen3 / rd1 / rd2 | Accuracy %: base / CoDA / Qwen3 / rd1 / rd2 |
|---|---:|---:|---:|---:|---|---|
| IFEval-541 | 1.001x | 0.924x | 1.192x / 1.173x | 1.173x / 1.154x | 1230 / 1220 / 1324 / 894 / 974 | 51.0 / 49.9 / 51.4 / 46.6 / 48.1 |
| MATH-500 | 0.904x | 0.936x | 1.081x / 1.090x | 1.042x / 1.088x | 688 / 755 / 729 / 571 / 598 | 71.6 / 69.8 / 71.2 / 70.8 / 70.2 |
| GPQA-Diamond | 0.949x | 0.963x | 1.156x / 1.111x | 1.158x / 1.138x | 1090 / 1137 / 1125 / 855 / 884 | 36.9 / 38.9 / 34.8 / 37.4 / 37.9 |
| AIME | 0.887x | 0.951x | 1.109x / 1.092x | 1.145x / 1.116x | 789 / 880 / 823 / 646 / 663 | 5.6 / 7.8 / 6.7 / 5.6 / 5.6 |
| HumanEval | 1.032x | 1.017x | 1.190x / 1.158x | 1.146x / 1.121x | 739 / 712 / 716 / 578 / 627 | 74.4 / 76.2 / 78.0 / 75.6 / 79.9 |
| MBPP | 0.966x | 0.923x | 1.095x / 1.067x | 1.093x / 1.080x | 1285 / 1331 / 1383 / 1062 / 1122 | 66.1 / 67.3 / 68.5 / 71.2 / 69.3 |
| GSM8K-500 | 0.911x | 0.936x | 1.093x / 1.074x | 1.107x / 1.095x | 889 / 976 / 942 / 755 / 770 | 91.6 / 91.0 / 91.6 / 91.6 / 92.0 |

## Against the plain 30B MoE (stock)

Exact (machine-independent): accuracy paired with the stock records, and 30B expert loads per token (stock / setup = times less work). Wall-clock speedup over stock is CHAINED: base over stock from the Phase 2 page (Phase 1 machine; stock timed there on IFEval only, computed from expert loads elsewhere) x rd over base measured on this machine. It is an estimate, not a single-machine measurement.

| Benchmark | Stock acc | rd1 acc (Δ vs stock [95% CI]) | rd2 acc (Δ vs stock [95% CI]) | Less 30B work than stock, rd1 / rd2 | Base over stock (Phase 2) | rd1 over stock (chained) | rd2 over stock (chained) |
|---|---:|---|---|---:|---:|---:|---:|
| IFEval-541 | 52.7 | 46.6 (−6.1 [−10.2, −1.8]) | 48.1 (−4.6 [−8.5, −0.7]) | 1.76x / 1.62x | 1.250x | 1.489x | 1.466x |
| MATH-500 | 72.4 | 70.8 (−1.6 [−4.8, +1.6]) | 70.2 (−2.2 [−5.0, +0.8]) | 1.49x / 1.43x | 1.186x | 1.282x | 1.236x |
| GPQA-Diamond | 37.4 | 37.4 (0.0 [−7.6, +8.1]) | 37.9 (+0.5 [−6.6, +7.1]) | 1.62x / 1.56x | 1.231x | 1.422x | 1.425x |
| AIME | 6.7 | 5.6 (−1.1 [−5.6, +2.2]) | 5.6 (−1.1 [−3.3, 0.0]) | 1.54x / 1.51x | 1.215x | 1.347x | 1.391x |
| HumanEval | 78.0 | 75.6 (−2.4 [−9.1, +3.7]) | 79.9 (+1.8 [−3.1, +6.7]) | 1.70x / 1.57x | 1.263x | 1.502x | 1.447x |
| MBPP | 65.4 | 71.2 (+5.8 [0.0, +11.7]) | 69.3 (+3.9 [−1.9, +9.7]) | 1.61x / 1.53x | 1.286x | 1.408x | 1.406x |
| GSM8K-500 | 90.2 | 91.6 (+1.4 [−0.6, +3.4]) | 92.0 (+1.8 [0.0, +3.8]) | 1.54x / 1.51x | 1.260x | 1.377x | 1.395x |
| All 7 (pooled) | 65.5 | 64.4 | 64.9 | 1.59x / 1.51x | 1.214x | 1.368x | 1.355x |

Same-machine check: stock was decoded on this machine for 13 GSM8K items during setup; on those items base is 1.214x, rd1 1.335x and rd2 1.371x faster than stock (measured).

## Host during each run, and the background-job check

| Benchmark | Setup | CPU throttled periods | Max GPU throttle time (s) | Items overlapping a backup / page job | Judge-pass time, overlapping vs other |
|---|---|---:|---:|---:|---:|
| IFEval-541 | rd1 | 2 | 0.0 | 142 of 541 | +1.05% |
| IFEval-541 | rd2 | 1 | 0.0 | 138 of 541 | −0.23% |
| MATH-500 | rd1 | 2 | 0.0 | 123 of 500 | +0.40% |
| MATH-500 | rd2 | 2 | 0.0 | 123 of 500 | −0.33% |
| GPQA-Diamond | rd1 | 1 | 0.0 | 61 of 198 | +0.07% |
| GPQA-Diamond | rd2 | 1 | 0.0 | 64 of 198 | +0.79% |
| AIME | rd1 | 1 | 0.0 | 27 of 90 | +1.03% |
| AIME | rd2 | 2 | 0.0 | 38 of 90 | +0.66% |
| HumanEval | rd1 | 2 | 0.0 | 27 of 164 | +0.28% |
| HumanEval | rd2 | 2 | 0.0 | 11 of 164 | +1.73% |
| MBPP | rd1 | 2 | 0.0 | 38 of 257 | −1.14% |
| MBPP | rd2 | 2 | 0.0 | 37 of 257 | +2.67% |
| GSM8K-500 | rd1 | 3 | 0.0 | 70 of 500 | +0.62% |
| GSM8K-500 | rd2 | 1 | 0.0 | 72 of 500 | +0.04% |

