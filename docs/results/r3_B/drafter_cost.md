# Drafter cost, exact, from the decode records (18 Sep 2026)

Every arm below is paired against **stock + DES48** on the same items. `30B` = unique expert-layer loads per generated
token (the paper's speed metric). `drafter` = drafter forwards per generated token (`drafter_n_by_kind`, all kinds)
x 364.64 expert-equivalents per forward (AGENTS.md section 8). `30B-only` divides stock+DES48's loads by the arm's 30B loads;
`total` divides by 30B + drafter. Records: `Nemotron-split-moe-train/r3_B/evals/step000710`, `reference_evals/da_decode`,
`r3_A/evals/stock` on the public HF dataset.

| dataset | stock+DES48 loads/tok | arm | 30B | drafter (fwd/tok) | 30B-only | total |
|---|---|---|---|---|---|---|
| HumanEval | 822.9 | Nemotron judge + DES48 | 657.0 | 136.3 (0.374) | 1.25x | **1.04x** |
| HumanEval | 822.9 | standard judge + DES48 | 533.6 | 123.9 (0.340) | 1.54x | 1.25x |
| HumanEval | 822.9 | blind accept | 782.5 | 133.1 (0.365) | 1.05x | 0.90x |
| MBPP | 1482.3 | Nemotron judge + DES48 | 1239.3 | 167.7 (0.460) | 1.20x | **1.05x** |
| MBPP | 1482.3 | standard judge + DES48 | 1366.9 | 167.1 (0.458) | 1.08x | 0.97x |
| MBPP | 1482.3 | blind accept | 1523.9 | 168.3 (0.462) | 0.97x | 0.88x |
| GSM8K-500 | 920.5 | Nemotron judge + DES48 | 755.5 | 122.3 (0.335) | 1.22x | **1.05x** |
| GSM8K-500 | 920.5 | standard judge + DES48 | 667.7 | 122.3 (0.335) | 1.38x | 1.17x |
| GSM8K-500 | 920.5 | blind accept | 1016.8 | 121.5 (0.333) | 0.91x | 0.81x |
| MATH-500 | 700.4 | Nemotron judge + DES48 | 620.1 | 108.7 (0.298) | 1.13x | **0.96x** |
| MATH-500 | 700.4 | standard judge + DES48 | 549.5 | 108.6 (0.298) | 1.27x | 1.06x |
| MATH-500 | 700.4 | blind accept | 764.0 | 108.8 (0.298) | 0.92x | 0.80x |
| GPQA-D (198) | 1256.5 | Nemotron judge + DES48 | 1112.5 | 109.0 (0.299) | 1.13x | **1.03x** |
| GPQA-D (198) | 1256.5 | standard judge + DES48 | 1120.0 | 108.8 (0.298) | 1.12x | 1.02x |
| GPQA-D (198) | 1256.5 | blind accept | 1348.4 | 108.5 (0.298) | 0.93x | 0.86x |
| AIME | 836.3 | Nemotron judge + DES48 | 725.2 | 106.2 (0.291) | 1.15x | **1.01x** |
| AIME | 836.3 | standard judge + DES48 | 672.9 | 106.1 (0.291) | 1.24x | 1.07x |
| AIME | 836.3 | blind accept | 927.3 | 106.2 (0.291) | 0.90x | 0.81x |

Reading: once the drafter's own forwards are counted, the Nemotron judge + DES48 is 0.96x to 1.05x of training-free
stock + DES48 on every set. The judge and drafter together are a net zero over the pool; the pool is the saving.
