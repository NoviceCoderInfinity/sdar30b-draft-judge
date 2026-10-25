# Evaluation table (canonical)

Generated 2026-09-09 05:27:46 UTC (10:57 IST) by `code/draftalign/eval_table.py` at commit `edaabad` from the record files under `logs/draftalign/da_decode/`. Re-run the script after any evaluation; do not edit by hand.

## Definitions

```
 
  acc          exact-match accuracy of the final answer (the branch's GSM8K/MATH/MBPP scorer), percent
  d_acc        paired accuracy difference vs the STOCK decode on the SAME items, percentage points, with a 95% paired
               bootstrap interval (2,000 resamples over items); "vs free" = the same against the training-free rule
  fwd/blk      MoE forwards per 32-token block, flush excluded, block-weighted = sum(passes) / sum(blocks) (analyze_da's pass/blk)
  fwd/blk(im)  the same as a mean over items of each item's own mean (the per-item definition used in chat tables)
  exp/tok      expert-layer executions per generated token = sum(unique_expert_layer_loads) / sum(generated tokens)
  ms/tok       wall-clock milliseconds per generated token = 1000 * sum(seconds) / sum(generated tokens)
  agree        drafter/final-token agreement, mean over blocks with a draft
  items        records on disk (a partial cell is scored on the items finished so far); shards = .done markers of 4 (or 1)
```

## GSM8K calibvalid = 100 frozen GSM8K TRAIN prompts (50 calibration + 50 validation), never in any training set

| judge | rule | drafter | items | shards | acc % | d_acc vs stock | d_acc vs free rule | fwd/blk | fwd/blk (im) | exp/tok | ms/tok | agree | cell |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| — | stock | — | 100 | 4/4 | 95.0 |  |  | 9.31 | 9.02 | 1091 | 187 |  | `gsm8k_B32_t95_stock` |
| — | stock + p32desf | — | 100 | 4/4 | 93.0 | -2.0 [-7.0, +3.0] (n=100) | -2.0 [-7.0, +3.0] (n=100) | 9.94 | 9.65 | 655 | 127 |  | `gsm8k_B32_t95_stock_p32desf` |
| — | stock + p48desf | — | 100 | 4/4 | 94.0 | -1.0 [-5.0, +2.0] (n=100) | -1.0 [-5.0, +2.0] (n=100) | 9.27 | 9.05 | 872 | 161 |  | `gsm8k_B32_t95_stock_p48desf` |
| — | blind_q30 | stock | 100 | 4/4 | 88.0 | -7.0 [-13.0, -3.0] (n=100) | -7.0 [-12.0, -2.0] (n=100) | 6.98 | 6.98 | 852 | 157 | 0.620 | `gsm8k_B32_t95_dv-s8_blind_q30` |
| — | blind_q40 | stock | 100 | 4/4 | 94.0 | -1.0 [-3.0, +0.0] (n=100) | -1.0 [-3.0, +0.0] (n=100) | 7.74 | 7.54 | 950 | 175 | 0.558 | `gsm8k_B32_t95_dv-s8_blind_q40` |
| — | blind_q50 | stock | 100 | 4/4 | 95.0 | +0.0 [+0.0, +0.0] (n=100) |  | 7.93 | 7.69 | 963 | 172 | 0.540 | `gsm8k_B32_t95_dv-s8_blind_q50` |
| — | blind_q50_da1 | stock | 100 | 4/4 | 91.0 | -4.0 [-9.0, +0.0] (n=100) | -4.0 [-9.0, +0.0] (n=100) | 8.14 | 7.84 | 972 | 179 | 0.661 | `gsm8k_B32_t95_dv-s8_blind_q50_da1` |
| — | blind_q50_rj | stock | 100 | 4/4 | 91.0 | -4.0 [-9.0, +0.0] (n=100) | -4.0 [-9.0, +0.0] (n=100) | 6.62 | 6.37 | 818 | 148 | 0.574 | `gsm8k_B32_t95_dv-s8_blind_q50_rj` |
| — | c95_a70_q50 | stock | 100 | 4/4 | 67.0 | -28.0 [-36.0, -19.0] (n=100) | -28.0 [-38.0, -19.0] (n=100) | 2.01 | 2.02 | 575 | 107 | 0.919 | `gsm8k_B32_t95_dv-s8_c95_a70_q50` |
| — | full_c95 | stock | 100 | 4/4 | 44.0 | -51.0 [-61.0, -42.0] (n=100) | -51.0 [-60.0, -42.0] (n=100) | 5.93 | 6.52 | 737 | 137 | 0.265 | `gsm8k_B32_t95_dv-s8_full_c95` |
| T1 step 600 | full_c95 | stock | 100 | 4/4 | 38.0 | -57.0 [-67.0, -46.0] (n=100) | -57.0 [-67.0, -47.0] (n=100) | 19.57 | 18.92 | 1909 | 346 | 0.572 | `gsm8k_B32_t95_dv-s8_full_c95_ad5d9add10` |
| T2a step 200 | full_c90_o99 | stock | 100 | 4/4 | 85.0 | -10.0 [-17.0, -4.0] (n=100) | -10.0 [-17.0, -4.0] (n=100) | 12.33 | 12.05 | 1484 | 277 | 0.521 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ad6fd3c2e9` |
| T2a step 400 | full_c90_o99 | stock | 100 | 4/4 | 93.0 | -2.0 [-6.0, +2.0] (n=100) | -2.0 [-6.0, +2.0] (n=100) | 8.70 | 8.40 | 1079 | 201 | 0.566 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ad7d6e7e82` |
| T2a step 600 | full_c90_o99 | stock | 100 | 4/4 | 93.0 | -2.0 [-6.0, +2.0] (n=100) | -2.0 [-6.0, +2.0] (n=100) | 8.23 | 8.06 | 1026 | 192 | 0.583 | `gsm8k_B32_t95_dv-s8_full_c90_o99_adbb1f79ea` |
| T2a step 800 | full_c90_o99 | stock | 100 | 4/4 | 94.0 | -1.0 [-3.0, +0.0] (n=100) | -1.0 [-3.0, +0.0] (n=100) | 7.90 | 7.70 | 992 | 185 | 0.579 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ad329be7a3` |
| T2a step 1,057 | full_c90 | stock | 100 | 4/4 | 89.0 | -6.0 [-12.0, -1.0] (n=100) | -6.0 [-11.0, -1.0] (n=100) | 5.75 | 5.55 | 747 | 139 | 0.554 | `gsm8k_B32_t95_dv-s8_full_c90_adb3106a4d` |
| T2a step 1,057 | full_c90_o99 | stock | 100 | 4/4 | 94.0 | -1.0 [-4.0, +2.0] (n=100) | -1.0 [-4.0, +2.0] (n=100) | 7.93 | 7.75 | 995 | 182 | 0.577 | `gsm8k_B32_t95_dv-s8_full_c90_o99_adb3106a4d` |
| T2a step 1,057 | full_c90_o99_ro | stock | 100 | 4/4 | 92.0 | -3.0 [-7.0, +0.0] (n=100) | -3.0 [-7.0, +0.0] (n=100) | 8.75 | 8.48 | 1067 | 196 | 0.547 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ro_adb3106a4d` |
| T2a step 1,057 | full_c90_ro | stock | 100 | 4/4 | 91.0 | -4.0 [-9.0, +1.0] (n=100) | -4.0 [-9.0, +0.0] (n=100) | 6.51 | 6.24 | 826 | 149 | 0.511 | `gsm8k_B32_t95_dv-s8_full_c90_ro_adb3106a4d` |
| T2a step 1,057 | full_c95 | stock | 100 | 4/4 | 91.0 | -4.0 [-9.0, +0.0] (n=100) | -4.0 [-9.0, +1.0] (n=100) | 6.90 | 6.67 | 878 | 164 | 0.539 | `gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d` |
| T2a step 1,057 | full_c95_o100 | stock | 100 | 4/4 | 93.0 | -2.0 [-6.0, +2.0] (n=100) | -2.0 [-6.0, +2.0] (n=100) | 8.87 | 8.67 | 1098 | 198 | 0.575 | `gsm8k_B32_t95_dv-s8_full_c95_o100_adb3106a4d` |
| T2a step 1,057 | full_c95_o99 | stock | 100 | 4/4 | 93.0 | -2.0 [-5.0, +0.0] (n=100) | -2.0 [-5.0, +0.0] (n=100) | 8.26 | 8.04 | 1032 | 192 | 0.572 | `gsm8k_B32_t95_dv-s8_full_c95_o99_adb3106a4d` |
| T2a step 1,057 | full_c95_ro | stock | 100 | 4/4 | 91.0 | -4.0 [-8.0, -1.0] (n=100) | -4.0 [-8.0, -1.0] (n=100) | 7.90 | 7.55 | 966 | 179 | 0.516 | `gsm8k_B32_t95_dv-s8_full_c95_ro_adb3106a4d` |
| T2b step 2,200 | full_c90_o99 | stock | 100 | 4/4 | 93.0 | -2.0 [-7.0, +3.0] (n=100) | -2.0 [-7.0, +2.0] (n=100) | 7.10 | 6.98 | 902 | 169 | 0.589 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ada7be11a8` |
| T2b step 2,400 | full_c90_o99 | stock | 100 | 4/4 | 93.0 | -2.0 [-5.0, +0.0] (n=100) | -2.0 [-5.0, +0.0] (n=100) | 7.29 | 7.05 | 919 | 168 | 0.576 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ad022382b2` |
| T2b step 2,600 | full_c90_o99 | stock | 100 | 4/4 | 91.0 | -4.0 [-9.0, +0.0] (n=100) | -4.0 [-9.0, +0.0] (n=100) | 7.24 | 7.03 | 918 | 172 | 0.579 | `gsm8k_B32_t95_dv-s8_full_c90_o99_adf51dafdc` |
| T2b step 2,600 | full_c90_o99 | fine-tuned d1 step 1,080 | 100 | 4/4 | 91.0 | -4.0 [-10.0, +1.0] (n=100) | -4.0 [-9.0, +1.0] (n=100) | 6.93 | 6.71 | 883 | 161 | 0.653 | `gsm8k_B32_t95_dv-s8_full_c90_o99_dra5250149_adb3106a4d` |
| T2b step 3,057 | full_c90 | stock | 100 | 4/4 | 90.0 | -5.0 [-10.0, -1.0] (n=100) | -5.0 [-9.0, -1.0] (n=100) | 4.92 | 4.73 | 665 | 125 | 0.557 | `gsm8k_B32_t95_dv-s8_full_c90_ad734af656` |
| T2b step 3,057 | full_c90_o99 | stock | 100 | 4/4 | 89.0 | -6.0 [-11.0, -2.0] (n=100) | -6.0 [-11.0, -2.0] (n=100) | 7.26 | 7.09 | 921 | 168 | 0.580 | `gsm8k_B32_t95_dv-s8_full_c90_o99_ad734af656` |
| T2b step 3,057 | full_c95 | stock | 100 | 4/4 | 94.0 | -1.0 [-4.0, +2.0] (n=100) | -1.0 [-5.0, +2.0] (n=100) | 6.00 | 5.85 | 781 | 148 | 0.543 | `gsm8k_B32_t95_dv-s8_full_c95_ad734af656` |
| T2b step 3,057 | full_c95_o99 | stock | 100 | 4/4 | 92.0 | -3.0 [-8.0, +1.0] (n=100) | -3.0 [-8.0, +1.0] (n=100) | 7.74 | 7.56 | 968 | 180 | 0.553 | `gsm8k_B32_t95_dv-s8_full_c95_o99_ad734af656` |
| T2b step 3,057 | full_c95_p32desf | stock | 100 | 4/4 | 88.0 | -7.0 [-13.0, -2.0] (n=100) | -7.0 [-13.0, -2.0] (n=100) | 6.67 | 6.46 | 464 | 101 | 0.541 | `gsm8k_B32_t95_dv-s8_full_c95_p32desf_ad734af656` |
| T2b step 3,057 | full_c95_p32desfcv60 | stock | 100 | 4/4 | 87.0 | -8.0 [-14.0, -3.0] (n=100) | -8.0 [-14.0, -2.0] (n=100) | 6.78 | 6.56 | 470 | 104 | 0.542 | `gsm8k_B32_t95_dv-s8_full_c95_p32desfcv60_ad734af656` |
| T2b step 3,057 | full_c95_p32desfcv75 | stock | 50 | 2/4 | 88.0 | -4.0 [-14.0, +6.0] (n=50) | -4.0 [-14.0, +6.0] (n=50) | 8.50 | 8.14 | 564 | 125 | 0.528 | `gsm8k_B32_t95_dv-s8_full_c95_p32desfcv75_ad734af656` |
| T2b step 3,057 | full_c95_p32desfcv90 | stock | 58 | 2/4 | 86.2 | -6.9 [-15.5, +0.0] (n=58) | -6.9 [-15.5, +0.0] (n=58) | 17.74 | 17.28 | 1095 | 236 | 0.528 | `gsm8k_B32_t95_dv-s8_full_c95_p32desfcv90_ad734af656` |
| T2b step 3,057 | full_c95_p48desf | stock | 100 | 4/4 | 94.0 | -1.0 [-5.0, +2.0] (n=100) | -1.0 [-5.0, +2.0] (n=100) | 6.09 | 5.89 | 613 | 124 | 0.536 | `gsm8k_B32_t95_dv-s8_full_c95_p48desf_ad734af656` |

## GSM8K-500 = 500 frozen GSM8K TEST items (stock decode reference on this split: 90.2 %, 10.06 fwd/blk, 1,164 exp/tok, 198 ms/tok — the analyze_da reference cell, recorded outside da_decode/)

| judge | rule | drafter | items | shards | acc % | d_acc vs stock | d_acc vs free rule | fwd/blk | fwd/blk (im) | exp/tok | ms/tok | agree | cell |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| — | stock + p32desf | — | 500 | 4/4 | 89.0 |  | -1.4 [-3.8, +1.0] (n=500) | 10.46 | 10.33 | 676 | 132 |  | `gsm8k_B32_t95_stock_p32desf` |
| — | stock + p48desf | — | 500 | 4/4 | 91.2 |  | +0.8 [-1.2, +3.0] (n=500) | 9.96 | 9.84 | 920 | 171 |  | `gsm8k_B32_t95_stock_p48desf` |
| — | blind_q50 | stock | 500 | 4/4 | 90.4 |  |  | 8.49 | 8.39 | 1017 | 184 | 0.543 | `gsm8k_B32_t95_dv-s8_blind_q50` |
| T2a step 1,057 | full_c90_o99 | stock | 158 | 0/4 | 89.2 |  | -2.5 [-7.0, +1.3] (n=158) | 8.69 | 8.63 | 1083 | 200 | 0.568 | `gsm8k_B32_t95_dv-s8_full_c90_o99_adb3106a4d` |
| T2a step 1,057 | full_c90_ro | stock | 382 | 3/4 | 89.0 |  | -2.4 [-5.5, +0.8] (n=382) | 6.82 | 6.73 | 858 | 160 | 0.517 | `gsm8k_B32_t95_dv-s8_full_c90_ro_adb3106a4d` |
| T2b step 3,057 | full_c95 | stock | 500 | 4/4 | 88.8 |  | -1.6 [-3.8, +0.6] (n=500) | 6.64 | 6.58 | 849 | 159 | 0.539 | `gsm8k_B32_t95_dv-s8_full_c95_ad734af656` |
| T2b step 3,057 | full_c95_p32desf | stock | 500 | 4/4 | 88.0 |  | -2.4 [-5.0, +0.4] (n=500) | 7.24 | 7.15 | 492 | 106 | 0.528 | `gsm8k_B32_t95_dv-s8_full_c95_p32desf_ad734af656` |
| T2b step 3,057 | full_c95_p48desf | stock | 500 | 4/4 | 87.8 |  | -2.6 [-5.0, -0.2] (n=500) | 6.76 | 6.70 | 668 | 134 | 0.546 | `gsm8k_B32_t95_dv-s8_full_c95_p48desf_ad734af656` |

## MATH-500 = 500 test items

| judge | rule | drafter | items | shards | acc % | d_acc vs stock | d_acc vs free rule | fwd/blk | fwd/blk (im) | exp/tok | ms/tok | agree | cell |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| — | stock | — | 500 | 4/4 | 72.4 |  |  | 8.52 | 8.94 | 854 | 150 |  | `math500_B32_t95_stock` |
| — | stock + p48desf | — | 110 | 0/4 | 72.7 | -1.8 [-8.2, +4.5] (n=110) | -6.4 [-13.6, +0.9] (n=110) | 8.64 | 9.17 | 711 | 134 |  | `math500_B32_t95_stock_p48desf` |
| — | blind_q50 | stock | 500 | 4/4 | 70.8 | -1.6 [-4.6, +1.4] (n=500) |  | 7.40 | 7.79 | 764 | 142 | 0.645 | `math500_B32_t95_dv-s8_blind_q50` |
| T2a step 1,057 | full_c90_o99 | stock | 44 | 0/4 | 63.6 | -11.4 [-25.0, +2.3] (n=44) | -13.6 [-27.3, +0.0] (n=44) | 10.32 | 10.19 | 1042 | 190 | 0.618 | `math500_B32_t95_dv-s8_full_c90_o99_ad191ea8d6` |
| T2a step 1,057 | full_c90_o99 | stock | 277 | 2/4 | 57.8 | -13.0 [-17.7, -8.3] (n=277) | -10.5 [-15.2, -5.4] (n=277) | 11.54 | 11.56 | 1175 | 219 | 0.579 | `math500_B32_t95_dv-s8_full_c90_o99_adb3106a4d` |
| T2b step 3,057 | full_c95 | stock | 500 | 4/4 | 68.0 | -4.4 [-7.6, -1.0] (n=500) | -2.8 [-6.4, +0.8] (n=500) | 6.29 | 6.59 | 680 | 127 | 0.652 | `math500_B32_t95_dv-s8_full_c95_ad734af656` |
| T2b step 3,057 | full_c95_p32desf | stock | 158 | 0/4 | 65.2 | -7.0 [-13.3, -0.6] (n=158) | -10.8 [-17.1, -4.4] (n=158) | 6.74 | 7.39 | 398 | 88 | 0.654 | `math500_B32_t95_dv-s8_full_c95_p32desf_ad734af656` |
| T2b step 3,057 | full_c95_p48desf | stock | 500 | 4/4 | 69.2 | -3.2 [-6.8, +0.2] (n=500) | -1.6 [-5.2, +2.0] (n=500) | 6.39 | 6.77 | 549 | 111 | 0.650 | `math500_B32_t95_dv-s8_full_c95_p48desf_ad734af656` |

## MBPP sanitized test = 257 items

| judge | rule | drafter | items | shards | acc % | d_acc vs stock | d_acc vs free rule | fwd/blk | fwd/blk (im) | exp/tok | ms/tok | agree | cell |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| — | stock | — | 257 | 1/1 | 65.4 |  |  | 11.93 | 12.28 | 1714 | 296 |  | `mbpp_B32_t95_stock` |
| — | stock + p32desf | — | 257 | 4/4 | 65.0 | -0.4 [-6.2, +4.7] (n=257) | +4.7 [-0.8, +10.5] (n=257) | 12.30 | 12.63 | 1130 | 217 |  | `mbpp_B32_t95_stock_p32desf` |
| — | stock + p48desf | — | 257 | 4/4 | 63.8 | -1.6 [-5.8, +3.1] (n=257) | +3.5 [-1.2, +8.2] (n=257) | 12.09 | 12.36 | 1482 | 273 |  | `mbpp_B32_t95_stock_p48desf` |
| — | blind_q50 | stock | 257 | 4/4 | 60.3 | -5.1 [-8.6, -1.6] (n=257) |  | 10.20 | 10.37 | 1524 | 279 | 0.606 | `mbpp_B32_t95_dv-s8_blind_q50` |
| T2b step 3,057 | full_c95 | stock | 257 | 4/4 | 51.4 | -14.0 [-20.2, -7.4] (n=257) | -8.9 [-14.8, -3.1] (n=257) | 10.43 | 10.70 | 1588 | 301 | 0.626 | `mbpp_B32_t95_dv-s8_full_c95_ad734af656` |
| T2b step 3,057 | full_c95_p32desf | stock | 257 | 4/4 | 55.6 | -9.7 [-15.6, -3.1] (n=257) | -4.7 [-11.3, +1.9] (n=257) | 10.74 | 10.97 | 1007 | 210 | 0.632 | `mbpp_B32_t95_dv-s8_full_c95_p32desf_ad734af656` |
| T2b step 3,057 | full_c95_p48desf | stock | 257 | 4/4 | 53.7 | -11.7 [-18.3, -5.4] (n=257) | -6.6 [-13.2, -0.4] (n=257) | 10.77 | 10.92 | 1367 | 272 | 0.645 | `mbpp_B32_t95_dv-s8_full_c95_p48desf_ad734af656` |

## Drafter fine-tune (SDAR-1.7B-Chat-b32, full fine-tune on the 30B's own answers) — gates

| drafter checkpoint | slot agreement vs stock (8-step draft replayed on the 30B's recorded blocks, paired held-out items) | standalone accuracy on calibvalid (drafter alone, block 32, thr .95) | passes/block alone |
|---|---|---|---:|
| d1 step 1,080 | 0.556 vs 0.579 (item-mean -0.007 [-0.028, +0.015], better/worse 41/57, n=99) | 78 / 100 | 10.66 |
| d1 step 1,000 | 0.561 vs 0.579 (item-mean -0.008 [-0.030, +0.015], better/worse 39/57, n=99) | 81 / 100 | 10.68 |
| stock d32 | — (baseline) | 87 / 100 | 13.00 |
