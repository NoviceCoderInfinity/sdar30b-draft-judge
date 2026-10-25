Box ratio (this 4x box / old 8x box, same 16 items, identical outputs): total 1.870; components moe_prefill 1.600, moe_judge 1.897, moe_kvflush 1.893, dr 1.901, other 1.444
Leave-one-out per-item error of the old-box time model: mean +0.2%, sd 5.3%, max |10.4|%
rd time vs S0 on the old box, same items (pooled over finished sets): rd1 x0.938, rd2 x0.898

SAME-BOX: rd vs today's judge on identical items, both decoded on this box under 4-GPU load
  rd1/humaneval    items  164  rd 8.61 tok/s vs judge 7.23 [full baseline] -> 1.191x  => old box 17.38 tok/s (1.50x stock)
  rd1/mbpp         items  257  rd 5.03 tok/s vs judge 4.70 [full baseline] -> 1.069x  => old box 9.12 tok/s (1.37x stock)
  rd1/gsm8k500     items  500  rd 6.83 tok/s vs judge 6.91 [full baseline] -> 0.989x  => old box 12.44 tok/s (1.25x stock)
  rd2/humaneval    items  164  rd 8.58 tok/s vs judge 7.23 [full baseline] -> 1.187x  => old box 17.32 tok/s (1.50x stock)
  rd2/mbpp         items  257  rd 5.12 tok/s vs judge 4.70 [full baseline] -> 1.089x  => old box 9.29 tok/s (1.40x stock)
  rd2/gsm8k500     items  500  rd 6.85 tok/s vs judge 6.91 [full baseline] -> 0.991x  => old box 12.47 tok/s (1.25x stock)

| cell | n | acc (S0 / stock) | passes/blk (S0) | exp/tok (S0) | tok/s this box | tok/s OLD 8x box: HEADLINE [source] / cost model / single-ratio (S0, stock) | wall this box 4x | wall OLD 8x |
|---|---|---|---|---|---|---|---|---|
| rd1/humaneval | 164 | 75.6 (74.4 / 78.0) | 4.58 (5.91) | 578 (739) | 8.61 | **17.38** [same-box] / cost 17.38 / ratio 16.12 (14.6, 11.6) | 11.3 min | 3.4 min |
| rd1/mbpp | 257 | 71.2 (66.1 / 65.4) | 7.17 (8.75) | 1062 (1285) | 5.03 | **9.12** [same-box] / cost 9.45 / ratio 9.39 (8.5, 6.6) | 13.1 min | 4.0 min |
| rd1/gsm8k500 | 500 | 91.6 (91.6 / 90.2) | 5.93 (7.12) | 755 (889) | 6.83 | **12.44** [same-box] / cost 13.89 / ratio 12.87 (12.6, 10.0) | 64.4 min | 16.2 min |
| rd2/humaneval | 164 | 79.9 (74.4 / 78.0) | 4.65 (5.91) | 627 (739) | 8.58 | **17.32** [same-box] / cost 16.64 / ratio 16.03 (14.6, 11.6) | 9.7 min | 3.3 min |
| rd2/mbpp | 257 | 69.3 (66.1 / 65.4) | 7.30 (8.75) | 1122 (1285) | 5.12 | **9.29** [same-box] / cost 9.40 / ratio 9.56 (8.5, 6.6) | 11.8 min | 3.7 min |
| rd2/gsm8k500 | 500 | 92.0 (91.6 / 90.2) | 6.03 (7.12) | 770 (889) | 6.85 | **12.47** [same-box] / cost 14.04 / ratio 12.90 (12.6, 10.0) | 62.6 min | 16.1 min |

Projected decode wall, all 14 cells: this box 4x 17.31 h; OLD 8x box 4.60 h (excl. setup/backup; measured where complete, projected elsewhere)
  rd1/humaneval    4x   11.3 min   old8x    3.4 min   [measured]
  rd1/mbpp         4x   13.1 min   old8x    4.0 min   [measured]
  rd1/gsm8k500     4x   64.4 min   old8x   16.2 min   [measured]
  rd1/ifeval541    4x  177.6 min   old8x   47.5 min   [S0 old-box item times x 0.938]
  rd1/math500      4x  140.1 min   old8x   36.1 min   [S0 old-box item times x 0.938]
  rd1/gpqa         4x   86.5 min   old8x   22.8 min   [S0 old-box item times x 0.938]
  rd1/aime         4x   38.7 min   old8x   10.7 min   [S0 old-box item times x 0.938]
  rd2/humaneval    4x    9.7 min   old8x    3.3 min   [measured]
  rd2/mbpp         4x   11.8 min   old8x    3.7 min   [measured]
  rd2/gsm8k500     4x   62.6 min   old8x   16.1 min   [measured]
  rd2/ifeval541    4x  169.5 min   old8x   45.4 min   [S0 old-box item times x 0.898]
  rd2/math500      4x  133.6 min   old8x   34.5 min   [S0 old-box item times x 0.898]
  rd2/gpqa         4x   82.6 min   old8x   21.8 min   [S0 old-box item times x 0.898]
  rd2/aime         4x   37.0 min   old8x   10.3 min   [S0 old-box item times x 0.898]
