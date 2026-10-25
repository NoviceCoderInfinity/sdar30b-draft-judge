# Results

Every number below is recomputed from the per-item eval records in the data repo
([Anupam-Rawat-IITB/sdar30b-draft-judge-data](https://huggingface.co/datasets/Anupam-Rawat-IITB/sdar30b-draft-judge-data)),
all decoded with the plain HF harness (`code/draftalign/run_da_decode.py`, block 32, threshold 0.95, bf16), on the
same 2,250 questions: GSM8K-500, MATH-500, HumanEval (164), MBPP (257), GPQA-Diamond (198), AIME (90), IFEval-541.

Definitions (token- or block-weighted sums over items, never means of per-item means):

```
passes/blk = sum(passes_per_block) / sum(n_blocks)                          30B forwards per 32-token block
exp/token  = sum(unique_expert_layer_loads) / sum(n_gen_to_eos)              30B experts loaded per generated token
compute    = vanilla exp/token / (setup exp/token + drafter cost per token)  drafter charged at its parameter count
             in 30B-expert units (SDAR-1.7B / CoDA-1.7B: 364.6 per forward, Qwen3-0.6B: 126.3)
```

## 1. Accuracy (%)

| Setup | gsm8k500 | math500 | humaneval | mbpp | gpqa | aime | ifeval541 | all 7 |
|---|---|---|---|---|---|---|---|---|
| vanilla | 90.2 | 72.4 | 78.0 | 65.4 | 37.4 | 6.7 | 52.7 | 65.5 |
| phase1 | 91.6 | 71.6 | 74.4 | 66.1 | 36.9 | 5.6 | 51.0 | 65.0 |
| phase2-coda | 91.0 | 69.8 | 76.2 | 67.3 | 38.9 | 7.8 | 49.9 | 64.7 |
| phase2-qwen3 | 91.6 | 71.2 | 78.0 | 68.5 | 34.8 | 6.7 | 51.4 | 65.4 |
| phase3-rd1 | 91.6 | 70.8 | 75.6 | 71.2 | 37.4 | 5.6 | 46.6 | 64.4 |
| phase3-rd2 | 92.0 | 70.2 | 79.9 | 69.3 | 37.9 | 5.6 | 48.1 | 64.9 |
| phase4-rd1 | 91.4 | 71.6 | 76.8 | 70.4 | 36.9 | 7.8 | 46.2 | 64.5 |
| phase4-rd2 | 91.6 | 70.6 | 75.0 | 70.4 | 36.9 | 7.8 | 46.0 | 64.2 |

## 2. 30B passes per block

| Setup | gsm8k500 | math500 | humaneval | mbpp | gpqa | aime | ifeval541 | all 7 |
|---|---|---|---|---|---|---|---|---|
| vanilla | 10.06 | 8.52 | 8.26 | 11.93 | 16.07 | 10.77 | 20.23 | 12.79 |
| phase1 | 7.12 | 6.38 | 5.91 | 8.75 | 12.00 | 7.87 | 15.57 | 9.72 |
| phase2-coda | 8.04 | 7.17 | 5.58 | 9.25 | 12.61 | 8.98 | 15.52 | 10.32 |
| phase2-qwen3 | 7.62 | 6.82 | 5.43 | 9.43 | 12.39 | 8.31 | 16.48 | 10.18 |
| phase3-rd1 | 5.93 | 5.24 | 4.58 | 7.17 | 9.21 | 6.38 | 11.31 | 7.64 |
| phase3-rd2 | 6.03 | 5.46 | 4.65 | 7.30 | 9.56 | 6.53 | 12.23 | 7.99 |
| phase4-rd1 | 5.77 | 5.23 | 4.67 | 7.12 | 9.49 | 6.20 | 11.31 | 7.68 |
| phase4-rd2 | 5.90 | 5.28 | 4.63 | 7.31 | 9.61 | 6.43 | 11.53 | 7.83 |

## 3. Speed vs vanilla

**Compute** (experts loaded per token, drafter counted; machine independent):

| Setup | gsm8k500 | math500 | humaneval | mbpp | gpqa | aime | ifeval541 | all 7 |
|---|---|---|---|---|---|---|---|---|
| phase1 | 1.15× | 1.07× | 1.13× | 1.18× | 1.15× | 1.11× | 1.18× | 1.12× |
| phase2-coda | 1.08× | 1.00× | 1.19× | 1.17× | 1.12× | 1.02× | 1.20× | 1.09× |
| phase2-qwen3 | 1.18× | 1.11× | 1.29× | 1.19× | 1.19× | 1.16× | 1.16× | 1.15× |
| phase3-rd1 | 1.13× | 1.06× | 1.18× | 1.11× | 1.11× | 1.09× | 1.17× | 1.09× |
| phase3-rd2 | 1.20× | 1.11× | 1.21× | 1.21× | 1.21× | 1.17× | 1.24× | 1.17× |
| phase4-rd1 | 1.15× | 1.06× | 1.13× | 1.11× | 1.11× | 1.14× | 1.21× | 1.11× |
| phase4-rd2 | 1.22× | 1.14× | 1.22× | 1.21× | 1.22× | 1.19× | 1.33× | 1.20× |

**Wall-clock per answer, one machine** (8× RTX PRO 6000 Blackwell; vanilla and Phase 4 decoded there, Phase 3's
recorded work priced at that machine's measured per-forward costs; 95% bootstrap ranges in `results/phase4_drafter/`):

| Setup | gsm8k500 | math500 | humaneval | mbpp | gpqa | aime | ifeval541 | all 7 |
|---|---|---|---|---|---|---|---|---|
| phase3-rd1 | 1.28× | 1.21× | 1.08× | 1.21× | 1.37× | 1.37× | 1.06× | 1.20× |
| phase3-rd2 | 1.30× | 1.26× | 1.32× | 1.29× | 1.43× | 1.42× | 1.13× | 1.26× |
| **phase4-rd1** | 1.31× | 1.28× | 1.26× | 1.22× | 1.39× | 1.43× | 1.10× | **1.25×** |
| **phase4-rd2** | 1.34× | 1.31× | 1.26× | 1.29× | 1.44× | 1.45× | 1.09× | **1.26×** |

Wall-clock gains exceed the compute gains because the drafter's forwards are cheap in time (small dense model) while
the compute column charges them at full parameter cost. Phases 1 and 2 were not timed on a common machine, so only
their compute speed-up is given.

Notes:
- IFEval is the weak spot of every judge setup: re-drafting (Phases 3–4) costs 4–6 points there, and repetition loops
  rise (vanilla 44, Phase 4 rd1 104 of 541).
- Phase 2: neither released drafter changes accuracy significantly (CoDA −0.3 points, p = 0.77; Qwen3-0.6B +0.4,
  p = 0.63, paired McNemar). The analysis (`docs/phase2_artifact/`, data repo `phase2/phase2/analysis/`) found that only
  training the drafter had real headroom, which became Phase 4.

## 4. Phase 5: judge retraining (failed)

The Phase 1 judge was retrained from its 128M checkpoint on 47,799 judge states recorded in SGLang on a fresh corpus
(wave C, 150,866 items), for one and then two epochs (to 147.61M and 155.66M tokens). The training-time probe looked
good (precision 99.63%, commit share 28.7%; Phase 1 judge 99.21% / 30.74%), but in real decoding every retrained
checkpoint loops until the length cap:

IFEval-541, first 16 items (fewer where the check was stopped), exact eval flags:

| Judge | Tokens | Items | Accuracy | Outputs cut at the length cap (looping) |
|---|---|---|---|---|
| Phase 1 (128M) | 128M | 16 | 56.2% | 1 |
| retrained | 136M | 13 | 7.7% | 12 |
| retrained | 147.61M | 16 | 6.2% | 15 |
| retrained | 155.66M | 15 | 6.7% | 14 |

The stopped full IFEval run of the 155.66M judge: 181 items, **1.7%** accuracy (phase4-rd1, i.e. the Phase 1 judge with the same drafter and flags, on the same
181 items: 48.1%), 173 of 181 outputs cut at the length cap.

Most likely cause (not proven): the states were recorded in SGLang while the judge is trained and evaluated in the HF
harness. Before any training, the HF trainer reproduced only 89.8% of the SGLang-recorded judge's decisions for the
same adapter, and that agreement fell to 72.6% by 155.66M. The training data, targets and loss rows were verified
correct; the training-time probe and the step-0 check both measure on the recorded states, so neither could see the
live-decode failure. Only IFEval was evaluated before the run was stopped. A valid retrain needs states recorded in
the training and evaluation engine (or on-policy re-recording) and a live-decode check on every checkpoint.
Details: data repo `phase5/logs/phase5/FAILURE_ANALYSIS.md`, `results/phase5/`.

The Phase 1 judge re-decoded on the Phase 5 machine reproduced its earlier records (GSM8K 495/500, HumanEval
164/164, MBPP 256/257 outputs identical; MATH 395/500, bf16 near-tie flips), so the harness and machine were sound.

## 5. Where the records are

| Setup | Data repo path |
|---|---|
| vanilla | `phase4/logs/phase4/decode/vanilla_here/<set>/` |
| phase1 | `phase2/assets/ref_S0_SDAR1.7B_judge128M_nopool/<set>/` (Phase 1 records) |
| phase2-coda / -qwen3 | `phase2/phase2/decode/F2_nopool/<set>/`, `phase2/phase2/decode/S1_nopool/<set>/` |
| phase3-rd1 / -rd2 | `phase3/logs/phase3_8gpu/decode/rd1/<set>/`, `.../rd2/<set>/` |
| phase4-rd1 / -rd2 | `phase4/logs/phase4/decode/rd1/<set>/`, `.../rd2/<set>/` |
| phase5 | `phase5/logs/phase5/eval/`, `phase5/logs/phase5/eval_diag/` |

`python scripts/summarize.py <folder>` reproduces the per-benchmark rows from any of these folders.
