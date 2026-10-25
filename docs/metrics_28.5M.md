# Judge metrics at 28.5M training tokens (Phase 1, wave A)

> Written 2026-09-29 (IST). Checkpoint `main/final`: **28,498,676 corpus tokens, step 7418**, adapter sha256 `957c48353ddf5593…`.
> Base `JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc` (block 32), drafter `JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05`.
> Weights: https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32 · Corpus, logs and eval records: https://huggingface.co/datasets/Arushhh/c1b-judge-corpus-waveA · This file name uses the trained token count (28.50M; the '28.7M' in the request was a typo).

Companion documents: [`AIME_unconstrained_token.md`](AIME_unconstrained_token.md) · [`DATA_GENERATION_CONTINUATION.md`](DATA_GENERATION_CONTINUATION.md) · [`PHASE1_HF_ARTIFACTS.md`](PHASE1_HF_ARTIFACTS.md)

Published pages (claude.ai artifacts; HTML copies are in the HF dataset under `logs/c1b/eval/checkpointB/`): results + trade-off chart https://claude.ai/artifact/6wBHs5rbZ5pQU96L55BuHg · training-data audit https://claude.ai/artifact/5J4RUMNTPNGmccse9gp6Cv

## 1. TL;DR

* The judge decodes with **1.25–1.34× fewer MoE passes per block** than stock on all 7 sets; counting the drafter's cost the net compute saving is **1.20–1.37×** (judge-only MoE compute 1.42–1.60×).
* Accuracy is statistically indistinguishable from stock on **6 of 7 sets** (95% paired bootstrap CI includes 0): GSM8K +0.6, MATH −0.8, MBPP +2.7, HumanEval −1.8, GPQA −3.0, AIME −1.1.
* **IFEval is the one significant failure: 37.9% vs 52.7% (−14.8 [−18.9, −10.7])**, with more looping (18.1% vs 12.6%) and more non-terminating answers (14.4% vs 8.5%).
* vs the round-3 judge: MBPP +16.7, HumanEval +11.6, MATH +3.6, GSM8K +2.0, GPQA +1.5 points.
* Training saw 28.50M of the 74.36M converted training tokens (38%); science and IF are exhausted (~100%), maths 36%, code 30%, chat 40%.

## 2. Evaluation results (checkpoint 28.5M)

Protocol: `run_da_decode.py --judge full --draft s8 --tau-commit .95 --pool 48 --pool-rule des --pool-flush` (DES48 expert pooling), greedy, block 32, generation limit 1,024 tokens (512 for MBPP/HumanEval). Stock/blind/round-3 on the six standard sets are `RESULTS.md` §1–3 (same model, harness and settings; earlier machine). IFEval stock and blind were run on this box. Δ = paired bootstrap over shared prompts, 2,000 resamples, seed 0.

| dataset | n | stock | blind | round-3 judge | judge 11.79M | **judge 28.5M** | Δ vs stock [95% CI] | stock passes/blk | judge passes/blk | speed-up passes/blk | MoE speed-up (judge only) | speed-up (judge + drafter) | judge exp/token | loop % | no-stop % |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K-500 | 500 | 90.2 | 90.4 | 88.8 | 91.0 | **90.8** | +0.6 [-1.6, +2.6] | 10.06 | 7.54 | 1.33× | 1.60× | 1.37× | 728 | 0.2 | 0.2 |
| MATH-500 | 500 | 72.4 | 70.8 | 68.0 | 72.8 (n=438) | **71.6** | -0.8 [-4.2, +2.6] | 8.52 | 6.50 | 1.31× | 1.54× | 1.29× | 555 | 7.8 | 9.8 |
| MBPP | 257 | 65.4 | 60.3 | 51.4 | 68.5 | **68.1** | +2.7 [-3.1, +8.2] | 11.93 | 9.54 | 1.25× | 1.47× | 1.29× | 1164 | 0.0 | 0.0 |
| HumanEval | 164 | 78.0 | 75.6 | 64.6 | 73.8 | **76.2** | -1.8 [-7.3, +3.7] | 8.26 | 6.58 | 1.25× | 1.43× | 1.20× | 688 | 0.6 | 0.6 |
| GPQA-Diamond | 198 | 37.4 | 35.4 | 32.8 | 31.6 (n=174) | **34.3** | -3.1 [-10.1, +4.0] | 16.07 | 12.61 | 1.27× | 1.42× | 1.28× | 972 | 13.1 | 11.6 |
| AIME | 90 | 6.7 | 5.6 | 5.6 | 6.7 | **5.6** | -1.1 [-3.3, +0.0] | 10.77 | 8.33 | 1.29× | 1.49× | 1.28× | 670 | 13.3 | 26.7 |
| IFEval-541 | 541 | 52.7 | 50.1 | — | 38.3 (n=407) | **37.9** | -14.8 [-18.9, -10.7] | 20.23 | 15.12 | 1.34× | 1.44× | 1.31× | 1095 | 18.1 | 14.4 |

Speed-up definitions (RESULTS.md): passes/blk = Σ(n_forward − n_prefill − n_flush)/Σ n_blocks; exp/token = distinct (layer, expert) loads per generated token (includes the DES48 saving); judge + drafter adds drafter forwards × 364.64 expert-equivalents (1.7B drafter ÷ one 30B expert).

### 2.1 All arms by speed-up (judge + drafter compute)

| dataset | blind Δacc / speed-up | round-3 Δacc / speed-up | judge 11.79M Δacc / speed-up | judge 28.5M Δacc / speed-up |
|---|---|---|---|---|
| GSM8K-500 | +0.2 / 1.02× | -1.4 / 1.20× | +0.8 / 1.35× | +0.6 / 1.37× |
| MATH-500 | -1.6 / 0.98× | -4.4 / 1.08× | +0.4 / 1.22× | -0.8 / 1.29× |
| MBPP | -5.1 / 1.01× | -14.0 / 0.98× | +3.1 / 1.28× | +2.7 / 1.29× |
| HumanEval | -2.4 / 1.07× | -13.4 / 1.34× | -4.2 / 1.27× | -1.8 / 1.20× |
| GPQA-Diamond | -2.0 / 0.95× | -4.6 / 0.99× | -5.8 / 1.12× | -3.1 / 1.28× |
| AIME | -1.1 / 0.97× | -1.1 / 1.10× | -0.0 / 1.22× | -1.1 / 1.28× |
| IFEval-541 | -2.6 / 1.00× | — | -14.4 / 1.21× | -14.8 / 1.31× |

Blind accept is nearly free in accuracy but, net of its drafter, barely faster than stock (0.95–1.07×). Only the trained judge buys real net speed.

### 2.2 Caveats

* Judge cells use DES48 expert pooling; stock does not. DES48 alone costs a little accuracy (MBPP stock+DES48 −1.6, RESULTS.md §4.1) and was never measured on IFEval, so part of every Δ may be pooling, not the judge.
* ms/token is wall-clock and machine-dependent: stock/blind/round-3 figures come from an earlier machine; compare speed by passes/blk and exp/token.
* The 11.79M checkpoint's evals were cut by a deadline on MATH (438/500), GPQA (174/198) and IFEval (407/541).
* AIME at 1,024 tokens truncates ~36% of stock answers. Rerun at 4,096 tokens ([`AIME_unconstrained_token.md`](AIME_unconstrained_token.md)): stock 7.8%, judge 6.7%, Δ −1.1 [−5.6, +2.2]; 1.36× fewer passes/blk, 1.26× net compute, 1.32× wall-clock.
* Not evaluated at 28.5M: **ChartQA-300 and FinQA-500** (deferred; judge + blind missing, stock FinQA shards 6–7 missing).

## 3. Training run

* Run `logs/c1b/train/main`: LoRA r32/α64 on q,k,v,o in all 48 layers + draft flag; experts and routers frozen; AdamW (0.9, 0.95), grad clip 1.0, LR 8e-5 warmup-stable-decay by corpus tokens (warmup 2M, horizon 1e9, decay in the last 15%, so every Phase-1 checkpoint is a stable-phase checkpoint); states/step ramp 32 → 128 at 8M → 512 at 32M (never reached); 8× RTX PRO 6000 Blackwell, torchrun DDP, manual all-reduce.
* Loss per state: **L = CE_w + β·Cert + λ·Calib** — CE weighted ×2 on override rows (code ×3, chat ×1); Cert = entropy (τ=0.5, [MASK] removed) averaged over correctly-predicted rows, β = 0.5 (chat 0); Calib = BCE(top-1 prob, correct), λ = 1 (`code/draftalign/losses.py:masked_ce_and_certainty`). Every component is logged per domain in `train.rank0.jsonl`.
* Timeline (IST, 2026-09-29): 02:19 start → 07:14 stop at 11.79M (checkpoint A, evaluated) → 11:23 resume → 11:51 stop at 14.47M → 12:09 resume with **domain weights maths 25 / code 25 / science 15 / IF 15 / chat 20** (science, IF repeatable up to 4 passes) → 13:45 stop/auto-relaunch at 23.19M → 14:45 stop at **28.50M**. (11:53–12:08 was lost to an O(N²) planner bug, fixed in `0e742f5`.)
* Throughput: ~460 tok/s at 32 states/step, ~1,500 tok/s at 128 states/step (≈5.2–5.7M tokens/h). GPUs ran at ~55–75% utilisation: the trainer is CPU/launch-bound; the grouped-GEMM MoE path (`fast_moe.py`) is exact (loss rel diff 0.0, grad cosine 1.0) but measured 0.85× speed, so it was not used.

### 3.1 Held-out probe (200 items per domain)

| tokens | all acc | maths | code | science | IF | chat | commit precision | override precision | chat override precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.84M | 0.458 | 0.561 | 0.693 | 0.453 | 0.268 | 0.330 | 0.990 | 0.899 | 0.829 |
| 8.00M | 0.472 | 0.581 | 0.710 | 0.470 | 0.277 | 0.336 | 0.989 | 0.882 | 0.896 |
| 11.79M | 0.479 | 0.593 | 0.727 | 0.477 | 0.275 | 0.339 | 0.991 | 0.911 | 0.893 |
| 14.47M | 0.484 | 0.594 | 0.732 | 0.480 | 0.283 | 0.347 | 0.988 | 0.890 | 0.774 |
| 23.19M | 0.481 | 0.594 | 0.721 | 0.481 | 0.282 | 0.340 | 0.991 | 0.906 | 0.861 |
| 28.50M | 0.484 | 0.591 | 0.717 | 0.493 | 0.286 | 0.350 | 0.987 | 0.875 | 0.874 |

Probe accuracy is flat from ~12M tokens (0.479 → 0.484); override precision peaked at 11.79M (0.911).

## 4. What the model has seen (exact replay of the trainer's plan)

Computed by `logs/c1b/dl/replay_seen.py` (rebuilds `plan_epoch` + `domain_weighted_units` on CPU; reproduces the trainer's 28,498,676 tokens exactly) and `seen_report.py`. Data: `logs/c1b/train/main/seen_report.{md,json}`, `seen_items.json` (per item).

**Totals:** prompts on disk 2,239,372 (wave A 287,222); generated 287,222 answers (153.02M tokens incl. prompts); rejected 121,233 (77.91M); converted to blocks 165,438 items (74.76M); train split 164,438 items (74.36M); **seen 28.50M = 38% of converted train tokens**, items touched 65,867 (40%). Trainer count 28.50M.

#### MATHS  — seen 7.70M of 21.57M converted train tokens (36%); items touched 15,552/43,668 (36%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 grade-school (GSM)** | **all** | **74,015** | **9,465** | **1,496** | **7,969 (2.14M)** | **0.77M** | **36%** | **36%** |
| | OpenMathInstruct-2 | 71,628 | 9,169 | 1,487 | 7,682 (2.07M) | 0.74M | 36% | 36% |
| | GSM8K train | 2,387 | 296 | 9 | 287 (0.07M) | 0.02M | 31% | 32% |
| **2 MATH-easy** | **all** | **262,853** | **33,730** | **18,804** | **14,690 (8.64M)** | **3.07M** | **36%** | **36%** |
| | OpenMathInstruct-2 | 262,000 | 33,620 | 18,798 | 14,592 (8.60M) | 3.05M | 36% | 36% |
| | MATH (Hendrycks) train Level 2 | 366 | 58 | 3 | 52 (0.02M) | 5.7K | 26% | 27% |
| | MATH (Hendrycks) train Level 3 | 343 | 37 | 2 | 32 (0.01M) | 6.2K | 43% | 42% |
| | MATH (Hendrycks) train Level 1 | 144 | 15 | 1 | 14 (4.4K) | 1.1K | 25% | 29% |
| **3 MATH-hard** | **all** | **281,476** | **36,122** | **18,286** | **17,544 (8.53M)** | **3.02M** | **36%** | **36%** |
| | NuminaMath-1.5 [cn_k12] | 103,515 | 13,332 | 7,638 | 5,692 (2.69M) | 0.96M | 36% | 36% |
| | DeepMath-103K | 74,754 | 9,558 | 5,328 | 4,014 (2.77M) | 0.96M | 35% | 35% |
| | NuminaMath-1.5 [orca_math] | 51,059 | 6,620 | 1,902 | 4,718 (1.53M) | 0.55M | 36% | 36% |
| | NuminaMath-1.5 [synthetic_math] | 47,761 | 6,080 | 2,998 | 3,010 (1.49M) | 0.53M | 36% | 35% |
| | NuminaMath-1.5 [metamath] | 3,665 | 453 | 383 | 70 (0.03M) | 8.9K | 29% | 33% |
| | MATH (Hendrycks) train Level 5 | 408 | 47 | 27 | 19 (0.01M) | 4.4K | 39% | 32% |
| | MATH (Hendrycks) train Level 4 | 314 | 32 | 10 | 21 (9.9K) | 4.1K | 41% | 38% |
| **4 olympiad** | **all** | **150,465** | **19,455** | **15,770** | **3,665 (2.36M)** | **0.84M** | **36%** | **36%** |
| | OpenMathReasoning | 64,000 | 8,331 | 7,398 | 923 (0.64M) | 0.22M | 35% | 34% |
| | NuminaMath-1.5 [olympiads] | 49,750 | 6,414 | 5,159 | 1,251 (0.75M) | 0.27M | 36% | 37% |
| | DeepMath-103K | 16,465 | 2,102 | 1,116 | 986 (0.64M) | 0.22M | 35% | 35% |
| | NuminaMath-1.5 [cn_contest] | 8,937 | 1,157 | 972 | 185 (0.12M) | 0.04M | 38% | 37% |
| | NuminaMath-1.5 [aops_forum] | 7,449 | 962 | 788 | 173 (0.12M) | 0.05M | 40% | 40% |
| | NuminaMath-1.5 [amc_aime] | 2,278 | 292 | 155 | 132 (0.09M) | 0.04M | 41% | 39% |
| | NuminaMath-1.5 [number_theory] | 588 | 77 | 72 | 5 (3.0K) | 1.5K | 49% | 60% |
| | NuminaMath-1.5 [inequalities] | 562 | 73 | 66 | 7 (5.8K) | 3.8K | 65% | 57% |
| | NuminaMath-1.5 [olympiads_ref] | 436 | 47 | 44 | 3 (2.1K) | 1.4K | 67% | 67% |

#### CODE  — seen 10.08M of 33.86M converted train tokens (30%); items touched 19,765/66,357 (30%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 easier: function-level (Python)** | **all** | **696,382** | **89,234** | **23,011** | **66,223 (33.80M)** | **10.04M** | **30%** | **30%** |
| | OpenCodeInstruct [self-instruct, generic] | 292,896 | 37,623 | 7,547 | 30,076 (15.70M) | 4.68M | 30% | 30% |
| | OpenCodeInstruct [evol-instruct, generic] | 225,647 | 28,789 | 9,670 | 19,119 (10.00M) | 2.97M | 30% | 30% |
| | OpenCodeInstruct [self-instruct, algorithmic] | 151,596 | 19,450 | 4,360 | 15,090 (7.19M) | 2.13M | 30% | 30% |
| | OpenCodeInstruct [evol-instruct, algorithmic] | 25,945 | 3,342 | 1,418 | 1,924 (0.91M) | 0.26M | 28% | 28% |
| | MBPP train/val [train] | 233 | 24 | 12 | 12 (2.5K) | 1.4K | 57% | 42% |
| | MBPP train/val [validation] | 57 | 6 | 4 | 2 (0.5K) | 0 | 0% | 0% |
| **2 competitive (CF-style, stdin/stdout)** | **all** | **13,616** | **1,752** | **1,418** | **334 (0.16M)** | **0.05M** | **29%** | **29%** |
| | CodeContests [CODEFORCES] | 4,125 | 565 | 527 | 38 (0.02M) | 7.4K | 38% | 37% |
| | TACO-verified [EASY] | 1,929 | 247 | 143 | 104 (0.05M) | 0.01M | 23% | 24% |
| | CodeContests [HACKEREARTH] | 1,162 | 140 | 99 | 41 (0.02M) | 5.0K | 29% | 29% |
| | CodeContests [AIZU] | 1,088 | 147 | 130 | 17 (0.01M) | 3.6K | 35% | 29% |
| | CodeContests [ATCODER] | 971 | 130 | 100 | 30 (0.01M) | 2.2K | 21% | 20% |
| | TACO-verified [MEDIUM_HARD] | 771 | 102 | 89 | 13 (7.3K) | 4.0K | 55% | 54% |
| | CodeContests [CODECHEF] | 630 | 73 | 42 | 31 (0.01M) | 4.7K | 35% | 35% |
| | TACO-verified [UNKNOWN_DIFFICULTY] | 606 | 68 | 48 | 20 (8.9K) | 3.5K | 40% | 35% |
| | TACO-verified [HARD] | 602 | 72 | 66 | 6 (3.3K) | 0.9K | 28% | 17% |
| | TACO-verified [MEDIUM] | 593 | 87 | 68 | 19 (0.01M) | 1.3K | 10% | 11% |
| | TACO-verified [VERY_HARD] | 291 | 31 | 30 | 1 (0.4K) | 0 | 0% | 0% |
| | open-r1/codeforces [rating 800] | 101 | 13 | 9 | 4 (1.0K) | 0.2K | 20% | 25% |
| | APPS [introductory] | 84 | 12 | 4 | 8 (3.6K) | 1.2K | 32% | 38% |
| | open-r1/codeforces [rating 900] | 34 | 5 | 3 | 2 (0.9K) | 0.5K | 53% | 50% |

#### SCIENCE  — seen 2.55M of 2.29M converted train tokens (112%); items touched 6,730/6,730 (100%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **free** | **all** | **213,112** | **27,344** | **23,061** | **4,283 (1.44M)** | **1.57M** | **112%** | **100%** |
| | OpenScienceReasoning-2 | 108,008 | 13,814 | 10,332 | 3,482 (1.05M) | 1.14M | 112% | 100% |
| | WebInstruct-verified [Physics] | 45,924 | 6,045 | 5,764 | 281 (0.14M) | 0.15M | 114% | 100% |
| | Nemotron-Science-v1 | 33,432 | 4,276 | 4,159 | 117 (0.08M) | 0.09M | 113% | 100% |
| | WebInstruct-verified [Chemistry] | 21,695 | 2,701 | 2,370 | 331 (0.15M) | 0.16M | 108% | 100% |
| | WebInstruct-verified [Biology] | 4,053 | 508 | 436 | 72 (0.01M) | 0.02M | 110% | 100% |
| **mc** | **all** | **54,130** | **6,998** | **4,351** | **2,647 (0.92M)** | **0.99M** | **112%** | **100%** |
| | OpenScienceReasoning-2 | 41,992 | 5,453 | 3,966 | 1,487 (0.47M) | 0.50M | 110% | 100% |
| | Nemotron-Science-v1 | 8,770 | 1,095 | 354 | 741 (0.32M) | 0.34M | 112% | 100% |
| | ai2_arc [ARC-Easy] | 2,250 | 293 | 19 | 274 (0.08M) | 0.10M | 117% | 100% |
| | ai2_arc [ARC-Challenge] | 1,118 | 157 | 12 | 145 (0.05M) | 0.05M | 113% | 100% |

#### IF  — seen 2.64M of 2.73M converted train tokens (97%); items touched 8,362/8,652 (97%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 constraint(s)** | **all** | **83,237** | **10,669** | **4,421** | **6,248 (1.92M)** | **1.81M** | **97%** | **97%** |
| | WildChat-4.8M | 82,377 | 10,537 | 4,353 | 6,184 (1.90M) | 1.79M | 97% | 97% |
| | Nemotron-RL-instruction_following | 369 | 64 | 33 | 31 (0.01M) | 0.01M | 98% | 97% |
| | IF_multi_constraints_upto5 | 317 | 45 | 23 | 22 (7.5K) | 7.4K | 100% | 100% |
| | RLVR-IFeval | 174 | 23 | 12 | 11 (3.0K) | 3.0K | 100% | 100% |
| **2 constraint(s)** | **all** | **51,162** | **6,691** | **4,511** | **2,180 (0.73M)** | **0.69M** | **97%** | **97%** |
| | WildChat-4.8M | 49,044 | 6,410 | 4,301 | 2,109 (0.70M) | 0.66M | 97% | 97% |
| | Nemotron-RL-instruction_following | 1,960 | 259 | 192 | 67 (0.03M) | 0.03M | 98% | 97% |
| | IF_multi_constraints_upto5 | 158 | 22 | 18 | 4 (2.0K) | 2.0K | 100% | 100% |
| **3 constraint(s)** | **all** | **18,906** | **2,434** | **2,010** | **424 (0.15M)** | **0.14M** | **96%** | **96%** |
| | WildChat-4.8M | 18,000 | 2,331 | 1,918 | 413 (0.15M) | 0.14M | 96% | 96% |
| | Nemotron-RL-instruction_following | 824 | 97 | 87 | 10 (2.8K) | 2.3K | 100% | 100% |
| | IF_multi_constraints_upto5 | 82 | 6 | 5 | 1 (0.2K) | 0.2K | 100% | 100% |

#### CHAT  — seen 5.52M of 13.91M converted train tokens (40%); items touched 15,458/39,031 (40%)

| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **chat** | **all** | **340,018** | **43,328** | **4,094** | **39,231 (13.98M)** | **5.52M** | **40%** | **40%** |
| | WildChat-1M | 187,995 | 23,976 | 2,745 | 21,229 (6.26M) | 2.48M | 40% | 40% |
| | ultrachat_200k | 147,000 | 18,669 | 1,299 | 17,370 (7.57M) | 2.99M | 40% | 40% |
| | oasst2 | 5,023 | 683 | 50 | 632 (0.15M) | 0.05M | 33% | 36% |

### 4.1 TACO funnel

`likaixin/TACO-verified` 12,898 problems → 4,792 prompts after filters (function-call problems, unparsable tests and multi-answer statements dropped) → 607 in wave A (4,185 wave B) → 163 kept (263 wrong answer, 139 failed, 34 no EOS, 8 timeout). The 30B solves EASY 42%, MEDIUM 22%, MEDIUM_HARD 13%, HARD 8%, VERY_HARD 3%. Competitive code is 0.5% of the code tokens trained.

## 5. IFEval: what goes wrong

Paired, 541 prompts, judge 28.5M vs stock: judge loses 116 prompts and gains 36; loops both 46, judge-only 52, stock-only 22. At 11.79M on 407 prompts: lost 80 (only 17 of them loops), gained 28; answers that loop have fewer forced commits (0.31/block) than those that don't (0.93/block), i.e. loops come from confident commits of self-consistent (repetitive) drafter context, not forced low-confidence commits. The judge matches the 30B's token on IF probe blocks only 28.6% of the time (maths 59%, code 72%) and agrees with the drafter on ~48% of IFEval slots. Training filtered looping / non-terminating chat and IF answers, so the judge never saw recovery-from-repetition states. IF was trained with β = 0.5 on WildChat-style prose (hypothesis: contributes to over-confidence). DES48 is a confound (not measured on IFEval).


## 6. How to resume training from here

The run can continue exactly where it stopped: same adapter, AdamW state, token count, WSD schedule position and domain-weighted plan position.

**State at the stop** (`main/final/trainer_state.json`): `step 7418`, `tokens 28,498,676`, `epoch 0`, `step_in_epoch 5256`, `epoch_tokens_start 0.0`, `dw_from_step 5256`, `dw_tokens_start 14,466,834`, `dw_step 2162`, plus `dom_tokens` and `milestones_done`. The domain-weighted plan is rebuilt deterministically from the index and resumes at `dw_step` (plan build ≈2 s).

**On a new box** (8× ≥96 GB GPUs):
```bash
git clone -b phase1-datacorpus https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git /workspace/dmge
cd /workspace/dmge && bash code/corpus1b/setup_box.sh          # venvs /venv/train (+ /venv/sgl), pinned models, eval sets
# secrets in ./.env (HF token, GITHUB_TOKEN, BACKUP_TO_HF=1, HF_PUSH_REPO=Arushhh/c1b-judge-corpus-waveA, HF_CKPT_REPO=Arushhh/c1b-judge-lora-sdar30b-b32)
set -a; . ./.env; set +a; export HF_HOME=/workspace/dmge/hf_home
# 1. the checkpoint (adapter + trainer_state.pt + json) -> logs/c1b/train/main/final
/venv/train/bin/python -c "from huggingface_hub import snapshot_download as s; s('Arushhh/c1b-judge-lora-sdar30b-b32', allow_patterns=['main/final/*'], local_dir='logs/c1b/train')"
# 2. the training set: index + item tars -> logs/c1b/trainset/waveA/{index.jsonl,items/}
/venv/train/bin/python -c "from huggingface_hub import snapshot_download as s; s('Arushhh/c1b-judge-corpus-waveA', repo_type='dataset', allow_patterns=['logs/c1b/trainset/waveA/index.jsonl','logs/c1b/trainset/waveA/items_tar/*','logs/c1b/trainset/waveA/build_summary.json'], local_dir='.')"
for t in logs/c1b/trainset/waveA/items_tar/*.tar; do tar -xf "$t" -C logs/c1b/trainset/waveA; done   # tars hold items/<h2>/*.npz
# check: 165,438 index lines; items/<2-hex>/*.npz as referenced by index 'file'
wc -l logs/c1b/trainset/waveA/index.jsonl
```
(Each `items_tar/<h2>.tar` holds `items/<h2>/*.npz`, matching the index's `file` field, verified 2026-09-29.) Copy `logs/c1b/train/main/config.json` from the HF dataset too if you want the exact previous config dump.

**Resume** (same command as `logs/c1b/dl/train_resume3.sh`; set `--stop-at` to a unix time and optionally move it later via `logs/c1b/train/main/STOP_AT`):
```bash
/venv/train/bin/torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
  --index logs/c1b/trainset/waveA/index.jsonl --out logs/c1b/train/main --lr 8e-5 --max-tokens 74759690 --epochs 1 \
  --stop-at <unix-time> --stop-at-file logs/c1b/train/main/STOP_AT --stop-file logs/c1b/train/main/STOP \
  --save-every-tokens 1.8e6 --keep-rolling 1000 --resume \
  --domain-weights maths=25,code=25,science=15,if=15,chat=20 --repeat-domains science,if --max-passes 4 \
  >> logs/c1b/train/main.log 2>&1
```
`--resume` picks the checkpoint with the most tokens under `--out` (`final` here). Controls: `echo <unix> > logs/c1b/train/main/STOP_AT` moves the stop without a restart; `touch logs/c1b/train/main/STOP` stops cleanly (saves `final`, runs the probe). Keep `nohup bash code/corpus1b/backup_hf.sh ckptloop` running to push+verify every checkpoint within 5 min.

**What remains in wave A:** science and IF are exhausted (they repeat, max 4 passes each); maths (~64% unseen), code (~70%) and chat (~60%) have unseen items. Continuing with the same weights keeps the 25/25/15/15/20 token mix by repeating science/IF. To change the mix or the loss (e.g. IF β = 0), start a new `--out` from this checkpoint with `--continue-from logs/c1b/train/main/final` (see the `train_c1b.py` docstring) so the ablation doesn't overwrite this run.

**Wave B / 1B plan:** after generating wave B (see [`DATA_GENERATION_CONTINUATION.md`](DATA_GENERATION_CONTINUATION.md)), continue the same run on the new index: `--index logs/c1b/trainset/waveB/index.jsonl --out logs/c1b/train/main_b --continue-from logs/c1b/train/main/final --max-tokens 1e9` (docstring of `code/corpus1b/train_c1b.py`).

**Throughput knobs worth fixing first:** the trainer is CPU-bound per rank (~99% CPU, GPUs ~55–75%); batching is 128 states/step until 32M tokens.

## 7. All checkpoints (HF model repo, each complete: adapter + optimizer/trainer state, verified file by file)

| checkpoint | step | corpus tokens | adapter sha256[:16] | link |
|---|---:|---:|---|---|
| `roll_00000278` | 278 | 0.500M | `0ba1fd989e859a3f` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00000278 |
| `roll_00000547` | 547 | 1.002M | `bd94e28eec1659fc` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00000547 |
| `roll_00000824` | 824 | 1.502M | `91df4656fae97e13` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00000824 |
| `tok_000001840000` | 1019 | 1.841M | `07c0e00bc50a00e6` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/tok_000001840000 |
| `roll_00001107` | 1107 | 2.004M | `02ce711b401e6c16` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00001107 |
| `roll_00001378` | 1378 | 2.505M | `0d591e678b1d3836` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00001378 |
| `roll_00001657` | 1657 | 3.006M | `9162bc3752691178` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00001657 |
| `roll_00001929` | 1929 | 3.507M | `d63c936977753596` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00001929 |
| `roll_00002200` | 2200 | 4.008M | `52ee1ed09446827a` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00002200 |
| `roll_00002470` | 2470 | 4.510M | `a63d85a48fb8fbec` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00002470 |
| `roll_00002739` | 2739 | 5.011M | `89fd0571314ac882` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00002739 |
| `roll_00003019` | 3019 | 5.512M | `f2e9790185b3f7da` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00003019 |
| `roll_00003291` | 3291 | 6.012M | `fe87d1665861f5f5` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00003291 |
| `roll_00003563` | 3563 | 6.513M | `aafbedef4c5385c7` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00003563 |
| `roll_00003840` | 3840 | 7.014M | `4526b971faa6ffb0` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00003840 |
| `roll_00004111` | 4111 | 7.514M | `4ed2c436a0696aac` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004111 |
| `tok_000008000000` | 4371 | 8.002M | `fdc67006542b3e4e` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/tok_000008000000 |
| `roll_00004373` | 4373 | 8.017M | `df25a40f5012e4a0` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004373 |
| `roll_00004442` | 4442 | 8.522M | `79d3f592c1dfa0f6` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004442 |
| `roll_00004512` | 4512 | 9.022M | `5988a46e513326e2` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004512 |
| `roll_00004579` | 4579 | 9.524M | `1b84751865dfd7f4` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004579 |
| `roll_00004648` | 4648 | 10.025M | `f20c53da6a99aac7` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004648 |
| `roll_00004718` | 4718 | 10.532M | `580af212df49003e` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004718 |
| `roll_00004787` | 4787 | 11.036M | `58a01800d308ea96` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004787 |
| `roll_00004856` | 4856 | 11.540M | `742353e31165c0a7` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00004856 |
| `tok_000011790000` | 4889 | 11.790M | `4f7831fdd854d1f1` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/tok_000011790000 |
| `roll_00005163` | 5163 | 13.797M | `1fa59bfe1397db11` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00005163 |
| `roll_00005566` | 5566 | 16.471M | `bb36e47b1117463d` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00005566 |
| `roll_00005876` | 5876 | 18.476M | `d687aa7a5723fa2d` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00005876 |
| `roll_00006185` | 6185 | 20.477M | `89038bd9eccf4c28` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00006185 |
| `roll_00006492` | 6492 | 22.478M | `fdd812e0efcea8f9` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00006492 |
| `roll_00006875` | 6875 | 24.988M | `ab177a448982f27c` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00006875 |
| `roll_00007152` | 7152 | 26.789M | `a7cad7bf99a54cf1` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/roll_00007152 |
| `final` | 7418 | 28.499M | `957c48353ddf5593` | https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32/tree/main/main/final |

Milestones kept forever: `tok_000001840000`, `tok_000008000000`, `tok_000011790000` (= checkpoint A, evaluated in `logs/c1b/eval/checkpointA`). `final` = 28.50M (evaluated in `logs/c1b/eval/checkpointB`).

## 8. Where everything is

| what | local path (box) | HF |
|---|---|---|
| eval summaries, per-shard records (GPQA records never uploaded) | `logs/c1b/eval/checkpoint{A,B}/` | https://huggingface.co/datasets/Arushhh/c1b-judge-corpus-waveA (same paths) |
| paired CIs, trade-off data, results page | `logs/c1b/eval/checkpointB/{paired_ci_vs_stock.json,tradeoff.json,judge-eval-results.html}` | same |
| AIME 4,096-token run | `logs/c1b/eval/aime4k/` | same |
| training logs, probe, config | `logs/c1b/train/main.log`, `logs/c1b/train/main/{train.rank*.jsonl,probe.jsonl,config.json}` | same |
| seen-data report + replay scripts | `logs/c1b/train/main/seen_*`, `logs/c1b/dl/{replay_seen,seen_report,seen_render}.py` | same |
| corpus (records, drafts, items, index, prompt lists) | `logs/c1b/gen`, `logs/c1b/draft`, `logs/c1b/trainset`, `data/c1b/prompt_lists` | same |
| handover notes | `HANDOVER.md`, `logs/c1b/HANDOFF.md` | same |
