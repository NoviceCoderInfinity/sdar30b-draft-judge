# AIME with a longer generation limit (4,096 tokens)

> Run 2026-09-29, 18:47–19:26 IST on 8× RTX PRO 6000 Blackwell. Judge = `main/final` (28,498,676 tokens, adapter
> sha256 `957c48353ddf5593`). Records: `logs/c1b/eval/aime4k/{aime_stock,aime_judge_des48}/` (HF dataset
> `Arushhh/c1b-judge-corpus-waveA`, same paths). Companion: [`metrics_28.5M.md`](metrics_28.5M.md).

## Why this run

The standard protocol caps AIME answers at **1,024 tokens** (`research_protocol.GEN_LEN["aime"]`). At that cap
36% of stock answers were cut off before finishing, so the 1,024-token AIME score says as much about the budget as
about the model. This run raises the cap to **4,096 tokens** with everything else unchanged.

## Setup

* 90 problems: `AI-MO/aimo-validation-aime` (AIME 2022–2024), one greedy run, block 32, commit threshold 0.95.
* Stock: `run_da_decode.py --stock`. Judge: `--drafter d32 --judge full --draft s8 --pool 48 --pool-rule des --pool-flush`
  (the same settings as every checkpointB cell).
* The limit is raised by `logs/c1b/dl/run_aime_long.py`, a wrapper that sets `GEN_LEN["aime"] = $AIME_GEN_LEN`
  (4096) in memory and calls the unmodified `run_da_decode.main()`. No shared code was edited.
* 8 shards per arm through `code/draftalign/gpu_queue.sh`. Stock shards took 13–16 min, judge shards 9–16 min.

## Results

| cell | n | acc % | passes/blk | exp/token | exp/token + drafter | mean answer length (tokens) | hit the limit % | loop % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock, 1,024 limit | 90 | 6.7 | 10.77 | 998 | 998 | 830 | 35.6 | 15.6 |
| judge 28.5M, 1,024 limit | 90 | 5.6 | 8.33 | 670 | 776 | 786 | 31.1 | 13.3 |
| **stock, 4,096 limit** | 90 | **7.8** | 8.50 | 781 | 781 | 1,202 | 10.0 | 16.7 |
| **judge 28.5M, 4,096 limit** | 90 | **6.7** | 6.25 | 513 | 618 | 1,180 | 10.0 | 15.6 |

* **Judge vs stock at 4,096 tokens: −1.1 points [−5.6, +2.2]** (paired bootstrap, 90 problems, 2,000 resamples,
  seed 0): within noise. Stock solves 7 problems, the judge 6; 5 are solved by both, 2 only by stock, 1 only by
  the judge.
* **Speed-up at 4,096 tokens:** 1.36× fewer passes per block, 1.52× less MoE compute (judge only), **1.26× net of
  the drafter**, in line with the other benchmarks.
* **The longer limit barely helps.** Stock goes from 6/90 to 7/90 (one extra problem) and the judge from 5/90 to
  6/90. Only 10% of answers now reach the limit and the average answer is ~1,200 tokens: most answers end on their
  own, long before 4,096. The low score is the model's, not the budget's.
* One stock answer that was wrong at 1,024 tokens is right at 4,096.

## Details

* **What "AIME" is here:** `AI-MO/aimo-validation-aime`, **90 problems = AIME 2022, 2023 and 2024, 30 per year**
  (AIME I and II, 15 problems each; source URLs point to the Art of Problem Solving wiki, e.g. `2022_AIME_I_Problems/Problem_1`).
  Each answer is an integer 0–999, scored by exact match. Mean prompt length: 131 tokens.
* **Tokens generated at the 4,096 limit:** stock 106,978 (mean 1,189 per problem), judge 105,089 (mean 1,168).
  81 of 90 answers ended with an end-of-sequence token in both arms; 9 hit the limit.
* **Wall-clock (same machine, same run, 1 GPU per shard):** stock 67.0 ms per generated token, judge 50.6 ms per
  token → **1.32× faster in wall-clock**, including the drafter. Total GPU time: stock 119.5 GPU-minutes, judge 88.7.
* **Speed-up summary at 4,096 tokens:**

  | measure | stock | judge 28.5M | speed-up |
  |---|---:|---:|---:|
  | MoE passes per 32-token block | 8.50 | 6.25 | 1.36× |
  | expert loads per token (MoE only) | 781 | 513 | 1.52× |
  | expert loads per token incl. drafter | 781 | 618 | 1.26× |
  | wall-clock ms per token | 67.0 | 50.6 | 1.32× |

## How this compares to published numbers

* **SDAR-30B-A3B-Chat has no published AIME score.** Its model card and paper (arXiv 2510.06303) report MMLU,
  GSM8K, MATH-500, MathBench, HumanEval, MBPP and IFEval only.
* The closest published figures (DreamReasoner-8B, arXiv 2606.19257, Table 3; 24K-token limit, threshold decoding,
  average of 8 runs):
  * **SDAR-8B-Chat, block 4: 13.8% (AIME 2024), 10.4% (AIME 2025).** On the SDAR model card the 8B and 30B-A3B
    Chat models score almost the same on maths (MATH-500 78.6 vs 77.8, GSM8K 91.3 vs 91.4).
  * SDAR-30B-A3B-**Sci** (a separate long-reasoning model): 73.4 / 59.2 at block 4, and **47.1 / 23.3 at block 32**,
    so larger blocks cost a lot on AIME.
* Our setting differs on every axis: block 32 (not 4), a 4,096-token limit (not 24K), one greedy run (not 8
  averaged), and AIME 2022–2024 (not 2024 / 2025 separately). A 7.8% stock score at block 32 is consistent with
  ~10–14% at block 4 for a Chat model of this size, but that is an inference, not a measurement.

## To match the published protocol next time

Block 4 with 4 denoising steps, a 16K–24K limit, 8+ sampled runs averaged, and AIME 2024 and 2025 reported
separately. At block 4 the draft-and-judge setup changes too (blocks of 4 tokens), so it needs its own run.
