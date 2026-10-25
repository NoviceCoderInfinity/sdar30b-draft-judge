# Models, weights and results — round 3 (2026-09-15)

Every model compared in round 3, where its weights and eval records live on the **public Hugging Face dataset of
Anupam Rawat** — [`Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts`](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts) —
and how far each one has been evaluated. The live results table (refreshed every 10 min while evals run) is
[`docs/results/r3_B/README.md`](results/r3_B/README.md); the full per-arm tables are [`docs/results/r3_B/eval_summary.md`](results/r3_B/eval_summary.md).

HF base for the links below: `https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/`

## 1. Models / settings

| # | model / setting | what it is | weights on HF (Anupam, public) | eval records on HF |
|---|---|---|---|---|
| 1 | **stock** | `JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc3…`, block 32, τ 0.95 | base model (JetLM) | `Nemotron-split-moe-train/reference_evals/da_decode/*_stock`, `Nemotron-split-moe-train/r3_A/evals/stock/` |
| 2 | **stock + DES48** | stock with a 48-expert DES pool per forward (`--pool 48 --pool-rule des --pool-flush`); no training | — | `reference_evals/da_decode/*_stock_p48desf`, `r3_A/evals/stock/*/stock_des48` |
| 3 | **ours = r3_B + DES48** | judge LoRA, calibrated recipe (β 0.5 + calibration BCE + override weights 2 / code 3), trained from r3_A step 700 for 710 steps → epoch 2; decoded with the 1.7B drafter (`--judge full --draft s8`) + DES48 | [`Nemotron-split-moe-train/r3_B/checkpoints/step000710`](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_B/checkpoints/step000710) (sha `ad8a6485…`; every 100 steps in the same folder) | `Nemotron-split-moe-train/r3_B/evals/step000710/` |
| 4 | **prev ours (round-2, GSM8K-heavy) + DES48** | `t2b_union_flag` step 3057 — the round-2 published judge | [`Nemotron-split-moe-train/round2_t2b_union_flag/step003057`](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/round2_t2b_union_flag/step003057) (sha `734af656…`) | `reference_evals/da_decode/*_dv-s8_full_c95_p48desf_ad734af656` |
| 5 | **r3_A + DES48** (no calibration) | judge LoRA, CE + certainty (β 2), same data as r3_B; planned 3 epochs, stopped at step 2000 (2.82 ep) | [`Nemotron-split-moe-train/r3_A/checkpoints`](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints): epoch 1 `step000700` (`8f3b73a4…`), epoch 2 `step001400` (`2e99a87b…`), last `step002000` (`d4e694e5…`) | `Nemotron-split-moe-train/r3_A/evals/step000700`, `step001400`, `step002000` |
| — | secondary: stock + DES32, blind accept, r3_B judge without DES | see `docs/results/r3_B/eval_summary.md` | — | same folders |

Training data, processed states and the code as run: `Nemotron-data-subset-training-diffusion-moe/`. Train logs:
`Nemotron-split-moe-train/{r3_A,r3_B}/train_logs/`. r3_B calibration evals (ECE/NLL vs mask ratio, reliability,
joint calibration) + results + report: `Nemotron-split-moe-train/r3_B/analysis/`. W&B project:
https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign (runs `r3_A`, `r3_B`).

Load a judge checkpoint exactly as the evals do:

```bash
hf download Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
  --include "Nemotron-split-moe-train/r3_B/checkpoints/step000710/*" --local-dir .
python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items humaneval \
  --adapter Nemotron-split-moe-train/r3_B/checkpoints/step000710 --judge full --draft s8 --pool 48 --pool-rule des --pool-flush
```

## 2. Eval coverage (full sets: HumanEval 164, MBPP 257, GSM8K-500, MATH-500, GPQA-Diamond 198, AIME 90)

Final, 19:24 IST, 15 Sep 2026. Every planned cell finished, including the plain-judge ablation (the eval queue was allowed to run past the nominal 18:00 shutdown). r3_A epoch rows on GSM8K/MATH were run on the first 100 / 50 items only, by design.

| model | HumanEval | MBPP | GSM8K-500 | MATH-500 | GPQA | AIME |
|---|---|---|---|---|---|---|
| stock | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| stock + DES48 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| stock + DES32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| blind accept | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **r3_B + DES48 (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **r3_B judge, no pool (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| round-2 judge + DES48 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r3_A epoch 1 (step 700) + DES48 | ✓ | ✓ | 100/500 (subset only) | 50/500 (subset only) | — | — |
| r3_A epoch 2 (step 1400) + DES48 | ✓ | ✓ | 100/500 (subset only) | 50/500 (subset only) | — | — |
| r3_A last (step 2000) + DES48 | ✓ | ✓ | — | — | — | — |

## 3. Results

**Per-dataset tables (stock, stock + DES48, stock + DES32, judge, judge + DES48 with every metric): [`docs/results/r3_B/FINAL_TABLES.md`](results/r3_B/FINAL_TABLES.md).**

**Artifacts:** results-only page (rows = models, ICLR abstract) https://claude.ai/artifact/BD1EFygV8vfBcnLKs7Zd4V · full report https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg
**GitHub:** branch `round3-2026-09-14` — results `docs/results/r3_B/` (README, eval_summary.md/json, ABSTRACT_ICLR2027.md, r3_results.html, calib/), model index `docs/MODELS_AND_RESULTS_2026-09-15.md`, this status file.
**Hugging Face (public, Anupam-Rawat-IITB):** https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train — `r3_B/checkpoints` (ours, final `step000710`), `r3_A/checkpoints`, `round2_t2b_union_flag/step003057`, `reference_evals/`, `r3_B/evals/`, `r3_B/analysis/`, `r3_A/evals/`.

Headline, accuracy (Δ vs stock, speedup = stock expert loads per token ÷ arm's):

| set | stock | stock + DES48 | r3_B + DES48 (ours) | r3_B judge, no pool (ours) | round-2 judge + DES48 | r3_A epoch 2 + DES48 |
|---|---:|---|---|---|---|---|
| HumanEval | 78.0 | 78.7 (+0.6, 1.19×) | 75.0 (-3.0, 1.50×) | 75.0 (-3.0, 1.22×) | 62.2 (-15.9, 1.84×) | 69.5 (-8.5, 1.49×) |
| MBPP | 65.4 | 63.8 (-1.6, 1.16×) | 61.1 (-4.3, 1.38×) | 62.3 (-3.1, 1.15×) | 53.7 (-11.7, 1.25×) | 61.9 (-3.5, 1.49×) |
| GSM8K-500 | 90.2 | 91.2 (+1.0, 1.26×) | 90.8 (+0.6, 1.54×) | 90.6 (+0.4, 1.20×) | 87.8 (-2.4, 1.74×) | 90.0 (-2.0, 1.74×) n=100 |
| MATH-500 | 72.4 | 70.6 (-1.8, 1.22×) | 69.2 (-3.2, 1.38×) | 67.0 (-5.4, 1.11×) | 69.2 (-3.2, 1.55×) | 68.0 (-8.0, 1.55×) n=50 |
| GPQA | 37.4 | 35.9 (-1.5, 1.10×) | 28.8 (-8.6, 1.24×) | 37.9 (+0.5, 1.05×) | 27.8 (-9.6, 1.23×) | — |
| AIME | 6.7 | 6.7 (+0.0, 1.19×) | 7.8 (+1.1, 1.38×) | 7.8 (+1.1, 1.11×) | 8.9 (+2.2, 1.48×) | — |

Reading: the calibrated judge supplies the accuracy recovery (HumanEval 75.0 vs 69.5 for the uncalibrated epoch-2 judge at the same ~1.5×), the DES48 pool supplies most of the speedup (without it the same judge gives only 1.05–1.22×), and on GPQA the plain judge matches stock (37.9 vs 37.4) while the pooled variants lose 8–10 points, so the GPQA gap comes from the 48-expert pool on that set, not from the judge.

## 4. Resuming the unfinished evals on another machine

Everything needed is public. Records are per question, so a shard cut by the shutdown resumes from where it stopped
(`run_da_decode.py` skips questions already present in the shard's `.jsonl`).

1. Weights: `hf download Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset --include "Nemotron-split-moe-train/r3_B/checkpoints/step000710/*" --local-dir .`
2. Partial records: the same repo, `Nemotron-split-moe-train/r3_B/evals/step000710/<set>/judge/` (plain judge) and `judge_des48/`; stock + DES baselines under `Nemotron-split-moe-train/r3_A/evals/stock/` and `Nemotron-split-moe-train/reference_evals/`. Copy them to `logs/draftalign/probes_r3/r3_B/step000710/<set>/<arm>/` and `logs/draftalign/probes_r3/r3_A/stock/<set>/<arm>/`.
3. Decode one shard (plain judge, sets: humaneval, mbpp, gsm8k500, aime, math500, gpqa; shard counts used: 2, 3, 6, 5, 12, 9):
   `CUDA_VISIBLE_DEVICES=0 python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items <set> --adapter <step000710 dir> --judge full --draft s8 --shard <k> --nshards <n> --no-trace [--score] --out logs/draftalign/probes_r3/r3_B/step000710/<set>/judge` — `--score` for every set except humaneval/mbpp, which are scored afterwards with `bash code/draftalign/run_score_r3.sh <dir> once <dir>/scored` (sandboxed).
   The exact queue lines that were pending are in `docs/results/r3_B/queue_pending_at_push.txt`.
4. Summaries: `python code/draftalign/eval_summary_r3B.py --step step000710`, then `python docs/report/build_results_page.py` and `python docs/report/build_r3_report.py --no-eval`.
