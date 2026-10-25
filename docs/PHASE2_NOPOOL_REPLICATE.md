# Phase 2 drafter swap, no expert pool: everything needed to replicate (2026-10-05)

## What the experiment is

The frozen SDAR-30B-A3B-Chat-b32 (`JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc`) writes text in 32-token blocks. A small
**drafter** guesses each block; a LoRA **judge** on the 30B (Phase 1, trained on a 128M-token corpus, weights hash
`35805974`) accepts, keeps or re-masks each guessed token, pass after pass. **No expert pool** (no DES48) anywhere.
Nothing is trained. Only the drafter changes:

| arm | drafter | revision | cost / forward (expert units) |
|---|---|---|---|
| reference A (Phase 1 records) | SDAR-1.7B-Chat-b32 | 2c83c05 | 364.64 |
| `F2_nopool` | Salesforce/CoDA-v0-Instruct (full-sequence diffusion, code-heavy corpus) | f2d40ce | 364.64 |
| `S1_nopool` | dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 (block 32) | 60f77bb | 126.32 |
| reference B | stock 30B decode, no drafter, no judge | — | — |

Benchmarks (full sets): HumanEval 164, MBPP 257, GSM8K-500, IFEval-541, MATH-500, GPQA-Diamond 198, AIME 90.

## Download everything (results, data, weights, code) with one command

```bash
pip install -U huggingface_hub   # provides `hf` (the old `huggingface-cli` no longer works in huggingface_hub 1.x)
hf download Arushhh/phase2-drafter-swap --repo-type dataset --local-dir phase2_bundle
```

Public, no token needed; 2,217 files, ~21 GB (tested 2026-10-05). Results page: https://claude.ai/artifact/JZBprrkKPKbX4XEgWyu8sK

## Where everything is

| what | where |
|---|---|
| code | GitHub `NoviceCoderInfinity/diffusion-moe-general-expert`, branch `phase2-drafter-swap` (`code/phase2/`) |
| text logs (records, running metrics, tables, reports, plots) | same repo, branch `phase2-logs` (replaced every 15 min) |
| everything incl. per-forward expert / judge / trace arrays | HF dataset `Arushhh/phase2-drafter-swap` (public; every 15 min) |
| judge weights + reference records | HF `Arushhh/phase2-drafter-swap` `assets/` (judge_128M_tok_000128000000, ref_S0_SDAR1.7B_judge128M_nopool, ref_stock_30B); originals: `Arushhh/c1b-judge-lora-sdar30b-b32`, `Arushhh/c1b-judge-corpus-waveA` |
| models / drafters | public HF repos at the pinned revisions above (`code/phase2/drafters.py`, `code/research_protocol.py`) |
| live metrics and plots | W&B `arushsharma536-spe-iit/phase2-drafter-swap` (one run per shard; `phase2-plots`) |

## Replicate on a new 8-GPU server

```bash
git clone -b phase2-drafter-swap git@github.com:NoviceCoderInfinity/diffusion-moe-general-expert.git dmge_p2
cd dmge_p2 && cp ~/.env .env      # HF_TOKEN; optional WANDB_API_KEY, DA_WANDB_PROJECT=phase2-drafter-swap, ENABLE_WANDB=1
tmux new -s p2 'bash code/phase2/replicate_nopool.sh 2>&1 | tee -a logs/phase2/replicate.log'
```

`replicate_nopool.sh` builds `/venv/p2`, downloads the 30B, the drafters, the references and the datasets, fetches the
128M judge and Phase 1's no-pool records, starts the three backup loops, decodes `S1_nopool` then `F2_nopool` on all
seven sets (8 shards each), then writes the tables and the analysis. Every step resumes after a restart.
About 1 h of setup, then about 3 h per drafter on 8x RTX PRO 6000.

## Outputs

- `logs/phase2/decode/<arm>/<set>/`: records, `*.running.jsonl`, `experts/`, `judge/`, `traces/` per item
- `logs/phase2/analysis/tables/TABLES.md` (`compare_table_p2.py`): accuracy, Δ vs reference A with 95% CI, experts per
  token, MoE-only / full-pipeline / drafter-only speedups, passes per block, draft tokens kept
- `logs/phase2/analysis/nopool/REPORT.md` (`analyze_nopool_p2.py`): is there a dip, why, scope for improvement
- `logs/phase2/plots/` (`plot_p2.py`) and W&B

Gotchas fixed on this branch: CoDA's `lm_head` must be loaded untied; the dLLM BD3LM layers need `past_key_value`
under transformers 4.54; record readers must skip `*.running.jsonl` (commits 193ac94, 955641a).
