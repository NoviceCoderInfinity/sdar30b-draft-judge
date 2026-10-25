# Retrieving everything from Phase 1 (one command, one directory)

```bash
curl -sL https://raw.githubusercontent.com/NoviceCoderInfinity/diffusion-moe-general-expert/phase1-datacorpus/code/corpus1b/ops/retrieve_all.sh | bash -s phase1_all
# options: WEIGHTS=final (only final checkpoints) | WEIGHTS=none ; ITEMS=0 (don't unpack training-item tars)
```

The result is one directory laid out exactly like the training box (`/workspace/dmge`):

| path | what | source |
|---|---|---|
| `code/`, `docs/`, `README.md`, `RESULTS.md` | all code (pipeline, trainer, decoder, ops scripts in `code/corpus1b/ops/`) and write-ups | GitHub `phase1-datacorpus` |
| `docs/results/phase1/` | eval summaries (11.79M, 28.5M, 128M, 128M no-pool, AIME 4K), no-pool vs DES48 table, gate_A, handoff, seen-data report | GitHub |
| `data/c1b/prompt_lists/` | the 2.24M prompts (5 domains, wave A/B), late-decontam lists; `data/c1b/eval/` eval sets | HF dataset |
| `logs/c1b/gen/wave{A,B}/gpu*/` | the 30B's generated answers (`records.*.jsonl.gz`), done markers, per-chunk stats | HF dataset |
| `logs/c1b/draft/wave{A,B}/gpu*/` | the 1.7B drafter's block drafts | HF dataset |
| `logs/c1b/trainset/wave{A,B}/` | training items (`index.jsonl`, `items_tar/*.tar` → `items/`), `index_128m.jsonl`, pinned probe holdout | HF dataset |
| `logs/c1b/train/{main,main_128m}/` | training logs, probe, config; **checkpoints** (86: 0→28.5M and 28.5M→128M, adapter + optimizer state) | HF dataset + HF model repo |
| `logs/c1b/eval/*/` | every per-question eval record and summary (checkpointA/B, checkpoint128M, checkpoint128M_nopool, probe128M, aime4k), incl. GPQA | HF dataset |
| `logs/c1b/analysis/`, `logs/c1b/dl/` | analyses (GPQA teacher-forced probe) and the exact ops scripts/logs | HF dataset |

Repos: code https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert (branch `phase1-datacorpus`) ·
data https://huggingface.co/datasets/Arushhh/c1b-judge-corpus-waveA · weights https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32
