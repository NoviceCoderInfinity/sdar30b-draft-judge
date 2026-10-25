# Phase 1 continuation: 128M → 228M (branch `phase1-continue-228m`)

More of the same SFT, nothing changed: the stock SDAR-30B answers new prompts (SGLang, threshold 0.95), the SDAR-1.7B
drafter drafts every kept block, `to_trainset.py` builds the items, and `train_c1b.py` continues the judge from
`main_128m/tok_000128000000` with the 128M run's exact recipe. Generation continues wave B where Phase 1 stopped.

## What exists (8 Oct 2026)

| | |
|---|---|
| Judge | 128,003,432 tokens, step 11,675 (`Arushhh/c1b-judge-lora-sdar30b-b32`, `main_128m/tok_000128000000`) |
| Converted corpus | 134.4M tokens: wave A 74.76M (165,438 items), wave B 59.67M (166,380 items) |
| 128M run's index | `index_128m.jsonl`, 106.2M tokens; ~6.7M of it never trained (the index builder replays the run to find them) |
| Generated, never converted | wave-B code from 30 Sep: 65,536 prompts decoded, 48,459 kept ≈ 24.3M tokens (converted first) |
| Never decoded | maths 616,768 · code ~553K · chat 257,243 prompts. Science and IF pools are exhausted: their 15% + 15% of the mix are further passes (≤ 4 per item, as before) |

## What this run does

1. **Setup** (`ops/c228_setup.sh`, ~1–1.5 h): CUDA 12.9 toolkit, the two venvs, pinned models, ~8 GB of this run's data
   from HF (prompt lists with sha256 check, wave-B state, indexes, item tars, the 128M checkpoint), SGLang smoke test
   (GSM8K-500: accuracy 90.2 ± 3, passes/block 10.06 ± 5%).
2. **Run** (`ops/c228_run.sh`): plan from the rental end (`ops/c228_plan.py`) → convert the pending code → generate
   maths, chat, code in time boxes (unbiased hash order) with drafts and items after each → build `index_228m.jsonl`
   (`build_index_228m.py`) → train to 228M (or until the planned stop) → base-decode eval vs 128M and stock
   (`ops/c228_eval.sh`, `ops/c228_compare.py`) → final push and checks (`ops/c228_verify.py`) → `READY_TO_KILL`.
   Data goes to HF every 20 min and after every stage; checkpoints every 5 min (every 2M tokens), verified file by file.

## Budget (8× RTX PRO 6000, rates measured in Phase 1)

| step | hours |
|---|---:|
| setup | 1.2 |
| convert pending wave-B code (24.3M) | 0.2 |
| generate maths 35.7M / chat 28.6M / code 11.4M | 3.5 / 1.3 / 0.2 |
| drafts for the new data | 0.9 |
| model loads, index, pushes | 1.2 |
| train 128M → 228M (9.5M tokens/h) | 10.5 |
| eval (7 benchmarks, base decode) | 2.5 |
| **total, MODE=full** | **≈ 22 h (~172 GPU-h), ~$255 at $11.7/h** |

MODE=box (a fixed rental): ~11 h converts ~51M and trains ~54M with a quick eval (GSM8K-500, HumanEval, MBPP); ~10 h
converts ~48M and trains ~47M. The plan is printed at the start (`logs/c1b/c228_state/plan.md`).

Measured rates (kept corpus tokens per GPU-hour): maths 1.28M, chat 2.73M, code 6.32M; drafts 68–94 blocks/s/GPU;
training 9.5M tokens/h. A slower host CPU (the trainer and decoder are CPU-launch-bound) can stretch training and eval.

## Box

8× RTX PRO 6000 Blackwell 96 GB (WS or Server), ≥ 64 CPU cores (fast single thread, e.g. EPYC 9xx4), ≥ 512 GB RAM,
≥ 500 GB disk, driver for CUDA ≥ 12.8. The prompt for the box agent is `docs/INSTANCE_PROMPT_C228.md`.
