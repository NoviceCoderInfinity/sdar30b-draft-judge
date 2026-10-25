# wide-when-sure

> # **MEASUREMENT RULE: SGLANG IS FOR DATA GENERATION ONLY. ALL BENCHMARK INFERENCE (TOKENS/S, PASSES, SECONDS PER ANSWER, ACCURACY) AND ALL TRAINING RUN ON THE HF/TORCH HARNESS (`run_da_decode.py`, `train_c1b.py`). NEVER REPORT A SPEED MEASURED IN SGLANG. NEVER TRAIN IN SGLANG.** (2026-10-21; see docs/PHASE5_JUDGE_ROUND.md)

> **Phase 1, 1B-token judge corpus (branch `phase1-datacorpus`, 2026-09-28): public Hugging Face artefacts.**
> Generated corpus (the 30B's own checked answers, block traces, 1.7B drafts, training items, prompt lists, eval sets, logs):
> https://huggingface.co/datasets/Arushhh/c1b-judge-corpus-waveA ·
> judge checkpoints (complete: LoRA adapter + optimizer/trainer state, verified file by file on the Hub):
> https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32 ·
> per-checkpoint links: [`docs/PHASE1_HF_ARTIFACTS.md`](docs/PHASE1_HF_ARTIFACTS.md) · how it is built: [`docs/CORPUS_1B_BUILD.md`](docs/CORPUS_1B_BUILD.md).

> **ICLR 2027 paper draft:** [`paper/`](paper/) — `main.tex` + compiled [`main.pdf`](paper/main.pdf) in the official ICLR 2027 style (anonymous, 9-page main text); every number is regenerated from `docs/results/r3_B/` by `paper/gen_tables.py`. Build with `make -C paper`. Overleaf (team): https://www.overleaf.com/project/6aa90cff1bb65519ab793cfb

> **Naming (2026-09-17).** *Standard split* = the earlier GSM8K/MATH training mix and its judge (`round2_t2b_union_flag/step003057`). *Nemotron split* = the code / GSM8K / verifiable-IF mix that `r3_A` and `r3_B` were trained on. Side-by-side comparison of the two, all arms, all six benchmarks: https://claude.ai/artifact/N1Teqns5a1sJR5vGZvwVhL. Run and folder names (`r3_*`, `round2_*`) are unchanged.
>
> **Nemotron split (2026-09-15) — results page https://claude.ai/artifact/BD1EFygV8vfBcnLKs7Zd4V · full report https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg · session handoff [`docs/HANDOFF_2026-09-15.md`](docs/HANDOFF_2026-09-15.md) · models, weights and results:** [`docs/MODELS_AND_RESULTS_2026-09-15.md`](docs/MODELS_AND_RESULTS_2026-09-15.md) · live results [`docs/results/r3_B/README.md`](docs/results/r3_B/README.md) · all weights & eval records on the public HF dataset [`Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts`](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts) (`Nemotron-split-moe-train/`: `r3_B/checkpoints`, `r3_A/checkpoints`, `round2_t2b_union_flag/step003057`, `reference_evals`).

Draft-aligned decoding for a 30B masked-diffusion MoE: can a small dense drafter
plus a tiny trained "judge" make **SDAR-30B-A3B-Chat-b32** write faster without
losing accuracy?

Curated snapshot, frozen 2026-09-09. Six decode settings, six evaluation
datasets, every number recomputed from the raw per-item records. Nothing in this
repository reports a checkpoint or a block width other than the two named above.

---

## Read this first: what this repository covers

Six decode settings of one checkpoint, `JetLM/SDAR-30B-A3B-Chat-b32`, all at
**block 32**, measured on the same frozen item sets:

| # | setting | drafter? | training? |
|---|---|:--:|:--:|
| 1 | **stock** — the model exactly as released | no | no |
| 2 | **blind accept** — the 1.7B drafter fills the block, the first stock pass keeps what it is confident in | yes | no |
| 3 | **trained judge** — a 27M LoRA + draft-flag adapter judges the draft | yes | **yes** |
| 4 | **judge + DES M=32** — the judge, plus a per-forward expert re-vote pooled to 32 | yes | **yes** |
| 5 | **judge + DES M=48** — the same, pooled to 48 | yes | **yes** |
| 6 | **stock + DES** — the expert pooling alone, no drafter and no adapter | no | no |

Two costs are being attacked: **passes/block** (how many times the 30B re-reads a
block) and **experts/token** (how much of the MoE wakes per generated token).
Settings 2–3 move the first, 4–6 move the second.

Every expert-work number for a drafter arm is reported twice — as measured, and
with the drafter's own forwards charged into the same unit ("+ drafter"). The
charged column is the honest one; see [RESULTS.md](RESULTS.md).

---

## What the project is

`JetLM/SDAR-30B-A3B-Chat-b32` is a **block-diffusion** language model: it fills a
whole 32-token block at once, re-running the network over the block until every
slot's top-1 probability clears a threshold. A block therefore costs many forward
passes (~10 on GSM8K), and each forward wakes 8 of 128 experts in each of 48
layers.

Two costs to attack:

* **passes/block** — how many times the 30B model re-reads the block.
* **experts/token** — how many (layer, expert) pairs execute per generated token.

The idea: a **dense 1.7B drafter** (`JetLM/SDAR-1.7B-Chat-b32`, same tokenizer and
[MASK] id) fills the block first. The 30B model then only has to *judge* a
concrete draft rather than denoise 32 blanks. Two ways to judge:

1. **Blind accept** (training-free): the block's first ordinary stock pass already
   produces a distribution at every masked slot. Read `q_i = p(draft token i)` off
   that same forward and accept the draft wherever `q_i > 0.50`. **Zero extra MoE
   forwards.**
2. **Trained judge**: a 26.7 M-parameter LoRA adapter, flag-gated to the tentative
   draft slots, judges a *draft-filled* block (no masks at all) and commits,
   keeps-as-tentative, or re-masks each slot.

Orthogonally, **DES pooling** restricts each block forward to a top-M expert pool
elected by the block's own rows (`--pool M --pool-rule des --pool-flush`), cutting
experts/token without touching the decode rule.

The name: the judge should commit **wide** (many slots at once) exactly **when
it is sure**, and fall back to masks when it is not.

---

## Result in one table

100 frozen GSM8K *train* prompts (`calibvalid`), all six settings on the **same
items**, block 32, threshold 0.95, adapter `t2b_union_flag/step003057`
(`_ad734af656`):

| setting | passes/blk | experts/token | + drafter charged | ms/token | acc % | Δacc vs stock [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| 1. stock | 9.31 | 1090.6 | 1090.6 | 187.1 | 95.0 | reference |
| 2. blind accept (training-free) | 7.93 | 962.7 | **1085.9** | 171.6 | 95.0 | +0.0 [+0.0, +0.0] |
| 3. trained judge | 6.00 | 780.8 | 903.8 | 148.2 | 94.0 | −1.0 [−4.0, +2.0] |
| 4. judge + DES M=32 | 6.67 | 463.9 | 587.8 | 100.9 | 88.0 | −7.0 [−13.0, −2.0] |
| 5. judge + DES M=48 | 6.09 | 612.7 | 735.5 | 124.0 | 94.0 | −1.0 [−4.0, +2.0] |
| 6. stock + DES M=32 (no drafter, no training) | 9.94 | **654.9** | **654.9** | 127.3 | 93.0 | −2.0 [−7.0, +3.0] |

n = 100 for every row; produced by `code/draftalign/compare_table.py` from the
records `code/draftalign/run_da_decode.py` wrote.

Three things this table says:

* **Blind accept is free and it is nothing.** It cuts passes/block by 15% and
  experts/token by 1.13x — but a drafter forward is not free. Charged at 364.64
  expert-equivalents (see [RESULTS.md](RESULTS.md) §3), the total is 1085.9 vs
  1090.6: **1.004x**. It costs nothing and it saves nothing.
* **The trained judge is real but small**, and it is not free either: 1090.6 →
  903.8 total = 1.21x, for −1.0 accuracy here and up to −14.0 on MBPP.
* **DES pooling alone beats the whole drafter line** on total compute: 654.9 with
  no drafter, no adapter and no training, versus 735.5 for the best judge arm.

---

## Round 3 (in progress)

As of 2026-09-14 a new judge adapter is being trained on a code-heavy mix. Nothing above comes
from it. Same checkpoint, same block width.

| domain | share | blocks (32 slots each) | source | a decode is kept only if |
|---|---:|---:|---|---|
| code | 35% | 8,297 | `nvidia/OpenCodeInstruct` — a frozen 12,000-item pool | the program passes the row's unit tests |
| maths | 35% | 8,400 | `openai/gsm8k` train, GSM8K only, re-decoded fresh | the answer is scored correct |
| instruction following | 30% | 6,861 | `nvidia/Nemotron-RL-instruction_following` + `allenai/IF_multi_constraints_upto5` + `allenai/RLVR-IFeval` | it passes every constraint under a strict IFEval-style checker |

**Initialisation:** fresh from the stock `JetLM/SDAR-30B-A3B-Chat-b32` @
`c351bbc37d240aa6871f167e8f92d694281b0c22`, with the stock drafter `JetLM/SDAR-1.7B-Chat-b32` @
`2c83c05d84be90a01c72443b2123ebf0376f6fd4`, block 32. No team checkpoint — not `step003057`, not
any other — is used as init or resumed from.

**Status:** preprocessing done (23,558 blocks, option B); training `r3_A` launched 2026-09-15 02:38 IST, 3 epochs on 8 GPUs — live status, every HF link (data + checkpoints) and resume instructions: [docs/ROUND3_STATUS_2026-09-15.md](docs/ROUND3_STATUS_2026-09-15.md). On the 2026-09-14 container,
anything that loads the model runs under `/venv/r3`, not `/venv/main`.

| read | for |
|---|---|
| [docs/TRAINING_RECIPE_2026-09-09.md](docs/TRAINING_RECIPE_2026-09-09.md) | the recipe, with the approved *Amendments 2026-09-14* at the top |
| [DATA.md](DATA.md) §7 | the three round-3 pools |
| [docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md](docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md) | every existing checkpoint and baseline, hash-verified |
| [WORKTREE.md](WORKTREE.md) §2.7, §3 | the round-3 commands, live status and the environment hazard |
| [AGENTS.md](AGENTS.md) §9 (l)–(s) | the round-3 footguns |
| **[HF: Nemotron-data-subset-training-diffusion-moe](https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-data-subset-training-diffusion-moe)** | the processed round-3 dataset (267 MB), checkpoints, sources, docs — public |
| [HUGGINGFACE.md](HUGGINGFACE.md) §9 | how the backup is laid out |

---

## Repository layout

```
README.md            this file
AGENTS.md            START HERE if you are an AI agent picking this up cold
ARCHITECTURE.md      the decode loop, what the LoRA touches, frozen/trained split
RESULTS.md           every number, its exact definition, its n, its producer
DATA.md              the six eval sets, their HF ids, the training pool provenance
EXCLUDED.md          what is NOT in this snapshot and where it still lives
HUGGINGFACE.md       every Hugging Face link: models, checkpoints, indices, records
WORKTREE.md          directory map, every command, and what is still open

code/
  draftalign/        the current research phase
    run_da_decode.py     decode/eval driver for all six settings   [ENTRY POINT]
    da_decode.py         the decode rules: blind, full judge, DES pooling
    adapter.py           manual LoRA + the draft-flag gate (no peft)
    train.py             trains the judge adapter                   [ENTRY POINT]
    train_forward.py     the custom forward training bypasses model.forward with
    dataset.py           state sampler: how a training state is corrupted
    losses.py            CE + certainty forcing, MASK column deleted
    drafter.py           the 1.7B drafter's block fill (s4 / s8 / dyn90)
    seqcache.py          the drafter's persistent KV cache
    moe_probe.py         fp32 slot metrics read off a judge forward
    eval_table.py        the canonical table -> results/reports/EVAL_TABLE.md
    compare_table.py     the head-to-head table incl. the "+drafter" column
    build_train_set.py   decode records -> training states
    build_drafter_set.py training set for the (rejected) drafter fine-tune
    train_drafter.py     full fine-tune of the 1.7B drafter (rejected at its gates)
    gpu_queue.sh         the job queue runner
    orchestrate_*.sh     the wave orchestrators actually used
    monitor_status.sh    read-only status snapshot
  research_protocol.py   MODELS, revisions, GEN_LEN, frozen item splits, hashing
  run_desvote.py         DES-Vote runner; owns build_items() for all six datasets
  run_research_baselines.py   the stock block-32 threshold baselines
  analyze_s1b.py         per-item scoring for all six datasets
  sdar_harness/          the stock decode + instrumented MoE (expert counters)
  vendor/                vendored OpenCompass dataset evaluators (scoring)
  compat/                flash_attn shim -- the model will not import without it
  configs/               frozen prompt sets + protocol json (hash-checked)
  tests/                 the decode-rule tests; these pin the stock equivalence

results/               reports, metrics, audits, figures (small, all committed)
docs/                   design docs, handoffs, the explainer deck
```

## How to reproduce

Prerequisites: 8x 96 GB GPUs (the runs used RTX PRO 6000 Blackwell), the two
models in a local HF cache, `/venv/main/bin/python`, and `code/` on the path via
the scripts themselves (each entry point inserts `code/` and `code/compat`).

```bash
# 1. stock baseline
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/run_da_decode.py \
    --model b32 --items gsm8k500 --stock --shard 0 --nshards 4 --gpu 0 --score

# 2. blind accept (training-free)
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/run_da_decode.py \
    --model b32 --drafter d32 --items gsm8k500 --draft s8 --judge blind \
    --tau-acc-q .50 --shard 0 --nshards 4 --gpu 0 --score

# 3. trained judge
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/run_da_decode.py \
    --model b32 --drafter d32 --items gsm8k500 \
    --adapter logs/draftalign/train/t2b_union_flag/step003057 \
    --judge full --draft s8 --shard 0 --nshards 4 --gpu 0 --score
```

All six invocations, verbatim from the queue files that produced the results, are
in [AGENTS.md](AGENTS.md). Settings 4–6 add the pooling flags
(`--pool 32|48 --pool-rule des --pool-flush`), with `--stock` in place of the
adapter for setting 6.

The stock reference cell on GSM8K-500 comes from a second script, because
`run_da_decode.py` hardcodes `BLOCK = 32` and takes its items differently:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/run_research_baselines.py \
    --model b32 --split gsm8k500 --cells B32:0.95 \
    --out logs/research_v1/baselines_b32 --shard 0 --nshards 3 --no-trace --score
```

Then score everything with `code/draftalign/eval_table.py` (canonical) and
`code/draftalign/compare_table.py` (head-to-head, includes the drafter charge).

## Where the bulk artifacts live

`logs/` (per-item records, per-block NPZ traces and judge checkpoints) and
`hf_cache/` (61 GB of model weights) are **not** in this repo. They are all on
Hugging Face, public and ungated: [HUGGINGFACE.md](HUGGINGFACE.md) has every
link, every checkpoint hash, and the `hf download` command for each piece.
[EXCLUDED.md](EXCLUDED.md) says what was left out and why.

Every table in [RESULTS.md](RESULTS.md) names the record path it came from, so a
reader who pulls the mirror can recompute it.

## Phase 2 drafter swap (branch `phase2-drafter-swap`): results bundle

Everything from the Phase 2 no-pool run (records, per-forward logs, analysis, tables, plots, datasets, judge weights, code snapshot):

```bash
pip install -U huggingface_hub
hf download Arushhh/phase2-drafter-swap --repo-type dataset --local-dir phase2_bundle
```

Guide and replication: `docs/PHASE2_NOPOOL_REPLICATE.md`. Results page: https://claude.ai/artifact/JZBprrkKPKbX4XEgWyu8sK
