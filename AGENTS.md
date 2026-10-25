# AGENTS.md — orientation for an AI agent picking this repo up cold

You are looking at a curated snapshot of a decoding-efficiency study on
`JetLM/SDAR-30B-A3B-Chat-b32`, a 30B block-diffusion MoE. Read this file before
touching anything. It is written to save you the two days of archaeology that
produced it.

`WORKTREE.md` is the companion: the directory map, every command in one place,
and the list of what is still open.

**Scope.** This repository covers exactly six decode settings of one checkpoint
(`JetLM/SDAR-30B-A3B-Chat-b32`) at block 32: stock, blind accept, trained judge,
judge + DES M=32, judge + DES M=48, and stock + DES. Everything here — code,
records, tables — belongs to those six. If a task asks you to compare against a
different checkpoint or a different block width, that work does not live in this
repository; say so rather than inventing a number.

**Round 3 (in progress, 2026-09-14)** trains a *new* judge on the same checkpoint and block
width, fresh from the stock base, on a 35 / 35 / 30 code / maths / IF mix
(`docs/TRAINING_RECIPE_2026-09-09.md`, *Amendments 2026-09-14*). Its entry points are §1.1, its
footguns §9 (l)–(s), its live status `WORKTREE.md` §3. No round-3 result exists yet; do not
quote one.

**Phase 2 (branch `phase2-drafter-swap`, 2026-09-29)** swaps the drafter for other released diffusion LMs on the
same frozen checkpoint and block width; the only thing it ever trains is the LoRA judge adapter. Everything lives in
`code/phase2/` and `docs/PHASE2_DRAFTER_SWAP.md`; its stop rules are frozen in `code/phase2/rules.json`. It runs in
`/venv/p2` (transformers 4.54.1 + the `LossKwargs` shim), not `/venv/r3`. No Phase 2 result exists yet.

**Phase 4 (branch `phase4-drafter-lora`, 2026-10-12)** trains the *drafter* for the first time: a LoRA (r 32, attention +
MLP, 34,865,152 parameters) on SDAR-1.7B, on the 1B corpus' waves A + B, for the rd1 / rd2 re-draft decoders. The judge
and its 128M LoRA stay frozen. Its adapter is **ungated**, so unlike the judge's it is merged for decoding (`--drafter
<train out>/merged`); `--redraft-ban-rejected` is its decode-time change. Everything lives in `code/drafter_lora/`
(stages: `p4.sh setup|smoke|train|gate1|gate2|gate3|gate4`) and `docs/PHASE4_DRAFTER_LORA.md`. Runs in `/venv/p2`.
No Phase 4 result exists yet.

**The single most important measurement:** expert *pooling* is where the cost
goes, not the drafter. `stock + DES M=32` is the cheapest arm at 654.9
experts/token and 93.0 on calibvalid, with no drafter and no training. The
trained judge buys its savings by giving up 1.4–14.0 accuracy points depending on
the dataset. See `README.md` and `RESULTS.md` §6.

---

## 1. Entry points

| file | what it is | run it how |
|---|---|---|
| `code/draftalign/run_da_decode.py` | **the** decode/eval driver — all six settings, one process per GPU, items sharded | CLI, §3 |
| `code/draftalign/da_decode.py` | the decode rules themselves (blind, full judge, DES pooling). Its module docstring is the spec; read it before changing any rule. | library |
| `code/draftalign/train.py` | trains the judge adapter (**not** `train_da.py` — that file does not exist) | `torchrun`, §4 |
| `code/draftalign/eval_table.py` | the canonical results table → `results/reports/EVAL_TABLE.md` | `python code/draftalign/eval_table.py` |
| `code/draftalign/compare_table.py` | head-to-head table, adds `exp/layer-fwd` and the honest `+drafter` column | `python code/draftalign/compare_table.py` |
| `code/run_research_baselines.py` | the stock block-32 threshold baselines (flat output dir) | CLI, §3.7 |
| `code/research_protocol.py` | `MODELS` (repo + pinned revision + cache dir), `GEN_LEN`, `MASK_ID`, frozen item splits, `config_hash`, `safe_out`, `LaunchLog` | library; `python code/research_protocol.py` prints the frozen-set hashes |
| `code/analyze_s1b.py` | per-item scoring for all six datasets (`per_item_correct`) via the vendored OpenCompass evaluators | library |
| `code/run_desvote.py` | DES-Vote runner; **owns `build_items()`**, the definition of all six eval sets | library + CLI |
| `code/draftalign/gpu_queue.sh` | the job queue runner (one command per free GPU, every 60 s) | §5 |

Import wiring: every entry point does
`sys.path.insert(0, code/)` and `sys.path.insert(0, code/compat/)` itself, then
`import research_protocol as P`. `research_protocol` sets `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and points `HF_MODULES_CACHE`
at the repo. **Run everything from the repository root**, not from `code/`.
**Exception:** a decode or scorer over a code pool executes generated programs in its cwd,
so round-3 code/IF shards run from a throwaway sandbox dir instead (§9 (p)). Outputs are
unaffected: `safe_out` joins a relative path to the repository root, not to the cwd.

### 1.1 Round-3 data pipeline (2026-09-14)

Every shell script below defaults `PY` to `/venv/r3/bin/python` and exports
`HF_HOME=/root/.cache/huggingface` (§9 (l)).

| file | what it is | run it how |
|---|---|---|
| `code/draftalign/build_opencode_pool.py` | freezes `code/configs/opencode_pool.json` (+ `.sha256`) from a local `nvidia/OpenCodeInstruct` snapshot; CPU only | `HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_opencode_pool.py` (its docstring) |
| `code/draftalign/build_if_pool.py` | freezes `code/configs/if_pool.json` (+ `.sha256`); CPU only | `HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_if_pool.py --n 6000` (its docstring) |
| `code/draftalign/if_scorer.py` | strict all-constraints IF scorer over the vendored IFEvalG checker (`code/vendor/ifeval/`) | library; runner wiring per `docs/IF_POOL_WIRING_PATCH.md` — TBD (IF pool build in progress) |
| `code/draftalign/launch_r3_datagen_gsm8k.sh` | Stage 1 for `gsm8k_pool:0:2400`, one shard per GPU (8); cwd = repo root, not sandboxed | `bash code/draftalign/launch_r3_datagen_gsm8k.sh` (`GPUS`, `NSHARDS` override) |
| `code/draftalign/launch_r3_datagen.sh` | Stage 1 for any pool slice, one shard per GPU in `GPUS`, each in its own sandbox cwd | `bash code/draftalign/launch_r3_datagen.sh <items> <name>` → `logs/draftalign/datagen/<name>_dv-s8_blind_q50_r3` |
| `code/draftalign/launch_r3_datagen_after.sh` | the same, but shard k starts on GPU k once the previous run's shard k writes `.done` | `nohup bash code/draftalign/launch_r3_datagen_after.sh <prev_dir> <items> <name> &` |
| `code/draftalign/queue_r3_lines.sh` | prints S stride-shard queue lines (sandbox cwd, absolute paths) for `sched_r3.sh` | `bash code/draftalign/queue_r3_lines.sh <items> <name> <S> >> <queue>` |
| `code/draftalign/sched_r3.sh` | two-level GPU scheduler: HI queue (code / IF shards) before LO queue (filler); never deletes its queue files | `nohup bash code/draftalign/sched_r3.sh <hi> <lo> <log> &` — stop: `touch <log>.stop` |
| `code/draftalign/build_r3_index.sh` | Stage 2 for one pool: `build_train_set.py --only-correct --max-seq 1200 --exclude-frozen` → `logs/draftalign/trainset_r3/<pool>/` | `bash code/draftalign/build_r3_index.sh gsm8k`, `… opencode <N>`, `… if <N>` |
| `code/draftalign/backup_r3_hf.py` | stages round-3 data, code, docs and env freezes and uploads them to the public HF folder `Nemotron-data-subset-training-diffusion-moe/` | `nohup /venv/r3/bin/python code/draftalign/backup_r3_hf.py --loop 30 > logs/draftalign/hf_backup_r3.log 2>&1 &` |

## 2. Models (pinned; `code/research_protocol.py::MODELS`)

| key | repo | revision | role |
|---|---|---|---|
| `b32` | `JetLM/SDAR-30B-A3B-Chat-b32` | `c351bbc37d240aa6871f167e8f92d694281b0c22` | the MoE under test: 48 layers, 128 experts/layer, top-8, hidden 2048, block 32 |
| `d32` | `JetLM/SDAR-1.7B-Chat-b32` | `2c83c05d84be90a01c72443b2123ebf0376f6fd4` | the dense drafter; same tokenizer, same `MASK_ID` |

`MASK_ID = 151669`, `EOS_IDS = [151645, 151643]`. `load_sdar()` asserts the
revision it loaded and asserts the tokenizer's mask id.

Round 3 trains from exactly these two stock revisions. The team checkpoints under
`logs/draftalign/train*/` are documentation only — never init, never `--resume`
(`docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md`).

## 3. The exact CLI for each of the six settings

Copied verbatim from `logs/draftalign/queue_matrix.txt` and the queue logs that
produced the published cells. Set `CUDA_VISIBLE_DEVICES` **outside**; `--gpu` is
only a label for the launch log. `--score` scores each item inline on CPU.

```bash
PY=/venv/main/bin/python
CK=logs/draftalign/train/t2b_union_flag/step003057      # the judge adapter

# (1) stock -- no drafter, no judge. The integrity twin: with --stock the same
#     loop must reproduce sdar_harness.decode.block_diffusion_generate token for
#     token (run_da_decode compares against the recorded stock cell, `ref_match`).
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --items gsm8k500 --stock \
  --shard 0 --nshards 4 --gpu 0 --score

# (2) blind accept -- TRAINING-FREE, zero extra MoE forwards
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --drafter d32 --items gsm8k500 --draft s8 --judge blind --tau-acc-q .50 \
  --shard 0 --nshards 4 --gpu 0 --score

# (3) trained judge
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --drafter d32 --items gsm8k500 --adapter $CK --judge full --draft s8 \
  --shard 0 --nshards 4 --gpu 0 --score

# (4) judge + DES pooling M=32
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --drafter d32 --items gsm8k500 --adapter $CK --judge full --draft s8 \
  --pool 32 --pool-rule des --pool-flush \
  --shard 0 --nshards 4 --gpu 0 --score

# (5) judge + DES pooling M=48   (identical, --pool 48)
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --drafter d32 --items gsm8k500 --adapter $CK --judge full --draft s8 \
  --pool 48 --pool-rule des --pool-flush \
  --shard 0 --nshards 4 --gpu 0 --score

# (6) stock + DES pooling alone -- no drafter, no adapter
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --items gsm8k500 --stock --pool 32 --pool-rule des --pool-flush \
  --shard 0 --nshards 4 --gpu 0 --score
```

### 3.7 A block size other than 32

`run_da_decode.py` has `BLOCK = 32` **hardcoded at module scope** (line ~57) and
no `--block` flag. The stock GSM8K-500 reference cell comes from the other
script, which shards its items differently:

```bash
CUDA_VISIBLE_DEVICES=0 $PY code/run_research_baselines.py \
  --model b32 --split gsm8k500 --cells B32:0.95 \
  --out logs/research_v1/baselines_b32 --shard 0 --nshards 3 --no-trace --score
```

`--cells` takes a comma list of `B<block>:<threshold>`; this repository uses
`B32:0.95` and `B32:0.90`. `--model` defaults to `b32`, the only 30B checkpoint
these scripts know about.

Records from `run_research_baselines.py` carry the same field set as
`run_da_decode.py`'s, so `eval_table.py`'s formulas apply unchanged. They land in
a **flat** directory, not one directory per cell.

### 3.8 Other flags worth knowing

| flag | meaning |
|---|---|
| `--draft {s4,s8,dyn90}` | drafter block-fill setting; `s8` = 8 drafter steps, used everywhere |
| `--threshold` | the stock commit threshold (strict `>`, bf16). Default 0.95 |
| `--tau-commit` / `--tau-override` | full judge: commit threshold, and a stricter one for commits that *override* the draft token |
| `--tau-acc-q` | blind: accept the draft when `q_i >` this. Default **0.30**; every published blind cell used **0.50** |
| `--full-remask-only` | full-judge ablation: never keep a tentative draft token |
| `--judge-passes K` | full-judge variant: K judge passes, then re-mask everything open and let the stock loop finish |
| `--draft-after-pass N` | blind variant ("anchored drafting"): N stock passes first, then draft only the still-open slots |
| `--tau-pool-cov` | full judge + pool: hold back commits on rows whose routing the pool pruned |
| `--limit` / `--no-trace` | cap items / skip the per-item NPZ trace |

## 4. Training the judge

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /venv/main/bin/torchrun --standalone --nproc_per_node=8 \
  code/draftalign/train.py \
  --index logs/draftalign/trainset_union_r2/index.jsonl \
  --out logs/draftalign/train/t2b_union_flag \
  --resume logs/draftalign/train/t2a_gsm8k_flag/step001057 \
  --steps 3057 --grad-accum 4 --pool-Ms 128 --no-train-gates \
  --draft-flag --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 \
  --max-seq 1200 --holdout-items 150 --eval-every 100 --save-every 200 \
  --log-every 10 --seed 1 --keep-checkpoints 2
```

(`code/draftalign/launch_t2b.sh` is this, wrapped.) One **full bf16 model per
rank** — no sharding, no DDP: the training forward bypasses `model.forward`
(see `train_forward.py`), so DDP's autograd hooks would never see the graph;
gradients are all-reduced manually.

Building the training index first:

```bash
bash code/draftalign/build_union_index.sh     # calls build_train_set.py 4x and concatenates
```

## 5. The job queue

```bash
nohup code/draftalign/gpu_queue.sh <queue_file> <log> ["0 1 2 3"] &
```

Each line of `<queue_file>` is one shell command with `{GPU}` substituted for a
free GPU index. A GPU counts as free when it holds < 1000 MiB and this runner did
not use it in the last 150 s. Stop it by removing the queue file.

**Round 3 uses `code/draftalign/sched_r3.sh` instead** (§1.1): two queue files,
`logs/draftalign/queue_r3_hi.txt` (code / IF shards) and `logs/draftalign/queue_r3_lo.txt`
(filler), polled every 20 s. A GPU holding < 1000 MiB with no live job of this scheduler takes
the next HI line, and a LO line only when HI is empty. It never deletes its queue files; a
started line moves to `<queue>.launched`. Stop it with `touch <log>.stop`. Lines come from
`queue_r3_lines.sh`; read §9 (o) before editing a live queue.

Typical queue line:

```
CUDA_VISIBLE_DEVICES={GPU} nohup /venv/main/bin/python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items math500 --adapter logs/draftalign/train/t2b_union_flag/step003057 --judge full --draft s8 --pool 32 --pool-rule des --pool-flush --shard 0 --nshards 4 --gpu {GPU} --score > logs/draftalign/train/mx_math500_judge_des32.shard0of4.log 2>&1
```

## 6. The cell-naming scheme

A **cell** is one (dataset × decode configuration). Its name is the record
filename and the directory name, built by `run_da_decode.py::cell_name`.

```
math500_B32_t95_dv-s8_full_c95_p48desf_ad734af656
│       │   │   │     │    │    │       └── _ad<hash8>  first 8 hex of sha256(adapter.safetensors)
│       │   │   │     │    │    └────────── _p<M><rule>[f]  pool M=48, rule "des", f = --pool-flush
│       │   │   │     │    └─────────────── c95  = --tau-commit 0.95 (as an integer percent)
│       │   │   │     └──────────────────── full = --judge full (the trained regime)
│       │   │   └────────────────────────── dv-s8 = draft-verify with the s8 drafter setting
│       │   └────────────────────────────── t95  = --threshold 0.95 (integer percent)
│       └────────────────────────────────── B32  = block size (always 32 from this script)
└────────────────────────────────────────── dataset family
```

Full grammar (`run_da_decode.py` lines ~76–154):

* stock: `{ds}_B32_t{t}_stock` + pool suffix + adapter suffix
  → `gsm8k_B32_t95_stock`, `gsm8k_B32_t95_stock_p32desf`
* blind: `{ds}_B32_t{t}_dv-{draft}_blind_q{tau_acc_q}` then optional
  `_d{tau_draft}`, `_rj` (rejudge), `_da{N}` (anchored drafting)
  → `gsm8k_B32_t95_dv-s8_blind_q50`, `gsm8k_B32_t95_dv-s8_blind_q50_da1`
* full: `{ds}_B32_t{t}_dv-{draft}_full_c{tau_commit}` then optional
  `_o{tau_override}` (only when it differs from `tau_commit`), `_ro`
  (`--full-remask-only`), `_mp{K}` (`--full-max-passes` ≠ 32), `_jp{K}`
  (`--judge-passes`)
  → `gsm8k_B32_t95_dv-s8_full_c90_o99_ad734af656`
* halves (the original two-forward judge, superseded):
  `{ds}_B32_t{t}_dv-{draft}_c{..}_a{..}_q{..}[_rj{k}]`
* a **fine-tuned drafter** adds `_dr<hash8>`; the stock `d32` adds nothing.
* pool suffix also takes `cv{pct}` when `--tau-pool-cov > 0`
  → `..._p32desfcv75_ad734af656`

Two `_ad` suffixes worth recognising: `_ad734af656` is **T2b step 3057**, the
checkpoint every current headline uses. `_adb3106a4d` is an *older* naming that
used the adapter's `config_hash` instead of its weight hash — that hash is the
same for every checkpoint of one recipe, so those cells silently mixed
checkpoints. This is why the scheme changed. Do not compare `_adb3106a4d` cells
across steps.

The **dataset family prefix** is *not* the `--items` value: `--items calibvalid`
and `--items gsm8k500` both produce `gsm8k_...` cells (the split goes in the
filename's second field instead). A record file is
`<cell>.<split>.shard{k}of{n}.jsonl`, e.g.
`gsm8k_B32_t95_stock.calibvalid.shard0of4.jsonl`.

## 7. Where records land

```
logs/draftalign/da_decode/<cell>/<cell>.<split>.shard{k}of{n}.jsonl        per-item records
logs/draftalign/da_decode/<cell>/<cell>.<split>.shard{k}of{n}.jsonl.done   shard completion marker
logs/draftalign/da_decode/<cell>/traces/<split>/<idx>.npz                  per-block trace
logs/draftalign/datagen/<name>/                                            training-data decodes
logs/draftalign/trainset_union_r2/index.jsonl                              the training index
logs/draftalign/train/<run>/step<NNNNNN>/adapter.safetensors               judge checkpoints
logs/research_v1/baselines_b32/<cell>.<split>.shard{k}of{n}.jsonl          run_research_baselines output (FLAT)
results/reports/EVAL_TABLE.md                                              regenerated by eval_table.py
results/audits/research_v1/launches.jsonl                                  every launch, via LaunchLog

# round 3 (2026-09-14)
logs/draftalign/datagen/<name>_dv-s8_blind_q50_r3/                         Stage 1 records, .done, shard logs, traces/  (e.g. gsm8k_pool_0_2400, opencode_pool_0_6000)
logs/draftalign/datagen/_failed/gsm8k_pool_0_2400_flashattn_importerror/   logs of the aborted first launch (§9 (l))
logs/draftalign/trainset_r3/<pool>/                                        Stage 2: index.jsonl, build_summary.json, items/
logs/draftalign/queue_r3_{hi,lo}.txt  (+ .launched)                        sched_r3.sh queues and what it started
logs/draftalign/sched_r3.log, logs/draftalign/hf_backup_r3.log             scheduler and HF-backup logs
/dev/shm/dlm/sandbox/<name>/shard<k>of<S>/                                 throwaway cwd of each code/IF shard (§9 (p))
```

None of `logs/` is in this repo (120 GB). See `HUGGINGFACE.md` for every mirror
link and the `hf download` line for each piece, `EXCLUDED.md` for what was cut.
On the 2026-09-14 container `logs` and `hf_cache` are symlinks into `/dev/shm/dlm/`
(§9 (m)); round-3 data is backed up to HF every 30 min (`HUGGINGFACE.md` §9).

A record is one JSON object per item. The fields the metrics use:
`n_forward, n_prefill, n_flush, n_blocks, passes_per_block[], n_gen_to_eos,
unique_expert_layer_loads, layer_forwards, seconds, correct, idx`, plus the
judge diagnostics `committed_judge_per_block[], full_commit_eq_draft_per_block[],
tentative_per_block[], remask_per_block[], full_passes_per_block[][{commit,
tentative, remask, forced}], drafter_n_by_kind{}, draft_agreement_per_block[]`,
plus provenance `cell, config_hash, model_key, model_revision, adapter,
adapter_hash, gpu, utc, ref_match`.

## 8. Metric definitions — use these verbatim

```
passes/blk = sum(n_forward - n_prefill - n_flush) / sum(n_blocks)
exp/token  = sum(unique_expert_layer_loads) / sum(n_gen_to_eos)
ms/token   = 1000 * sum(seconds) / sum(n_gen_to_eos)
"+ drafter" charges drafter forwards at 364.64 expert-equivalents per forward.
```

All three are **block-** or **token-weighted sums over items**, not means of
per-item means. Notes:

* `sum(passes_per_block)` is identically `n_forward - n_prefill - n_flush`;
  the code computes it the first way (`da_decode.py`, `BlockTrace`), the
  definition above is the equivalent identity. Verified numerically on records.
* `eval_table.py` publishes a **second** column, `fwd/blk (im)`, the mean over
  items of each item's own mean. It differs (calibvalid stock: 9.31 vs 9.02).
  Every number in `RESULTS.md` is the block-weighted one. Do not mix them.
* `364.64 = 1,720,573,440 / 4,718,592` = (SDAR-1.7B parameters *excluding* its
  token-embedding matrix) / (one 30B expert's parameters = 3 × 2048 × 768).
  Defined at `code/draftalign/compare_table.py:23`. `drafter_forwards` is
  `sum(drafter_n_by_kind.values())` — prefill + draft + redraft + flush.
* `unique_expert_layer_loads` counts **distinct (layer, expert) pairs actually
  dispatched**, read from `sdar_harness.moe_instr.SINK` (the very
  `selected_experts` tensor the dispatch indexes). Under a pool it counts the
  *restricted* set, i.e. the experts that really ran.
* Accuracy is exact-match through the vendored OpenCompass evaluators
  (`code/vendor/oc_datasets/`), per item, via `analyze_s1b.per_item_correct`.
* Paired Δaccuracy is a bootstrap over items, **2,000 resamples, seed 0**, on the
  items the two cells share. The point estimate is stable; the CI endpoints
  wobble by ±0.2–0.3 between runs because the RNG is shared across rows.

## 9. Known footguns

**(a) `BLOCK = 32` is hardcoded in `run_da_decode.py`.** Module scope, no flag,
and it will not warn you. Every result in this repository is a block-32 result.

**(b) The dataset-family map must list all six datasets — and every training pool.**
`run_da_decode.py` lines ~333–339 (and the identical block in
`run_research_baselines.py`, lines ~196–202) map the split name to a scorer family:

```python
ds = ("mbpp"      if base.startswith("mbpp")      else
      "mbpp"      if base.startswith("opencode")  else   # opencode_pool: gold = unit-test asserts, MBPP sandbox
      "math500"   if base.startswith("math")      else
      "aime"      if base.startswith("aime")      else
      "gpqa"      if base.startswith("gpqa")      else
      "humaneval" if base.startswith("humaneval") else
      "gsm8k")
```

The fall-through is **`gsm8k`**, not an error. Add a dataset without adding a
branch and every item is scored with the GSM8K numeric-answer extractor, which
on code or multiple-choice output yields a silent floor of near-zero accuracy
that looks like a real (catastrophic) result. If you add a dataset, edit this map
in **both** runners, add the branch in `analyze_s1b.per_item_correct`, and add a
`GEN_LEN` entry in `research_protocol.py` (a missing key raises, which is the one
place it fails loudly).

Round 3 (2026-09-14) added the `opencode*` → `mbpp` line to **both** runners. The IF line
(`if*` → the IF scorer), with its `POOLS`, `GEN_LEN` and `per_item_correct` entries, is
specified by `docs/IF_POOL_WIRING_PATCH.md` — **TBD (IF pool build in progress)**. Until it
lands, `run_da_decode.py` stops on an `if_pool` split with a `KeyError` at `GEN_LEN[base]`;
a partial patch that registers `POOLS` and `GEN_LEN` but not the family line would score every
IF item with the GSM8K extractor — the silent case above. `DATA.md` §6 tracks each step per pool.

**(b2) The reporting scripts have their own dataset lists, and they are shorter.**
`eval_table.py`'s `splits` dict covers only `calibvalid, gsm8k500, math500, mbpp`;
`compare_table.py`'s `SPLITS` likewise. **AIME, GPQA and HumanEval cells exist on
disk but appear in no generated report.** If you regenerate the tables and a
dataset vanishes, this is why — add it to those dicts.

**(c) `gpu_queue.sh` deletes its own queue file when it drains.**
`if [ ! -s "$Q" ]; then ... rm -f "$Q"; break; fi`. Removing the queue file is
also the documented way to *stop* the runner, so the two are the same event and
your queue is gone either way. **Always keep a `.bak`** — the live tree has
`queue_matrix.txt.bak`, `queue.paused.bak_1h30`, `queue_baselines.txt.bak` for
exactly this reason. Also note the orchestrators pause the queue by
`mv queue.txt queue.paused` and resume with the reverse; a pause plus a drain
can lose work if you are not holding a backup.

**(d) A monitoring shell can stall the whole wave.**
The orchestrators gate on `pgrep -fc 'run_da_decode.py --model'` /
`pgrep -fc 'train.py --index'` and wait for the count to hit zero. Any process
whose *command line* contains that literal string is counted — including your own
`grep`, and including a long-lived monitoring shell that has the pattern in its
`bash -c` argument. The wave then waits forever for a job that is not running.
Use bracketed patterns everywhere: `train[.]py`, `gpu_q[u]eue.sh`,
`run_da_decode[.]py`. `monitor_status.sh` shows the safe idiom — snapshot `ps`
once, then match against the snapshot:

```bash
PSOUT="$(ps -eo pid,args --no-headers | grep -v shell-snapshots)"
cnt(){ printf '%s\n' "$PSOUT" | grep -c -- "$1"; }
```

**(e) A flag-gated adapter must NOT be merged.** `adapter.py::merge_lora`
*refuses* on an adapter whose metadata says `draft_embed: true`, because a merged
weight applies at every position — which is precisely what the per-position gate
exists to prevent. `run_da_decode.load_adapter_into` loads such an adapter
un-merged and arms the flag only on judge passes. Every other forward runs with
an all-False flag, under which the model is bit-identical to the frozen base.
If you "optimise" by merging, you silently change every prefill, flush and stock
denoise pass.

**(f) `_ad<hash>` is the adapter **weights** hash, not its config hash.** Cells
named with the config hash (`_adb3106a4d`) collide across checkpoints of the same
recipe. `adapter_weights_hash()` exists because of that bug.

**(g) Two independent stock `calibvalid` decodes exist**, one under
`logs/research_v1/baselines_b32/` and one under `logs/draftalign/da_decode/`.
They are bit-identical in `gen_ids` and `passes_per_block` for all 100 items;
only wall-clock differs (184.3 vs 187.1 ms/token). `compare_table.py` uses 187.1.
Quote the one that matches the arm you are comparing against.

**(h) `--pool-rule des` is a per-forward re-vote, not a static coreset.** It
sums the **unnormalised** top-k router weights and re-elects the pool on every
forward, per layer (DES-Vote Algorithm 3). A static per-block coreset frozen from
a single readout forward is a different algorithm with different numbers; do not
mix the two.

**(i) `--stock` is an integrity twin, and it is checked.** With `--stock`,
`run_da_decode.py` must reproduce `sdar_harness.decode.block_diffusion_generate`
token for token and pass for pass; it compares every item against the recorded
stock cell and writes `ref_match` into the record. If you change the decode loop
and `ref_match.gen_ids` goes False on a `--stock` run, you broke the baseline.
`code/tests/test_da_decode.py` pins the same property offline.

**(j) `HF_*_OFFLINE=1` is set on import of `research_protocol`.** Nothing will
download. Both models (b32, d32) must already be in `hf_cache/hub`, and datasets
in the HF datasets cache.

**(k) Never write outside the repo.** `research_protocol.safe_out()` refuses any
path outside the repository root and refuses `/workspace/stage0`,
`/workspace/dlm-general-expert-exp1` and `/workspace/.hf_home` explicitly.
Use it for every output path.

**(l) On the 2026-09-14 NGC container, `/venv/main` cannot load the model — use `/venv/r3`.**
The image's system Python carries a real `flash_attn` 2.7.4.post1, and `/venv/main` there is a
`--system-site-packages` venv (transformers 4.52.4) that sees it. transformers then imports
`flash_attn.layers.rotary` (`modeling_flash_attention_utils.py` line 38, reached from the remote
`modeling_sdar_moe.py` line 36); `code/compat` is first on `sys.path`, so `flash_attn` resolves
to the shim, which has no `layers`, and the load dies with `ModuleNotFoundError: No module named
'flash_attn.layers'`. The first round-3 launch died this way on all 8 shards (logs in
`logs/draftalign/datagen/_failed/gsm8k_pool_0_2400_flashattn_importerror/`). `/venv/r3` is an
isolated venv (no system site-packages) with the team's exact pip freeze; every round-3 launcher
defaults to it. Datasets must be read with `HF_HOME=/root/.cache/huggingface` — the launchers
export it, and under (j)'s offline flags a wrong `HF_HOME` cannot fall back to a download.
Details: `WORKTREE.md`, *Known environment hazard*.

**(m) `logs/` and `hf_cache/` live in RAM.** On the 2026-09-14 container both are symlinks
into `/dev/shm/dlm/` (tmpfs, wiped on restart), as are the OpenCodeInstruct snapshot
(`/dev/shm/dlm/data/`), the env freeze (`/dev/shm/dlm/env/`), the backup stage and the
sandboxes. A restart loses every decode record and the model weights. `backup_r3_hf.py --loop 30`
pushes round-3 data to Hugging Face every 30 min (`HUGGINGFACE.md` §9); anything newer than the
last sync is gone, and the weights must be pulled again (`HUGGINGFACE.md` §2).

**(n) Round-3 decodes are stride-sharded; trim by shards, never by killing one.**
`run_da_decode.py` takes `items[shard::nshards]`, so any set of finished shards is a uniform
sample of the slice — `opencode_pool:0:6000` is queued as 32 shards of 187–188 items so it can
be cut to the block target once yield is measured. Trim by deleting the not-yet-started lines
from the queue. `build_r3_index.sh` refuses unless every record file present has its
`.jsonl.done`, so a shard killed mid-run blocks Stage 2 until its partial `.jsonl` is moved
out. Its `opencode N` / `if N` argument names the **decoded slice** (`opencode 6000` reads
`opencode_pool_0_6000_…` with `--split-name opencode_pool:0:6000`), not the number of items that
finished.

**(o) `sched_r3.sh` never deletes its queue files — and consumes a line at launch, not on
success.** Unlike `gpu_queue.sh` (c), a started line is removed from `<queue>` and appended to
`<queue>.launched`. So: a line whose command fails at once (a missing `.env`, a typo) is gone
from the queue and must be re-appended by hand; one pop removes *every* identical copy of the
line (`grep -v -x -F`), so duplicates launch once; `touch <log>.stop` ends the scheduler but not
the jobs it started; it exits on its own once both queues are empty and none of its jobs is
alive, after which appended lines wait for a restart. A GPU counts as free below 1000 MiB, so a
job started outside the scheduler is only seen once it holds memory. HI wins only at launch — a
running LO job is never pre-empted.

**(p) Never run a code-pool decode or scorer from the repository root.** `--score` on a code
split executes the generated program: the vendored `oc_datasets/mbpp.py::execution` runs it in a
`multiprocessing.Process` with `exec` and a timeout, but with **no `chdir` and no reliability
guard**, and `analyze_s1b._humaneval_score` calls the same `execution()`. The child inherits the
parent's cwd, so a program that creates, overwrites or deletes files does it wherever the decode
was launched. It happened on 2026-09-14: a 12,000-item OpenCodeInstruct reference-solution check
launched from the repository root dumped 190 files and directories into it (`test*.json`,
`*.db`, `*.pkl`, `output*.html`, `workflow_deletion_log.txt`, `test_path/`). No tracked file was
modified or deleted, and the files were removed by an exact creation-time list. The fix needed no
code change: `queue_r3_lines.sh`, `launch_r3_datagen.sh` and `launch_r3_datagen_after.sh` run
each shard with cwd = its own throwaway `/dev/shm/dlm/sandbox/<name>/shard<k>of<S>` and absolute
repo paths. That is safe because every output path in the decode path resolves against the
repository root through `safe_out`. Do the same for any ad-hoc scorer run over `mbpp`,
`humaneval` or `opencode_pool`.

**(q) An OpenCodeInstruct cell is named `mbpp_…`.** The family map sends `opencode*` to `mbpp`,
and `cell_name`, the W&B group (`group=cell`) and the default `ref_cell` all take the family:
records are `mbpp_B32_t95_dv-s8_blind_q50.opencode_pool:0:6000.shard{k}of32.jsonl`. Only the
split token and the `_r3` datagen directory distinguish them from the MBPP eval cell
`mbpp_B32_t95_dv-s8_blind_q50`. Never glob `mbpp_*` across `datagen/` and `da_decode/` together.

**(r) The round-3 backup is public, and a secret-scan hit stops it.** `backup_r3_hf.py` uploads
to the **public** `Arushhh/dlm-exp1-drafter-voting-research-artifacts`. It stages an explicit
allowlist only and scans the stage for the `.env` values of `HF_TOKEN` / `WANDB_API_KEY` and for
`hf_[A-Za-z0-9]{30,}` before every upload; a hit raises `SystemExit`, which **ends `--loop`**
(a transient Hub error is only logged). If the folder README's "Last sync" stops advancing, read
`logs/draftalign/hf_backup_r3.log`.

**(s) `launch_r3_datagen_after.sh` needs the shard count of the run it waits on.** Shard k waits
for `<prev_dir>/*.shard<k>of<N>.jsonl.done` with N = the number of GPUs in *this* launch; if the
previous run used a different `--nshards`, the wait never ends.

## 10. Tests

```bash
/venv/main/bin/python -m pytest code/tests -q     # from the repository root
```

`code/tests/` is CPU-only and does not load a model. It pins the things that are
easy to break: stock equivalence (`test_da_decode.py`), the full-judge commit
rule (`test_da_decode_full.py`), anchored drafting
(`test_da_decode_anchored.py`), the draft-flag gate's identity at an all-False
flag (`test_draftalign_flag.py`), the training state sampler
(`test_draftalign_train.py`), the DES vote (`test_desvote.py`), and the pool
rules (`test_protocol_pools.py`). Run them before and after any decode-rule
change.

`test_if_scorer.py` (2026-09-14) pins the round-3 IF scorer: a compliant and a violating
response per instruction id, the all-constraints rule, reasoning stripping, the fail-closed
record wrapper, gold validation, determinism, the checker quirks behind
`build_if_pool.EXCLUDED_IDS`, and the frozen `if_pool.json`'s integrity (skipped while the pool
is absent). It needs the checker's pure-Python deps and NLTK `punkt_tab` in `/root/nltk_data`,
installed in both `/venv/main` and `/venv/r3` (`code/vendor/ifeval/PROVENANCE.md`).

## 11. What was being run when this snapshot was taken

A `gpu_queue.sh` wave over `logs/draftalign/queue_matrix.txt`: the 6-setting ×
6-dataset matrix (DES32 / DES48 under stock and under the judge). Several
math500 pool cells are partial — `n` is stated per row in `RESULTS.md`, and a
partial cell is never averaged with a complete one.

`docs/DRAFT_ALIGNED_DESIGN_2026-09-06.md` is the design document for the judge.
`docs/ICLR2027_DATA_PLAN.md` is the forward data plan for the next training
round: what to draw, from where, how much. Round 3 does not follow its phases: it draws per
`docs/TRAINING_RECIPE_2026-09-09.md` → *Amendments 2026-09-14*, and what it is running now is in
`WORKTREE.md` §3.
