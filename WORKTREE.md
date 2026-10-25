# WORKTREE — the map, the commands, and what is still open

One file. If you are an agent picking this repository up cold, read `AGENTS.md`
for *how the code works* and this file for *where everything is and what to run*.

Scope: six decode settings of **one** checkpoint, `JetLM/SDAR-30B-A3B-Chat-b32`,
all at **block 32**. Nothing here reports any other checkpoint or block width.

Round 3 (in progress, 2026-09-14) trains a new judge on that same checkpoint and block,
fresh from the stock base: commands §2.7, status §3.

---

## 1. Directory map

| path | what is in it | you need it when |
|---|---|---|
| `README.md` | the six settings, the headline numbers, how to reproduce | first read |
| `AGENTS.md` | entry points, pinned models, the exact CLI, record schema, footguns | before running or changing anything |
| `ARCHITECTURE.md` | the decode loop, what the LoRA touches, frozen vs trained split | changing a decode rule or the adapter |
| `RESULTS.md` | every number, its exact definition, its `n`, its producer | quoting a result |
| `DATA.md` | the six eval sets, their HF ids, the training-pool provenance | building or auditing training data |
| `EXCLUDED.md` | what is *not* here (120 GB of `logs/`, 61 GB of weights) and the mirror | looking for a record file |
| `WORKTREE.md` | this file | orienting |
| `code/draftalign/` | **the current pipeline** — decode driver, decode rules, adapter, trainer, data build, tables | almost everything |
| `code/sdar_harness/` | the stock decode loop and the instrumented MoE (expert counters) | verifying stock equivalence, expert accounting |
| `code/tests/` | CPU-only tests; no model is loaded | before and after any decode-rule change |
| `code/configs/` | frozen prompt sets + the protocol JSON (sha-checked on load) | never edit by hand |
| `code/compat/` | `flash_attn` shim — **the model will not import without it** | never delete |
| `code/vendor/` | vendored OpenCompass evaluators, used for scoring | scoring |
| `code/research_protocol.py` | `MODELS`, revisions, `GEN_LEN`, frozen splits, `safe_out`, `config_hash` | any script |
| `code/run_desvote.py` | owns `build_items()` — the definition of all six eval sets | any script (imported by `research_protocol`) |
| `code/run_research_baselines.py` | the stock block-32 threshold cells; flat output dir | the GSM8K-500 stock reference |
| `docs/` | the judge design doc, the training explainer, the trainer/probe specs, the forward data plan, the round-3 training recipe | understanding *why* |
| `results/reports/` | `EVAL_TABLE.md` (generated) and the calibvalid output samples | reading generated tables |
| `results/metrics/draftalign/` | drafter accuracy/agreement, holdout precision, progression | drafter and judge gate metrics |
| `results/figures/draftalign/` | the six figures and the scripts that draw them | rebuilding a figure |
| `results/audits/draftalign/` | the sanity-check run | checking the harness |
| `docs/TRAINING_RECIPE_2026-09-09.md` | the round-3 recipe, with the approved *Amendments 2026-09-14* at the top | before any round-3 work |
| `docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md` | every checkpoint and baseline record, hash-verified; why round 3 starts from stock | choosing an init or a comparison |
| `docs/IF_RECIPE_2026-09-14.md`, `docs/IF_POOL_WIRING_PATCH.md` | the verifiable-IF recipe and the runner wiring for `if_pool` — **TBD (IF pool build in progress)** | the IF slice |
| `code/configs/opencode_pool.json`, `code/configs/if_pool.json` | the frozen round-3 code and IF pools, each with a `.sha256` | never edit by hand — rebuild with `build_opencode_pool.py` / `build_if_pool.py` |
| `code/vendor/ifeval/` | vendored IFEvalG checker (open-instruct); provenance in its `PROVENANCE.md` | scoring IF (`code/draftalign/if_scorer.py`) |
| `code/draftalign/*_r3*` | the seven round-3 scripts: three datagen launchers, `queue_r3_lines.sh`, `sched_r3.sh`, `build_r3_index.sh`, `backup_r3_hf.py` | §2.7 |
| `/venv/r3` | isolated venv with the team's pip freeze | anything that loads the model on the 2026-09-14 container (§3, *Known environment hazard*) |
| `/dev/shm/dlm/` | RAM-backed working tree, **wiped on restart**: `hf_cache/`, `logs/` (both symlinked from the repo root), `data/` (the OpenCodeInstruct snapshot), `env/`, `sandbox/`, `hf_backup_stage/` | knowing what a restart loses |

**Not in this repository:** `logs/` (120 GB — per-item records, NPZ traces, judge
checkpoints, datagen decodes, training indices) and `hf_cache/` (61 GB — model
weights). Both are on the public HF mirror named in `EXCLUDED.md` §4. Every table
in `RESULTS.md` names the record path it came from, so a reader who pulls the
mirror can recompute it. On the 2026-09-14 container both are symlinks into
`/dev/shm/dlm/`; round-3 data is pushed to the public HF folder
`Nemotron-data-subset-training-diffusion-moe/` every 30 min (`HUGGINGFACE.md` §9).

---

## 2. Every command you need

```bash
PY=/venv/main/bin/python
CK=logs/draftalign/train/t2b_union_flag/step003057      # the judge adapter
```

Run everything **from the repository root**. Set `CUDA_VISIBLE_DEVICES` outside
the command; `--gpu` is only a label for the launch log. `--score` scores each
item inline on CPU.

On the 2026-09-14 NGC container use `PY=/venv/r3/bin/python`: `/venv/main` there fails at
model import (§3, *Known environment hazard*). A decode that scores code (`mbpp`,
`humaneval`, `opencode_pool`) executes generated programs in its cwd — run it from a sandbox
dir, not the repository root (§2.7; `AGENTS.md` §9 (p)).

### 2.1 The six decode settings

```bash
# (1) stock — no drafter, no judge. Integrity twin: with --stock this loop must
#     reproduce sdar_harness.decode.block_diffusion_generate token for token.
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --items gsm8k500 --stock \
  --shard 0 --nshards 4 --gpu 0 --score

# (2) blind accept — training-free, zero extra MoE forwards
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

# (6) stock + DES pooling alone — no drafter, no adapter
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/run_da_decode.py \
  --model b32 --items gsm8k500 --stock --pool 32 --pool-rule des --pool-flush \
  --shard 0 --nshards 4 --gpu 0 --score
```

`--items` takes `gsm8k500 | math500 | mbpp | humaneval | gpqa | aime | calibvalid`.

### 2.2 The stock GSM8K-500 reference cell

`run_da_decode.py` has `BLOCK = 32` hardcoded at module scope and shards items
differently, so the GSM8K-500 stock reference comes from the other script:

```bash
CUDA_VISIBLE_DEVICES=0 $PY code/run_research_baselines.py \
  --model b32 --split gsm8k500 --cells B32:0.95 \
  --out logs/research_v1/baselines_b32 --shard 0 --nshards 3 --no-trace --score
```

Its records carry the same field set, so `eval_table.py`'s formulas apply
unchanged. They land in a **flat** directory, not one directory per cell.

### 2.3 Build the training data (block splitting)

The training set is built from decode records: each block of a *correct* decoded
answer becomes one training state, corrupted with draft tokens and masks.

```bash
bash code/draftalign/build_union_index.sh   # calls build_train_set.py 4x, concatenates
```

Reads the datagen decodes, writes
`logs/draftalign/trainset_union_r2/index.jsonl` plus a `build_summary.json` per
pool. `code/draftalign/dataset.py` is the state sampler (how a state is
corrupted); `code/draftalign/losses.py` is CE + certainty forcing with the
`[MASK]` column deleted. `DATA.md` documents the pools and the provenance.

### 2.4 Train the judge adapter

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
(`train_forward.py`), so DDP's autograd hooks would never see the graph;
gradients are all-reduced manually.

### 2.5 Tables and checks

```bash
$PY code/draftalign/eval_table.py       # canonical -> results/reports/EVAL_TABLE.md
$PY code/draftalign/compare_table.py    # head-to-head, adds exp/layer-fwd and "+ drafter"
$PY code/research_protocol.py           # prints the frozen-set hashes and the protocol sha
$PY -m pytest code/tests/ -q            # CPU only, loads no model
```

### 2.6 Running a wave across GPUs

```bash
bash code/draftalign/gpu_queue.sh <queue-file>   # one command per free GPU, polls every 60 s
bash code/draftalign/monitor_status.sh           # read-only status snapshot
```

**Footgun:** `gpu_queue.sh` deletes the queue file when it drains. Write the next
queue *before* the current wave empties, or the cards idle. Round 3's `sched_r3.sh` (§2.7)
never deletes its queue files.

### 2.7 Round 3 — pools, decode, index, backup (2026-09-14)

Every round-3 shell script defaults `PY` to `/venv/r3/bin/python` and exports
`HF_HOME=/root/.cache/huggingface`. Stage 1 and Stage 2 are `DATA.md` §4's, unchanged; the
pools are `DATA.md` §7.

**Code and IF shards never run with cwd = the repository root:** `--score` executes generated
programs in the cwd (`AGENTS.md` §9 (p)), so `queue_r3_lines.sh`, `launch_r3_datagen.sh` and
`launch_r3_datagen_after.sh` run each shard in `/dev/shm/dlm/sandbox/<name>/shard<k>of<S>` with
absolute repo paths.

```bash
# 0. the frozen pools (CPU only; both already built -- commands as in each builder's docstring)
HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_opencode_pool.py
HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_if_pool.py --n 6000

# 1. Stage 1 -- blind-rule decode, stock 30B + stock 1.7B drafter
bash code/draftalign/launch_r3_datagen_gsm8k.sh                                   # gsm8k_pool:0:2400, 8 shards (running)
bash code/draftalign/launch_r3_datagen.sh <items> <name>                          # any slice, one sandboxed shard per GPU in $GPUS
nohup bash code/draftalign/launch_r3_datagen_after.sh <prev_dir> <items> <name> & # shard k starts when <prev_dir>'s shard k is .done

# 1b. what round 3 queues: stride shards for the two-level scheduler
bash code/draftalign/queue_r3_lines.sh opencode_pool:0:6000 opencode_pool_0_6000 32 >> logs/draftalign/queue_r3_hi.txt
bash code/draftalign/queue_r3_lines.sh gsm8k_pool:2400:4400 gsm8k_pool_2400_4400 32 >> logs/draftalign/queue_r3_lo.txt
nohup bash code/draftalign/sched_r3.sh logs/draftalign/queue_r3_hi.txt logs/draftalign/queue_r3_lo.txt logs/draftalign/sched_r3.log &
touch logs/draftalign/sched_r3.log.stop                                           # stops the scheduler, not its running jobs

# 2. Stage 2 -- one index per pool, only once every shard present has its .done
bash code/draftalign/build_r3_index.sh gsm8k              # -> logs/draftalign/trainset_r3/gsm8k/
bash code/draftalign/build_r3_index.sh opencode 6000      # N names the decoded slice, not the items finished
bash code/draftalign/build_r3_index.sh if <N>             # TBD (IF pool build in progress)

# 3. back up to the public HF folder every 30 min
nohup /venv/r3/bin/python code/draftalign/backup_r3_hf.py --loop 30 > logs/draftalign/hf_backup_r3.log 2>&1 &

# the IF scorer's CPU tests
/venv/r3/bin/python -m pytest code/tests/test_if_scorer.py -q
```

The live `queue_r3_lo.txt` lines were generated before the sandbox change and run from the
repository root; they decode GSM8K, whose vendored scorer (`code/vendor/oc_datasets/{gsm8k,math}.py`)
contains no `exec` or `eval`.

---

## 3. Status — what is done, what is open

### Complete

| item | evidence |
|---|---|
| All six settings on GSM8K-500, MATH-500, MBPP, HumanEval, GPQA-Diamond, AIME and calibvalid | `RESULTS.md` §1–§4 |
| Judge adapter trained and frozen at `step003057` (26,740,736 params, sha256[:16] `734af65681ab6bb8`) | `RESULTS.md` §7, `EXCLUDED.md` §4 |
| Training index built (`trainset_union_r2`), mix documented by block and by item | `RESULTS.md` §5, `DATA.md` |
| Stock integrity twin passes — `--stock` reproduces the reference loop token for token | `ref_match` in every stock record; `code/tests/test_da_decode.py` |
| Drafter fine-tune evaluated and **rejected at its gates** | `results/metrics/draftalign/` |
| Checkpoint inventory re-verified (2026-09-14): all 14 judge adapter sha256 match Hugging Face; benchmark record counts match `RESULTS.md` | `docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md` §4 |
| Round-3 code pool frozen — 12,000 `nvidia/OpenCodeInstruct` items, sha256 `d636ff2e…` — and wired: `opencode*` → `mbpp` scorer in both runners, `GEN_LEN` 512, `POOLS` | `code/configs/opencode_pool.json`, `DATA.md` §7.2, the `git diff` of `run_da_decode.py`, `run_research_baselines.py`, `research_protocol.py` |
| OpenCodeInstruct reference-output scorer gate: 193 / 200 (the 7 misses are row defects) | `DATA.md` §7.2 |
| IF checker vendored; strict all-constraints scorer; CPU tests | `code/vendor/ifeval/PROVENANCE.md`, `code/draftalign/if_scorer.py`, `code/tests/test_if_scorer.py` |
| Round-3 environment: `/venv/r3` (team freeze, no system site-packages) | *Known environment hazard* below |

### Running — round 3 (state on 2026-09-14)

| item | state | where |
|---|---|---|
| GSM8K decode `gsm8k_pool:0:2400`, 8 shards | running — so far ~92–94% correct, 6.30 blocks/item, 16.1 s/item on one RTX PRO 6000 Blackwell | `logs/draftalign/datagen/gsm8k_pool_0_2400_dv-s8_blind_q50_r3/` |
| OpenCodeInstruct decode `opencode_pool:0:6000`, 32 stride shards | queued, HI | `logs/draftalign/queue_r3_hi.txt` |
| arm-C GSM8K filler `gsm8k_pool:2400:4400`, 32 shards | queued, LO | `logs/draftalign/queue_r3_lo.txt` |
| `sched_r3.sh` over both queues | running | `logs/draftalign/sched_r3.log` |
| verifiable-IF pool | built — 1,803 items as of sha256 `2d3761e9…`; wiring and decode TBD (IF pool build in progress) | `code/configs/if_pool.json`, `DATA.md` §7.3 |
| HF backup, `backup_r3_hf.py --loop 30` | running; first sync 2026-09-14T14:44:54Z | `logs/draftalign/hf_backup_r3.log`, `HUGGINGFACE.md` §9 |
| Stage 2 (`build_r3_index.sh`) | not started — each pool waits for its decode | `logs/draftalign/trainset_r3/` |
| training — 6 GPUs train + 2 evaluate, checkpoint and evaluate every 100 steps, W&B | not started | recipe *Amendment* A4 |

### Open

| # | item | why it matters |
|---|---|---|
| 1 | **24 unrun eval shards**: `judge + DES-32` and `judge + DES-48` on **AIME** and **HumanEval**, 12 shards each | those two datasets have no pooled judge cell, so §4's coverage is 4 of 6 sets |
| 2 | **Several MATH-500 pool cells are partial** — `n` is stated per row in `RESULTS.md` | a partial cell must never be averaged with a complete one |
| 3 | `EVAL_TABLE.md` covers only calibvalid / gsm8k500 / math500 / mbpp | AIME, GPQA and HumanEval appear in **no** generated report; `RESULTS.md` §1–§4 is their only written record |
| 4 | `results/reports/EVAL_TABLE.md` was slightly stale at snapshot time for the partial math500 cells | regenerate with `eval_table.py` before quoting it |
| 5 | Forward data plan (`docs/ICLR2027_DATA_PLAN.md`) not executed | round 3 draws per the recipe's *Amendments 2026-09-14*, not per that plan's phases |
| 6 | **Round-3 datagen running; Stage 2 and training not started** | `docs/TRAINING_RECIPE_2026-09-09.md`, *Amendments 2026-09-14* — the preprocessing gate (`build_r3_index.sh` per pool) comes before training; see "Running" above |
| 7 | **Checkpoint choice is operating-point dependent** | `docs/TRAINING_RECIPE_2026-09-09.md` §1.1 — at `c90_o99` step 2200 beats 3057 on both axes; at `c95` step 3057 beats 2200 by 4.0 points at DES-48 for the same compute. No `t2a` checkpoint has been evaluated at `c95` |
| 8 | **IF slice not wired or decoded** — `POOLS`, `GEN_LEN`, the `if*` line in both runners' family map, the scorer branch | `docs/IF_POOL_WIRING_PATCH.md` — TBD (IF pool build in progress). Until then `run_da_decode.py` fails on an `if_pool` split at `GEN_LEN` |
| 9 | **IF yield vs the 9,600-block target** | `code/configs/if_pool.json` (sha256 `2d3761e9…`) holds 1,803 items; 9,600 blocks needs ≥ 5.32 kept blocks per item even at a 100% pass rate. Pass rate and blocks/item are unmeasured |
| 10 | **Code yield unmeasured** | `opencode_pool:0:6000` is trimmed to 11,200 blocks once measured; stride shards make any finished subset a uniform sample (`AGENTS.md` §9 (n)) |
| 11 | **Batch for 6 training ranks** | the recipe's 8 × 4 = 32 states/step is unreachable on 6 GPUs; 6 × 5 = 30 or 6 × 6 = 36 — TBD in the training phase (*Amendment* A4) |

### Known environment hazard

**Current — the 2026-09-14 NGC container. Use `/venv/r3`.**

| | |
|---|---|
| symptom | `ModuleNotFoundError: No module named 'flash_attn.layers'` at model load. The first round-3 launch died this way on all 8 shards; logs in `logs/draftalign/datagen/_failed/gsm8k_pool_0_2400_flashattn_importerror/` |
| cause | the system Python carries a real `flash_attn` 2.7.4.post1, and `/venv/main` on this container is a `--system-site-packages` venv (transformers 4.52.4, NGC torch `2.14.0a0+4fdf77b940.nv26.08`) that sees it. transformers then runs `from flash_attn.layers.rotary import apply_rotary_emb` (`modeling_flash_attention_utils.py` line 38, imported by the remote `modeling_sdar_moe.py` line 36); `code/compat` is first on `sys.path`, so `flash_attn` is the shim, which has no `layers` |
| fix | `/venv/r3`: `python3 -m venv /venv/r3`, **no system site-packages**, replicating the team's exact pip freeze — torch 2.11.0+cu128, transformers 4.52.4, tokenizers 0.21.4, huggingface_hub 0.36.2, datasets 5.0.1, accelerate 1.14.0, wandb 0.29.0. Every round-3 launcher defaults to it. The freeze file is `/dev/shm/dlm/env/team_requirements_2026-09-09.txt`, backed up to HF under `strategy/env/` together with `venv_r3_freeze.txt` |
| IF checker deps | `nltk` 3.9.2, `langdetect` 1.0.9, `immutabledict` 1.2.0, `absl-py` 2.5.0 in both venvs; NLTK `punkt` / `punkt_tab` in `/root/nltk_data` (`code/vendor/ifeval/PROVENANCE.md`) |
| datasets | read with `HF_HOME=/root/.cache/huggingface`, where the dataset caches are; the launchers export it, and with `HF_DATASETS_OFFLINE=1` nothing falls back to a download |
| volatility | `hf_cache` and `logs` are symlinks into `/dev/shm/dlm/` — RAM-backed, **wiped on restart** — as are the OpenCodeInstruct snapshot, the env freeze and the sandboxes. Round-3 data is backed up to HF every 30 min (`HUGGINGFACE.md` §9); anything newer than the last sync is lost |

**Earlier — the 2026-09-09 environment, kept for the record.**

`transformers` was upgraded to **5.16.0** in `/venv/main` on 2026-09-09. In that
version `DynamicCache` is no longer subscriptable, which breaks the decode path:
`code/tests/test_da_decode.py` and `test_draftalign.py` fail with
`TypeError: 'DynamicCache' object is not subscriptable`, as do
`sdar_harness/rowinv.py` lines 119 and 217. **Pin `transformers` back before
running anything on GPU.** A healthy CPU suite is `3 failed, 281 passed,
4 skipped`; those 3 are pre-existing and unrelated (a `SimpleNamespace` fixture
missing `drafter` in `run_da_decode.cell_name`, and a JSON decode in
`test_train_drafter.py`).

On the 2026-09-14 container `/venv/main` has transformers 4.52.4, so this particular breakage
does not apply there; the hazard above does.
