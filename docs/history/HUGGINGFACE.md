# HUGGINGFACE

Everything this project pushed to or pulled from Hugging Face: the two model
repos under test, every trained checkpoint, the training indices, the loss
functions, and the raw decode records.

**All of it is public and ungated.** This GitHub repository is private; the
Hugging Face side is not, so nothing below needs a token, an approval, or an
account. Every link is a plain URL you can paste into a browser or hand to
`hf download`.

Verified against the live Hugging Face API on **2026-09-09**. Sizes are exact
bytes; hashes are the first 16 hex of the file's sha256 as Hugging Face stores it.

**Re-verified 2026-09-14** (`docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md` §4): all 14 judge
`adapter.safetensors` sha256 match Hugging Face, and the benchmark record counts match
`RESULTS.md`. Round-3 training data is backed up into the same public dataset — §9.

---

## 1. The repos, at a glance

| what | repo | type | public? |
|---|---|---|:--:|
| the 30B MoE under test | https://huggingface.co/JetLM/SDAR-30B-A3B-Chat-b32 | model | yes |
| the 1.7B dense drafter | https://huggingface.co/JetLM/SDAR-1.7B-Chat-b32 | model | yes |
| **everything we produced** | https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts | dataset | **yes** |
| hourly backup of a subset | https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-logs | dataset | no |
| the friend's Stage-0 routing traces | https://huggingface.co/datasets/Anupam-Rawat-IITB/stage0-routing-logs | dataset | yes |
| round-3 training data backup (a folder of the public artifacts dataset) | https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-data-subset-training-diffusion-moe | dataset folder | **yes** — §9 |

**Use the public one.** `dlm-exp1-drafter-voting-research-artifacts` is a strict
superset of the private backup: 59,538 files / 127.85 GB versus 18,427 files, and
the private one carries no `code/`, no `docs/`, and only 15,469 of the 55,950
`logs/draftalign/` files. Nothing in this document lives only behind the private
repo. The private repo exists because a cron job (`code/sync_voting.sh` in the
research tree) writes to both; it is a backup, not a second source of truth.

Repo state this document was written against:

```
Arushhh/dlm-exp1-drafter-voting-research-artifacts
  commit 2fcc0418b7c8915f61919cd413a9c118054fccc7
  59,538 files, 127,850,076,008 bytes (127.85 GB), lastModified 2026-09-09T11:35:43Z
```

That is the 2026-09-09 state. From 2026-09-14 the round-3 backup (§9) commits to this repo
every 30 min, so the live file count and commit are newer.

---

## 2. The two models (upstream — not ours, pinned by revision)

Both are pinned in `code/research_protocol.py::MODELS`, loaded with
`local_files_only=True`, and the revision is asserted at load. Pull them at
exactly these revisions or the numbers in RESULTS.md are not reproducible.

**MoE under test — `b32`, 64 files:**

```
https://huggingface.co/JetLM/SDAR-30B-A3B-Chat-b32
revision c351bbc37d240aa6871f167e8f92d694281b0c22
```

**Dense drafter — `d32`, 16 files, same tokenizer and same [MASK] id (151669):**

```
https://huggingface.co/JetLM/SDAR-1.7B-Chat-b32
revision 2c83c05d84be90a01c72443b2123ebf0376f6fd4
```

Pull both:

```bash
hf download JetLM/SDAR-30B-A3B-Chat-b32 \
    --revision c351bbc37d240aa6871f167e8f92d694281b0c22 \
    --local-dir hf_cache/SDAR-30B-A3B-Chat-b32

hf download JetLM/SDAR-1.7B-Chat-b32 \
    --revision 2c83c05d84be90a01c72443b2123ebf0376f6fd4 \
    --local-dir hf_cache/SDAR-1.7B-Chat-b32
```

Together they are ~61 GB. `trust_remote_code=True` is required — the repos ship
their own `modeling_sdar*.py`, and `code/compat/flash_attn/` must be on the path
or the import fails (see ARCHITECTURE.md). On an image that also has a real `flash_attn`
installed, the shim makes the import fail instead — use an isolated venv (WORKTREE.md,
*Known environment hazard*).

---

## 3. Trained checkpoints — the judge adapters

All judge checkpoints live inside the **public** dataset, under
`logs/draftalign/train/`. Every one is a manual LoRA (no `peft`): rank 32,
alpha 64, dropout 0, on `q_proj, k_proj, v_proj, o_proj` of all 48 layers.
The router gate, the 128 experts, the embeddings and the output head are frozen.

### 3.1 t2b_union_flag — the published judge

This is the adapter behind settings 3, 4 and 5 in README.md and every "judge"
row in RESULTS.md. Trained on the union index (GSM8K + MATH + MBPP), 3,057
steps, `world_size=7`, `draft_flag=true`, `train_gates=false`, 385 tensors,
26,740,736 fp32 parameters.

| step | file | bytes | sha256[:16] | cell tag |
|---|---|---:|---|---|
| 2200 | logs/draftalign/train/t2b_union_flag/step002200/adapter.safetensors | 107,005,856 | a7be11a8f5843cd1 | `_ada7be11a8` |
| 2400 | logs/draftalign/train/t2b_union_flag/step002400/adapter.safetensors | 107,005,856 | 022382b2a5c63f32 | `_ad022382b2` |
| 2600 | logs/draftalign/train/t2b_union_flag/step002600/adapter.safetensors | 107,005,856 | f51dafdc064761dd | `_adf51dafdc` |
| 2800 | logs/draftalign/train/t2b_union_flag/step002800/adapter.safetensors | 107,005,856 | 38a9ef557a7d4fb8 | `_ad38a9ef55` |
| 3000 | logs/draftalign/train/t2b_union_flag/step003000/adapter.safetensors | 107,005,856 | 3e97af0249eca7f4 | `_ad3e97af02` |
| **3057** | **logs/draftalign/train/t2b_union_flag/step003057/adapter.safetensors** | **107,005,856** | **734af65681ab6bb8** | **`_ad734af656`** |

Each step directory also holds `adapter_config.json` (378 B) and
`trainer_state.pt` (214,260,746 B — optimizer state, only needed to resume).
The run directory also holds `config.json` (the full recipe, `config_hash`
`f8ee21d32a2e7980`), `train.rank{0..6}.jsonl` (the per-step loss trace, ~100 KB
each) and `wandb/` (four run directories).

`config.json`'s `resume` field is **self-referential**: it names
`logs/draftalign/train/t2b_union_flag`, the run's own output directory, not
`t2a_gsm8k_flag/step001057`. The two-stage curriculum (T2a to step 1057, then the union mix)
is documented in the round-3 recipe §4.1 and `ARCHITECTURE.md` §6 but is not evidenced by that
field (`docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md` §4.1).

Pull just the published adapter — 107 MB, not the whole 127 GB:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train/t2b_union_flag/step003057/adapter*" \
    --local-dir .
```

That gives you `logs/draftalign/train/t2b_union_flag/step003057/`, which is
exactly the `--adapter` argument in README.md's reproduce block:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/run_da_decode.py \
    --model b32 --drafter d32 --items gsm8k500 \
    --adapter logs/draftalign/train/t2b_union_flag/step003057 \
    --judge full --draft s8 --shard 0 --nshards 4 --gpu 0 --score
```

Loading it by hand is three lines — `code/draftalign/adapter.py::load_adapter`
reads `adapter_config.json`, attaches the LoRA with the recorded shape, and
asserts the metadata matches before it copies any weights:

```python
from draftalign.adapter import load_adapter, merge_lora
info = load_adapter(moe, "logs/draftalign/train/t2b_union_flag/step003057")
merge_lora(moe)          # inference model is a stock module tree again
```

### 3.2 t2a_gsm8k_flag — GSM8K-only judge (superseded)

Same recipe, 1,057 steps, GSM8K index only. Kept because the threshold sweep and
the progression plots in `results/metrics/draftalign/` were measured on it.

| step | bytes | sha256[:16] | cell tag |
|---|---:|---|---|
| 200 | 107,005,856 | 6fd3c2e933b67c6f | `_ad6fd3c2e9` |
| 400 | 107,005,856 | 7d6e7e823792d329 | `_ad7d6e7e82` |
| 600 | 107,005,856 | bb1f79ea97f033ed | `_adbb1f79ea` |
| 800 | 107,005,856 | 329be7a372295e6c | `_ad329be7a3` |
| 1000 | 107,005,856 | 712a8ebcb2eaba49 | `_ad712a8ebc` |
| 1057 | 107,005,856 | 191ea8d69b509628 | `_ad191ea8d6` |

Under `logs/draftalign/train/t2a_gsm8k_flag/step<NNNNNN>/`; `trainer_state.pt` is
214,255,306 B each. Run `config_hash` `4fbb68e5d5f3fdaf`.

### 3.3 t1_gsm8k_r1 — the first judge, different shape

`train_gates=true` and **no** draft flag, so it has 432 tensors, not 385, and the
adapter file is larger. It is not loadable by a `draft_flag` cell and vice versa —
`load_adapter` asserts on the mismatch.

| step | bytes | sha256[:16] |
|---|---:|---|
| 400 | 157,334,472 | 5b25cba2452a6bae |
| 600 | 157,334,472 | e97175e953ef153c |

Under `logs/draftalign/train/t1_gsm8k_r1/step<NNNNNN>/`; `trainer_state.pt` is
314,942,923 B each. Run `config_hash` `de58eb0854715659`.

### 3.4 Reading the `_ad…` tag in a record directory name

Every decode cell under `logs/draftalign/da_decode/` is named with an `_ad<hash8>`
suffix, so a cell can never silently mix checkpoints. **`hash8` is the first 8
hex of the sha256 of `adapter.safetensors`** (`run_da_decode.py::adapter_weights_hash`),
which is what makes the tables above a complete lookup key.

Two legacy tags predate that scheme and are named by the recipe's `config_hash`
instead, which is shared by every checkpoint of one recipe and therefore
**ambiguous**:

* `_adb3106a4d` — `b3106a4df420f375` is the config hash of *both* the T2a and the
  T2b recipe. A cell with this tag does not, by its name alone, say which
  checkpoint produced it.
* `_ad5d9add10` — `5d9add1007ae47b3`, the T1 recipe.

Any cell reported in RESULTS.md uses the unambiguous weights-hash form.

### 3.5 The judge ablation runs — launched, no checkpoints yet

`logs/draftalign/train_ablation/` holds `A_math000`, `C_math310`, `E_math1000`
(config + wandb, `train.rank0.jsonl` is **0 bytes**) and `B_math125.log`,
`D_math625.log`. These are the MATH-share ablation runs started 2026-09-09
09:09–09:10 UTC. **No `adapter.safetensors` exists for any of them** — they had
not reached a save step when this snapshot was taken. Their frozen training
indices *are* published (§5).

---

## 4. Trained checkpoints — the drafter fine-tune (rejected)

`logs/draftalign/train_drafter/` holds two **full fine-tunes of the 1.7B dense
drafter**, saved in fp32 as sharded `safetensors` plus the complete tokenizer and
remote-code files, so each step directory is a loadable model on its own.

**Both were rejected at their gates and no published result uses them.** They are
here because their gate metrics (`results/metrics/draftalign/drafter_accuracy_*.json`,
`drafter_agreement_*.json`) are reported by `eval_table.py`, and this is the
provenance of those numbers.

| run | steps published | per-step weights | `trainer_state.pt` |
|---|---|---:|---:|
| `d1` | step001000, step001080 | 4,969,539,560 + 3,157,455,608 B | 16,254,193,795 B |
| `d2_cleanprefix` | step000200, step000400, step000600 | 4,969,539,560 + 3,157,455,608 B | 16,254,193,859 B |

sha256[:16] of `model-00001-of-00002.safetensors` / `model-00002-of-00002.safetensors`:

```
d1/step001000              aa5fe4e886bee276 / 9159bfb10519d517
d1/step001080              8e95f75cb2803e70 / 656de6fbb4f30e63
d2_cleanprefix/step000200  49ea67f2792991af / 1de854fd67912278
d2_cleanprefix/step000400  5e845c5424727244 / 2e211dae7b298808
d2_cleanprefix/step000600  6e3e6f01fc5688e4 / a84e7ecbadb7bc53
```

The `step000600` pair was missing from this list until 2026-09-14. It comes from the live
Hugging Face inventory at revision `2fcc0418b7c8915f61919cd413a9c118054fccc7`
(`docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md` §2.4, §4.1), with byte sizes consistent with
the other two `d2_cleanprefix` steps. None of the ten drafter shard hashes was re-computed
locally — the weights are not in the local mirror.

Recipes: `d1` — index `drafterset_union/index.jsonl`, lr 1e-5, 1,080 steps,
`only_correct=true`, `config_hash` `7815f0cdd2410890`. `d2_cleanprefix` — index
`drafterset_union_v2/index.jsonl`, `conditioning=clean_prefix`, lr 5e-6, early
stop patience 2 (stopped at 600 of a planned 1,602), `config_hash`
`5dee343e8f7141a1`.

Weights only, one step (~8.1 GB, skips the 16 GB optimizer state):

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train_drafter/d1/step001080/*" \
    --exclude "*trainer_state.pt" --local-dir .
```

---

## 5. Training data — the indices, not a HF `datasets` split

The judge's training set is **the model's own correct answers, re-decoded under
the blind rule, then corrupted on the fly**. It is not a packaged HF dataset; it
is a set of `index.jsonl` files plus the per-item NPZ traces they point at, all
under the public dataset. DATA.md §4 documents how they were built.

| file (in the public dataset) | bytes | what |
|---|---:|---|
| logs/draftalign/trainset_union_r2/index.jsonl | 3,971,882 | **the judge's training index** — 8,638 items, 65,521 blocks |
| logs/draftalign/trainset_union_r2/gsm8k_a/index.jsonl | 1,182,768 | GSM8K 0–3000 |
| logs/draftalign/trainset_union_r2/gsm8k_b/index.jsonl | 1,686,239 | GSM8K 3000–7273 |
| logs/draftalign/trainset_union_r2/math/index.jsonl | 951,055 | MATH, every third |
| logs/draftalign/trainset_union_r2/mbpp/index.jsonl | 24,124 | MBPP, all 120 |
| logs/draftalign/trainset_union_r2/index.jsonl.broken-flatpaths | 3,844,186 | superseded; kept for provenance, **do not train on it** |
| logs/draftalign/drafterset_union/index.jsonl | 2,969,695 | drafter fine-tune `d1` |
| logs/draftalign/drafterset_union_v2/index.jsonl | 3,285,294 | drafter fine-tune `d2_cleanprefix` |

Each `<source>/build_summary.json` (~820 B) carries the kept/dropped counts that
DATA.md §4 tabulates; `<source>/items/*.npz` are the per-block traces the index
points at.

The five MATH-share ablation indices, with their exact block counts from
`logs/draftalign/trainset_union_r2/ablation_manifest.json`:

| index | bytes | MATH blocks | GSM8K blocks | total | items |
|---|---:|---:|---:|---:|---:|
| index_A_math000.jsonl | 581,878 | 0 | 8,005 | 8,005 | 1,346 |
| index_B_math125.jsonl | 549,359 | 1,011 | 7,000 | 8,011 | 1,248 |
| index_C_math310.jsonl | 502,071 | 2,487 | 5,520 | 8,007 | 1,106 |
| index_D_math625.jsonl | 413,415 | 5,006 | 3,000 | 8,006 | 839 |
| index_E_math1000.jsonl | 318,293 | 8,001 | 0 | 8,001 | 553 |

The raw decodes those indices were built from are in
`logs/draftalign/datagen/`, four directories:
`gsm8k_pool_0_3000_dv-s8_blind_q50`, `gsm8k_pool_3000_7273_dv-s8_blind_q50`,
`math_pool_0_7500_3_dv-s8_blind_q50`, `mbpp_pool_dv-s8_blind_q50`.

All the indices, and nothing else (~15 MB):

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/trainset_union_r2/*.jsonl" \
    --include "logs/draftalign/trainset_union_r2/*/index.jsonl" \
    --include "logs/draftalign/trainset_union_r2/*/build_summary.json" \
    --include "logs/draftalign/trainset_union_r2/ablation_manifest.json" \
    --local-dir .
```

**The six evaluation datasets** are ordinary public HF datasets and are *not*
mirrored here — their exact ids, revisions, configs and splits are in DATA.md §1
(`openai/gsm8k`, `HuggingFaceH4/MATH-500`, `google-research-datasets/mbpp`,
`AI-MO/aimo-validation-aime`, `fingertap/GPQA-Diamond`, `openai_humaneval`).
The training pools come from `openai/gsm8k` and `EleutherAI/hendrycks_math`
(DATA.md §3). Round 3's pools add `nvidia/OpenCodeInstruct`, `allenai/IF_multi_constraints_upto5`
and `allenai/RLVR-IFeval` (DATA.md §7); the selected rows are backed up under §9's `unprocessed/`.

---

## 6. Loss functions

The loss code is **in this repository**, and mirrored inside the public dataset
at both `code/draftalign/losses.py` and `wide-when-sure/code/draftalign/losses.py`.

**Judge loss** — `code/draftalign/losses.py::masked_ce_and_certainty`:

```
loss = mean_i CE(logits_i, T_i)  +  beta * mean_{i : argmax_i == T_i} H(softmax(logits_i / temp))
```

The second term is dParallel's certainty forcing (arXiv 2509.26488), applied
**only on rows the model already gets right**, so it sharpens a correct
distribution instead of pushing a wrong one. The `[MASK]` column (token id
151669) is **deleted** — not set to `-inf`, which would make the term
`0 * log(0)` = NaN — before the entropy softmax, because otherwise the model can
drive the entropy down by piling probability on a token that is never a legal
output. Logits are fp32 and cover only the ~32 active block rows
(`active_logits` applies `lm_head` to those rows, so a `[S, 151936]` tensor is
never built).

Hyperparameters, identical in all three judge runs (t1, t2a, t2b):
`beta = 2.0`, `temp = 0.5`, `label_smoothing = 0.0`, `mask_id = 151669`.

`eval_row_stats` in the same file computes the held-out diagnostics: argmax
accuracy and the precision of rows with `top1 > tau` (`tau = 0.95`).

**Drafter fine-tune loss** — `code/draftalign/train_drafter.py`: plain
cross-entropy at the **masked positions only**, against the MoE's own tokens, with
the prompt clean and never a loss row, `mask_weighting="mean"`. It reuses
`losses.active_logits` and has no certainty term.

---

## 7. The decode records behind every number

`logs/draftalign/da_decode/` holds one directory per cell — 75 of them — named
`<split>_B32_t95_<arm>[_<pool>][_ad<hash8>]`. Every table in RESULTS.md names the
record path it was computed from, so a cell can be recomputed without re-decoding.
The six published settings on `calibvalid` and the six eval sets are all in there,
plus the stock/blind/judge/DES variants and the threshold sweep.

`logs/draftalign/ab_probe/`, `logs/draftalign/probes/` (four routing probes) and
`logs/research_v1/` (2,257 files from the earlier DES-Vote phase) are the
supporting probes.

A curated copy of **this repository** is also inside the dataset, under
`wide-when-sure/`, so the mirror is self-describing.

Pull the whole thing (127.85 GB — you almost certainly want `--include` instead):

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts \
    --repo-type dataset --local-dir .
```

One cell, e.g. the published judge on GSM8K-500:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_ad734af656/*" \
    --local-dir .
```

Everything except the NPZ traces and the optimizer states, which is most of the
bulk:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --exclude "*.npz" "*trainer_state.pt" "*model-0000*.safetensors" \
    --local-dir .
```

Download into the repository root and the paths line up with what the code
expects — `logs/` is in `.gitignore`, so a pulled mirror never enters git.

---

## 8. The friend's Stage-0 traces

Separate project, separate account, public, 3,473 files:

```
https://huggingface.co/datasets/Anupam-Rawat-IITB/stage0-routing-logs
```

Per-block routing `.npz` traces, `results/`, `configs/`, `export/`, and a `code/`
snapshot that includes `code/sdar_harness/coreset_des.py` — a dynamic DES-Vote
implementation that was never run and was removed from git. Nothing in
RESULTS.md is computed from it.

---

## 9. Round-3 data backup — `Nemotron-data-subset-training-diffusion-moe/`

Everything round-3 preprocessing produces is pushed into a folder of the **public** artifacts
dataset:

```
https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-data-subset-training-diffusion-moe
```

`code/draftalign/backup_r3_hf.py --loop 30` stages the tree under
`/dev/shm/dlm/hf_backup_stage/`, scans it for secrets and uploads it as one commit
(`r3 data backup <UTC time>`) every 30 minutes. The working copies are on `/dev/shm` and are
wiped on restart; this folder is the durable copy. First sync logged 2026-09-14T14:44:54Z.

```bash
nohup /venv/r3/bin/python code/draftalign/backup_r3_hf.py --loop 30 > logs/draftalign/hf_backup_r3.log 2>&1 &
/venv/r3/bin/python code/draftalign/backup_r3_hf.py          # one sync
```

| path in the folder | contents |
|---|---|
| `README.md` | regenerated every sync: target mix, pipeline, layout, restore commands, and **live status tables** — per datagen dir: records, correct, shards done; per trainset pool: kept items, usable blocks, dropped |
| `unprocessed/SOURCES.md` | upstream repo ids, revisions, licences and the selection rule of each slice |
| `unprocessed/gsm8k/<slice>.jsonl` | `idx, split, prompt, gold` of every GSM8K pool slice decoded |
| `unprocessed/opencode/` | `opencode_pool.json` + `opencode_raw_rows_<sha12>.parquet` — the selected upstream rows with all their original columns |
| `unprocessed/if/` | `if_pool.json` + `.sha256` |
| `processed/datagen/<cell dir>/` | per-item records (`*.jsonl`), `.done` markers, shard logs, and `traces.tar` (the per-block NPZ traces) |
| `processed/datagen/_failed/` | logs only, from the aborted first launch |
| `processed/trainset_r3/<pool>/` | `index.jsonl`, `build_summary.json` (+ `manifest.json` when present), and `items.tar` (the training-state NPZs) |
| `strategy/docs/` | the round-3 recipe, the data plan, `DATA.md`, `HUGGINGFACE.md`, the checkpoint doc, and the IF recipe and wiring patch (each copied when present) |
| `strategy/code/` | the pipeline code as it ran, the pool configs, `code/vendor/ifeval/` |
| `strategy/code_changes.diff`, `strategy/git_state.txt` | `git diff` of every edit to tracked files; base commit and `git status` |
| `strategy/env/` | `team_requirements_2026-09-09.txt` (the team freeze) and `venv_r3_freeze.txt` |

Behaviour worth knowing:

* Files modified in the last 10 s wait for the next sync. `traces.tar` and `items.tar` are
  rebuilt only when their directory's content signature changes; the NPZs inside are already
  compressed, so tar only keeps the repo's file count down.
* Only an explicit allowlist is staged, and the stage is scanned for the `.env` values of
  `HF_TOKEN` / `WANDB_API_KEY` and for `hf_…` token shapes before upload. A hit aborts the
  upload **and ends `--loop`**; a transient Hub error is logged and retried next cycle.
* Full upstream datasets are referenced by revision, not re-uploaded.
* Planned for the training phase (recipe *Amendment* A4): every 100-step checkpoint is backed up
  to this public repo too. Its path in the repo is not fixed yet.

Restore:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "Nemotron-data-subset-training-diffusion-moe/*" --local-dir .
tar -xf Nemotron-data-subset-training-diffusion-moe/processed/datagen/<cell>/traces.tar -C <cell>/     # -> <cell>/traces/<split>/<idx>.npz
tar -xf Nemotron-data-subset-training-diffusion-moe/processed/trainset_r3/<pool>/items.tar -C <pool>/  # -> <pool>/items/*.npz
```

## 10. Phase 5 backup — `Arushhh/phase5-loop-guard`

Public HF dataset: **https://huggingface.co/datasets/Arushhh/phase5-loop-guard**

Live incremental backup of the Phase 5 judge round (Texas box, instance 57349095). Pushed **every 15 min** by `code/judge_round/backup_loop_p5.sh` → `backup_p5.py` (an explicit allowlist; before every upload each file is scanned for the VALUES of HF_TOKEN / WANDB_API_KEY / GITHUB_TOKEN and for `hf_[A-Za-z0-9]{30,}` — any hit refuses the upload).

What is in it, at the repo's own relative paths:
- `logs/c1b/gen/waveC/` — wave C generation records (vanilla 30B answers)
- `logs/c1b/draft/waveC/` — first drafts (Phase 4 merged drafter)
- `logs/c1b/trainset/waveC/` — packed training items as a single **`items.tar`** (the ~150k small item files are tarred to stay under the Hub's API rate limit) + `index.jsonl`; `logs/c1b/trainset/fresh/` — the combined training `index.jsonl` + `build_summary.json`
- `logs/phase5/judge_sglang/record/` and `logs/phase5/judge_sglang/record_run1/` — the recorded judge per-pass states (`gpu<k>/states.NNNNN.npz`)
- `logs/phase5/judge/rec_main/` — the retrained judge checkpoints (flag-gated LoRA adapter)
- `logs/phase5/` — logs (`events.tsv`, STATUS history, `corpus_verify.md`, `cp6_prep/` selection & exclusions)
- `results/` — result tables and metrics

NOT uploaded: optimizer states (`*.pt`); the transient `judge_side/`, `judge_trace/`, `drafter_side/`, `drafter_trace/` dirs (created and deleted mid-run, so they can vanish during an upload); `.env` / any secret.

GitHub mirror: branch **`phase5-box-s1`** (code, `STATUS.md`, `results/`), branch **`phase5-logs`** (small text logs only, ≤5 MB).

Restore:
```
hf download Arushhh/phase5-loop-guard --repo-type dataset --local-dir .
tar -xf logs/c1b/trainset/waveC/items.tar -C logs/c1b/trainset/waveC/   # -> items/*  (packed training items)
```

**Full reproduce / resume-from-any-checkpoint:** see **docs/PHASE5_RESUME.md** — clone `phase5-box-s1` at the pinned commit, `hf download` the exact paths, verify against `logs/phase5/MANIFEST.sha256`, env from `logs/phase5/env/`, then resume training (`train_c1b_rec --continue-from` from `logs/phase5/judge/start_128m` or `rec_main/tok_000136000000…160M`) or run the rd1 eval. The new checkpoints include `trainer_state.pt` (optimizer) so a resume is exact; `start_128m/` is the 128M starting point copied into this dataset.

## Round 3 (2026-09-15)

All round-3 model weights, eval records, logs and analysis: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train — index in `docs/MODELS_AND_RESULTS_2026-09-15.md`.
