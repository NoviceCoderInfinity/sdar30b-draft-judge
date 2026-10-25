# CHECKPOINTS AND BASELINES

> **2026-09-15:** the published judge `t2b_union_flag/step003057` (sha `734af656…`) is now also on the public Anupam Rawat HF dataset: https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/round2_t2b_union_flag/step003057 and its eval records under `Nemotron-split-moe-train/reference_evals/`. Other historical checkpoints below remain on the team (Arushhh) repo.


Every checkpoint and baseline record that exists in this project as of round 3's
start, in one place: the two upstream base models round 3 actually trains from,
every team-trained checkpoint the project has produced to date, and the
existing benchmark records that define the "already measured" space round 3 is
extending.

Sources for this document: `HUGGINGFACE.md` §2–§4, `RESULTS.md` (full skim for
checkpoint provenance), `docs/TRAINING_RECIPE_2026-09-09.md` §1.1 and §4.1, the
inventory pulled from the live Hugging Face API at revision
`2fcc0418b7c8915f61919cd413a9c118054fccc7`, and a local mirror of
adapters/configs/train logs at `/dev/shm/dlm/artifacts_mirror/logs/draftalign/`
and `logs/research_v1/baselines_b32/`, cross-checked against that inventory by
recomputing sha256 locally. Verification results are in §5.

---

## 0. Read this first

> **Round 3 trains fresh from the stock base models.** None of the
> team-trained checkpoints in §3 are used as init or resumed from — not
> `t2b_union_flag/step003057`, not any other step. They are in this document
> as **reference and provenance only**: the dose-response curve, the ablation
> record, and the reason two drafter fine-tunes were rejected. See
> `docs/TRAINING_RECIPE_2026-09-09.md` §4.1 for why round 3 starts from stock.
>
> **Block size is 32 everywhere** — both base models, every team-trained
> checkpoint, and every benchmark record in this document, unless a table row
> is explicitly marked otherwise (one legacy baseline family in §4.2 is block
> 4, called out where it appears).

---

## 1. Base models (upstream — not ours, pinned by revision)

Both are pinned in `code/research_protocol.py::MODELS`, loaded with
`local_files_only=True`, revision asserted at load. This is what round 3
trains from.

| model | repo | revision | shape | notes |
|---|---|---|---|---|
| MoE under test | `JetLM/SDAR-30B-A3B-Chat-b32` | `c351bbc37d240aa6871f167e8f92d694281b0c22` | 30B MoE, ~3B active, 48 layers, 128 experts, top-8 routing | block 32 |
| Dense drafter | `JetLM/SDAR-1.7B-Chat-b32` | `2c83c05d84be90a01c72443b2123ebf0376f6fd4` | dense, 1.7B | same tokenizer as the MoE, same `[MASK]` id **151669**, block 32 |

```bash
hf download JetLM/SDAR-30B-A3B-Chat-b32 \
    --revision c351bbc37d240aa6871f167e8f92d694281b0c22 \
    --local-dir hf_cache/SDAR-30B-A3B-Chat-b32

hf download JetLM/SDAR-1.7B-Chat-b32 \
    --revision 2c83c05d84be90a01c72443b2123ebf0376f6fd4 \
    --local-dir hf_cache/SDAR-1.7B-Chat-b32
```

**Local cache on this machine:** `hf_cache/hub` — a symlink to
`/dev/shm/dlm/hf_cache` (RAM-backed `tmpfs`, wiped on container restart).
Confirmed present at verification time:

```
hf_cache -> /dev/shm/dlm/hf_cache
hf_cache/hub/models--JetLM--SDAR-30B-A3B-Chat-b32
hf_cache/hub/models--JetLM--SDAR-1.7B-Chat-b32
```

---

## 2. Team-trained checkpoints, by run

Four training rounds have produced checkpoints or attempted to; one produced
none. All are LoRA judge adapters on the 30B (rank 32, alpha 64, dropout 0, on
`q_proj,k_proj,v_proj,o_proj` × 48 layers) except the two drafter fine-tunes,
which are full fp32 fine-tunes of the 1.7B.

| run | kind | steps published | status | used in RESULTS.md? |
|---|---|---|---|---|
| `t1_gsm8k_r1` | LoRA judge, `train_gates=true` | 400, 600 (of 1057 target) | **rejected** — incompatible shape | no |
| `t2a_gsm8k_flag` | LoRA judge, `draft_flag=true` | 200/400/600/800/1000/1057 | **superseded** | provenance only (threshold sweep, progression plots) |
| `t2b_union_flag` | LoRA judge, `draft_flag=true` | 2200/2400/2600/2800/3000/3057 | **step003057 = published judge**; others superseded | **yes — every judge number in RESULTS.md and the deck** |
| `train_drafter/d1` | full fine-tune, 1.7B drafter | 1000, 1080 | **rejected** at gate | no (gate metrics only) |
| `train_drafter/d2_cleanprefix` | full fine-tune, 1.7B drafter | 200, 400, 600 | **rejected** at gate | no (gate metrics only) |
| `train_ablation/{A,B,C,D,E}` | LoRA judge, MATH-share ablation | — | **no checkpoint** — launched, not reached a save step | no |

### 2.1 t1_gsm8k_r1 — first judge attempt (rejected)

`train_gates=true` and no `draft_flag`, so this run's adapter has 432 tensors,
not 385, and is **not loadable by a `draft_flag` cell and vice versa**
(`load_adapter` asserts on the mismatch). Abandoned in favor of the
`draft_flag` design used from `t2a` onward; never appears in RESULTS.md.

Recipe essentials, from `logs/draftalign/train/t1_gsm8k_r1/config.json`
(`config_hash de58eb0854715659`, verified present locally, 967 B, byte count
matches the live inventory):

| index | steps (target) | world_size | grad_accum | lr_lora | lr_gate | seed | draft_flag | train_gates | resume |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|---|
| `logs/draftalign/trainset_gsm8k_r1/index.jsonl` | 1057 | 8 | 4 | 2e-5 | 1e-5 | 0 | *(key absent → false)* | **true** | `null` |

Checkpoints — base path `logs/draftalign/train/t1_gsm8k_r1/step<NNNNNN>/`:

| step | file | bytes | sha256[:16] | local sha256 check | status |
|---|---|---:|---|:--:|---|
| 400 | `logs/draftalign/train/t1_gsm8k_r1/step000400/adapter.safetensors` | 157,334,472 | `5b25cba2452a6bae` | **match** | rejected |
| 600 | `logs/draftalign/train/t1_gsm8k_r1/step000600/adapter.safetensors` | 157,334,472 | `e97175e953ef153c` | **match** | rejected |

Each step also has `adapter_config.json` (354 B) and `trainer_state.pt`
(314,942,923 B). No `_ad<hash8>` decode cell was observed for either hash —
this run was never carried into a `da_decode` cell.

```
step000400  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t1_gsm8k_r1/step000400
step000600  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t1_gsm8k_r1/step000600
```

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train/t1_gsm8k_r1/step000400/adapter*" \
    --include "logs/draftalign/train/t1_gsm8k_r1/step000600/adapter*" \
    --local-dir .
```

### 2.2 t2a_gsm8k_flag — GSM8K-only judge (superseded)

Same LoRA/`draft_flag` shape used from here on. Kept because the threshold
sweep and the progression plots in `results/metrics/draftalign/` were measured
on it, and because its step 1057 is the first stage of the two-stage
curriculum that produced `t2b_union_flag` (see §2.3.2).

Recipe essentials, from
`logs/draftalign/train/t2a_gsm8k_flag/config.json` (`config_hash
4fbb68e5d5f3fdaf`, verified present locally, 1037 B, matches the live
inventory):

| index | steps | world_size | grad_accum | lr_lora | seed | draft_flag | train_gates | resume | warmup | holdout_items |
|---|---:|---:|---:|---:|---:|:--:|:--:|---|---:|---:|
| `logs/draftalign/trainset_gsm8k_r1/index.jsonl` | 1057 | 8 | 4 | 2e-5 | 0 | **true** | false | `null` | 100 | 100 |

Checkpoints — base path `logs/draftalign/train/t2a_gsm8k_flag/step<NNNNNN>/`:

| step | file | bytes | sha256[:16] | `_ad` tag | local sha256 check | status |
|---|---|---:|---|---|:--:|---|
| 200 | `.../step000200/adapter.safetensors` | 107,005,856 | `6fd3c2e933b67c6f` | `_ad6fd3c2e9` | **match** | superseded |
| 400 | `.../step000400/adapter.safetensors` | 107,005,856 | `7d6e7e823792d329` | `_ad7d6e7e82` | **match** | superseded |
| 600 | `.../step000600/adapter.safetensors` | 107,005,856 | `bb1f79ea97f033ed` | `_adbb1f79ea` | **match** | superseded |
| 800 | `.../step000800/adapter.safetensors` | 107,005,856 | `329be7a372295e6c` | `_ad329be7a3` | **match** | superseded |
| 1000 | `.../step001000/adapter.safetensors` | 107,005,856 | `712a8ebcb2eaba49` | `_ad712a8ebc` | **match** | superseded |
| **1057** | `.../step001057/adapter.safetensors` | 107,005,856 | `191ea8d69b509628` | `_ad191ea8d6` | **match** | superseded — curriculum stage-1 endpoint (see §2.3.2) |

`adapter_config.json` is 378 B and `trainer_state.pt` is 214,255,306 B at
every step (path prefix `logs/draftalign/train/t2a_gsm8k_flag/step<NNNNNN>/`).
Two legacy/ambiguous decode-cell tags also refer to this recipe: `_adb3106a4d`
(config hash of *both* T2a and T2b — ambiguous, do not use to identify a
checkpoint) and `_ad5d9add10` (this run's own config hash, also ambiguous
across its steps).

**Note:** no `t2a` checkpoint has been evaluated at the headline `c95`
operating point at all (`docs/TRAINING_RECIPE_2026-09-09.md` §1.1, §6) — every
`t2a`-vs-`t2b` comparison that exists today is at the looser `c90_o99` point.

```
step000200  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step000200
step000400  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step000400
step000600  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step000600
step000800  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step000800
step001000  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step001000
step001057  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2a_gsm8k_flag/step001057
```

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step000200/adapter*" \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step000400/adapter*" \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step000600/adapter*" \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step000800/adapter*" \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step001000/adapter*" \
    --include "logs/draftalign/train/t2a_gsm8k_flag/step001057/adapter*" \
    --local-dir .
```

### 2.3 t2b_union_flag — **the published judge**

**This is the adapter behind settings 3, 4 and 5 in README.md and every
"judge" row in RESULTS.md.** Trained on the union index (GSM8K + MATH + MBPP),
3,057 steps, `world_size=7`, `draft_flag=true`, `train_gates=false`, 385
tensors, 26,740,736 fp32 parameters (0.089% of the model).

Recipe essentials, from `logs/draftalign/train/t2b_union_flag/config.json`
(`config_hash f8ee21d32a2e7980`, verified present locally, 1070 B, matches
the live inventory):

| index | steps | world_size | grad_accum | lr_lora | seed | draft_flag | train_gates | resume (literal config value) | warmup | holdout_items |
|---|---:|---:|---:|---:|---:|:--:|:--:|---|---:|---:|
| `logs/draftalign/trainset_union_r2/index.jsonl` | 3057 | 7 | 5 | 2e-5 | 1 | **true** | false | `logs/draftalign/train/t2b_union_flag` | 50 | 150 |

> **On "resume":** `config.json`'s `resume` field is **self-referential** — it
> names the run's own output directory, not `t2a_gsm8k_flag/step001057`. See
> the discrepancy note in §5. The `world_size 8→7`, `grad_accum 4→5`,
> `seed 0→1`, `warmup 100→50`, `holdout_items 100→150` deltas vs. `t2a` above
> are all confirmed directly against both configs and match the "six things
> changed at once" confound recorded in `docs/TRAINING_RECIPE_2026-09-09.md`
> §1.2.

#### 2.3.1 Checkpoints — base path `logs/draftalign/train/t2b_union_flag/step<NNNNNN>/`

| step | file | bytes | sha256[:16] | `_ad` tag | local sha256 check | status |
|---|---|---:|---|---|:--:|---|
| 2200 | `.../step002200/adapter.safetensors` | 107,005,856 | `a7be11a8f5843cd1` | `_ada7be11a8` | **match** | superseded — see §2.3.3 |
| 2400 | `.../step002400/adapter.safetensors` | 107,005,856 | `022382b2a5c63f32` | `_ad022382b2` | **match** | superseded |
| 2600 | `.../step002600/adapter.safetensors` | 107,005,856 | `f51dafdc064761dd` | `_adf51dafdc` | **match** | superseded |
| 2800 | `.../step002800/adapter.safetensors` | 107,005,856 | `38a9ef557a7d4fb8` | `_ad38a9ef55` | **match** | superseded |
| 3000 | `.../step003000/adapter.safetensors` | 107,005,856 | `3e97af0249eca7f4` | `_ad3e97af02` | **match** | superseded |
| **3057** | `.../step003057/adapter.safetensors` | **107,005,856** | **`734af65681ab6bb8`** | **`_ad734af656`** | **match** | **PUBLISHED JUDGE** |

`adapter_config.json` is 378 B and `trainer_state.pt` is 214,260,746 B at
every step.

```
step002200  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step002200
step002400  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step002400
step002600  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step002600
step002800  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step002800
step003000  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step003000
step003057  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train/t2b_union_flag/step003057
```

Pull just the published adapter (107 MB, not the 127 GB dataset):

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train/t2b_union_flag/step003057/adapter*" \
    --local-dir .
```

All six steps in one call:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train/t2b_union_flag/step002200/adapter*" \
    --include "logs/draftalign/train/t2b_union_flag/step002400/adapter*" \
    --include "logs/draftalign/train/t2b_union_flag/step002600/adapter*" \
    --include "logs/draftalign/train/t2b_union_flag/step002800/adapter*" \
    --include "logs/draftalign/train/t2b_union_flag/step003000/adapter*" \
    --include "logs/draftalign/train/t2b_union_flag/step003057/adapter*" \
    --local-dir .
```

#### 2.3.2 It is a two-stage curriculum

`t2b_union_flag` is **not** a single training run in isolation: per
`docs/TRAINING_RECIPE_2026-09-09.md` §4.1, step 3057 is the result of `t2a`
(100% GSM8K) trained to step 1057, **then resumed into the union mix**
(GSM8K + MATH + MBPP) for a further ~2000 steps. Any arm branching off 3057
therefore reports a *curriculum* result, not a clean single-mix result — this
is exactly why round 3 (§0 above) insists on a fresh-from-stock init instead
of resuming from either 1057 or 3057.

#### 2.3.3 Open question: checkpoint 2200 vs. 3057

**Unresolved — flagged explicitly in `docs/TRAINING_RECIPE_2026-09-09.md`
§1.1.** At the `c90_o99` operating point, compute gain saturates by step 2200
while accuracy keeps falling through 3057:

| checkpoint | accuracy (`c90_o99`) | fwd/blk | compute vs. stock |
|---|---:|---:|---:|
| `t2b` step 2200 | 93.0 | 8.27 | 1.266× |
| `t2b` step 3057 | 89.0 | 8.43 | 1.242× |

But re-run at the **headline `c95` operating point** (the deck's settings
3/4/5), the two checkpoints cannot be told apart on the `judge alone` and
`judge + DES 32` arms (McNemar exact p = 1.000 both times); the `judge + DES
48` row favors 3057 (+4.0) but rests on only 6 discordant items out of 100 and
is not significant (p = 0.219, CI lower bound at zero). **Net: whether the
extra ~857 steps from 2200→3057 helped, hurt, or did nothing to accuracy is an
open question**, not settled by either operating point alone. See recipe §1.1
for the full analysis and caveats (n=100, calibvalid is threshold-tuning data,
no `t2a` checkpoint has ever been run at `c95`).

### 2.4 Drafter fine-tunes — d1 and d2_cleanprefix (rejected)

Full fp32 fine-tunes of the 1.7B dense drafter (not LoRA — sharded
`safetensors` plus a complete tokenizer/remote-code snapshot per step, so each
step directory is a standalone loadable model). **Both were rejected at their
gates and no published result in RESULTS.md uses them.** They are documented
because their gate metrics
(`results/metrics/draftalign/drafter_accuracy_*.json`,
`drafter_agreement_*.json`) are reported by `eval_table.py`, and this is the
provenance of those numbers.

Recipe essentials — **run-level `config.json` is not in the local mirror for
either run** (see §5); the fields below are sourced from `HUGGINGFACE.md` §4
only and were not independently re-verified against a local file:

| run | index | lr | steps | notes | config_hash |
|---|---|---:|---:|---|---|
| `d1` | `logs/draftalign/drafterset_union/index.jsonl` | 1e-5 | 1,080 | `only_correct=true` | `7815f0cdd2410890` |
| `d2_cleanprefix` | `logs/draftalign/drafterset_union_v2/index.jsonl` | 5e-6 | early-stopped at 600 of planned 1,602 (patience 2) | `conditioning=clean_prefix` | `5dee343e8f7141a1` |

Checkpoints — two safetensors shards per step
(`model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`,
4,969,539,560 B + 3,157,455,608 B in every case):

| run/step | sha256[:16] shard 1 | sha256[:16] shard 2 | trainer_state.pt bytes | local check | status |
|---|---|---|---:|---|---|
| `d1/step001000` | `aa5fe4e886bee276` | `9159bfb10519d517` | 16,254,193,795 | weights not mirrored locally; **size of `trainer_state.pt` verified locally (match)** | rejected |
| `d1/step001080` | `8e95f75cb2803e70` | `656de6fbb4f30e63` | 16,254,193,795 | weights not mirrored locally; **size of `trainer_state.pt` verified locally (match)** | rejected |
| `d2_cleanprefix/step000200` | `49ea67f2792991af` | `1de854fd67912278` | 16,254,193,859 | weights not mirrored locally; **size of `trainer_state.pt` verified locally (match)** | rejected |
| `d2_cleanprefix/step000400` | `5e845c5424727244` | `2e211dae7b298808` | 16,254,193,859 | weights not mirrored locally; **size of `trainer_state.pt` verified locally (match)** | rejected |
| `d2_cleanprefix/step000600` | `6e3e6f01fc5688e4` | `a84e7ecbadb7bc53` | 16,254,193,859 | weights not mirrored locally; **size of `trainer_state.pt` verified locally (match)** | rejected — **not listed in `HUGGINGFACE.md`'s hash table, see §5** |

Base path: `logs/draftalign/train_drafter/<run>/step<NNNNNN>/`.

```
d1/step001000              https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train_drafter/d1/step001000
d1/step001080              https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train_drafter/d1/step001080
d2_cleanprefix/step000200  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train_drafter/d2_cleanprefix/step000200
d2_cleanprefix/step000400  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train_drafter/d2_cleanprefix/step000400
d2_cleanprefix/step000600  https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/2fcc0418b7c8915f61919cd413a9c118054fccc7/logs/draftalign/train_drafter/d2_cleanprefix/step000600
```

Weights only, one step (~8.1 GB, skips the 16 GB optimizer state):

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train_drafter/d1/step001080/*" \
    --exclude "*trainer_state.pt" --local-dir .
```

All five published steps:

```bash
hf download Arushhh/dlm-exp1-drafter-voting-research-artifacts --repo-type dataset \
    --include "logs/draftalign/train_drafter/d1/step001000/*" \
    --include "logs/draftalign/train_drafter/d1/step001080/*" \
    --include "logs/draftalign/train_drafter/d2_cleanprefix/step000200/*" \
    --include "logs/draftalign/train_drafter/d2_cleanprefix/step000400/*" \
    --include "logs/draftalign/train_drafter/d2_cleanprefix/step000600/*" \
    --exclude "*trainer_state.pt" --local-dir .
```

### 2.5 Ablation runs — no checkpoints

`logs/draftalign/train_ablation/` holds the MATH-share ablation runs started
2026-09-09 09:09–09:10 UTC. **No `adapter.safetensors` exists for any of
them** — none had reached a save step when the HF snapshot was taken.
`train.rank0.jsonl` is 0 bytes for the three that have a run directory at all.

| run | run directory on HF? | `config.json` on HF? | local mirror has? | MATH blocks | GSM8K blocks | total blocks | items |
|---|:--:|:--:|---|---:|---:|---:|---:|
| `A_math000` | yes | yes (1,050 B) | wandb/ logs only — **no config.json locally** | 0 | 8,005 | 8,005 | 1,346 |
| `B_math125` | no — `.log` file only | — | not mirrored | 1,011 | 7,000 | 8,011 | 1,248 |
| `C_math310` | yes | yes (1,050 B) | wandb/ logs only — **no config.json locally** | 2,487 | 5,520 | 8,007 | 1,106 |
| `D_math625` | no — `.log` file only | — | not mirrored | 5,006 | 3,000 | 8,006 | 839 |
| `E_math1000` | yes | yes (1,052 B) | wandb/ logs only — **no config.json locally** | 8,001 | 0 | 8,001 | 553 |

Block counts are from `logs/draftalign/trainset_union_r2/ablation_manifest.json`
via `HUGGINGFACE.md` §5. Status for all five: **no checkpoint**.

---

## 3. Baseline / benchmark records — the existing "benchmarked model space"

This is what round 3 is being measured against: every stock / blind-accept /
trained-judge decode cell that already exists on disk, pulled from the two
local mirror trees named in the task. **Update:** the per-item JSONL payloads
for `da_decode/` are now present locally (305 real `.jsonl` files, not just
`.jsonl.done` sentinels), so every count below is a **direct recount**, not a
citation. Method: for each cell and each split found in its filenames
(`<cell>.<split>.shard<k>of<n>.jsonl`), every non-empty line across all
matching shards was parsed as JSON, keyed by `(split, idx)`, and deduplicated
on that key; any `*.oldcfg-*` file would have been skipped (none were found
anywhere in the 305 files). The `research_v1/baselines_b32/` counts in §3.2
were re-verified with this same dedup method and are unchanged from the
earlier pass. See §4.

### 3.1 `logs/draftalign/da_decode/` — stock, blind-accept, judge cells

All three arms are present for all six eval datasets, plus `calibvalid` —
which is **not** its own top-level cell directory (there is no
`calibvalid_*` directory; that was a mis-framing in the previous version of
this table). It is a `split` embedded inside each `gsm8k_B32_t95_*` cell
directory alongside the `gsm8k500` split, e.g.
`gsm8k_B32_t95_stock/gsm8k_B32_t95_stock.calibvalid.shard0of4.jsonl`.
Canonical cell names: `<ds>_B32_t95_stock`, `<ds>_B32_t95_dv-s8_blind_q50`,
`<ds>_B32_t95_dv-s8_full_c95_ad734af656` (the published-judge cell). Counts
below are **real, deduplicated-by-`(split, idx)` record counts**, not shard
tallies:

| dataset | stock n | blind-accept n | judge (`_ad734af656`) n | `RESULTS.md` n | match? |
|---|---:|---:|---:|---:|:--:|
| GSM8K-500 | *(see note 1)* | 500 | 500 | 500 | ✅ |
| MATH-500 | 500 | 500 | 500 | 500 | ✅ |
| MBPP | 257 | 257 | 257 | 257 | ✅ |
| HumanEval | 164 | 164 | 164 | 164 | ✅ |
| GPQA-Diamond | 198 | 198 | 198 | 198 | ✅ |
| AIME | 90 | 90 | 90 | 90 | ✅ |
| calibvalid | 100 | 100 | 100 | 100 | ✅ |

**All 17 in-scope (cell, split) counts recounted above match `RESULTS.md`
exactly. Zero mismatches, zero partial cells** (none fall short of their
dataset size: gsm8k500 500, math500 500, mbpp 257, humaneval 164, gpqa 198,
aime 90, calibvalid 100). Recount diagnostics for all 17: 0 duplicate
`(split, idx)` keys, 0 JSON parse errors, 0 records missing an `idx` field,
0 `*.oldcfg-*` files encountered anywhere in the 305-file set.

*Note 1 — GSM8K-500 stock:* `gsm8k_B32_t95_stock/` (in `da_decode/`) never
carries a `gsm8k500` split by design — per `RESULTS.md` §1, the GSM8K-500
stock reference lives only in `logs/research_v1/baselines_b32/` (§3.2 below,
n=500, confirmed there). This is not a gap in this mirror; it matches the
producer note in `RESULTS.md` §7.

> **Data-hygiene flag — stray split found, not counted above.**
> `gsm8k_B32_t95_stock/` also contains 6 records under a filename split-token
> of `aime` (`gsm8k_B32_t95_stock.aime.shard{0,1}of4.jsonl` — only 2 of the 4
> shards its own filenames imply exist). This is neither a valid GSM8K record
> nor a usable AIME cell: 6 records is far short of AIME's 90-item size, and
> it sits inside the *GSM8K* stock directory, not `aime_B32_t95_stock/` (the
> real AIME stock cell, separate and complete at 90/90 above). It reads as
> misplaced/leftover output rather than a labeled cell, is **excluded from
> every count in this document**, and does not affect any `RESULTS.md`
> number. Flagging for cleanup rather than silently dropping it.

75 cells exist in total under `da_decode/` on HF (`HUGGINGFACE.md` §7); the
local mirror's 305 real `.jsonl` files cover a wider set than the 17
tabulated above — also present are pooled (`p32desf`/`p48desf`), other
`_ad<hash8>` checkpoint tags, and `c90_o99`-style threshold-sweep variants,
none of which were in scope for this table.

### 3.2 `logs/research_v1/baselines_b32/` — real payloads, counted directly

Only stock GSM8K cells are mirrored here (no blind-accept or judge cells in
this tree). Line counts below are `wc -l` across every `*.shard*of3.jsonl` in
the directory (`.done` marker files excluded from the count):

| cell | split | shards | n (lines, counted locally) | block size |
|---|---|:--:|---:|:--:|
| `gsm8k_B32_t90_stock` | calibvalid | 3 | **100** | 32 |
| `gsm8k_B32_t95_stock` | calibvalid | 3 | **100** | 32 |
| `gsm8k_B32_t95_stock` | gsm8k500 | 3 | **500** | 32 |
| `gsm8k_B4_t90_stock` | calibvalid | 3 | **100** | **4 — not block 32** |
| `gsm8k_B4_t90_stock` | gsm8k500 | 3 | **500** | **4 — not block 32** |

Both counts cross-check exactly against `RESULTS.md` §1: GSM8K-500 n=500,
calibvalid n=100. `RESULTS.md` §1 also notes this `gsm8k_B32_t95_stock`
family is the one actually used for the GSM8K-500 stock row (not the
`da_decode` twin) because the two are bit-identical in `gen_ids` and
`passes_per_block` and only wall-clock differs (184.3 ms/token here vs. 187.1
in `da_decode`; every RESULTS.md comparison uses the 187.1 figure so both
sides of a pair share the same run family). The `B4` rows are a legacy
block-4 comparison point, unrelated to round 3's block-32 checkpoints —
included here only because they physically exist in this directory.

---

## 4. Verification summary

| check | result |
|---|---|
| `t1_gsm8k_r1` adapter.safetensors × 2 steps — sha256[:16] vs. TSV | **match**, both steps |
| `t2a_gsm8k_flag` adapter.safetensors × 6 steps — sha256[:16] vs. TSV | **match**, all 6 steps |
| `t2b_union_flag` adapter.safetensors × 6 steps — sha256[:16] vs. TSV, incl. published `step003057` | **match**, all 6 steps |
| **Total adapter.safetensors checked** | **14 / 14 match, 0 mismatches** |
| `train_drafter/{d1,d2_cleanprefix}` model-*.safetensors (10 files) | **not present in local mirror** — cannot independently verify; TSV and `HUGGINGFACE.md` agree with each other for the 4 pairs `HUGGINGFACE.md` lists |
| `train_drafter/{d1,d2_cleanprefix}` `trainer_state.pt` × 5 (byte size only, not hashed — out of the requested scope) | **present locally, size matches TSV exactly** for all 5 |
| `train_drafter/{d1,d2_cleanprefix}` `config.json`, run-level (×2) and per-step (×5) | **missing from local mirror** — listed in the live TSV inventory but not downloaded into `artifacts_mirror` |
| `train_ablation/{A_math000,C_math310,E_math1000}` `config.json` (×3) | **missing from local mirror** — listed in TSV (1,050 / 1,050 / 1,052 B) but only `wandb/` logs were mirrored |
| `da_decode/` jsonl payloads for the 17 in-scope (cell, split) rows in §3.1 | **present and complete** — 305 real files now in the mirror; all 17 recounted directly, all 17 match `RESULTS.md` exactly, 0 partial |
| `da_decode/gsm8k_B32_t95_stock/` stray `aime`-labeled split (6 records, 2/4 shards) | **present but out of scope / misplaced** — not part of any dataset's canonical cell, excluded from all counts, flagged in §3.1 for cleanup |
| `research_v1/baselines_b32/` jsonl payloads | **present and complete**, 15/15 real files, counted directly (recounted with the same dedup method as `da_decode/`; unchanged from the earlier pass — 100/100/500/100/500, 0 dupes) |

### 4.1 Discrepancies between `HUGGINGFACE.md` and the live inventory

1. **`HUGGINGFACE.md` §4's sha256 listing for `d2_cleanprefix` is incomplete.**
   The prose table above it states three steps were published
   ("step000200, step000400, step000600"), but the fenced sha256 block only
   gives pairs for step000200 and step000400 — **step000600 is omitted**. The
   live TSV inventory (same revision, `2fcc0418b7c8915f61919cd413a9c118054fccc7`)
   does contain step000600's shard hashes: `6e3e6f01fc5688e4` /
   `a84e7ecbadb7bc53`, byte sizes consistent with the other two steps. This
   document's §2.4 table includes the missing row, sourced from the TSV.
2. **`t2b_union_flag/config.json`'s `resume` field does not encode the
   documented curriculum handoff.** `HUGGINGFACE.md` §3.1 and
   `docs/TRAINING_RECIPE_2026-09-09.md` §4.1 both describe `step003057` as a
   two-stage curriculum — `t2a_gsm8k_flag` to step 1057, then resumed into
   the union mix. But the run's own `config.json` sets `"resume":
   "logs/draftalign/train/t2b_union_flag"` — **its own output directory**,
   not `logs/draftalign/train/t2a_gsm8k_flag/step001057`. Nor does the live
   inventory show a `step001057` checkpoint anywhere inside
   `t2b_union_flag/`'s own tree (its published steps start at 2200). The
   curriculum fact is stated consistently across two narrative documents, but
   is **not independently evidenced by this config field** — the actual
   cross-run weight handoff must happen through a mechanism this config key
   doesn't capture. Flagging rather than resolving; see §2.3.2.

No other discrepancies were found: all config hashes (`de58eb0854715659`,
`4fbb68e5d5f3fdaf`, `f8ee21d32a2e7980`), all `t1`/`t2a`/`t2b` adapter hashes,
and all byte counts checked against the live TSV agree with `HUGGINGFACE.md`.
