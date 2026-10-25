# EXCLUDED

What is **not** in this repository, why, and where it still lives.

---

## 1. Bulk artifacts — excluded by size

| what | size | where it is |
|---|---:|---|
| `logs/` — per-item decode records, per-block NPZ traces, judge checkpoints, datagen decodes, training indices | **120 GB** | HF mirror (§4) |
| `hf_cache/` — the b32 and d32 model weights and the `transformers` remote-code modules | **61 GB** | the public HF model repos (§4) |
| round-3 data (2026-09-14, growing) — selected source rows, datagen records and traces, `trainset_r3` indices | in progress | the public HF folder `Nemotron-data-subset-training-diffusion-moe/` (§4) |

Every table in `RESULTS.md` names the record path it was computed from, so a
reader who pulls the mirror can recompute it. The record schema is documented in
`AGENTS.md` §7.

## 2. File types stripped everywhere

Listed in `.gitignore` so they cannot creep back:

| pattern | why |
|---|---|
| `*.jsonl` | per-item records belong with `logs/`, not in git |
| `*.npz` | per-block traces; the JSON summaries beside them are kept |
| `__pycache__/`, `*.pyc`, `.pytest_cache/` | build artifacts |
| `.env`, `*.pem`, `*.key`, `wandb/` | credentials; W&B runs are mirrored on wandb.ai |

**Raw agent session transcripts are not published here.** They are agent scratch,
they carry no measurement that is not already in `RESULTS.md`, and even redacted
they are the highest-risk place for a pasted credential to survive.

## 3. Code retained even though nothing imports it

* `code/compat/flash_attn/` — reached only via `sys.path`, never by name.
  **The model cannot be imported without it** (see `ARCHITECTURE.md` §1).
* `code/vendor/oc_datasets/__init__.py`, `code/vendor/opencompass/__init__.py`,
  `code/tests/__init__.py` — package markers.
* `code/build_prompts.py` — produced `code/configs/prompts.json`; it is the
  provenance of every frozen split.
* `code/draftalign/train_drafter.py`, `build_drafter_set.py`,
  `eval_drafter_accuracy.py`, `eval_drafter_agreement.py` — the 1.7B drafter
  fine-tune. It was **rejected at its gates** and no published cell uses it, but
  its gate metrics are in `results/metrics/draftalign/` and `eval_table.py`
  prints them, so the code that produced them is kept.
* All `code/draftalign/orchestrate_*.sh`, `launch_*.sh`, `monitor_status.sh`,
  `requeue_anchored_after_pause.sh` — these *are* how the published runs were
  launched, and they are the only record of the wave structure.
* `code/draftalign/build_opencode_pool.py`, `build_if_pool.py` (2026-09-14) — produced
  `code/configs/opencode_pool.json` and `if_pool.json`; they are the provenance of the
  round-3 pools, as `build_prompts.py` is of `prompts.json`.
* `code/draftalign/launch_r3_datagen*.sh`, `queue_r3_lines.sh`, `sched_r3.sh`,
  `build_r3_index.sh`, `backup_r3_hf.py` (2026-09-14) — how the round-3 data is being
  decoded, indexed and backed up.

**Import closure holds:** no file in `code/` imports anything that is missing,
and every file parses.

## 4. Where the excluded artifacts live

Full detail — every checkpoint step with its byte size and sha256, the training
indices, the loss functions, and a `hf download` line for each — is in
**HUGGINGFACE.md**. The short version:

| artifact | location | access |
|---|---|---|
| `logs/`, `results/` — the full mirror | `https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts` | public |
| the 30B MoE under test | `https://huggingface.co/JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc37d240aa6871f167e8f92d694281b0c22` | public |
| the 1.7B drafter | `https://huggingface.co/JetLM/SDAR-1.7B-Chat-b32` @ `2c83c05d84be90a01c72443b2123ebf0376f6fd4` | public |
| round-3 data — selected source rows, datagen records + `traces.tar`, `trainset_r3` indices + `items.tar`, and the code, docs and env freezes as run | `https://huggingface.co/datasets/Arushhh/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-data-subset-training-diffusion-moe`, synced every 30 min by `code/draftalign/backup_r3_hf.py --loop 30` (HUGGINGFACE.md §9) | public |
| round-3 working copies on the 2026-09-14 container — `logs/`, `hf_cache/`, the OpenCodeInstruct snapshot | `/dev/shm/dlm/` — RAM-backed, **wiped on restart**; anything newer than the last sync is lost | local only |
| full upstream round-3 sources | `nvidia/OpenCodeInstruct` @ `8f3ba5bafe4d…`, `allenai/IF_multi_constraints_upto5` @ `2e3a77407b7f…`, `allenai/RLVR-IFeval` @ `47c03c73621c…`, `openai/gsm8k` @ `740312add88f…` — referenced by revision, not re-uploaded | public |

The judge checkpoint itself —
`logs/draftalign/train/t2b_union_flag/step003057/adapter.safetensors`,
107,005,856 bytes, sha256[:16] `734af65681ab6bb8` — is in the public artifacts
mirror under that path. It is 26,740,736 fp32 parameters; everything needed to
load it is in `code/draftalign/adapter.py::load_adapter`. The other 12 judge
checkpoints and the two rejected drafter fine-tunes are published there too —
see HUGGINGFACE.md §3 and §4.

## 5. Verification performed on this repository

* **Secret scan.** `grep -rIn` over every file for the literal substrings `hf_`,
  `ghp_`, `github_pat`, `sk-` and for token-shaped regexes
  (`hf_[A-Za-z0-9]{20,}`, `ghp_…`, `github_pat_…`, `sk-…`), plus
  `HF_TOKEN=`/`WANDB_API_KEY=`/`api_key=` followed by a literal. **Zero real
  matches.**
* **No credential files.** `find` for `.env`, `*.pem`, `*.key`, `*credential*`
  → nothing. (2026-09-14: the working tree now holds a gitignored `.env`, sourced by the
  round-3 launchers and read by `backup_r3_hf.py` for `HF_TOKEN`; it is not committed.)
* **No bulk data.** `find` for `*.jsonl`, `*.npz`, `*.tar.gz` → nothing.
* **Import closure.** Every `.py` in `code/` parses and resolves.
