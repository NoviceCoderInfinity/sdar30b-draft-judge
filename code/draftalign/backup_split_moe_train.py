#!/usr/bin/env python
"""Back up round-3 training artifacts to their OWN folder, separate from the data backup (user, 2026-09-15).

Public HF dataset Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts:
  Nemotron-data-subset-training-diffusion-moe/   <- data backup (backup_r3_hf.py)
  Nemotron-split-moe-train/r3_A/checkpoints/     <- pushed by ckpt_sync_r3.sh (one dir per 100 steps)
  Nemotron-split-moe-train/r3_A/{train_logs,evals,README.md}  <- this script
Runs as the logged-in HF account (not .env's HF_TOKEN).
  /venv/r3/bin/python code/draftalign/backup_split_moe_train.py              # train logs + README + evals
  /venv/r3/bin/python code/draftalign/backup_split_moe_train.py --evals-only
"""
import glob, io, json, os, sys, time
from huggingface_hub import HfApi, CommitOperationAdd
os.environ.pop("HF_TOKEN", None)
REPO = "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"
DST = "Nemotron-split-moe-train/r3_A"; RUN = "logs/draftalign/train/r3_A"; TRAIN_EX = 22734; STATES = 32
api = HfApi(); assert api.whoami()["name"] == "Anupam-Rawat-IITB", "not logged in as Anupam-Rawat-IITB"
ops = []
if "--evals-only" not in sys.argv:
    steps = sorted({i.path.split("/")[-1] for i in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"{DST}/checkpoints")})
    for f in ["train.log", "config.json"] + sorted(os.path.basename(x) for x in glob.glob(f"{RUN}/train.rank*.jsonl")):
        ops.append(CommitOperationAdd(path_in_repo=f"{DST}/train_logs/{f}", path_or_fileobj=f"{RUN}/{f}"))
    last = [l for l in open(f"{RUN}/train.log") if l.startswith("[step")]
    last_step = int(last[-1].split("/")[0][6:]) if last else 0
    base = f"https://huggingface.co/datasets/{REPO}/tree/main/{DST}/checkpoints/"
    rows = "\n".join(f"| {int(s[4:])} | {int(s[4:]) * STATES / TRAIN_EX:.2f} | [{s}]({base}{s}) |" for s in steps)
    readme = f"""# Nemotron-split-moe-train — round-3 judge LoRA training (`r3_A`)

Training artifacts only. The data backup (raw + processed data, strategy docs) lives next door in
[`Nemotron-data-subset-training-diffusion-moe/`](https://huggingface.co/datasets/{REPO}/tree/main/Nemotron-data-subset-training-diffusion-moe).

- Base: `JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc3…`, drafter `JetLM/SDAR-1.7B-Chat-b32`, block 32. LoRA r32/α64 q,k,v,o + draft flag, gates frozen.
- 8 GPUs × grad-accum 4 = 32 states/step; train split 22,734 examples → **1 epoch ≈ 710.4 steps**. Planned 2,175 steps (3.06 ep).
- History (IST, 15 Sep 2026): launched 02:38 → stopped at step 1,560 at 08:16 (holdout looked flat) → downstream evals on steps 700/1400 →
  **resumed from `step001500`** after the evals (downstream code accuracy was still improving epoch 1 → 2) → **stopped for good at step 2,000 at 11:44 IST**
  (user decision: 3 epochs of this loss did not fix confident wrong overrides). Successor run: [`r3_B`](https://huggingface.co/datasets/{REPO}/tree/main/Nemotron-split-moe-train/r3_B), new loss from `step000700`. Latest logged step: **{last_step}**.
- Epoch-boundary checkpoints: epoch 1 ≈ `step000700`, epoch 2 ≈ `step001400`.
- W&B: https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_A · code/docs: GitHub `NoviceCoderInfinity/diffusion-moe-general-expert`, branch `round3-2026-09-14`.

Layout: `checkpoints/stepNNNNNN/{{adapter.safetensors, adapter_config.json, trainer_state.pt}}` · `train_logs/` (train.log, per-rank jsonl — steps 1510–1560 appear twice, once per run segment; config.json) · `evals/` (downstream eval records, `eval_summary.{{json,md}}`).

| step | epoch | checkpoint |
|---:|---:|---|
{rows}

Resume: `torchrun --standalone --nproc_per_node=8 code/draftalign/train.py --resume <stepNNNNNN dir> ...` (flags in `train_logs/config.json`).

_Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_
"""
    ops.append(CommitOperationAdd(path_in_repo=f"{DST}/README.md", path_or_fileobj=io.BytesIO(readme.encode())))
ev = "logs/draftalign/probes_r3/r3_A"
for f in glob.glob(f"{ev}/**/*.jsonl", recursive=True) + glob.glob(f"{ev}/*.json") + glob.glob(f"{ev}/*.md") + glob.glob(f"{ev}/*.txt"):
    if "/wandb/" in f: continue
    ops.append(CommitOperationAdd(path_in_repo=f"{DST}/evals/{os.path.relpath(f, ev)}", path_or_fileobj=f))
for f in sorted(glob.glob("logs/draftalign/plateau_probe_r3/*.json")):   # plateau diagnosis (probe_plateau_r3.py)
    ops.append(CommitOperationAdd(path_in_repo=f"{DST}/analysis/plateau_probe/{os.path.basename(f)}", path_or_fileobj=f))
for f in ("docs/report/r3_report.html", "logs/draftalign/resume_r3.log"):
    if os.path.exists(f):
        ops.append(CommitOperationAdd(path_in_repo=f"{DST}/analysis/{os.path.basename(f)}", path_or_fileobj=f))
# ---- r3_B: the calibrated-loss retrain from r3_A step000700 (user, 11:45 IST 15 Sep) ----
RUN_B, DST_B = "logs/draftalign/train/r3_B", "Nemotron-split-moe-train/r3_B"
if "--evals-only" not in sys.argv and os.path.exists(f"{RUN_B}/config.json"):
    for f in ["train.log", "config.json"] + sorted(os.path.basename(x) for x in glob.glob(f"{RUN_B}/train.rank*.jsonl")):
        ops.append(CommitOperationAdd(path_in_repo=f"{DST_B}/train_logs/{f}", path_or_fileobj=f"{RUN_B}/{f}"))
    lb = [l for l in open(f"{RUN_B}/train.log") if l.startswith("[step")]
    lsb = int(lb[-1].split("/")[0][6:]) if lb else 0
    try:
        steps_b = sorted({i.path.split("/")[-1] for i in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"{DST_B}/checkpoints")})
    except Exception:
        steps_b = []
    base_b = f"https://huggingface.co/datasets/{REPO}/tree/main/{DST_B}/checkpoints/"
    rows_b = "\n".join(f"| {int(s[4:])} | {1 + int(s[4:]) * STATES / TRAIN_EX:.2f} | [{s}]({base_b}{s}) |" for s in steps_b) or "| – | – | first checkpoint at step 100 |"
    readme_b = f"""# Nemotron-split-moe-train — `r3_B`: calibrated judge-LoRA retrain from epoch 1

Same model, data (22,734 train / 824 holdout examples, seed 1), LoRA and schedule shape as [`r3_A`](https://huggingface.co/datasets/{REPO}/tree/main/Nemotron-split-moe-train/r3_A);
**starts from `r3_A/step000700` (epoch 1) with a fresh optimizer and lr schedule** (`--init-adapter`). Why: r3_A's confident (>0.95) OVERRIDES of the
draft were right only ~78–81% of the time (code ~77%) and did not improve with more epochs, while confident confirms got worse (99.1% → 96.7%).

Loss changes (all new flags; r3_A's loss is bit-identical when they are off):
- certainty weight `--beta 2.0 → 0.5`
- `--calib 1.0`: + BCE(top-1 prob of the MASK-suppressed softmax, [that token == target]) — a confident WRONG token costs −log(1−p)
- `--override-weight 2.0`, `--code-override-weight 3.0`: loss weight on slots where the draft token is present and wrong
- holdout eval every 100 steps now logs `draft_pass1_ovr_{{acc,conf_frac,conf_precision}}` for checkpoint selection

Planned {710} steps = 1 epoch → ends at **epoch 2**. `r3_B` step N ≈ epoch 1 + N/710.4. Latest logged step: **{lsb}**.

| step | epoch | checkpoint |
|---:|---:|---|
{rows_b}

_Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_
"""
    ops.append(CommitOperationAdd(path_in_repo=f"{DST_B}/README.md", path_or_fileobj=io.BytesIO(readme_b.encode())))
evB = "logs/draftalign/probes_r3/r3_B"       # r3_B final-checkpoint evals (eval_after_r3B.sh); new stock DES baselines land in r3_A/stock
for f in glob.glob(f"{evB}/**/*.jsonl", recursive=True) + glob.glob(f"{evB}/*.json") + glob.glob(f"{evB}/*.md"):
    if "/wandb/" in f: continue
    ops.append(CommitOperationAdd(path_in_repo=f"{DST_B}/evals/{os.path.relpath(f, evB)}", path_or_fileobj=f))
api.create_commit(REPO, repo_type="dataset", operations=ops, commit_message="Nemotron-split-moe-train: r3_A + r3_B train logs, evals, README")
print(f"[split_backup] committed {len(ops)} ops -> https://huggingface.co/datasets/{REPO}/tree/main/{DST}", flush=True)
