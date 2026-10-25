#!/usr/bin/env python
"""Round-3 data backup -> public HF dataset repo, folder `Nemotron-data-subset-training-diffusion-moe/`.

Stages everything round-3 preprocessing produced into a RAM staging tree and uploads it with one commit:

  strategy/     recipes, data plans, IF recipe + wiring patch, checkpoint doc, the pipeline code as it ran,
                `git diff` of every edit to tracked files, the env (team freeze + /venv/r3 freeze)
  unprocessed/  the exact source rows selected (GSM8K pool slices, OpenCodeInstruct frozen pool + its raw
                parquet rows, verifiable-IF frozen pool) + SOURCES.md (repo ids, revisions, licences)
  processed/    datagen/<cell dir>/  per-item decode records (*.jsonl, *.done, shard logs) + traces.tar
                trainset_r3/<pool>/  index.jsonl, build_summary.json, items.tar
                final index (if built)
  README.md     regenerated each sync with a live status table

Traces/items are packed as tar (npz inside is already compressed) so the repo's file count stays small.
Secrets are never staged (explicit allowlist) and the stage is scanned for the .env token values before upload.

    /venv/r3/bin/python code/draftalign/backup_r3_hf.py              # one sync
    nohup /venv/r3/bin/python code/draftalign/backup_r3_hf.py --loop 30 > logs/draftalign/hf_backup_r3.log 2>&1 &
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDER = "Nemotron-data-subset-training-diffusion-moe"
REPO = "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"   # moved from Arushhh/ on 2026-09-15 (user)
STAGE = os.path.join("/dev/shm/dlm/hf_backup_stage", FOLDER)
ENV_KEEP = "/dev/shm/dlm/env/team_requirements_2026-09-09.txt"

DOCS = ["docs/TRAINING_RECIPE_2026-09-09.md", "docs/ICLR2027_DATA_PLAN.md", "DATA.md", "HUGGINGFACE.md",
        "docs/IF_RECIPE_2026-09-14.md", "docs/IF_POOL_WIRING_PATCH.md", "docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md"]
CODE = ["code/research_protocol.py", "code/run_desvote.py", "code/analyze_s1b.py", "code/run_research_baselines.py",
        "code/draftalign/run_da_decode.py", "code/draftalign/da_decode.py", "code/draftalign/build_train_set.py",
        "code/draftalign/dataset.py", "code/draftalign/build_opencode_pool.py", "code/draftalign/build_if_pool.py",
        "code/draftalign/if_scorer.py", "code/draftalign/launch_r3_datagen_gsm8k.sh", "code/draftalign/launch_r3_datagen.sh",
        "code/draftalign/launch_r3_datagen_after.sh", "code/draftalign/sched_r3.sh", "code/draftalign/queue_r3_lines.sh",
        "code/draftalign/build_r3_index.sh", "code/draftalign/backup_r3_hf.py"]
CONFIGS = ["code/configs/opencode_pool.json", "code/configs/opencode_pool.json.sha256",
           "code/configs/if_pool.json", "code/configs/if_pool.json.sha256", "code/configs/prompts.json.sha256"]


def load_env():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


def cp(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def signature(paths):
    """(count, bytes, newest mtime) -- files touched in the last 10 s are left for the next sync."""
    now, n, b, m = time.time(), 0, 0, 0.0
    for p in paths:
        st = os.stat(p)
        if now - st.st_mtime < 10:
            continue
        n, b, m = n + 1, b + st.st_size, max(m, st.st_mtime)
    return [n, b, m]


def tar_dir(src_dir, tar_path, base):
    """Pack src_dir as <base>/... into tar_path, only when its content signature changed."""
    files = sorted(p for p in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True) if os.path.isfile(p))
    sig = signature(files)
    sig_path = tar_path + ".sig.json"
    if os.path.exists(tar_path) and os.path.exists(sig_path) and json.load(open(sig_path)) == sig:
        return False
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    now = time.time()
    with tarfile.open(tar_path + ".tmp", "w") as tf:
        for p in files:
            if now - os.path.getmtime(p) >= 10:
                tf.add(p, arcname=os.path.join(base, os.path.relpath(p, src_dir)))
    os.replace(tar_path + ".tmp", tar_path)
    json.dump(sig, open(sig_path, "w"))
    return True


def stage_strategy():
    out = os.path.join(STAGE, "strategy")
    for f in DOCS:
        cp(os.path.join(ROOT, f), os.path.join(out, "docs", os.path.basename(f)))
    for f in CODE + CONFIGS:
        cp(os.path.join(ROOT, f), os.path.join(out, f))
    vend = os.path.join(ROOT, "code", "vendor", "ifeval")
    if os.path.isdir(vend):
        shutil.copytree(vend, os.path.join(out, "code", "vendor", "ifeval"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout
    os.makedirs(os.path.join(out, "env"), exist_ok=True)
    open(os.path.join(out, "code_changes.diff"), "w").write(git("diff"))
    open(os.path.join(out, "git_state.txt"), "w").write(
        f"base commit {git('rev-parse', 'HEAD').strip()} (origin github.com/NoviceCoderInfinity/diffusion-moe-general-expert)\n"
        f"tracked edits: code_changes.diff\nuntracked/new files (copied under strategy/code):\n{git('status', '--short')}")
    cp(ENV_KEEP, os.path.join(out, "env", "team_requirements_2026-09-09.txt"))
    frz = subprocess.run(["/venv/r3/bin/pip", "freeze"], capture_output=True, text=True).stdout
    open(os.path.join(out, "env", "venv_r3_freeze.txt"), "w").write(frz)


def stage_unprocessed(datagen_dirs):
    out = os.path.join(STAGE, "unprocessed")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "code", "compat")); sys.path.insert(0, os.path.join(ROOT, "code"))
    import research_protocol as P
    # GSM8K pool slices actually decoded (prompt as fed to the model + gold), one jsonl per slice
    for d in datagen_dirs:
        for split in sorted({os.path.basename(t) for t in glob.glob(os.path.join(d, "traces", "*"))}):
            if not split.startswith("gsm8k_pool"):
                continue
            dst = os.path.join(out, "gsm8k", split.replace(":", "_") + ".jsonl")
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as fh:
                    for it in P.items_for_split(split):
                        fh.write(json.dumps({k: it[k] for k in ("idx", "split", "prompt", "gold")}) + "\n")
    # OpenCodeInstruct: the frozen pool + the raw upstream rows it selected (all original columns)
    pool = os.path.join(ROOT, "code", "configs", "opencode_pool.json")
    if os.path.exists(pool):
        cp(pool, os.path.join(out, "opencode", "opencode_pool.json"))
        sha = hashlib.sha256(open(pool, "rb").read()).hexdigest()
        raw = os.path.join(out, "opencode", f"opencode_raw_rows_{sha[:12]}.parquet")
        if not os.path.exists(raw):
            import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
            spec = json.load(open(pool))
            items = spec.get("items", spec) if isinstance(spec, dict) else spec
            ids = pa.array(sorted({str(it["id"]) for it in items}))
            parts = []
            for f in sorted(glob.glob("/dev/shm/dlm/data/OpenCodeInstruct/data/*.parquet")):
                t = pq.read_table(f)
                parts.append(t.filter(pc.is_in(t["id"], value_set=ids)))
            pq.write_table(pa.concat_tables(parts), raw, compression="zstd")
    # verifiable IF: the frozen pool carries the raw prompt + ground-truth constraint spec per row
    for f in ("if_pool.json", "if_pool.json.sha256"):
        cp(os.path.join(ROOT, "code", "configs", f), os.path.join(out, "if", f))
    open(os.path.join(out, "SOURCES.md"), "w").write(SOURCES)


SOURCES = """# Sources of the round-3 subset

| slice | upstream | revision | licence | selection |
|---|---|---|---|---|
| maths | `openai/gsm8k` config `main`, split `train` | `740312add88f781978c0658806c59bc2815b9866` | MIT | `research_protocol.gsm8k_pool_items` (7,273 train rows minus the 200 frozen ranks), slices as decoded |
| code | `nvidia/OpenCodeInstruct` split `train` | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | CC-BY-4.0 | `code/draftalign/build_opencode_pool.py` -> `opencode_pool.json` (score==1.0, contamination drop vs MBPP-257/HumanEval-164, length fit, dedup) |
| IF | `allenai/IF_multi_constraints_upto5`, `allenai/RLVR-IFeval` | see `if_pool.json` | ODC-BY | `code/draftalign/build_if_pool.py` -> `if_pool.json` (program-verifiable constraints only; see strategy/docs/IF_RECIPE_2026-09-14.md) |
| IF (considered, not verifiable as-is) | `Arushhh/Llama-HybridDiffusion-processed-data-run1` / `Nemotron-IF-Domain-Cascade` | `903d9e9466737a17fb45227a4789ab8519a28b5a` | mixed source terms | used only if its constraints can be recovered at >=99% audited precision |
| models | `JetLM/SDAR-30B-A3B-Chat-b32` (30B MoE, block 32) and `JetLM/SDAR-1.7B-Chat-b32` (drafter) | `c351bbc37d240aa6871f167e8f92d694281b0c22` / `2c83c05d84be90a01c72443b2123ebf0376f6fd4` | upstream | stock, never trained by us |

Full upstream datasets are public at the revisions above and are referenced, not re-uploaded.
"""


def stage_processed():
    out = os.path.join(STAGE, "processed")
    status = {"datagen": {}, "trainset": {}}
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "logs", "draftalign", "datagen", "*")) if os.path.isdir(d))
    for d in dirs:
        name = os.path.basename(d)
        dst = os.path.join(out, "datagen", name)
        if name == "_failed":                                  # provenance of the aborted first launch: logs only
            for f in glob.glob(os.path.join(d, "*", "*.log")):
                cp(f, os.path.join(dst, os.path.relpath(f, d)))
            continue
        recs = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        for f in recs + glob.glob(os.path.join(d, "*.done")) + glob.glob(os.path.join(d, "shard*.log")):
            cp(f, os.path.join(dst, os.path.basename(f)))
        if os.path.isdir(os.path.join(d, "traces")):
            tar_dir(os.path.join(d, "traces"), os.path.join(dst, "traces.tar"), "traces")
        n = c = 0
        for f in recs:
            for line in open(f):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    n += 1; c += int(bool(r.get("correct")))
        status["datagen"][name] = {"records": n, "correct": c, "shards": len(recs),
                                   "shards_done": len(glob.glob(os.path.join(d, "*.done")))}
    for d in sorted(glob.glob(os.path.join(ROOT, "logs", "draftalign", "trainset_r3", "*"))):
        name = os.path.basename(d)
        dst = os.path.join(out, "trainset_r3", name)
        if os.path.isfile(d):
            cp(d, dst); continue
        for f in ("index.jsonl", "build_summary.json", "manifest.json"):
            cp(os.path.join(d, f), os.path.join(dst, f))
        if os.path.isdir(os.path.join(d, "items")):
            tar_dir(os.path.join(d, "items"), os.path.join(dst, "items.tar"), "items")
        bs = os.path.join(d, "build_summary.json")
        if os.path.exists(bs):
            status["trainset"][name] = json.load(open(bs)).get("summary", {})
    return dirs, status


def write_readme(status):
    rows = "\n".join(f"| `{k}` | {v['records']} | {v['correct']} | {v['shards_done']}/{v['shards']} |"
                     for k, v in status["datagen"].items()) or "| — | | | |"
    trows = "\n".join(f"| `{k}` | {v.get('kept_items', '')} | {v.get('usable_blocks', '')} | {json.dumps(v.get('dropped', {}))} |"
                      for k, v in status["trainset"].items()) or "| — (not built yet) | | | |"
    open(os.path.join(STAGE, "README.md"), "w").write(f"""# Nemotron-data-subset-training-diffusion-moe

Round-3 training data for the draft-aligned judge adapter on **JetLM/SDAR-30B-A3B-Chat-b32** (30B MoE, ~3B active,
**block 32**), with the stock **JetLM/SDAR-1.7B-Chat-b32** drafter. Recipe: `strategy/docs/TRAINING_RECIPE_2026-09-09.md`.
Code: github.com/NoviceCoderInfinity/diffusion-moe-general-expert (base commit + diff in `strategy/`).

Last sync: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}

## Target mix (blocks of 32 slots)

| domain | share | blocks | source |
|---|---:|---:|---|
| code | 35% | 11,200 | nvidia/OpenCodeInstruct (unit-test verified) |
| maths | 35% | 11,200 | openai/gsm8k train pool (GSM8K only) |
| instruction following | 30% | 9,600 | program-verifiable IF prompts (strict all-constraints check) |

## Pipeline (DATA.md §4, unchanged)

1. **Decode once** with the training-free blind rule: stock 30B + stock 1.7B drafter, `--draft s8 --judge blind --tau-acc-q .50`,
   block 32, threshold 0.95, greedy; the model's own answer is scored inline (`--score`).
2. **Build blocks**: `build_train_set.py --only-correct --max-seq 1200 --exclude-frozen` — only answers the 30B got right
   (GSM8K exact answer, code passes its unit tests, IF passes every constraint) become 32-token training blocks.
3. **Mix**: per-pool block caps to 35/35/30.

## Layout

- `unprocessed/` — the exact source rows selected, plus `SOURCES.md` (ids, revisions, licences)
- `processed/datagen/<cell>/` — per-item decode records (`*.jsonl`), completion markers, shard logs, `traces.tar` (per-block NPZ traces)
- `processed/trainset_r3/<pool>/` — `index.jsonl`, `build_summary.json`, `items.tar` (training states)
- `strategy/` — recipes and docs, pipeline code as run, `code_changes.diff`, environment freezes

## Status — decode

| cell dir | records | correct | shards done |
|---|---:|---:|---:|
{rows}

## Status — training sets

| pool | kept items | usable blocks | dropped |
|---|---:|---:|---|
{trows}

## Restore

```bash
hf download {REPO} --repo-type dataset --include "{FOLDER}/*" --local-dir .
tar -xf {FOLDER}/processed/datagen/<cell>/traces.tar -C <cell>/      # -> <cell>/traces/<split>/<idx>.npz
tar -xf {FOLDER}/processed/trainset_r3/<pool>/items.tar -C <pool>/   # -> <pool>/items/*.npz (index.jsonl paths are relative)
```
""")


def secret_scan():
    needles = [os.environ.get(k, "") for k in ("HF_TOKEN", "WANDB_API_KEY")]
    needles = [n.encode() for n in needles if len(n) >= 16]
    pat = re.compile(rb"hf_[A-Za-z0-9]{30,}")
    for p in glob.glob(os.path.join(STAGE, "**", "*"), recursive=True):
        if not os.path.isfile(p) or p.endswith(".sig.json"):
            continue
        with open(p, "rb") as fh:
            blob = fh.read()
        if any(n in blob for n in needles) or pat.search(blob):
            raise SystemExit(f"[backup] ABORT: secret-like content in {p}")


def sync(repo):
    load_env()
    os.environ["HF_HUB_OFFLINE"] = "0"
    t0 = time.time()
    dirs, status = stage_processed()
    stage_unprocessed(dirs)
    stage_strategy()
    write_readme(status)
    secret_scan()
    from huggingface_hub import HfApi
    os.environ.pop("HF_TOKEN", None)   # .env holds a different account; use the logged-in HF account
    api = HfApi()
    assert api.whoami()["name"] == "Anupam-Rawat-IITB", "not logged in as Anupam-Rawat-IITB"
    info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=STAGE, path_in_repo=FOLDER,
                             ignore_patterns=["*.sig.json", "*.tmp"],
                             commit_message=f"r3 data backup {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}")
    print(f"[backup] {time.strftime('%FT%TZ', time.gmtime())} synced in {time.time() - t0:.0f}s -> {info}", flush=True)
    print(json.dumps(status["datagen"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--loop", type=float, default=0, help="minutes between syncs (0 = once)")
    args = ap.parse_args()
    while True:
        try:
            sync(args.repo)
        except SystemExit:
            raise
        except Exception as e:                                          # keep looping on transient hub errors
            print(f"[backup] {time.strftime('%FT%TZ', time.gmtime())} sync failed: {e!r}", flush=True)
        if not args.loop:
            break
        time.sleep(60 * args.loop)


if __name__ == "__main__":
    main()
