#!/usr/bin/env python
"""Back up Phase 2's outputs (logs/phase2) to a Hugging Face dataset repo, once or every N minutes.

The box's disk goes away with the rental; this is the durable copy. What is staged (under /dev/shm/p2_backup_stage,
never inside the repository):
    every *.json / *.jsonl / *.md / *.log / *.txt under logs/phase2 (records, running metrics, stop events, gates,
    probes, summaries, pipeline and job logs), and the plots' *.png under logs/phase2/plots (plot_p2.py)
    one .tar per many-file directory: decode traces/, per-item experts/, re-drafted items/ (rebuilt only when the
    directory's content signature changes)
Not staged: logs/phase2/data (the round-3 training set: already on HF), logs/phase2/ref (downloaded references),
logs/phase2/_hf (download cache), checkpoints.

Before every upload the stage is scanned for the .env values of HF_TOKEN / WANDB_API_KEY / GITHUB_TOKEN and for
`hf_[A-Za-z0-9]{30,}`; a hit aborts (and ends --loop), exactly like round 3's backup (AGENTS.md 9(r)).

Repo: --repo, else $P2_HF_REPO, else <token owner>/phase2-drafter-swap (created PRIVATE if missing), path prefix
`phase2/`.

    /venv/p2/bin/python code/phase2/backup_p2.py              # one sync
    nohup /venv/p2/bin/python code/phase2/backup_p2.py --loop 30 > logs/phase2/hf_backup.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "logs", "phase2")
STAGE = "/dev/shm/p2_backup_stage"
TEXT_EXT = (".json", ".jsonl", ".md", ".log", ".txt", ".done", ".stopped")
PLOT_DIR, PLOT_EXT = "plots", (".png",)                    # plot_p2.py's PNGs are staged too
TAR_DIRS = ("traces", "experts", "judge", "items")
SKIP_TOP = ("data", "ref", "_hf")
TOKEN_RE = re.compile(rb"hf_[A-Za-z0-9]{30,}")


def env_secrets() -> list:
    vals = []
    for f in (os.path.join(ROOT, ".env"), os.path.expanduser("~/.env")):
        if os.path.exists(f):
            for line in open(f):
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    if k.strip() in ("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN") and len(v.strip().strip('"\'')) >= 8:
                        vals.append(v.strip().strip('"\'').encode())
    for k in ("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN"):
        v = os.environ.get(k)
        if v and len(v) >= 8:
            vals.append(v.encode())
    return list(set(vals))


def dir_signature(d: str) -> str:
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            h.update(f"{os.path.relpath(p, d)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def stage() -> dict:
    os.makedirs(STAGE, exist_ok=True)
    n_text = n_tar = 0
    now = time.time()
    for root, dirs, files in os.walk(SRC):
        rel = os.path.relpath(root, SRC)
        top = rel.split(os.sep)[0]
        if top in SKIP_TOP:
            dirs[:] = []
            continue
        for d in list(dirs):
            if d in TAR_DIRS:
                dirs.remove(d)
                src = os.path.join(root, d)
                dst = os.path.join(STAGE, rel, d + ".tar")
                sig = dir_signature(src)
                sigf = dst + ".sig"
                if os.path.exists(dst) and os.path.exists(sigf) and open(sigf).read() == sig:
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tarfile.open(dst + ".tmp", "w") as tf:
                    tf.add(src, arcname=d)
                os.replace(dst + ".tmp", dst)
                open(sigf, "w").write(sig)
                n_tar += 1
        for f in files:
            if not (f.endswith(TEXT_EXT) or (top in (PLOT_DIR, "analysis") and f.endswith(PLOT_EXT))
                    or (top == "analysis" and f.endswith(".py"))) or f == "STOP":
                continue
            p = os.path.join(root, f)
            try:
                if now - os.stat(p).st_mtime < 5:              # being written: next sync
                    continue
            except OSError:
                continue
            dst = os.path.join(STAGE, rel, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
            n_text += 1
    return {"text_files": n_text, "tars_rebuilt": n_tar}


def scan(secrets: list) -> list:
    hits = []
    for root, _, files in os.walk(STAGE):
        for f in files:
            if f.endswith(".sig"):
                continue
            p = os.path.join(root, f)
            if f.endswith(".tar"):
                continue                                          # tars hold npz arrays only (traces, experts, items)
            b = open(p, "rb").read()
            if TOKEN_RE.search(b) or any(s in b for s in secrets):
                hits.append(os.path.relpath(p, STAGE))
    return hits


def resolve_repo(api, repo: str) -> str:
    if repo:
        return repo
    who = api.whoami()["name"]
    return f"{who}/phase2-drafter-swap"


def sync(repo_arg: str, private: bool) -> str:
    from huggingface_hub import HfApi
    os.environ["HF_HUB_OFFLINE"] = "0"
    api = HfApi()
    repo = resolve_repo(api, repo_arg or os.environ.get("P2_HF_REPO", ""))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    st = stage()
    hits = scan(env_secrets())
    if hits:
        raise SystemExit(f"[backup] SECRET FOUND in {hits[:5]}: nothing uploaded, backup stopped")
    readme = os.path.join(STAGE, "README.md")
    with open(readme, "w") as fh:
        fh.write("# Phase 2 (drafter swap) outputs\n\nBranch `phase2-drafter-swap`, docs/PHASE2_DRAFTER_SWAP.md.\n"
                 f"Last sync: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    api.upload_folder(folder_path=STAGE, repo_id=repo, repo_type="dataset", path_in_repo="phase2",
                      ignore_patterns=["*.sig", "*.tmp"],
                      commit_message=f"phase 2 backup {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}")
    print(f"[backup] {time.strftime('%H:%M:%S UTC', time.gmtime())} -> https://huggingface.co/datasets/{repo} "
          f"({st['text_files']} text files, {st['tars_rebuilt']} tars rebuilt)", flush=True)
    return repo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--loop", type=float, default=0, help="minutes between syncs (0 = once)")
    args = ap.parse_args(argv)
    if not os.path.isdir(SRC):
        print(f"[backup] nothing to back up: {SRC} missing", flush=True)
        return 0
    while True:
        try:
            sync(args.repo, private=not args.public)
        except SystemExit:
            raise
        except Exception as e:                                     # a transient Hub error: log, retry next cycle
            print(f"[backup] {time.strftime('%H:%M:%S UTC', time.gmtime())} upload failed: {e!r}", flush=True)
            if not args.loop:
                return 1
        if not args.loop:
            return 0
        time.sleep(60 * args.loop)


if __name__ == "__main__":
    sys.exit(main())
