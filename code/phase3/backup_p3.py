#!/usr/bin/env python
"""Phase 3 backup: logs, results and the code that ran -> public Hugging Face dataset; results -> GitHub branch.

    /venv/p2/bin/python code/phase3/backup_p3.py [--final]

Hugging Face (needs HF_TOKEN): repo P3_HF_REPO (default Arushhh/phase3-redraft), public unless P3_HF_PRIVATE=1.
  Uploads logs/phase3 (decode records, job logs, setup logs and checks, analysis, run log; never the /dev/shm sandbox),
  results/phase3, and a snapshot of the code that ran (code/phase3, the decode files, the new test).
GitHub (needs GITHUB_TOKEN): commits results/phase3 to the current branch (phase3-redraft) and pushes to
  P3_GH_REPO (default NoviceCoderInfinity/diffusion-moe-general-expert).
Before every upload a secret scan looks for the HF / GitHub token values and hf_ / ghp_ / github_pat_ strings; a hit
stops the upload (exit 3) and names only the file. Tokens are never printed.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
PATTERNS = [re.compile(rb"hf_[A-Za-z0-9]{30,}"), re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
            re.compile(rb"github_pat_[A-Za-z0-9_]{30,}")]
CODE_FILES = ["code/draftalign/da_decode.py", "code/draftalign/drafter.py", "code/draftalign/run_da_decode.py",
              "code/tests/test_da_decode_redraft.py"]


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def scan(root):
    """Files under `root` that contain a token value or a token-shaped string."""
    secrets = [os.environ[k].encode() for k in ("HF_TOKEN", "GITHUB_TOKEN") if len(os.environ.get(k, "")) >= 8]
    hits = []
    for d, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(d, fn)
            try:
                if os.path.getsize(p) > 512 * 2 ** 20:
                    continue
                b = open(p, "rb").read()
            except OSError:
                continue
            if any(s in b for s in secrets) or any(pt.search(b) for pt in PATTERNS):
                hits.append(p)
    return hits


def hf_backup(final):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[backup] no HF_TOKEN: Hugging Face upload skipped"); return
    from huggingface_hub import HfApi
    repo = os.environ.get("P3_HF_REPO", "Arushhh/phase3-redraft")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=os.environ.get("P3_HF_PRIVATE") == "1", exist_ok=True)
    stage = "/dev/shm/p3_stage"
    shutil.rmtree(stage, ignore_errors=True)
    if os.path.isdir("logs/phase3"):
        shutil.copytree("logs/phase3", os.path.join(stage, "logs/phase3"))
    if os.path.isdir("results/phase3"):
        shutil.copytree("results/phase3", os.path.join(stage, "results/phase3"))
    shutil.copytree("code/phase3", os.path.join(stage, "code/phase3"), ignore=shutil.ignore_patterns("__pycache__"))
    for f in CODE_FILES:
        os.makedirs(os.path.dirname(os.path.join(stage, f)), exist_ok=True)
        shutil.copy(f, os.path.join(stage, f))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(stage, "README.md"), "w") as fh:
        fh.write("# Phase 3: re-draft between judge passes\n\n"
                 "SDAR-1.7B drafter + Phase 1's 128M LoRA judge on JetLM/SDAR-30B-A3B-Chat-b32 (block 32, threshold 0.95, "
                 "no expert pool), re-drafting the [MASK] slots from committed tokens after every judge pass (rd1) or every "
                 "2 passes (rd2), on 7 benchmarks with one 8x RTX PRO 6000 box.\n\n"
                 f"- code: github.com/NoviceCoderInfinity/diffusion-moe-general-expert, branch phase3-redraft, commit {head}\n"
                 "- results/phase3/TABLE.md: the tables; metrics.json: every number\n"
                 "- logs/phase3/decode/<setup>/<benchmark>/: per-item decode records; jobs/: shard logs; setup/: setup and checks\n"
                 f"- last sync: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{' (final)' if final else ''}\n")
    hits = scan(stage)
    if hits:
        print(f"[backup] SECRET SCAN HIT, nothing uploaded: {hits}"); sys.exit(3)
    api.upload_folder(folder_path=stage, repo_id=repo, repo_type="dataset",
                      commit_message=f"phase 3 {'final ' if final else ''}backup {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    print(f"[backup] Hugging Face: uploaded to datasets/{repo}")


def gh_backup(final):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[backup] no GITHUB_TOKEN: GitHub push skipped"); return
    if not os.path.isdir("results/phase3"):
        print("[backup] no results/phase3 yet: GitHub push skipped"); return
    hits = scan("results/phase3")
    if hits:
        print(f"[backup] SECRET SCAN HIT, nothing pushed: {hits}"); sys.exit(3)
    git = ["git", "-c", f"user.name={os.environ.get('P3_GIT_NAME', 'Arush777')}",
           "-c", f"user.email={os.environ.get('P3_GIT_EMAIL', '23je0154@iitism.ac.in')}"]
    subprocess.run(git + ["add", "results/phase3"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("[backup] GitHub: nothing new to commit")
    else:
        msg = f"phase 3: {'final ' if final else ''}results {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
        subprocess.run(git + ["commit", "-q", "-m", msg], check=True)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    repo = os.environ.get("P3_GH_REPO", "NoviceCoderInfinity/diffusion-moe-general-expert")
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    p = subprocess.run(["git", "push", url, f"HEAD:{branch}"], capture_output=True, text=True)
    print("[backup] GitHub push:", (p.stdout + p.stderr).replace(token, "***").strip() or "ok")
    if p.returncode:
        raise SystemExit(f"git push failed ({p.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    load_env()
    print(f"[backup] {time.strftime('%H:%M:%S UTC', time.gmtime())}{' final' if args.final else ''}")
    err = None
    for fn in (hf_backup, gh_backup):
        try:
            fn(args.final)
        except SystemExit as e:
            if e.code == 3:
                raise
            err = e; print(f"[backup] {fn.__name__} failed: {e}")
        except Exception as e:                                   # keep the other backup going
            err = e; print(f"[backup] {fn.__name__} failed: {type(e).__name__}: {e}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
