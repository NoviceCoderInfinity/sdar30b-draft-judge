#!/usr/bin/env python
"""Back up phase 4's outputs to the Hugging Face dataset $P4_HF_REPO (default Arushhh/phase4-drafter-lora).

    set -a; . ./.env; set +a; /venv/p2/bin/python code/drafter_lora/backup_p4.py [--dry-run] [--loop 30]

An explicit allowlist only (adapter checkpoints, the merged drafter, held-out pid list, gate JSONs, Gate 1 drafts,
decode records, small logs).  Before every upload each selected file up to --scan-max-mb is scanned for the VALUES
of HF_TOKEN / WANDB_API_KEY / GITHUB_TOKEN (from the environment / .env) and for hf_[A-Za-z0-9]{30,}; any hit refuses
the upload and exits with code 2 (ending --loop).  Values are never printed.  The repo is created private unless
P4_HF_PRIVATE=0.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
import time
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALLOW = [
    "logs/phase4/train/*/config.json", "logs/phase4/train/*/model_info.json", "logs/phase4/train/*/holdout_pids.json",
    "logs/phase4/train/*/train.rank0.jsonl", "logs/phase4/train/*/speed.json", "logs/phase4/train/*/check_25pct.json",
    "logs/phase4/train/*/STOP.json", "logs/phase4/train/*/DONE.json", "logs/phase4/train/*.log",
    "logs/phase4/train/main/ckpt/*/adapter*", "logs/phase4/train/main/ckpt/*/trainer_state.json",
    "logs/phase4/train/main/ckpt/latest", "logs/phase4/train/main/final_adapter/*", "logs/phase4/train/main/merged/*",
    "logs/phase4/smoke/*.json", "logs/phase4/smoke/*.log", "logs/phase4/smoke/parity/*.json",
    "logs/phase4/smoke/train/*.json", "logs/phase4/smoke/train/*.jsonl", "logs/phase4/smoke/speedprobe/*.json",
    "logs/phase4/gate1/shards/*.json", "logs/phase4/gate1/drafts/*.npz", "logs/phase4/gate1/logs/*.log",
    "logs/phase4/gate2.log", "logs/phase4/decode/**/*.jsonl", "logs/phase4/decode/**/*.jsonl.done",
    "logs/phase4/decode/**/*.log", "logs/phase4/p4.log", "logs/phase4/events.tsv",
    "results/phase4_drafter/**", "STATUS.md",
    "logs/phase4/**",                    # operator 2026-10-12: put EVERYTHING of phase 4 on HF (logs, ckpts, optimizer)
]
DENY = [".env", "*.env", "**/.env"]                             # (was also "*.pt": operator asked for everything)
TOKEN_RE = re.compile(rb"hf_[A-Za-z0-9]{30,}")
SECRET_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN")
# A .wandb file written in the last LIVE_S seconds belongs to a live W&B run: it can grow between hashing and upload, and
# the Hub then rejects the WHOLE commit ("an LFS pointer pointed to a file that does not exist").  It is uploaded by the
# first pass after its run ends.
LIVE_S = 900


def read_env() -> dict:
    env = {}
    for path in (os.path.join(ROOT, ".env"), "/workspace/.env"):
        if os.path.isfile(path):
            for line in open(path):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    env.update({k: os.environ[k] for k in SECRET_KEYS if os.environ.get(k)})
    return env


def _match(rel: str, pats) -> bool:
    return any(fnmatch.fnmatch(rel, p) or (p.endswith("/**") and rel.startswith(p[:-3] + "/")) for p in pats)


def selected() -> List[str]:
    out = []
    for top in ("logs/phase4", "results/phase4_drafter", "."):
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base):
            continue
        walk = os.walk(base) if top != "." else [(ROOT, [], os.listdir(ROOT))]
        for d, _dirs, files in walk:
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), ROOT)
                p = os.path.join(ROOT, rel)
                if _match(rel, ALLOW) and not _match(rel, DENY) and os.path.isfile(p) and not (
                        rel.endswith(".wandb") and time.time() - os.path.getmtime(p) < LIVE_S):
                    out.append(rel)
    return sorted(set(out))


def scan(files: List[str], secrets: List[bytes], max_mb: float) -> List[str]:
    hits = []
    for rel in files:
        p = os.path.join(ROOT, rel)
        if os.path.getsize(p) > max_mb * 2 ** 20:
            continue
        data = open(p, "rb").read()
        if TOKEN_RE.search(data) or any(s and s in data for s in secrets):
            hits.append(rel)
    return hits


def once(args) -> int:
    env = read_env()
    secrets = [env[k].encode() for k in SECRET_KEYS if env.get(k) and len(env[k]) >= 8]
    files = selected()
    hits = scan(files, secrets, args.scan_max_mb)
    if hits:
        print(f"[backup] REFUSED: {len(hits)} file(s) contain a secret or an hf_ token: {hits[:10]}", flush=True)
        return 2
    repo = os.environ.get("P4_HF_REPO") or env.get("P4_HF_REPO") or "Arushhh/phase4-drafter-lora"
    size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
    print(f"[backup] {len(files)} files, {size / 2 ** 30:.2f} GiB -> {repo} (scan clean)", flush=True)
    if args.dry_run:
        for f in files[:50]:
            print("  ", f)
        return 0
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi
    api = HfApi(token=env.get("HF_TOKEN"))
    private = (os.environ.get("P4_HF_PRIVATE") or env.get("P4_HF_PRIVATE") or "1") != "0"
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=repo, repo_type="dataset", allow_patterns=files,
                      commit_message=f"phase 4 backup {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
                      + (f" ({args.tag})" if args.tag else ""))
    print("[backup] uploaded", flush=True)
    return 0


def scan_staged() -> int:
    """`git diff --cached` scanned like an upload: 0 = clean, 2 = a secret value or an hf_ token is staged."""
    import subprocess
    env = read_env()
    secrets = [env[k].encode() for k in SECRET_KEYS if env.get(k) and len(env[k]) >= 8]
    diff = subprocess.run(["git", "diff", "--cached"], cwd=ROOT, capture_output=True, check=True).stdout
    names = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True,
                           check=True).stdout.decode().split()
    bad = [n for n in names if _match(n, DENY)]
    hit = bool(TOKEN_RE.search(diff) or any(x in diff for x in secrets))
    if hit or bad:
        print(f"[scan] REFUSE: staged diff holds a secret or an hf_ token ({hit}), or a denied file ({bad})", flush=True)
        return 2
    print(f"[scan] staged diff clean ({len(names)} files)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-staged", action="store_true", help="scan `git diff --cached` before a push; exit 2 on a hit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop", type=float, default=0, help="minutes between backups (0 = once)")
    ap.add_argument("--scan-max-mb", type=float, default=512)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.scan_staged:
        return scan_staged()
    while True:
        try:
            rc = once(args)
        except Exception as e:                       # a transient Hub error is logged, not fatal to --loop
            print(f"[backup] error: {type(e).__name__}: {e}", flush=True)
            rc = 1
        if rc == 2 or not args.loop:
            return rc
        time.sleep(60 * args.loop)


if __name__ == "__main__":
    sys.exit(main())
