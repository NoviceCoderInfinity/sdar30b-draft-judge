#!/usr/bin/env python
"""Back up phase 5's outputs to the PUBLIC Hugging Face dataset $P5_HF_REPO (default Arushhh/phase5-loop-guard).
Copy of code/drafter_lora/backup_p4.py (operator, 23 Oct 2026) with the Phase 5 allowlist and the same secret scan.

    set -a; . /workspace/.env; set +a; /venv/p2/bin/python code/judge_round/backup_p5.py [--dry-run] [--loop 15]

An explicit allowlist only: logs/phase5 and the fresh-data folders (logs/c1b/gen/waveC, logs/c1b/draft/waveC,
logs/c1b/trainset/waveC, logs/c1b/trainset/fresh, logs/c1b/fresh, data/c1b/prompt_lists_fresh), plus results/ and
STATUS.md.  Before every upload each selected file up to --scan-max-mb is scanned for the VALUES of HF_TOKEN /
WANDB_API_KEY / GITHUB_TOKEN (from the environment / .env) and for hf_[A-Za-z0-9]{30,}; any hit refuses the upload and
exits with code 2 (ending --loop).  Values are never printed.  The repo is created PUBLIC unless P5_HF_PRIVATE=1.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import time
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOPS = ["logs/phase5", "logs/c1b/gen/waveC", "logs/c1b/draft/waveC", "logs/c1b/trainset/waveC", "logs/c1b/trainset/fresh",
        "logs/c1b/fresh", "data/c1b/prompt_lists_fresh", "results"]
ALLOW = [t + "/**" for t in TOPS] + ["STATUS.md", "logs/c1b/fresh_targets.json"]
DENY = [".env", "*.env", "**/.env", "**/*.pt"]      # no optimizer states here: they are not phase 5 outputs
# Transient side/trace dirs the judge/drafter engines create and DELETE during a run (judge_sglang.py): a file listed
# for upload can vanish before it is read ("... is not a file", 23 Oct 20:16 IST rc=1). Excluded in selected() by PATH
# (not via DENY, so the staged-diff secret scan never flags them as "denied" and stops the loop — 20:48 IST).
TRANSIENT = ("/judge_side/", "/judge_trace/", "/drafter_side/", "/drafter_trace/")
TOKEN_RE = re.compile(rb"hf_[A-Za-z0-9]{30,}")
SECRET_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN")
LIVE_S = 900   # a .wandb file written in the last LIVE_S seconds belongs to a live run (see backup_p4.py)
# Any LARGE file (> LFS_MB, stored as LFS by the Hub) still being written (modified in the last LIVE_S seconds) is skipped
# for the same reason: the SGLang trace.*.jsonl files grow during the upload and the Hub then rejects the whole commit
# ("an LFS pointer pointed to a file that does not exist", 23 Oct 2026 15:17 IST). They are uploaded once they stop growing.
LFS_MB = 10
# Directories with very many files (e.g. the ~150k packed training items) are uploaded as ONE <dir>.tar, not as loose
# files: the Hub rate-limits at 1000 API requests / 5 min and one loose-file upload makes ~1 request per file, so 150k
# files stall the whole commit (23 Oct 2026 18:05 IST, --tag loop stuck 85 min / HTTP 429). The tar is rebuilt only when
# the directory is newer than it, so a static dir is tarred once and re-uploaded unchanged (the Hub then skips its bytes).
TAR_DIRS = ["logs/c1b/trainset/waveC/items"]
DENY += [d + "/**" for d in TAR_DIRS]        # the loose files are excluded; the sibling <dir>.tar is uploaded instead


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


def ensure_tars() -> None:
    """For each TAR_DIRS entry build <dir>.tar (beside the dir), but only when the dir is newer than the tar, so a
    static dir is tarred once and the same bytes re-upload (the Hub dedups them). Reproducible enough: same content,
    same path. The loose files are excluded by the DENY patterns added above."""
    for rel in TAR_DIRS:
        src = os.path.join(ROOT, rel)
        if not os.path.isdir(src):
            continue
        tar = src.rstrip("/") + ".tar"
        newest = max((os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(src) for f in fs), default=0.0)
        if os.path.exists(tar) and os.path.getmtime(tar) >= newest:
            continue
        tmp = tar + ".tmp"
        subprocess.run(["tar", "--sort=name", "-cf", tmp, "-C", os.path.dirname(src), os.path.basename(src)], check=True)
        os.replace(tmp, tar)
        print(f"[backup] tarred {rel} -> {os.path.relpath(tar, ROOT)} ({os.path.getsize(tar) / 2 ** 20:.0f} MiB)", flush=True)


def selected() -> List[str]:
    out = []
    for top in TOPS + ["."]:
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base):
            continue
        walk = os.walk(base) if top != "." else [(ROOT, [], os.listdir(ROOT))]
        for d, _dirs, files in walk:
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), ROOT)
                p = os.path.join(ROOT, rel)
                if any(t in "/" + rel for t in TRANSIENT):
                    continue                                 # transient judge/drafter side & trace files (deleted mid-run)
                # DENY includes **/*.pt, but the NEW judge checkpoints' trainer_state.pt (optimizer state) MUST reach HF
                # so a box loss leaves training resumable; every other *.pt stays denied. Secret scan still runs on it.
                denied = _match(rel, DENY) and not (rel.endswith("trainer_state.pt") and rel.startswith("logs/phase5/judge/"))
                # the LFS-fresh skip (large file modified in the last LIVE_S s) is for files that GROW during upload;
                # judge checkpoints under logs/phase5/judge/ are written atomically (save_ckpt: tmp dir then os.replace),
                # so they are complete the instant they appear and must NOT be delayed — exempt them from the fresh skip.
                fresh_big = (os.path.getsize(p) > LFS_MB * 2 ** 20 and time.time() - os.path.getmtime(p) < LIVE_S
                             and not rel.startswith("logs/phase5/judge/"))
                if _match(rel, ALLOW) and not denied and os.path.isfile(p) and not os.path.islink(p) and not (
                        rel.endswith(".wandb") and time.time() - os.path.getmtime(p) < LIVE_S) and not fresh_big:
                    out.append(rel)
    # logs/c1b/fresh_targets.json sits beside the fresh/ folder, not inside a TOP
    extra = os.path.join(ROOT, "logs/c1b/fresh_targets.json")
    if os.path.isfile(extra):
        out.append("logs/c1b/fresh_targets.json")
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
    ensure_tars()
    files = selected()
    hits = scan(files, secrets, args.scan_max_mb)
    if hits:
        print(f"[backup] REFUSED: {len(hits)} file(s) contain a secret or an hf_ token: {hits[:10]}", flush=True)
        return 2
    repo = os.environ.get("P5_HF_REPO") or env.get("P5_HF_REPO") or "Arushhh/phase5-loop-guard"
    size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
    print(f"[backup] {len(files)} files, {size / 2 ** 30:.2f} GiB -> {repo} (scan clean)", flush=True)
    if args.dry_run:
        for f in files[:60]:
            print("  ", f)
        return 0
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi
    api = HfApi(token=env.get("HF_TOKEN"))
    private = (os.environ.get("P5_HF_PRIVATE") or env.get("P5_HF_PRIVATE") or "0") == "1"
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=repo, repo_type="dataset", allow_patterns=files,
                      commit_message=f"phase 5 backup {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
                      + (f" ({args.tag})" if args.tag else ""))
    print("[backup] uploaded", flush=True)
    return 0


def scan_staged(git_dir: str = ROOT) -> int:
    """`git diff --cached` (in git_dir, default the repo root) scanned like an upload: 0 = clean, 2 = a secret value or
    an hf_ token is staged."""
    import subprocess
    env = read_env()
    secrets = [env[k].encode() for k in SECRET_KEYS if env.get(k) and len(env[k]) >= 8]
    diff = subprocess.run(["git", "diff", "--cached"], cwd=git_dir, capture_output=True, check=True).stdout
    names = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=git_dir, capture_output=True,
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
    ap.add_argument("--git-dir", default=ROOT, help="with --scan-staged: the checkout to scan (e.g. the phase5-logs worktree)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop", type=float, default=0, help="minutes between backups (0 = once)")
    ap.add_argument("--scan-max-mb", type=float, default=512)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.scan_staged:
        return scan_staged(args.git_dir)
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
