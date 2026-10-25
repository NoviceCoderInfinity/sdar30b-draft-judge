#!/usr/bin/env python
"""8-GPU Phase 3 backup: GitHub branch phase3-redraft-8gpu + HF dataset Arushhh/phase3-redraft (same paths).

    /venv/p2/bin/python code/phase3_8gpu/backup_8gpu.py [--tag TEXT]

GitHub: commits logs/phase3_8gpu, results/phase3_8gpu and code/phase3_8gpu (force-added: logs/ and *.jsonl are
gitignored) to the current branch, which must be phase3-redraft-8gpu, pushes it, and checks that the remote branch head
equals the local commit. Files over 95 MB stay out of git (Hugging Face only).
Hugging Face: uploads logs/phase3_8gpu and results/phase3_8gpu (copied to /dev/shm first, so a file being appended is
read once) to the same paths of the dataset, in one commit that never deletes anything there; then checks that every
copied file is on the Hub with the same size.
Before either upload a secret scan looks for the token values (environment and /workspace/.env) and for hf_ / ghp_ /
github_pat_ strings in everything to be uploaded; a hit stops the backup (exit 3) and names only the file. Tokens are
never printed. One backup at a time (flock). Every backup appends start / end / status to logs/phase3_8gpu/bg_jobs.tsv,
so the analysis can check whether any decode item overlapped a backup.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
BRANCH = "phase3-redraft-8gpu"
HF_REPO = os.environ.get("P3G_HF_REPO", "Arushhh/phase3-redraft")
GIT_PATHS = ["logs/phase3_8gpu", "results/phase3_8gpu", "code/phase3_8gpu",
             "code/draftalign/run_da_decode.py", "code/draftalign/gpu_meter.py"]   # the decode code that ran (GPU-time patch)
CODE_SNAPSHOT = ["code/phase3_8gpu", "code/draftalign/run_da_decode.py", "code/draftalign/gpu_meter.py",
                 "code/draftalign/da_decode.py", "code/draftalign/seqcache.py"]       # copied to logs/phase3_8gpu/code_snapshot on HF
HF_PATHS = ["logs/phase3_8gpu", "results/phase3_8gpu"]
STAGE = "/dev/shm/p3g_stage"
LOCK = "/dev/shm/p3g_backup.lock"
JOBS = "logs/phase3_8gpu/bg_jobs.tsv"
STATUS = "logs/phase3_8gpu/backup_status.json"
PATTERNS = [re.compile(rb"hf_[A-Za-z0-9]{30,}"), re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
            re.compile(rb"github_pat_[A-Za-z0-9_]{30,}")]
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01SV2MwpQfwFFB5q7DoTR8qR")


def say(msg):
    print(f"[backup {time.strftime('%H:%M:%S', time.gmtime())} UTC] {mask(msg)}", flush=True)


def mask(s):
    s = str(s)
    for k in ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY"):
        v = os.environ.get(k, "")
        if len(v) >= 8:
            s = s.replace(v, "***")
    return re.sub(r"(hf|ghp|github_pat)_[A-Za-z0-9_]{8,}", "***", s)


def load_env():
    for p in (os.path.join(ROOT, ".env"), "/workspace/.env"):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def scan(paths):
    secrets = [os.environ[k].encode() for k in ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
               if len(os.environ.get(k, "")) >= 8]
    hits = []
    for root in paths:
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


def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {mask(r.stderr.strip()[-400:])}")
    return r.stdout.strip()


def github(tag):
    br = git("rev-parse", "--abbrev-ref", "HEAD")
    assert br == BRANCH, f"checkout is on {br}, not {BRANCH}: refusing to commit"
    paths = [p for p in GIT_PATHS if os.path.exists(p)]
    git("add", "-f", "--", *paths)
    big = [p for p in git("diff", "--cached", "--name-only").splitlines()
           if os.path.exists(p) and os.path.getsize(p) > 95 * 2 ** 20]
    for p in big:
        git("rm", "--cached", "-q", "--", p)
        say(f"kept out of git (> 95 MB, Hugging Face only): {p}")
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        n = len(git("diff", "--cached", "--name-only").splitlines())
        git("commit", "-q", "-m", f"phase 3 8gpu: backup {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC ({tag})" + TRAILER)
        say(f"github: committed {n} files")
    else:
        say("github: nothing new to commit")
    for attempt, wait_s in enumerate((30, 60, 0), 1):
        try:
            git("push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")
            break
        except RuntimeError as e:
            if not wait_s:
                raise
            say(f"github push attempt {attempt} failed ({e}); retrying in {wait_s}s")
            time.sleep(wait_s)
    local = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    assert remote == local, f"push did not land: remote {remote[:9]} != local {local[:9]}"
    say(f"github: {BRANCH} at {local[:9]} (remote head verified)")


def huggingface(tag):
    if not os.environ.get("HF_TOKEN"):
        say("huggingface: no HF_TOKEN, skipped")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    shutil.rmtree(STAGE, ignore_errors=True)
    want = {}
    for p in HF_PATHS:
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(STAGE, p), ignore=shutil.ignore_patterns("*.lock", "__pycache__"))
    snap = os.path.join(STAGE, "logs/phase3_8gpu/code_snapshot")
    for c in CODE_SNAPSHOT:
        if os.path.isdir(c):
            shutil.copytree(c, os.path.join(snap, c), ignore=shutil.ignore_patterns("__pycache__", "*.new"), dirs_exist_ok=True)
        elif os.path.exists(c):
            os.makedirs(os.path.dirname(os.path.join(snap, c)), exist_ok=True)
            shutil.copy(c, os.path.join(snap, c))
    for d, _, files in os.walk(STAGE):
        for fn in files:
            full = os.path.join(d, fn)
            want[os.path.relpath(full, STAGE)] = os.path.getsize(full)
    if not want:
        say("huggingface: nothing to upload")
        return
    hits = scan([STAGE])
    assert not hits, f"secret scan hit in the staged copy: {hits}"
    for attempt, wait_s in enumerate((20, 40, 80, 160, 0), 1):
        try:
            info = api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=STAGE, path_in_repo="",
                                     commit_message=f"phase 3 8gpu: backup {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC ({tag})")
            break
        except Exception as e:                       # noqa: BLE001 (network / TLS / server errors: retry)
            if not wait_s:
                raise
            say(f"huggingface upload attempt {attempt} failed ({type(e).__name__}: {str(e)[:160]}); retrying in {wait_s}s")
            time.sleep(wait_s)
    have = {}
    for p in HF_PATHS:
        if any(k.startswith(p + "/") for k in want):
            for e in api.list_repo_tree(HF_REPO, path_in_repo=p, repo_type="dataset", recursive=True):
                if getattr(e, "size", None) is not None and e.__class__.__name__ == "RepoFile":
                    have[e.path] = e.size
    bad = [k for k, v in want.items() if have.get(k) != v]
    assert not bad, f"{len(bad)} files not on the Hub with the right size, e.g. {bad[:3]}"
    say(f"huggingface: {len(want)} files verified on {HF_REPO} ({getattr(info, 'oid', '') or ''})"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="scheduled")
    args = ap.parse_args()
    load_env()
    os.makedirs(os.path.dirname(JOBS), exist_ok=True)
    lock = open(LOCK, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    t0 = time.time()
    status = "ok"
    try:
        hits = scan([p for p in GIT_PATHS if os.path.exists(p)])
        if hits:
            say(f"SECRET SCAN HIT -- nothing uploaded; files: {hits}")
            status = "secret_scan_hit"
            return 3
        try:
            github(args.tag)
        except Exception as e:                       # keep going: Hugging Face is the second copy
            status = "github_failed"
            say(f"github FAILED: {e}")
        try:
            huggingface(args.tag)
        except Exception as e:
            status = "hf_failed" if status == "ok" else "both_failed"
            say(f"huggingface FAILED: {e}")
        return 0 if status == "ok" else 1
    finally:
        try:
            import json
            st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            st["last_attempt_utc"] = now; st["last_status"] = status
            if status in ("ok", "hf_failed"):
                st["github_ok_utc"] = now
            if status in ("ok", "github_failed"):
                st["hf_ok_utc"] = now
            st["backups_total"] = st.get("backups_total", 0) + 1
            st["backups_failed"] = st.get("backups_failed", 0) + (status != "ok")
            json.dump(st, open(STATUS, "w"), indent=1)
        except Exception:                            # noqa: BLE001
            pass
        with open(JOBS, "a") as fh:
            fh.write(f"{t0:.3f}\t{time.time():.3f}\tbackup\t{status}\t{args.tag}\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    sys.exit(main())
