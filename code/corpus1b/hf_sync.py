#!/usr/bin/env python
"""Push the 1B-corpus artefacts to the PUBLIC Hugging Face repos, verify them file by file, prune local checkpoints.

    /venv/train/bin/python code/corpus1b/hf_sync.py data     # corpus -> $HF_PUSH_REPO (dataset repo)
    /venv/train/bin/python code/corpus1b/hf_sync.py ckpt     # checkpoints -> $HF_CKPT_REPO (model repo) + prune
    /venv/train/bin/python code/corpus1b/hf_sync.py links    # docs/PHASE1_HF_ARTIFACTS.md (links for GitHub)
    /venv/train/bin/python code/corpus1b/hf_sync.py status   # local vs verified checkpoints, disk

data: generation records (never the traces), drafts, the training index + items packed as one tar per hash
      folder (the Hub dislikes 100k+ small files), prompt lists, eval sets, manifests, stage-1 reports, trainer
      logs/probes/configs and eval summaries. GPQA eval records are never uploaded (the set asks not to be
      published in plain text); .env and anything that looks like a token never are either: every uploaded
      text file is scanned first and a hit aborts the upload.
ckpt: every COMPLETE checkpoint (adapter.safetensors + adapter_config.json + trainer_state.pt + trainer_state.json,
      i.e. usable and resumable) of every run under logs/c1b/train/<run>/ goes to <run>/<ckpt>/ in one commit,
      with the run's config.json. It is then verified against the Hub file by file (size + LFS sha256, or the git
      blob sha1 for small files) and a marker is written to logs/c1b/hf_sync/. Local checkpoints older than the
      KEEP (5) newest of their run are deleted only when their marker matches the directory AND a fresh Hub
      listing still verifies. Nothing else is ever deleted.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ["HF_HUB_OFFLINE"] = "0"
KEEP = 5
CKPT_FILES = ("adapter.safetensors", "adapter_config.json", "trainer_state.pt", "trainer_state.json")
MARK_DIR = os.path.join(ROOT, "logs", "c1b", "hf_sync", "ckpt")
LINKS_MD = os.path.join(ROOT, "docs", "PHASE1_HF_ARTIFACTS.md")
TEXT_EXT = (".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml", ".csv", ".sh", ".py")


def say(msg: str) -> None:
    print(f"[hf_sync {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


# ---------------------------------------------------------------------------
# secrets: never let a token reach a public repo
# ---------------------------------------------------------------------------
def _secret_patterns() -> List[bytes]:
    pats = []
    for k in ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "HF_PUSH_TOKEN"):
        v = os.environ.get(k, "")
        if len(v) >= 12:
            pats.append(v.encode())
    return pats


_RX = re.compile(rb"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")


def scan_for_secrets(paths: List[str]) -> List[str]:
    pats, hits = _secret_patterns(), []
    for p in paths:
        if not p.endswith(TEXT_EXT) or os.path.getsize(p) > 2_000_000_000:
            continue
        with open(p, "rb") as fh:
            prev = b""
            while True:
                b = fh.read(1 << 24)
                if not b:
                    break
                chunk = prev + b
                if any(x in chunk for x in pats) or _RX.search(chunk):
                    hits.append(p)
                    break
                prev = b[-256:]
    return hits


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
DATA_ALLOW = [
    "logs/c1b/gen/wave*/gpu*/records.*.jsonl.gz", "logs/c1b/gen/wave*/gpu*/done.txt", "logs/c1b/gen/wave*/gpu*/stats.jsonl",
    "logs/c1b/draft/wave*/gpu*/drafts.*.npz", "logs/c1b/draft/wave*/gpu*/done.txt",
    "logs/c1b/trainset/wave*/index.jsonl", "logs/c1b/trainset/wave*/build_summary.json",
    "logs/c1b/trainset/wave*/items_tar/*.tar",
    "logs/c1b/trainset/index_*.jsonl", "logs/c1b/trainset/holdout_pids_*.txt",       # merged indexes (128M, 228M)
    "logs/c1b/train/*/seen_items.json", "logs/c1b/c228_state/*.md", "logs/c1b/c228_state/*.json",
    "logs/c1b/eval/*/compare.*",
    "data/c1b/prompt_lists/*.jsonl", "data/c1b/prompt_lists/manifest.json", "data/c1b/prompt_lists/spotcheck/**",
    "data/c1b/eval/**", "data/c1b/sources_manifest.json",
    "logs/c1b/stage1/*.jsonl", "logs/c1b/stage1/*.log", "logs/c1b/waveA_state/*.jsonl",
    "logs/c1b/train/*/probe.jsonl", "logs/c1b/train/*/config.json", "logs/c1b/train/*/train.rank0.jsonl",
    "logs/c1b/train/*.log", "logs/c1b/train/lr_choice.txt",
    "logs/c1b/eval/*/summary.*", "logs/c1b/eval/*/queue.txt.bak", "logs/c1b/eval/*/*/*.jsonl", "logs/c1b/gate_A.md",
    "logs/c1b/*.log",
]
DATA_IGNORE = ["**/.env", "**/*.env", "**/*.tmp*", "**/trace/**", "**/gpqa*/**", "**/*gpqa*.jsonl",
               "**/.cache/**", "logs/c1b/backup_hf.log", "logs/c1b/hf_sync/**"]
# files appended IN PLACE while other steps run (the index, logs, done/stats files, trainer logs): upload_large_folder
# hashes a file, then commits it later; if it grew meanwhile the Hub rejects every commit ("LFS pointer pointed to a
# file that does not exist") and huggingface_hub retries forever. These go up from snapshot copies, in one commit.
APPENDED = ["**/index.jsonl", "**/*.log", "**/done.txt", "**/stats.jsonl", "**/train.rank0.jsonl", "**/probe.jsonl",
            "**/verify.jsonl", "logs/c1b/stage1/*.jsonl"]
STAGE = "/dev/shm/c1b_hf_stage"


def pack_items(wave_dir: str) -> int:
    """items/<h2>/*.npz -> items_tar/<h2>.tar, rebuilt only when the folder's file count changed."""
    src, dst = os.path.join(wave_dir, "items"), os.path.join(wave_dir, "items_tar")
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    n = 0
    for h2 in sorted(os.listdir(src)):
        files = sorted(f for f in os.listdir(os.path.join(src, h2)) if f.endswith(".npz"))
        cnt = os.path.join(dst, f"{h2}.count")
        if os.path.exists(cnt) and open(cnt).read().strip() == str(len(files)):
            continue
        tmp = os.path.join(dst, f"{h2}.tar.tmp")
        with tarfile.open(tmp, "w") as tf:
            for f in files:
                tf.add(os.path.join(src, h2, f), arcname=f"items/{h2}/{f}")
        os.replace(tmp, os.path.join(dst, f"{h2}.tar"))
        open(cnt, "w").write(str(len(files)))
        n += 1
    return n


def cmd_data(a) -> int:
    repo = os.environ["HF_PUSH_REPO"]
    os.chdir(ROOT)
    for w in glob.glob("logs/c1b/trainset/wave*"):
        k = pack_items(w)
        if k:
            say(f"packed {k} item folders of {w}")
    import fnmatch
    cand = []
    for pat in DATA_ALLOW:
        cand += [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]
    cand = sorted({p for p in cand if not any(fnmatch.fnmatch(p, ig) for ig in DATA_IGNORE)})
    hits = scan_for_secrets(cand)
    if hits:
        say(f"SECRET SCAN HIT in {hits[:5]}: nothing uploaded")
        return 3
    h = api()
    h.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    say(f"data -> {repo}: {len(cand)} candidate files, {sum(os.path.getsize(p) for p in cand) / 1e9:.2f} GB")
    appended = [p for p in cand if any(fnmatch.fnmatch(p, g) for g in APPENDED)]
    h.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=ROOT, allow_patterns=DATA_ALLOW,
                          ignore_patterns=DATA_IGNORE + APPENDED, print_report=False)
    if appended:                                                  # snapshot copies of the growing files, one commit
        shutil.rmtree(STAGE, ignore_errors=True)
        for p in appended:
            os.makedirs(os.path.dirname(os.path.join(STAGE, p)), exist_ok=True)
            shutil.copy2(p, os.path.join(STAGE, p))
        h.upload_folder(repo_id=repo, repo_type="dataset", folder_path=STAGE,
                        commit_message=f"snapshot of {len(appended)} growing files (index, logs, done/stats)")
        shutil.rmtree(STAGE, ignore_errors=True)
    say("data ok")
    return 0


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def git_blob_sha1(path: str) -> str:
    h = hashlib.sha1()
    h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(d: str) -> Dict[str, list]:
    return {f: [os.path.getsize(os.path.join(d, f)), os.stat(os.path.join(d, f)).st_mtime_ns]
            for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}


def complete(d: str) -> bool:
    return all(os.path.isfile(os.path.join(d, f)) for f in CKPT_FILES)


def local_ckpts() -> List[dict]:
    out = []
    for run in sorted(glob.glob(os.path.join(ROOT, "logs", "c1b", "train", "*"))):
        if not os.path.isdir(run) or os.path.basename(run) in ("bench",):
            continue
        for d in sorted(glob.glob(os.path.join(run, "*"))):
            name = os.path.basename(d)
            if not os.path.isdir(d) or name.endswith(".tmp") or not re.match(r"^(tok_\d{12}|roll_\d{8}|final)$", name):
                continue
            if not complete(d):
                continue
            try:
                st = json.load(open(os.path.join(d, "trainer_state.json")))
            except (ValueError, OSError):
                continue
            out.append({"run": os.path.basename(run), "name": name, "dir": d, "tokens": float(st.get("tokens", 0.0)),
                        "step": int(st.get("step", 0)), "mtime": os.path.getmtime(os.path.join(d, "trainer_state.json"))})
    return out


def marker_path(c: dict) -> str:
    return os.path.join(MARK_DIR, f"{c['run']}__{c['name']}.json")


def remote_verify(h, repo: str, prefix: str, local_dir: str) -> Optional[dict]:
    """{file: sha} if every checkpoint file of local_dir is on the Hub under prefix with the same content."""
    try:
        tree = {f.path: f for f in h.list_repo_tree(repo, path_in_repo=prefix, recursive=True, expand=True,
                                                    repo_type="model") if hasattr(f, "size")}
    except Exception as e:                                              # noqa: BLE001
        say(f"listing {prefix} failed: {type(e).__name__}: {e}")
        return None
    got = {}
    for f in CKPT_FILES:
        lp, rp = os.path.join(local_dir, f), f"{prefix}/{f}"
        r = tree.get(rp)
        if r is None or int(r.size) != os.path.getsize(lp):
            return None
        lfs = getattr(r, "lfs", None)
        if lfs:
            s = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
            if s != sha256(lp):
                return None
            got[f] = "sha256:" + s
        else:
            if getattr(r, "blob_id", None) != git_blob_sha1(lp):
                return None
            got[f] = "git:" + r.blob_id
    return got


def model_card(repo: str, verified: List[dict]) -> str:
    rows = "\n".join(f"| `{m['path_in_repo']}` | {m['tokens'] / 1e6:.2f}M | {m['step']} | {m['verified_utc']} |"
                     for m in sorted(verified, key=lambda m: (m["run"], m["tokens"])))
    return f"""---
license: apache-2.0
base_model: JetLM/SDAR-30B-A3B-Chat-b32
tags: [lora, block-diffusion, judge, sdar]
---
# Phase-1 judge checkpoints (1B-token corpus, wave A)

LoRA judge adapters for the stock **JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc** (block 32), trained with
`code/corpus1b/train_c1b.py` from github.com/NoviceCoderInfinity/diffusion-moe-general-expert (branch
`phase1-datacorpus`). The drafter they judge is the stock **JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05**.

Every folder `<run>/<checkpoint>/` is a complete checkpoint: `adapter.safetensors` + `adapter_config.json`
(LoRA r32 / alpha 64 on q, k, v, o in all 48 layers + the draft flag) and `trainer_state.pt` (AdamW state,
token count) + `trainer_state.json`, so it can be both used and resumed (`--resume` / `--continue-from`).
`<run>/config.json` is the run's full configuration. The adapter is flag-gated: load it with
`draftalign.adapter.load_adapter` and never merge it (AGENTS.md 9(e)). `tok_<12 digits>` = corpus tokens seen.
The training corpus is the public dataset repo `{os.environ.get('HF_PUSH_REPO', '')}`.

| checkpoint | tokens | step | verified on the Hub (UTC) |
|---|---:|---:|---|
{rows}
"""


def links_doc(verified: List[dict]) -> str:
    ck, ds = os.environ.get("HF_CKPT_REPO", ""), os.environ.get("HF_PUSH_REPO", "")
    rows = "\n".join(f"| {m['run']} | [`{m['name']}`](https://huggingface.co/{ck}/tree/main/{m['path_in_repo']}) | "
                     f"{m['tokens'] / 1e6:.2f}M | {m['verified_utc']} |"
                     for m in sorted(verified, key=lambda m: (m["run"], m["tokens"])))
    return f"""# Phase 1 (1B-token judge corpus): artefacts on Hugging Face

Auto-generated by `code/corpus1b/hf_sync.py links`; every checkpoint listed was verified on the Hub file by file.

* **Judge checkpoints (weights, public):** https://huggingface.co/{ck}
* **Corpus (generation records, drafts, training items, prompt lists, eval sets, logs; public):**
  https://huggingface.co/datasets/{ds}

Base model: `JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc` (block 32); drafter: `JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05`.
Each checkpoint folder is complete (adapter + optimizer/trainer state): usable and resumable.

| run | checkpoint | corpus tokens | verified (UTC) |
|---|---|---:|---|
{rows}
"""


def verified_markers() -> List[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(MARK_DIR, "*.json")))]


def cmd_ckpt(a) -> int:
    repo = os.environ["HF_CKPT_REPO"]
    h = api()
    h.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    os.makedirs(MARK_DIR, exist_ok=True)
    rc = 0
    # every complete checkpoint is pushed as soon as it lands, newest first (the box may die at any time); the KEEP (5)
    # newest of each run stay on the local disk, older ones are deleted by prune only once verified on the Hub
    cks = sorted(local_ckpts(), key=lambda c: (c["tokens"], c["mtime"]), reverse=True)
    for c in cks:
        mp, fp = marker_path(c), fingerprint(c["dir"])
        if os.path.exists(mp) and json.load(open(mp)).get("fingerprint") == fp:
            continue
        prefix = f"{c['run']}/{c['name']}"
        hits = scan_for_secrets([os.path.join(c["dir"], f) for f in os.listdir(c["dir"])])
        if hits:
            say(f"SECRET SCAN HIT in {hits}: {prefix} not uploaded"); rc = 3; continue
        say(f"upload {prefix} ({sum(v[0] for v in fp.values()) / 1e6:.0f} MB)")
        try:
            info = h.upload_folder(repo_id=repo, repo_type="model", folder_path=c["dir"], path_in_repo=prefix,
                                   allow_patterns=list(CKPT_FILES),
                                   commit_message=f"{prefix}: {c['tokens'] / 1e6:.2f}M tokens, step {c['step']}")
            cfg = os.path.join(ROOT, "logs", "c1b", "train", c["run"], "config.json")
            if os.path.exists(cfg):
                h.upload_file(path_or_fileobj=cfg, path_in_repo=f"{c['run']}/config.json", repo_id=repo, repo_type="model")
        except Exception as e:                                          # noqa: BLE001
            say(f"upload {prefix} failed: {type(e).__name__}: {e}"); rc = 1; continue
        if fingerprint(c["dir"]) != fp:
            say(f"{prefix} changed during upload; retry next round"); rc = 1; continue
        got = remote_verify(h, repo, prefix, c["dir"])
        if got is None:
            say(f"VERIFY FAILED for {prefix}; kept locally, retry next round"); rc = 1; continue
        m = {"repo": repo, "run": c["run"], "name": c["name"], "path_in_repo": prefix, "tokens": c["tokens"],
             "step": c["step"], "files": got, "fingerprint": fp,
             "commit": getattr(info, "oid", None) or str(info), "verified_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
        json.dump(m, open(mp + ".tmp", "w"), indent=1); os.replace(mp + ".tmp", mp)
        say(f"verified {prefix} on the Hub ({len(got)} files)")
    # model card + links doc
    ver = verified_markers()
    try:
        h.upload_file(path_or_fileobj=model_card(repo, ver).encode(), path_in_repo="README.md", repo_id=repo,
                      repo_type="model", commit_message="model card")
    except Exception as e:                                              # noqa: BLE001
        say(f"model card upload failed: {e}")
    write_links(ver)
    rc = max(rc, prune(h, repo))
    return rc


def prune(h, repo: str) -> int:
    """Delete local checkpoints older than the KEEP newest of their run, only when verified on the Hub (now)."""
    rc = 0
    by_run: Dict[str, List[dict]] = {}
    for c in local_ckpts():
        by_run.setdefault(c["run"], []).append(c)
    for run, cs in by_run.items():
        cs.sort(key=lambda c: (c["tokens"], c["mtime"]), reverse=True)
        for c in cs[KEEP:]:
            mp = marker_path(c)
            if not os.path.exists(mp):
                say(f"keep {run}/{c['name']}: not yet verified on the Hub"); continue
            m = json.load(open(mp))
            if m.get("fingerprint") != fingerprint(c["dir"]):
                say(f"keep {run}/{c['name']}: changed since verification"); continue
            if remote_verify(h, repo, m["path_in_repo"], c["dir"]) is None:
                say(f"keep {run}/{c['name']}: Hub copy does not verify now"); rc = 1; continue
            shutil.rmtree(c["dir"])
            say(f"pruned local {run}/{c['name']} (verified on the Hub: {m['path_in_repo']})")
    return rc


def write_links(ver: Optional[List[dict]] = None) -> None:
    ver = verified_markers() if ver is None else ver
    txt = links_doc(ver)
    if not os.path.exists(LINKS_MD) or open(LINKS_MD).read() != txt:
        os.makedirs(os.path.dirname(LINKS_MD), exist_ok=True)
        open(LINKS_MD, "w").write(txt)


def cmd_links(a) -> int:
    write_links()
    return 0


def cmd_status(a) -> int:
    for c in local_ckpts():
        mp = marker_path(c)
        v = os.path.exists(mp) and json.load(open(mp)).get("fingerprint") == fingerprint(c["dir"])
        print(f"{c['run']}/{c['name']}: {c['tokens'] / 1e6:.2f}M tokens, verified={v}")
    du = shutil.disk_usage(ROOT)
    print(f"disk free {du.free / 1e9:.0f} GB of {du.total / 1e9:.0f} GB; verified markers: {len(verified_markers())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("data", "ckpt", "links", "status"))
    a = ap.parse_args(argv)
    if a.cmd in ("data", "ckpt"):             # one Hub writer at a time: two upload_large_folder runs on one folder
        import fcntl                          # would share (and corrupt) its .cache/huggingface/upload state
        os.makedirs(os.path.dirname(MARK_DIR), exist_ok=True)
        lock = open(os.path.join(os.path.dirname(MARK_DIR), "hub.lock"), "w")
        t0 = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        if time.time() - t0 > 5:
            say(f"waited {time.time() - t0:.0f} s for another Hub writer")
    return {"data": cmd_data, "ckpt": cmd_ckpt, "links": cmd_links, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
