#!/usr/bin/env python
"""Hugging Face downloads with readable progress in a log file, retries, and a size check.

    from drafter_lora.hf_progress import download
    download("JetLM/SDAR-30B-A3B-Chat-b32", revision="c351bbc...", cache_dir="hf_cache/hub", label="30B judge base")

    # CLI (p4.sh setup uses it for the two pinned models before setup_p3_8gpu.sh, which then finds them cached)
    HF_HUB_OFFLINE=0 python code/drafter_lora/hf_progress.py --models b32,d32

What it prints, every --every seconds (default 20) and at the end, one plain line (no tqdm carriage returns, so
`tail -f` of a log stays readable):
    [dl 30B judge base] 23.41/61.07 GB (38.3%)  files 7/25  112.4 MB/s  ETA 5m35s  elapsed 3m28s
Before starting it lists the files the call will fetch (allow_patterns applied) and their total size; afterwards
it checks every file exists with the listed size.  Up to --retries attempts (huggingface_hub resumes partial files).
`hf_transfer` is switched on when it is importable (HF_HUB_ENABLE_HF_TRANSFER=1); otherwise the default backend
(hf_xet for Xet-backed repos) is used.  The token comes from $HF_TOKEN when set (public repos need none); it is never
printed.  Does not import research_protocol (that sets HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import threading
import time
from typing import Iterable, List, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _fmt_t(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _du(paths: Iterable[str]) -> int:
    tot = 0
    for p in paths:
        if os.path.isfile(p):
            tot += os.path.getsize(p)
            continue
        for d, _ds, fs in os.walk(p):
            for f in fs:
                fp = os.path.join(d, f)
                if not os.path.islink(fp):
                    try:
                        tot += os.path.getsize(fp)
                    except OSError:
                        pass
    return tot


def plan(repo: str, repo_type: str, revision: Optional[str], allow: Optional[Sequence[str]]):
    """[(path, size)] the download will fetch."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    files = []
    for f in api.list_repo_tree(repo, repo_type=repo_type, revision=revision, recursive=True):
        size = getattr(f, "size", None)
        if size is None:                                   # a folder
            continue
        if allow and not any(fnmatch.fnmatch(f.path, p) for p in allow):
            continue
        files.append((f.path, int(size)))
    return files


def download(repo: str, *, repo_type: str = "model", revision: Optional[str] = None,
             cache_dir: Optional[str] = None, local_dir: Optional[str] = None,
             allow_patterns: Optional[Sequence[str]] = None, label: Optional[str] = None,
             every: float = 20.0, retries: int = 4, max_workers: int = 16) -> str:
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        backend = "hf_transfer"
    except ImportError:
        backend = "default (hf_xet when the repo is Xet-backed)"
    if not sys.stdout.isatty():
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # our lines instead of tqdm's \r in log files
    from huggingface_hub import snapshot_download
    label = label or repo
    files = plan(repo, repo_type, revision, allow_patterns)
    total = sum(s for _, s in files)
    print(f"[dl {label}] {repo} ({repo_type}{'@' + revision[:8] if revision else ''}): {len(files)} files, "
          f"{total / 1e9:.2f} GB; backend {backend}", flush=True)

    if local_dir:
        tops = sorted({os.path.join(local_dir, os.path.dirname(p).split("/")[0] if "/" in p else p) for p, _ in files})
        roots = tops + [os.path.join(local_dir, ".cache", "huggingface")]
        targets = [os.path.join(local_dir, p) for p, _ in files]
    else:
        base = os.path.join(cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
                            f"{repo_type}s--" + repo.replace("/", "--"))
        roots, targets = [os.path.join(base, "blobs")], None
        commit = revision if revision and len(revision) == 40 else None
        ref = os.path.join(base, "refs", revision or "main")
        if commit is None and os.path.isfile(ref):
            commit = open(ref).read().strip()
        if commit:                                         # the snapshot's symlinks: what is complete already
            targets = [os.path.join(base, "snapshots", commit, p) for p, _ in files]
    base_bytes = _du(roots)

    def present() -> int:
        """Bytes of planned files already complete (a cached or resumed download is not counted as 0)."""
        if targets is None:
            return 0
        return sum(s for (p, s), t in zip(files, targets) if os.path.isfile(t) and os.path.getsize(t) == s)

    already = present()
    if already:
        print(f"[dl {label}] {already / 1e9:.2f} GB of it already on disk", flush=True)
    sizes = dict(files)
    t0 = time.time()
    stop = threading.Event()

    def done_files() -> int:
        if targets is None:
            return -1
        return sum(1 for (p, s), t in zip(files, targets) if os.path.isfile(t) and os.path.getsize(t) == s)

    def line(final: bool = False) -> str:
        new_b = max(0, _du(roots) - base_bytes)            # bytes written by this call (rate)
        got = present() if final else already + new_b
        got = min(got, total) if total else got
        el = time.time() - t0
        rate = new_b / el if el > 0 else 0.0
        eta = 0.0 if got >= total else ((total - got) / rate if rate > 0 else float("inf"))
        nf = done_files()
        files_s = f"  files {nf}/{len(files)}" if nf >= 0 else ""
        eta_s = _fmt_t(eta) if eta != float("inf") else "?"
        return (f"[dl {label}] {got / 1e9:.2f}/{total / 1e9:.2f} GB ({100 * got / max(total, 1):.1f}%){files_s}  "
                f"{rate / 1e6:.1f} MB/s  ETA {eta_s}  elapsed {_fmt_t(el)}")

    def monitor():
        while not stop.wait(every):
            print(line(), flush=True)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    path, last = None, None
    try:
        for attempt in range(1, retries + 1):
            try:
                path = snapshot_download(repo, repo_type=repo_type, revision=revision, cache_dir=cache_dir,
                                         local_dir=local_dir, allow_patterns=list(allow_patterns) if allow_patterns
                                         else None, token=os.environ.get("HF_TOKEN") or None,
                                         max_workers=max_workers)
                break
            except Exception as e:                       # network blips: snapshot_download resumes partial files
                last = e
                print(f"[dl {label}] attempt {attempt}/{retries} failed: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                time.sleep(min(60, 10 * attempt))
        else:
            raise RuntimeError(f"download of {repo} failed after {retries} attempts") from last
    finally:
        stop.set()
        th.join(timeout=1)
    if local_dir:
        bad = [p for p, s in files if not (os.path.isfile(os.path.join(local_dir, p))
                                           and os.path.getsize(os.path.join(local_dir, p)) == sizes[p])]
    else:
        bad = [p for p, s in files if not (os.path.isfile(os.path.join(path, p))
                                           and os.path.getsize(os.path.join(path, p)) == sizes[p])]
    print(line(final=True).replace("[dl ", "[dl done ", 1), flush=True)
    if bad:
        raise RuntimeError(f"{label}: {len(bad)} files missing or of the wrong size after download: {bad[:5]}")
    return path


MODELS = {   # research_protocol.MODELS, repeated here so this file never imports research_protocol
    "b32": ("JetLM/SDAR-30B-A3B-Chat-b32", "c351bbc37d240aa6871f167e8f92d694281b0c22", "30B judge base"),
    "d32": ("JetLM/SDAR-1.7B-Chat-b32", "2c83c05d84be90a01c72443b2123ebf0376f6fd4", "1.7B drafter"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="b32,d32", help="research_protocol.MODELS keys, into hf_cache/hub")
    ap.add_argument("--every", type=float, default=20.0)
    args = ap.parse_args(argv)
    for k in [m for m in args.models.split(",") if m]:
        repo, rev, label = MODELS[k]
        download(repo, revision=rev, cache_dir=os.path.join(ROOT, "hf_cache", "hub"), label=label, every=args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
