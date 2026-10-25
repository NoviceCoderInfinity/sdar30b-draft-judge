#!/usr/bin/env python
"""Download what phase 4 needs beyond code/phase3_8gpu/setup_p3_8gpu.sh (which brings the env, both pinned models,
the 128M judge adapter and the benchmark sets), then check it.

    set -a; . ./.env; set +a; HF_HUB_OFFLINE=0 /venv/p2/bin/python code/drafter_lora/fetch_p4.py [--what all]

  corpus   Arushhh/c1b-judge-corpus-waveA (dataset): logs/c1b/trainset/{waveA,waveB}/{index.jsonl, build_summary.json,
           items_tar/*.tar} -> untarred into logs/c1b/trainset/<wave>/items/; checks 165,438 + 166,380 index rows and
           that every row's npz exists (2.3 GB of tars on the Hub; the runbook's "11 GB" overstates it)
  phase3   Arushhh/phase3-redraft (dataset): logs/phase3_8gpu/decode/{base,rd1,rd2}/** -> logs/phase4/ref/phase3/
  vanilla  Arushhh/phase2-drafter-swap (dataset): assets/ref_stock_30B/** -> logs/phase4/ref/phase2/

Does NOT import research_protocol (it sets HF_HUB_OFFLINE=1 on import).  The token is read from $HF_TOKEN and never
printed.  Re-runnable: finished steps leave logs/phase4/fetch/.done_<what>.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAVES = {"waveA": 165_438, "waveB": 166_380}
DONE = os.path.join(ROOT, "logs", "phase4", "fetch")


def _dl(repo: str, patterns, local_dir: str, label: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from drafter_lora.hf_progress import download         # progress lines, retries, size check
    os.makedirs(local_dir, exist_ok=True)
    download(repo, repo_type="dataset", local_dir=local_dir, allow_patterns=list(patterns), label=label)


def _safe_members(tf: tarfile.TarFile, dest: str):
    root = os.path.realpath(dest)
    for m in tf.getmembers():
        target = os.path.realpath(os.path.join(dest, m.name))
        if not (target == root or target.startswith(root + os.sep)) or m.issym() or m.islnk():
            raise RuntimeError(f"refusing tar member outside {dest}: {m.name}")
        yield m


def corpus() -> dict:
    pats = []
    for w in WAVES:
        pats += [f"logs/c1b/trainset/{w}/index.jsonl", f"logs/c1b/trainset/{w}/build_summary.json",
                 f"logs/c1b/trainset/{w}/items_tar/*"]
    _dl("Arushhh/c1b-judge-corpus-waveA", pats, ROOT, "corpus waves A+B")
    out = {}
    for w, want in WAVES.items():
        d = os.path.join(ROOT, "logs", "c1b", "trainset", w)
        tars = sorted(glob.glob(os.path.join(d, "items_tar", "*.tar")))
        t0, n_items = time.time(), 0
        for j, t in enumerate(tars):
            mark = t + ".extracted"
            if not os.path.exists(mark):
                with tarfile.open(t) as tf:
                    members = list(_safe_members(tf, d))
                    tf.extractall(d, members=members)
                    n_items += len(members)
                open(mark, "w").close()
            if (j + 1) % 32 == 0 or j + 1 == len(tars):
                print(f"[untar {w}] {j + 1}/{len(tars)} tars, {n_items} files extracted this run, "
                      f"{time.time() - t0:.0f}s", flush=True)
        rows = [json.loads(l) for l in open(os.path.join(d, "index.jsonl")) if l.strip()]
        missing = [r["file"] for r in rows if not os.path.isfile(os.path.join(d, r["file"]))]
        print(f"[check {w}] {len(rows)} index rows (runbook: {want}), {len(missing)} without an npz", flush=True)
        out[w] = {"rows": len(rows), "want": want, "missing_npz": len(missing), "first_missing": missing[:3]}
        assert len(rows) == want, f"{w}: {len(rows)} index rows, the runbook says {want}"
        assert not missing, f"{w}: {len(missing)} index rows have no npz (first {missing[:3]})"
    return out


def phase3() -> dict:
    dest = os.path.join(ROOT, "logs", "phase4", "ref", "phase3")
    _dl("Arushhh/phase3-redraft", [f"logs/phase3_8gpu/decode/{a}/*" for a in ("base", "rd1", "rd2")], dest,
        "Phase 3 base/rd1/rd2 records")
    base = os.path.join(dest, "logs", "phase3_8gpu", "decode")
    return {a: {s: len(glob.glob(os.path.join(base, a, s, "*.jsonl"))) for s in sorted(os.listdir(os.path.join(base, a)))}
            for a in ("base", "rd1", "rd2") if os.path.isdir(os.path.join(base, a))}


def vanilla() -> dict:
    dest = os.path.join(ROOT, "logs", "phase4", "ref", "phase2")
    _dl("Arushhh/phase2-drafter-swap", ["assets/ref_stock_30B/*"], dest, "vanilla 30B records")
    files = glob.glob(os.path.join(dest, "assets", "ref_stock_30B", "**", "*.jsonl"), recursive=True)
    return {"jsonl_files": len(files), "dirs": sorted({os.path.relpath(os.path.dirname(f), dest) for f in files})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all", help="all | corpus,phase3,vanilla")
    args = ap.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.makedirs(DONE, exist_ok=True)
    steps = ("corpus", "phase3", "vanilla") if args.what == "all" else tuple(args.what.split(","))
    report = {}
    for s in steps:
        if os.path.exists(os.path.join(DONE, f".done_{s}")):
            print(f"[fetch] {s}: done earlier", flush=True)
            continue
        print(f"[fetch] {s} ...", flush=True)
        report[s] = {"corpus": corpus, "phase3": phase3, "vanilla": vanilla}[s]()
        open(os.path.join(DONE, f".done_{s}"), "w").close()
        print(f"[fetch] {s}: {json.dumps(report[s])[:600]}", flush=True)
    with open(os.path.join(DONE, "report.json"), "a") as fh:
        fh.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
