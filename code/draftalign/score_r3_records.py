#!/usr/bin/env python
"""Post-hoc scoring of code-pool decode records, run as an UNPRIVILEGED user (see run_score_r3.sh).

`run_da_decode.py --score` executes generated programs inside the decode process tree
(oc_datasets/mbpp.py::execution = multiprocessing child + exec, no chdir, same user = root). Some
OpenCodeInstruct programs write/delete files, including through absolute paths. Code pools are therefore
decoded WITHOUT --score, and scored here with the identical call the inline scorer makes,

    per_item_correct(ds, [rec])[rec["idx"]]            (run_da_decode.py: `scorer`)

from a non-root process whose cwd is a throwaway directory, so a program cannot touch root-owned files.

Reads  <records>/*.jsonl whose .done marker exists (finished shards only), skipping names already in --skip-dir.
Writes <out>/<same name> with "correct" filled ("score_error" on exception) plus <name>.done.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor


def family(name: str) -> str:
    """Scorer family from the record FILE name: run_da_decode names records <family>_B32_...<split>.shardKofN.jsonl,
    and the family prefix is exactly the `ds` the inline scorer used (mbpp | math500 | aime | gpqa | humaneval |
    ifeval | gsm8k) -- so re-scoring uses the identical dispatch."""
    fam = os.path.basename(name).split("_B32_")[0]
    assert fam in ("mbpp", "math500", "aime", "gpqa", "humaneval", "ifeval", "gsm8k"), f"unknown family in {name}"
    return fam


def _score(job):
    code_dir, sandbox, ds, rec = job
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    os.chdir(tempfile.mkdtemp(prefix="item_", dir=sandbox))       # a fresh cwd per item
    from analyze_s1b import per_item_correct
    try:
        return int(per_item_correct(ds, [rec])[rec["idx"]]), None
    except Exception as e:                                         # same fallback as run_da_decode
        import traceback; return None, traceback.format_exc()[-400:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-dir", default=None, help="names already scored there are skipped")
    ap.add_argument("--code", required=True, help="read-only copy of code/ (analyze_s1b + vendor)")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    sandbox = os.getcwd()
    os.makedirs(args.out, exist_ok=True)
    done_names = {os.path.basename(p)[:-len(".done")] for p in glob.glob(os.path.join(args.records, "*.jsonl.done"))}
    skip = set(os.listdir(args.skip_dir)) if args.skip_dir and os.path.isdir(args.skip_dir) else set()
    todo = sorted(n for n in done_names if n not in skip and not os.path.exists(os.path.join(args.out, n + ".done")))
    print(f"[score] {len(done_names)} finished shards, {len(todo)} to score", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name in todo:
            recs = [json.loads(l) for l in open(os.path.join(args.records, name)) if l.strip()]
            jobs = [(args.code, sandbox, family(name), r) for r in recs]
            n_ok = n_err = 0
            with open(os.path.join(args.out, name + ".tmp"), "w") as fh:
                for r, (c, err) in zip(recs, ex.map(_score, jobs)):
                    r["correct"] = c
                    if err is not None:
                        r["score_error"] = err; n_err += 1
                    n_ok += int(bool(c))
                    fh.write(json.dumps(r) + "\n")
            os.replace(os.path.join(args.out, name + ".tmp"), os.path.join(args.out, name))
            open(os.path.join(args.out, name + ".done"), "w").write("ok\n")
            print(f"[score] {name}: {len(recs)} records, {n_ok} correct, {n_err} scorer errors", flush=True)


if __name__ == "__main__":
    main()
