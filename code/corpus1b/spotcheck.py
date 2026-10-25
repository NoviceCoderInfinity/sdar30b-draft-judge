#!/usr/bin/env python
"""Spot-check export for the 1B corpus (docs/CORPUS_1B_BUILD.md): per source, the 200 rows with the
smallest common.order_key(pid) (a deterministic, file-order-independent sample), for a manual "does the
key actually match the prompt" review -- plus automatic sanity counts that don't need a human.

Reads data/c1b/prompt_lists/<domain>.jsonl (build_prompt_lists.py's output); writes
data/c1b/spotcheck/<tag>.jsonl (fields: pid, source, source_id, prompt, key, meta) and prints/returns the
sanity-count summary (also written to data/c1b/spotcheck/summary.json): empty/missing answers, and 'mc' keys
whose answer letter falls outside 0..n_options-1.

Usage:
    python code/corpus1b/spotcheck.py                       # every domain found in data/c1b/prompt_lists
    python code/corpus1b/spotcheck.py --domains chat,maths
    python code/corpus1b/spotcheck.py --n 200 --out data/c1b/spotcheck

TODO: not run end-to-end on this Mac (no full prompt_lists to read yet) -- exercised only by
code/corpus1b/tests/test_data_builders.py on a handful of synthetic rows written to a tmp dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402

N_PER_SOURCE = 200


def _is_empty_answer(key: dict) -> bool:
    t = key.get("type")
    if t == "math" or t == "free":
        return not str(key.get("answer", "")).strip()
    if t == "mc":
        return not str(key.get("answer", "")).strip()
    if t == "asserts":
        return not str(key.get("tests", "")).strip()
    if t == "humaneval":
        return not str(key.get("test", "")).strip()
    if t == "stdio":
        return not key.get("cases")
    if t == "ifeval":
        return not str(key.get("gold", "")).strip()
    return False


def _mc_out_of_range(key: dict) -> bool:
    if key.get("type") != "mc":
        return False
    answer, n = str(key.get("answer", "")), key.get("n_options")
    if not answer or n is None:
        return False
    idx = ord(answer[0].upper()) - ord("A")
    return not (0 <= idx < int(n))


def sample_by_source(rows: List[dict], n: int = N_PER_SOURCE) -> Dict[str, List[dict]]:
    by_tag: Dict[str, List[dict]] = {}
    for r in rows:
        by_tag.setdefault(common.source_tag_of(r["pid"]), []).append(r)
    out = {}
    for tag, rs in by_tag.items():
        rs.sort(key=lambda r: common.order_key(r["pid"]))
        out[tag] = rs[:n]
    return out


def sanity_counts(rows: List[dict]) -> Dict[str, int]:
    return {"n_rows": len(rows), "empty_answer": sum(1 for r in rows if _is_empty_answer(r["key"])),
            "mc_letter_out_of_range": sum(1 for r in rows if _mc_out_of_range(r["key"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default=None, help="comma-separated; default: every *.jsonl in --prompt-lists-dir")
    ap.add_argument("--prompt-lists-dir", default=common.PROMPT_DIR)
    ap.add_argument("--n", type=int, default=N_PER_SOURCE)
    ap.add_argument("--out", default=os.path.join(common.DATA_DIR, "spotcheck"))
    args = ap.parse_args(argv)

    if args.domains:
        domain_files = [(d, os.path.join(args.prompt_lists_dir, f"{d}.jsonl")) for d in args.domains.split(",")]
    else:
        domain_files = []
        if os.path.isdir(args.prompt_lists_dir):
            for fn in sorted(os.listdir(args.prompt_lists_dir)):
                if fn.endswith(".jsonl"):
                    domain_files.append((fn[:-len(".jsonl")], os.path.join(args.prompt_lists_dir, fn)))

    all_rows: List[dict] = []
    for domain, path in domain_files:
        if not os.path.exists(path):
            print(f"[spotcheck] WARNING: {path} not found, skipping domain {domain}", file=sys.stderr)
            continue
        all_rows.extend(common.read_jsonl(path))

    samples = sample_by_source(all_rows, args.n)
    summary: Dict[str, dict] = {}
    for tag, rs in samples.items():
        out_path = os.path.join(args.out, f"{tag}.jsonl")
        common.write_jsonl(out_path, [{"pid": r["pid"], "source": r["source"], "source_id": r["source_id"],
                                       "prompt": r["prompt"], "key": r["key"], "meta": r.get("meta", {})}
                                      for r in rs])
        summary[tag] = sanity_counts(rs)
        print(f"[spotcheck] {tag:16s} {summary[tag]}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
