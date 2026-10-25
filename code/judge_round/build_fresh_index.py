#!/usr/bin/env python
"""Fresh judge round, step 5: one training index = the fresh wave-C items + a repeated slice of the old waves.

    /venv/p2/bin/python code/judge_round/build_fresh_index.py \
        --fresh logs/c1b/trainset/waveC/index.jsonl \
        --repeat-index logs/c1b/trainset/waveA/index.jsonl --repeat-index logs/c1b/trainset/waveB/index.jsonl \
        --repeat "science=8.0e6" \
        --exclude-pids logs/phase4/train/main/holdout_pids.json --exclude-pids logs/c1b/trainset/holdout_pids_waveA.txt \
        --out logs/c1b/trainset/fresh/index.jsonl

What goes in, in this order:
  1. every row of every --fresh index (wave-C prompts that were never decoded before);
  2. for each --repeat DOMAIN=TOKENS, rows of that domain from the --repeat-index files (science has no fresh prompts, so it
     is REPEATED from waves A / B): pids named in any --exclude-pids file are skipped (a .txt with one pid per line, or a
     .json that is a list or a {"pids": [...]} object like holdout_pids.json), the rest are sorted by
     corpus1b/common.order_key(pid) and taken until the corpus-token sum reaches the budget (so the last item overshoots
     by less than one item; a pool smaller than the budget is taken whole and reported as a shortfall).
Each repeated row gets "repeated": true and keeps its original "wave". "file" of every row is rewritten to be relative to
the OUTPUT index's directory (the trainer resolves it from there, draftalign/dataset.py), "idx" is renumbered 0..n-1 in
output order, everything else is untouched.  Fresh rows whose pid is in an exclusion file are kept but counted and warned
about (they should not exist).  A pid that appears twice in the output, or an item file that does not exist, is an error.

Also writes <out dir>/build_summary.json (items and corpus tokens per domain, per (wave, repeated), per cell; totals; the
budgets and their shortfall; the exclusion counts) and prints it as one table.  --strict exits 3 on a budget shortfall.
CPU only, no model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CORPUS = os.path.join(CODE, "corpus1b")
for p in (CODE, CORPUS):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402

DOM_ORDER = ("maths", "code", "chat", "if", "science")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def read_rows(path: str) -> List[dict]:
    """The rows of an index.jsonl, with `_abs` = the absolute path of the row's item file."""
    root = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("file", "pid", "domain", "corpus_tokens"):
                if k not in r:
                    raise SystemExit(f"{path}:{ln}: row has no {k!r}")
            r["_abs"] = r["file"] if os.path.isabs(r["file"]) else os.path.abspath(os.path.join(root, r["file"]))
            rows.append(r)
    return rows


def load_pids(path: str) -> List[str]:
    """pids from a .txt (one per line; blank lines and '#' comments skipped) or a .json (a list, or {"pids": [...]})."""
    if path.endswith(".json"):
        d = json.load(open(path))
        if isinstance(d, dict):
            if "pids" not in d:
                raise SystemExit(f"{path}: a JSON object without a 'pids' key")
            d = d["pids"]
        if not isinstance(d, list):
            raise SystemExit(f"{path}: expected a JSON list of pids")
        return [str(x) for x in d]
    out = []
    for line in open(path):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def parse_budgets(specs: Sequence[str]) -> Dict[str, float]:
    """['science=8.0e6', 'chat=1e6,if=2e6'] -> {'science': 8e6, ...}; a domain twice is an error."""
    out: Dict[str, float] = {}
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"--repeat expects DOMAIN=TOKENS, got {part!r}")
            d, v = part.split("=", 1)
            d, val = d.strip(), float(v)
            if d in out:
                raise SystemExit(f"--repeat names domain {d!r} twice")
            if val < 0:
                raise SystemExit(f"--repeat {d}: a negative budget")
            out[d] = val
    return out


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def take_repeats(pool: Iterable[dict], budget: float) -> Tuple[List[dict], int]:
    """Rows of one domain in common.order_key(pid) order until the corpus-token sum reaches `budget`; (rows, tokens)."""
    chosen, got = [], 0
    for r in sorted(pool, key=lambda r: C.order_key(str(r["pid"]))):
        if got >= budget:
            break
        chosen.append(r)
        got += int(r["corpus_tokens"])
    return chosen, got


def build(fresh_paths: Sequence[str], repeat_paths: Sequence[str], budgets: Dict[str, float],
          exclude_paths: Sequence[str], out_path: str) -> Tuple[List[dict], dict]:
    """The output rows (idx / file rewritten, `_abs` dropped) and the summary dict; nothing is written."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    excl_per_file = {p: load_pids(p) for p in exclude_paths}
    excl = {pid for v in excl_per_file.values() for pid in v}

    fresh = [r for p in fresh_paths for r in read_rows(p)]
    fresh_in_excl = sum(1 for r in fresh if str(r["pid"]) in excl)

    pool_all = [r for p in repeat_paths for r in read_rows(p)]
    repeated: List[dict] = []
    per_budget: Dict[str, dict] = {}
    excluded_rows: Dict[str, int] = {}
    for d, budget in budgets.items():
        rows_d = [r for r in pool_all if r["domain"] == d]
        keep = [r for r in rows_d if str(r["pid"]) not in excl]
        excluded_rows[d] = len(rows_d) - len(keep)
        chosen, got = take_repeats(keep, budget)
        for r in chosen:
            r["repeated"] = True
        repeated += chosen
        per_budget[d] = {"requested_tokens": budget, "taken_tokens": got, "taken_items": len(chosen),
                         "pool_items": len(rows_d), "pool_items_after_exclusion": len(keep),
                         "pool_tokens_after_exclusion": sum(int(r["corpus_tokens"]) for r in keep),
                         "shortfall_tokens": max(0.0, budget - got)}

    rows = fresh + repeated
    seen: Dict[str, int] = {}
    for i, r in enumerate(rows):
        pid = str(r["pid"])
        if pid in seen:
            raise SystemExit(f"pid {pid} appears twice in the output (rows {seen[pid]} and {i}); "
                             f"the trainer's loader asserts pids are unique")
        seen[pid] = i
    missing = [r["_abs"] for r in rows if not os.path.isfile(r["_abs"])]
    if missing:
        raise SystemExit(f"{len(missing)} item files do not exist, first: {missing[:3]}")
    for i, r in enumerate(rows):
        r["file"] = os.path.relpath(r.pop("_abs"), out_dir)
        r["idx"] = i

    cells: Counter = Counter()
    toks: Counter = Counter()
    for r in rows:
        key = (str(r["domain"]), str(r.get("wave")), bool(r.get("repeated", False)))
        cells[key] += 1
        toks[key] += int(r["corpus_tokens"])

    def _agg(keyfn):
        a: Dict[str, dict] = defaultdict(lambda: {"items": 0, "corpus_tokens": 0})
        for k, n in cells.items():
            a[keyfn(k)]["items"] += n
            a[keyfn(k)]["corpus_tokens"] += toks[k]
        return dict(a)

    by_domain = _agg(lambda k: k[0])
    total_tok = sum(toks.values())
    for v in by_domain.values():
        v["token_share"] = round(v["corpus_tokens"] / total_tok, 4) if total_tok else 0.0
    by_group = _agg(lambda k: f"{k[1]}/{'repeated' if k[2] else 'fresh'}")
    fresh_tok = sum(t for k, t in toks.items() if not k[2])
    summary = {
        "out": os.path.abspath(out_path),
        "inputs": {"fresh": list(fresh_paths), "repeat_index": list(repeat_paths), "exclude_pids": list(exclude_paths)},
        "totals": {"items": len(rows), "corpus_tokens": total_tok,
                   "fresh_items": len(fresh), "fresh_corpus_tokens": fresh_tok,
                   "repeated_items": len(repeated), "repeated_corpus_tokens": total_tok - fresh_tok},
        "by_domain": by_domain,
        "by_wave_repeated": by_group,
        "cells": [{"domain": d, "wave": w, "repeated": rp, "items": n, "corpus_tokens": toks[(d, w, rp)]}
                  for (d, w, rp), n in sorted(cells.items())],
        "budgets": per_budget,
        "exclusions": {"files": {p: len(set(v)) for p, v in excl_per_file.items()}, "unique_pids": len(excl),
                       "repeat_pool_rows_excluded": excluded_rows, "fresh_rows_in_exclusion": fresh_in_excl},
    }
    return rows, summary


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_outputs(rows: List[dict], summary: dict, out_path: str) -> str:
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)
    sp = os.path.join(out_dir, "build_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    return sp


def table(summary: dict) -> str:
    cells = summary["cells"]
    doms = sorted({c["domain"] for c in cells}, key=lambda d: (DOM_ORDER.index(d) if d in DOM_ORDER else 99, d))

    def get(d, rep):
        n = sum(c["items"] for c in cells if c["domain"] == d and c["repeated"] == rep)
        t = sum(c["corpus_tokens"] for c in cells if c["domain"] == d and c["repeated"] == rep)
        return n, t

    L = [f"{'domain':8s} {'fresh items':>11s} {'fresh Mtok':>10s} {'repeat items':>12s} {'repeat Mtok':>11s} "
         f"{'total items':>11s} {'total Mtok':>10s} {'share':>6s}"]
    for d in doms:
        fn, ft = get(d, False)
        rn, rt = get(d, True)
        L.append(f"{d:8s} {fn:11d} {ft / 1e6:10.2f} {rn:12d} {rt / 1e6:11.2f} {fn + rn:11d} {(ft + rt) / 1e6:10.2f} "
                 f"{summary['by_domain'][d]['token_share']:6.3f}")
    t = summary["totals"]
    L.append(f"{'TOTAL':8s} {t['fresh_items']:11d} {t['fresh_corpus_tokens'] / 1e6:10.2f} {t['repeated_items']:12d} "
             f"{t['repeated_corpus_tokens'] / 1e6:11.2f} {t['items']:11d} {t['corpus_tokens'] / 1e6:10.2f} {1.0:6.3f}")
    waves = ", ".join(f"{k} {v['items']} items {v['corpus_tokens'] / 1e6:.2f}M" for k, v in sorted(summary["by_wave_repeated"].items()))
    L.append(f"by wave: {waves}")
    for d, b in summary["budgets"].items():
        L.append(f"repeat budget {d}: requested {b['requested_tokens'] / 1e6:.2f}M, taken {b['taken_tokens'] / 1e6:.2f}M "
                 f"({b['taken_items']} of {b['pool_items_after_exclusion']} eligible items), "
                 f"shortfall {b['shortfall_tokens'] / 1e6:.2f}M")
    e = summary["exclusions"]
    L.append(f"exclusions: {e['unique_pids']} unique pids from {len(e['files'])} files; repeat-pool rows skipped "
             f"{e['repeat_pool_rows_excluded']}; fresh rows in an exclusion list {e['fresh_rows_in_exclusion']}")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="append", required=True, help="index.jsonl of fresh items (repeatable): every row is kept")
    ap.add_argument("--repeat-index", action="append", default=[], help="index.jsonl of old items to repeat from (repeatable)")
    ap.add_argument("--repeat", action="append", default=[], help="DOMAIN=CORPUS_TOKENS to repeat, e.g. science=8.0e6 (repeatable)")
    ap.add_argument("--exclude-pids", action="append", default=[], help=".txt (one pid per line) or .json list / {'pids': [...]} (repeatable)")
    ap.add_argument("--out", required=True, help="output index.jsonl; build_summary.json goes beside it")
    ap.add_argument("--strict", action="store_true", help="exit 3 if a --repeat budget cannot be met")
    a = ap.parse_args(argv)
    budgets = parse_budgets(a.repeat)
    if budgets and not a.repeat_index:
        raise SystemExit("--repeat needs at least one --repeat-index")
    rows, summary = build(a.fresh, a.repeat_index, budgets, a.exclude_pids, a.out)
    sp = write_outputs(rows, summary, a.out)
    print(table(summary))
    print(f"[fresh-index] {len(rows)} rows -> {os.path.abspath(a.out)}\n[fresh-index] summary -> {sp}")
    if summary["exclusions"]["fresh_rows_in_exclusion"]:
        print(f"[fresh-index] WARNING: {summary['exclusions']['fresh_rows_in_exclusion']} fresh rows are in an exclusion list "
              f"(they were kept)", file=sys.stderr)
    short = {d: b["shortfall_tokens"] for d, b in summary["budgets"].items() if b["shortfall_tokens"] > 0}
    if short:
        print(f"[fresh-index] WARNING: repeat budget not met (tokens short): {short}", file=sys.stderr)
        if a.strict:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
