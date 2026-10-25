#!/usr/bin/env python
"""Judge round, step 1: a training subset of the corpus whose corpus tokens cover the round's budget.

    /venv/p2/bin/python code/judge_round/select_subset.py --index logs/c1b/trainset/waveA/index.jsonl \
        --index logs/c1b/trainset/waveB/index.jsonl --exclude logs/phase4/train/main/holdout_pids.json \
        --tokens 34e6 --out logs/phase5/judge/subset_pids.json

Items are drawn per domain in the corpus' own proportions (by corpus tokens), shuffled with numpy default_rng(seed),
never from the Phase 4 held-out set, until the token budget is met.  The output has the same {"pids": [...]} shape as
holdout_pids.json, so eval_agree.py --holdout-pids reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, CODE)
from drafter_lora import data as D   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--exclude", default=None, help="holdout_pids.json: never pick these")
    ap.add_argument("--tokens", type=float, default=34e6, help="corpus tokens (prompt + answer) to cover")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    entries = D.load_entries(a.index)
    excl = set(D.load_holdout(a.exclude)) if a.exclude else set()
    pool = [e for e in entries if str(e["pid"]) not in excl and int(e.get("usable", 1))]
    by_dom = defaultdict(list)
    for e in pool:
        by_dom[e["domain"]].append(e)
    tok_dom = {d: sum(int(e["corpus_tokens"]) for e in es) for d, es in by_dom.items()}
    tot = sum(tok_dom.values())
    rng = np.random.default_rng(a.seed)
    chosen, got = [], {}
    for d, es in sorted(by_dom.items()):
        budget = a.tokens * tok_dom[d] / tot
        order = rng.permutation(len(es))
        s = 0
        for i in order:
            if s >= budget:
                break
            chosen.append(str(es[i]["pid"])); s += int(es[i]["corpus_tokens"])
        got[d] = {"items": sum(1 for p in chosen if p in {str(x["pid"]) for x in es}), "corpus_tokens": s}
    chosen.sort()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"pids": chosen, "per_domain": got, "tokens_target": a.tokens, "seed": a.seed,
               "excluded": len(excl), "pool_items": len(pool)}, open(a.out, "w"), indent=1)
    print(f"[subset] {len(chosen)} items, {sum(v['corpus_tokens'] for v in got.values()) / 1e6:.1f}M corpus tokens -> {a.out}")
    for d, v in sorted(got.items()):
        print(f"  {d:8s} {v['items']:7d} items {v['corpus_tokens'] / 1e6:6.1f}M tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
