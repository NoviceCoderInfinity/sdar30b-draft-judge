#!/usr/bin/env python
"""Progress counts for the 128M -> 228M continuation (CPU; run from the repo root with /venv/train/bin/python).

    python code/corpus1b/ops/c228_count.py [domain]      # prints JSON; with a domain, only that domain

  generated_M[d]  kept corpus tokens (prompt + answer) of wave-B records written since this run started
                  (record files whose mtime >= START in logs/c1b/c228_state/start.txt)
  converted_M[d]  corpus tokens of wave-B training items NOT in index_128m (all new conversions, incl. 30 Sep code)
  drafted_frac    share of kept wave-B records that have drafts (done.txt of logs/c1b/draft/waveB)
"""
import collections as C
import glob
import json
import os
import sys

sys.path.insert(0, "code/corpus1b")
import common as CM                                                  # noqa: E402


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    start = float(open("logs/c1b/c228_state/start.txt").read().split()[0])
    gen, kept_all = C.Counter(), 0
    for f in glob.glob("logs/c1b/gen/waveB/gpu*/records.*.jsonl.gz"):
        new = os.path.getmtime(f) >= start
        for r in CM.read_jsonl(f):
            if not r.get("kept") or (only and r.get("domain") != only):
                continue
            kept_all += 1
            if new:
                gen[r["domain"]] += int(r.get("prompt_len", 0)) + int(r.get("n_gen", 0))
    drafted = set()
    for f in glob.glob("logs/c1b/draft/waveB/gpu*/done.txt"):
        drafted.update(open(f).read().split())
    in128 = set()
    if os.path.exists("logs/c1b/trainset/index_128m.jsonl"):
        in128 = {json.loads(l)["pid"] for l in open("logs/c1b/trainset/index_128m.jsonl") if l.strip()}
    conv = C.Counter()
    if os.path.exists("logs/c1b/trainset/waveB/index.jsonl"):
        for l in open("logs/c1b/trainset/waveB/index.jsonl"):
            if not l.strip():
                continue
            e = json.loads(l)
            if e["pid"] not in in128 and (not only or e.get("domain") == only):
                conv[e.get("domain")] += int(e.get("corpus_tokens", 0))
    out = {"generated_M": {d: round(v / 1e6, 3) for d, v in gen.items()},
           "converted_M": {d: round(v / 1e6, 3) for d, v in conv.items()},
           "converted_total_M": round(sum(conv.values()) / 1e6, 3),
           "drafted_records": len(drafted), "kept_records": kept_all}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
