#!/usr/bin/env python
"""Kept generations + teacher-forced drafts -> the training index the judge trainer reads.

    /venv/train/bin/python code/corpus1b/to_trainset.py --wave A

Reads logs/c1b/gen/wave<W>/gpu*/records.*.jsonl.gz (gen_sglang.py; kept rows carry prompt_ids and every block's
32 token ids + the stock pass that committed each slot) and logs/c1b/draft/wave<W>/gpu*/drafts.*.npz
(draft_sglang.py), and writes, per kept item, one npz in the exact format of draftalign/build_train_set.py
(validated by its own build_item) plus two 1B-corpus arrays:

    prompt_ids tokens valid committed n_pass block_start draft has_draft accepted usable prompt_length
    block_length correct idx      (build_train_set)
    draft_conf [nb, 32] uint8     drafter confidence, round(255 p)          (1B corpus)
    domain                        the item's domain                         (1B corpus)

to logs/c1b/trainset/wave<W>/items/<h2>/<pid>.npz, and one line per item to logs/c1b/trainset/wave<W>/index.jsonl
with, besides build_train_set's fields, `domain`, `pid`, `wave`, `source`, `corpus_tokens` (prompt + kept answer:
the unit of the 1B plan) and `n_gen`.

Slot validity: the prompt tail of block 0 is not generated (invalid); in the last block, every slot AFTER the
first EOS is invalid (never a loss row). committed[b, p, j] = the stock pass p committed slot j.
`--max-seq` 1,856 keeps every block of a <= 800-token prompt + <= 1,024-token answer usable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402
import research_protocol as P                                            # noqa: E402
from draftalign.build_train_set import build_item                        # noqa: E402

BLOCK = C.BLOCK


def load_drafts(draft_root: str) -> Dict[str, Tuple[str, int, int]]:
    """pid -> (npz path, row offset, nb) over every drafts.*.npz (arrays are read lazily, one file at a time)."""
    where: Dict[str, Tuple[str, int, int]] = {}
    for path in sorted(glob.glob(os.path.join(draft_root, "gpu*", "drafts.*.npz"))):
        z = np.load(path, allow_pickle=True)
        off = 0
        for pid, nb in zip(z["pids"].tolist(), z["nb"].tolist()):
            where[str(pid)] = (path, off, int(nb))
            off += int(nb)
    return where


class DraftCache:
    """Keeps the most recently used drafts file in memory (records and drafts are both roughly pid-ordered)."""

    def __init__(self):
        self.path, self.z = None, None

    def get(self, path: str, off: int, nb: int) -> Tuple[np.ndarray, np.ndarray]:
        if path != self.path:
            z = np.load(path, allow_pickle=True)
            self.z = {"draft": z["draft"], "conf": z["conf"]}
            self.path = path
        return self.z["draft"][off:off + nb], self.z["conf"][off:off + nb]


def item_arrays(rec: dict) -> dict:
    """The trace-shaped arrays of one kept generation record (what build_item validates)."""
    prompt_ids = [int(x) for x in rec["prompt_ids"]]
    L = len(prompt_ids)
    lo0 = int(rec["lo0"])
    assert lo0 == (L // BLOCK) * BLOCK, (rec["pid"], lo0, L)
    nb = len(rec["blocks"])
    tokens = np.asarray([b["ids"] for b in rec["blocks"]], np.int32)
    steps = np.asarray([b["steps"] for b in rec["blocks"]], np.int32)
    assert tokens.shape == (nb, BLOCK) and steps.shape == (nb, BLOCK), rec["pid"]
    valid = np.ones((nb, BLOCK), bool)
    tail = L - lo0
    valid[0, :tail] = False
    # slots after the first EOS of the answer are not trained on
    gen = tokens.reshape(-1)[tail:]
    e = C.first_eos(gen.tolist())
    if e >= 0:
        cut = tail + e + 1                                   # flat position just after the EOS
        flat_valid = valid.reshape(-1)
        flat_valid[cut:] = False
        valid = flat_valid.reshape(nb, BLOCK)
    n_pass = (steps.max(axis=1) + 1).astype(np.int32)
    Pn = int(max(1, n_pass.max()))
    committed = np.zeros((nb, Pn, BLOCK), bool)
    bi, sj = np.nonzero(valid)
    committed[bi, steps[bi, sj], sj] = True
    return {"prompt_ids": prompt_ids, "trace": {
        "tokens": tokens, "valid": valid, "committed": committed, "n_pass": n_pass,
        "block_start": (lo0 + BLOCK * np.arange(nb)).astype(np.int32),
        "prompt_length": np.int32(L), "block_length": np.int32(BLOCK)}}


def iter_kept(gen_root: str, domains=None) -> Iterator[dict]:
    for path in sorted(glob.glob(os.path.join(gen_root, "gpu*", "records.*.jsonl.gz"))):
        for r in C.read_jsonl(path):
            if r.get("kept") and (domains is None or r.get("domain") in domains):
                yield r


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", choices=("A", "B", "C"), required=True)
    ap.add_argument("--gen", default=None, help="default logs/c1b/gen/wave<W>")
    ap.add_argument("--draft", default=None, help="default logs/c1b/draft/wave<W>")
    ap.add_argument("--out", default=None, help="default logs/c1b/trainset/wave<W>")
    ap.add_argument("--max-seq", type=int, default=1856)
    ap.add_argument("--domains", default=None,
                    help="only these domains' kept records (comma-separated): the next domain may still be generating")
    ap.add_argument("--require-draft", action="store_true", default=True,
                    help="drop items whose drafts are missing (default; every kept item should have them)")
    args = ap.parse_args(argv)

    gen_root = P.safe_out(args.gen or os.path.join("logs", "c1b", "gen", f"wave{args.wave}"))
    draft_root = P.safe_out(args.draft or os.path.join("logs", "c1b", "draft", f"wave{args.wave}"))
    out = P.safe_out(args.out or os.path.join("logs", "c1b", "trainset", f"wave{args.wave}"))
    os.makedirs(out, exist_ok=True)
    index_path = os.path.join(out, "index.jsonl")
    done = set()
    if os.path.exists(index_path):                                   # resume: keep what is already written
        done = {e["pid"] for e in C.read_jsonl(index_path)}
    where = load_drafts(draft_root)
    cache = DraftCache()
    import postdecontam                                               # late decontamination (math_test): never trained on
    late = postdecontam.all_flagged()
    counts = {"kept_records": 0, "written": len(done), "no_draft": 0, "bad": 0, "no_usable_block": 0,
              "excluded_late_decontam": 0}
    tokens = 0
    idx = len(done)
    with open(index_path, "a") as fh:
        for rec in iter_kept(gen_root, set(args.domains.split(",")) if args.domains else None):
            counts["kept_records"] += 1
            pid = rec["pid"]
            if pid in done:
                continue
            if pid in late:
                counts["excluded_late_decontam"] += 1
                continue
            if pid not in where:
                counts["no_draft"] += 1
                if args.require_draft:
                    continue
                drafts = conf = None
            else:
                drafts, conf = cache.get(*where[pid])
            try:
                a = item_arrays(rec)
                item = build_item({"idx": idx, "correct": 1, "split": f"c1b_{rec['domain']}"}, a["trace"],
                                  a["prompt_ids"], drafts, args.max_seq, BLOCK)
            except AssertionError as e:
                counts["bad"] += 1
                print(f"[skip] {pid}: {e}", flush=True)
                continue
            if not bool(item["usable"].any()):
                counts["no_usable_block"] += 1
                continue
            item["draft_conf"] = (np.zeros_like(item["draft"], dtype=np.uint8) if conf is None
                                  else np.asarray(conf, np.uint8))
            item["domain"] = np.array(rec["domain"])
            h2 = C.sha256_hex(pid)[:2]
            rel = os.path.join("items", h2, f"{pid}.npz")
            os.makedirs(os.path.join(out, "items", h2), exist_ok=True)
            np.savez_compressed(os.path.join(out, rel), **item)
            corpus_tokens = int(rec["prompt_len"]) + int(rec["n_gen"])
            tokens += corpus_tokens
            entry = {"file": rel, "idx": idx, "split": f"c1b_{rec['domain']}", "source": rec.get("source"),
                     "domain": rec["domain"], "pid": pid, "wave": args.wave, "correct": 1,
                     "n_blocks": int(item["tokens"].shape[0]), "usable_blocks": int(item["usable"].sum()),
                     "draft_blocks": int((item["has_draft"] & item["usable"]).sum()),
                     "usable": [bool(v) for v in item["usable"]],
                     "has_draft": [bool(v) for v in item["has_draft"]],
                     "block_start": [int(v) for v in item["block_start"]],
                     "seq_len": int(item["block_start"][-1]) + BLOCK,
                     "prompt_length": int(item["prompt_length"]),
                     "corpus_tokens": corpus_tokens, "n_gen": int(rec["n_gen"])}
            fh.write(json.dumps(entry) + "\n")
            idx += 1
            counts["written"] += 1
    summary = {"wave": args.wave, "domains": args.domains, "out": out, "counts": counts, "corpus_tokens_this_run": tokens}
    with open(os.path.join(out, "build_summary.json"), "w") as f2:
        json.dump(summary, f2, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
