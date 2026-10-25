#!/usr/bin/env python
"""Judge round, step 3: the corpus items with the TRAINED drafter's first drafts in place of the stored ones.

    /venv/p2/bin/python code/judge_round/swap_drafts.py --index logs/c1b/trainset/waveA/index.jsonl \
        --index logs/c1b/trainset/waveB/index.jsonl --pids logs/phase5/judge/subset_pids.json \
        --drafts logs/phase5/judge/drafts --redrafts logs/phase5/judge/redrafts --out logs/c1b/trainset/p4drafts

For every pid in --pids with a dumped draft (eval_agree.py --trained-only --dump-drafts, draft_sglang's npz format),
the item npz is copied to <out>/items/<h2>/<pid>.npz with `draft` and `draft_conf` replaced (same shape, same drafted
slots, asserted), and its index row is appended to <out>/index.jsonl with `file` pointing at the copy.  Nothing else
in the item changes: tokens, valid, committed (stock's commit order), domain, correctness are the judge's labels and
stay exactly as the 1B corpus wrote them.  The judge trainer (corpus1b/train_c1b.py --index <out>/index.jsonl) then
samples its usual pass-p states, with the new drafter's tokens at the uncommitted slots.

With --redrafts (redraft_items.py's redrafts.shard*.npz: pids, nb, pass_p [sum nb] int16, redraft [sum nb, 32] int32,
rconf [sum nb, 32] uint8), an item that has both a draft and a re-draft record also gets `redraft` [nb, 32] int32 (D1 at
the hole slots, -1 elsewhere), `redraft_conf` [nb, 32] uint8 and `redraft_pass` [nb] int16 (the vanilla pass the
re-draft state was built at, 0 = none), which judge_round/sampler_k3.py reads.  Items with a draft but no re-draft are
written without them; build_summary.json counts both ("with_redraft", "without_redraft").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, CODE)
from drafter_lora import data as D                       # noqa: E402
from drafter_lora.judge_probe import load_drafts         # noqa: E402


def load_redrafts(d: str) -> dict:
    """pid -> (pass_p [nb] int16, redraft [nb, 32] int32, rconf [nb, 32] uint8) from redrafts.shard*.npz."""
    out: dict = {}
    for f in sorted(glob.glob(os.path.join(d, "redrafts.shard*.npz"))):
        z = np.load(f, allow_pickle=False)
        nbs = z["nb"].tolist()
        assert len(z["pass_p"]) == len(z["redraft"]) == len(z["rconf"]) == sum(nbs), f"inconsistent shard {f}"
        off = 0
        for pid, nb in zip(z["pids"].tolist(), nbs):
            assert str(pid) not in out, f"pid {pid} appears in two re-draft shards (second: {f})"
            out[str(pid)] = (z["pass_p"][off:off + nb].astype(np.int16), z["redraft"][off:off + nb].astype(np.int32),
                             z["rconf"][off:off + nb].astype(np.uint8))
            off += nb
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--redrafts", default=None, help="redraft_items.py output dir (redrafts.shard*.npz), optional")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    entries = {str(e["pid"]): e for e in D.load_entries(a.index)}
    pids = D.load_holdout(a.pids)
    drafts = load_drafts(a.drafts)
    redrafts = load_redrafts(a.redrafts) if a.redrafts else {}
    assert redrafts or not a.redrafts, f"--redrafts {a.redrafts}: no redrafts.shard*.npz there"
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    n_ok = n_missing = n_bad = n_rd = 0
    tokens = 0
    with open(os.path.join(out, "index.jsonl"), "w") as fh:
        for pid in pids:
            if pid not in drafts:
                n_missing += 1
                continue
            e = entries[pid]
            d = dict(np.load(e["_path"], allow_pickle=False).items())
            dr, cf = drafts[pid]
            if dr.shape != d["draft"].shape or not np.array_equal(dr >= 0, d["draft"] >= 0):
                n_bad += 1
                continue
            d["draft"] = dr.astype(d["draft"].dtype)
            d["draft_conf"] = cf.astype(np.uint8)
            if pid in redrafts:
                pp, rd, rc = redrafts[pid]
                nb = d["draft"].shape[0]
                assert rd.shape == rc.shape == d["draft"].shape and pp.shape == (nb,), \
                    f"{pid}: re-draft shapes {pp.shape} {rd.shape} {rc.shape} vs draft {d['draft'].shape}"
                d["redraft"], d["redraft_conf"], d["redraft_pass"] = rd, rc, pp
                n_rd += 1
            rel = os.path.join("items", os.path.basename(os.path.dirname(e["_path"])), os.path.basename(e["_path"]))
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **d)
            row = {k: v for k, v in e.items() if not k.startswith("_")}
            row["file"] = rel
            row["draft_source"] = "phase4_merged_dr3d174340"
            fh.write(json.dumps(row) + "\n")
            n_ok += 1
            tokens += int(e.get("corpus_tokens", 0))
    summary = {"items": n_ok, "missing_draft": n_missing, "shape_mismatch": n_bad, "corpus_tokens": tokens,
               "with_redraft": n_rd, "without_redraft": n_ok - n_rd,
               "redrafts": os.path.abspath(a.redrafts) if a.redrafts else None,
               "index": os.path.join(out, "index.jsonl")}
    json.dump(summary, open(os.path.join(out, "build_summary.json"), "w"), indent=1)
    print(f"[swap] {n_ok} items ({tokens / 1e6:.1f}M corpus tokens), {n_missing} without a draft, {n_bad} shape mismatches, "
          f"{n_rd} with a re-draft / {n_ok - n_rd} without -> {out}")
    return 0 if n_ok and not n_bad else 1


if __name__ == "__main__":
    sys.exit(main())
