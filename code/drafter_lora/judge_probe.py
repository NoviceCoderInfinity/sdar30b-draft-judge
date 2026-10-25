#!/usr/bin/env python
"""Gate 2: the FROZEN judge's commit precision on the trained drafter's first drafts vs on the stored (untrained) ones.

    torchrun --standalone --nproc_per_node=8 code/drafter_lora/judge_probe.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --holdout-pids logs/phase4/train/main/holdout_pids.json --drafts logs/phase4/gate1/drafts \
        --adapter logs/phase2/ref/c1b_128M/tok_000128000000 --out results/phase4_drafter/gate2.json

The held-out pass-1 states are built twice by draftalign/dataset.py's StateSampler (regime draft_pass1: every open slot
holds the draft, nothing committed), once with the corpus' stored drafts and once with the trained drafter's drafts
(dumped by eval_agree.py --dump-drafts, same items, same s8 replay), and the judge -- SDAR-30B + the 128M LoRA,
flag-gated, never merged -- is probed exactly as corpus1b/train_c1b.py's probe() does: per domain, the share of rows
with top-1 > 0.95 (commit share) and the precision of those rows.  Both probes draw the same examples (same entries,
same seed), so the comparison is paired.

Both waves number items from idx 0; the sampler and train_fast.group_states key items by (split, idx), so the
held-out entries are re-numbered 0..n-1 here before the sampler sees them.

Gate 2 (runbook section 6): commit precision within --max-precision-drop (0.5 points) of the stored-draft value, and
the commit share should rise.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign import train_fast as TF                                  # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler               # noqa: E402
from draftalign.train_drafter import dist_init                           # noqa: E402
from drafter_lora import data as D                                       # noqa: E402


def load_drafts(d: str) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for f in sorted(glob.glob(os.path.join(d, "drafts.shard*.npz"))):
        z = np.load(f)
        off = 0
        for pid, nb in zip(z["pids"].tolist(), z["nb"].tolist()):
            out[str(pid)] = (z["draft"][off:off + nb].astype(np.int32), z["conf"][off:off + nb].astype(np.uint8))
            off += nb
    return out


class SwappedDrafts(StateSampler):
    """StateSampler whose items carry another drafter's first drafts (and confidences)."""

    def __init__(self, cfg, entries, drafts: Dict[str, tuple]):
        self._swap = drafts
        super().__init__(cfg, entries)

    def _item(self, i: int) -> dict:
        d = dict(super()._item(i))
        dr, cf = self._swap[str(self.entries[i]["pid"])]
        assert dr.shape == d["draft"].shape, (self.entries[i]["pid"], dr.shape, d["draft"].shape)
        assert np.array_equal(dr >= 0, d["draft"] >= 0), "drafted slots differ from the stored drafts'"
        d["draft"], d["draft_conf"] = dr, cf
        return d


def heldout_entries(index_paths: List[str], pids: List[str]) -> List[dict]:
    keep = set(pids)
    rows = [e for e in D.load_entries(index_paths) if str(e["pid"]) in keep]
    rows.sort(key=lambda e: str(e["pid"]))
    out = []
    for j, e in enumerate(rows):
        e = dict(e)
        e["file"] = e["_path"]                         # absolute: StateSampler joins only relative paths
        e["idx"] = j                                   # unique across waves (see the docstring)
        out.append(e)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--holdout-pids", required=True)
    ap.add_argument("--drafts", required=True, help="eval_agree.py --dump-drafts directory")
    ap.add_argument("--adapter", default="logs/phase2/ref/c1b_128M/tok_000128000000")
    ap.add_argument("--model", default="b32")
    ap.add_argument("--n-per-domain", type=int, default=256, help="probe states per domain (train_c1b default)")
    ap.add_argument("--seed", type=int, default=1, help="train_c1b's default probe seed")
    ap.add_argument("--max-seq", type=int, default=1856, help="train_c1b's default")
    ap.add_argument("--max-precision-drop", type=float, default=0.5, help="points")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import train_c1b as TC                                         # corpus1b/train_c1b.py: probe()
    info = dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    pids = D.load_holdout(args.holdout_pids)
    drafts = load_drafts(P.safe_out(args.drafts))
    missing = [p for p in pids if p not in drafts]
    assert not missing, f"{len(missing)} held-out items have no dumped draft (first: {missing[:3]})"
    entries = heldout_entries(args.index, pids)
    cfg = SamplerConfig(index_path="(phase4 held-out)", seed=args.seed, max_seq=args.max_seq)
    stored = StateSampler(cfg, entries)
    swapped = SwappedDrafts(cfg, entries, drafts)
    assert stored.examples == swapped.examples

    _tok, model = P.load_sdar(args.model, str(device))
    meta = A.load_adapter(model, P.safe_out(args.adapter))["meta"]
    model.eval()
    conf_embed = bool(meta.get("conf_embed", False))
    lcfg = TF.LossCfg()
    t0 = time.time()
    res = {}
    for tag, s in (("stored", stored), ("trained", swapped)):
        res[tag] = TC.probe(model, s, lcfg, args.n_per_domain, 32, args.seed, rank, world, device, conf_embed)
        if rank == 0:
            print(f"[{tag}] commit share {res[tag]['all/conf_frac']:.4f} precision "
                  f"{res[tag]['all/conf_precision']:.4f} acc {res[tag]['all/acc']:.4f}", flush=True)
    if rank == 0:
        a, b = res["stored"], res["trained"]
        drop = 100 * (a["all/conf_precision"] - b["all/conf_precision"])
        status = "PASS" if drop <= args.max_precision_drop else "FAIL"
        verdict = {"gate": 2, "precision_stored": a["all/conf_precision"], "precision_trained": b["all/conf_precision"],
                   "precision_drop_points": drop, "max_drop_points": args.max_precision_drop,
                   "commit_share_stored": a["all/conf_frac"], "commit_share_trained": b["all/conf_frac"],
                   "commit_share_rises": bool(b["all/conf_frac"] > a["all/conf_frac"]), "status": status}
        rep = {"verdict": verdict, "stored": a, "trained": b, "adapter": os.path.abspath(args.adapter),
               "adapter_meta": meta, "n_heldout_items": len(entries), "examples": len(stored.examples),
               "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out = P.safe_out(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
        print(json.dumps(verdict, indent=2), flush=True)
        rc = 0 if status == "PASS" else 3
    else:
        rc = 0
    if world > 1:
        rc_t = torch.tensor([rc], device=device)
        torch.distributed.broadcast(rc_t, src=0)
        rc = int(rc_t)
        torch.distributed.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
