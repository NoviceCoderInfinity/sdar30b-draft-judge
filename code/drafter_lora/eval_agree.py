#!/usr/bin/env python
"""Gate 1 (offline) and the smoke test's parity check: the s8 first-draft replay on the held-out items.

    # one shard per GPU (p4.sh gate1 runs k = 0..7, then --aggregate)
    CUDA_VISIBLE_DEVICES=k python code/drafter_lora/eval_agree.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --holdout-pids logs/phase4/train/main/holdout_pids.json --trained logs/phase4/train/main/final_adapter \
        --redraft-probe --dump-drafts logs/phase4/gate1/drafts --shard k --nshards 8 --out logs/phase4/gate1/shards/shard_k.json
    python code/drafter_lora/eval_agree.py --aggregate logs/phase4/gate1/shards --out results/phase4_drafter/gate1.json

    # smoke parity: the UNTRAINED drafter's HF replay vs the stored SGLang drafts, 200 held-out items (>= 95%)
    ... eval_agree.py ... --parity-only --n-items 200 --shard k --nshards 8 --out logs/phase4/smoke/parity/shard_k.json

Both drafters go through the SAME code (drafter_lora/replay.py: SeqCache with the clean 30B prefix, draft_block s8).
`--trained` is either an adapter directory (drafter_lora/lora.py format: the base is loaded once and the adapter is
switched off for the untrained pass -- bit-identical to the stock drafter) or a merged full HF directory (then the
stock d32 is loaded as a second model).

Reported, overall and per domain (counts summed over shards, ratios computed once):
  first-draft agreement with T (untrained, trained, delta in points); retention = share of the slots the untrained
  drafter got right that the trained one still gets right; re-draft probe (holes re-drafted from the true tokens of
  the first-draft-correct slots), agreement and repeat-of-the-rejected-token rate, with and without the ban; parity
  of the untrained replay with the stored drafts; agreement of the stored drafts themselves.
Gate 1 (runbook section 6): delta >= +3.0 points (target +7); retention >= 97% (fail below 95%).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

from drafter_lora import data as D                                       # noqa: E402
from drafter_lora import lora as LA                                      # noqa: E402
from drafter_lora.replay import parity_with_stored, replay_item, stored_agreement   # noqa: E402

COUNT_KEYS = ("items", "slots", "agree0", "agree1", "right0", "kept", "par_slots", "par_equal", "stored_slots",
              "stored_agree", "rd_holes0", "rd_agree0", "rd_repeat0", "rd_agree_ban0", "rd_repeat_ban0",
              "rd_holes1", "rd_agree1", "rd_repeat1", "rd_agree_ban1", "rd_repeat_ban1", "fwd0", "fwd1")


def _add(acc: Dict[str, Dict[str, float]], dom: str, row: Dict[str, float]) -> None:
    for d in (dom, "all"):
        a = acc.setdefault(d, {k: 0.0 for k in COUNT_KEYS})
        for k, v in row.items():
            a[k] += float(v)


def load_models(trained: str, device: str, model_key: str = "d32"):
    """(untrained_runner, trained_runner, info): each runner is (model, adapter_off?)."""
    tok, base = P.load_sdar(model_key, device)
    base.eval()
    if trained is None:
        return (base, False), None, {"trained": None}
    if os.path.isfile(os.path.join(trained, LA.ADAPTER_JSON)):
        meta = LA.load(base, trained)
        base.eval()
        return (base, True), (base, False), {"trained": os.path.abspath(trained), "kind": "adapter", "meta": meta}
    from draftalign.eval_drafter_agreement import load_drafter
    _t, merged, info = load_drafter(trained, device)
    merged.eval()
    return (base, False), (merged, False), {"trained": os.path.abspath(trained), "kind": "merged", **info}


def run(model_off, item, device, probe):
    model, off = model_off
    if off:
        with LA.adapter_off((1, 1), device):
            return replay_item(model, item, device, redraft_probe=probe)
    return replay_item(model, item, device, redraft_probe=probe)


def shard_main(args) -> dict:
    device = args.device
    entries = D.load_entries(args.index)
    by_pid = {str(e["pid"]): e for e in entries}
    pids = D.load_holdout(args.holdout_pids)
    if args.n_items:
        rng = np.random.default_rng(args.seed)
        pids = sorted(pids[int(i)] for i in rng.choice(len(pids), min(args.n_items, len(pids)), replace=False))
    mine = pids[args.shard::args.nshards]
    m0, m1, info = load_models(None if args.parity_only else args.trained, device)
    acc: Dict[str, Dict[str, float]] = {}
    dump = {"pids": [], "nb": [], "draft": [], "conf": []}
    t0 = time.time()
    for n, pid in enumerate(mine):
        it = D.load_item(by_pid[pid])
        dom = it["domain"]
        r0 = run(m0, it, device, args.redraft_probe and not args.parity_only)
        par = parity_with_stored(it, r0["drafts"])
        sa = stored_agreement(it)
        row = {"items": 1, "slots": int(r0["valid"].sum()), "agree0": int(r0["right"].sum()),
               "par_slots": par["slots"], "par_equal": par["equal"], "stored_slots": sa["slots"],
               "stored_agree": sa["agree"], "fwd0": r0["n_fwd"]}
        if r0["redraft"]:
            rd = r0["redraft"]
            row.update({"rd_holes0": rd["holes"], "rd_agree0": rd["agree"], "rd_repeat0": rd["repeat"],
                        "rd_agree_ban0": rd["agree_ban"], "rd_repeat_ban0": rd["repeat_ban"]})
        if m1 is not None:
            r1 = run(m1, it, device, args.redraft_probe)
            row.update({"agree1": int(r1["right"].sum()), "right0": int(r0["right"].sum()),
                        "kept": int((r0["right"] & r1["right"]).sum()), "fwd1": r1["n_fwd"]})
            if r1["redraft"]:
                rd = r1["redraft"]
                row.update({"rd_holes1": rd["holes"], "rd_agree1": rd["agree"], "rd_repeat1": rd["repeat"],
                            "rd_agree_ban1": rd["agree_ban"], "rd_repeat_ban1": rd["repeat_ban"]})
            if args.dump_drafts:
                dump["pids"].append(pid); dump["nb"].append(r1["drafts"].shape[0])
                dump["draft"].append(r1["drafts"].astype(np.int32)); dump["conf"].append(r1["conf"])
        _add(acc, dom, row)
        if (n + 1) % 25 == 0:
            a = acc["all"]
            msg = f"untrained {a['agree0'] / max(a['slots'], 1):.4f}"
            if m1 is not None:
                msg += f" trained {a['agree1'] / max(a['slots'], 1):.4f}"
            print(f"[shard {args.shard}] {n + 1}/{len(mine)} {msg} ({time.time() - t0:.0f}s)", flush=True)
    if args.dump_drafts and dump["pids"]:
        dd = P.safe_out(args.dump_drafts)
        os.makedirs(dd, exist_ok=True)
        np.savez_compressed(os.path.join(dd, f"drafts.shard{args.shard}of{args.nshards}.npz"),
                            pids=np.array(dump["pids"]), nb=np.array(dump["nb"], np.int32),
                            draft=np.concatenate(dump["draft"]), conf=np.concatenate(dump["conf"]).astype(np.uint8))
    res = {"shard": args.shard, "nshards": args.nshards, "n_items": len(mine), "counts": acc, "info": info,
           "parity_only": bool(args.parity_only), "redraft_probe": bool(args.redraft_probe),
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return res


def ratios(c: Dict[str, float]) -> Dict[str, float]:
    f = lambda a, b: (c[a] / c[b]) if c.get(b) else float("nan")            # noqa: E731
    out = {"items": c["items"], "slots": c["slots"],
           "agreement_untrained": f("agree0", "slots"), "agreement_trained": f("agree1", "slots"),
           "retention": f("kept", "right0"), "parity_with_stored": f("par_equal", "par_slots"),
           "stored_draft_agreement": f("stored_agree", "stored_slots"),
           "redraft_agreement_untrained": f("rd_agree0", "rd_holes0"),
           "redraft_repeat_untrained": f("rd_repeat0", "rd_holes0"),
           "redraft_agreement_untrained_ban": f("rd_agree_ban0", "rd_holes0"),
           "redraft_agreement_trained": f("rd_agree1", "rd_holes1"),
           "redraft_repeat_trained": f("rd_repeat1", "rd_holes1"),
           "redraft_agreement_trained_ban": f("rd_agree_ban1", "rd_holes1"),
           "redraft_repeat_trained_ban": f("rd_repeat_ban1", "rd_holes1")}
    out["delta_points"] = 100 * (out["agreement_trained"] - out["agreement_untrained"])
    return out


def aggregate(args) -> dict:
    files = sorted(glob.glob(os.path.join(args.aggregate, "shard_*.json")))
    assert files, f"no shard_*.json under {args.aggregate}"
    shards = [json.load(open(f)) for f in files]
    ns = {s["nshards"] for s in shards}
    assert len(ns) == 1 and len(shards) == ns.pop(), f"{len(shards)} shard files for nshards {ns}"
    acc: Dict[str, Dict[str, float]] = {}
    for s in shards:
        for d, row in s["counts"].items():
            a = acc.setdefault(d, {k: 0.0 for k in COUNT_KEYS})
            for k, v in row.items():
                a[k] += v
    rep = {"overall": ratios(acc["all"]), "per_domain": {d: ratios(c) for d, c in sorted(acc.items()) if d != "all"},
           "shards": len(shards), "info": shards[0].get("info"), "seconds_max_shard": max(s["seconds"] for s in shards)}
    o = rep["overall"]
    if shards[0].get("parity_only"):
        ok = o["parity_with_stored"] >= args.parity_min
        rep["verdict"] = {"check": "smoke parity (untrained HF replay vs stored SGLang drafts)",
                          "parity": o["parity_with_stored"], "min": args.parity_min, "pass": bool(ok)}
    else:
        d, r = o["delta_points"], o["retention"]
        status = ("FAIL" if (d < args.min_delta_points or r < args.retention_fail) else
                  "ASK" if r < args.retention_min else "PASS")
        rep["verdict"] = {"gate": 1, "delta_points": d, "min_delta_points": args.min_delta_points,
                          "target_delta_points": 7.0, "retention": r, "retention_min": args.retention_min,
                          "retention_fail_below": args.retention_fail, "status": status}
    return rep


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", default=[])
    ap.add_argument("--holdout-pids", default=None, help="<train out>/holdout_pids.json")
    ap.add_argument("--trained", default=None, help="adapter dir (drafter_lora) or merged HF dir")
    ap.add_argument("--parity-only", action="store_true", help="untrained replay vs stored drafts only")
    ap.add_argument("--redraft-probe", action="store_true")
    ap.add_argument("--dump-drafts", default=None, help="dir for the trained drafter's drafts (Gate 2 input)")
    ap.add_argument("--n-items", type=int, default=0, help="a seeded subset of the held-out items (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--aggregate", default=None, help="a dir of shard JSONs -> one report")
    ap.add_argument("--min-delta-points", type=float, default=3.0)
    ap.add_argument("--retention-min", type=float, default=0.97)
    ap.add_argument("--retention-fail", type=float, default=0.95)
    ap.add_argument("--parity-min", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.aggregate:
        res = aggregate(args)
    else:
        assert args.index and args.holdout_pids, "--index and --holdout-pids are required for a shard"
        assert args.parity_only or args.trained, "--trained is required (or --parity-only)"
        res = shard_main(args)
    out = P.safe_out(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res.get("verdict") or {k: v for k, v in res.items() if k != "counts"}, indent=2), flush=True)
    print(f"[out] {out}", flush=True)
    v = res.get("verdict") or {}
    return 0 if (v.get("pass", True) and v.get("status", "PASS") == "PASS") else 3


if __name__ == "__main__":
    sys.exit(main())
