#!/usr/bin/env python
"""Per-domain holdout diagnostics for one judge checkpoint, run on an EVAL GPU (recipe §4.2), no trainer edit.

Rebuilds exactly the trainer's held-out item set (same index, seed and --holdout-items -> same
StateSampler.split_holdout), then runs train.evaluate() separately on each domain (the index entries'
`source` field: gsm8k / opencode / if). Prints one JSON line per domain with the same keys the trainer
logs (argmax accuracy, top1>tau precision, mean top1) so the curves are directly comparable.

    CUDA_VISIBLE_DEVICES=6 /venv/r3/bin/python code/draftalign/eval_holdout_r3.py \
        --index logs/draftalign/trainset_r3/index.jsonl --seed 1 --holdout-items 150 \
        --adapter logs/draftalign/train/r3_A/step000100 --out logs/draftalign/probes_r3/r3_A/holdout.jsonl
"""
from __future__ import annotations

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # code/
for p in (os.path.join(HERE, "compat"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch                                                                # noqa: E402
import research_protocol as P                                               # noqa: E402
from draftalign import adapter as A                                         # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                  # noqa: E402
from draftalign.train import evaluate                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--holdout-items", type=int, default=150)
    ap.add_argument("--p-pass1", type=float, default=0.4); ap.add_argument("--p-allmask", type=float, default=0.0)
    ap.add_argument("--p-draftfull", type=float, default=0.3); ap.add_argument("--max-seq", type=int, default=1200)
    ap.add_argument("--eval-states", type=int, default=64); ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--model", default="b32")
    args = ap.parse_args()
    scfg = SamplerConfig(index_path=args.index, seed=args.seed, p_pass1=args.p_pass1, p_allmask=args.p_allmask,
                         p_draftfull=args.p_draftfull, mask_id=P.MASK_ID, max_seq=args.max_seq)
    full = StateSampler(scfg)
    _, eval_s = full.split_holdout(args.holdout_items)                      # identical to the trainer's holdout
    by_dom = {}
    for e in eval_s.entries:
        by_dom.setdefault(e.get("source", "?"), []).append(e)
    device = torch.device("cuda:0")
    tok, model = P.load_sdar(args.model, str(device))
    info = A.load_adapter(model, args.adapter)                              # un-merged: flag-gated adapter
    model.eval(); model.config.use_cache = False
    step = int(os.path.basename(args.adapter.rstrip("/")).replace("step", "") or 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        for dom, entries in sorted(by_dom.items()):
            s = StateSampler(scfg, entries, eval_s.root)
            ev = evaluate(model, s, args.eval_states, 0, "full", P.MASK_ID, device, tau=args.tau, seed=0,
                          draft_flag=A.is_flag_gated(model))
            row = {"step": step, "adapter": args.adapter, "domain": dom, "holdout_items": len(entries),
                   "holdout_examples": len(s), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
            fh.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
