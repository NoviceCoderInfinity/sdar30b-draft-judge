#!/usr/bin/env python
"""Phase 2, step 4: the calibration probe of a judge adapter on one drafter's drafts (one GPU).

Loads the pinned stock 30B model + the adapter (default: the S0 judge, r3_B step 710), builds round 3's held-out
items from a (re-drafted) index -- `split_holdout(150)` at seed 1, the items r3_B never trained on -- and runs
`p2probe.probe` on draft-filled pass-1 states. Writes <out>/probe.json and checks the "probe" stop rules
(commit precision < 0.99 is a WARNING here: the flowchart goes on to the zero-shot decode either way).

    CUDA_VISIBLE_DEVICES=0 /venv/train/bin/python code/phase2/probe_p2.py \
        --index logs/phase2/redraft/F1/index.jsonl --adapter logs/phase2/ref/r3_B/step000710 \
        --out logs/phase2/probe/F1__r3_B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import torch                                                             # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign import train_fast as F                                   # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler               # noqa: E402
from phase2 import p2probe                                               # noqa: E402
from phase2.stoprules import STOP_EXIT, StopMonitor                      # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="b32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout-items", type=int, default=150)
    ap.add_argument("--states-per-domain", type=int, default=768)
    ap.add_argument("--max-seq", type=int, default=1200)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--calib", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    scfg = SamplerConfig(index_path=P.safe_out(args.index), seed=args.seed, p_pass1=0.4, p_allmask=0.0,
                         p_draftfull=0.3, mask_id=P.MASK_ID, max_seq=args.max_seq)
    full = StateSampler(scfg)
    _, eval_s = full.split_holdout(args.holdout_items)
    print(f"[probe] {len(full.entries)} items in the index, {len(eval_s.entries)} held out", flush=True)
    t0 = time.time()
    _tok, model = P.load_sdar(args.model, args.device)
    got = A.load_adapter(model, P.safe_out(args.adapter))
    model.eval()
    lcfg = F.LossCfg(beta=args.beta, temp=0.5, calib=args.calib, mask_id=P.MASK_ID)
    # the experts the judge routes to on these states (prefix pass and the draft-filled block pass, per layer)
    from phase2.explog import TrainExpertCapture
    import numpy as np
    cap = TrainExpertCapture(model, torch.device(args.device))
    cap.install()
    try:
        with cap.armed_scope():
            pr = p2probe.probe(model, eval_s, lcfg, args.states_per_domain, 32, args.seed, 0, 1,
                               torch.device(args.device), tau=args.tau)
        win = cap.window(1)
    finally:
        cap.uninstall()
    np.savez_compressed(os.path.join(out, "experts.npz"), prefix=win["prefix"], block=win["block"])
    pr.update({k: v for k, v in TrainExpertCapture.metrics(win).items()})
    res = {"index": args.index, "adapter": args.adapter, "adapter_meta": got["meta"], "holdout_items": len(eval_s.entries),
           "seconds": round(time.time() - t0, 1), **pr}
    with open(os.path.join(out, "probe.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in pr.items() if k.startswith("all/") and not isinstance(v, list)}, indent=1),
          flush=True)
    mon = StopMonitor("probe", out, context={"index": args.index, "adapter": args.adapter})
    d = mon.check({k: v for k, v in pr.items() if isinstance(v, float)}, counts={"n": 1})
    return STOP_EXIT if d.stop else 0


if __name__ == "__main__":
    sys.exit(main())
