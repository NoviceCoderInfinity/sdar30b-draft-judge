#!/usr/bin/env python
"""Real-model check of the grouped-GEMM expert forward (fast_moe.py) before the trainer switches to it.

    CUDA_VISIBLE_DEVICES=0 /venv/train/bin/python code/corpus1b/check_fast_moe.py --adapter logs/c1b/train/main/final

The pinned 30B + the given adapter on ONE GPU; the same states (one micro-batch per domain mix, from the index) run
through the stock expert loop, then through the patched forward: the trainer's loss (train_fast.micro_batch_loss,
layer checkpointing on, as in training) and the LoRA gradients are compared. PASS = loss within 0.1% and gradient
cosine >= 0.999. Both passes are timed (after one warm-up each), so the speed-up is measured, not estimated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402
import torch                                                             # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign import train_fast as F                                   # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry   # noqa: E402
import fast_moe                                                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--index", default="logs/c1b/trainset/waveA/index.jsonl")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    _tok, model = P.load_sdar("b32", str(dev))
    A.attach_lora(model, r=32, alpha=64.0, train_gates=False, draft_flag=True, conf_embed=False)
    A.load_adapter(model, P.safe_out(a.adapter))
    A.freeze_base(model)
    model.train()
    model.config.use_cache = False
    params = [p for _, p in A.trainable_parameters(model)]
    lcfg = F.LossCfg(beta=0.5, chat_beta=0.0, temp=0.5, calib=1.0, override_weight=2.0, code_override_weight=3.0,
                     chat_override_weight=1.0, mask_id=P.MASK_ID)
    s = StateSampler(SamplerConfig(index_path=a.index, seed=1, p_pass1=0.4, p_allmask=0.0, p_draftfull=0.3,
                                   mask_id=P.MASK_ID, block=32, max_seq=1856))
    by_item = s.item_examples()
    picked, per_dom = [], {}
    for i in s.item_order(0):                       # up to 4 states per domain, every domain represented
        d = domain_of_entry(s.entries[i])
        if per_dom.get(d, 0) < 4:
            take = by_item[i][:4 - per_dom.get(d, 0)]
            picked.extend(take); per_dom[d] = per_dom.get(d, 0) + len(take)
        if len(per_dom) >= 5 and all(v >= 4 for v in per_dom.values()):
            break
    sts = [s.state(k, epoch=0) for k in picked[:a.states]]

    def run():
        for p in params:
            p.grad = None
        torch.cuda.synchronize(); t0 = time.time()
        loss, _stats, _ = F.micro_batch_loss(model, sts, lcfg, block=32, checkpoint_layers=True, conf_embed=False)
        loss.backward()
        torch.cuda.synchronize()
        g = torch.cat([p.grad.detach().float().flatten() for p in params if p.grad is not None])
        return float(loss), g, time.time() - t0

    run()                                           # warm-up (loop)
    res0 = [run() for _ in range(a.reps)]
    mode = fast_moe.patch_moe(model, mode="auto")
    run()                                           # warm-up (fast; first _grouped_mm / bmm call)
    res1 = [run() for _ in range(a.reps)]
    l0, g0, _ = res0[-1]
    l1, g1, _ = res1[-1]
    t0 = min(r[2] for r in res0); t1 = min(r[2] for r in res1)
    rel = abs(l1 - l0) / max(abs(l0), 1e-12)
    cos = float(torch.nn.functional.cosine_similarity(g0, g1, dim=0))
    ok = rel <= 1e-3 and cos >= 0.999
    rep = {"mode": mode, "states": len(sts), "loss_loop": l0, "loss_fast": l1, "rel_diff_pct": 100 * rel,
           "grad_cosine": cos, "sec_loop": t0, "sec_fast": t1, "speedup": t0 / max(t1, 1e-9), "PASS": ok}
    print("[check_fast_moe] " + json.dumps(rep), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
