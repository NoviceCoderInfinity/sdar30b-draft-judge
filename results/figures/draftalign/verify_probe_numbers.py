#!/usr/bin/env python
"""CPU re-derivation of the three probe numbers the report quotes that no tracked
analysis script produces: the next-token copy rate at filled slots, the blind-marginal
acceptance sweep, and the stock pass-1 commit count.

Reads only the recorded probe records
  logs/draftalign/ab_probe/gsm8k_B32_t95_stock.calibvalid/shard*of4.jsonl
(one JSON object per block; `moe.<kind>.argmax` is the fp32 argmax with the [MASK]
logit suppressed, `moe.P0.top1` the stock pass-1 top-1 probability, `moe.P0.p_draft`
the stock pass-1 probability of the s8 drafter's token) and the recorded stock traces
  logs/draftalign/da_decode/gsm8k_B32_t95_stock/traces/calibvalid/*.npz.

Nothing is loaded on a GPU and no model is instantiated.
Usage: /venv/main/bin/python results/figures/draftalign/verify_probe_numbers.py
"""
import glob
import json
import os

import numpy as np

ROOT = "/workspace/dlm-exp1-voting"
PROBE = f"{ROOT}/logs/draftalign/ab_probe/gsm8k_B32_t95_stock.calibvalid/shard*of4.jsonl"
STOCK = f"{ROOT}/logs/draftalign/da_decode/gsm8k_B32_t95_stock/traces/calibvalid/*.npz"
FILL = {"P1": "draft", "P2a": "draft", "P2b": "draft", "P3": "self", "P4a": "self",
        "P4b": "self", "P5": "final", "P6a": "final", "P6b": "final"}

blocks = []
for f in sorted(glob.glob(PROBE)):
    for line in open(f):
        r = json.loads(line)
        if "b" in r:
            blocks.append(r)
print("blocks read:", len(blocks), "| items:", len({b["idx"] for b in blocks}))

# ---------------------------------------------------------------- 1. next-token copy
print("\n== 1. what a FILLED slot outputs on the stock model (valid slots, i < 31) ==")
print("%-4s %-6s %8s %8s %8s" % ("kind", "fill", "next", "own", "n"))
for kind in ("P1", "P5", "P3"):
    nxt = own = n = 0
    for b in blocks:
        v = np.asarray(b["valid"], bool)
        am = np.asarray(b["moe"][kind]["argmax"], int)
        src = np.asarray(b[{"draft": "drafter", "self": "self_draft", "final": "final"}
                           [FILL[kind]]]["s8"]["tokens"] if FILL[kind] == "draft"
                          else b["self_draft"] if FILL[kind] == "self" else b["final"], int)
        for i in range(31):
            if not v[i]:
                continue
            n += 1
            nxt += int(am[i] == src[i + 1])
            own += int(am[i] == src[i])
    print("%-4s %-6s %8.4f %8.4f %8d" % (kind, FILL[kind], nxt / n, own / n, n))

# ---------------------------------------------------------- 2. blind-marginal sweep
print("\n== 2. blind-marginal acceptance beyond the stock pass-1 commits ==")
print("the stock pass-1 commit set is P0.top1 > 0.95; a draft slot is ACCEPTED when it "
      "is still open and P0.p_draft > gamma")
print("%-6s %10s %10s %12s %12s" % ("gamma", "extra/blk", "precision", "wrong/blk", "share"))
nblk = len(blocks)
for gamma in (0.30, 0.40, 0.50, 0.70):
    extra = right = 0
    for b in blocks:
        v = np.asarray(b["valid"], bool)
        top1 = np.asarray(b["moe"]["P0"]["top1"], float)
        q = np.asarray(b["moe"]["P0"]["p_draft"], float)
        D = np.asarray(b["drafter"]["s8"]["tokens"], int)
        T = np.asarray(b["final"], int)
        acc = v & ~(top1 > 0.95) & (q > gamma)
        extra += int(acc.sum())
        right += int((acc & (D == T)).sum())
    print("%-6.2f %10.2f %10.4f %12.2f %12.4f"
          % (gamma, extra / nblk, right / max(1, extra), (extra - right) / nblk,
             extra / sum(int(np.asarray(b["valid"], bool).sum()) for b in blocks) * nblk
             / nblk))

# ---------------------------------------------------------- 3. stock pass-1 commits
print("\n== 3. stock decode, pass 1 (from the recorded stock traces of the same 100 items) ==")
nb = c1 = slots = right1 = 0
passes = []
for p in sorted(glob.glob(STOCK)):
    d = np.load(p)
    for b in range(int(d["tokens"].shape[0])):
        v = d["valid"][b]
        if not v.any():
            continue
        nb += 1
        slots += int(v.sum())
        c1 += int((d["committed"][b, 0] & v).sum())
        right1 += int(((d["proposed"][b, 0] == d["tokens"][b]) & v).sum())
        passes.append(int(d["n_pass"][b]))
print("blocks %d | generated slots/block %.2f | pass-1 commits/block %.2f "
      "| pass-1 argmax already final %.4f" % (nb, slots / nb, c1 / nb, right1 / slots))
print("passes/block mean %.2f median %d p90 %d" % (np.mean(passes), int(np.median(passes)),
                                                   int(np.percentile(passes, 90))))
