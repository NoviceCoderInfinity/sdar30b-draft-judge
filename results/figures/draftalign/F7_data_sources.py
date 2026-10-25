#!/usr/bin/env python
"""F7 -- the training corpus: blocks per source and the stock drafter's per-slot
agreement with the big model, per source.

Data: the round-1 and round-2 data-generation records (one JSON line per prompt),
  logs/draftalign/datagen/<cell>/*.jsonl
written by run_da_decode.py under the training-free blind-accept rule (draft s8,
gamma 0.50). Fields used: `correct`, `n_blocks`, `n_valid_per_block`,
`draft_agreement_per_block` (per block, the share of that block's generated slots
where the drafter's token equals the decode's final token).

Round-2 generation was still running when this figure was made; the counts for the
three unfinished sources are therefore a snapshot, and the figure says so. Round 1's
own count is cross-checked against logs/draftalign/trainset_gsm8k_r1/build_summary.json.

Usage: /venv/main/bin/python results/figures/draftalign/F7_data_sources.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402

SOURCES = [
    ("GSM8K\n0-3000\n(round 1)", "gsm8k_pool_0_3000_dv-s8_blind_q50", True),
    ("GSM8K\n3000-7273", "gsm8k_pool_3000_7273_dv-s8_blind_q50", False),
    ("MATH\n(every 3rd\nof 7500)", "math_pool_0_7500_3_dv-s8_blind_q50", False),
    ("MBPP", "mbpp_pool_dv-s8_blind_q50", True),
]
MAX_SEQ = 1200                              # build_train_set drops blocks ending past this

rows = []
for label, cell, done in SOURCES:
    recs = []
    for f in sorted(glob.glob(f"{ROOT}/logs/draftalign/datagen/{cell}/*.jsonl")):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    if not recs:
        continue
    n_items = len(recs)
    ok = [r for r in recs if r.get("correct")]
    blocks = agr = 0
    agrs = []
    for r in ok:                            # only correct items enter the training set
        pl = int(r["prompt_tokens"])
        lo0 = (pl // 32) * 32
        for b, nv in enumerate(r["n_valid_per_block"]):
            if lo0 + 32 * (b + 1) > MAX_SEQ or nv == 0:
                continue
            blocks += 1
            agrs.append(float(r["draft_agreement_per_block"][b]))
    rows.append(dict(label=label, cell=cell, done=done, items=n_items, correct=len(ok),
                     blocks=blocks, agree=float(np.mean(agrs)) if agrs else float("nan"),
                     agree_sd=float(np.std(agrs)) if agrs else float("nan"),
                     agrs=np.array(agrs)))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(len(rows))

ax = axes[0]
cols = [C["r2a"] if r["done"] else C["grey"] for r in rows]
ax.bar(x, [r["blocks"] for r in rows], color=cols, width=0.62)
for i, r in enumerate(rows):
    ax.annotate("%d blocks\n%d/%d items kept" % (r["blocks"], r["correct"], r["items"]),
                (i, r["blocks"]), textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows], fontsize=6.6)
ax.set_ylabel("usable 32-token blocks")
ax.set_ylim(0, 20500)
ax.set_title("training blocks per source")

ax = axes[1]
parts = ax.violinplot([r["agrs"] for r in rows], positions=x, widths=0.75, showextrema=False,
                      showmedians=True)
for i, b in enumerate(parts["bodies"]):
    b.set_facecolor(cols[i])
    b.set_alpha(0.45)
    b.set_edgecolor(cols[i])
parts["cmedians"].set_color(C["stock"])
for i, r in enumerate(rows):
    ax.annotate("mean %.3f" % r["agree"], (i, 1.02), ha="center", fontsize=6.8, color=cols[i])
ax.axhline(0.5579, color=C["free"], ls="--", lw=0.9)
ax.text(len(rows) - 0.55, 0.575, "probe, GSM8K frozen train: 0.558", ha="right", fontsize=6.8,
        color=C["free"])
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows], fontsize=6.6)
ax.set_ylabel("drafter / big-model slot agreement per block")
ax.set_ylim(-0.03, 1.10)
ax.set_title("how often the 1.7B drafter's token is the final token")

fig.suptitle("Training corpus (grey = generation still running when this figure was made)",
             y=1.03, fontsize=9.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F7_data_sources.png")
fig.savefig(out)
print("wrote", out)
for r in rows:
    print("%-22s items %5d correct %5d blocks %6d agreement %.4f (sd %.3f) %s"
          % (r["cell"][:22], r["items"], r["correct"], r["blocks"], r["agree"], r["agree_sd"],
             "COMPLETE" if r["done"] else "PARTIAL"))
print("total blocks in the snapshot:", sum(r["blocks"] for r in rows))
