#!/usr/bin/env python
"""F6 -- the round-2a judge's precision and commit share against the commit threshold,
split into CONFIRMATIONS (the judge's argmax equals the drafter's token) and OVERRIDES
(it differs).

Data: results/metrics/draftalign/holdout_precision_t2a.json, produced on CPU-side
recorded states by code/draftalign/analyze_holdout_precision.py over 300 held-out
states of the round-1 GSM8K training set with the step-1,057 round-2a adapter.
Two regimes are stored; the figure shows `draft_pass1` (every open slot holds the
drafter's token -- the exact state the judge meets at inference pass 1).

Usage: /venv/main/bin/python results/figures/draftalign/F6_confirm_vs_override.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402

J = json.load(open(f"{ROOT}/results/metrics/draftalign/holdout_precision_t2a.json"))
R = J["regimes"]["draft_pass1"]
taus = sorted(float(t) for t in R["by_tau"])
g = lambda t, *k: (lambda d: d)(R["by_tau"][("%g" % t) if ("%g" % t) in R["by_tau"] else str(t)])


def get(t, *keys):
    key = "%g" % t
    if key not in R["by_tau"]:
        key = str(t)
    d = R["by_tau"][key]
    for k in keys:
        d = d[k]
    return d


conf_p = [get(t, "confirm", "precision") for t in taus]
ovr_p = [get(t, "override", "precision") for t in taus]
all_p = [get(t, "precision") for t in taus]
conf_s = [get(t, "confirm", "share_of_open") for t in taus]
ovr_s = [get(t, "override", "share_of_open") for t in taus]

n_slots = 29.0                              # generated slots per block (stock decode, 100 train prompts)
wrong_per_block = [n_slots * (get(t, "commit_share") * (1 - get(t, "precision"))) for t in taus]

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))

ax = axes[0]
ax.plot(taus, conf_p, "o-", color=C["r2a"], label="confirmations (argmax = draft)")
ax.plot(taus, ovr_p, "s-", color=C["r1"], label="overrides (argmax ≠ draft)")
ax.plot(taus, all_p, "^--", color=C["stock"], lw=1.0, ms=4, label="all commits")
ax.axhline(0.99, color=C["grey"], lw=0.8, ls=":")
ax.set_xlabel("commit threshold τ")
ax.set_ylabel("precision of a confident commit")
ax.set_ylim(0.72, 1.005)
ax.set_title("precision")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(taus, conf_s, "o-", color=C["r2a"], label="confirmations")
ax.plot(taus, ovr_s, "s-", color=C["r1"], label="overrides")
ax.plot(taus, [get(t, "commit_share") for t in taus], "^--", color=C["stock"], lw=1.0, ms=4,
        label="all commits")
ax.set_xlabel("commit threshold τ")
ax.set_ylabel("share of open slots committed")
ax.set_ylim(0, 0.52)
ax.set_title("commit share")
ax.legend(fontsize=7)

ax = axes[2]
ax.plot(taus, wrong_per_block, "o-", color=C["accent"])
for t, y in zip(taus, wrong_per_block):
    ax.annotate("%.2f" % y, (t, y), textcoords="offset points", xytext=(0, 6), fontsize=6.5,
                ha="center", color=C["accent"])
ax.set_xlabel("commit threshold τ (same for both kinds)")
ax.set_ylabel("wrong confident commits per block")
ax.set_ylim(0, 0.95)
ax.set_title("expected damage per 29-slot block")

fig.suptitle("Round-2a judge on 300 held-out pass-1 states: almost every wrong confident "
             "commit is an override", y=1.04, fontsize=9.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F6_confirm_vs_override.png")
fig.savefig(out)
print("wrote", out)
print("open slots", R["open_slots"], "| draft right", round(R["draft_correct_rate"], 4),
      "| judge argmax right", round(R["argmax_acc_all"], 4))
for t in taus:
    print("tau %-6s commit %.3f prec %.4f | confirm %.3f/%.4f | override %.3f/%.4f | wrong/blk %.3f"
          % (t, get(t, "commit_share"), get(t, "precision"), get(t, "confirm", "share_of_open"),
             get(t, "confirm", "precision"), get(t, "override", "share_of_open"),
             get(t, "override", "precision"),
             n_slots * get(t, "commit_share") * (1 - get(t, "precision"))))
# the two asymmetric rules the decoder was actually run with
for tc, to in ((0.95, 0.99), (0.90, 0.99)):
    w = n_slots * (get(tc, "confirm", "share_of_open") * (1 - get(tc, "confirm", "precision"))
                   + get(to, "override", "share_of_open") * (1 - get(to, "override", "precision")))
    s = n_slots * (get(tc, "confirm", "share_of_open") + get(to, "override", "share_of_open"))
    print("confirm %.2f / override %.2f -> %.2f commits/block, %.3f wrong/block" % (tc, to, s, w))
