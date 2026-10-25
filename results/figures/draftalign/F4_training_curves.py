#!/usr/bin/env python
"""F4 -- round 1 vs round 2a training curves, parsed from the two torchrun logs.

Data (no GPU, no model):
  logs/draftalign/train/t1_gsm8k_r1.log     round 1 (ungated LoRA + trainable router
                                            gates, no draft flag), stopped at step 600
  logs/draftalign/train/t2a_gsm8k_flag.log  round 2a (draft-flag vector + flag-gated
                                            LoRA, gates frozen), all 1,057 steps
Lines parsed:
  "[step k/1057] loss L ce C cert Z acc A top1 T uniq U M m lr R ..."   every 10 steps
  "[eval k] allmask_pass1_* draft_pass1_*"                              every 100 steps
`draft_pass1` = held-out states in which every open slot holds the drafter's token
(the inference pass-1 state); `allmask_pass1` = every open slot [MASK] (stock regime).
conf_frac = share of open slots whose top-1 probability exceeds 0.95;
conf_precision = of those, the share whose argmax equals the target.

Usage: /venv/main/bin/python results/figures/draftalign/F4_training_curves.py
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402

STEP = re.compile(r"^\[step (\d+)/\d+\] loss ([\d.]+) ce ([\d.]+) cert ([\d.]+) acc ([\d.]+) top1 ([\d.]+)")
EVAL = re.compile(r"^\[eval (\d+)\] (.*)$")


def parse(path):
    steps, evals = [], []
    for line in open(path, errors="ignore"):
        m = STEP.match(line)
        if m:
            steps.append(tuple(float(x) for x in m.groups()))
            continue
        m = EVAL.match(line)
        if m:
            d = {"step": int(m.group(1))}
            for kv in m.group(2).split():
                k, v = kv.split("=")
                d[k] = float(v)
            evals.append(d)
    return np.array(steps), evals


s1, e1 = parse(f"{ROOT}/logs/draftalign/train/t1_gsm8k_r1.log")
s2, e2 = parse(f"{ROOT}/logs/draftalign/train/t2a_gsm8k_flag.log")
col = lambda ev, k: np.array([d[k] for d in ev])

fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.95))

ax = axes[0]
ax.plot(s1[:, 0], s1[:, 2], color=C["r1"], lw=1.1, label="round 1")
ax.plot(s2[:, 0], s2[:, 2], color=C["r2a"], lw=1.1, label="round 2a")
ax.set_yscale("log")
ax.set_xlabel("optimiser step")
ax.set_ylabel("training cross-entropy (nats)")
ax.set_title("cross-entropy on open slots")
ax.legend(fontsize=7.5)

ax = axes[1]
ax.plot(s1[:, 0], s1[:, 4], color=C["r1"], lw=1.1, label="round 1")
ax.plot(s2[:, 0], s2[:, 4], color=C["r2a"], lw=1.1, label="round 2a")
ax.set_xlabel("optimiser step")
ax.set_ylabel("share of open slots with argmax = target")
ax.set_title("training accuracy")
ax.set_ylim(0.25, 0.82)
ax.legend(fontsize=7.5)

ax = axes[2]
ax.plot(col(e1, "step"), col(e1, "draft_pass1_acc"), "o-", ms=3.5, color=C["r1"], label="round 1, draft slots")
ax.plot(col(e2, "step"), col(e2, "draft_pass1_acc"), "o-", ms=3.5, color=C["r2a"], label="round 2a, draft slots")
ax.plot(col(e1, "step"), col(e1, "allmask_pass1_acc"), "s--", ms=3, lw=0.9, color=C["r1"],
        alpha=0.65, label="round 1, all-mask slots")
ax.plot(col(e2, "step"), col(e2, "allmask_pass1_acc"), "s--", ms=3, lw=0.9, color=C["r2a"],
        alpha=0.65, label="round 2a, all-mask slots")
ax.axhline(0.556, color=C["grey"], ls=":", lw=1.0)
ax.text(1040, 0.565, "copy-the-draft baseline 0.556", ha="right", fontsize=6.8, color=C["grey"])
ax.axhline(0.5323, color=C["stock"], ls="-.", lw=0.8)
ax.text(1040, 0.497, "stock at all-mask slots 0.532", ha="right", fontsize=6.8, color=C["stock"])
ax.set_xlabel("optimiser step")
ax.set_ylabel("held-out argmax accuracy")
ax.set_title("held-out accuracy (64 states / regime)")
ax.set_ylim(0.10, 0.63)
ax.legend(fontsize=6.5, loc="lower right")

ax = axes[3]
ax.plot(col(e1, "step"), col(e1, "draft_pass1_conf_frac"), "o-", ms=3.5, color=C["r1"], label="round 1, share > .95")
ax.plot(col(e2, "step"), col(e2, "draft_pass1_conf_frac"), "o-", ms=3.5, color=C["r2a"], label="round 2a, share > .95")
ax.plot(col(e1, "step"), col(e1, "draft_pass1_conf_precision"), "^--", ms=3.5, lw=0.9, color=C["r1"],
        label="round 1, precision")
ax.plot(col(e2, "step"), col(e2, "draft_pass1_conf_precision"), "^--", ms=3.5, lw=0.9, color=C["r2a"],
        label="round 2a, precision")
ax.set_xlabel("optimiser step")
ax.set_ylabel("share of / precision at confident slots")
ax.set_title("confidence at draft slots")
ax.set_ylim(0.1, 1.03)
ax.legend(fontsize=6.5, loc="center right")

fig.suptitle("Round 1 (ungated adapter, no draft marker) vs round 2a (draft flag + gated LoRA): "
             "same data, same budget", y=1.06, fontsize=9.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F4_training_curves.png")
fig.savefig(out)
print("wrote", out)
print("round 1: %d step lines (last step %d), %d evals" % (len(s1), s1[-1, 0], len(e1)))
print("round 2a: %d step lines (last step %d), %d evals" % (len(s2), s2[-1, 0], len(e2)))
for tag, ev in (("r1", e1), ("r2a", e2)):
    for d in ev:
        print("%-4s eval %4d draft_acc %.4f conf %.4f prec %.4f | allmask_acc %.4f conf %.4f prec %.4f"
              % (tag, d["step"], d["draft_pass1_acc"], d["draft_pass1_conf_frac"],
                 d["draft_pass1_conf_precision"], d["allmask_pass1_acc"],
                 d["allmask_pass1_conf_frac"], d["allmask_pass1_conf_precision"]))
