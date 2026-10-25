#!/usr/bin/env python
"""F1 -- one block of draft-aligned decoding: prompt -> drafter block -> judge pass ->
commit / keep tentative / re-mask -> repeat -> flush.

The boxes are a schematic of the loop implemented in code/draftalign/da_decode.py
(`da_generate`, `judge_full_block`, `full_commit_rule`). The slot strip at the bottom
is NOT schematic: it is the round-2a judge's actual pass-1 verdict on GSM8K frozen-train
item 9, block index 2, recomputed here from the recorded trace
logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d/traces/calibvalid/00009.npz
by applying `full_commit_rule` (commit if top-1 > 0.95; otherwise keep the draft token
as tentative context when the judge's argmax equals it, else re-mask).

Usage: /venv/main/bin/python results/figures/draftalign/F1_pipeline.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

TR = f"{ROOT}/logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d/traces/calibvalid/00009.npz"
B, TAU = 2, 0.95
d = np.load(TR)
D, A, P = d["draft_tokens"][B], d["judge_argmax"][B], d["judge_p"][B]
commit = P > TAU
tentative = ~commit & (A == D)
remask = ~commit & ~tentative
n_c, n_t, n_r = int(commit.sum()), int(tentative.sum()), int(remask.sum())

fig = plt.figure(figsize=(10.6, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(-1, 52)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color=tc)


def arrow(x1, y1, x2, y2, text="", ls="-", col=None, rad=0.0, fs=7.0, dx=0, dy=1.2):
    col = col or C["stock"]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color=col, ls=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color=col)


# --- top row: the loop ------------------------------------------------------
Y = 38
box(2, Y, 15, 8, "prompt\n+ blocks already\nwritten", "#F2F2F2", C["stock"])
box(22, Y, 17, 8, "1.7B drafter\n8 masked steps\n→ 32 draft tokens D", "#E6F2FA", C["free"])
box(44, Y, 20, 8, "30B judge: ONE forward\nevery open slot holds D\ndraft flag on those slots",
    "#E3F5EE", C["r2a"])
box(69, Y, 14, 8, "per-slot rule\ntop-1 > τ ?", "#FFF3E8", C["r1"])
box(87, Y, 11, 8, "kv flush\n(1 forward)", "#F2F2F2", C["stock"])

arrow(17.6, Y + 4, 21.4, Y + 4)
arrow(39.6, Y + 4, 43.4, Y + 4)
arrow(64.6, Y + 4, 68.4, Y + 4)
arrow(83.6, Y + 4, 86.4, Y + 4, "all 32\ncommitted", fs=6.6, dy=0.6)

# --- the three outcomes -----------------------------------------------------
Y2 = 25
box(44, Y2, 15, 7, "COMMIT\nslot ← judge's argmax\nnever revisited", "#D7F0E5", C["r2a"], fs=7.2)
box(61, Y2, 15, 7, "TENTATIVE\nkeep the draft token\nas context, still open", "#FDE9D9", C["r1"], fs=7.2)
box(78, Y2, 15, 7, "RE-MASK\nslot ← [MASK]\nstill open", "#E3EEF8", C["mask"], fs=7.2)

arrow(74, Y - 0.4, 53, Y2 + 8.0, col=C["r2a"], rad=0.16)
arrow(76, Y - 0.4, 68.5, Y2 + 8.0, col=C["r1"])
arrow(78, Y - 0.4, 85.5, Y2 + 8.0, col=C["mask"], rad=-0.16)
ax.text(56.5, Y2 + 9.2, "top-1 > τ", fontsize=6.8, color=C["r2a"], ha="center")
ax.text(69.5, Y2 + 9.2, "argmax = draft", fontsize=6.8, color=C["r1"], ha="center")
ax.text(84.0, Y2 + 9.2, "argmax ≠ draft", fontsize=6.8, color=C["mask"], ha="center")
arrow(76, Y2 - 0.4, 46, Y - 0.4, col=C["stock"], rad=-0.32)
ax.text(58, Y2 - 4.2, "the block re-enters the judge in its new state: committed slots hold their tokens,\n"
        "tentative slots hold the draft, re-masked slots hold [MASK]  (≥ 1 commit per pass, so it terminates)",
        fontsize=6.9, color=C["stock"], ha="center")

# --- the slot strip ---------------------------------------------------------
Y3 = 8
ax.text(2, Y3 + 9.4, "the judge's actual pass-1 verdict on one real block "
                     "(GSM8K frozen-train item 9, block 3, commit τ = 0.95)",
        fontsize=8.2, ha="left")
w = 2.85
for i in range(32):
    x = 2 + i * w
    if commit[i]:
        fc, ec, lab = "#D7F0E5", C["r2a"], "C"
    elif tentative[i]:
        fc, ec, lab = "#FDE9D9", C["r1"], "T"
    else:
        fc, ec, lab = "#E3EEF8", C["mask"], "M"
    ax.add_patch(plt.Rectangle((x, Y3), w * 0.86, 4.6, fc=fc, ec=ec, lw=0.8))
    ax.text(x + w * 0.43, Y3 + 2.3, lab, ha="center", va="center", fontsize=6.6, color=ec)
    ax.text(x + w * 0.43, Y3 - 1.3, str(i), ha="center", va="center", fontsize=5.4, color=C["grey"])
ax.text(2, Y3 - 4.6,
        "C commit (%d slots)      T tentative, draft kept as context (%d)      "
        "M re-masked (%d).   This block needed 8 judge passes in all." % (n_c, n_t, n_r),
        fontsize=7.2, ha="left")
ax.text(2, Y3 + 6.0,
        "the judge settles the prefix-forced head of the block in one pass and keeps most of the rest "
        "as tentative draft context rather than blanking it.", fontsize=6.9, ha="left", color=C["grey"])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F1_pipeline.png")
fig.savefig(out)
print("wrote", out)
print("pass-1 verdict: commit %d, tentative %d, re-mask %d (of 32)" % (n_c, n_t, n_r))
print("recorded judge passes for this block:", int(np.load(TR)["n_pass"][B]) or "see the jsonl record")
