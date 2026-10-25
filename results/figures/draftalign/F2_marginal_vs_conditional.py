#!/usr/bin/env python
"""F2 -- marginal vs conditional, GSM8K frozen-train item 9, block index 2 (the third block).

Data (recorded traces, no GPU):
  logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_blind_q50/traces/calibvalid/00009.npz
      judge_q[2]  = q_i, the STOCK model's pass-1 probability of the drafter's token,
                    read from the empty-block forward (the MARGINAL).
      judge_p[2]  = the stock pass-1 top-1 probability at slot i.
      draft_tokens[2], tokens[2] = the drafter's block and the decode's final block.
  logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d/traces/calibvalid/00009.npz
      judge_p[2]  = the ROUND-2a judge's pass-1 top-1 probability with the block
                    draft-filled (the CONDITIONAL).
The drafter's block and the final tokens are byte-identical between the two cells for
this block (checked in the script), so the two readings describe the same 32 slots.

Usage: /venv/main/bin/python results/figures/draftalign/F2_marginal_vs_conditional.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402

BLIND = f"{ROOT}/logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_blind_q50/traces/calibvalid/00009.npz"
FULL = f"{ROOT}/logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d/traces/calibvalid/00009.npz"
B = 2                                       # block index 2 = the third generated block

b, f = np.load(BLIND), np.load(FULL)
assert np.array_equal(b["draft_tokens"][B], f["draft_tokens"][B]), "drafts differ between cells"
assert np.array_equal(b["tokens"][B], f["tokens"][B]), "final tokens differ between cells"

D, T = b["draft_tokens"][B], b["tokens"][B]
right = D == T
q_stock = b["judge_q"][B]                   # marginal: stock p(draft token), empty block
p_stock = b["judge_p"][B]                   # stock pass-1 top-1
p_judge = f["judge_p"][B]                   # conditional: trained judge top-1, draft-filled

# the token strings are hard-coded from a CPU tokenizer decode of the same arrays
# (AutoTokenizer JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc) so the figure needs no model files
LBL = ["1", "/", "2", " her", " hourly", " wage", ".", " This", " means", " her",
       " overtime", " rate", " is", " $", "1", "8", " +", " ($", "1", "8", "/", "2",
       ")", " =", " $", "1", "8", " +", " $", "9", " =", " $"]

x = np.arange(32)
w = 0.4
fig, ax = plt.subplots(figsize=(9.2, 3.5))

for i in range(32):
    col = C["right"] if right[i] else C["wrong"]
    ax.bar(x[i] - w / 2, q_stock[i], w, color=col, alpha=0.45, edgecolor=col, linewidth=0.5)
    ax.bar(x[i] + w / 2, p_judge[i], w, color=col, alpha=1.0, edgecolor="none")

ax.plot(x, p_stock, ls="none", marker="_", ms=9, mew=1.3, color=C["stock"], zorder=5)
ax.axhline(0.95, color=C["grey"], lw=0.9, ls="--", zorder=1)
ax.text(31.6, 0.965, "commit threshold 0.95", ha="right", va="bottom", fontsize=7.5, color=C["grey"])

ax.set_xticks(x)
ax.set_xticklabels([r.replace(" ", "␣") for r in LBL], rotation=90, fontsize=6.5)
ax.set_xlim(-0.8, 31.8)
ax.set_ylim(0, 1.06)
ax.set_ylabel("probability at pass 1")
ax.set_xlabel("slot of the block (label = the final token)")
ax.set_title("Item 9, block 3: the stock marginal is flat where the trained judge is peaked", pad=8)

handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C["grey"], alpha=0.45, ec=C["grey"], lw=0.5),
    plt.Rectangle((0, 0), 1, 1, fc=C["grey"]),
    plt.Line2D([], [], ls="none", marker="_", ms=9, mew=1.3, color=C["stock"]),
    plt.Rectangle((0, 0), 1, 1, fc=C["right"]),
    plt.Rectangle((0, 0), 1, 1, fc=C["wrong"]),
]
labels = ["stock pass-1 p(draft token), empty block  [marginal]",
          "round-2a judge pass-1 top-1, draft-filled  [conditional]",
          "stock pass-1 top-1 (its own best guess)",
          "draft token was right", "draft token was wrong"]
ax.legend(handles, labels, ncol=2, fontsize=7.2, loc="lower left", bbox_to_anchor=(0.0, -0.55))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F2_marginal_vs_conditional.png")
fig.savefig(out)
print("wrote", out)
print("draft==final:", int(right.sum()), "/ 32 =", round(float(right.mean()), 4))
print("slots 7-16 (all draft-right): mean marginal q %.3f, mean judge p %.3f"
      % (q_stock[7:17].mean(), p_judge[7:17].mean()))
print("slots 17-31 (14/15 draft-wrong): mean marginal q %.3f, mean judge p %.3f"
      % (q_stock[17:].mean(), p_judge[17:].mean()))
