#!/usr/bin/env python
"""F5 -- the accuracy / cost frontier on the 100 frozen GSM8K train prompts.

Every point is one complete decode of the same 100 prompts, so accuracy differences
are paired per item. The numbers are read from the JSON that
`code/draftalign/analyze_da.py` writes for the recorded decode cells under
logs/draftalign/da_decode/; this script re-runs that analysis so the figure cannot
drift from the records.

Usage: /venv/main/bin/python results/figures/draftalign/F5_frontier.py
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import C, ROOT, style          # noqa: E402

style()
import matplotlib.pyplot as plt             # noqa: E402

D = f"{ROOT}/logs/draftalign/da_decode"
POINTS = [
    # (cell dir, label, colour key, marker)
    ("gsm8k_B32_t95_stock", "stock decode", "stock", "o"),
    ("gsm8k_B32_t95_dv-s8_blind_q50", "free rule γ .50", "free", "D"),
    ("gsm8k_B32_t95_dv-s8_blind_q40", "free rule γ .40", "free", "d"),
    ("gsm8k_B32_t95_dv-s8_blind_q30", "free rule γ .30", "free", "v"),
    ("gsm8k_B32_t95_dv-s8_full_c95_ad5d9add10", "round 1, commit .95", "r1", "X"),
    ("gsm8k_B32_t95_dv-s8_full_c95_adb3106a4d", "2a: confirm .95 / override .95", "r2a", "s"),
    ("gsm8k_B32_t95_dv-s8_full_c90_adb3106a4d", "2a: confirm .90 / override .90", "r2a", "^"),
    ("gsm8k_B32_t95_dv-s8_full_c90_ro_adb3106a4d", "2a: .90 / .90, re-mask only", "r2a", "<"),
    ("gsm8k_B32_t95_dv-s8_full_c95_o99_adb3106a4d", "2a: confirm .95 / override .99", "r2a", "P"),
    ("gsm8k_B32_t95_dv-s8_full_c90_o99_adb3106a4d", "2a: confirm .90 / override .99", "r2a", "*"),
    ("gsm8k_B32_t95_dv-s8_full_c95_o100_adb3106a4d", "2a: confirm .95 / override .995", "r2a", "h"),
    ("gsm8k_B32_t95_dv-s8_c95_a70_q50", "halves judge (draft context)", "accent", "x"),
]

tmp = os.path.join(tempfile.gettempdir(), "F5_da_summary.json")
subprocess.run(["/venv/main/bin/python", f"{ROOT}/code/draftalign/analyze_da.py"]
               + [f"{D}/{c}" for c, *_ in POINTS] + ["--out", tmp],
               check=True, capture_output=True, cwd=ROOT)
S = json.load(open(tmp))
cells = {k[: -len(".calibvalid")]: v for k, v in S["cells"].items() if k.endswith(".calibvalid")}
paired = {k[: -len(".calibvalid")]: v for k, v in S["paired_vs_ref"].items()
          if k.endswith(".calibvalid")}

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
OFF = {"gsm8k_B32_t95_dv-s8_full_c95_ad5d9add10", "gsm8k_B32_t95_dv-s8_c95_a70_q50"}
FREE = ["gsm8k_B32_t95_dv-s8_blind_q30", "gsm8k_B32_t95_dv-s8_blind_q40",
        "gsm8k_B32_t95_dv-s8_blind_q50", "gsm8k_B32_t95_stock"]

for ax, xk, xlab in ((axes[0], "passes_per_block", "block forwards per block (judge + denoise, flush excluded)"),
                     (axes[1], "expert_layer_exec_per_token", "MoE expert-layer executions per generated token")):
    ax.plot([cells[c][xk] for c in FREE], [cells[c]["accuracy"] for c in FREE],
            "-", color=C["free"], lw=1.0, alpha=0.5, zorder=1)
    for cell, label, ck, mk in POINTS:
        s_ = cells[cell]
        acc = s_["accuracy"]
        if cell in OFF:
            continue
        d = paired.get(cell, {})
        lo, hi = d.get("ci95", [0.0, 0.0])
        delta = d.get("delta", 0.0)
        yerr = [[abs(delta - lo)], [abs(hi - delta)]] if d else None
        ax.errorbar(s_[xk], acc, yerr=yerr, fmt=mk, ms=7.5, color=C[ck], capsize=2.5,
                    elinewidth=0.9, mec="white", mew=0.6, zorder=3,
                    label=label if ax is axes[0] else None)
    ax.set_xlabel(xlab)
    ax.set_ylabel("accuracy on the 100 frozen train prompts (%)")
    ax.set_ylim(84.5, 98.2)
    ax.axhline(95.0, color=C["stock"], lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], 95.3, "stock accuracy 95.0", ha="right", fontsize=6.8, color=C["stock"])

axes[0].text(0.02, 0.03,
             "off the scale:\n  round-1 judge  38% at 19.57 passes / 1909 exec\n"
             "  stock halves judge  67% at 2.01 passes / 575 exec",
             transform=axes[0].transAxes, fontsize=6.6, va="bottom", color=C["r1"])
axes[0].legend(fontsize=6.6, loc="upper left", ncol=1)
fig.suptitle("Accuracy against cost, 100 frozen GSM8K train prompts (y axis truncated at 84.5; "
             "the two failed decodes are off the scale).\nBars are the 95% paired bootstrap "
             "interval of the accuracy difference from the stock decode on the same items.",
             y=1.06, fontsize=8.6)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F5_frontier.png")
fig.savefig(out)
print("wrote", out)
print("%-46s %6s %8s %8s %9s %8s" % ("decode", "acc", "Δ vs stock", "passes", "exec/tok", "ms/tok"))
for cell, label, ck, mk in POINTS:
    s = cells[cell]
    d = paired.get(cell, {})
    ci = d.get("ci95", [0, 0])
    print("%-46s %6.1f %8.1f [%.0f,%.0f] %8.2f %9.0f %8.0f"
          % (label, s["accuracy"],
             d.get("delta", 0.0), ci[0], ci[1], s["passes_per_block"],
             s["expert_layer_exec_per_token"], s["ms_per_token"]))
