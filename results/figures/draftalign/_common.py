"""Shared style for the draft-aligned report figures (CPU only, no model loading)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/workspace/dlm-exp1-voting"

# colour-blind-safe, consistent across every figure of the report
C = {
    "stock":   "#4C4C4C",   # stock decode
    "free":    "#0072B2",   # training-free blind-accept rule
    "r1":      "#D55E00",   # round 1 judge
    "r2a":     "#009E73",   # round 2a judge
    "right":   "#009E73",   # draft token was right
    "wrong":   "#D55E00",   # draft token was wrong
    "grey":    "#9A9A9A",
    "accent":  "#CC79A7",
    "mask":    "#56B4E9",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
