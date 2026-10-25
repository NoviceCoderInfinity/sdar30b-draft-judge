#!/usr/bin/env python
"""MEASURED cost of the judge LoRA per 30B forward, from the decode records only (no GPU).

    /venv/p2/bin/python code/drafter_lora/judge_lora_cost.py --out results/phase4_drafter/judge_lora_cost.json

The judge's 128M LoRA is flag-gated (code/draftalign/adapter.py, LoRALinear.forward): when no position of the block
holds a tentative draft token, every q/k/v/o projection returns the frozen base output and the LoRA kernels are skipped
(`if flag is not None and not CTX.any: return out`).  So judge forwards split into
    flagged   (>= 1 drafted slot: the LoRA runs)   and   unflagged (no drafted slot: the LoRA is skipped).
Which pass is which, from each record's full_passes_per_block (one dict per judge pass, aligned with the record's
judge forwards = forwards[] entries of kind 3, in order):
    pass 0 of a block: flagged (the first draft fills every open slot)
    pass j > 0:        flagged  iff  tentative(j-1) + redraft(j-1) > 0   (slots still holding a draft token)
Per forward: ms = forwards[k][1] (cuda.synchronize + perf_counter around the 30B forward), e = forwards[k][2] /
forwards[k][3] = distinct experts per layer.  The time of a forward grows with e, and flagged and unflagged passes have
different e, so the LoRA cost is measured AT MATCHED e:
    bins of width 1 expert/layer; in each bin with >= 30 forwards of both kinds, diff = mean ms(flagged) - mean
    ms(unflagged); LoRA cost = the mean of the bin diffs weighted by the bin's number of unflagged forwards.
Check: the least-squares fit ms = a + d * flagged + c * e (d should equal the matched-bin cost).
Control: unflagged judge forwards vs the vanilla 30B's denoise forwards decoded on this box
(logs/phase4/decode/vanilla_here), matched the same way (should be ~0).
rd1 re-drafts every hole after every pass, so (almost) every judge pass is flagged there and d is not identifiable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402

from drafter_lora.compare_gates import SETS, load_dir                    # noqa: E402

NAME = {"ifeval541": "IFEval", "math500": "MATH-500", "gpqa": "GPQA", "aime": "AIME", "humaneval": "HumanEval",
        "mbpp": "MBPP", "gsm8k500": "GSM8K"}
JUDGE, DENOISE = 3, 1                                                    # forwards[k][0] kind codes
MIN_N = 30


def judge_forwards(R):
    ms, e, fl, bad = [], [], [], 0
    for r in R.values():
        J = [f for f in r["forwards"] if f[0] == JUDGE]
        fp = r["full_passes_per_block"]
        if len(J) != sum(len(b) for b in fp):
            bad += 1
            continue
        k = 0
        for b in fp:
            for j, _ in enumerate(b):
                f = J[k]; k += 1
                ms.append(f[1]); e.append(f[2] / f[3])
                fl.append(1.0 if j == 0 else float(b[j - 1]["tentative"] + b[j - 1]["redraft"] > 0))
    return np.array(ms), np.array(e), np.array(fl), bad


def matched(ms1, e1, ms0, e0):
    """mean ms(group 1) - mean ms(group 0) at matched experts/layer (1-wide bins, >= MIN_N of each)."""
    if not len(ms0) or not len(ms1):
        return None
    lo, hi = np.floor(min(e1.min(), e0.min())), np.ceil(max(e1.max(), e0.max()))
    d, w = [], []
    for x in np.arange(lo, hi, 1.0):
        m1 = (e1 >= x) & (e1 < x + 1); m0 = (e0 >= x) & (e0 < x + 1)
        if m1.sum() >= MIN_N and m0.sum() >= MIN_N:
            d.append(ms1[m1].mean() - ms0[m0].mean()); w.append(int(m0.sum()))
    if not d:
        return None
    return {"ms": float(np.average(d, weights=w)), "bins": len(d), "bin_min": float(min(d)), "bin_max": float(max(d)),
            "n_matched_group0": int(sum(w))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4", default="logs/phase4/decode")
    ap.add_argument("--vanilla", default="logs/phase4/decode/vanilla_here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.chdir(os.path.abspath(os.path.join(CODE, "..")))
    rep = {"definition": __doc__, "arms": {}}
    van = {}
    for s in SETS:
        V = load_dir(os.path.join(a.vanilla, s))
        if V:
            F = [f for r in V.values() for f in r["forwards"] if f[0] == DENOISE]
            van[s] = (np.array([f[1] for f in F]), np.array([f[2] / f[3] for f in F]), len(V))
    for arm in ("rd1", "rd2"):
        per = {}
        for s in SETS:
            R = load_dir(os.path.join(a.p4, arm, s))
            if not R:
                continue
            ms, e, fl, bad = judge_forwards(R)
            o = {"n_records": len(R), "alignment_failures": bad, "n_judge_forwards": int(len(ms)),
                 "n_unflagged": int((fl == 0).sum()), "unflagged_pct": float(100 * (fl == 0).mean()),
                 "mean_ms_flagged": float(ms[fl == 1].mean()),
                 "mean_ms_unflagged": float(ms[fl == 0].mean()) if (fl == 0).any() else None,
                 "lora_cost_matched": matched(ms[fl == 1], e[fl == 1], ms[fl == 0], e[fl == 0])}
            if 0 < (fl == 0).sum() < len(fl):
                X = np.c_[np.ones_like(ms), fl, e]
                c = np.linalg.lstsq(X, ms, rcond=None)[0]
                o["fit_ms_a_d_c"] = [float(x) for x in c]
            if s in van:
                vms, ve, vn = van[s]
                o["vanilla_denoise"] = {"n_records": vn, "n_forwards": int(len(vms)), "mean_ms": float(vms.mean())}
                o["unflagged_minus_vanilla_matched"] = matched(ms[fl == 0], e[fl == 0], vms, ve)
                o["flagged_minus_vanilla_matched"] = matched(ms[fl == 1], e[fl == 1], vms, ve)
            per[s] = o
        rep["arms"][arm] = per
    json.dump(rep, open(P.safe_out(a.out), "w"), indent=1)

    f = lambda x, d=1: "—" if x is None else f"{x:+.{d}f}"                         # noqa: E731
    L = ["Judge LoRA cost per 30B forward, at matched experts per layer (ms)", "",
         "| Benchmark | Mode | Judge forwards | Without LoRA (no drafted slot) | LoRA cost: flagged − unflagged [bin range, bins] | "
         "Fit d | Unflagged − vanilla forward | Flagged − vanilla forward |", "|" + "---|" * 8]
    for s in SETS:
        for arm in ("rd1", "rd2"):
            o = rep["arms"][arm].get(s)
            if not o:
                continue
            m = o["lora_cost_matched"]
            mc = "—" if not m else f"**{m['ms']:+.1f}** [{m['bin_min']:+.1f}, {m['bin_max']:+.1f}], {m['bins']}"
            fit = f(o["fit_ms_a_d_c"][1]) if "fit_ms_a_d_c" in o else "—"
            u = o.get("unflagged_minus_vanilla_matched"); g = o.get("flagged_minus_vanilla_matched")
            L.append(f"| {NAME[s]} ({o['n_records']}) | {arm} | {o['n_judge_forwards']:,} | {o['n_unflagged']:,} "
                     f"({o['unflagged_pct']:.1f}%) | {mc} | {fit} | {f(u['ms'] if u else None)} | {f(g['ms'] if g else None)} |")
    open(os.path.splitext(P.safe_out(a.out))[0] + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
