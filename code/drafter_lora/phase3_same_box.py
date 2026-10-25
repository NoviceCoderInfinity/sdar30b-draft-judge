#!/usr/bin/env python
"""Phase 3 rd1 / rd2 timed for THIS machine, from Phase 3's recorded work.  Phase 3 was decoded on another box and was
not re-run here; its WORK is machine-independent and recorded per forward (decoding is deterministic: the vanilla
re-decode here matched the earlier records on 2,250/2,250 questions), so its time here = that work priced at this
machine's per-forward costs, measured on the Phase 4 records of this machine.

    /venv/p2/bin/python code/drafter_lora/phase3_same_box.py --out results/phase4_drafter/phase3_same_box.json

Per benchmark, fitted on every 30B forward of Phase 4 rd1 + rd2 on this machine (forwards[k] = [kind, ms, distinct
experts summed over layers, layers, positions]; ctx = prompt tokens + 32 x block index; e = experts per layer):
    judge forward  (kind 3)   ms = a + d x [any drafted slot, i.e. the judge LoRA runs] + b x ctx/1000 + c x e
    KV-flush       (kind 2)   ms = a' + b' x ctx/1000 + c' x e
    prefill        (kind 0)   = Phase 4's measured prefill of the same question on this machine (same prompt)
    drafter                   = Phase 4's measured ms per drafter forward of each kind (same mode) x the forward count
    host / other              = Phase 4's measured (seconds - 30B seconds - drafter seconds) per 30B forward (same mode)
Flagged pass: pass 0 of a block, or tentative + redraft of the previous pass > 0 (code/draftalign/adapter.py skips the
LoRA otherwise).  The same pricing applied to the vanilla records (kind 1 priced as an unflagged judge forward, its
own host time) and to Phase 4 itself checks the model against measured times.
Reported:  P3 vs vanilla = sum(measured vanilla seconds) / sum(priced Phase 3 seconds);  P4 vs P3 = sum(priced Phase 3
seconds) / sum(measured Phase 4 seconds); 95% range: paired bootstrap over questions (2,000, default_rng(0)), which
covers question sampling, not the pricing model's error (see the model checks).
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
from drafter_lora.vanilla_same_box import ratio_ci                       # noqa: E402

NAME = {"ifeval541": "IFEval", "math500": "MATH-500", "gpqa": "GPQA", "aime": "AIME", "humaneval": "HumanEval",
        "mbpp": "MBPP", "gsm8k500": "GSM8K"}
P3DIR = "logs/phase4/ref/phase3/logs/phase3_8gpu/decode"


def trace(r):
    """(kind, ms, experts/layer, ctx, flagged) per 30B forward."""
    fpb, nv = r.get("full_passes_per_block") or [], r["n_valid_per_block"]
    b = j = 0
    for f in r["forwards"]:
        k, ms, e = f[0], f[1], f[2] / max(f[3], 1)
        ctx = r["prompt_tokens"] + 32 * b
        if k == 3:
            fl = nv[b] if j == 0 else fpb[b][j - 1]["tentative"] + fpb[b][j - 1].get("redraft", 0)
            yield 3, ms, e, ctx, 1.0 if fl > 0 else 0.0
            j += 1
        elif k == 2:
            yield 2, ms, e, ctx, 0.0
            b += 1; j = 0
        else:
            assert k in (0, 1), f"unexpected forward kind {k}"
            yield k, ms, e, ctx, 0.0


def fit(records):
    J, F = [], []
    for R in records:
        for r in R.values():
            for k, ms, e, ctx, fl in trace(r):
                if k == 3:
                    J.append((ms, fl, ctx / 1000, e))
                elif k == 2:
                    F.append((ms, ctx / 1000, e))
    J, F = np.array(J), np.array(F)
    cj = np.linalg.lstsq(np.c_[np.ones(len(J)), J[:, 1], J[:, 2], J[:, 3]], J[:, 0], rcond=None)[0]
    cf = np.linalg.lstsq(np.c_[np.ones(len(F)), F[:, 1], F[:, 2]], F[:, 0], rcond=None)[0]
    return cj, cf


def price(r, cj, cf, prefill_ms, drafter_ms, other_ms):
    """Seconds this record's work takes at the fitted per-forward costs."""
    t, n = 0.0, 0
    for k, ms, e, ctx, fl in trace(r):
        n += 1
        if k in (1, 3):
            t += cj[0] + cj[1] * fl + cj[2] * ctx / 1000 + cj[3] * e
        elif k == 2:
            t += cf[0] + cf[1] * ctx / 1000 + cf[2] * e
        else:
            t += prefill_ms if prefill_ms is not None else ms
    t += sum(c * drafter_ms.get(kk, 0.0) for kk, c in (r.get("drafter_n_by_kind") or {}).items())
    return (t + other_ms * n) / 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.chdir(os.path.abspath(os.path.join(CODE, "..")))
    rep = {"definition": __doc__, "per_set": {}, "pooled": {}}
    pool = {arm: {"v": [], "p3": [], "p4": [], "p4_pred": [], "v_pred": []} for arm in ("rd1", "rd2")}
    for s in SETS:
        P4 = {arm: load_dir(f"logs/phase4/decode/{arm}/{s}") for arm in ("rd1", "rd2")}
        V = load_dir(f"logs/phase4/decode/vanilla_here/{s}")
        cj, cf = fit([P4["rd1"], P4["rd2"]])
        rep["per_set"][s] = {"fit_judge_a_flag_ctx_experts": [float(x) for x in cj],
                             "fit_flush_a_ctx_experts": [float(x) for x in cf]}
        for arm in ("rd1", "rd2"):
            PR, P3 = P4[arm], load_dir(f"{P3DIR}/{arm}/{s}")
            ids = sorted(set(PR) & set(P3) & set(V))
            dms, dn, oth, nfw = {}, {}, 0.0, 0
            for r in PR.values():
                for kk, v in (r.get("drafter_ms_by_kind") or {}).items():
                    dms[kk] = dms.get(kk, 0.0) + v
                for kk, v in (r.get("drafter_n_by_kind") or {}).items():
                    dn[kk] = dn.get(kk, 0) + v
                oth += r["seconds"] - r["moe_seconds"] - (r.get("drafter_seconds") or 0.0); nfw += len(r["forwards"])
            drafter_ms = {kk: dms[kk] / dn[kk] for kk in dms if dn.get(kk)}
            other_ms = 1000 * oth / nfw
            v_other = 1000 * sum(V[i]["seconds"] - V[i]["moe_seconds"] for i in ids) / sum(len(V[i]["forwards"]) for i in ids)
            pre = {i: next((f[1] for f in PR[i]["forwards"] if f[0] == 0), None) for i in ids}
            vpre = {i: next((f[1] for f in V[i]["forwards"] if f[0] == 0), None) for i in ids}
            p3 = np.array([price(P3[i], cj, cf, pre[i], drafter_ms, other_ms) for i in ids])
            p4p = np.array([price(PR[i], cj, cf, pre[i], drafter_ms, other_ms) for i in ids])
            vp = np.array([price(V[i], cj, cf, vpre[i], {}, v_other) for i in ids])
            p4 = np.array([PR[i]["seconds"] for i in ids]); v = np.array([V[i]["seconds"] for i in ids])
            for k, x in (("v", v), ("p3", p3), ("p4", p4), ("p4_pred", p4p), ("v_pred", vp)):
                pool[arm][k].append(x)
            rep["per_set"][s][arm] = {
                "n": len(ids), "p3_vs_vanilla": ratio_ci(v, p3), "p4_vs_p3": ratio_ci(p3, p4),
                "p4_vs_vanilla_measured": ratio_ci(v, p4),
                "check_p4_priced_vs_measured_pct": float(100 * (p4p.sum() / p4.sum() - 1)),
                "check_vanilla_priced_vs_measured_pct": float(100 * (vp.sum() / v.sum() - 1)),
                "p3_seconds_per_answer": float(p3.mean()), "p4_seconds_per_answer": float(p4.mean()),
                "vanilla_seconds_per_answer": float(v.mean())}
    for arm in ("rd1", "rd2"):
        c = {k: np.concatenate(x) for k, x in pool[arm].items()}
        rep["pooled"][arm] = {
            "n": int(len(c["v"])), "p3_vs_vanilla": ratio_ci(c["v"], c["p3"]), "p4_vs_p3": ratio_ci(c["p3"], c["p4"]),
            "p4_vs_vanilla_measured": ratio_ci(c["v"], c["p4"]),
            "check_p4_priced_vs_measured_pct": float(100 * (c["p4_pred"].sum() / c["p4"].sum() - 1)),
            "check_vanilla_priced_vs_measured_pct": float(100 * (c["v_pred"].sum() / c["v"].sum() - 1)),
            "p3_seconds_per_answer": float(c["p3"].mean()), "p4_seconds_per_answer": float(c["p4"].mean()),
            "vanilla_seconds_per_answer": float(c["v"].mean())}
    json.dump(rep, open(P.safe_out(a.out), "w"), indent=1)
    f3 = lambda x: f"{x[0]:.2f}× [{x[1]:.2f}, {x[2]:.2f}]"                    # noqa: E731
    L = ["Speed vs the vanilla 30B MoE on this machine (vanilla = 1.00×): Phase 4 measured here; Phase 3 = its recorded "
         "work priced at this machine's measured per-forward costs", "",
         "| Benchmark (questions) | Mode | Phase 3 vs vanilla | Phase 4 vs vanilla (measured) | Phase 4 vs Phase 3 | "
         "Seconds/answer: vanilla | Phase 3 | Phase 4 | Model check: Phase 4 | vanilla |", "|" + "---|" * 10]
    for s in list(SETS) + ["pooled"]:
        for arm in ("rd1", "rd2"):
            o = rep["pooled"][arm] if s == "pooled" else rep["per_set"][s][arm]
            lab = f"**All 7 pooled ({o['n']:,})**" if s == "pooled" else f"{NAME[s]} ({o['n']})"
            L.append(f"| {lab} | {arm} | {f3(o['p3_vs_vanilla'])} | {f3(o['p4_vs_vanilla_measured'])} | {f3(o['p4_vs_p3'])} | "
                     f"{o['vanilla_seconds_per_answer']:.1f} | {o['p3_seconds_per_answer']:.1f} | {o['p4_seconds_per_answer']:.1f} | "
                     f"{o['check_p4_priced_vs_measured_pct']:+.1f}% | {o['check_vanilla_priced_vs_measured_pct']:+.1f}% |")
    open(os.path.splitext(P.safe_out(a.out))[0] + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
