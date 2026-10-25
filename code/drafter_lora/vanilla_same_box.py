#!/usr/bin/env python
"""MEASURED same-machine speed of Phase 4 vs the vanilla 30B (both decoded on this box, same items).  No extrapolation.

    /venv/p2/bin/python code/drafter_lora/vanilla_same_box.py --out results/phase4_drafter/vanilla_same_box.json \
        [--include-partial]

For question i of benchmark b (records matched by idx):  w = `seconds` (cuda.synchronize + perf_counter around the whole
decode), g = `gpu_active_s` (NVML GPM hardware counters), V = vanilla 30B stock decode on this box
(logs/phase4/decode/vanilla_here), P = Phase 4 rd1 or rd2 on this box (logs/phase4/decode/{rd1,rd2}).  Over a set of
questions I (one benchmark, or all benchmarks used, pooled):

    speedup (wall-clock)  S_w(I) = sum_{i in I} w_V(i) / sum_{i in I} w_P(i)
    speedup (GPU time)    S_g(I) = sum_{i in I} g_V(i) / sum_{i in I} g_P(i)
    speedup (tok/s)       S_t(I) = [sum_I t_P(i) / sum_I w_P(i)] / [sum_I t_V(i) / sum_I w_V(i)],  t = n_gen_to_eos
                          (decoding speed per generated token; differs from S_w when the two answers' lengths differ)
    exact decomposition (W = wall seconds, M = 30B MoE seconds, N = generated tokens, F = 30B forwards, all summed
    over I):  S_w = (N_V/N_P) x [(F/N)_V/(F/N)_P] x [(M/F)_V/(M/F)_P] x [(M/W)_P/(M/W)_V]
              = answer length x 30B forwards per token x time per 30B forward x time outside 30B forwards
    95% range: paired bootstrap over the questions of I (resample i with replacement; 2,000 resamples, numpy
    default_rng(0)); the same procedure as every accuracy range of this run.
Every other quantity (accuracy and its paired delta, time / GPU / CPU / MoE / drafter seconds per answer, experts
activated, judge passes per block, loops, tok/s) is report_p4.measure on the same paired questions, i.e. the
definitions of measured_rd1.json / measured_rd2.json.  Also: the share of vanilla questions whose tokens are identical
to the earlier vanilla records (ref_match, checked per question by run_da_decode --ref-dir).
Benchmarks used: those whose vanilla decode here is complete (.complete marker); with --include-partial also those
still decoding, on the questions finished so far (marked "complete": false, n = questions so far), once at least
--min-partial (default 30) questions are done: a handful of questions gives meaningless ratios.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402

from drafter_lora.compare_gates import SETS, load_dir                    # noqa: E402
from drafter_lora.report_p4 import measure, paired_delta                 # noqa: E402

NAME = {"ifeval541": "IFEval", "math500": "MATH-500", "gpqa": "GPQA", "aime": "AIME", "humaneval": "HumanEval",
        "mbpp": "MBPP", "gsm8k500": "GSM8K"}
B = 2000


def tps_ci(tp, wp, tv, wv):
    rng = np.random.default_rng(0)
    n = len(tp)
    f = lambda ix: (tp[ix].sum() / wp[ix].sum()) / (tv[ix].sum() / wv[ix].sum())      # noqa: E731
    bs = [f(rng.integers(0, n, n)) for _ in range(B)]
    return [float(f(np.arange(n))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ratio_ci(num: np.ndarray, den: np.ndarray):
    rng = np.random.default_rng(0)
    n = len(num)
    bs = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        bs.append(num[ix].sum() / den[ix].sum())
    return [float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def compare(V, Pn, ids):
    v, p = measure(V, ids), measure(Pn, ids)
    wv = np.array([V[i]["seconds"] for i in ids]); wp = np.array([Pn[i]["seconds"] for i in ids])
    gv = np.array([V[i]["gpu_active_s"] for i in ids]); gp = np.array([Pn[i]["gpu_active_s"] for i in ids])
    tv = np.array([V[i]["n_gen_to_eos"] for i in ids], float); tp = np.array([Pn[i]["n_gen_to_eos"] for i in ids], float)
    dec = {"length": v["tokens"] / p["tokens"],
           "fwd_per_token": (v["n_forward_30b"] / v["tokens"]) / (p["n_forward_30b"] / p["tokens"]),
           "time_per_fwd": (v["moe_s"] / v["n_forward_30b"]) / (p["moe_s"] / p["n_forward_30b"]),
           "outside_30b": (p["moe_s"] / p["seconds_total"]) / (v["moe_s"] / v["seconds_total"])}
    dec["product"] = dec["length"] * dec["fwd_per_token"] * dec["time_per_fwd"] * dec["outside_30b"]
    dec["ms_per_fwd_vanilla"] = 1000 * v["moe_s"] / v["n_forward_30b"]
    dec["ms_per_fwd_phase4"] = 1000 * p["moe_s"] / p["n_forward_30b"]
    return {"n": len(ids), "decomposition": dec, "speedup_wall": ratio_ci(wv, wp), "speedup_gpu": ratio_ci(gv, gp),
            "speedup_tok_s": tps_ci(tp, wp, tv, wv), "tokens_ratio_p4_over_vanilla": float(tp.sum() / tv.sum()),
            "dacc_p4_minus_vanilla": [float(x) for x in paired_delta(Pn, V, ids)],
            "moe_s_ratio_vanilla_over_p4": v["moe_s"] / p["moe_s"],
            "cpu_thread_s_ratio_vanilla_over_p4": v["cpu_thread_s"] / p["cpu_thread_s"],
            "experts_per_token_ratio_p4_over_vanilla": p["experts_per_token"] / v["experts_per_token"],
            "judge_passes_reduction_pct": 100 * (1 - p["judge_passes_per_block"] / v["judge_passes_per_block"]),
            "vanilla": v, "phase4": p}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vanilla", default="logs/phase4/decode/vanilla_here")
    ap.add_argument("--p4", default="logs/phase4/decode")
    ap.add_argument("--include-partial", action="store_true")
    ap.add_argument("--min-partial", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.chdir(os.path.abspath(os.path.join(CODE, "..")))
    complete = {s: os.path.exists(os.path.join(a.vanilla, s, ".complete")) for s in SETS}
    V = {s: load_dir(os.path.join(a.vanilla, s)) for s in SETS}
    used = [s for s in SETS if V[s] and (complete[s] or (a.include_partial and len(V[s]) >= a.min_partial))]
    rep = {"definition": __doc__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sets_used": used, "complete": {s: complete[s] for s in used}, "n_vanilla_here": {s: len(V[s]) for s in SETS},
           "token_identity_vs_vanilla_records": {}, "vanilla_alone": {}, "arms": {}}
    for s in used:
        rm = [r.get("ref_match") for r in V[s].values()]
        rep["token_identity_vs_vanilla_records"][s] = {
            "n": len(rm), "with_reference": sum(m is not None for m in rm),
            "identical_tokens": sum(bool(m and m.get("gen_ids")) for m in rm),
            "identical_all_fields": sum(bool(m and all(m.values())) for m in rm)}
        rep["vanilla_alone"][s] = measure(V[s], sorted(V[s]))
    for arm in ("rd1", "rd2"):
        per, pooled_V, pooled_P = {}, {}, {}
        for s in used:
            Pn = load_dir(os.path.join(a.p4, arm, s))
            ids = sorted(set(V[s]) & set(Pn))
            if not ids:
                continue
            per[s] = compare(V[s], Pn, ids)
            per[s]["complete"] = complete[s]
            for i in ids:
                k = len(pooled_V); pooled_V[k], pooled_P[k] = V[s][i], Pn[i]
        out = {"per_set": per}
        if pooled_V:
            out["pooled"] = compare(pooled_V, pooled_P, list(pooled_V))
            out["pooled"]["sets"] = list(per)
            out["pooled"]["complete"] = all(complete[s] for s in per)
        rep["arms"][arm] = out
    json.dump(rep, open(P.safe_out(a.out), "w"), indent=1)

    f3 = lambda c: f"**{c[0]:.2f}×** [{c[1]:.2f}, {c[2]:.2f}]"                     # noqa: E731
    f1 = lambda x: "—" if x is None else f"{x:.1f}"                                   # noqa: E731
    rows = []
    for s in used + ["pooled"]:
        for arm in ("rd1", "rd2"):
            o = rep["arms"].get(arm, {})
            c = o.get("pooled") if s == "pooled" else o.get("per_set", {}).get(s)
            if not c:
                continue
            if s == "pooled":
                lab = f"**Pooled, {len(c['sets'])} benchmarks ({c['n']})**"
            else:
                lab = f"{NAME[s]} ({c['n']}{'' if c['complete'] else ' so far'})"
            rows.append((lab, arm, c))
    L = ["Speed and accuracy, Phase 4 vs vanilla 30B, both decoded on this machine, same questions",
         "",
         "| Benchmark (questions) | Mode | Speedup, wall-clock [95% range] | Speedup, GPU time [95% range] | "
         "Speedup, tok/s [95% range] | Accuracy %: vanilla | Phase 4 | Δ points [95% range] | "
         "Judge passes/block: vanilla | Phase 4 | Tokens/answer: vanilla | Phase 4 |",
         "|" + "---|" * 12]
    for lab, arm, c in rows:
        v, p, d = c["vanilla"], c["phase4"], c["dacc_p4_minus_vanilla"]
        L.append(f"| {lab} | {arm} | {f3(c['speedup_wall'])} | {f3(c['speedup_gpu'])} | {f3(c['speedup_tok_s'])} | "
                 f"{v['accuracy_pct']:.1f} | {p['accuracy_pct']:.1f} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] | "
                 f"{v['judge_passes_per_block']:.2f} | {p['judge_passes_per_block']:.2f} | "
                 f"{v['tokens'] / v['n']:.0f} | {p['tokens'] / p['n']:.0f} |")
    L += ["", "Time and compute per answer (seconds), same questions",
          "",
          "| Benchmark (questions) | Mode | Wall: vanilla | Phase 4 | GPU-active: vanilla | Phase 4 | CPU (decode thread): "
          "vanilla | Phase 4 | 30B MoE forwards: vanilla | Phase 4 | Drafter forwards: Phase 4 | Experts/token: vanilla | "
          "Phase 4 | Loops: vanilla | Phase 4 |",
          "|" + "---|" * 15]
    for lab, arm, c in rows:
        v, p = c["vanilla"], c["phase4"]
        L.append(f"| {lab} | {arm} | {f1(v['seconds_per_answer'])} | {f1(p['seconds_per_answer'])} | "
                 f"{f1(v['gpu_active_s_per_answer'])} | {f1(p['gpu_active_s_per_answer'])} | "
                 f"{f1(v['cpu_thread_s_per_answer'])} | {f1(p['cpu_thread_s_per_answer'])} | "
                 f"{f1(v['moe_s_per_answer'])} | {f1(p['moe_s_per_answer'])} | {f1(p['drafter_s_per_answer'])} | "
                 f"{v['experts_per_token']:.0f} | {p['experts_per_token']:.0f} | {v['loops']} | {p['loops']} |")
    L += ["", "Exact decomposition of the wall-clock speedup S_w (product of the four factors = S_w)", "",
          "| Benchmark (questions) | Mode | Answer length N_V/N_P | 30B forwards per token (F/N)_V/(F/N)_P | "
          "Time per 30B forward (ms): vanilla | Phase 4 | ratio | Time outside 30B forwards (drafter, host) | Product = S_w |",
          "|" + "---|" * 9]
    for lab, arm, c in rows:
        d = c["decomposition"]
        L.append(f"| {lab} | {arm} | {d['length']:.3f} | {d['fwd_per_token']:.3f} | {d['ms_per_fwd_vanilla']:.1f} | "
                 f"{d['ms_per_fwd_phase4']:.1f} | {d['time_per_fwd']:.3f} | {d['outside_30b']:.3f} | **{d['product']:.3f}** |")
    ti = rep["token_identity_vs_vanilla_records"]
    L.append("\nVanilla decoded here vs the earlier vanilla records, questions with identical tokens: "
             + "; ".join(f"{NAME[s]} {t['identical_tokens']}/{t['with_reference']}" for s, t in ti.items()))
    open(os.path.splitext(P.safe_out(a.out))[0] + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
