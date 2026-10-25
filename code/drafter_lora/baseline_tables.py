#!/usr/bin/env python
"""Three result tables for the operator (13 Oct 06:15 IST), from the measured files of report_p4.py and speedup_chain.py.

    /venv/p2/bin/python code/drafter_lora/baseline_tables.py --out results/phase4_drafter/baseline_tables.json

  Table V   every entry against the vanilla 30B MoE (baseline = vanilla): Phase 3 rd1 / Phase 4 rd1 / Phase 3 rd2 /
            Phase 4 rd2 -- speed (chained, x vanilla), accuracy delta vs vanilla [95% CI], judge passes per block,
            30B experts activated per token, looping outputs, length-cap hits (vanilla's own value shown first)
  Table rd1 Phase 4 rd1 with TWO baselines side by side: Phase 3 rd1 and vanilla
  Table rd2 the same for rd2
Only benchmarks whose Gate 4 decode is complete enter a table (others are listed as running / queued); pooled rows use
the complete benchmarks only.  Speed: vs Phase 3 = GPU-active time ratio (cross-machine, about 5% optimistic);
vs vanilla = the chained estimate of speedup_chain.py, and -- for Phase 4, once the vanilla 30B has been decoded on this
box (vanilla_same_box.json, complete benchmarks only; the pooled value only when it covers every benchmark of the
table) -- the MEASURED same-machine wall-clock ratio sum(seconds vanilla) / sum(seconds Phase 4) [95% paired bootstrap].
Phase 3 was decoded on the Phase 3 machine: with phase3_same_box.json present, its speed on THIS machine (vs vanilla,
and Phase 4 vs Phase 3) is its recorded per-forward work priced at this machine's measured per-forward costs
(phase3_same_box.py; the pricing reproduces the measured Phase 4 and vanilla times within -0.7% to +2.5%) and replaces
the chained / cross-machine figures in the .md (they stay in the .json as *_chained).  Counts and accuracy are measured
on the same items.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FULL = {"ifeval541": 541, "math500": 500, "gpqa": 198, "aime": 90, "humaneval": 164, "mbpp": 257, "gsm8k500": 500}
NAME = {"ifeval541": "IFEval", "math500": "MATH-500", "gpqa": "GPQA", "aime": "AIME", "humaneval": "HumanEval",
        "mbpp": "MBPP", "gsm8k500": "GSM8K"}


def agg(per, sets, who):
    """Pooled counts over `sets` from report_p4 aggregates (who = new | phase3 | vanilla)."""
    P = [per[s][who] for s in sets]
    blocks = sum(p["blocks"] for p in P)
    tokens = sum(p["tokens"] for p in P)
    n = sum(p["n"] for p in P)
    out = {"n": n, "accuracy_pct": 100.0 * sum(p["correct"] for p in P) / n,
           "judge_passes_per_block": sum(p["judge_passes_per_block"] * p["blocks"] for p in P) / blocks,
           "experts_per_token": sum(p["experts_activated_total"] for p in P) / tokens,
           "loops": sum(p["loops"] for p in P), "length_cap_hits": sum(p["length_cap_hits"] for p in P)}
    if all(p.get("drafter_fwd_per_block") is not None for p in P):
        out["drafter_fwd_per_block"] = sum(p["drafter_fwd_per_block"] * p["blocks"] for p in P) / blocks
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/phase4_drafter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = json.load(open(os.path.join(a.dir, "speedup_chain.json")))["arms"]
    M = {arm: json.load(open(os.path.join(a.dir, f"measured_{arm}.json"))) for arm in ("rd1", "rd2")}
    vsb = os.path.join(a.dir, "vanilla_same_box.json")
    VSB = json.load(open(vsb))["arms"] if os.path.exists(vsb) else {}
    p3b = os.path.join(a.dir, "phase3_same_box.json")
    P3S = json.load(open(p3b)) if os.path.exists(p3b) else None

    def p3s(arm, s, key):
        """phase3_same_box.json value [v, lo, hi] for a set or 'pooled', or None."""
        if not P3S:
            return None
        o = P3S["pooled"].get(arm) if s == "pooled" else P3S["per_set"].get(s, {}).get(arm)
        return (o or {}).get(key)
    done = {arm: [s for s in FULL if "n" in ch[arm]["per_set"].get(s, {})
                  and (M[arm]["vs_phase3"]["per_set"].get(s) or {}).get("n_paired", 0) >= FULL[s]] for arm in M}
    status = {arm: {s: ("done" if s in done[arm] else
                        f"running ({(M[arm]['vs_phase3']['per_set'].get(s) or {}).get('n_paired', 0)}/{FULL[s]})"
                        if (M[arm]["vs_phase3"]["per_set"].get(s) or {}).get("n_paired", 0) else "queued")
                    for s in FULL} for arm in M}
    rep = {"definition": __doc__, "status": status, "vanilla_table": [], "rd_tables": {"rd1": [], "rd2": []}}

    def measured(arm, s):
        """Phase 4 wall-clock speed vs vanilla measured on this box, [value, lo, hi], or None."""
        o = VSB.get(arm, {})
        if s == "pooled":
            p = o.get("pooled")
            return p["speedup_wall"] if p and p.get("complete") and sorted(p.get("sets", [])) == sorted(done[arm]) else None
        c = o.get("per_set", {}).get(s)
        return c["speedup_wall"] if c and c.get("complete") else None

    def row_for(arm, s):
        """(Phase 4 counts, Phase 3 counts, vanilla counts, chain entry) for a set or 'pooled'."""
        if s == "pooled":
            sets = done[arm]
            return (agg(M[arm]["vs_phase3"]["per_set"], sets, "new"), agg(M[arm]["vs_phase3"]["per_set"], sets, "phase3"),
                    agg(M[arm]["vs_vanilla"]["per_set"], sets, "vanilla"), ch[arm].get("pooled", {}))
        p3 = M[arm]["vs_phase3"]["per_set"][s]; pv = M[arm]["vs_vanilla"]["per_set"][s]
        return (agg({s: p3}, [s], "new"), agg({s: p3}, [s], "phase3"), agg({s: pv}, [s], "vanilla"),
                ch[arm]["per_set"][s])

    common = [s for s in FULL if s in done["rd1"] and s in done["rd2"]]
    for s in common + ["pooled"]:
        r = {"benchmark": "pooled" if s == "pooled" else NAME[s]}
        for arm in ("rd1", "rd2"):
            if s == "pooled" and not done[arm]:
                continue
            n4, n3, nv, c = row_for(arm, s) if s != "pooled" else row_for(arm, "pooled")
            r.setdefault("vanilla", {"accuracy_pct": nv["accuracy_pct"], "judge_passes_per_block": nv["judge_passes_per_block"],
                                     "experts_per_token": nv["experts_per_token"], "loops": nv["loops"],
                                     "length_cap_hits": nv["length_cap_hits"], "n": nv["n"]})
            for tag, nn, sp, da, sm in (("p3_" + arm, n3, c.get("p3_vs_vanilla"), c.get("dacc_p3_vs_vanilla"),
                                         p3s(arm, s, "p3_vs_vanilla")),
                                        ("p4_" + arm, n4, c.get("p4_vs_vanilla"), c.get("dacc_p4_vs_vanilla"),
                                         measured(arm, s))):
                r[tag] = {"speed_vs_vanilla_chained": sp, "speed_vs_vanilla_measured": sm,
                          "accuracy_pct": nn["accuracy_pct"], "dacc_vs_vanilla": da,
                          "judge_passes_per_block": nn["judge_passes_per_block"],
                          "passes_vs_vanilla_pct": 100 * (nn["judge_passes_per_block"] / nv["judge_passes_per_block"] - 1),
                          "experts_per_token": nn["experts_per_token"],
                          "experts_vs_vanilla_pct": 100 * (nn["experts_per_token"] / nv["experts_per_token"] - 1),
                          "loops": nn["loops"], "length_cap_hits": nn["length_cap_hits"]}
        r["sets_in_pool"] = {arm: done[arm] for arm in M} if s == "pooled" else None
        rep["vanilla_table"].append(r)
    for arm in ("rd1", "rd2"):
        for s in done[arm] + (["pooled"] if done[arm] else []):
            n4, n3, nv, c = row_for(arm, s)
            if s == "pooled":
                d3, dv = c.get("dacc_p4_vs_p3"), c.get("dacc_p4_vs_vanilla")
                a_ratio = (c["p4_vs_vanilla"] / c["p3_vs_vanilla"]) if c.get("p3_vs_vanilla") else None
            else:
                d3 = M[arm]["vs_phase3"]["per_set"][s]["delta_accuracy_pts_ci95"]; dv = c.get("dacc_p4_vs_vanilla")
                a_ratio = c.get("A_p4_over_p3_gpu")
            rep["rd_tables"][arm].append({
                "benchmark": "pooled" if s == "pooled" else NAME[s], "n": n4["n"],
                "speed_vs_p3_cross_machine_gpu": a_ratio, "speed_vs_vanilla_chained": c.get("p4_vs_vanilla"),
                "speed_vs_vanilla_measured": measured(arm, s), "speed_vs_p3_this_machine": p3s(arm, s, "p4_vs_p3"),
                "accuracy": {"p4": n4["accuracy_pct"], "p3": n3["accuracy_pct"], "vanilla": nv["accuracy_pct"],
                             "d_vs_p3": d3, "d_vs_vanilla": dv},
                "judge_passes_per_block": {"p4": n4["judge_passes_per_block"], "p3": n3["judge_passes_per_block"],
                                           "vanilla": nv["judge_passes_per_block"],
                                           "vs_p3_pct": 100 * (n4["judge_passes_per_block"] / n3["judge_passes_per_block"] - 1),
                                           "vs_vanilla_pct": 100 * (n4["judge_passes_per_block"] / nv["judge_passes_per_block"] - 1)},
                "drafter_fwd_per_block": {"p4": n4.get("drafter_fwd_per_block"), "p3": n3.get("drafter_fwd_per_block"),
                                          "vanilla": 0.0,
                                          "vs_p3_pct": 100 * (n4["drafter_fwd_per_block"] / n3["drafter_fwd_per_block"] - 1)},
                "experts_per_token": {"p4": n4["experts_per_token"], "p3": n3["experts_per_token"],
                                      "vanilla": nv["experts_per_token"],
                                      "vs_p3_pct": 100 * (n4["experts_per_token"] / n3["experts_per_token"] - 1),
                                      "vs_vanilla_pct": 100 * (n4["experts_per_token"] / nv["experts_per_token"] - 1)},
                "loops": {"p4": n4["loops"], "p3": n3["loops"], "vanilla": nv["loops"]},
                "length_cap_hits": {"p4": n4["length_cap_hits"], "p3": n3["length_cap_hits"], "vanilla": nv["length_cap_hits"]}})
    json.dump(rep, open(a.out, "w"), indent=1)

    f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"                       # noqa: E731
    ci = lambda d: "–" if not d else f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"   # noqa: E731
    mx = lambda d: "–" if not d else f"**{d[0]:.2f}×** [{d[1]:.2f}, {d[2]:.2f}]"  # noqa: E731
    L = ["## Table V: everything vs the vanilla 30B MoE (baseline = vanilla)", "",
         "| Benchmark | Vanilla acc % | Speed ×vanilla on this machine (vanilla = 1.00×): P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 | Δacc vs vanilla: P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 | Judge passes/blk: vanilla | P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 | Experts/token: vanilla | P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 | Loops: vanilla | P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 |",
         "|" + "---|" * 25]
    for r in rep["vanilla_table"]:
        v = r["vanilla"]; g = [r.get(k, {}) for k in ("p3_rd1", "p4_rd1", "p3_rd2", "p4_rd2")]
        L.append(f"| {r['benchmark']} | {v['accuracy_pct']:.1f} | "
                 + " | ".join(mx(x.get("speed_vs_vanilla_measured")) for x in g)
                 + " | " + " | ".join(ci(x.get('dacc_vs_vanilla')) for x in g)
                 + f" | {v['judge_passes_per_block']:.2f} | " + " | ".join(f"{x['judge_passes_per_block']:.2f} ({x['passes_vs_vanilla_pct']:+.0f}%)" for x in g)
                 + f" | {v['experts_per_token']:.0f} | " + " | ".join(f"{x['experts_per_token']:.0f} ({x['experts_vs_vanilla_pct']:+.0f}%)" for x in g)
                 + f" | {v['loops']} | " + " | ".join(str(x["loops"]) for x in g) + " |")
    for arm in ("rd1", "rd2"):
        L += ["", f"## Table {arm}: Phase 4 {arm} vs its two baselines (Phase 3 {arm} | vanilla)", "",
              "| Benchmark (n) | Speed on this machine: vs P3 | vs vanilla | Accuracy %: P4 | P3 | vanilla | Δ vs P3 | Δ vs vanilla | Judge passes/blk: P4 | P3 | vanilla | vs P3 | vs vanilla | Drafter fwd/blk: P4 | P3 | vs P3 | Experts/token: P4 | P3 | vanilla | vs P3 | vs vanilla | Loops: P4 | P3 | vanilla |",
              "|" + "---|" * 24]
        for r in rep["rd_tables"][arm]:
            ac, jp, df, ex, lo = r["accuracy"], r["judge_passes_per_block"], r["drafter_fwd_per_block"], r["experts_per_token"], r["loops"]
            L.append(f"| {r['benchmark']} ({r['n']}) | {mx(r.get('speed_vs_p3_this_machine'))} | {mx(r.get('speed_vs_vanilla_measured'))} | {ac['p4']:.1f} | {ac['p3']:.1f} | {ac['vanilla']:.1f} | {ci(ac['d_vs_p3'])} | {ci(ac['d_vs_vanilla'])} | "
                     f"{jp['p4']:.2f} | {jp['p3']:.2f} | {jp['vanilla']:.2f} | {jp['vs_p3_pct']:+.1f}% | {jp['vs_vanilla_pct']:+.1f}% | "
                     f"{f(df['p4'], 1)} | {f(df['p3'], 1)} | {df['vs_p3_pct']:+.1f}% | {ex['p4']:.0f} | {ex['p3']:.0f} | {ex['vanilla']:.0f} | {ex['vs_p3_pct']:+.1f}% | {ex['vs_vanilla_pct']:+.1f}% | "
                     f"{lo['p4']} | {lo['p3']} | {lo['vanilla']} |")
        pend = [f"{NAME[s]}: {st}" for s, st in status[arm].items() if st != "done"]
        if pend:
            L.append(f"\n{arm} not yet in the table: " + "; ".join(pend))
    L += ["", "Speed on this machine (vanilla 30B = 1.00×), time per answer, [95% paired bootstrap]. Phase 4 and the vanilla 30B"
          " were decoded on this machine (vanilla_same_box.json). Phase 3 was decoded on the Phase 3 machine; its recorded"
          " per-forward work is priced at this machine's measured per-forward costs (phase3_same_box.json), a pricing that"
          " reproduces the measured Phase 4 times within ±1.0% and the vanilla times within -0.7% to +2.5% per benchmark."]
    open(os.path.splitext(a.out)[0] + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
