#!/usr/bin/env python
"""Phase 3 and Phase 4 speed vs the vanilla 30B MoE by CHAINING per-benchmark factors (operator, 13 Oct 05:00 IST).
Time per answer throughout (the plan's speed measure; tokens/s is inflated by looping outputs).

    /venv/p2/bin/python code/drafter_lora/speedup_chain.py --out results/phase4_drafter/speedup_chain.json

  C  Phase 1 base over vanilla  = base_over_stock_phase2 (tok/s, the Phase 2 page's figure for the Phase 1 machine,
                                  from results/phase3_8gpu/final_numbers.json; partly computed there) x
                                  (vanilla tokens / base tokens on the same items) -> a time-per-answer ratio
  B  Phase 3 rd over base       = sum(base seconds) / sum(rd seconds), same items, both decoded on the Phase 3 box
                                  (measured, same machine)
  A  Phase 4 rd over Phase 3 rd = sum(Phase 3 gpu_active_s) / sum(Phase 4 gpu_active_s), same items (NVML GPM GPU-active
                                  time, same GPU model; CROSS-MACHINE: probably flatters Phase 4 by ~5%, see REPORT.md)
  Phase 3 vs vanilla = B x C ;  Phase 4 vs vanilla = A x B x C   (two multiplications)
Pooled: estimated vanilla time summed over benchmarks / the arm's time summed (each benchmark's time x its factor).
Accuracy: measured, paired with the vanilla records on the same items (bootstrap 2,000, default_rng(0)).
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

from drafter_lora.compare_gates import SETS, load_dir, vanilla_glob      # noqa: E402

FULL = {"ifeval541": 541, "math500": 500, "gpqa": 198, "aime": 90, "humaneval": 164, "mbpp": 257, "gsm8k500": 500}
NAME = {"ifeval541": "IFEval", "math500": "MATH-500", "gpqa": "GPQA", "aime": "AIME", "humaneval": "HumanEval",
        "mbpp": "MBPP", "gsm8k500": "GSM8K"}


def tot(R, ids, k):
    return float(sum(R[i][k] for i in ids))


def acc(R, ids):
    return 100.0 * float(np.mean([bool(R[i]["correct"]) for i in ids]))


def delta_ci(A, B, ids):
    d = np.array([int(bool(A[i]["correct"])) - int(bool(B[i]["correct"])) for i in ids])
    rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4", default="logs/phase4/decode", help="Phase 4 Gate 4 records: <dir>/{rd1,rd2}/<set>/")
    ap.add_argument("--p3", default="logs/phase4/ref/phase3/logs/phase3_8gpu/decode")
    ap.add_argument("--vanilla", default="logs/phase4/ref/phase2/assets/ref_stock_30B/*_B32_t95_stock.{set}.shard*.jsonl")
    ap.add_argument("--base1", default="logs/phase2/ref/S0_128M_nopool_by_set/{set}/*.jsonl",
                    help="Phase 1 base records (Phase 1 machine), for the token-count conversion of C")
    ap.add_argument("--p3-final", default="results/phase3_8gpu/final_numbers.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.chdir(P.REPO_ROOT if hasattr(P, "REPO_ROOT") else os.path.abspath(os.path.join(CODE, "..")))
    fin = json.load(open(a.p3_final))["vs_stock"]["benchmarks"]
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "definition": __doc__, "arms": {}}
    for arm in ("rd1", "rd2"):
        per, pool = {}, {"p4_gpu": 0.0, "p4_est_van": 0.0, "p3_sec": 0.0, "p3_est_van": 0.0, "ids": []}
        PA, PB, PV = {}, {}, {}
        for s in SETS:
            N = load_dir(os.path.join(a.p4, arm, s))
            if len(N) < FULL[s]:
                per[s] = {"status": f"running or pending ({len(N)}/{FULL[s]} records)"}
                continue
            R3, B3 = load_dir(os.path.join(a.p3, arm, s)), load_dir(os.path.join(a.p3, "base", s))
            V, B1 = load_dir(vanilla_glob(a.vanilla, s)), load_dir(a.base1.format(set=s))
            ids = sorted(set(N) & set(R3) & set(B3) & set(V))
            idc = sorted(set(V) & set(B1))
            A = tot(R3, ids, "gpu_active_s") / tot(N, ids, "gpu_active_s")
            B = tot(B3, ids, "seconds") / tot(R3, ids, "seconds")
            c_tps = fin[s]["base_over_stock_phase2"]
            C = c_tps * tot(V, idc, "n_gen_to_eos") / tot(B1, idc, "n_gen_to_eos")
            per[s] = {"n": len(ids), "A_p4_over_p3_gpu": A, "B_p3_over_base_time": B, "C_base_over_vanilla_time": C,
                      "C_tps_phase2": c_tps, "p3_vs_vanilla": B * C, "p4_vs_vanilla": A * B * C,
                      "acc_vanilla": acc(V, ids), "acc_p3": acc(R3, ids), "acc_p4": acc(N, ids),
                      "dacc_p3_vs_vanilla": delta_ci(R3, V, ids), "dacc_p4_vs_vanilla": delta_ci(N, V, ids),
                      "dacc_p4_vs_p3": delta_ci(N, R3, ids)}
            g4, s3 = tot(N, ids, "gpu_active_s"), tot(R3, ids, "seconds")
            pool["p4_gpu"] += g4; pool["p4_est_van"] += g4 * A * B * C
            pool["p3_sec"] += s3; pool["p3_est_van"] += s3 * B * C
            for i in ids:
                k = len(PA); PA[k], PB[k], PV[k] = N[i], R3[i], V[i]
        done = [s for s in SETS if "n" in per[s]]
        out = {"per_set": per, "sets_done": done}
        if done:
            ix = list(PA)
            out["pooled"] = {"n": len(ix), "sets": done,
                             "p3_vs_vanilla": pool["p3_est_van"] / pool["p3_sec"],
                             "p4_vs_vanilla": pool["p4_est_van"] / pool["p4_gpu"],
                             "acc_vanilla": acc(PV, ix), "acc_p3": acc(PB, ix), "acc_p4": acc(PA, ix),
                             "dacc_p3_vs_vanilla": delta_ci(PB, PV, ix), "dacc_p4_vs_vanilla": delta_ci(PA, PV, ix),
                             "dacc_p4_vs_p3": delta_ci(PA, PB, ix)}
        rep["arms"][arm] = out
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    lines = ["| Mode | Benchmark (n) | Phase 3 vs vanilla | Phase 4 vs vanilla | Accuracy %: vanilla / Phase 3 / Phase 4 |",
             "|---|---|---|---|---|"]
    for arm, o in rep["arms"].items():
        for s in SETS:
            v = o["per_set"][s]
            if "n" not in v:
                lines.append(f"| {arm} | {NAME[s]} | {v['status']} | | |")
                continue
            lines.append(f"| {arm} | {NAME[s]} ({v['n']}) | {v['p3_vs_vanilla']:.2f}x | {v['p4_vs_vanilla']:.2f}x | "
                         f"{v['acc_vanilla']:.1f} / {v['acc_p3']:.1f} / {v['acc_p4']:.1f} |")
        if "pooled" in o:
            p = o["pooled"]
            lines.append(f"| {arm} | **pooled ({p['n']}, {len(p['sets'])} sets)** | **{p['p3_vs_vanilla']:.2f}x** | "
                         f"**{p['p4_vs_vanilla']:.2f}x** | {p['acc_vanilla']:.1f} / {p['acc_p3']:.1f} / {p['acc_p4']:.1f} |")
    open(os.path.splitext(a.out)[0] + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
