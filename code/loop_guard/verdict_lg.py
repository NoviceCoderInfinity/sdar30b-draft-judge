"""Pre-registered verdict for phase 5 experiment 1 (the loop guard), read from report_p4.py's JSON.

The report was made with --phase3 pointing at PHASE 4's records of the same arm on the same box (so its "vs_phase3"
block is the paired comparison against Phase 4) and --vanilla at the vanilla records decoded on the same box.

Rule (docs/PHASE5_LOOP_GUARD.md), per arm, on IFEval-541:
  PASS     loops <= 85  and  paired delta accuracy vs Phase 4 >= +2.0 points
  KILL     loops >  95  or   paired delta accuracy vs Phase 4 <  +1.0 points
  BETWEEN  otherwise (report; the operator decides)
Targets, reported but not required: loops <= 60; the 95% range of the delta vs vanilla covers 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
from drafter_lora.compare_gates import load_dir   # noqa: E402

SET = "ifeval541"
PASS_LOOPS, KILL_LOOPS, PASS_DACC, KILL_DACC, TARGET_LOOPS = 85, 95, 2.0, 1.0, 60


def guard_counts(new_dir: str) -> dict:
    R = load_dir(os.path.join(new_dir, SET))
    slots = sum(sum(r.get("loopguard_slots_per_block") or [0]) for r in R.values())
    changed = sum(sum(r.get("loopguard_changed_per_block") or [0]) for r in R.values())
    blocks = sum(r["n_blocks"] for r in R.values())
    n_lg = sum(1 for r in R.values() if r.get("loop_guard"))
    return {"records": len(R), "records_with_loop_guard": n_lg, "guard_slots_per_block": slots / blocks if blocks else None,
            "guard_changed_per_block": changed / blocks if blocks else None, "guard_changed_total": changed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("rd1", "rd2"))
    ap.add_argument("--report", required=True, help="report_p4.py JSON for this arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = json.load(open(a.report))
    p4 = rep["vs_phase3"]["per_set"][SET]            # paired vs Phase 4 (same box)
    van = (rep.get("vs_vanilla") or {}).get("per_set", {}).get(SET)
    new, ref = p4["new"], p4["phase3"]
    d_p4 = p4["delta_accuracy_pts_ci95"]
    loops = int(new["loops"])
    if loops <= PASS_LOOPS and d_p4[0] >= PASS_DACC:
        status = "PASS"
    elif loops > KILL_LOOPS or d_p4[0] < KILL_DACC:
        status = "KILL"
    else:
        status = "BETWEEN"
    out = {
        "arm": a.arm, "set": SET, "status": status, "n_paired": p4["n_paired"],
        "rule": {"pass": f"loops <= {PASS_LOOPS} and dAcc vs Phase 4 >= +{PASS_DACC}",
                 "kill": f"loops > {KILL_LOOPS} or dAcc vs Phase 4 < +{KILL_DACC}", "target_loops": TARGET_LOOPS},
        "loops": {"phase4": int(ref["loops"]), "guard": loops, "target_met": loops <= TARGET_LOOPS},
        "length_cap_hits": {"phase4": int(ref["length_cap_hits"]), "guard": int(new["length_cap_hits"])},
        "accuracy_pct": {"phase4": ref["accuracy_pct"], "guard": new["accuracy_pct"]},
        "delta_acc_vs_phase4_pts_ci95": d_p4,
        "delta_acc_vs_vanilla_pts_ci95": (van["delta_accuracy_pts_ci95"] if van else None),
        "vanilla_ci_covers_0": (bool(van["delta_accuracy_pts_ci95"][2] >= 0) if van else None),
        "judge_passes_per_block": {"phase4": ref["judge_passes_per_block"], "guard": new["judge_passes_per_block"],
                                   "reduction_pct": p4["judge_passes_reduction_pct"]},
        "drafter_fwd_per_block": {"phase4": ref["drafter_fwd_per_block"], "guard": new["drafter_fwd_per_block"]},
        "seconds_per_answer": {"phase4": ref["seconds_per_answer"], "guard": new["seconds_per_answer"],
                               "ratio_phase4_over_guard": (ref["seconds_per_answer"] / new["seconds_per_answer"]
                                                           if new["seconds_per_answer"] else None),
                               "note": "Phase 4 was timed on vast container 55572817 (8x RTX PRO 6000 Server Edition, 2x AMD EPYC "
                                       "9555, 755 GB). The ratio is a same-box speed only if this box matches that; otherwise "
                                       "it is cross-machine. Judge passes and drafter forwards per block are machine-independent."},
        "tokens_per_answer": {"phase4": ref["tokens"] / ref["n"], "guard": new["tokens"] / new["n"]},
        "redraft_repeat_rate": {"phase4": ref.get("redraft_repeat_rate"), "guard": new.get("redraft_repeat_rate")},
        "guard": guard_counts(rep["new"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"[verdict] {a.arm} IFEval: {status}; loops {ref['loops']} -> {loops}; dAcc vs Phase 4 {d_p4[0]:+.1f} "
          f"[{d_p4[1]:+.1f}, {d_p4[2]:+.1f}]; passes/block {ref['judge_passes_per_block']:.2f} -> "
          f"{new['judge_passes_per_block']:.2f}; s/answer {ref['seconds_per_answer']:.1f} -> {new['seconds_per_answer']:.1f} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
