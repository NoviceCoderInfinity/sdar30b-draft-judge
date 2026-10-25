#!/usr/bin/env python
"""The smoke test's verdict (runbook section 4), read from the smoke runs' outputs.  Exit 0 = all pass, 3 = stop & ask.

    python code/drafter_lora/smoke_check.py --train logs/phase4/smoke/train --parity logs/phase4/smoke/parity.json \
        --speed logs/phase4/smoke/speedprobe/speed.json --out logs/phase4/smoke/smoke.json

  1. trainable parameters == 34,865,152 (model_info.json, written before the assert in the trainer)
  2. the KL term is exactly 0 at the first step (lora_B starts at zero)
  3. the loss falls over the run (mean CE of the last --window logged steps below the first --window), no NaN / inf
  4. the untrained drafter's HF replay matches the stored SGLang drafts on >= 95% of slots (eval_agree --parity-only)
  5. the measured speed projects the full pass at <= --max-hours (speed.json of a short full-configuration run)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

EXPECTED = 34_865_152


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--parity", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    checks = {}
    mi = json.load(open(os.path.join(a.train, "model_info.json")))
    checks["trainable_count"] = {"value": mi["trainable"]["total"], "want": EXPECTED,
                                 "pass": mi["trainable"]["total"] == EXPECTED}
    rows = [json.loads(l) for l in open(os.path.join(a.train, "train.rank0.jsonl")) if l.strip()]
    steps = [r for r in rows if "ce" in r]
    first = steps[0] if steps else {}
    checks["kl_zero_at_step1"] = {"value": first.get("kl"), "pass": first.get("kl") == 0.0 and first.get("step") == 1}
    finite = all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in steps)
    w = min(a.window, max(1, len(steps) // 2))
    ce_a = sum(r["ce"] for r in steps[:w]) / w if steps else float("nan")
    ce_b = sum(r["ce"] for r in steps[-w:]) / w if steps else float("nan")
    checks["loss_falls"] = {"steps": len(steps), "ce_first": ce_a, "ce_last": ce_b, "finite": finite,
                            "pass": bool(finite and steps and ce_b < ce_a)}
    par = json.load(open(a.parity))
    pv = par["verdict"]
    checks["replay_parity"] = {"value": pv["parity"], "want": pv["min"], "pass": bool(pv["pass"]),
                               "slots": par["overall"].get("slots")}
    sp = json.load(open(a.speed))
    checks["speed_projection"] = {"projected_hours": sp["projected_hours"], "sec_per_step": sp["sec_per_step"],
                                  "pass_steps": sp["pass_steps"], "max_hours": a.max_hours,
                                  "pass": sp["projected_hours"] <= a.max_hours, "note": "measured on this box"}
    ok = all(c["pass"] for c in checks.values())
    rep = {"pass": ok, "checks": checks}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    for k, c in checks.items():
        print(f"[smoke] {'PASS' if c['pass'] else 'FAIL'}  {k}: " +
              ", ".join(f"{kk}={vv}" for kk, vv in c.items() if kk != "pass"), flush=True)
    print(f"[smoke] {'ALL PASS' if ok else 'STOP AND ASK'} -> {a.out}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
