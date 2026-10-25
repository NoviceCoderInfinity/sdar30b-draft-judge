#!/usr/bin/env python
"""Canary summary (8-GPU Phase 3 re-run). Every canary decodes HumanEval item 0 with the setup judge check's settings
(SDAR-1.7B + 128M judge, no --judge-diag, exactly the Phase 1 record's configuration) on all 8 GPUs at once, after
setup and after every run. MEASURED: the record's `seconds` (synchronize + perf_counter around the whole decode).

    /venv/p2/bin/python code/phase3_8gpu/canary_check.py logs/phase3_8gpu/canary

Prints every canary: mean seconds over the 8 GPUs, vs the first canary (machine drift during this run) and vs the
Phase 1 record of the same item (this machine vs the Phase 1 machine on identical work); writes summary.json.
Exit 3 when the latest canary is more than 3% slower than the first (the project lead's STOP rule; > 2% is flagged).
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "logs/phase2/ref/S0_128M_nopool_by_set/humaneval")


def recs(path):
    out = []
    for f in sorted(glob.glob(path)):
        if ".running." in f:
            continue
        for line in open(f):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def main(d):
    ref = [r for r in recs(os.path.join(REF, "*.jsonl")) if int(r["idx"]) == 0]
    ref = ref[0] if ref else None
    rows = []
    for c in sorted(glob.glob(os.path.join(d, "[0-9][0-9]_*"))):
        per = {}
        for g in sorted(glob.glob(os.path.join(c, "gpu*"))):
            rr = [r for r in recs(os.path.join(g, "*.jsonl")) if int(r["idx"]) == 0]
            if rr:
                per[os.path.basename(g)] = rr[-1]
        if not per:
            continue
        secs = [r["seconds"] for r in per.values()]
        pre = [r["ms_by_kind"]["prefill"] for r in per.values()]
        row = {"canary": os.path.basename(c), "gpus": len(per), "utc_last": max(r["utc"] for r in per.values()),
               "seconds_by_gpu": {k: r["seconds"] for k, r in per.items()}, "mean_s": sum(secs) / len(secs),
               "min_s": min(secs), "max_s": max(secs), "mean_prefill_ms": sum(pre) / len(pre),
               "tokens": sorted({r["n_gen_to_eos"] for r in per.values()}),
               "identical_to_phase1": sum(bool((r.get("ref_match") or {}).get("gen_ids")) for r in per.values())}
        rows.append(row)
    if not rows:
        print("[canary] no canary records yet")
        return 0
    first = rows[0]
    for r in rows:
        r["vs_first"] = r["mean_s"] / first["mean_s"]
        if ref:
            r["vs_phase1_record"] = r["mean_s"] / ref["seconds"]
            r["prefill_vs_phase1_record"] = r["mean_prefill_ms"] / ref["ms_by_kind"]["prefill"]
    out = {"item": "humaneval idx 0", "phase1_record_seconds": ref and ref["seconds"],
           "phase1_record_prefill_ms": ref and ref["ms_by_kind"]["prefill"], "canaries": rows}
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    last = rows[-1]
    msg = (f"[canary] {last['canary']}: mean {last['mean_s']:.3f}s over {last['gpus']} GPUs (min {last['min_s']:.3f}, "
           f"max {last['max_s']:.3f}); vs first canary {last['vs_first']:.4f}")
    if ref:
        msg += (f"; vs Phase 1 record {last['vs_phase1_record']:.4f} (prefill {last['prefill_vs_phase1_record']:.4f}); "
                f"identical to Phase 1 {last['identical_to_phase1']}/{last['gpus']}")
    print(msg, flush=True)
    if ref and len(rows) == 1 and not (0.98 <= last["vs_phase1_record"] <= 1.02):
        print(f"[canary] MACHINE FLAG: this machine / Phase 1 machine on identical work = {last['vs_phase1_record']:.4f} "
              f"(outside 0.98-1.02)", flush=True)
        open(os.path.join(d, "MACHINE_FLAG"), "w").write(msg + "\n")
    if last["vs_first"] > 1.02:
        print(f"[canary] FLAG: {100 * (last['vs_first'] - 1):.2f}% slower than the first canary", flush=True)
    return 3 if last["vs_first"] > 1.03 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
