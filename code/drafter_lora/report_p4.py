#!/usr/bin/env python
"""Phase 4: every MEASURED per-item quantity of the rd1 / rd2 decodes, summed from the raw records (nothing estimated).

    /venv/p2/bin/python code/drafter_lora/report_p4.py --arm rd1 --new logs/phase4/decode/rd1 \
        --phase3 logs/phase4/ref/phase3/logs/phase3_8gpu/decode/rd1 \
        --vanilla 'logs/phase4/ref/phase2/assets/ref_stock_30B/*_B32_t95_stock.{set}.shard*.jsonl' \
        --out results/phase4_drafter/measured_rd1.json

Reports, per benchmark and pooled, for the new decode and each reference ON THE SAME ITEMS (paired by idx):
  accuracy            sum(correct) / n; paired delta, 95% CI from 2,000 bootstrap resamples, numpy default_rng(0)
  time per answer     sum(seconds) / n; seconds = torch.cuda.synchronize + perf_counter around the whole decode
  tok/s, ms/token     sum(n_gen_to_eos) / sum(seconds); 1000 sum(seconds) / sum(n_gen_to_eos)
  GPU time            sum(gpu_active_s): NVML GPM graphics-active share x the item's interval (hardware counters)
  CPU time            sum(cpu_thread_s) (decoding thread) and sum(cpu_process_s) (whole process), OS CPU clocks;
                      both include CUDA's spin-waiting at synchronisations
  MoE time            sum(moe_seconds): every 30B forward timed with cuda.synchronize + perf_counter; by kind from
                      ms_by_kind (prefill, judge, kvflush, ...)
  drafter time        sum(drafter_seconds): every drafter forward timed the same way; by kind (prefill, draft,
                      redraft, kvflush) from drafter_ms_by_kind
  experts activated   sum(unique_expert_layer_loads) = distinct (layer, expert) pairs dispatched, counted from the
                      router's selected_experts per forward; per token, per 30B forward, per layer-forward
  judge passes / block = sum(n_forward - n_prefill - n_flush) / sum(n_blocks)   (AGENTS.md section 8)
  drafter forwards / block (draft + redraft), loops (compare_gates.loops), length-cap hits, re-draft repeat rate
Definitions follow code/phase3_8gpu/report_8gpu.py and AGENTS.md section 8 verbatim.

Ratios of TIMES between records from different machines are labelled "cross_machine": the Phase 3 records come from
the Phase 3 box and the vanilla records mostly from a Workstation-Edition box, both slower than this one on identical
work.  Counts (accuracy, passes, experts) do not depend on the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402

from drafter_lora.compare_gates import SETS, load_dir, loops, vanilla_glob   # noqa: E402


def _sum(R, ids, key) -> Optional[float]:
    v = [R[i].get(key) for i in ids]
    return float(sum(v)) if all(x is not None for x in v) else None     # None unless EVERY item has it


def _kind(R, ids, key) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in ids:
        for k, ms in (R[i].get(key) or {}).items():
            out[k] = out.get(k, 0.0) + float(ms)
    return {k: round(v / 1e3, 3) for k, v in sorted(out.items())}       # seconds


def measure(R: Dict[int, dict], ids: List[int]) -> Optional[dict]:
    if not ids:
        return None
    n = len(ids)
    tok = sum(R[i]["n_gen_to_eos"] for i in ids)
    sec = sum(R[i]["seconds"] for i in ids)
    blocks = sum(R[i]["n_blocks"] for i in ids)
    nfw = sum(R[i]["n_forward"] for i in ids)
    loads = sum(R[i]["unique_expert_layer_loads"] for i in ids)
    lf = sum(R[i]["layer_forwards"] for i in ids)
    gpu, cth, cpr = _sum(R, ids, "gpu_active_s"), _sum(R, ids, "cpu_thread_s"), _sum(R, ids, "cpu_process_s")
    moe, dr = _sum(R, ids, "moe_seconds"), _sum(R, ids, "drafter_seconds")
    slots = sum(sum(R[i].get("redraft_slots_per_block") or [0]) for i in ids)
    old = sum(sum(R[i].get("redraft_eq_old_per_block") or [0]) for i in ids)
    return {
        "n": n, "correct": int(sum(bool(R[i]["correct"]) for i in ids)),
        "accuracy_pct": 100 * float(np.mean([bool(R[i]["correct"]) for i in ids])),
        "tokens": tok, "blocks": blocks,
        "seconds_total": sec, "seconds_per_answer": sec / n, "tok_s": tok / sec if sec else None,
        "ms_per_token": 1000 * sec / tok if tok else None,
        "gpu_active_s": gpu, "gpu_share_pct": (100 * gpu / sec) if gpu is not None and sec else None,
        "gpu_active_s_per_answer": (gpu / n) if gpu is not None else None,
        "cpu_thread_s": cth, "cpu_process_s": cpr,
        "cpu_thread_s_per_answer": (cth / n) if cth is not None else None,
        "moe_s": moe, "moe_s_per_answer": (moe / n) if moe is not None else None,
        "moe_s_by_kind": _kind(R, ids, "ms_by_kind"),
        "drafter_s": dr, "drafter_s_per_answer": (dr / n) if dr is not None else None,
        "drafter_s_by_kind": _kind(R, ids, "drafter_ms_by_kind"),
        "outside_forwards_s": (sec - moe - dr) if (moe is not None and dr is not None) else None,
        "judge_passes_per_block": sum(R[i]["n_forward"] - R[i]["n_prefill"] - R[i]["n_flush"] for i in ids) / blocks,
        "n_forward_30b": nfw, "n_judge": sum(R[i].get("n_judge", 0) for i in ids),
        "drafter_forwards_by_kind": {k: int(sum((R[i].get("drafter_n_by_kind") or {}).get(k, 0) for i in ids))
                                     for k in sorted({k for i in ids for k in (R[i].get("drafter_n_by_kind") or {})})},
        "drafter_fwd_per_block": sum(sum((R[i].get("drafter_n_by_kind") or {}).get(k, 0) for k in ("draft", "redraft"))
                                     for i in ids) / blocks,
        "experts_activated_total": loads, "experts_per_token": loads / tok if tok else None,
        "experts_per_30b_forward": loads / nfw if nfw else None,
        "experts_per_layer_forward": loads / lf if lf else None,
        "experts_by_kind": {k: int(sum((R[i].get("uniq_by_kind") or {}).get(k, 0) for i in ids))
                            for k in sorted({k for i in ids for k in (R[i].get("uniq_by_kind") or {})})},
        "loops": int(sum(loops(R[i]) for i in ids)), "length_cap_hits": int(sum(bool(R[i].get("truncated")) for i in ids)),
        "redraft_repeat_rate": (old / slots) if slots else None,
        "gpu_names": sorted({str(R[i].get("gpu")) for i in ids}),
    }


def paired_delta(A, B, ids) -> List[float]:
    d = np.array([int(bool(A[i]["correct"])) - int(bool(B[i]["correct"])) for i in ids])
    rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def _ratio(a, b):
    return (a / b) if (a is not None and b) else None


def versus(new: dict, ref: dict, label: str) -> dict:
    per, pool_new, pool_ref = {}, {}, {}
    for s in new:
        if s not in ref:
            continue
        ids = sorted(set(new[s]) & set(ref[s]))
        if not ids:
            continue
        a, b = measure(new[s], ids), measure(ref[s], ids)
        per[s] = {"n_paired": len(ids), "new": a, label: b, "delta_accuracy_pts_ci95": paired_delta(new[s], ref[s], ids),
                  "judge_passes_reduction_pct": 100 * (1 - a["judge_passes_per_block"] / b["judge_passes_per_block"]),
                  "experts_per_token_ratio": _ratio(a["experts_per_token"], b["experts_per_token"]),
                  "cross_machine_time_ratio_new_over_ref": _ratio(a["seconds_total"], b["seconds_total"]),
                  "cross_machine_tok_s_ratio_new_over_ref": _ratio(a["tok_s"], b["tok_s"])}
        for i in ids:
            pool_new[(s, i)], pool_ref[(s, i)] = new[s][i], ref[s][i]
    out = {"per_set": per}
    if pool_new:
        keys = list(pool_new)
        A = {j: pool_new[k] for j, k in enumerate(keys)}
        B = {j: pool_ref[k] for j, k in enumerate(keys)}
        ix = list(range(len(keys)))
        a, b = measure(A, ix), measure(B, ix)
        out["pooled"] = {"n_paired": len(ix), "new": a, label: b, "delta_accuracy_pts_ci95": paired_delta(A, B, ix),
                         "judge_passes_reduction_pct": 100 * (1 - a["judge_passes_per_block"] / b["judge_passes_per_block"]),
                         "experts_per_token_ratio": _ratio(a["experts_per_token"], b["experts_per_token"]),
                         "cross_machine_time_ratio_new_over_ref": _ratio(a["seconds_total"], b["seconds_total"]),
                         "cross_machine_tok_s_ratio_new_over_ref": _ratio(a["tok_s"], b["tok_s"])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--new", required=True, help="dir with one sub-dir per benchmark (this box)")
    ap.add_argument("--phase3", default=None, help="Phase 3 records of the same arm, one sub-dir per benchmark")
    ap.add_argument("--vanilla", default=None, help="glob template with {set} (flat vanilla bundle)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sets = [s for s in a.sets.split(",") if s]
    new = {s: r for s, r in ((s, load_dir(os.path.join(a.new, s))) for s in sets) if r}
    rep = {"arm": a.arm, "new": os.path.abspath(a.new), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_new": {s: len(r) for s, r in new.items()},
           "new_alone": {"per_set": {s: measure(r, sorted(r)) for s, r in new.items()}}}
    allk = {(s, i): new[s][i] for s in new for i in new[s]}
    if allk:
        keys = list(allk)
        rep["new_alone"]["pooled"] = measure({j: allk[k] for j, k in enumerate(keys)}, list(range(len(keys))))
    if a.phase3:
        p3 = {s: load_dir(os.path.join(a.phase3, s)) for s in sets}
        rep["vs_phase3"] = versus(new, p3, "phase3")
    if a.vanilla:
        van = {s: load_dir(vanilla_glob(a.vanilla, s)) for s in sets}
        rep["vs_vanilla"] = versus(new, van, "vanilla")
    out = P.safe_out(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    p = rep["new_alone"].get("pooled") or {}
    print(f"[report_p4] {a.arm}: {p.get('n')} items, accuracy {p.get('accuracy_pct')}, judge passes/blk "
          f"{p.get('judge_passes_per_block')}, s/answer {p.get('seconds_per_answer')}, GPU s {p.get('gpu_active_s')}, "
          f"CPU thread s {p.get('cpu_thread_s')}, MoE s {p.get('moe_s')}, drafter s {p.get('drafter_s')}, experts/token "
          f"{p.get('experts_per_token')} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
