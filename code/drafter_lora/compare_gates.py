#!/usr/bin/env python
"""Gates 3 and 4: the trained drafter's rd1 / rd2 decodes against Phase 3 (and vanilla), paired on the same items.

    # Gate 3: rd1 on items[k::32], k = 0..7, of every benchmark (a uniform quarter)
    python code/drafter_lora/compare_gates.py --gate 3 --arm rd1 \
        --new logs/phase4/decode/rd1_gate3 --phase3 ref/phase3/logs/phase3_8gpu/decode/rd1 \
        --out results/phase4_drafter/gate3.json
    # Gate 4: rd1 or rd2 on all items, plus vanilla
    python code/drafter_lora/compare_gates.py --gate 4 --arm rd2 \
        --new logs/phase4/decode/rd2 --phase3 ref/phase3/logs/phase3_8gpu/decode/rd2 \
        --vanilla 'ref/phase2/assets/ref_stock_30B/{set}' --out results/phase4_drafter/gate4_rd2.json

Records: <dir>/<set>/**.jsonl (run_da_decode.py records, one JSON per item; `.running.` files skipped), matched by
`idx`.  Definitions, as code/phase3_8gpu/report_8gpu.py and redraft_why.py:
  judge passes / block = sum(n_forward - n_prefill - n_flush) / sum(n_blocks)
  drafter forwards / block = sum(draft + redraft drafter forwards) / sum(n_blocks)  (+ _all: prefill, flush too)
  time per answer = sum(seconds) over the paired items; tok/s = sum(n_gen_to_eos) / sum(seconds)
  delta accuracy = paired new - reference, 95% CI from 2,000 bootstrap resamples, numpy default_rng(0)
  looping output = more than half of the output's tokens inside a 12-token sequence that already appeared earlier
  re-draft repeat rate = sum(redraft_eq_old) / sum(redraft_slots)
Gate 3 (runbook section 6): >= 5% fewer judge passes per block (fail below 2%), the pooled paired accuracy CI covers 0
or better, IFEval looping outputs no higher than Phase 3 on the same items.
Gate 4 success criteria (section 7) are reported against Phase 3 and, when --vanilla is given, against vanilla.
"""
from __future__ import annotations

import argparse
import glob
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

SETS = ("ifeval541", "math500", "gpqa", "aime", "humaneval", "mbpp", "gsm8k500")
NGRAM = 12


FAMILY = {"ifeval541": "ifeval", "gsm8k500": "gsm8k"}      # cell-name family of a set (run_da_decode.cell_name)


def vanilla_glob(template: str, s: str) -> str:
    """The vanilla template for set `s`, restricted to that set's own cell family.  The flat vanilla bundle also holds
    gsm8k_B32_t95_stock.aime.* (GSM8K settings on the AIME split), which a bare '*_B32...' glob would mix into aime."""
    t = template.format(set=s)
    base = os.path.basename(t)
    if base.startswith("*_"):
        t = os.path.join(os.path.dirname(t), FAMILY.get(s, s) + base[1:])
    return t


def load_dir(d: str) -> Dict[int, dict]:
    """Records under a directory (recursive), or of the files a glob pattern names (when `d` contains '*')."""
    out: Dict[int, dict] = {}
    files = glob.glob(d) if "*" in d else glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)
    for f in sorted(files):
        if ".running." in os.path.basename(f):
            continue
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[int(r["idx"])] = r
    return out


def loops(r: dict) -> bool:
    ids = r["gen_ids"][:r["n_gen_to_eos"]]
    seen, rp = set(), 0
    for k in range(len(ids) - NGRAM + 1):
        t = tuple(ids[k:k + NGRAM])
        if t in seen:
            rp += 1
        else:
            seen.add(t)
    return len(ids) > 2 * NGRAM and rp / (len(ids) - NGRAM + 1) > 0.5


def agg(R: Dict[int, dict], ids: List[int]) -> Optional[dict]:
    if not ids:
        return None
    blocks = sum(R[i]["n_blocks"] for i in ids)
    tok = sum(R[i]["n_gen_to_eos"] for i in ids)
    sec = sum(R[i]["seconds"] for i in ids)
    slots = sum(sum(R[i].get("redraft_slots_per_block") or [0]) for i in ids)
    old = sum(sum(R[i].get("redraft_eq_old_per_block") or [0]) for i in ids)
    return {"n": len(ids), "correct": int(sum(bool(R[i]["correct"]) for i in ids)),
            "acc": 100 * float(np.mean([bool(R[i]["correct"]) for i in ids])),
            "passes_blk": sum(R[i]["n_forward"] - R[i]["n_prefill"] - R[i]["n_flush"] for i in ids) / blocks,
            "drafter_fwd_blk": sum(sum((R[i].get("drafter_n_by_kind") or {}).get(k, 0) for k in ("draft", "redraft"))
                                   for i in ids) / blocks,                    # the runbook's column (25.8 / 17.2)
            "drafter_fwd_blk_all": sum(sum((R[i].get("drafter_n_by_kind") or {}).values()) for i in ids) / blocks,
            "seconds": sec, "sec_per_item": sec / len(ids), "tokens": tok, "tok_s": tok / sec if sec else None,
            "loops": int(sum(loops(R[i]) for i in ids)), "capped": int(sum(bool(R[i].get("truncated")) for i in ids)),
            "redraft_repeat": (old / slots) if slots else None, "blocks": blocks}


def paired_delta(A: Dict[int, dict], B: Dict[int, dict], ids: List[int]) -> List[float]:
    d = np.array([int(bool(A[i]["correct"])) - int(bool(B[i]["correct"])) for i in ids])
    rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def compare(new: Dict[str, Dict[int, dict]], ref: Dict[str, Dict[int, dict]], label: str) -> dict:
    per, pooled_new, pooled_ref, keys = {}, {}, {}, []
    for s in new:
        if s not in ref:
            continue
        ids = sorted(set(new[s]) & set(ref[s]))
        if not ids:
            continue
        a, b = agg(new[s], ids), agg(ref[s], ids)
        per[s] = {"n_paired": len(ids), "new": a, label: b, "delta_acc": paired_delta(new[s], ref[s], ids),
                  "passes_reduction_pct": 100 * (1 - a["passes_blk"] / b["passes_blk"]),
                  "time_ratio": a["seconds"] / b["seconds"] if b["seconds"] else None}
        for i in ids:
            k = (s, i)
            pooled_new[k], pooled_ref[k] = new[s][i], ref[s][i]
            keys.append(k)
    out = {"per_set": per}
    if keys:
        ix = list(range(len(keys)))
        A = {j: pooled_new[k] for j, k in enumerate(keys)}
        B = {j: pooled_ref[k] for j, k in enumerate(keys)}
        a, b = agg(A, ix), agg(B, ix)
        out["pooled"] = {"n_paired": len(keys), "new": a, label: b, "delta_acc": paired_delta(A, B, ix),
                         "passes_reduction_pct": 100 * (1 - a["passes_blk"] / b["passes_blk"]),
                         "time_ratio": a["seconds"] / b["seconds"] if b["seconds"] else None}
    return out


def verdict_gate3(c: dict, min_red: float, fail_red: float) -> dict:
    p = c["pooled"]
    red, lo = p["passes_reduction_pct"], p["delta_acc"][1]
    hi = p["delta_acc"][2]
    ife = c["per_set"].get("ifeval541")
    loops_ok = None if ife is None else ife["new"]["loops"] <= ife["phase3"]["loops"]
    status = "PASS"
    if red < fail_red or hi < 0 or loops_ok is False:
        status = "FAIL"
    elif red < min_red:
        status = "ASK"
    return {"gate": 3, "passes_reduction_pct": red, "min_pct": min_red, "fail_below_pct": fail_red,
            "pooled_delta_acc": p["delta_acc"], "ci_covers_0_or_better": bool(hi >= 0),
            "ifeval_loops_new": None if ife is None else ife["new"]["loops"],
            "ifeval_loops_phase3": None if ife is None else ife["phase3"]["loops"], "ifeval_loops_ok": loops_ok,
            "status": status, "ci_lower": lo}


def success_gate4(c3: dict, cv: Optional[dict]) -> dict:
    p = c3["pooled"]
    ife3 = c3["per_set"].get("ifeval541")
    out = {"ifeval_loops_new": None if ife3 is None else ife3["new"]["loops"], "ifeval_loops_max": 60,
           "faster_per_answer_than_phase3": bool(p["new"]["seconds"] < p["phase3"]["seconds"]),
           "tok_s_not_lower": bool((p["new"]["tok_s"] or 0) >= (p["phase3"]["tok_s"] or 0)),
           "passes_reduction_pct": p["passes_reduction_pct"],
           "passes_reduction_ok": bool(p["passes_reduction_pct"] >= 5.0)}
    out["ifeval_loops_ok"] = None if ife3 is None else bool(ife3["new"]["loops"] <= 60)
    if cv is not None and cv.get("pooled"):
        ifev = cv["per_set"].get("ifeval541")
        out["ifeval_ci_vs_vanilla"] = None if ifev is None else ifev["delta_acc"]
        out["ifeval_not_significantly_below_vanilla"] = None if ifev is None else bool(ifev["delta_acc"][2] >= 0)
        out["pooled_ci_vs_vanilla"] = cv["pooled"]["delta_acc"]
        out["pooled_ci_covers_0_or_better"] = bool(cv["pooled"]["delta_acc"][2] >= 0)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", type=int, choices=(3, 4), required=True)
    ap.add_argument("--arm", choices=("rd1", "rd2"), default="rd1")
    ap.add_argument("--new", required=True, help="dir with one sub-dir per benchmark")
    ap.add_argument("--phase3", required=True, help="Phase 3 records of the same arm, one sub-dir per benchmark")
    ap.add_argument("--vanilla", default=None,
                    help="vanilla 30B records: a dir or glob template with {set}; the HF bundle is FLAT and mixes in "
                         "pooled cells, so use '<ref>/assets/ref_stock_30B/*_B32_t95_stock.{set}.shard*.jsonl'")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--min-reduction-pct", type=float, default=5.0)
    ap.add_argument("--fail-reduction-pct", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sets = [s for s in args.sets.split(",") if s]
    new = {s: load_dir(os.path.join(args.new, s)) for s in sets}
    p3 = {s: load_dir(os.path.join(args.phase3, s)) for s in sets}
    new = {s: r for s, r in new.items() if r}
    rep = {"gate": args.gate, "arm": args.arm, "new": os.path.abspath(args.new), "phase3": os.path.abspath(args.phase3),
           "n_new": {s: len(r) for s, r in new.items()}, "n_phase3": {s: len(r) for s, r in p3.items()},
           "vs_phase3": compare(new, p3, "phase3"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cv = None
    if args.vanilla:
        van = {s: load_dir(vanilla_glob(args.vanilla, s)) for s in sets}
        cv = compare(new, van, "vanilla")
        rep["vs_vanilla"] = cv
        rep["n_vanilla"] = {s: len(r) for s, r in van.items()}
    assert rep["vs_phase3"].get("pooled"), "no item is shared between --new and --phase3"
    rc = 0
    if args.gate == 3:
        rep["verdict"] = verdict_gate3(rep["vs_phase3"], args.min_reduction_pct, args.fail_reduction_pct)
        rc = 0 if rep["verdict"]["status"] == "PASS" else 3
    else:
        rep["success"] = success_gate4(rep["vs_phase3"], cv)
    out = P.safe_out(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
    print(json.dumps(rep.get("verdict") or rep.get("success"), indent=2), flush=True)
    print(f"[out] {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
