#!/usr/bin/env python
"""Summarise an evaluation queue (eval_ckpt.sh): one row per cell directory under logs/c1b/eval/<tag>/.

    /venv/train/bin/python code/corpus1b/eval_summary.py --dir logs/c1b/eval/checkpointA

Metrics exactly as AGENTS.md section 8 defines them (block-/token-weighted sums over items):
    acc %       mean of the records' `correct` (scored inline by run_da_decode --score); None for chat
    passes/blk  sum(n_forward - n_prefill - n_flush) / sum(n_blocks)
    exp/token   sum(unique_expert_layer_loads) / sum(n_gen_to_eos)
    ms/token    1000 * sum(seconds) / sum(n_gen_to_eos)
    loop %      answers where some 32-token stretch occurs >= 3 times (common.has_loop)
    no-stop %   answers with no EOS inside the generation budget
Released-model references for the paper's sets are round 3's stock cells (RESULTS.md section 1).
Writes <dir>/summary.json and <dir>/summary.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402

# RESULTS.md section 1: the released model (stock, block 32, threshold 0.95), round 3
RELEASED = {"gsm8k500": {"acc": 90.2, "ppb": 10.06, "exp_tok": 1164.2},
            "math500": {"acc": 72.4, "ppb": 8.52, "exp_tok": 853.5},
            "mbpp": {"acc": 65.4, "ppb": 11.93, "exp_tok": 1714.0},
            "humaneval": {"acc": 78.0, "ppb": 8.26, "exp_tok": 982.3},
            "gpqa": {"acc": 37.4, "ppb": 16.07, "exp_tok": 1381.1},
            "aime": {"acc": 6.7, "ppb": 10.77, "exp_tok": 997.7}}


def cell_rows(cell_dir: str):
    seen = {}
    for path in glob.glob(os.path.join(cell_dir, "**", "*.jsonl"), recursive=True):
        for r in C.read_jsonl(path):
            if "idx" in r:
                seen[int(r["idx"])] = r
    return list(seen.values())


def summarise(rows):
    if not rows:
        return None
    tok = sum(int(r.get("n_gen_to_eos") or 0) for r in rows)
    blocks = sum(int(r.get("n_blocks") or 0) for r in rows)
    passes = sum(int(r.get("n_forward", 0)) - int(r.get("n_prefill", 0)) - int(r.get("n_flush", 0)) for r in rows)
    corr = [r.get("correct") for r in rows if r.get("correct") is not None]
    loops = sum(C.has_loop(r.get("gen_ids") or []) for r in rows)
    nostop = sum(1 for r in rows if not r.get("stopped_eos", True))
    return {"n": len(rows), "acc": 100.0 * sum(corr) / len(corr) if corr else None,
            "ppb": passes / max(1, blocks),
            "exp_tok": sum(int(r.get("unique_expert_layer_loads") or 0) for r in rows) / max(1, tok),
            "ms_tok": 1000.0 * sum(float(r.get("seconds") or 0) for r in rows) / max(1, tok),
            "loop_pct": 100.0 * loops / len(rows), "nostop_pct": 100.0 * nostop / len(rows)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    root = a.dir if os.path.isabs(a.dir) else os.path.join(C.ROOT, a.dir)
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) == "logs":
            continue
        s = summarise(cell_rows(d))
        if s:
            name = os.path.basename(d)
            s["set"] = name.split("_")[0]
            ref = RELEASED.get(s["set"])
            if ref and s["acc"] is not None:
                s["released_acc"] = ref["acc"]
                s["d_acc_vs_released"] = s["acc"] - ref["acc"]
                s["speedup_ppb_vs_released"] = ref["ppb"] / max(1e-9, s["ppb"])
            out[name] = s
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    lines = ["| cell | n | acc % | vs released | passes/blk | x fewer passes | exp/token | ms/token | loop % | no-stop % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"
    for name, s in out.items():
        lines.append(f"| {name} | {s['n']} | {f(s['acc'])} | {f(s.get('d_acc_vs_released'))} | {f(s['ppb'], 2)} | "
                     f"{f(s.get('speedup_ppb_vs_released'), 2)} | {f(s['exp_tok'])} | {f(s['ms_tok'])} | "
                     f"{f(s['loop_pct'])} | {f(s['nostop_pct'])} |")
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


if __name__ == "__main__":
    main()
