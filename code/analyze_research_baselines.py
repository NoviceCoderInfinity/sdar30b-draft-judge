#!/usr/bin/env python
"""Score and summarise run_research_baselines.py cells.

Per cell: n, accuracy (vendored scorer), item-paired 95% bootstrap CI against a
reference cell, wall seconds/item, ms per forward by kind, forwards per
generated token, expert-layer executions per generated token, tokens per
denoising pass, passes per block, EOS/truncation rates, output length.

Usage: /venv/main/bin/python code/analyze_research_baselines.py logs/research_v1/baselines [more dirs...] \
           --ref gsm8k_B32_t95_stock --out results/metrics/baselines_summary.json
Cells from directories without a model tag are the b32 checkpoint; use
--tag DIR=LABEL to label directories (e.g. logs/research_v1/baselines_b32=b32).
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

from analyze_s1b import per_item_correct, boot_ci   # noqa: E402

N_LAYERS = 48


def load_dir(d):
    cells = defaultdict(dict); done = defaultdict(lambda: True)
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        base = os.path.basename(f)[:-6]
        cell = ".".join(base.split(".")[:2])              # <cell name>.<split>
        done[cell] = done[cell] and os.path.exists(f + ".done")
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            assert r["idx"] not in cells[cell], f"duplicate idx {r['idx']} in {cell}"
            cells[cell][r["idx"]] = r
    return cells, done


def summarise(rows, ds):
    rows = sorted(rows, key=lambda r: r["idx"])
    corr = {r["idx"]: r["correct"] for r in rows if r.get("correct") is not None}
    missing = [r for r in rows if r.get("correct") is None]
    if missing:
        corr.update(per_item_correct(ds, missing))
    n = len(rows)
    tok = np.array([r["n_gen_to_eos"] for r in rows], float)
    fwd = np.array([r["n_forward"] for r in rows], float)
    den = np.array([r["n_denoise"] for r in rows], float)
    secs = np.array([r["seconds"] for r in rows], float)
    uniq = np.array([r["unique_expert_layer_loads"] for r in rows], float)
    lf = np.array([r["layer_forwards"] for r in rows], float)
    ms = {k: sum(r["ms_by_kind"][k] for r in rows) for k in ("prefill", "denoise", "kvflush")}
    nk = {k: sum(r["n_" + {"prefill": "prefill", "denoise": "denoise", "kvflush": "flush"}[k]] for r in rows) for k in ms}
    cps = [c for r in rows for c in r["commits_per_pass"]]
    ppb = [p for r in rows for p in r["passes_per_block"]]
    uniq_den = np.array([r["uniq_by_kind"]["denoise"] for r in rows], float)
    return {"n": n, "accuracy": 100 * np.mean([corr[r["idx"]] for r in rows]), "correct": corr,
            "s_per_item": secs.mean(), "ms_per_forward": 1000 * secs.sum() / fwd.sum(),
            "ms_per_forward_by_kind": {k: (ms[k] / nk[k] if nk[k] else None) for k in ms},
            "forwards_per_item": fwd.mean(), "denoise_per_item": den.mean(),
            "forwards_per_token": fwd.sum() / tok.sum(), "denoise_per_token": den.sum() / tok.sum(),
            "ms_per_token": 1000 * secs.sum() / tok.sum(),
            "expert_layer_exec_per_token": uniq.sum() / tok.sum(),
            "expert_layer_exec_per_token_denoise_only": uniq_den.sum() / tok.sum(),
            "uniq_experts_per_layer_forward": uniq.sum() / lf.sum(),
            "uniq_experts_per_layer_forward_denoise": uniq_den.sum() / (den.sum() * N_LAYERS),
            "tokens_per_pass": float(np.mean(cps)), "passes_per_block": float(np.mean(ppb)),
            "gen_tokens_mean": tok.mean(), "eos_rate": float(np.mean([r["stopped_eos"] for r in rows])),
            "truncated_rate": float(np.mean([r["truncated"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="gsm8k_B32_t95_stock")
    ap.add_argument("--tag", action="append", default=[], help="DIR=LABEL")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "metrics", "research_v1", "baselines_summary.json"))
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    tags = dict(t.split("=", 1) for t in args.tag)
    stats, done_map = {}, {}
    for d in args.dirs:
        cells, done = load_dir(d)
        label = tags.get(d, tags.get(os.path.abspath(d), "b32"))
        for cell, rows in cells.items():
            if not done[cell] and not args.partial:
                continue
            ds = "mbpp" if cell.startswith("mbpp") else "gsm8k"
            if cell.split(".")[1] != "mbpp" and "_B" not in cell.split(".")[0]:
                continue
            key = f"{label}:{cell}"
            stats[key] = summarise(list(rows.values()), ds); stats[key]["complete"] = done[cell]; stats[key]["dir"] = d
            done_map[key] = done[cell]
    refs = [k for k in stats if k.split(":")[1].split(".")[0] == args.ref]
    comps = {}
    for k, s in stats.items():
        for rk in refs:
            if rk == k or k.split(".")[-1] != rk.split(".")[-1]:      # pair only within the same split
                continue
            a, b = s["correct"], stats[rk]["correct"]
            common = sorted(set(a) & set(b))
            if not common:
                continue
            ci = boot_ci([a[i] - b[i] for i in common])
            comps[f"{k} - {rk}"] = {"n": len(common), "delta": ci[2], "ci95": [ci[0], ci[1]],
                                    "speed_ratio_ms_per_token": stats[rk]["ms_per_token"] / s["ms_per_token"],
                                    "expert_work_ratio_per_token": s["expert_layer_exec_per_token"] / stats[rk]["expert_layer_exec_per_token"]}
    out = {"cells": {k: {a: b for a, b in v.items() if a != "correct"} for k, v in stats.items()}, "paired_vs_ref": comps}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    hdr = f"{'cell':34}{'n':>4}{'acc':>7}{'s/it':>7}{'ms/fwd':>8}{'fwd/tok':>9}{'ms/tok':>8}{'exp/tok':>9}{'uniq/lf':>9}{'tok/pass':>9}{'pass/blk':>9}{'gen':>6}{'eos':>6}{'trunc':>7}"
    print(hdr); print("-" * len(hdr))
    for k in sorted(stats):
        s = stats[k]
        print(f"{k + ('' if s['complete'] else '*'):34}{s['n']:>4}{s['accuracy']:>7.1f}{s['s_per_item']:>7.1f}{s['ms_per_forward']:>8.0f}{s['forwards_per_token']:>9.3f}"
              f"{s['ms_per_token']:>8.0f}{s['expert_layer_exec_per_token']:>9.0f}{s['uniq_experts_per_layer_forward']:>9.1f}{s['tokens_per_pass']:>9.2f}"
              f"{s['passes_per_block']:>9.2f}{s['gen_tokens_mean']:>6.0f}{s['eos_rate']:>6.2f}{s['truncated_rate']:>7.2f}")
    print("(* partial)")
    for k, c in comps.items():
        print(f"{k}: n={c['n']} delta {c['delta']:+.1f} CI [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}] | speed x{c['speed_ratio_ms_per_token']:.2f} | expert work x{c['expert_work_ratio_per_token']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
