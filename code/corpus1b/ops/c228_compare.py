#!/usr/bin/env python
"""Per-item comparison of the 228M judge (base decode, no pool) with the 128M judge and with stock SDAR-30B.

    /venv/train/bin/python code/corpus1b/ops/c228_compare.py --dir logs/c1b/eval/checkpoint228M_nopool

References (downloaded once into logs/c1b/eval/refs/, public HF):
  128M judge, same decode: Arushhh/phase3-redraft  logs/phase3_8gpu/decode/base/<set>/  (token-identical to Phase 1)
  stock 30B:               Arushhh/phase2-drafter-swap  assets/ref_stock_30B/<prefix>_B32_t95_stock.<set>.shard*.jsonl
Accuracy deltas are paired over the shared items, 95% bootstrap (2,000 resamples, numpy default_rng(0)), exactly as the
Phase 3 report; passes/block = sum(n_forward - n_prefill - n_flush) / sum(n_blocks). Writes <dir>/compare.{md,json}.
"""
import argparse
import glob
import json
import os

import numpy as np

SETS = ["ifeval541", "math500", "gpqa", "aime", "humaneval", "mbpp", "gsm8k500"]
STOCK_PREFIX = {"ifeval541": "ifeval", "gsm8k500": "gsm8k"}
REFS = "logs/c1b/eval/refs"


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        if ".running." in f or f.endswith(".done"):
            continue
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[int(r["idx"])] = r
    return out


def fetch_refs():
    if os.path.exists(f"{REFS}/.ok"):
        return
    from huggingface_hub import snapshot_download
    tok = os.environ.get("HF_TOKEN")
    snapshot_download("Arushhh/phase3-redraft", repo_type="dataset", local_dir=f"{REFS}/p3", token=tok,
                      allow_patterns=[f"logs/phase3_8gpu/decode/base/{s}/*.jsonl" for s in SETS])
    snapshot_download("Arushhh/phase2-drafter-swap", repo_type="dataset", local_dir=f"{REFS}/p2", token=tok,
                      allow_patterns=[f"assets/ref_stock_30B/{STOCK_PREFIX.get(s, s)}_B32_t95_stock.{s}.shard*.jsonl"
                                      for s in SETS])
    open(f"{REFS}/.ok", "w").write("ok\n")


def paired(A, B, ids):
    d = np.array([int(bool(A[i].get("correct"))) - int(bool(B[i].get("correct"))) for i in ids])
    if len(d) == 0:
        return None
    rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def ppb(R, ids):
    num = sum(R[i]["n_forward"] - R[i]["n_prefill"] - R[i]["n_flush"] for i in ids)
    den = sum(R[i]["n_blocks"] for i in ids)
    return num / den if den else float("nan")


def acc(R, ids):
    return 100.0 * float(np.mean([bool(R[i].get("correct")) for i in ids])) if ids else float("nan")


def fmt(v):
    return "·" if v is None else f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    fetch_refs()
    rows, pooled = [], {"n": 0, "new": 0, "b128": 0, "stock": 0}
    allN, allB, allS, ids_all = {}, {}, {}, []
    for s in SETS:
        N = load(f"{a.dir}/{s}_judge/*.jsonl")
        if not N:
            continue
        B = load(f"{REFS}/p3/logs/phase3_8gpu/decode/base/{s}/*.jsonl")
        St = load(f"{REFS}/p2/assets/ref_stock_30B/{STOCK_PREFIX.get(s, s)}_B32_t95_stock.{s}.shard*.jsonl")
        ids = sorted(i for i in N if i in B and i in St)
        if not ids:
            continue
        r = {"set": s, "n": len(ids), "acc_new": acc(N, ids), "acc_128m": acc(B, ids), "acc_stock": acc(St, ids),
             "d_vs_128m": paired(N, B, ids), "d_vs_stock": paired(N, St, ids),
             "ppb_new": ppb(N, ids), "ppb_128m": ppb(B, ids), "ppb_stock": ppb(St, ids)}
        r["passes_cut_vs_128m"] = r["ppb_128m"] / r["ppb_new"]
        r["passes_cut_vs_stock"] = r["ppb_stock"] / r["ppb_new"]
        rows.append(r)
        for i in ids:
            k = (s, i)
            allN[k], allB[k], allS[k] = N[i], B[i], St[i]
            ids_all.append(k)
    if not rows:
        print("no comparable items"); return
    tot = {"set": "ALL (pooled)", "n": len(ids_all), "acc_new": acc(allN, ids_all), "acc_128m": acc(allB, ids_all),
           "acc_stock": acc(allS, ids_all), "d_vs_128m": paired(allN, allB, ids_all), "d_vs_stock": paired(allN, allS, ids_all),
           "ppb_new": ppb(allN, ids_all), "ppb_128m": ppb(allB, ids_all), "ppb_stock": ppb(allS, ids_all)}
    tot["passes_cut_vs_128m"] = tot["ppb_128m"] / tot["ppb_new"]
    tot["passes_cut_vs_stock"] = tot["ppb_stock"] / tot["ppb_new"]
    rows.append(tot)
    head = (f"# 228M judge vs 128M judge vs stock (base decode, no pool; paired items)\n\n"
            f"Checkpoint: {open(os.path.join(a.dir, 'checkpoint.txt')).read().strip()}\n\n"
            "| set | n | acc 228M | acc 128M | acc stock | Δ vs 128M [95% CI] | Δ vs stock [95% CI] | passes/blk 228M | 128M | stock | fewer passes vs 128M | vs stock |\n"
            "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|\n")
    lines = [f"| {r['set']} | {r['n']} | {r['acc_new']:.1f} | {r['acc_128m']:.1f} | {r['acc_stock']:.1f} | {fmt(r['d_vs_128m'])} | "
             f"{fmt(r['d_vs_stock'])} | {r['ppb_new']:.2f} | {r['ppb_128m']:.2f} | {r['ppb_stock']:.2f} | "
             f"{r['passes_cut_vs_128m']:.3f}x | {r['passes_cut_vs_stock']:.3f}x |" for r in rows]
    open(os.path.join(a.dir, "compare.md"), "w").write(head + "\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(a.dir, "compare.json"), "w"), indent=2)
    print(head + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
