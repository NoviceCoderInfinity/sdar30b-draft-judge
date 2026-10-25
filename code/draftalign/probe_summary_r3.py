#!/usr/bin/env python
"""Summarise round-3 checkpoint probes -> <probes>/summary.jsonl and <probes>/best.json.

For each probed step and dataset: accuracy, passes/blk, exp/token (AGENTS.md §8 definitions) and the PAIRED
delta vs the team's stock records on the same idx (mirror under REF). best.json = the step with the highest
mean Δacc across the three probe sets (ties: lower exp/token); ckpt_sync_r3.sh keeps that checkpoint on disk.
"""
from __future__ import annotations
import glob, json, os, sys

REF = "/dev/shm/dlm/artifacts_mirror/logs/draftalign/da_decode"
PROBES = {"gsm8k500": "gsm8k_B32_t95_stock", "mbpp": "mbpp_B32_t95_stock", "humaneval": "humaneval_B32_t95_stock",
          "aime": "aime_B32_t95_stock", "gpqa": "gpqa_B32_t95_stock"}
MAIN = ("gsm8k500", "mbpp", "humaneval")     # best/score over these three; aime/gpqa reported when present
ARMS = ("judge", "judge_des32", "judge_des48")   # probe arms; best = judge_des48 (the deck's headline arm)


def recs(pattern):
    out = {}
    for f in glob.glob(pattern):
        if ".oldcfg" in f: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[int(r["idx"])] = r
    return out


def metrics(rs):
    rs = list(rs)
    nb = sum(r["n_blocks"] for r in rs) or 1; ng = sum(r["n_gen_to_eos"] for r in rs) or 1
    return {"n": len(rs), "acc": 100.0 * sum(int(bool(r.get("correct"))) for r in rs) / max(len(rs), 1),
            "passes_blk": sum(r["n_forward"] - r["n_prefill"] - r["n_flush"] for r in rs) / nb,
            "exp_tok": sum(r["unique_expert_layer_loads"] for r in rs) / ng}


def main(probes_dir):
    stock = {ds: recs(f"{REF}/{cell}/{cell}.{ds}.shard*.jsonl") for ds, cell in PROBES.items()}
    rows = []
    for sd in sorted(glob.glob(os.path.join(probes_dir, "step*"))):
        step = int(os.path.basename(sd)[4:]); row = {"step": step}
        for ds in PROBES:
            for arm in ARMS:
                d = os.path.join(sd, ds, arm, "scored")
                if not glob.glob(os.path.join(d, "*.jsonl.done")): continue
                mine = recs(os.path.join(d, "*.jsonl")); m = metrics(mine.values())
                paired = [(int(bool(mine[i].get("correct"))), int(bool(stock[ds][i].get("correct")))) for i in mine if i in stock[ds]]
                m["d_acc"] = 100.0 * (sum(a for a, _ in paired) - sum(b for _, b in paired)) / max(len(paired), 1)
                sm = metrics(stock[ds][i] for i in mine if i in stock[ds]); m["speedup_exp"] = sm["exp_tok"] / max(m["exp_tok"], 1e-9)
                row[f"{ds}/{arm}"] = m
        keys = [f"{ds}/judge_des48" for ds in MAIN]
        if all(k in row for k in keys):
            row["score"] = sum(row[k]["d_acc"] for k in keys) / len(keys)
            row["exp_tok_mean"] = sum(row[k]["exp_tok"] for k in keys) / len(keys)
        rows.append(row)
    with open(os.path.join(probes_dir, "summary.jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    done = [r for r in rows if "score" in r]
    if done:
        best = max(done, key=lambda r: (r["score"], -r["exp_tok_mean"]))
        json.dump({"step": best["step"], "score": best["score"], "exp_tok_mean": best["exp_tok_mean"]},
                  open(os.path.join(probes_dir, "best.json"), "w"))
    for r in rows:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else {kk: round(vv, 2) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
