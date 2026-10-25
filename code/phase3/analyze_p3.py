#!/usr/bin/env python
"""Phase 3 tables: the re-draft setups decoded on this box against today's judge and stock (earlier records).

    /venv/p2/bin/python code/phase3/analyze_p3.py

Reads logs/phase3/decode/<arm>/<set>/*.jsonl (rd1, rd2), Phase 1's no-pool SDAR-1.7B + 128M judge records
(logs/phase2/ref/S0_128M_nopool_by_set/<set>/) and the stock records (logs/phase2/ref/stock_flat/). Writes
logs/phase3/analysis/{TABLE.md, metrics.json} and copies both (plus the setup checks) to results/phase3/.

Accuracy, passes per block, expert loads per token and drafter forwards per token do not depend on the machine, so
they compare across every setup. Tokens per second compares only setups decoded in THIS run (same box); the earlier
records were timed on other machines, so their times are not set against these.
Paired accuracy deltas: bootstrap over items, 2000 resamples, seed 0 (pooled: resampled within each benchmark).
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np

SETS = [("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500"), ("ifeval541", "IFEval-541"),
        ("math500", "MATH-500"), ("gpqa", "GPQA-Diamond"), ("aime", "AIME")]
DRAFTER_EQ = 364.64          # expert-equivalents per SDAR-1.7B forward (compare_table.py)
REF = ["stock", "judge_p1"]
LABEL = {"stock": "Stock 30B (earlier records)", "judge_p1": "SDAR-1.7B + judge (Phase 1 records)",
         "rd1": "re-draft every pass", "rd2": "re-draft every 2 passes", "judge": "SDAR-1.7B + judge (this box)"}
OUT = "logs/phase3/analysis"


def load(paths):
    recs = {}
    for f in sorted(paths):
        if ".running." in f:
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs[int(r["idx"])] = r
    return recs


def arms_here():
    return sorted(os.path.basename(d) for d in glob.glob("logs/phase3/decode/*") if os.path.isdir(d))


def sources(key):
    src = {"stock": glob.glob(f"logs/phase2/ref/stock_flat/*_B32_t95_stock.{key}.shard*.jsonl"),
           "judge_p1": glob.glob(f"logs/phase2/ref/S0_128M_nopool_by_set/{key}/*.jsonl")}
    for a in arms_here():
        src[a] = glob.glob(f"logs/phase3/decode/{a}/{key}/*.jsonl")
    return src


def tot(R, f):
    return sum(f(r) for r in R)


def metrics(recs):
    if not recs:
        return None
    R = list(recs.values())
    tok = tot(R, lambda r: r["n_gen_to_eos"]); sec = tot(R, lambda r: r["seconds"]); nb = tot(R, lambda r: r["n_blocks"])
    passes = tot(R, lambda r: r["n_forward"] - r["n_prefill"] - r["n_flush"])
    loads = tot(R, lambda r: r["unique_expert_layer_loads"])
    dfw = tot(R, lambda r: sum((r.get("drafter_n_by_kind") or {}).values()))
    forced = tot(R, lambda r: sum(1 for blk in (r.get("full_passes_per_block") or []) for p in blk if p.get("forced")))
    m = {"n": len(R), "correct": tot(R, lambda r: int(bool(r.get("correct")))), "tokens": tok, "seconds": sec,
         "blocks": nb, "passes": passes, "loads": loads, "drafter_fwd": dfw, "forced": forced}
    rd = {k: tot(R, lambda r, k=k: sum(r.get(k) or [])) for k in
          ("redraft_rounds_per_block", "redraft_slots_per_block", "redraft_eq_judge_per_block", "redraft_eq_old_per_block")}
    m.update({"rd_rounds": rd["redraft_rounds_per_block"], "rd_slots": rd["redraft_slots_per_block"],
              "rd_eq_judge": rd["redraft_eq_judge_per_block"], "rd_eq_old": rd["redraft_eq_old_per_block"]})
    cf = [0, 0]; cm = [0, 0]
    for r in R:
        d = r.get("judge_diag") or {}
        for acc, k in ((cf, "copy_flag"), (cm, "copy_mask")):
            if k in d:
                acc[0] += d[k][0]; acc[1] += d[k][1]
    m.update({"copy_flag": cf, "copy_mask": cm, "gpu": R[0].get("gpu")})
    return derive(m)


def derive(m):
    m = dict(m)
    m["acc"] = 100.0 * m["correct"] / m["n"] if m["n"] else None
    m["tps"] = m["tokens"] / m["seconds"] if m["seconds"] else None
    m["ms_tok"] = 1000.0 * m["seconds"] / m["tokens"] if m["tokens"] else None
    m["passes_blk"] = m["passes"] / m["blocks"] if m["blocks"] else None
    m["exp_tok"] = m["loads"] / m["tokens"] if m["tokens"] else None
    m["dfw_tok"] = m["drafter_fwd"] / m["tokens"] if m["tokens"] else None
    m["charged_tok"] = (m["loads"] + DRAFTER_EQ * m["drafter_fwd"]) / m["tokens"] if m["tokens"] else None
    m["forced_blk"] = m["forced"] / m["blocks"] if m["blocks"] else None
    m["rd_rounds_blk"] = m["rd_rounds"] / m["blocks"] if m["blocks"] else None
    m["rd_slots_blk"] = m["rd_slots"] / m["blocks"] if m["blocks"] else None
    m["rd_eq_judge_share"] = m["rd_eq_judge"] / m["rd_slots"] if m["rd_slots"] else None
    m["rd_eq_old_share"] = m["rd_eq_old"] / m["rd_slots"] if m["rd_slots"] else None
    m["copy_flag_rate"] = 100.0 * m["copy_flag"][0] / m["copy_flag"][1] if m["copy_flag"][1] else None
    m["copy_mask_rate"] = 100.0 * m["copy_mask"][0] / m["copy_mask"][1] if m["copy_mask"][1] else None
    return m


def pooled(ms):
    ms = [m for m in ms if m]
    if not ms:
        return None
    keys = ("n", "correct", "tokens", "seconds", "blocks", "passes", "loads", "drafter_fwd", "forced",
            "rd_rounds", "rd_slots", "rd_eq_judge", "rd_eq_old")
    p = {k: sum(m[k] for m in ms) for k in keys}
    p["copy_flag"] = [sum(m["copy_flag"][0] for m in ms), sum(m["copy_flag"][1] for m in ms)]
    p["copy_mask"] = [sum(m["copy_mask"][0] for m in ms), sum(m["copy_mask"][1] for m in ms)]
    p["gpu"] = ms[0].get("gpu")
    return derive(p)


def diffs(a, b):
    ids = sorted(set(a) & set(b))
    return np.array([int(bool(a[i].get("correct"))) - int(bool(b[i].get("correct"))) for i in ids], float)


def paired(pairs, n_boot=2000, seed=0):
    """pairs: [(recs_a, recs_b)] per benchmark -> delta of accuracy (points) with a 95% CI, resampled per benchmark."""
    ds = [d for d in (diffs(a, b) for a, b in pairs) if len(d)]
    if not ds:
        return None
    N = sum(len(d) for d in ds)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n_boot)
    for d in ds:
        acc += d[rng.integers(0, len(d), size=(n_boot, len(d)))].sum(1)
    return {"n": N, "delta": 100.0 * sum(d.sum() for d in ds) / N,
            "lo": 100.0 * float(np.percentile(acc / N, 2.5)), "hi": 100.0 * float(np.percentile(acc / N, 97.5))}


def f(v, p=1, suf=""):
    return "—" if v is None else f"{v:.{p}f}{suf}"


def sd(v, p=1):
    if v is None:
        return "—"
    s = f"{abs(v):.{p}f}"
    return s if float(s) == 0 else ("+" if v > 0 else "−") + s


def ci(d):
    return "—" if not d else f"{sd(d['delta'])} [{sd(d['lo'])}, {sd(d['hi'])}]"


def main():
    os.makedirs(OUT, exist_ok=True)
    here = [a for a in arms_here()]
    cols = REF + here
    judges = ["judge_p1"] + here                       # the setups that compete (stock is the reference)
    data, recs_all, table, out = {}, {}, [], {"arms_this_box": here, "sets": {}}
    for key, name in SETS:
        src = sources(key)
        recs = {a: load(src.get(a, [])) for a in cols}
        recs_all[key] = recs
        data[key] = {a: metrics(recs[a]) for a in cols}
    # ---- Table 1: accuracy, passes per block, tokens per second ------------------------------------------------
    lines = ["# Phase 3: re-draft between judge passes", "",
             "SDAR-1.7B drafter + Phase 1's 128M judge, block 32, threshold 0.95, s8 first draft, no expert pool. "
             "rd1 / rd2 re-draft the [MASK] slots after every judge pass / every 2 passes, from committed tokens only.", ""]
    chk = "logs/phase3/setup/checks.json"
    if os.path.exists(chk):
        lines += [f"Setup checks: `{json.dumps(json.load(open(chk)))}`", ""]
    gpus = {data[k][a]["gpu"] for k, _ in SETS for a in here if data[k].get(a)}
    lines += [f"This box: {', '.join(sorted(g for g in gpus if g)) or '—'}. Tokens per second compares only the setups "
              f"decoded here ({', '.join(here) or 'none yet'}); stock and today's judge times come from other machines and are not shown.", "",
              "## Table 1", "",
              "| Benchmark | Measure | " + " | ".join(LABEL.get(a, a) for a in cols) + " |",
              "|---|---|" + "---:|" * len(cols)]
    full = [k for k, _ in SETS if all(data[k].get(a) for a in cols)]     # benchmarks every setup has finished
    all_name = f"All benchmarks ({len(full)} of {len(SETS)} with every setup finished)"
    for key, name in SETS + [("ALL", all_name)]:
        if key == "ALL":
            if not full:
                continue
            ms = {a: pooled([data[k][a] for k in full]) for a in cols}
            dv = {a: paired([(recs_all[k][a], recs_all[k]["stock"]) for k in full]) for a in judges}
        else:
            ms = data[key]
            dv = {a: paired([(recs_all[key][a], recs_all[key]["stock"])]) for a in judges}
        if not any(ms.values()):
            continue
        n = ms["stock"]["n"] if ms.get("stock") else (next(m for m in ms.values() if m)["n"])
        accs = {a: ms[a]["acc"] for a in judges if ms.get(a) and ms[a]["acc"] is not None}
        best_acc = max((round(v, 1) for v in accs.values()), default=None)
        pbs = {a: ms[a]["passes_blk"] for a in judges if ms.get(a)}
        best_pb = min((round(v, 2) for v in pbs.values()), default=None)
        tpss = {a: ms[a]["tps"] for a in here if ms.get(a)}
        best_tps = max((round(v, 2) for v in tpss.values()), default=None)
        row_a, row_p, row_t = [], [], []
        for a in cols:
            m = ms.get(a)
            if not m:
                row_a.append("—"); row_p.append("—"); row_t.append("—"); continue
            s = f(m["acc"])
            if a != "stock":
                s += f" ({ci(dv.get(a)) if key == 'ALL' else sd(dv[a]['delta'] if dv.get(a) else None)})"
            if a in accs and round(m["acc"], 1) == best_acc:
                s = f"**{s}**"
            row_a.append(s)
            st_pb = ms["stock"]["passes_blk"] if ms.get("stock") else None
            s = f(m["passes_blk"], 2)
            if a != "stock" and st_pb:
                s += f" ({st_pb / m['passes_blk']:.2f}×)"
            if a in pbs and round(m["passes_blk"], 2) == best_pb:
                s = f"**{s}**"
            row_p.append(s)
            if a in here:
                s = f"{m['tps']:.2f} ({m['ms_tok']:.1f} ms)"
                if round(m["tps"], 2) == best_tps and len(tpss) > 1:
                    s = f"**{s}**"
                row_t.append(s)
            else:
                row_t.append("other machine")
        lines.append(f"| {name} (n = {n}) | Accuracy, % (Δ vs stock) | " + " | ".join(row_a) + " |")
        lines.append("|  | Passes / block (× fewer than stock) | " + " | ".join(row_p) + " |")
        lines.append("|  | Tokens / s (ms / token), this box | " + " | ".join(row_t) + " |")
    lines += ["", "Bold: the best of the judge setups in that row (stock is the reference and never bold). "
              "Accuracy deltas on single benchmarks are point estimates; the all-benchmarks row adds the 95% CI.", ""]
    # ---- Table 2: against today's judge, and the re-draft / copy diagnostics ------------------------------------
    lines += ["## Table 2: against today's judge (Phase 1 records)", "",
              "| Benchmark | Setup | Δ accuracy vs today's judge [95% CI] | Passes / block (today → setup) | Expert loads / token | "
              "Drafter forwards / token | Charged loads / token | Forced passes / block | Re-draft rounds / block | "
              "Re-drafted slots / block | Re-draft = judge's token | Re-draft = rejected draft | Copy rate, draft slots | Copy rate, mask slots |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, name in SETS + [("ALL", all_name)]:
        for a in here:
            if key == "ALL":
                if not full:
                    continue
                m = pooled([data[k][a] for k in full]); j = pooled([data[k]["judge_p1"] for k in full])
                d = paired([(recs_all[k][a], recs_all[k]["judge_p1"]) for k in full])
            else:
                m = data[key].get(a); j = data[key].get("judge_p1")
                d = paired([(recs_all[key][a], recs_all[key]["judge_p1"])])
            if not m:
                continue
            pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"
            lines.append(f"| {name} | {a} | {ci(d)} | {f(j['passes_blk'], 2) if j else '—'} → {f(m['passes_blk'], 2)} | "
                         f"{f(j['exp_tok'], 0) if j else '—'} → {f(m['exp_tok'], 0)} | {f(j['dfw_tok'], 3) if j else '—'} → {f(m['dfw_tok'], 3)} | "
                         f"{f(j['charged_tok'], 0) if j else '—'} → {f(m['charged_tok'], 0)} | {f(j['forced_blk'], 2) if j else '—'} → {f(m['forced_blk'], 2)} | "
                         f"{f(m['rd_rounds_blk'], 2)} | {f(m['rd_slots_blk'], 2)} | {pct(m['rd_eq_judge_share'])} | {pct(m['rd_eq_old_share'])} | "
                         f"{f(m['copy_flag_rate'], 2, '%')} | {f(m['copy_mask_rate'], 2, '%')} |")
    lines += ["", "Re-draft = judge's token: share of re-drafted tokens equal to the judge's top choice at the pass before "
              "(the drafter switched to what the judge wanted). Copy rate: share of open slots whose judge argmax equals the "
              "next slot's token (the untrained next-token copy); compare rd1 with rd2, not with zero.", ""]
    for key, _ in SETS:
        out["sets"][key] = {a: data[key][a] for a in cols}
    out["all_sets_used"] = full
    out["all"] = {a: pooled([data[k][a] for k in full]) for a in cols}
    out["delta_vs_stock_all"] = {a: paired([(recs_all[k][a], recs_all[k]["stock"]) for k in full]) for a in judges}
    out["delta_vs_judge_all"] = {a: paired([(recs_all[k][a], recs_all[k]["judge_p1"]) for k in full]) for a in here}
    with open(os.path.join(OUT, "TABLE.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    os.makedirs("results/phase3", exist_ok=True)
    for fn in ("TABLE.md", "metrics.json"):
        shutil.copy(os.path.join(OUT, fn), os.path.join("results/phase3", fn))
    if os.path.exists(chk):
        shutil.copy(chk, "results/phase3/setup_checks.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
