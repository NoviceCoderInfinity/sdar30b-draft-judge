#!/usr/bin/env python
"""Round-3 downstream eval summary: r3_A judge checkpoints vs stock, from whatever records exist right now.

    /venv/r3/bin/python code/draftalign/eval_summary_r3.py            # -> probes_r3/r3_A/eval_summary.{json,md}
    /venv/r3/bin/python code/draftalign/eval_summary_r3.py --print    # also echo the markdown

Read-only over every record directory; writes only eval_summary.json and eval_summary.md. Safe to re-run at any
time: shards still being written are read line by line (a torn last line is skipped), missing directories are
"pending", and code sets without a finished `scored/` shard are "unscored" (efficiency shown, accuracy withheld).

Layout (see docs/report/build_r3_report.py for the rendered view):
  judge arms   logs/draftalign/probes_r3/r3_A/<step>/<ds>/<arm>/*.shard*of*.jsonl      arm in judge, judge_des32, judge_des48
  new stock    logs/draftalign/probes_r3/r3_A/stock/<ds>/<arm>/                         arm in stock, stock_des32, stock_des48
  team stock   /dev/shm/dlm/artifacts_mirror/logs/draftalign/da_decode/<cell>/<cell>.<split>.shard*.jsonl
  prev round   same mirror, cells *_dv-s8_full_c95[_p32desf|_p48desf]_ad734af656 (T2b step 3057, documentation only)

Correctness: humaneval/mbpp are decoded without --score, so `correct` is authoritative only in <arm>/scored/*.jsonl
with a matching .jsonl.done; gsm8k500/math500 are scored inline. `scored/` is preferred whenever it exists.
A scored record carrying `score_error` is INVALID (scorer failure, not "incorrect"): if any scored record of a code
cell has one, the whole cell's accuracy is withheld and the cell reads "unscored" (scoring pending) until re-scored.

Metrics (AGENTS.md section 8, verbatim): passes/blk = sum(n_forward - n_prefill - n_flush) / sum(n_blocks);
exp/token = sum(unique_expert_layer_loads) / sum(n_gen_to_eos); ms/token = 1000 sum(seconds) / sum(n_gen_to_eos).
Every comparison is PAIRED: both sides are computed on exactly the idx set they share. Delta-acc CI = bootstrap over
shared items, 2,000 resamples, numpy default_rng(0) per comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PR = os.path.join(ROOT, "logs", "draftalign", "probes_r3", "r3_A")
MIRROR = "/dev/shm/dlm/artifacts_mirror/logs/draftalign/da_decode"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
PREV_TAG = "_ad734af656"   # T2b step 3057 = the published round-2 judge (docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md 2.3.1)

# key, label, planned n, code set (scored/ needed), mirror family, mirror split
DATASETS = [
    ("humaneval", "HumanEval", 164, True, "humaneval", "humaneval"),
    ("mbpp", "MBPP", 257, True, "mbpp", "mbpp"),
    ("gsm8k500", "GSM8K-500 (first 100)", 100, False, "gsm8k", "gsm8k500"),
    ("math500", "MATH-500 (first 50)", 50, False, "math500", "math500"),
]
DS = {d[0]: d for d in DATASETS}
# known checkpoints; any other logs/draftalign/probes_r3/r3_A/step* dir is picked up generically as a normal step
STEP_META = {700: ("epoch 1", "normal", ""), 1400: ("epoch 2", "normal", ""),
             1500: ("dropped", "dropped", "dropped at 09:40 IST so training could resume at 09:53 IST (user decision)")}
# 11:08 IST: the step001900 final eval was cancelled (training runs to the 12:20 shutdown); any later step* dir still shows up generically
FINAL_DS = ("humaneval", "mbpp")


def discover_steps():
    found = set()
    for d in glob.glob(os.path.join(PR, "step*")):
        b = os.path.basename(d)[4:]
        if os.path.isdir(d) and b.isdigit():
            found.add(int(b))
    out = []
    for n in sorted(found | set(STEP_META)):
        label, mode, note = STEP_META.get(n, (f"step {n:,}", "normal", ""))
        out.append((f"step{n:06d}", label, n, mode, note))
    return out


def apply_mode(c, mode, note, ds):
    c["mode"], c["note"] = mode, note
    if mode == "scheduled" and c["status"] == "pending":
        c["status"] = "scheduled" if ds in FINAL_DS else "n/a"
    elif mode == "dropped" and c["status"] in ("pending", "partial", "unscored"):
        c["status"] = "dropped"
    return c
POOLS = [(None, ""), (32, "_des32"), (48, "_des48")]
MIRROR_POOL = {None: "", 32: "_p32desf", 48: "_p48desf"}
ARM_LABEL = {"stock": "stock", "stock_des32": "stock + DES32", "stock_des48": "stock + DES48",
             "judge": "judge", "judge_des32": "judge + DES32", "judge_des48": "judge + DES48"}
SHARD_RE = re.compile(r"\.shard(\d+)of(\d+)\.jsonl$")


# ----------------------------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # torn last line of a shard still being written
                    continue
    except OSError:
        pass
    return out


def correct_of(r):
    if r.get("score_error") is not None:
        return None
    c = r.get("correct")
    return None if c is None else int(bool(c))


def load_arm_dir(d, code):
    """-> dict(recs={idx: rec}, scored={idx: 0/1}, source, shards_done, shards_total, last_utc)."""
    files = sorted(f for f in glob.glob(os.path.join(d, "*.shard*of*.jsonl")) if ".oldcfg" not in f)
    nshards = max([int(SHARD_RE.search(f).group(2)) for f in files if SHARD_RE.search(f)] or [0])
    done = sum(os.path.exists(f + ".done") for f in files)
    recs = {}
    for f in files:
        for r in read_jsonl(f):
            if "idx" in r:
                recs[int(r["idx"])] = r
    sdir = os.path.join(d, "scored")
    sfiles = sorted(f for f in glob.glob(os.path.join(sdir, "*.jsonl")) if os.path.exists(f + ".done"))
    scored, source, score_errors = {}, "none", 0
    if sfiles:
        source = "scored"
        for f in sfiles:
            for r in read_jsonl(f):
                if "idx" not in r:
                    continue
                i = int(r["idx"]); recs.setdefault(i, r)
                if r.get("score_error") is not None:
                    score_errors += 1
                    continue
                c = correct_of(r)
                if c is not None:
                    scored[i] = c
        if score_errors and code:          # scorer was broken for this cell: withhold accuracy until it is re-scored
            scored = {}
    elif recs and not code:
        source = "inline"
        scored = {i: c for i, r in recs.items() if (c := correct_of(r)) is not None}
    elif recs:
        source = "unscored"
    utcs = [r.get("utc") for r in recs.values() if r.get("utc")]
    return {"recs": recs, "scored": scored, "source": source, "shards_done": done, "shards_total": nshards,
            "n_scored_shards": len(sfiles), "score_errors": score_errors,
            "last_utc": max(utcs) if utcs else None, "path": os.path.relpath(d, ROOT)}


def load_mirror(cell, split):
    files = sorted(f for f in glob.glob(os.path.join(MIRROR, cell, f"{cell}.{split}.shard*.jsonl")) if ".oldcfg" not in f)
    recs = {}
    for f in files:
        for r in read_jsonl(f):
            if "idx" in r:
                recs[int(r["idx"])] = r
    scored = {i: c for i, r in recs.items() if (c := correct_of(r)) is not None}
    nshards = max([int(SHARD_RE.search(f).group(2)) for f in files if SHARD_RE.search(f)] or [0])
    return {"recs": recs, "scored": scored, "source": "mirror" if recs else "none",
            "shards_done": sum(os.path.exists(f + ".done") for f in files), "shards_total": nshards,
            "n_scored_shards": 0, "score_errors": 0, "last_utc": None, "path": f"artifacts_mirror/da_decode/{cell} ({split})"}


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
def eff(recs):
    rs = list(recs)
    if not rs:
        return {"n": 0, "passes_blk": None, "exp_tok": None, "ms_tok": None, "sec_item": None}
    nb = sum(r.get("n_blocks", 0) for r in rs); ng = sum(r.get("n_gen_to_eos", 0) for r in rs)
    return {"n": len(rs),
            "passes_blk": sum(r["n_forward"] - r["n_prefill"] - r["n_flush"] for r in rs) / nb if nb else None,
            "exp_tok": sum(r["unique_expert_layer_loads"] for r in rs) / ng if ng else None,
            "ms_tok": 1000.0 * sum(r.get("seconds", 0.0) for r in rs) / ng if ng else None,
            "sec_item": sum(r.get("seconds", 0.0) for r in rs) / len(rs)}


def bootstrap_ci(diff, n_boot=2000, seed=0):
    diff = np.asarray(diff, dtype=float)
    if diff.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    return [100.0 * float(np.percentile(means, 2.5)), 100.0 * float(np.percentile(means, 97.5))]


def compare(a, b):
    """Paired a vs b on the idx they share. a/b = loaded arm dicts (restricted already)."""
    if not a["recs"] or not b["recs"]:
        return None
    common = sorted(set(a["recs"]) & set(b["recs"]))
    if not common:
        return None
    ea, eb = eff(a["recs"][i] for i in common), eff(b["recs"][i] for i in common)
    both = [i for i in common if i in a["scored"] and i in b["scored"]]
    out = {"n_common": len(common), "n_common_scored": len(both),
           "passes_blk": ea["passes_blk"], "ref_passes_blk": eb["passes_blk"],
           "exp_tok": ea["exp_tok"], "ref_exp_tok": eb["exp_tok"],
           "speedup_exp": (eb["exp_tok"] / ea["exp_tok"]) if ea["exp_tok"] and eb["exp_tok"] else None,
           "sec_item": ea["sec_item"], "ref_sec_item": eb["sec_item"],
           "acc": None, "ref_acc": None, "d_acc": None, "ci": None, "a_only": None, "b_only": None}
    if both:
        xa = np.array([a["scored"][i] for i in both]); xb = np.array([b["scored"][i] for i in both])
        out.update(acc=100.0 * xa.mean(), ref_acc=100.0 * xb.mean(), d_acc=100.0 * (xa - xb).mean(),
                   ci=bootstrap_ci(xa - xb), a_only=int(((xa == 1) & (xb == 0)).sum()),
                   b_only=int(((xa == 0) & (xb == 1)).sum()))
    return out


def restrict(arm, idx):
    if idx is None:
        return arm
    s = set(idx)
    return dict(arm, recs={i: r for i, r in arm["recs"].items() if i in s},
                scored={i: c for i, c in arm["scored"].items() if i in s})


def cell_summary(arm, planned, code):
    n_rec, n_sc = len(arm["recs"]), len(arm["scored"])
    if n_rec == 0:
        status = "pending"
    elif code and not arm["scored"]:
        status = "unscored"
    elif n_sc >= planned and n_rec >= planned:
        status = "complete"
    else:
        status = "partial"
    e = eff(arm["recs"].values())
    acc = 100.0 * sum(arm["scored"].values()) / n_sc if n_sc else None
    hashes = sorted({r.get("adapter_hash") for r in arm["recs"].values() if r.get("adapter_hash")})
    return {"status": status, "n_planned": planned, "n_records": n_rec, "n_scored": n_sc, "source": arm["source"],
            "shards": f"{arm['shards_done']}/{arm['shards_total']}" if arm["shards_total"] else "-",
            "scored_shards": arm["n_scored_shards"], "score_errors": arm.get("score_errors", 0), "acc": acc, "passes_blk": e["passes_blk"], "exp_tok": e["exp_tok"],
            "ms_tok": e["ms_tok"], "sec_item": e["sec_item"], "last_utc": arm["last_utc"], "path": arm["path"],
            "adapter_hash": hashes[0][:16] if hashes else None}


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------
def build():
    now = dt.datetime.now(dt.timezone.utc)
    steps = discover_steps()
    res = {"generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generated_ist": now.astimezone(IST).strftime("%Y-%m-%d %H:%M IST"),
           "datasets": [{"key": k, "label": l, "planned": n, "code": c} for k, l, n, c, _, _ in DATASETS],
           "steps": [{"key": s, "label": l, "step": n, "mode": m, "note": no} for s, l, n, m, no in steps],
           "arm_labels": ARM_LABEL, "prev_round_tag": PREV_TAG,
           "universe": {}, "stock": {}, "prev_round": {}, "cells": {}, "direction": {}, "coverage": {}}
    for key, label, planned, code, fam, split in DATASETS:
        # --- load every source for this dataset
        judge = {s: {f"judge{suf}": load_arm_dir(os.path.join(PR, s, key, f"judge{suf}"), code) for _, suf in POOLS}
                 for s, *_ in steps}
        stock, prev = {}, {}
        for M, suf in POOLS:
            local = load_arm_dir(os.path.join(PR, "stock", key, f"stock{suf}"), code)
            stock[f"stock{suf}"] = local if local["recs"] else load_mirror(f"{fam}_B32_t95_stock{MIRROR_POOL[M]}", split)
            prev[f"judge{suf}"] = load_mirror(f"{fam}_B32_t95_dv-s8_full_c95{MIRROR_POOL[M]}{PREV_TAG}", split)
        # --- the item universe the round-3 wave covers
        observed = set()
        for s in judge:
            for a in judge[s].values():
                observed |= set(a["recs"])
        for nm, a in stock.items():
            if a["source"] != "mirror":
                observed |= set(a["recs"])
        ref_idx = sorted(next((a["recs"] for a in stock.values() if a["source"] == "mirror" and a["recs"]), {}))
        if key in ("humaneval", "mbpp", "math500") and len(ref_idx) >= planned:
            universe, how = ref_idx[:planned], f"first {planned} idx of the reference set"
        elif observed:
            universe, how = sorted(observed), "idx observed in round-3 records so far"
        else:
            universe, how = ref_idx[:planned], "provisional: first idx of the reference set (no round-3 records yet)"
        res["universe"][key] = {"n": len(universe), "how": how}
        stock = {k: restrict(v, universe) for k, v in stock.items()}
        prev = {k: restrict(v, universe) for k, v in prev.items()}
        judge = {s: {k: restrict(v, universe) for k, v in arms.items()} for s, arms in judge.items()}

        # --- stock rows
        res["stock"][key] = {}
        for M, suf in POOLS:
            nm = f"stock{suf}"; c = cell_summary(stock[nm], planned, code and stock[nm]["source"] != "mirror")
            c["vs_stock"] = compare(stock[nm], stock["stock"]) if nm != "stock" else None
            res["stock"][key][nm] = c
        # --- previous round (documentation only)
        res["prev_round"][key] = {}
        for M, suf in POOLS:
            nm = f"judge{suf}"; c = cell_summary(prev[nm], planned, False)
            c["vs_stock"] = compare(prev[nm], stock["stock"])
            c["vs_pool_stock"] = compare(prev[nm], stock[f"stock{suf}"]) if suf else None
            c["cell"] = f"{fam}_B32_t95_dv-s8_full_c95{MIRROR_POOL[M]}{PREV_TAG}"
            res["prev_round"][key][nm] = c
        # --- judge cells per checkpoint
        res["cells"].setdefault(key, {})
        for s, _, _, mode, note in steps:
            res["cells"][key][s] = {}
            for M, suf in POOLS:
                nm = f"judge{suf}"; a = judge[s][nm]; c = apply_mode(cell_summary(a, planned, code), mode, note, key)
                c["vs_stock"] = compare(a, stock["stock"])
                c["vs_pool_stock"] = compare(a, stock[f"stock{suf}"]) if suf else None
                c["vs_prev"] = compare(a, prev[nm])
                res["cells"][key][s][nm] = c
        # --- epoch direction: epoch 2 vs epoch 1, latest vs epoch 2 (same arm, paired)
        res["direction"][key] = {}
        for M, suf in POOLS:
            nm = f"judge{suf}"
            res["direction"][key][nm] = {
                "e2_vs_e1": compare(judge["step001400"][nm], judge["step000700"][nm]),
                "latest_vs_e2": compare(judge["step001500"][nm], judge["step001400"][nm]) if "step001500" in judge else None,
                "final_vs_e2": compare(judge["step001900"][nm], judge["step001400"][nm]) if "step001900" in judge else None}
    # --- coverage roll-up
    tot = {"complete": 0, "partial": 0, "unscored": 0, "pending": 0, "dropped": 0, "scheduled": 0, "n/a": 0}
    for key in res["cells"]:
        for s in res["cells"][key]:
            for c in res["cells"][key][s].values():
                tot[c["status"]] += 1
        for nm, c in res["stock"][key].items():
            tot[c["status"]] += 1
    res["coverage"] = tot
    return res


# ----------------------------------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------------------------------
def f(x, nd=1, sign=False, suffix=""):
    if x is None:
        return "—"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}") + suffix


def dacc(cmp):
    if not cmp or cmp.get("d_acc") is None:
        return "—"
    lo, hi = cmp["ci"]
    return f"{cmp['d_acc']:+.1f} [{lo:+.1f}, {hi:+.1f}] (n={cmp['n_common_scored']})"


def status_txt(c):
    if c["status"] in ("dropped", "scheduled", "n/a"):
        return {"n/a": "not planned"}.get(c["status"], c["status"].upper()) + (f" ({c['n_scored']}/{c['n_planned']} ran)" if c["n_records"] else "")
    if c["status"] == "complete":
        return "complete"
    if c["status"] == "pending":
        return "PENDING"
    if c["status"] == "unscored":
        return f"SCORING PENDING ({c['n_records']} decoded" + (f", {c['score_errors']} invalid scorer records ignored" if c.get("score_errors") else "") + ")"
    return f"PARTIAL {c['n_scored']}/{c['n_planned']} scored, {c['n_records']} decoded"


def to_markdown(res):
    L = [f"# Round-3 downstream eval summary (r3_A)", "",
         f"Generated {res['generated_ist']} ({res['generated_utc']}). Every Δ is paired on the shared idx; "
         "stock rows are on the dataset universe below. Metric definitions: AGENTS.md §8.", "",
         f"Coverage: {res['coverage']}", ""]
    hdr = "| arm | status | n acc / n eff | acc % | Δacc vs stock [95% CI] | passes/blk | exp/token | speedup (exp) | s/item |"
    sep = "|---|---|---:|---:|---|---:|---:|---:|---:|"
    for s, slabel, step in [(d["key"], d["label"], d["step"]) for d in res["steps"]]:
        L += [f"## {s} ({slabel})", ""]
        for d in res["datasets"]:
            k = d["key"]; u = res["universe"][k]
            L += [f"### {d['label']} — universe n={u['n']} ({u['how']})", "", hdr, sep]
            rows = [(nm, res["stock"][k][nm]) for nm in ("stock", "stock_des32", "stock_des48")] + \
                   [(nm, res["cells"][k][s][nm]) for nm in ("judge", "judge_des32", "judge_des48")]
            for nm, c in rows:
                v = c.get("vs_stock")
                L.append(f"| {ARM_LABEL[nm]} | {status_txt(c)} | {c['n_scored']} / {c['n_records']} | {f(c['acc'])} | "
                         f"{dacc(v) if nm != 'stock' else 'ref'} | {f(c['passes_blk'], 2)} | {f(c['exp_tok'])} | "
                         f"{f(v['speedup_exp'], 3, suffix='x') if v else ('1.000x' if nm == 'stock' and c['n_records'] else '—')} | "
                         f"{f(c['sec_item'])} |")
            L.append("")
    L += ["## Epoch direction (same arm, paired on shared items)", "",
          "| dataset | arm | epoch 2 − epoch 1 Δacc [CI] | e2-only right / e1-only right | exp/token e1 → e2 | final (step 1900) − epoch 2 Δacc |",
          "|---|---|---|---:|---|---|"]
    for d in res["datasets"]:
        for nm in ("judge", "judge_des32", "judge_des48"):
            e = res["direction"][d["key"]][nm]["e2_vs_e1"]; l2 = res["direction"][d["key"]][nm].get("final_vs_e2")
            flips = f"{e['a_only']} / {e['b_only']}" if e and e.get("a_only") is not None else "—"
            ex = f"{f(e['ref_exp_tok'])} → {f(e['exp_tok'])}" if e else "—"
            L.append(f"| {d['label']} | {ARM_LABEL[nm]} | {dacc(e)} | {flips} | {ex} | {dacc(l2)} |")
    L += ["", f"## Previous round — team checkpoint T2b step 3057 (`{PREV_TAG}`), documentation only", "",
          "| dataset | arm | n | acc % | Δacc vs stock [CI] | exp/token | speedup (exp) |", "|---|---|---:|---:|---|---:|---:|"]
    for d in res["datasets"]:
        for nm in ("judge", "judge_des32", "judge_des48"):
            c = res["prev_round"][d["key"]][nm]; v = c["vs_stock"]
            if c["status"] == "pending":
                L.append(f"| {d['label']} | {ARM_LABEL[nm]} | — | no cell | — | — | — |"); continue
            L.append(f"| {d['label']} | {ARM_LABEL[nm]} | {c['n_scored']} | {f(c['acc'])} | {dacc(v)} | {f(c['exp_tok'])} | "
                     f"{f(v['speedup_exp'], 3, suffix='x') if v else '—'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=PR)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    res = build()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, text in (("eval_summary.json", json.dumps(res, indent=1)), ("eval_summary.md", to_markdown(res))):
        tmp = os.path.join(args.out_dir, f".{name}.tmp")
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, os.path.join(args.out_dir, name))
    print(f"[eval_summary] {res['generated_ist']} coverage {res['coverage']} -> {args.out_dir}/eval_summary.{{json,md}}")
    if args.print:
        sys.stdout.write(to_markdown(res))
    return res


if __name__ == "__main__":
    main()
