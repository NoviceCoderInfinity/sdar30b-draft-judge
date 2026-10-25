#!/usr/bin/env python
"""Per-arm report of what ran inside the decode: experts, router probabilities, judge and drafter probabilities.

For every decoded set of an arm (logs/phase2/decode/<ARM>/<set>/) it reads the records, the decode traces and the
Phase 2 logs, and writes logs/phase2/report/<ARM>/experts.{md,json}:

  EXPERTS, per forward kind (prefill = the prompt, judge = draft-filled judge passes, denoise = stock passes,
  kvflush = storing the final block):
    forwards per item; experts per layer in ONE forward (mean and max); experts per forward over all 48 layers;
    union of experts per layer over the whole item; normalised load entropy; max-load share; router top-8 mass
    (how much of the router's probability the 8 chosen experts carry, before renormalisation)
  per set: experts per layer per block forward, the union per item, experts loaded per token (and charged with the
    drafter), the per-layer profile (48 values: mean experts per block forward in each layer), the 20 most loaded
    (layer, expert) pairs
  JUDGE probabilities: mean top-1 probability and share of slots above 0.95 at judge pass 1, 2, 3, ...; passes per block
  DRAFTER probabilities (from the traces): mean confidence of its tokens, and its reliability -- confidence bins vs
    how often the drafted token equals the 30B's final token (10 bins)

    /venv/p2/bin/python code/phase2/expert_report.py --arm F1 --out logs/phase2/report/F1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402

KINDS = ("prefill", "judge", "denoise", "kvflush")


def mean(xs):
    v = [float(x) for x in xs if x is not None and x == x]
    return sum(v) / len(v) if v else None


def read_records(d: str):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.shard*of*.jsonl"))):
        if f.endswith(".running.jsonl"):                # the running metrics, not records
            continue
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                recs[(r.get("split"), int(r["idx"]))] = r
    return list(recs.values())


def drafter_reliability(trace_dir: str, max_items: int = 2000):
    """Drafter confidence vs correctness (drafted token == the 30B's final token), over valid slots."""
    cnt, sconf, sok = np.zeros(10), np.zeros(10), np.zeros(10)
    n_items = 0
    for f in sorted(glob.glob(os.path.join(trace_dir, "*", "*.npz")))[:max_items]:
        z = np.load(f)
        if "p_draft" not in z.files:
            continue
        dt, pd, tok, va = z["draft_tokens"], z["p_draft"], z["tokens"], z["valid"].astype(bool)
        m = va & (dt >= 0) & np.isfinite(pd)
        if not m.any():
            continue
        c = pd[m].astype(np.float64)
        ok = (dt[m] == tok[m]).astype(np.float64)
        b = np.clip((c * 10).astype(int), 0, 9)
        np.add.at(cnt, b, 1); np.add.at(sconf, b, c); np.add.at(sok, b, ok)
        n_items += 1
    tot = cnt.sum()
    if not tot:
        return None
    return {"items": n_items, "slots": int(tot), "mean_confidence": float(sconf.sum() / tot),
            "accuracy": float(sok.sum() / tot), "ece": float(np.abs(sconf - sok).sum() / tot),
            "bins": [[round(i / 10, 1), int(c), (float(s / c) if c else None), (float(o / c) if c else None)]
                     for i, (c, s, o) in enumerate(zip(cnt, sconf, sok))]}


def set_report(recs: list, set_dir: str) -> dict:
    out = {"n_items": len(recs)}
    gen = sum(int(r.get("n_gen_to_eos", 0)) for r in recs)
    out["exp_per_token"] = sum(int(r.get("unique_expert_layer_loads", 0)) for r in recs) / gen if gen else None
    out["charged_exp_per_token"] = mean(r.get("charged_exp_per_token") for r in recs)
    out["exp_per_fwd_per_layer_block"] = mean(r.get("exp_uniq_per_layer_forward_block") for r in recs)
    out["exp_union_per_layer_item"] = mean(r.get("exp_union_per_layer_item") for r in recs)
    kinds = {}
    for k in KINDS:
        v = [r["exp_by_kind"][k] for r in recs if isinstance(r.get("exp_by_kind"), dict) and k in r["exp_by_kind"]]
        if v:
            kinds[k] = {key: mean(x.get(key) for x in v) for key in
                        ("n_forwards", "per_fwd_uniq_per_layer", "per_fwd_uniq_total", "uniq_per_layer", "entropy",
                         "max_share", "top8_mass")}
            kinds[k]["per_fwd_uniq_per_layer_max"] = max((x.get("per_fwd_uniq_per_layer_max") or 0) for x in v)
    out["by_kind"] = kinds
    prof = [r["exp_per_layer_profile_block"] for r in recs if r.get("exp_per_layer_profile_block")]
    out["per_layer_profile_block"] = [round(float(v), 2) for v in np.mean(np.array(prof), 0)] if prof else None
    top = Counter()
    for r in recs:
        for layer, e, c in r.get("exp_top10") or []:
            top[(layer, e)] += c
    out["top20_layer_expert"] = [[l, e, c] for (l, e), c in top.most_common(20)]
    jp = defaultdict(list); jc = defaultdict(list)
    for r in recs:
        for i, v in enumerate(r.get("judge_p_mean_by_pass") or []):
            jp[i].append(v)
        for i, v in enumerate(r.get("judge_commit_frac_by_pass") or []):
            jc[i].append(v)
    out["judge_by_pass"] = [{"pass": i + 1, "mean_top1_prob": mean(jp[i]), "commit_frac_above_0.95": mean(jc[i]),
                             "items": len(jp[i])} for i in sorted(jp)]
    out["judge_passes_per_block"] = mean(
        (int(r["n_forward"]) - int(r["n_prefill"]) - int(r["n_flush"])) / max(1, int(r["n_blocks"])) for r in recs)
    out["drafter_reliability"] = drafter_reliability(os.path.join(set_dir, "traces"))
    sc = [int(r["correct"]) for r in recs if r.get("correct") is not None]
    out["accuracy"] = sum(sc) / len(sc) if sc else None
    return out


def fmt(x, n=2):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--root", default="logs/phase2/decode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    base = P.safe_out(os.path.join(args.root, args.arm))
    rep = {"arm": args.arm, "sets": {}}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(d):
            recs = read_records(d)
            if recs:
                rep["sets"][os.path.basename(d)] = set_report(recs, d)
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "experts.json"), "w"), indent=1)
    md = [f"# Experts, router and probabilities: {args.arm}", ""]
    md += ["## Per set", "",
           "| set | items | accuracy | experts/token | charged experts/token | experts per layer, one block forward "
           "| union per layer per item | judge passes / block |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, r in rep["sets"].items():
        md.append(f"| {s} | {r['n_items']} | {fmt(r['accuracy'], 3)} | {fmt(r['exp_per_token'], 1)} | "
                  f"{fmt(r['charged_exp_per_token'], 1)} | {fmt(r['exp_per_fwd_per_layer_block'])} | "
                  f"{fmt(r['exp_union_per_layer_item'])} | {fmt(r['judge_passes_per_block'])} |")
    md += ["", "## Experts by forward kind (averaged over items)", "",
           "| set | kind | forwards / item | experts per layer, one forward (mean / max) | experts per forward, 48 layers "
           "| union per layer per item | load entropy | max-load share | router top-8 mass |",
           "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for s, r in rep["sets"].items():
        for k, v in r["by_kind"].items():
            md.append(f"| {s} | {k} | {fmt(v['n_forwards'], 1)} | {fmt(v['per_fwd_uniq_per_layer'])} / "
                      f"{v['per_fwd_uniq_per_layer_max']} | {fmt(v['per_fwd_uniq_total'], 1)} | {fmt(v['uniq_per_layer'])} | "
                      f"{fmt(v['entropy'], 3)} | {fmt(v['max_share'], 3)} | {fmt(v['top8_mass'], 3)} |")
    md += ["", "## Judge probabilities by pass", "", "| set | pass | mean top-1 prob | share > 0.95 | items |",
           "|---|---:|---:|---:|---:|"]
    for s, r in rep["sets"].items():
        for j in r["judge_by_pass"][:6]:
            md.append(f"| {s} | {j['pass']} | {fmt(j['mean_top1_prob'], 3)} | {fmt(j['commit_frac_above_0.95'], 3)} | "
                      f"{j['items']} |")
    md += ["", "## Drafter confidence vs correctness (drafted token == the 30B's final token)", "",
           "| set | slots | mean confidence | accuracy | ECE |", "|---|---:|---:|---:|---:|"]
    for s, r in rep["sets"].items():
        d = r["drafter_reliability"]
        if d:
            md.append(f"| {s} | {d['slots']} | {fmt(d['mean_confidence'], 3)} | {fmt(d['accuracy'], 3)} | {fmt(d['ece'], 3)} |")
    md += ["", "## Most loaded (layer, expert) pairs in block forwards (top 10 per set)", ""]
    for s, r in rep["sets"].items():
        md.append(f"- **{s}**: " + ", ".join(f"L{l}/E{e} ({c})" for l, e, c in r["top20_layer_expert"][:10]))
    md += ["", "Per-layer profiles and reliability bins: experts.json. Raw data: decode/<arm>/<set>/experts, judge, traces."]
    open(os.path.join(out, "experts.md"), "w").write("\n".join(md) + "\n")
    print(f"[report] {args.arm}: {len(rep['sets'])} sets -> {out}/experts.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
