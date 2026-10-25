#!/usr/bin/env python
"""Phase 2 verdict for one drafter arm against S0, by the rules fixed in rules.json (pass_fail).

Reads, for each benchmark, the arm's decode records and the S0 records (same items, same judge adapter unless
the arm was retrained, same tau / pool), plus stock records for the speedup, the re-draft summary (agreement) and
the probe. Writes <out>/gate.json and <out>/gate.md:

    per set: n, accuracy (arm, S0), paired delta [95% CI], experts/token, drafter forwards/token,
             charged experts/token (each drafter at its OWN cost per forward), charged speedup vs stock (arm, S0),
             passes/block, draft agreement, stopped shards
    verdicts: drafter_fits, alignment_hurt (in-domain sets or probe precision), retrain_needed,
              cross_domain_gain (MATH-500 near-domain, GPQA out of domain), size_pays_off

    /venv/train/bin/python code/phase2/gate_p2.py --arm F1 --arm-dir logs/phase2/decode/F1 \
        --s0-dir logs/phase2/decode/S0 --stock-dir logs/phase2/decode/stock \
        --redraft logs/phase2/redraft/F1/summary.json --probe logs/phase2/probe/F1__r3_B/probe.json \
        --out logs/phase2/gate/F1_zeroshot
Every arm is reported, stopped or not (rules.json: "report every run").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

from phase2 import drafters as P2D                                       # noqa: E402
from phase2.stoprules import load_rules, paired_delta                    # noqa: E402

SETS = ("humaneval", "mbpp", "gsm8k500", "math500", "gpqa", "ifeval541", "aime")


def load_set(roots: str, split: str):
    """Records of `split` under the first of the comma-separated `roots` that has any (<root>/<cell dir>/<records>)
    -> ({idx: rec}, stopped shard count). A local rerun listed first wins over a downloaded reference."""
    for root in [r for r in roots.split(",") if r]:
        recs, stopped = _load_one(P.safe_out(root), split)
        if recs:
            return recs, stopped
    return {}, 0


def _load_one(root: str, split: str):
    recs, stopped = {}, 0
    for f in sorted(glob.glob(os.path.join(root, "*", f"*.{split}.shard*of*.jsonl"))):
        if f.endswith(".running.jsonl"):                # the running metrics, not records
            continue
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                recs[int(r["idx"])] = r
        stopped += int(os.path.exists(f + ".stopped"))
    return recs, stopped


def agg(recs: dict, eq: float) -> dict:
    R = list(recs.values())
    if not R:
        return {"n": 0}
    gen = sum(int(r["n_gen_to_eos"]) for r in R)
    uniq = sum(int(r["unique_expert_layer_loads"]) for r in R)
    n_dr = sum(sum((r.get("drafter_n_by_kind") or {}).values()) for r in R)
    blocks = sum(int(r["n_blocks"]) for r in R)
    passes = sum(int(r["n_forward"]) - int(r["n_prefill"]) - int(r["n_flush"]) for r in R)
    sc = [int(r["correct"]) for r in R if r.get("correct") is not None]
    ag = [a for r in R for a in r.get("draft_agreement_per_block", []) if a is not None]
    return {"n": len(R), "acc": sum(sc) / len(sc) if sc else None, "exp_tok": uniq / gen if gen else None,
            "drafter_fwd_tok": n_dr / gen if gen else None,
            "charged_exp_tok": (uniq + n_dr * eq) / gen if gen else None,
            "passes_per_block": passes / blocks if blocks else None,
            "draft_agreement": sum(ag) / len(ag) if ag else None,
            "ms_per_token": 1000.0 * sum(r["seconds"] for r in R) / gen if gen else None,
            "exp_per_fwd_per_layer_block": _mean(r.get("exp_uniq_per_layer_forward_block") for r in R),
            "exp_union_per_layer_item": _mean(r.get("exp_union_per_layer_item") for r in R),
            "judge_p_mean": _mean(r.get("judge_p_mean") for r in R)}


def _mean(xs):
    v = [float(x) for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="phase2.drafters key of the arm")
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--s0-dir", required=True, help="comma list of roots; the first holding a set's records is used")
    ap.add_argument("--stock-dir", default=None, help="comma list of roots, as --s0-dir")
    ap.add_argument("--redraft", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    args = ap.parse_args(argv)
    rules = load_rules()
    pf = rules["pass_fail"]
    bs = pf["bootstrap"]
    eq_arm, eq_s0 = P2D.cost_per_forward(args.arm), P2D.cost_per_forward("S0")
    rows, verdict = {}, {"arm": args.arm, "rules_sha256": rules["_sha256"]}
    for s in args.sets.split(","):
        a, a_stop = load_set(args.arm_dir, s)
        b, b_stop = load_set(args.s0_dir, s)
        st, _ = load_set(args.stock_dir, s) if args.stock_dir else ({}, 0)
        ra, rb, rs = agg(a, eq_arm), agg(b, eq_s0), agg(st, 0.0)
        shared = sorted(i for i in a if i in b and a[i].get("correct") is not None and b[i].get("correct") is not None)
        d = paired_delta([int(a[i]["correct"]) for i in shared], [int(b[i]["correct"]) for i in shared],
                         resamples=bs["resamples"], seed=bs["seed"], ci=bs["ci"])
        row = {"arm": ra, "s0": rb, "stock": rs, "paired": d, "stopped_shards": a_stop}
        if rs.get("exp_tok"):
            for k, r in (("arm", ra), ("s0", rb)):
                if r.get("charged_exp_tok"):
                    row[f"{k}_speedup_charged"] = rs["exp_tok"] / r["charged_exp_tok"]
                    row[f"{k}_speedup_exp"] = rs["exp_tok"] / r["exp_tok"]
        rows[s] = row
    ind = [s for s in pf["in_domain_sets"] if s in rows and rows[s]["paired"]["n"]]
    hurt_sets = [s for s in ind if rows[s]["paired"]["hi"] < 0]
    probe = json.load(open(P.safe_out(args.probe))) if args.probe and os.path.exists(P.safe_out(args.probe)) else None
    prec = probe.get("all/conf_precision") if probe else None
    floor = pf["alignment_hurt"]["commit_precision_floor"]
    verdict["alignment_hurt"] = bool(hurt_sets) or (prec is not None and prec < floor)
    verdict["alignment_hurt_sets"] = hurt_sets
    verdict["probe_conf_precision"] = prec
    verdict["retrain_needed"] = bool(hurt_sets)
    gains = [s for s in pf["out_of_mix_sets"] if s in rows and rows[s]["paired"]["n"] and rows[s]["paired"]["lo"] > 0]
    verdict["cross_domain_gain_sets"] = gains
    verdict["out_of_domain_gain"] = any(s in pf["out_of_domain_sets"] for s in gains)
    pays = {s: (rows[s].get("arm_speedup_charged"), rows[s].get("s0_speedup_charged")) for s in rows}
    verdict["size_pays_off_sets"] = [s for s, (x, y) in pays.items() if x and y and x > y and s not in hurt_sets]
    if args.redraft and os.path.exists(P.safe_out(args.redraft)):
        rd = json.load(open(P.safe_out(args.redraft)))["per_domain"].get("all", {})
        verdict["agreement"], verdict["prev_agreement"] = rd.get("agreement"), rd.get("prev_agreement")
        r = rd.get("ratio_to_prev")
        verdict["drafter_fits"] = (r is not None and r >= pf["drafter_fits"]["min_ratio_to_s0"])
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump({"verdict": verdict, "sets": rows}, open(os.path.join(out, "gate.json"), "w"), indent=2)
    f = lambda x, n=3: "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))   # noqa: E731
    md = [f"# Phase 2 gate: {args.arm} vs S0", "", f"rules sha256 `{rules['_sha256'][:12]}`", "",
          "| set | n | acc arm | acc S0 | Δ [95% CI] | exp/tok arm | charged exp/tok arm | charged speedup arm / S0 | "
          "passes/blk | agreement | stopped shards |", "|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|"]
    for s, r in rows.items():
        p_ = r["paired"]
        md.append(f"| {s} | {p_['n']} | {f(r['arm'].get('acc'))} | {f(r['s0'].get('acc'))} | "
                  f"{f(p_['delta'])} [{f(p_['lo'])}, {f(p_['hi'])}] | {f(r['arm'].get('exp_tok'), 1)} | "
                  f"{f(r['arm'].get('charged_exp_tok'), 1)} | {f(r.get('arm_speedup_charged'), 2)} / "
                  f"{f(r.get('s0_speedup_charged'), 2)} | {f(r['arm'].get('passes_per_block'), 2)} | "
                  f"{f(r['arm'].get('draft_agreement'))} | {r['stopped_shards']} |")
    md += ["", "## Experts and judge (arm / S0)", "",
           "| set | experts per layer, one block forward | union per layer per item | judge mean top-1 prob |",
           "|---|---|---|---|"]
    for s, r in rows.items():
        a_, b_ = r["arm"], r["s0"]
        md.append(f"| {s} | {f(a_.get('exp_per_fwd_per_layer_block'), 2)} / {f(b_.get('exp_per_fwd_per_layer_block'), 2)} | "
                  f"{f(a_.get('exp_union_per_layer_item'), 2)} / {f(b_.get('exp_union_per_layer_item'), 2)} | "
                  f"{f(a_.get('judge_p_mean'))} / {f(b_.get('judge_p_mean'))} |")
    md += ["", "S0 values are '—' where the S0 reference is round 3's records (they predate the expert and judge logs).",
           "", "## Verdict", "", "```json", json.dumps(verdict, indent=2), "```"]
    open(os.path.join(out, "gate.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
