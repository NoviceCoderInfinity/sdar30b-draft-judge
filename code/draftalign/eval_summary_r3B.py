#!/usr/bin/env python
"""r3_B final-checkpoint eval summary on the FULL sets, against stock, stock + DES, blind accept and r3_A epochs 1/2.

    /venv/r3/bin/python code/draftalign/eval_summary_r3B.py [--step step000710] [--print]
        -> logs/draftalign/probes_r3/r3_B/eval_summary.{json,md}

Reuses eval_summary_r3.py's readers and metrics unchanged (paired Δacc on shared idx, bootstrap CI with 2,000
resamples, AGENTS.md §8 passes/blk and exp/token). Code sets count as scored only through `scored/` shards.
Baselines come from whichever source holds the most records for the cell: the round-3 local dir
(probes_r3/r3_A/stock/<ds>/<arm>), the team mirror (da_decode), or research_v1/baselines_b32.
r3_A epoch rows exist only where the eval wave ran them (HumanEval, MBPP full; GSM8K first 100; MATH first 50),
so their Δ is paired on those items. exp/token counts the 30B's expert loads only (the drafter is not charged).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary_r3 as S  # noqa: E402

ROOT = S.ROOT
PRA = S.PR
PRB = os.path.join(ROOT, "logs", "draftalign", "probes_r3", "r3_B")
BASES = [S.MIRROR, "/dev/shm/dlm/artifacts_mirror/logs/research_v1/baselines_b32"]
# key, label, n, code set, mirror family, mirror split
DATASETS = [("humaneval", "HumanEval", 164, True, "humaneval", "humaneval"),
            ("mbpp", "MBPP (sanitized)", 257, True, "mbpp", "mbpp"),
            ("gsm8k500", "GSM8K-500", 500, False, "gsm8k", "gsm8k500"),
            ("math500", "MATH-500", 500, False, "math500", "math500"),
            ("gpqa", "GPQA-Diamond", 198, False, "gpqa", "gpqa"),
            ("aime", "AIME", 90, False, "aime", "aime")]
LABEL = {"stock": "stock", "stock_des32": "stock + DES32", "stock_des48": "stock + DES48",
         "blind": "blind accept (no training)", "prev_judge_des48": "round-2 judge + DES48 (T2b 3057)",
         "r3A_e1_judge_des48": "r3_A epoch 1 judge + DES48", "r3A_e2_judge_des48": "r3_A epoch 2 judge + DES48",
         "r3A_s2000_judge_des48": "r3_A last ckpt (step 2000, 2.82 ep) judge + DES48",
         "r3B_judge": "r3_B epoch 2 judge", "r3B_judge_des48": "r3_B epoch 2 judge + DES48"}
EFF_KEYS = ("n_forward", "n_prefill", "n_flush", "unique_expert_layer_loads", "n_blocks", "n_gen_to_eos")
_eff = S.eff
S.eff = lambda recs: _eff([r for r in recs if all(k in r for k in EFF_KEYS)])   # older records may lack a field


def load_cell(cell, split):
    """Team records for `cell`/`split` from every base; the source with the most records wins."""
    best = {"recs": {}, "scored": {}, "source": "none", "shards_done": 0, "shards_total": 0,
            "n_scored_shards": 0, "score_errors": 0, "last_utc": None, "path": cell}
    for base in BASES:
        files = sorted(f for f in glob.glob(os.path.join(base, cell, f"{cell}.{split}.shard*.jsonl"))
                       + glob.glob(os.path.join(base, f"{cell}.{split}.shard*.jsonl")) if ".oldcfg" not in f)
        recs = {}
        for f in files:
            for r in S.read_jsonl(f):
                if "idx" in r:
                    recs[int(r["idx"])] = r
        if len(recs) > len(best["recs"]):
            best = {"recs": recs, "scored": {i: c for i, r in recs.items() if (c := S.correct_of(r)) is not None},
                    "source": "mirror", "shards_done": sum(os.path.exists(f + ".done") for f in files),
                    "shards_total": len(files), "n_scored_shards": 0, "score_errors": 0, "last_utc": None,
                    "path": f"{os.path.basename(base)}/{cell} ({split})"}
    return best


def latest_step(run_dir):
    steps = sorted(os.path.basename(d) for d in glob.glob(os.path.join(run_dir, "step*")) if os.path.isdir(d))
    return steps[-1] if steps else None


def build(run_dir, step):
    now = dt.datetime.now(dt.timezone.utc)
    res = {"generated_ist": now.astimezone(S.IST).strftime("%Y-%m-%d %H:%M IST"), "run_dir": os.path.relpath(run_dir, ROOT),
           "step": step, "labels": LABEL, "datasets": {}}
    for key, label, n, code, fam, split in DATASETS:
        arms = {}
        for M, suf in S.POOLS:
            local = S.load_arm_dir(os.path.join(PRA, "stock", key, f"stock{suf}"), code)
            team = load_cell(f"{fam}_B32_t95_stock{S.MIRROR_POOL[M]}", split)
            arms[f"stock{suf}"] = local if local["recs"] and len(local["recs"]) >= len(team["recs"]) else team
        arms["blind"] = load_cell(f"{fam}_B32_t95_dv-s8_blind_q50", split)
        arms["prev_judge_des48"] = load_cell(f"{fam}_B32_t95_dv-s8_full_c95_p48desf{S.PREV_TAG}", split)
        arms["r3A_e1_judge_des48"] = S.load_arm_dir(os.path.join(PRA, "step000700", key, "judge_des48"), code)
        arms["r3A_e2_judge_des48"] = S.load_arm_dir(os.path.join(PRA, "step001400", key, "judge_des48"), code)
        arms["r3A_s2000_judge_des48"] = S.load_arm_dir(os.path.join(PRA, "step002000", key, "judge_des48"), code)
        for nm in ("judge", "judge_des48"):
            arms[f"r3B_{nm}"] = S.load_arm_dir(os.path.join(run_dir, step or "-", key, nm), code)
        ref = sorted(arms["stock"]["recs"])
        universe = ref[:n] if len(ref) >= n else None
        arms = {k: S.restrict(v, universe) for k, v in arms.items()}
        rows = {}
        for nm, a in arms.items():
            c = S.cell_summary(a, n, code and a["source"] != "mirror")
            c["vs_stock"] = S.compare(a, arms["stock"]) if nm != "stock" else None
            c["vs_stock_des48"] = S.compare(a, arms["stock_des48"]) if nm not in ("stock", "stock_des48") else None
            rows[nm] = c
        res["datasets"][key] = {"label": label, "n": n, "universe": len(universe) if universe else None, "rows": rows}
    return res


def to_markdown(res):
    L = [f"# r3_B eval summary — `{res['run_dir']}/{res['step']}` (full sets)", "",
         f"Generated {res['generated_ist']}. Δ = paired on shared items, 95% bootstrap CI. exp/token = 30B expert loads per token "
         "(drafter not charged); speedup = stock exp/token ÷ arm exp/token on the shared items.", ""]
    hdr = "| arm | status | n scored / decoded | acc % | Δacc vs stock [CI] | Δacc vs stock + DES48 [CI] | passes/blk | exp/token | speedup vs stock |"
    for key, d in res["datasets"].items():
        L += [f"## {d['label']} (n = {d['n']})", "", hdr, "|---|---|---:|---:|---|---|---:|---:|---:|"]
        for nm, c in d["rows"].items():
            v = c["vs_stock"]
            L.append(f"| {LABEL[nm]} | {S.status_txt(c)} | {c['n_scored']} / {c['n_records']} | {S.f(c['acc'])} | "
                     f"{'ref' if nm == 'stock' else S.dacc(v)} | {'ref' if nm == 'stock_des48' else ('—' if nm == 'stock' else S.dacc(c['vs_stock_des48']))} | "
                     f"{S.f(c['passes_blk'], 2)} | {S.f(c['exp_tok'])} | "
                     f"{S.f(v['speedup_exp'], 2, suffix='×') if v else ('1.00×' if nm == 'stock' and c['n_records'] else '—')} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=PRB, help="probes dir of the run (default probes_r3/r3_B)")
    ap.add_argument("--step", default=None, help="checkpoint dir name under --run-dir (default: latest)")
    ap.add_argument("--out-dir", default=None, help="default = --run-dir")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    step = args.step or latest_step(args.run_dir)
    res = build(args.run_dir, step)
    out = args.out_dir or args.run_dir
    os.makedirs(out, exist_ok=True)
    for name, text in (("eval_summary.json", json.dumps(res, indent=1)), ("eval_summary.md", to_markdown(res))):
        tmp = os.path.join(out, f".{name}.tmp")
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, os.path.join(out, name))
    print(f"[eval_summary_r3B] {res['generated_ist']} step {step} -> {out}/eval_summary.{{json,md}}")
    if args.print:
        sys.stdout.write(to_markdown(res))


if __name__ == "__main__":
    main()
