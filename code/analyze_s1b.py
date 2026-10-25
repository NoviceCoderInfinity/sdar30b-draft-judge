#!/usr/bin/env python
"""Stage 1 Arm B analysis: per-item scoring, paired bootstrap CIs, verdicts, plots.

Scoring uses the vendored opencompass path (as P1 did):
  gsm8k    MATHEvaluator(v2).is_equiv on math_postprocess_v2(pred)
           vs gsm8k_dataset_postprocess(gold)
  math500  MATHEvaluator(v2).is_equiv on math_postprocess_v2(pred) vs the
           dataset's own `answer` field. NOT comparable to any published number:
           no SDAR config for MATH-500 was gated in P1. Relative across M only.
  mbpp     MBPPEvaluator (executes the asserts); per-item `is_correct` from its
           own `details` dict.

Pass criterion (set 2026-09-03). Mask-vote and draft-vote run the SAME items, so
the comparison is paired: per item d_i = correct_draft_i - correct_mask_i, and a
95% CI on mean(d) by bootstrap over items, 10,000 resamples.
  WIN  CI excludes 0 in favour of draft-vote
  TIE  CI includes 0
  LOSS CI excludes 0 against
Arm B passes iff draft-vote WINS **pooled** at M=16 and M=24, and shows no LOSS
on any dataset at any M <= 32. M=48, 64 and unrestricted are curve shape only.

Bytes are totals over the WHOLE block decode, never step-1 alone:
  MoE      = sum over steps of (unique experts per layer) x 9.44 MB, / n_blocks
  drafter  = 3.4 GB once per block          (RESIDENT -- headline)
             4 x 3.4 GB per block           (CONSERVATIVE -- reported alongside)
"""
from __future__ import annotations
import re

import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oc_datasets.math import math_postprocess_v2, MATHEvaluator      # noqa: E402
from oc_datasets.gsm8k import gsm8k_dataset_postprocess              # noqa: E402
from oc_datasets.mbpp import MBPPEvaluator                           # noqa: E402

EXPERT_MB, DRAFTER_GB, DRAFT_FWD = 9.44, 3.4, 4
RES = os.environ.get("S1B_RESULTS_DIR", "results")
PLOTS = f"{RES}/plots"
PASS1_M = [16, 24, 32]
ALL_M = [16, 24, 32, 48, 64]
N_BOOT = 10000


# ---------------------------------------------------------------------------
# per-item scoring
# ---------------------------------------------------------------------------
def per_item_correct(ds, rows):
    """-> dict idx -> 0/1"""
    if ds == "ifeval":
        from draftalign.if_scorer import score_ifeval_record     # strict IFEvalG, every constraint must pass
        return {x["idx"]: score_ifeval_record(x) for x in rows}
    if ds == "mbpp":
        r = MBPPEvaluator().score([x["pred"] for x in rows], [x["gold"] for x in rows])
        det = r.get("details", {})
        return {rows[i]["idx"]: int(bool(det[str(i)]["is_correct"]))
                for i in range(len(rows)) if str(i) in det}
    if ds == "gpqa":
        out = {}
        for x in rows:
            out[x["idx"]] = int(_gpqa_letter(x["pred"]) == str(x["gold"]).strip().upper())
        return out
    if ds == "humaneval":
        return _humaneval_score(rows)
    if ds == "chat":
        # corpus-1b held-out chat800: no gold, no correctness. `int(None)` raises inside both runners'
        # `rec["correct"] = int(scorer(rec))`, which they already catch (`except Exception: rec["correct"]
        # = None; rec["score_error"] = ...`) -- the net effect is exactly `correct: None`, no crash.
        return {x["idx"]: None for x in rows}
    if ds == "finqa":
        return {x["idx"]: _finqa_score(x["pred"], x["gold"]) for x in rows}
    if ds == "chartqa":
        return {x["idx"]: _chartqa_score(x["pred"], x["gold"]) for x in rows}
    ev = MATHEvaluator(version="v2")
    out = {}

    def _aime_canon(v):
        """AIME answers are integers 0-999 conventionally written 0-padded to 3 digits.
        Canonicalise both sides so 033 == 33; leaves anything non-numeric untouched."""
        t = str(v).strip()
        return str(int(t)) if t.isdigit() else t

    for x in rows:
        p = math_postprocess_v2(x["pred"])
        g = gsm8k_dataset_postprocess(x["gold"]) if ds == "gsm8k" else x["gold"]
        if ds == "aime":
            p, g = _aime_canon(p), _aime_canon(g)
        try:
            ok = bool(ev.is_equiv(p, g))
        except Exception:
            ok = False
        out[x["idx"]] = int(ok)
    return out


def boot_ci(d, n_boot=N_BOOT, seed=0):
    """95% bootstrap CI on mean(d), in accuracy POINTS."""
    d = np.asarray(d, float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1) * 100.0
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            float(d.mean() * 100.0))


def verdict(ci):
    if ci is None:
        return "n/a"
    lo, hi, _ = ci
    if lo > 0:
        return "WIN"
    if hi < 0:
        return "LOSS"
    return "TIE"


# ---------------------------------------------------------------------------
def main(logdir):
    cells = defaultdict(list)
    for f in sorted(glob.glob(f"{logdir}/*.jsonl")):
        if not os.path.exists(f + ".done"):
            continue
        base = os.path.basename(f)[:-len(".jsonl")].split(".")[0]
        ds, vote, m = base.split("_")
        cells[(ds, vote, int(m[1:]))] += [json.loads(l) for l in open(f) if l.strip()]

    stats, correct = {}, {}
    for key in sorted(cells):
        ds, vote, M = key
        rows = sorted(cells[key], key=lambda r: r["idx"])
        c = per_item_correct(ds, rows)
        correct[key] = c
        u = sum(r["unique_expert_layer_loads"] for r in rows)
        lf = sum(r["layer_forwards"] for r in rows)
        nb = sum(r["n_blocks"] for r in rows)
        cps = [x for r in rows for x in r["committed_per_step"]]
        spb = [x for r in rows for x in r["steps_per_block"]]
        secs = sum(r["seconds"] for r in rows)
        moe_gb = u * EXPERT_MB / 1024 / nb
        dres = DRAFTER_GB if vote == "draft" else 0.0
        dcon = DRAFT_FWD * DRAFTER_GB if vote == "draft" else 0.0
        stats[key] = {
            "dataset": ds, "vote": vote, "M": M, "n": len(rows),
            "accuracy": round(100.0 * sum(c.values()) / max(len(c), 1), 2),
            "tokens_per_step": round(float(np.mean(cps)), 3),
            "steps_per_block": round(float(np.mean(spb)), 2),
            "unique_experts_per_forward": round(u / lf, 2),
            "moe_GB_per_block": round(moe_gb, 2),
            "total_GB_resident": round(moe_gb + dres, 2),
            "total_GB_conservative": round(moe_gb + dcon, 2),
            "seconds_per_item": round(secs / len(rows), 2),
            "ms_per_moe_forward": round(
                1000 * secs / sum(r["n_moe_forward"] for r in rows), 1),
        }

    # ---- paired comparisons ----
    paired, pooled_d = {}, defaultdict(list)
    for M in ALL_M:
        for ds in sorted({k[0] for k in stats}):
            a, b = correct.get((ds, "draft", M)), correct.get((ds, "mask", M))
            if not a or not b:
                continue
            common = sorted(set(a) & set(b))
            d = [a[i] - b[i] for i in common]
            ci = boot_ci(d)
            paired[f"{ds}|M{M}"] = {
                "dataset": ds, "M": M, "n_paired": len(common),
                "acc_draft": round(100 * np.mean([a[i] for i in common]), 2),
                "acc_mask": round(100 * np.mean([b[i] for i in common]), 2),
                "delta_points": round(ci[2], 2),
                "ci95": [round(ci[0], 2), round(ci[1], 2)],
                "verdict": verdict(ci)}
            pooled_d[M] += d
    for M, d in pooled_d.items():
        ci = boot_ci(d)
        paired[f"POOLED|M{M}"] = {
            "dataset": "POOLED", "M": M, "n_paired": len(d),
            "delta_points": round(ci[2], 2),
            "ci95": [round(ci[0], 2), round(ci[1], 2)],
            "verdict": verdict(ci)}

    # ---- cross-M: "equal accuracy at a smaller M" ----
    cross = {}
    for ds in sorted({k[0] for k in stats}):
        for Md in ALL_M:
            for Mm in ALL_M:
                if Md >= Mm:
                    continue
                a, b = correct.get((ds, "draft", Md)), correct.get((ds, "mask", Mm))
                if not a or not b:
                    continue
                common = sorted(set(a) & set(b))
                ci = boot_ci([a[i] - b[i] for i in common])
                cross[f"{ds}|draft@M{Md}_vs_mask@M{Mm}"] = {
                    "dataset": ds, "M_draft": Md, "M_mask": Mm,
                    "acc_draft": round(100 * np.mean([a[i] for i in common]), 2),
                    "acc_mask": round(100 * np.mean([b[i] for i in common]), 2),
                    "delta_points": round(ci[2], 2),
                    "ci95": [round(ci[0], 2), round(ci[1], 2)],
                    "verdict": verdict(ci)}

    # ---- overall pass ----
    # The gate is only meaningful once ALL THREE datasets have contributed every
    # pass-1 cell; a pooled CI over one dataset is not the pooled CI.
    EXPECTED_DS = {"mbpp", "gsm8k", "math500"}
    complete = all((ds, vote, M) in stats
                   for ds in EXPECTED_DS for vote in ("mask", "draft")
                   for M in PASS1_M)
    pooled_win = all(paired.get(f"POOLED|M{M}", {}).get("verdict") == "WIN"
                     for M in (16, 24) if f"POOLED|M{M}" in paired)
    have_both = complete and all(f"POOLED|M{M}" in paired for M in (16, 24))
    any_loss = [k for k, v in paired.items()
                if v["dataset"] != "POOLED" and v["M"] <= 32 and v["verdict"] == "LOSS"]
    if not have_both:
        missing = sorted({ds for ds in EXPECTED_DS for vote in ("mask", "draft")
                          for M in PASS1_M if (ds, vote, M) not in stats})
        overall = f"INCOMPLETE (awaiting: {', '.join(missing)})"
    else:
        overall = "PASS" if (pooled_win and not any_loss) else "FAIL"

    out = {"cells": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in stats.items()},
           "paired": paired, "cross_M": cross,
           "arm_b_verdict": overall,
           "losses_at_M_le_32": any_loss,
           "notes": {
               "math500": "not comparable to published; relative across M only",
               "bytes": "totals over the whole block decode; resident headlines",
               "criterion": "pooled WIN at M=16 and M=24, no dataset LOSS at M<=32"}}
    json.dump(out, open(f"{RES}/S1B_metrics.json", "w"), indent=2)

    # ---- print ----
    hdr = (f"{'dataset':9}{'vote':7}{'M':>4}{'n':>5}{'acc':>8}{'tok/stp':>8}"
           f"{'stp/blk':>8}{'uniq/fwd':>9}{'GB res':>8}{'GB cons':>8}{'s/item':>8}")
    print(hdr); print("-" * len(hdr))
    for k in sorted(stats, key=lambda k: (k[0], k[2], k[1])):
        r = stats[k]
        print(f"{r['dataset']:9}{r['vote']:7}{r['M']:>4}{r['n']:>5}{r['accuracy']:>8.2f}"
              f"{r['tokens_per_step']:>8.2f}{r['steps_per_block']:>8.2f}"
              f"{r['unique_experts_per_forward']:>9.2f}{r['total_GB_resident']:>8.2f}"
              f"{r['total_GB_conservative']:>8.2f}{r['seconds_per_item']:>8.1f}")

    print(f"\n{'comparison':18}{'n':>6}{'draft':>8}{'mask':>8}{'delta':>8}"
          f"{'95% CI':>18}  verdict")
    print("-" * 74)
    for M in ALL_M:
        for k in [k for k in paired if paired[k]["M"] == M]:
            v = paired[k]
            ad = f"{v['acc_draft']:.2f}" if "acc_draft" in v else "  -  "
            am = f"{v['acc_mask']:.2f}" if "acc_mask" in v else "  -  "
            ci = f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
            print(f"{k:18}{v['n_paired']:>6}{ad:>8}{am:>8}"
                  f"{v['delta_points']:>+8.2f}{ci:>18}  {v['verdict']}")
    if cross:
        print(f"\ncross-M (draft at a SMALLER coreset vs mask at a larger one)")
        print(f"{'comparison':38}{'draft':>8}{'mask':>8}{'delta':>8}{'95% CI':>18}  verdict")
        print("-" * 88)
        for k, v in sorted(cross.items(), key=lambda kv: (kv[1]["dataset"], kv[1]["M_mask"], kv[1]["M_draft"])):
            ci = f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
            print(f"{k:38}{v['acc_draft']:>8.2f}{v['acc_mask']:>8.2f}"
                  f"{v['delta_points']:>+8.2f}{ci:>18}  {v['verdict']}")

    print(f"\nARM B: {overall}" + (f"  (losses: {any_loss})" if any_loss else ""))

    # ---- figures ----
    os.makedirs(PLOTS, exist_ok=True)
    dss = sorted({k[0] for k in stats})
    if not dss:
        return
    fig, axes = plt.subplots(2, len(dss), figsize=(4.7 * len(dss), 8.2), squeeze=False)
    col = {"mask": "#C3626B", "draft": "#2D6E8E"}
    for j, ds in enumerate(dss):
        a1, a2 = axes[0][j], axes[1][j]
        unr = stats.get((ds, "none", 0))
        for vote in ("mask", "draft"):
            pts = [(M, stats[(ds, vote, M)]) for M in ALL_M if (ds, vote, M) in stats]
            if not pts:
                continue
            a1.plot([p[0] for p in pts], [p[1]["accuracy"] for p in pts],
                    marker="o", lw=2, color=col[vote], label=f"{vote}-vote")
            a2.plot([p[1]["total_GB_resident"] for p in pts],
                    [p[1]["accuracy"] for p in pts], marker="o", lw=2,
                    color=col[vote], label=f"{vote}-vote")
            for M, r in pts:
                a2.annotate(f"M={M}", (r["total_GB_resident"], r["accuracy"]),
                            textcoords="offset points", xytext=(5, -10), fontsize=7)
        if unr:
            for ax in (a1, a2):
                ax.axhline(unr["accuracy"], ls="--", color="#8496A6",
                           label=f"unrestricted = {unr['accuracy']:.2f}")
            a2.scatter([unr["total_GB_resident"]], [unr["accuracy"]], marker="v",
                       s=70, color="#8496A6", zorder=5)
        a1.set_xlabel("coreset size M"); a1.set_ylabel("accuracy (%)")
        a1.set_title(f"{ds}: accuracy vs M", fontsize=11)
        a1.grid(alpha=.25); a1.legend(fontsize=8)
        a2.set_xlabel("total bytes per block (GB, resident drafter)")
        a2.set_ylabel("accuracy (%)")
        a2.set_title(f"{ds}: accuracy vs bytes", fontsize=11)
        a2.grid(alpha=.25); a2.legend(fontsize=8)
    fig.suptitle("Stage 1 Arm B — block-level expert coresets  "
                 "(bytes: whole-block decode, drafter resident 3.4 GB/block)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/fig_s1b.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote S1B_metrics.json, plots/fig_s1b.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("S1B_LOGS_DIR", "logs/s1b"))


def _boxed_content(s):
    """Content of the LAST \\boxed{...}, brace-balanced so \\boxed{\\text{C}} works."""
    i = s.rfind("\\boxed")
    if i < 0:
        return ""
    j = s.find("{", i)
    if j < 0:
        return ""
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return s[j + 1:]


def _gpqa_letter(pred):
    """Extract the chosen option letter A-D. Strips LaTeX commands first so \\mathrm{B}
    does not yield the 'a' of 'mathrm'. Falls back to 'Answer: X' then a lone trailing letter."""
    s = str(pred)
    box = _boxed_content(s)
    if box:
        cleaned = re.sub(r"\\[a-zA-Z]+", " ", box)          # drop \text, \mathrm, ...
        cleaned = re.sub(r"[^A-Za-z]", " ", cleaned)
        for tok in cleaned.split():
            if len(tok) == 1 and tok.upper() in "ABCD":
                return tok.upper()
    m = re.findall(r"(?i)answer\s*(?:is)?\s*[:\-]?\s*\(?([A-Da-d])\)?[^A-Za-z]", s + " ")
    if m:
        return m[-1].upper()
    m = re.findall(r"(?m)^\s*\(?([A-Da-d])\)?[\.\)]?\s*$", s)
    return m[-1].upper() if m else ""


def _humaneval_score(rows):
    """Execute each completion against HumanEval's own unit tests (pass@1).

    `execution()` already isolates each program in its own timeout-guarded process, so for the
    small batches the inline scorer uses (1 row per call) we must NOT also open a worker pool --
    an unbounded ProcessPoolExecutor spawns one worker per core (384 here) per scored item.
    """
    import json as _json
    from oc_datasets.mbpp import MBPPEvaluator, execution
    ev = MBPPEvaluator()
    progs, idxs = [], []
    for x in rows:
        meta = _json.loads(x["gold"])
        code = ev._process_answer(str(x["pred"]))
        progs.append(code + "\n\n" + meta["test"] + "\n\ncheck(" + meta["entry_point"] + ")\n")
        idxs.append(x["idx"])
    out = {}
    if len(progs) <= 4:
        for i, pr in enumerate(progs):
            _, verdict = execution(pr, i, 10)
            out[idxs[i]] = int(verdict == "pass")
        return out
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(8, len(progs))) as pool:
        for f in [pool.submit(execution, pr, i, 10) for i, pr in enumerate(progs)]:
            i, verdict = f.result()
            out[idxs[i]] = int(verdict == "pass")
    return out


# ---------------------------------------------------------------------------
# corpus-1b held-out eval scorers (2026-09-28): finqa500, chartqa300 (docs in
# corpus1b/eval_sets.py -- gold field, prompt shape, join/order keys)
# ---------------------------------------------------------------------------
def _plain_number(s):
    """Lenient numeric parse shared by the finqa/chartqa branches: strips '$', ',' and a trailing '%'."""
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("$", "").strip()
    if t.endswith("%"):
        t = t[:-1].strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _rel_close(a, b, rel):
    return abs(a - b) <= rel * abs(b) if b else abs(a) <= rel


def _finqa_score(pred, gold):
    """FinQA gold (dreamerdeo/finqa's own `answer` field, e.g. "94" or "14%") vs the model's last
    \\boxed{} (or the raw text if unboxed). 1% relative tolerance; a percentage gold matches EITHER the
    percent-as-written prediction ("42%" gold vs pred "42") or the percent-as-fraction one (pred "0.42"),
    since a correct model answer is written either way about equally often. Falls back to case-insensitive
    exact string match when either side does not parse as a number."""
    box = _boxed_content(str(pred))
    p_text = box if box else str(pred)
    pn, gn = _plain_number(p_text), _plain_number(gold)
    if pn is None or gn is None:
        return int(p_text.strip().lower() == str(gold).strip().lower())
    if _rel_close(pn, gn, 0.01):
        return 1
    if str(gold).strip().endswith("%") and _rel_close(pn, gn / 100.0, 0.01):
        return 1
    return 0


def _chartqa_score(pred, gold):
    """ChartQA's own "relaxed accuracy": a numeric answer within 5% relative tolerance, else a
    case-insensitive exact string match. `pred` is read from the last \\boxed{} as with finqa."""
    box = _boxed_content(str(pred))
    p_text = box if box else str(pred)
    pn, gn = _plain_number(p_text), _plain_number(gold)
    if pn is not None and gn is not None:
        return int(_rel_close(pn, gn, 0.05))
    return int(p_text.strip().lower() == str(gold).strip().lower())
