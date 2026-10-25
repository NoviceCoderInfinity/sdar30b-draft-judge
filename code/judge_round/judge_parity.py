#!/usr/bin/env python
"""Parity of the SGLang judge decoder (code/judge_round/judge_sglang.py --mode decode) against the HF harness decoder
(code/draftalign/run_da_decode.py: SDAR-30B judge with the flag-gated LoRA, SDAR-1.7B drafter, full judge, rd1 re-draft,
--redraft-ban-rejected, --loop-guard 12) on the SAME items.  SGLang is for data generation only: this check says whether
its records may stand in for the harness's decode as training data, never that it may be used for a benchmark number.

    python code/judge_round/judge_parity.py --sglang logs/phase5/judge_sgl/decode --harness logs/phase5/judge_sgl/harness \
        [--out report.json] [--min-token-agreement 0.95] [--max-acc-gap 3.0] [--max-passes-gap 0.05] \
        [--require-accuracy] [--min-items 1] [--ignore-split]
    exit 0 = PASS, 3 = FAIL (the failing criterion is named), 2 = bad arguments / no readable records.

Inputs (DIR_OR_GLOB: a directory = every *.jsonl / *.jsonl.gz under it, recursively; or a glob; or one file; `.running.`
files and lines that are not records, i.e. without `idx` and `gen_ids`, are skipped).  Two record formats, joined on
(split, idx) (--ignore-split: on idx alone):
  harness  one JSON per item written by run_da_decode.py (da_decode.make_record): gen_ids (ALL non-mask ids of the last
           block, filler after the first EOS included), pred, gold, n_blocks, passes_per_block (judge + rejudge + stock
           denoise forwards of each block, kv flush excluded), drafter_n_by_kind, redraft_*_per_block, loopguard_*_per_block,
           full_fallback_per_block, and `correct` when run with --score.
  sglang   one JSON per item written by judge_sglang.py --mode decode: gen_ids (trimmed to the first EOS, inclusive), pred,
           n_blocks, n_judge, n_draft_fwd, n_redraft_rounds, n_redraft_slots, n_redraft_fwd, n_loopguard_slots,
           n_loopguard_changed, full_fallback, passes_per_block.
Both sides are cut at the first EOS (151645 / 151643) inclusive before anything is compared, so the harness's post-EOS filler
never counts.  Metrics (overall and per split):
  exact            share of items whose cut gen_ids are identical
  prefix           mean over items of (common prefix length / shorter length)
  tokagr           mean over items of (position-wise equal tokens over the shorter length / shorter length); 1.0 when both
                   are empty, 0.0 when only one is.  tokagr_pooled = the same, token-weighted.
  |dlen|           mean |length difference| in tokens
  pass/blk         sum(passes_per_block) / sum(n_blocks) per side (n_judge if the list is missing); gap = (sglang - harness) / harness
  dfwd/blk         drafter forwards per block, first drafts + re-drafts (harness: drafter_n_by_kind draft + redraft, as
                   compare_gates; sglang: n_draft_fwd + n_redraft_fwd)
  acc              accuracy in percent over the items the harness scored (`correct` not null): the harness side is its own
                   `correct`; the SGLang side is `analyze_s1b.per_item_correct` (the scorer behind run_da_decode --score) on the
                   SGLang `pred` with the harness record's `gold`.  Null when the harness records carry no correctness.
  loops            outputs with more than half of their tokens inside a 12-token sequence that already appeared earlier
                   (compare_gates.loops, applied to the cut gen_ids of each side)
  also reported: re-draft rounds / slots, loop-guard slots / changed per block, items that fell back to the stock rule.
Verdict on the OVERALL row: PASS when tokagr >= --min-token-agreement, |accuracy gap| <= --max-acc-gap points, and
|pass/blk relative gap| <= --max-passes-gap.  An accuracy that cannot be computed is reported as null and is NOT a failure,
unless --require-accuracy.  Accuracy moves in steps of 100/n points: with few items one discordant item can exceed the gap
(the report says so); judge a smoke run on its token and pass numbers.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import json
import os
import sys
import tempfile
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))

EOS_IDS = (151645, 151643)     # research_protocol.EOS_IDS (pinned by tests/test_judge_parity.py)
NGRAM = 12                     # drafter_lora/compare_gates.NGRAM (pinned by the same test)
EPS = 1e-9
CODE_SETS = ("mbpp", "humaneval")      # the scorer executes the generated program: run it from a scratch cwd (AGENTS.md 9 (p))


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def expand(spec: str) -> List[str]:
    """Files named by DIR_OR_GLOB, sorted, `.running.` files dropped."""
    hits = sorted(glob.glob(spec, recursive=True)) if any(c in spec for c in "*?[") else [spec]
    out: List[str] = []
    for h in hits:
        if os.path.isdir(h):
            for pat in ("**/*.jsonl", "**/*.jsonl.gz"):
                out += glob.glob(os.path.join(h, pat), recursive=True)
        elif os.path.isfile(h):
            out.append(h)
    return sorted({f for f in out if ".running." not in os.path.basename(f)})


def _lines(path: str) -> Iterable[str]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield line


def cut_at_eos(ids: Iterable) -> List[int]:
    """ids up to and including the first EOS (the harness's n_gen_to_eos rule); all of them when there is none."""
    out: List[int] = []
    for t in ids:
        t = int(t)
        out.append(t)
        if t in EOS_IDS:
            break
    return out


def _num(v) -> Optional[float]:
    """A scalar count, or the sum of a per-block list; None when the field is absent / null."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return float(sum(float(x or 0) for x in v))
    return None


def _first(r: dict, *keys: str) -> Optional[float]:
    """The first key present in the record, as a number (lists summed): scalar sglang keys first, harness per-block lists after."""
    for k in keys:
        if k in r:
            return _num(r[k])
    return None


def _any(v) -> bool:
    return any(bool(x) for x in v) if isinstance(v, (list, tuple)) else bool(v)


def _drafter_fwd(r: dict) -> Optional[float]:
    """First drafts + re-drafts, prefill and flush excluded (compare_gates' drafter forwards / block)."""
    bk = r.get("drafter_n_by_kind")
    if isinstance(bk, dict):
        return float(sum(int(bk.get(k, 0) or 0) for k in ("draft", "redraft")))
    first = _first(r, "n_draft_fwd", "drafter_forwards_per_block")
    if first is None:
        return None
    return first + (_first(r, "n_redraft_fwd", "redraft_forwards_per_block") or 0.0)


def digest(r: dict) -> dict:
    """The few fields the comparison needs (a harness record also carries per-forward lists of several MB)."""
    ids = cut_at_eos(r["gen_ids"])
    ppb = r.get("passes_per_block")
    passes = _num(ppb) if isinstance(ppb, (list, tuple)) else _num(r.get("n_judge"))
    blocks = _num(r.get("n_blocks"))
    if blocks is None and isinstance(ppb, (list, tuple)):
        blocks = float(len(ppb))
    c = r.get("correct")
    return {"split": str(r.get("split", "")), "idx": int(r["idx"]), "ids": ids, "pred": r.get("pred"), "gold": r.get("gold"),
            "blocks": blocks, "passes": passes, "draft_fwd": _drafter_fwd(r),
            "redraft_rounds": _first(r, "n_redraft_rounds", "redraft_rounds_per_block"),
            "redraft_slots": _first(r, "n_redraft_slots", "redraft_slots_per_block"),
            "lg_slots": _first(r, "n_loopguard_slots", "loopguard_slots_per_block"),
            "lg_changed": _first(r, "n_loopguard_changed", "loopguard_changed_per_block"),
            "fallback": _any(r["full_fallback"]) if "full_fallback" in r else
                        (_any(r["full_fallback_per_block"]) if "full_fallback_per_block" in r else None),
            "correct": None if c is None else int(bool(c))}


def read_records(spec: str) -> Tuple[Dict[Tuple[str, int], dict], dict]:
    """-> {(split, idx): digest}, info.  A later file wins a duplicate key (counted)."""
    files = expand(spec)
    recs: Dict[Tuple[str, int], dict] = {}
    n_lines = n_dup = n_skip = 0
    for f in files:
        for line in _lines(f):
            try:
                r = json.loads(line)
                if not (isinstance(r, dict) and "idx" in r and isinstance(r.get("gen_ids"), list)):
                    raise ValueError
                d = digest(r)
            except (ValueError, TypeError, KeyError):
                n_skip += 1                       # a truncated last line, or a jsonl that is not a record file
                continue
            n_lines += 1
            k = (d["split"], d["idx"])
            n_dup += k in recs
            recs[k] = d
    return recs, {"spec": spec, "files": len(files), "records": len(recs), "lines": n_lines, "duplicates": n_dup,
                  "skipped_lines": n_skip}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def agreement(a: List[int], b: List[int]) -> dict:
    """Prefix and position-wise agreement of two token lists over the shorter length."""
    m = min(len(a), len(b))
    if m == 0:
        v = 1.0 if len(a) == len(b) else 0.0
        return {"exact": len(a) == len(b), "prefix": v, "tok": v, "matches": 0, "min_len": 0, "dlen": abs(len(a) - len(b))}
    p = 0
    while p < m and a[p] == b[p]:
        p += 1
    eq = sum(1 for i in range(m) if a[i] == b[i])
    return {"exact": a == b, "prefix": p / m, "tok": eq / m, "matches": eq, "min_len": m, "dlen": abs(len(a) - len(b))}


def is_looping(ids: List[int]) -> bool:
    """drafter_lora/compare_gates.loops: more than half of the tokens inside a 12-token sequence seen earlier."""
    seen, rp = set(), 0
    for k in range(len(ids) - NGRAM + 1):
        t = tuple(ids[k:k + NGRAM])
        if t in seen:
            rp += 1
        else:
            seen.add(t)
    return len(ids) > 2 * NGRAM and rp / (len(ids) - NGRAM + 1) > 0.5


def dataset_of(split: str) -> str:
    """run_da_decode.main's items-name -> scorer-family map (kept in step with it by hand)."""
    base = str(split).split(":")[0]
    return ("mbpp" if base.startswith("mbpp") else
            "mbpp" if base.startswith("opencode") else
            "ifeval" if base.startswith("if") else
            "math500" if base.startswith("math") else
            "aime" if base.startswith("aime") else
            "gpqa" if base.startswith("gpqa") else
            "humaneval" if base.startswith("humaneval") else
            "finqa" if base.startswith("finqa") else
            "chartqa" if base.startswith("chartqa") else
            "chat" if base.startswith("chat") else
            "gsm8k")


def make_scorer() -> Callable[[str, dict], Optional[int]]:
    """score(ds, row) -> 0 / 1 / None: run_da_decode's `per_item_correct(ds, [r])[r['idx']]` (needs /venv/p2's deps)."""
    for p in (os.path.join(CODE, "compat"), CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from analyze_s1b import per_item_correct

    def score(ds: str, row: dict) -> Optional[int]:
        with (sandbox() if ds in CODE_SETS else contextlib.nullcontext()):
            v = per_item_correct(ds, [row])[row["idx"]]
        return None if v is None else int(v)
    return score


@contextlib.contextmanager
def sandbox():
    """A scratch cwd while generated programs run (a program may write files wherever it is launched)."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="parity_sandbox_") as d:
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _side_sum(items: List[dict], key: str) -> Optional[float]:
    """Sum of a field over the items; None unless EVERY item has it (a partial sum would bias a per-block rate)."""
    vals = [it[key] for it in items]
    return None if any(v is None for v in vals) else float(sum(vals))


def _rate(items: List[dict], key: str) -> Optional[float]:
    s, b = _side_sum(items, key), _side_sum(items, "blocks")
    return None if s is None or b is None else _ratio(s, b)


def _gap(sg: Optional[float], h: Optional[float]) -> Optional[float]:
    return None if sg is None or h is None or h == 0 else (sg - h) / h


def stats(pairs: List[Tuple[dict, dict]], scorer: Optional[Callable], max_acc_gap: float) -> dict:
    """Metrics of joined (sglang, harness) digests."""
    n = len(pairs)
    out: dict = {"n": n}
    if n == 0:
        return out
    ag = [agreement(s["ids"], h["ids"]) for s, h in pairs]
    out.update({"exact_matches": sum(a["exact"] for a in ag), "exact_rate": sum(a["exact"] for a in ag) / n,
                "prefix_agreement": sum(a["prefix"] for a in ag) / n, "token_agreement": sum(a["tok"] for a in ag) / n,
                "token_agreement_pooled": _ratio(sum(a["matches"] for a in ag), sum(a["min_len"] for a in ag)),
                "mean_abs_len_diff": sum(a["dlen"] for a in ag) / n,
                "mean_len_sglang": sum(len(s["ids"]) for s, _ in pairs) / n,
                "mean_len_harness": sum(len(h["ids"]) for _, h in pairs) / n})
    S, H = [s for s, _ in pairs], [h for _, h in pairs]
    for name, key in (("passes_per_block", "passes"), ("drafter_fwd_per_block", "draft_fwd"),
                      ("redraft_rounds_per_block", "redraft_rounds"), ("redraft_slots_per_block", "redraft_slots"),
                      ("loopguard_slots_per_block", "lg_slots"), ("loopguard_changed_per_block", "lg_changed")):
        a, b = _rate(S, key), _rate(H, key)
        out[name] = {"sglang": a, "harness": b, "rel_gap": _gap(a, b)}
    out["blocks"] = {"sglang": _side_sum(S, "blocks"), "harness": _side_sum(H, "blocks")}
    out["loops"] = {"sglang": sum(is_looping(s["ids"]) for s in S), "harness": sum(is_looping(h["ids"]) for h in H)}
    out["fallback_items"] = {side: (None if any(x["fallback"] is None for x in L) else sum(bool(x["fallback"]) for x in L))
                             for side, L in (("sglang", S), ("harness", H))}
    out["accuracy"] = accuracy(pairs, scorer, max_acc_gap)
    return out


def accuracy(pairs: List[Tuple[dict, dict]], scorer: Optional[Callable], max_acc_gap: float) -> dict:
    """Paired accuracy over the items the harness scored.  The SGLang pred is scored like the harness scores its own records."""
    null = {"n_scored": 0, "n_harness_scored": sum(h["correct"] is not None for _, h in pairs), "sglang": None,
            "harness": None, "gap_points": None, "sglang_only_correct": None, "harness_only_correct": None, "score_errors": 0}
    scored = [(s, h) for s, h in pairs if h["correct"] is not None]
    if not scored or scorer is None:
        return null
    sg_ok, h_ok, errors = [], [], 0
    for s, h in scored:
        row = {"idx": s["idx"], "pred": s["pred"], "gold": h["gold"] if h["gold"] is not None else s["gold"],
               "gen_ids": s["ids"], "n_gen_to_eos": len(s["ids"])}
        try:
            v = scorer(dataset_of(h["split"] or s["split"]), row)
        except Exception:                                                    # the harness records a score_error as correct: None
            v = None
        if v is None:
            errors += 1
            continue
        sg_ok.append(int(v))
        h_ok.append(h["correct"])
    m = len(sg_ok)
    if m == 0:
        return {**null, "score_errors": errors}
    a, b = 100.0 * sum(sg_ok) / m, 100.0 * sum(h_ok) / m
    res = {"n_scored": m, "n_harness_scored": len(scored), "sglang": a, "harness": b, "gap_points": a - b,
           "sglang_only_correct": sum(1 for x, y in zip(sg_ok, h_ok) if x and not y),
           "harness_only_correct": sum(1 for x, y in zip(sg_ok, h_ok) if y and not x), "score_errors": errors}
    if 100.0 / m > max_acc_gap + EPS:
        res["resolution_warning"] = (f"accuracy moves in steps of {100.0 / m:.1f} points at n={m}: one discordant item "
                                     f"already exceeds --max-acc-gap {max_acc_gap}")
    return res


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def verdict(overall: dict, min_tok: float, max_acc: float, max_pass: float, require_accuracy: bool,
            min_items: int) -> dict:
    """PASS / FAIL on the overall row; `failures` names every failing criterion."""
    crit: Dict[str, dict] = {}
    n = overall.get("n", 0)
    crit["joined_items"] = {"value": n, "limit": min_items, "op": ">=", "ok": n >= min_items}
    tok = overall.get("token_agreement")
    crit["token_agreement"] = {"value": tok, "limit": min_tok, "op": ">=", "ok": tok is not None and tok >= min_tok - EPS}
    pg = (overall.get("passes_per_block") or {}).get("rel_gap")
    crit["passes_per_block_rel_gap"] = {"value": pg, "limit": max_pass, "op": "|x|<=", "ok": pg is not None and abs(pg) <= max_pass + EPS}
    ag = (overall.get("accuracy") or {}).get("gap_points")
    if ag is None:
        crit["accuracy_gap_points"] = {"value": None, "limit": max_acc, "op": "|x|<=", "ok": False if require_accuracy else None,
                                       "note": "accuracy not available" + ("" if require_accuracy else " (not a failure)")}
    else:
        crit["accuracy_gap_points"] = {"value": ag, "limit": max_acc, "op": "|x|<=", "ok": abs(ag) <= max_acc + EPS}
    bad = {k: c for k, c in crit.items() if c["ok"] is False}
    msgs = []
    for k, c in bad.items():
        if c["value"] is None:
            msgs.append(f"{k}: not computable")
        else:
            msgs.append(f"{k} {c['value']:.4g} violates {c['op']} {c['limit']:.4g}")
    return {"status": "FAIL" if bad else "PASS", "criteria": crit, "failures": list(bad), "message": "; ".join(msgs)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def compare(sg: Dict[Tuple[str, int], dict], hr: Dict[Tuple[str, int], dict], scorer: Optional[Callable], max_acc_gap: float,
            ignore_split: bool = False) -> Tuple[dict, dict]:
    """-> (overall stats, per-split stats), plus the join counts under overall['join']."""
    def key(k):
        return ("", k[1]) if ignore_split else k
    sgk, hk = {key(k): v for k, v in sg.items()}, {key(k): v for k, v in hr.items()}
    shared = sorted(set(sgk) & set(hk))
    pairs = [(sgk[k], hk[k]) for k in shared]
    overall = stats(pairs, scorer, max_acc_gap)
    overall["join"] = {"sglang": len(sgk), "harness": len(hk), "joined": len(shared), "sglang_only": len(set(sgk) - set(hk)),
                       "harness_only": len(set(hk) - set(sgk))}
    per: Dict[str, dict] = {}
    splits = sorted({p[0]["split"] if not ignore_split else "all" for p in pairs}) if pairs else []
    for sp in splits:
        sub = pairs if ignore_split else [p for p in pairs if p[0]["split"] == sp]
        per[sp] = stats(sub, scorer, max_acc_gap)
        per[sp]["join"] = {"sglang": sum(1 for k in sgk if ignore_split or k[0] == sp), "harness": sum(1 for k in hk if ignore_split or k[0] == sp),
                           "joined": len(sub)}
    return overall, per


def _f(v, spec: str, none: str = "-") -> str:
    return none if v is None else format(v, spec)


def table(rep: dict) -> str:
    """The compact text table of a report (sgl = SGLang, har = HF harness)."""
    lines = [f"{'split':12s} {'n':>4s} {'exact':>6s} {'prefix':>6s} {'tokagr':>6s} {'|dlen|':>6s} | "
             f"{'pass/b sgl':>10s} {'har':>6s} {'gap':>7s} | {'dfwd/b sgl':>10s} {'har':>6s} {'gap':>7s} | "
             f"{'acc% sgl':>8s} {'har':>6s} {'gap pt':>7s} {'n':>4s} | {'loops sgl':>9s} {'har':>4s}"]
    for name, s in list(rep["per_split"].items()) + [("ALL", rep["overall"])]:
        if not s.get("n"):
            continue
        p, d, a = s["passes_per_block"], s["drafter_fwd_per_block"], s["accuracy"]
        lines.append(f"{name:12s} {s['n']:4d} {s['exact_rate']:6.3f} {s['prefix_agreement']:6.3f} {s['token_agreement']:6.3f} "
                     f"{s['mean_abs_len_diff']:6.1f} | {_f(p['sglang'], '.3f'):>10s} {_f(p['harness'], '.3f'):>6s} {_f(p['rel_gap'], '+.1%'):>7s} | "
                     f"{_f(d['sglang'], '.3f'):>10s} {_f(d['harness'], '.3f'):>6s} {_f(d['rel_gap'], '+.1%'):>7s} | "
                     f"{_f(a['sglang'], '.1f'):>8s} {_f(a['harness'], '.1f'):>6s} {_f(a['gap_points'], '+.1f'):>7s} {a['n_scored']:4d} | "
                     f"{s['loops']['sglang']:9d} {s['loops']['harness']:4d}")
    o = rep["overall"]
    if o.get("n"):
        lines.append("per block, sglang/harness: " + "  ".join(
            f"{k.replace('_per_block', '')} {_f(o[k]['sglang'], '.3f')}/{_f(o[k]['harness'], '.3f')}"
            for k in ("redraft_rounds_per_block", "redraft_slots_per_block", "loopguard_slots_per_block", "loopguard_changed_per_block"))
            + f"  |  items with stock fallback {_f(o['fallback_items']['sglang'], 'd')}/{_f(o['fallback_items']['harness'], 'd')}")
        w = o["accuracy"].get("resolution_warning")
        if w:
            lines.append("note: " + w)
    j = o["join"]
    lines.append(f"join: sglang {j['sglang']}, harness {j['harness']}, joined {j['joined']}, sglang-only {j['sglang_only']}, harness-only {j['harness_only']}")
    v = rep["verdict"]
    lines.append(f"VERDICT: {v['status']}" + (f"  ({v['message']})" if v["message"] else ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sglang", required=True, help="SGLang decode records: a dir, a glob or a file (judge_sglang.py --mode decode --out)")
    ap.add_argument("--harness", required=True, help="HF harness records of the same items (run_da_decode.py --out): dir, glob or file")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--min-token-agreement", type=float, default=0.95)
    ap.add_argument("--max-acc-gap", type=float, default=3.0, help="|sglang - harness| accuracy, in points")
    ap.add_argument("--max-passes-gap", type=float, default=0.05, help="|relative gap| of judge passes per block")
    ap.add_argument("--require-accuracy", action="store_true", help="FAIL when accuracy cannot be computed (default: report null)")
    ap.add_argument("--min-items", type=int, default=1, help="FAIL unless at least this many items are shared")
    ap.add_argument("--ignore-split", action="store_true", help="join on idx alone (when the two sides name the split differently)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sg, si = read_records(args.sglang)
    hr, hi = read_records(args.harness)
    if not sg or not hr:
        print(f"no records: sglang {si['records']} ({args.sglang}), harness {hi['records']} ({args.harness})", file=sys.stderr)
        return 2
    for side, info in (("sglang", si), ("harness", hi)):
        if info["duplicates"]:
            print(f"warning: {info['duplicates']} duplicate (split, idx) records on the {side} side (the later file wins): "
                  f"is {info['spec']} one run's output?", file=sys.stderr)
    sp = lambda R: sorted({k[0] for k in R})
    has_correct = any(h["correct"] is not None for h in hr.values())
    scorer = None
    scorer_error = None
    if has_correct:
        try:
            scorer = make_scorer()
        except Exception as e:                                                # no /venv/p2 deps: accuracy stays null
            scorer_error = f"{type(e).__name__}: {e}"
            print(f"warning: the harness scorer is unavailable ({scorer_error}); accuracy is null", file=sys.stderr)
    overall, per = compare(sg, hr, scorer, args.max_acc_gap, args.ignore_split)
    if overall.get("n", 0) == 0:
        print(f"no shared (split, idx): sglang splits {sp(sg)}, harness splits {sp(hr)}"
              + ("" if args.ignore_split else "; --ignore-split joins on idx alone"), file=sys.stderr)
    overall.setdefault("accuracy", {"n_scored": 0, "sglang": None, "harness": None, "gap_points": None})
    rep = {"sglang": si, "harness": hi,
           "thresholds": {"min_token_agreement": args.min_token_agreement, "max_acc_gap": args.max_acc_gap,
                          "max_passes_gap": args.max_passes_gap, "require_accuracy": args.require_accuracy,
                          "min_items": args.min_items, "ignore_split": args.ignore_split},
           "harness_has_correctness": has_correct, "scorer_error": scorer_error,
           "overall": overall, "per_split": per, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rep["verdict"] = verdict(overall, args.min_token_agreement, args.max_acc_gap, args.max_passes_gap, args.require_accuracy,
                             args.min_items)
    print(table(rep) if overall.get("n") else f"VERDICT: {rep['verdict']['status']}  ({rep['verdict']['message']})")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"[out] {args.out}")
    return 0 if rep["verdict"]["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
