#!/usr/bin/env python
"""Tables for run_da_decode.py cells (Draft-Verify decode and its --stock twin).

Per cell (all shard*.jsonl under the cell dir): n, accuracy, accuracy PAIRED against the stock
reference records on the same items (95% bootstrap CI, points), passes/block (EVERY block forward
except the kv flush: judge + rejudge + stock denoise; `stock_passes_per_block` keeps the denoise-only
count), MoE forwards per block and per generated token by kind (prefill / judge / rejudge /
denoise / kvflush), drafter forwards per token by kind, unique experts per layer-forward (all /
denoise / judge), expert-layer executions per token, accepted-draft and committed-at-judge
fractions per block, draft agreement, ms per token (MoE / drafter / wall), EOS + truncation rates,
the judge mode / expert-pool (M, rule, mean size actually used) / adapter of the cell, the full
judge's tentative-kept and re-masked slots per block, and -- for stock cells -- the per-item
integrity match against the reference records.

Usage: /venv/main/bin/python code/draftalign/analyze_da.py logs/draftalign/da_decode/<cell> [more cells...] \
           [--ref-dir logs/research_v1/baselines_b32] [--ref-cell gsm8k_B32_t95_stock] [--out summary.json]
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

N_LAYERS = 48
MOE_KINDS = ("prefill", "judge", "rejudge", "denoise", "kvflush")
DR_KINDS = ("prefill", "draft", "redraft", "flush")
N_BOOT = 10000


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def load_jsonl_records(files):
    recs = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            assert r["idx"] not in recs, f"duplicate idx {r['idx']} in {f}"
            recs[int(r["idx"])] = r
    return recs


def load_cell(cell_dir: str):
    """-> {(cell, split): {"recs": {idx: rec}, "complete": bool}} for every <cell>.<split>.shard*.jsonl in the dir."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(cell_dir, "*.shard*of*.jsonl"))):
        base = os.path.basename(f)[:-6]
        parts = base.split(".")
        groups[(parts[0], parts[1])].append(f)
    out = {}
    for key, files in groups.items():
        out[key] = {"recs": load_jsonl_records(files), "complete": all(os.path.exists(f + ".done") for f in files),
                    "files": files}
    return out


def load_ref(ref_dir: str, ref_cell: str, split: str):
    return load_jsonl_records(sorted(glob.glob(os.path.join(ref_dir, f"{ref_cell}.{split}.shard*of*.jsonl"))))


def ensure_correct(recs: dict, ds: str):
    missing = [r for r in recs.values() if r.get("correct") is None]
    if missing:
        from analyze_s1b import per_item_correct
        for idx, c in per_item_correct(ds, missing).items():
            recs[idx]["correct"] = c
    return {i: r["correct"] for i, r in recs.items() if r.get("correct") is not None}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def summarise(recs: dict, ds: str) -> dict:
    rows = [recs[i] for i in sorted(recs)]
    n = len(rows)
    tok = float(sum(r["n_gen_to_eos"] for r in rows))
    nblk = float(sum(r["n_blocks"] for r in rows))
    corr = ensure_correct(recs, ds)
    s = {"n": n, "accuracy": 100 * np.mean([corr[r["idx"]] for r in rows if r["idx"] in corr]) if corr else float("nan"),
         "correct": corr, "gen_tokens_mean": tok / n,
         "eos_rate": float(np.mean([r["stopped_eos"] for r in rows])),
         "truncated_rate": float(np.mean([r["truncated"] for r in rows])),
         "passes_per_block": float(np.mean([p for r in rows for p in r["passes_per_block"]])),
         "stock_passes_per_block": float(np.mean([p for r in rows for p in r.get("stock_passes_per_block", r["passes_per_block"])])),
         "tokens_per_pass": float(np.mean([c for r in rows for c in r["commits_per_pass"]])) if any(r["commits_per_pass"] for r in rows) else float("nan"),
         "s_per_item": float(np.mean([r["seconds"] for r in rows])),
         "ms_per_token": _safe_div(1000 * sum(r["seconds"] for r in rows), tok),
         "moe_ms_per_token": _safe_div(1000 * sum(r.get("moe_seconds", sum(r["ms_by_kind"].values()) / 1e3) for r in rows), tok),
         "drafter_ms_per_token": _safe_div(1000 * sum(r.get("drafter_seconds", 0.0) for r in rows), tok)}
    # MoE forwards by kind
    nk = {"prefill": sum(r["n_prefill"] for r in rows), "denoise": sum(r["n_denoise"] for r in rows),
          "kvflush": sum(r["n_flush"] for r in rows), "judge": sum(r.get("n_judge", 0) for r in rows),
          "rejudge": sum(r.get("n_rejudge", 0) for r in rows)}
    s["moe_forwards_per_token"] = _safe_div(sum(r["n_forward"] for r in rows), tok)
    s["moe_forwards_per_token_by_kind"] = {k: _safe_div(v, tok) for k, v in nk.items()}
    s["moe_forwards_per_block"] = _safe_div(sum(r["n_forward"] - r["n_prefill"] for r in rows), nblk)
    s["ms_per_forward_by_kind"] = {k: _safe_div(sum(r["ms_by_kind"].get(k, 0.0) for r in rows), nk[k]) for k in nk}
    # drafter forwards by kind
    dk = {k: sum(r.get("drafter_n_by_kind", {}).get(k, 0) for r in rows) for k in DR_KINDS}
    s["drafter_forwards_per_token"] = _safe_div(sum(dk.values()), tok)
    s["drafter_forwards_per_token_by_kind"] = {k: _safe_div(v, tok) for k, v in dk.items()}
    s["drafter_ms_per_forward_by_kind"] = {k: _safe_div(sum(r.get("drafter_ms_by_kind", {}).get(k, 0.0) for r in rows), dk[k]) for k in DR_KINDS}
    # experts
    uniq = float(sum(r["unique_expert_layer_loads"] for r in rows)); lf = float(sum(r["layer_forwards"] for r in rows))
    uniq_den = float(sum(r["uniq_by_kind"].get("denoise", 0) for r in rows))
    uniq_j = float(sum(r["uniq_by_kind"].get("judge", 0) + r["uniq_by_kind"].get("rejudge", 0) for r in rows))
    s["uniq_per_layer_forward_all"] = _safe_div(uniq, lf)
    s["uniq_per_layer_forward_denoise"] = _safe_div(uniq_den, nk["denoise"] * N_LAYERS)
    s["uniq_per_layer_forward_judge"] = _safe_div(uniq_j, (nk["judge"] + nk["rejudge"]) * N_LAYERS)
    s["expert_layer_exec_per_token"] = _safe_div(uniq, tok)
    # draft / judge
    nv = float(sum(sum(r["n_valid_per_block"]) for r in rows))
    acc = float(sum(sum(r.get("accepted_per_block", [])) for r in rows))
    acc_eq = float(sum(sum(r.get("accepted_eq_per_block", [])) for r in rows))
    cj = float(sum(sum(r.get("committed_judge_per_block", [])) for r in rows))
    forced = float(sum(sum(r.get("forced_per_block", [])) for r in rows))
    blind = float(sum(sum(r.get("blind_accept_per_block", [])) for r in rows))
    s["blind_accept_frac"] = _safe_div(blind, nv); s["blind_accept_per_block"] = _safe_div(blind, nblk)
    s["judge_modes"] = sorted({str(r.get("judge")) for r in rows})
    s["accepted_frac"] = _safe_div(acc, nv); s["accepted_eq_frac"] = _safe_div(acc_eq, nv)
    s["committed_judge_frac"] = _safe_div(cj, nv); s["forced_frac_blocks"] = _safe_div(forced, nblk)
    s["accepted_per_block"] = _safe_div(acc, nblk); s["committed_judge_per_block"] = _safe_div(cj, nblk)
    agree_blk = [a for r in rows for a in r.get("draft_agreement_per_block", []) if a is not None]
    agree_item = [np.mean([a for a in r.get("draft_agreement_per_block", []) if a is not None] or [np.nan]) for r in rows]
    s["draft_agreement_blocks"] = float(np.mean(agree_blk)) if agree_blk else float("nan")
    s["draft_agreement_item_ci"] = boot_ci(agree_item)
    s["rejudge_rounds_per_block"] = _safe_div(sum(sum(r.get("rejudge_rounds_per_block", [])) for r in rows), nblk)
    s["rejudge_accepted_per_block"] = _safe_div(sum(sum(r.get("rejudge_accepted_per_block", [])) for r in rows), nblk)
    s["judge_raw_mask_argmax"] = int(sum(r.get("judge_raw_mask_argmax", 0) for r in rows))
    # judge mode / expert pool / adapter of the cell (a cell may never mix them)
    s["judge_mode"] = sorted({str(r.get("judge_mode", r.get("judge"))) for r in rows})
    s["pool_M"] = sorted({str(r.get("pool_M")) for r in rows})
    s["pool_rule"] = sorted({str(r.get("pool_rule")) for r in rows})
    s["pool_flush"] = sorted({bool(r.get("pool_flush", False)) for r in rows})
    s["adapter"] = sorted({str(r.get("adapter")) for r in rows})
    s["adapter_hash"] = sorted({str(r.get("adapter_hash")) for r in rows})
    # mean pool size actually used, over the forwards that were pooled (field 5 of each record)
    psz = [f[5] for r in rows for f in r.get("forwards", []) if len(f) > 5 and f[5]]
    s["pool_size_used"] = float(np.mean(psz)) if psz else float("nan")
    s["pooled_forward_frac"] = _safe_div(len(psz), sum(len(r.get("forwards", [])) for r in rows))
    # full judge: tentative draft tokens kept / slots re-masked / commits that took the draft token
    tent = float(sum(sum(r.get("tentative_per_block", [])) for r in rows))
    rem = float(sum(sum(r.get("remask_per_block", [])) for r in rows))
    ceq = float(sum(sum(r.get("full_commit_eq_draft_per_block", [])) for r in rows))
    fb = float(sum(sum(r.get("full_fallback_per_block", [])) for r in rows))
    s["tentative_per_block"] = _safe_div(tent, nblk); s["remask_per_block"] = _safe_div(rem, nblk)
    s["tentative_frac"] = _safe_div(tent, nv); s["remask_frac"] = _safe_div(rem, nv)
    s["full_commit_eq_draft_per_block"] = _safe_div(ceq, nblk); s["full_commit_eq_draft_frac"] = _safe_div(ceq, nv)
    s["full_fallback_per_block"] = _safe_div(fb, nblk)
    # integrity vs the reference records (stock twin)
    rm = [r["ref_match"] for r in rows if r.get("ref_match")]
    s["ref_matched_items"] = len(rm)
    s["ref_gen_ids_match"] = int(sum(m["gen_ids"] for m in rm))
    s["ref_passes_match"] = int(sum(m["passes_per_block"] for m in rm))
    return s


def paired_accuracy(a: dict, b: dict):
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.array([a[i] - b[i] for i in common], float)
    m, lo, hi = boot_ci(d)
    return {"n": len(common), "delta": 100 * m, "ci95": [100 * lo, 100 * hi],
            "acc_a": 100 * float(np.mean([a[i] for i in common])), "acc_b": 100 * float(np.mean([b[i] for i in common]))}


def fmt(x, nd=2, w=7):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return " " * (w - 3) + "nan"
    return f"{x:{w}.{nd}f}"


def print_table(stats: dict, comps: dict):
    hdr = (f"{'cell':44}{'n':>4}{'acc':>6} {'d_acc [95% CI]':>22}{'pass/blk':>9}{'den/blk':>8}{'moeF/blk':>9}{'moeF/tok':>9}"
           f"{'judge/tok':>10}{'den/tok':>8}{'drF/tok':>8}{'uniq/lf':>8}{'u_den':>6}{'u_jdg':>6}{'exp/tok':>8}"
           f"{'acc%':>6}{'cj%':>5}{'agree':>7}{'ms/tok':>7}{'moe':>6}{'drft':>6}{'gen':>5}{'eos':>5}{'trunc':>6}")
    print(hdr); print("-" * len(hdr))
    for k in sorted(stats):
        s = stats[k]; c = comps.get(k)
        d = f"{c['delta']:+5.1f} [{c['ci95'][0]:+5.1f},{c['ci95'][1]:+5.1f}] n={c['n']}" if c else "ref"
        print(f"{k + ('' if s['complete'] else '*'):44}{s['n']:>4}{s['accuracy']:>6.1f} {d:>22}{fmt(s['passes_per_block'], 2, 9)}"
              f"{fmt(s['stock_passes_per_block'], 2, 8)}{fmt(s['moe_forwards_per_block'], 2, 9)}{fmt(s['moe_forwards_per_token'], 3, 9)}"
              f"{fmt(s['moe_forwards_per_token_by_kind']['judge'] + s['moe_forwards_per_token_by_kind']['rejudge'], 3, 10)}"
              f"{fmt(s['moe_forwards_per_token_by_kind']['denoise'], 3, 8)}{fmt(s['drafter_forwards_per_token'], 3, 8)}"
              f"{fmt(s['uniq_per_layer_forward_all'], 1, 8)}{fmt(s['uniq_per_layer_forward_denoise'], 1, 6)}{fmt(s['uniq_per_layer_forward_judge'], 1, 6)}"
              f"{fmt(s['expert_layer_exec_per_token'], 0, 8)}{fmt(100 * s['accepted_frac'], 1, 6)}{fmt(100 * s['committed_judge_frac'], 0, 5)}"
              f"{fmt(s['draft_agreement_blocks'], 3, 7)}{fmt(s['ms_per_token'], 0, 7)}{fmt(s['moe_ms_per_token'], 0, 6)}{fmt(s['drafter_ms_per_token'], 0, 6)}"
              f"{fmt(s['gen_tokens_mean'], 0, 5)}{fmt(s['eos_rate'], 2, 5)}{fmt(s['truncated_rate'], 2, 6)}")
    print("(* partial; pass/blk = every block forward except the flush (judge + denoise), den/blk = stock denoise "
          "passes only; acc% = accepted-draft slots / valid slots; cj% = committed-at-judge slots / valid slots; "
          "u_den / u_jdg = unique experts per layer-forward in denoise / judge forwards; ms/tok = wall, moe, drafter)")
    for k in sorted(stats):
        s = stats[k]
        pm = ",".join(s.get("pool_M", ["None"]))
        print(f"{k}: judge {','.join(s.get('judge_mode', ['?']))}; pool M {pm} rule {','.join(s.get('pool_rule', ['None']))}"
              f" flush {','.join(str(x) for x in s.get('pool_flush', [False]))}; pool size used "
              f"{fmt(s.get('pool_size_used'), 1, 1).strip()} on {100 * (s.get('pooled_forward_frac') or 0):.0f}% of forwards; "
              f"adapter {','.join(s.get('adapter', ['None']))} ({','.join(s.get('adapter_hash', ['None']))}); "
              f"full judge: tentative {s.get('tentative_per_block', float('nan')):.2f}/blk "
              f"({100 * (s.get('tentative_frac') or 0):.1f}% of valid), remask {s.get('remask_per_block', float('nan')):.2f}/blk "
              f"({100 * (s.get('remask_frac') or 0):.1f}%), commits taking the draft "
              f"{s.get('full_commit_eq_draft_per_block', float('nan')):.2f}/blk, stock fallback "
              f"{s.get('full_fallback_per_block', float('nan')):.2f}/blk")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="cell directories written by run_da_decode.py")
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "logs", "research_v1", "baselines_b32"))
    ap.add_argument("--ref-cell", default=None, help="default <ds>_B32_t95_stock")
    ap.add_argument("--out", default=None, help="json summary path")
    ap.add_argument("--partial", action="store_true", help="include incomplete cells")
    args = ap.parse_args(argv)

    stats, comps, refs_loaded = {}, {}, {}
    for d in args.cells:
        for (cell, split), g in load_cell(d).items():
            if not g["complete"] and not args.partial:
                print(f"[skip] {cell}.{split} in {d} is incomplete (use --partial)"); continue
            ds = "mbpp" if split == "mbpp" else "gsm8k"
            key = f"{cell}.{split}"
            s = summarise(g["recs"], ds); s["complete"] = g["complete"]; s["dir"] = d
            stats[key] = s
            ref_cell = args.ref_cell or f"{ds}_B32_t95_stock"
            rk = f"ref:{ref_cell}.{split}"
            if rk not in refs_loaded:
                ref = load_ref(args.ref_dir, ref_cell, split)
                refs_loaded[rk] = ref
                if ref:
                    rs = summarise(ref, ds); rs["complete"] = True; rs["dir"] = args.ref_dir
                    stats[rk] = rs
            ref = refs_loaded[rk]
            if ref and rk in stats:
                pc = paired_accuracy(s["correct"], stats[rk]["correct"])
                if pc:
                    comps[key] = pc
    if not stats:
        print("no cells"); return
    print_table(stats, comps)
    for k, s in stats.items():
        if s["ref_matched_items"]:
            print(f"{k}: stock integrity vs reference: gen_ids identical on {s['ref_gen_ids_match']}/{s['ref_matched_items']} items, "
                  f"passes identical on {s['ref_passes_match']}/{s['ref_matched_items']}")
    for k, s in stats.items():
        if not k.startswith("ref:") and not np.isnan(s["draft_agreement_blocks"]):
            m, lo, hi = s["draft_agreement_item_ci"]
            print(f"{k}: judge {','.join(s['judge_modes'])}; draft agreement item-mean {m:.3f} [{lo:.3f},{hi:.3f}]; accepted {s['accepted_per_block']:.2f}/blk "
                  f"(eq {100 * s['accepted_eq_frac']:.1f}% of valid), committed-at-judge {s['committed_judge_per_block']:.2f}/blk, "
                  f"forced-progress blocks {100 * s['forced_frac_blocks']:.1f}%, rejudge rounds {s['rejudge_rounds_per_block']:.2f}/blk "
                  f"(accepted {s['rejudge_accepted_per_block']:.2f}/blk); blind accepts {s['blind_accept_per_block']:.2f}/blk "
                  f"({100 * s['blind_accept_frac']:.1f}% of valid); raw MASK argmax at judge {s['judge_raw_mask_argmax']}")
            print(f"    ms/forward: moe " + ", ".join(f"{kk} {fmt(v, 0, 1).strip()}" for kk, v in s["ms_per_forward_by_kind"].items())
                  + "; drafter " + ", ".join(f"{kk} {fmt(v, 1, 1).strip()}" for kk, v in s["drafter_ms_per_forward_by_kind"].items()))
    if args.out:
        out = {"cells": {k: {a: b for a, b in v.items() if a != "correct"} for k, v in stats.items()}, "paired_vs_ref": comps}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
