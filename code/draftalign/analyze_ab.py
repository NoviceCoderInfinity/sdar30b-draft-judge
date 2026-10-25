#!/usr/bin/env python
"""Tables for the AB probe (probe_ab.py output). Reads every shard*.jsonl under the cell dir.

  1. drafter agreement with the FINAL tokens per setting (slot-level rate, block-mean, item-mean
     with a 95% bootstrap CI over items), forwards/block, ms/forward vs MoE ms/forward,
     calibration AUC of p_commit for draft==final;
  2. per MoE forward kind (halves merged: each valid slot read from the forward where it was
     [MASK]): argmax==final, mean top-1, commit rate at .95/.90, acceptance (argmax==fill),
     self-confirmation (top1>.95 & argmax==fill & fill!=final), correction (fill!=final &
     argmax==final); each next to the P0 mask-init reference with the same fill and an
     item-paired 95% bootstrap CI on the difference;
  3. expert tables: unique experts per forward, coverage by M and vote rule (mean over layers,
     min over layers), unique-expert recall, paired CIs vs P0;
  4. integrity: P0 (bf16 greedy) vs recorded pass-1 proposals.

Usage: /venv/main/bin/python code/draftalign/analyze_ab.py [--cell logs/draftalign/ab_probe/<cell>]
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from draftalign.drafter import SETTINGS            # noqa: E402
from draftalign.moe_probe import MS, KINDS         # noqa: E402

MERGED = {"P0": ("P0",), "P1": ("P1",), "P2": ("P2a", "P2b"), "P3": ("P3",), "P4": ("P4a", "P4b"),
          "P5": ("P5",), "P6": ("P6a", "P6b")}
FILL_OF = {"P0": "draft", "P1": "draft", "P2": "draft", "P3": "self", "P4": "self", "P5": "final", "P6": "final"}
EXPERT_KINDS = ["P0", "P1", "P2a", "P2b", "P2", "P3", "P4a", "P4b", "P4", "P5", "P6a", "P6b", "P6"]
N_BOOT = 10000


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def load_cell(cell_dir: str):
    items, blocks = {}, []
    for f in sorted(glob.glob(os.path.join(cell_dir, "shard*of*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("rec") == "item":
                items[int(r["idx"])] = r
            else:
                blocks.append(r)
    ok = {i for i, r in items.items() if r.get("status") == "ok"}
    blocks = [b for b in blocks if int(b["idx"]) in ok]
    return items, blocks


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_ci(a, b, n=N_BOOT, seed=0):
    """CI on mean(a - b) over items (a, b aligned per item)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return boot_ci(d, n, seed)


def auc(score, label):
    """Rank AUC (Mann-Whitney) with tie-averaged ranks; nan if one class is missing."""
    s = np.asarray(score, float); y = np.asarray(label, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def item_means(per_item: dict):
    """{idx: [values]} -> aligned (idx list, mean per item)."""
    idxs = sorted(per_item)
    return idxs, np.array([np.mean(per_item[i]) if len(per_item[i]) else np.nan for i in idxs])


def fmt(x, nd=3):
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:6.{nd}f}"


def fmt_ci(m, lo, hi, nd=3):
    return f"{m:+.{nd}f} [{lo:+.{nd}f},{hi:+.{nd}f}]"


# ---------------------------------------------------------------------------
# per-block slot arrays
# ---------------------------------------------------------------------------
def slot_arrays(block: dict, kind: str):
    """Per VALID slot: argmax, top1, fill token, final token for a (merged) kind."""
    valid = np.asarray(block["valid"], bool)
    final = np.asarray(block["final"], int)
    parts = MERGED[kind]
    if len(parts) == 1:
        m = block["moe"][parts[0]]
        argmax, top1 = np.asarray(m["argmax"], int), np.asarray(m["top1"], float)
    else:
        a, b = block["moe"][parts[0]], block["moe"][parts[1]]
        ma = np.asarray(a["masked"], bool)
        argmax = np.where(ma, np.asarray(a["argmax"], int), np.asarray(b["argmax"], int))
        top1 = np.where(ma, np.asarray(a["top1"], float), np.asarray(b["top1"], float))
    src = FILL_OF[kind]
    fill = {"draft": np.asarray(block["drafter"]["s8"]["tokens"], int), "self": np.asarray(block["self_draft"], int),
            "final": final}[src]
    return {"argmax": argmax[valid], "top1": top1[valid], "fill": fill[valid], "final": final[valid]}


def slot_metrics_np(sa: dict, thr_hi=0.95) -> dict:
    acc = sa["argmax"] == sa["final"]
    accept = sa["argmax"] == sa["fill"]
    wrong = sa["fill"] != sa["final"]
    return {"acc": acc, "top1": sa["top1"], "c95": sa["top1"] > 0.95, "c90": sa["top1"] > 0.90,
            "accept": accept, "selfconf": (sa["top1"] > thr_hi) & accept & wrong, "correct": wrong & acc}


def p0_reference(block: dict, src: str) -> dict:
    """P0 slot metrics read against the fill source `src` (mask-init reference for that source)."""
    sa = slot_arrays(block, "P0")
    valid = np.asarray(block["valid"], bool)
    fill = {"draft": np.asarray(block["drafter"]["s8"]["tokens"], int), "self": np.asarray(block["self_draft"], int),
            "final": np.asarray(block["final"], int)}[src]
    sa = {**sa, "fill": fill[valid]}
    return slot_metrics_np(sa)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_drafter(items, blocks):
    print("\n== 1. drafter vs FINAL tokens (valid slots) ==")
    moe_ms = [b["moe"][k]["ms"] for b in blocks for k, _, _ in KINDS]
    moe_ms_p0 = [b["moe"]["P0"]["ms"] for b in blocks]
    print(f"MoE probe forward: {np.mean(moe_ms):.1f} ms mean over all kinds ({np.mean(moe_ms_p0):.1f} ms P0), "
          f"flush {np.mean([b['moe_flush_ms'] for b in blocks]):.1f} ms; prefill {np.mean([it['moe_ms_by_kind'].get('prefill', 0) for it in items.values() if it.get('status') == 'ok']):.1f} ms/item")
    print(f"{'setting':8s} {'slot-rate':>9s} {'block-mean':>10s} {'item-mean [95% CI]':>26s} {'fwd/blk':>8s} {'steps':>6s} {'ms/fwd':>7s} "
          f"{'AUC(p_commit)':>13s} {'p_final|wrong':>13s} {'rawMASK':>7s} {'n_slots':>7s}")
    for s in SETTINGS:
        agree, blk_rate, per_item = [], [], defaultdict(list)
        pc, pfl_wrong, nf, ns, ms, rawm = [], [], [], [], [], 0
        for b in blocks:
            v = np.asarray(b["valid"], bool)
            d = b["drafter"][s]
            a = (np.asarray(d["tokens"], int) == np.asarray(b["final"], int))[v]
            agree.append(a); blk_rate.append(a.mean()); per_item[b["idx"]].append(a.mean())
            pc.append(np.asarray(d["p_commit"], float)[v])
            pfl_wrong.append(np.asarray(d["p_final_last"], float)[v][~a])
            nf.append(d["n_fwd"]); ns.append(d["n_steps"]); ms += d["ms"]; rawm += d["raw_mask_argmax"]
        agree = np.concatenate(agree); pc = np.concatenate(pc); pfl_wrong = np.concatenate(pfl_wrong)
        _, im = item_means(per_item)
        m, lo, hi = boot_ci(im)
        print(f"{s:8s} {agree.mean():9.4f} {np.mean(blk_rate):10.4f} {m:8.4f} [{lo:.4f},{hi:.4f}] {np.mean(nf):8.2f} {np.mean(ns):6.2f} "
              f"{np.mean(ms):7.1f} {auc(pc, agree):13.4f} {pfl_wrong.mean() if pfl_wrong.size else float('nan'):13.4f} {rawm:7d} {agree.size:7d}")
    print(f"drafter flush {np.mean([b['drafter_flush_ms'] for b in blocks]):.1f} ms; prefill "
          f"{np.mean([it['drafter_ms_by_kind'].get('prefill', 0) for it in items.values() if it.get('status') == 'ok']):.1f} ms/item")


def table_kinds(items, blocks):
    print("\n== 2. MoE forward kinds (valid slots; halves merged over their masked slots; fill = draft s8 | self = recorded pass-1 | final) ==")
    cols = ["acc", "top1", "c95", "c90", "accept", "selfconf", "correct"]
    print(f"{'kind':5s} {'fill':6s} " + " ".join(f"{c:>8s}" for c in cols) + "   | paired diff vs P0(same fill) [95% CI]: " + " ".join(cols))
    per_kind_items = {}
    for kind in MERGED:
        src = FILL_OF[kind]
        pooled = defaultdict(list); pi = defaultdict(lambda: defaultdict(list)); pi0 = defaultdict(lambda: defaultdict(list))
        for b in blocks:
            m = slot_metrics_np(slot_arrays(b, kind)); r = p0_reference(b, src)
            for c in cols:
                pooled[c].append(m[c]); pi[c][b["idx"]].append(m[c].mean()); pi0[c][b["idx"]].append(r[c].mean())
        vals = {c: float(np.concatenate(pooled[c]).mean()) for c in cols}
        diffs = []
        for c in cols:
            _, a = item_means(pi[c]); _, r0 = item_means(pi0[c])
            dm, lo, hi = paired_ci(a, r0)
            diffs.append(fmt_ci(dm, lo, hi))
        per_kind_items[kind] = pi
        print(f"{kind:5s} {src:6s} " + " ".join(f"{vals[c]:8.4f}" for c in cols) + "   | " + " ".join(diffs))
    n_slots = sum(int(b["n_valid"]) for b in blocks)
    rawm = {k: sum(b["moe"][k]["raw_mask_argmax"] for b in blocks) for k, _, _ in KINDS}
    fim = {k: sum(b["moe"][k]["fill_is_mask"] for b in blocks) for k, _, _ in KINDS}
    print(f"valid slots {n_slots}; raw fp32 argmax == [MASK] per kind: {rawm}; fill token == [MASK] per kind: {fim}")


def table_experts(items, blocks):
    print("\n== 3. experts: unique per forward (all 32 positions), coverage of R_final (valid picks) by top-M pool, unique-expert recall ==")
    L = len(blocks[0]["experts"]["P0"]["recall"])
    hdr = f"{'kind':5s} {'uniq/layer':>10s} {'recall':>7s} " + " ".join(f"{'t8:M' + str(M):>7s}" for M in MS) + " " + \
          " ".join(f"{'full:M' + str(M):>8s}" for M in MS) + "   | min over layers (t8): " + " ".join(f"M{M}" for M in MS)
    print(hdr)
    stats = {}
    for k in EXPERT_KINDS:
        if k not in blocks[0]["experts"]:
            continue
        uniq = np.array([b["experts"][k].get("uniq", [np.nan] * L) for b in blocks], float)      # [nb, L]
        rec = np.array([b["experts"][k]["recall"] for b in blocks], float)
        cov = {rule: {M: np.array([b["experts"][k]["cov"][rule][str(M)] for b in blocks], float) for M in MS} for rule in ("top8", "full")}
        stats[k] = (uniq, rec, cov)
        per_layer_t8 = {M: np.nanmean(cov["top8"][M], 0) for M in MS}
        print(f"{k:5s} {np.nanmean(uniq):10.2f} {np.nanmean(rec):7.4f} " + " ".join(f"{np.nanmean(cov['top8'][M]):7.4f}" for M in MS) + " " +
              " ".join(f"{np.nanmean(cov['full'][M]):8.4f}" for M in MS) + "   | " + " ".join(f"{per_layer_t8[M].min():.3f}" for M in MS))
    print("\npaired diff vs P0 (top-8 vote rule, mean over layers, item-paired 95% CI):")
    idx_of = [b["idx"] for b in blocks]
    def item_mean(arr):
        d = defaultdict(list)
        for i, v in zip(idx_of, np.nanmean(arr, 1)):
            d[i].append(v)
        return item_means(d)[1]
    base = stats["P0"]
    for k in EXPERT_KINDS:
        if k == "P0" or k not in stats:
            continue
        uniq, rec, cov = stats[k]
        cells = [fmt_ci(*paired_ci(item_mean(cov["top8"][M]), item_mean(base[2]["top8"][M]))) for M in MS]
        cells.append("recall " + fmt_ci(*paired_ci(item_mean(rec), item_mean(base[1]))))
        print(f"{k:5s} " + "  ".join(f"M{M} {c}" for M, c in zip(MS, cells[:-1])) + "  " + cells[-1])


def table_integrity(items, blocks):
    print("\n== 4. integrity ==")
    mism = [b["integrity"]["p0_mismatch"] for b in blocks]
    n_slots = sum(int(b["n_valid"]) for b in blocks)
    st = defaultdict(int)
    for it in items.values():
        st[it.get("status", "?")] += 1
    print(f"items by status: {dict(st)}; blocks analysed {len(blocks)}; valid slots {n_slots}")
    print(f"P0 bf16-argmax vs recorded pass-1 proposals: {sum(mism)} mismatching slots in {sum(1 for m in mism if m)} blocks "
          f"({sum(mism) / max(n_slots, 1):.5f} of valid slots)")
    nfwd = defaultdict(list)
    for it in items.values():
        if it.get("status") != "ok":
            continue
        for k, v in it["moe_n_by_kind"].items():
            nfwd["moe/" + k].append(v / it["n_blocks"])
        for k, v in it["drafter_n_by_kind"].items():
            nfwd["drafter/" + k].append(v / it["n_blocks"])
    print("forwards per block: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(nfwd.items())))
    pm = [it.get("peak_mem_gib") for it in items.values() if it.get("peak_mem_gib") is not None]
    if pm:
        print(f"peak memory {max(pm):.1f} GiB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=os.path.join(ROOT, "logs", "draftalign", "ab_probe", "gsm8k_B32_t95_stock.calibvalid"))
    args = ap.parse_args(argv)
    items, blocks = load_cell(args.cell)
    print(f"cell {args.cell}: {len(items)} item records, {len(blocks)} blocks from {sum(1 for i in items.values() if i.get('status') == 'ok')} ok items")
    if not blocks:
        print("no blocks"); return
    table_drafter(items, blocks)
    table_kinds(items, blocks)
    table_experts(items, blocks)
    table_integrity(items, blocks)


if __name__ == "__main__":
    main()
