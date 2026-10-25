#!/usr/bin/env python
"""The head-to-head comparison table: stock vs DES-pooled stock vs judge, per split.

Columns beyond eval_table.py's:
  exp/layer-fwd  unique expert-layer loads / layer_forwards -- how many DISTINCT experts each MoE
                 layer actually executes per forward.  Under a top-M pool this is the number to read
                 against M: it says how much of the pool is really used.
  +drafter       exp/tok with the drafter's own compute converted to expert-equivalents:
                 drafter_forwards * 364.6 / tokens, where 364.6 = 1,720,573,440 / 4,718,592 =
                 (SDAR-1.7B params excluding its token-embedding matrix) / (one 30B expert's params,
                 3 x hidden 2048 x moe_intermediate 768).  Stock rows have no drafter, so it equals
                 exp/tok there.  This is the honest total-compute column: a drafter-assisted rule is
                 only cheaper if it wins HERE, not on exp/tok alone.
  lost/gain      paired against that split's stock decode on the SHARED items: how many items stock
                 got right and this arm got wrong / wrong and this arm got right.  net = gain - lost.
"""
from __future__ import annotations
import glob, json, os, random, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "logs", "draftalign", "da_decode")
REF = os.path.join(ROOT, "logs", "research_v1", "baselines_b32")
DRAFTER_EXPERT_EQUIV = 1_720_573_440 / 4_718_592          # = 364.64


def load(cell, split, root=D):
    recs = {}
    pats = [f"{root}/*/{cell}.{split}.shard*of*.jsonl", f"{root}/{cell}.{split}.shard*of*.jsonl"]
    for p in pats:
        for f in glob.glob(p):
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("correct") is None:
                    continue
                recs[r["idx"]] = r
    return recs


def metrics(recs):
    if not recs:
        return None
    R = list(recs.values())
    def S(key):
        t = 0
        for r in R:
            v = r.get(key)
            if isinstance(v, list):
                t += sum(x for x in v if isinstance(x, (int, float)))
            elif isinstance(v, dict):
                t += sum(v.values())
            elif v:
                t += v
        return t
    toks = sum(max(1, r["n_gen_to_eos"]) for r in R)
    uniq, lf = S("unique_expert_layer_loads"), S("layer_forwards")
    dfw = sum(sum((r.get("drafter_n_by_kind") or {}).values()) for r in R)
    return {
        "n": len(R),
        "fwd_blk": S("passes_per_block") / max(S("n_blocks"), 1),
        "exp_layer_fwd": uniq / max(lf, 1),
        "exp_tok": uniq / toks,
        "plus_drafter": (uniq + dfw * DRAFTER_EXPERT_EQUIV) / toks,
        "ms_tok": 1000.0 * S("seconds") / toks,
        "acc": 100.0 * sum(1 for r in R if r["correct"]) / len(R),
    }


def paired(arm, base, boot=2000, seed=0):
    """Δacc on shared items + 95% paired bootstrap, and the lost/gain counts."""
    ids = sorted(set(arm) & set(base))
    if not ids:
        return None
    d = [int(bool(arm[i]["correct"])) - int(bool(base[i]["correct"])) for i in ids]
    lost = sum(1 for x in d if x < 0)
    gain = sum(1 for x in d if x > 0)
    mean = 100.0 * sum(d) / len(d)
    rng = random.Random(seed)
    bs = sorted(100.0 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(boot))
    return mean, bs[int(0.025 * boot)], bs[int(0.975 * boot)], lost, gain, len(ids)


JUDGE = "_ad734af656"
SPLITS = [
    ("GSM8K calibvalid (100 frozen TRAIN prompts, thresholds tuned here)", "calibvalid", [
        ("stock",              "gsm8k_B32_t95_stock", D),
        ("stock + DES M=32",   "gsm8k_B32_t95_stock_p32desf", D),
        ("stock + DES M=48",   "gsm8k_B32_t95_stock_p48desf", D),
        ("free rule blind_q50","gsm8k_B32_t95_dv-s8_blind_q50", D),
        ("judge c95",          f"gsm8k_B32_t95_dv-s8_full_c95{JUDGE}", D),
        ("judge + DES M=32",   f"gsm8k_B32_t95_dv-s8_full_c95_p32desf{JUDGE}", D),
        ("judge + DES M=48",   f"gsm8k_B32_t95_dv-s8_full_c95_p48desf{JUDGE}", D)]),
    ("GSM8K-500 (500 held-out TEST items)", "gsm8k500", [
        ("stock",              "gsm8k_B32_t95_stock", REF),
        ("stock + DES M=32",   "gsm8k_B32_t95_stock_p32desf", D),
        ("stock + DES M=48",   "gsm8k_B32_t95_stock_p48desf", D),
        ("free rule blind_q50","gsm8k_B32_t95_dv-s8_blind_q50", D),
        ("judge c95",          f"gsm8k_B32_t95_dv-s8_full_c95{JUDGE}", D),
        ("judge + DES M=32",   f"gsm8k_B32_t95_dv-s8_full_c95_p32desf{JUDGE}", D),
        ("judge + DES M=48",   f"gsm8k_B32_t95_dv-s8_full_c95_p48desf{JUDGE}", D)]),
    ("MBPP sanitized test (257 items)", "mbpp", [
        ("stock",              "mbpp_B32_t95_stock", D),
        ("stock + DES M=32",   "mbpp_B32_t95_stock_p32desf", D),
        ("stock + DES M=48",   "mbpp_B32_t95_stock_p48desf", D),
        ("free rule blind_q50","mbpp_B32_t95_dv-s8_blind_q50", D),
        ("judge c95",          f"mbpp_B32_t95_dv-s8_full_c95{JUDGE}", D),
        ("judge + DES M=32",   f"mbpp_B32_t95_dv-s8_full_c95_p32desf{JUDGE}", D),
        ("judge + DES M=48",   f"mbpp_B32_t95_dv-s8_full_c95_p48desf{JUDGE}", D)]),
    ("MATH-500 (500 test items)", "math500", [
        ("stock",              "math500_B32_t95_stock", D),
        ("free rule blind_q50","math500_B32_t95_dv-s8_blind_q50", D),
        ("judge c95",          f"math500_B32_t95_dv-s8_full_c95{JUDGE}", D)]),
]

W = (18, 8, 14, 9, 10, 8, 7, 22, 10)
HDR = ("cell", "fwd/blk", "exp/layer-fwd", "exp/tok", "+drafter", "ms/tok", "acc", "Δacc [95% CI]", "lost/gain")


def row(cells):
    return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, W)))


def main():
    for title, split, arms in SPLITS:
        print(f"\n### {title}")
        print(row(HDR)); print("-" * (sum(W) + 2 * (len(W) - 1)))
        base = None
        for label, cell, root in arms:
            recs = load(cell, split, root)
            m = metrics(recs)
            if not m:
                print(row((label, "—", "—", "—", "—", "—", "—", "CELL MISSING", "—"))); continue
            if base is None:
                base = recs
                dtxt, lg = "reference", "—"
            else:
                p = paired(recs, base)
                dtxt = (f"{p[0]:+.1f} [{p[1]:+.1f}, {p[2]:+.1f}]" + (f" n={p[5]}" if p[5] != m["n"] else "")) if p else "—"
                lg = f"{p[3]}/{p[4]}" if p else "—"
            print(row((f"{label}" + ("" if m["n"] >= 100 or split == "calibvalid" else ""),
                       f"{m['fwd_blk']:.2f}", f"{m['exp_layer_fwd']:.2f}", f"{m['exp_tok']:.1f}",
                       f"{m['plus_drafter']:.1f}", f"{m['ms_tok']:.1f}", f"{m['acc']:.1f}", dtxt, lg)))
            print(row(("", "", "", "", "", "", f"n={m['n']}", "", "")))
    print(f"\n+drafter uses {DRAFTER_EXPERT_EQUIV:.1f} expert-equivalents per drafter forward "
          f"(1,720,573,440 / 4,718,592).")


if __name__ == "__main__":
    main()
