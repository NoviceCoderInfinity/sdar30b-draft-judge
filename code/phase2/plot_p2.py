#!/usr/bin/env python
"""Phase 2 plots (CPU only): what every arm's decode, re-draft and probe have logged so far, as PNGs and on W&B.

Reads only what exists (the run is partial for hours); every plot whose inputs are missing is skipped with a
one-line note. Inputs (all optional):
    logs/phase2/decode/<ARM>/<set>/   records *.<set>.shard*of*.jsonl, running metrics *.running.jsonl,
                                      experts/<set>/<idx>.npz, judge/<set>/<idx>.npz, traces/<set>/<idx>.npz
    S0 reference of a set             logs/phase2/decode/S0/<set> if it has records, else ref/S0_by_set/<set>,
                                      else round 3's ref/S0_flat (the same order as run_phase2.sh)
    stock reference of a set          logs/phase2/decode/stock/<set>, else ref/stock_flat_by_set/<set>, else
                                      ref/stock_flat/*_B32_t95_stock.<set>.shard*
    logs/phase2/redraft/<ARM>/summary.json   (else its agreement.shard*.jsonl, merged here)
    logs/phase2/probe/<ARM>__r3_B/probe.json ; logs/phase2/report/<ARM>/experts.json
Large npz directories are subsampled (--max-items per set, evenly spaced over the sorted files: deterministic).

Outputs: logs/phase2/plots/<ARM>/<set>__<plot>.png, logs/phase2/plots/<ARM>/<plot>.png (re-draft, probe),
logs/phase2/plots/compare/<set>__arms.png and compare/redraft_agreement.png, plus plots/summary.json (the numbers
behind the comparison). With ENABLE_WANDB=1 and WANDB_API_KEY (from the environment or the repo's .env, never
printed) every PNG is logged as a wandb.Image to ONE run, id/name "phase2-plots", resumed in place (project
$DA_WANDB_PROJECT, default phase2-drafter-swap; the key's default entity). A W&B failure only warns.

    CUDA_VISIBLE_DEVICES= /venv/p2/bin/python code/phase2/plot_p2.py --once
    CUDA_VISIBLE_DEVICES= nohup /venv/p2/bin/python code/phase2/plot_p2.py --loop 30 > logs/phase2/plots.log 2>&1 &
--loop ends cleanly when logs/phase2/STOP exists.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(CODE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

from phase2.stoprules import paired_delta                                # noqa: E402

ARMS = ("S0", "F2", "F1", "S1")
SETS = ("humaneval", "mbpp", "gsm8k500", "ifeval541", "math500", "gpqa", "aime")
KINDS = ("prefill", "judge", "denoise", "kvflush")
N_LAYERS, N_EXPERTS = 48, 128
# fixed colour per entity (never by rank); the reference categorical order
ARM_COLOR = {"S0": "#2a78d6", "F2": "#eb6834", "F1": "#1baf7a", "S1": "#4a3aa7", "stock": "#8a8985"}
KIND_COLOR = {"prefill": "#2a78d6", "judge": "#eb6834", "denoise": "#1baf7a", "kvflush": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEEP = ("idx", "correct", "n_gen_to_eos", "unique_expert_layer_loads", "n_forward", "n_prefill", "n_flush",
        "n_blocks", "drafter_n_by_kind", "drafter_eq_per_forward", "charged_exp_per_token", "exp_by_kind",
        "exp_per_layer_profile_block", "judge_p_mean_by_pass", "judge_commit_frac_by_pass",
        "draft_agreement_per_block", "seconds")
WB_RUN_ID = "phase2-plots"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9, "axes.titlesize": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "lines.linewidth": 2.0, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


# ---------------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------------
def rpath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def out_path(p: str) -> str:
    """Repo-relative outputs go through research_protocol.safe_out; an absolute path (tests) only must not be a
    colleague's tree."""
    if not os.path.isabs(p):
        return P.safe_out(p)
    p = os.path.abspath(p)
    assert not any(p.startswith(b) for b in P.FORBIDDEN), f"never write into a colleague's tree: {p}"
    return p


def read_jsonl(path: str) -> list:
    """Rows of a jsonl being written: unparsable (partial) lines are skipped."""
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows


def record_files(d: str, split: str, pattern: str = "*") -> list:
    """Decode record shards of `split` in d (never the *.running.jsonl next to them)."""
    return [f for f in sorted(glob.glob(os.path.join(d, f"{pattern}.{split}.shard*of*.jsonl")))
            if not f.endswith(".running.jsonl")]


def load_records(d: str, split: str, pattern: str = "*") -> dict:
    recs = {}
    for f in record_files(d, split, pattern):
        for r in read_jsonl(f):
            if isinstance(r, dict) and "idx" in r and "n_gen_to_eos" in r:
                recs[int(r["idx"])] = {k: r.get(k) for k in KEEP}
    return recs


def load_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def subsample(files: list, n: int) -> list:
    if n <= 0 or len(files) <= n:
        return files
    return [files[i] for i in sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int).tolist()))]


def npz_files(d: str, kind: str, split: str, n: int) -> list:
    return subsample(sorted(glob.glob(os.path.join(d, kind, split, "*.npz"))), n)


def _mean(xs):
    v = [float(x) for x in xs if x is not None and x == x]
    return sum(v) / len(v) if v else None


class Sources:
    """Where each arm's / reference's records of a set live."""

    def __init__(self, a):
        self.a = a
        self.decode = rpath(a.decode_root)

    def arm_dir(self, arm: str, split: str):
        """The directory holding an arm's records of `split`: <base>/<set> or <base> itself (a flat out dir)."""
        base = self.a.extra.get(arm) or os.path.join(self.decode, arm)
        for d in (os.path.join(base, split), base):
            if record_files(d, split) or glob.glob(os.path.join(d, f"*.{split}.shard*of*.running.jsonl")):
                return d
        return None

    def s0(self, split: str):
        d = self.arm_dir("S0", split)
        if d:
            return load_records(d, split), "decode/S0"
        for root, pat, tag in ((os.path.join(rpath(self.a.s0_ref_root), split), "*", "ref/S0_by_set"),
                               (rpath(self.a.s0_flat), "*", "ref/S0_flat")):
            r = load_records(root, split, pat)
            if r:
                return r, tag
        return {}, None

    def stock(self, split: str):
        d = self.arm_dir("stock", split)
        if d:
            return load_records(d, split), "decode/stock"
        for root, pat, tag in ((os.path.join(rpath(self.a.stock_ref_root), split), "*", "ref/stock_flat_by_set"),
                               (rpath(self.a.stock_flat), "*_B32_t95_stock", "ref/stock_flat")):
            r = load_records(root, split, pat)
            if r:
                return r, tag
        return {}, None


# ---------------------------------------------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------------------------------------------
def cost_per_forward(arm: str, recs: dict) -> float:
    v = [r.get("drafter_eq_per_forward") for r in recs.values() if r.get("drafter_eq_per_forward") is not None]
    if v:
        return float(v[0])
    if arm == "stock":
        return 0.0
    try:
        from phase2 import drafters as P2D
        return float(P2D.cost_per_forward(arm))
    except Exception:                                                     # noqa: BLE001
        return 0.0


def agg(recs: dict, eq: float) -> dict:
    R = list(recs.values())
    if not R:
        return {"n": 0}
    gen = sum(int(r.get("n_gen_to_eos") or 0) for r in R)
    uniq = sum(int(r.get("unique_expert_layer_loads") or 0) for r in R)
    n_dr = sum(sum((r.get("drafter_n_by_kind") or {}).values()) for r in R)
    blocks = sum(int(r.get("n_blocks") or 0) for r in R)
    passes = sum(int(r.get("n_forward") or 0) - int(r.get("n_prefill") or 0) - int(r.get("n_flush") or 0) for r in R)
    sc = [int(r["correct"]) for r in R if r.get("correct") is not None]
    ag = [a for r in R for a in (r.get("draft_agreement_per_block") or []) if a is not None]
    return {"n": len(R), "acc": sum(sc) / len(sc) if sc else None, "n_scored": len(sc),
            "exp_tok": uniq / gen if gen else None, "charged_exp_tok": (uniq + n_dr * eq) / gen if gen else None,
            "passes_per_block": passes / blocks if blocks else None,
            "draft_agreement": sum(ag) / len(ag) if ag else None}


def passes_per_item(recs: dict) -> list:
    return [(int(r.get("n_forward") or 0) - int(r.get("n_prefill") or 0) - int(r.get("n_flush") or 0))
            / max(1, int(r.get("n_blocks") or 0)) for r in recs.values() if r.get("n_blocks")]


def expert_stats(files: list) -> dict:
    """Per stage: summed per-layer experts per forward, forwards, hit frequency [L, E], router mass [L]."""
    prof = defaultdict(lambda: np.zeros(N_LAYERS)); nf = defaultdict(int)
    hits = defaultdict(lambda: np.zeros((N_LAYERS, N_EXPERTS))); ms = defaultdict(lambda: np.zeros(N_LAYERS))
    mn = defaultdict(lambda: np.zeros(N_LAYERS)); n_items = 0
    for f in files:
        try:
            z = np.load(f)
            names = [str(x) for x in z["kind_names"]] if "kind_names" in z.files else \
                ["prefill", "denoise", "kvflush", "judge", "rejudge"]
            if "hit_bits" in z.files:
                bits = np.unpackbits(z["hit_bits"], axis=-1)[..., :N_EXPERTS].astype(bool)     # [F, L, E]
                kinds = z["kinds"].astype(int)
                for k in np.unique(kinds):
                    nm = names[k] if k < len(names) else str(k)
                    b = bits[kinds == k]
                    prof[nm][:b.shape[1]] += b.sum(-1).sum(0)
                    hits[nm][:b.shape[1], :b.shape[2]] += b.sum(0)
                    nf[nm] += b.shape[0]
            elif "counts" in z.files:                                     # no bitmask: counts > 0 per item
                for k, nm in enumerate(names[:z["counts"].shape[0]]):
                    c = z["counts"][k]
                    if c.sum():
                        hits[nm] += (c > 0); prof[nm] += (c > 0).sum(-1); nf[nm] += 1
            if "mass_sum" in z.files:
                for k, nm in enumerate(names[:z["mass_sum"].shape[0]]):
                    ms[nm] += z["mass_sum"][k]; mn[nm] += z["mass_n"][k]
            n_items += 1
        except Exception as e:                                            # noqa: BLE001  (a file being written)
            print(f"[plot] unreadable {f}: {e!r}", flush=True)
    out = {"items": n_items, "profile": {}, "freq": {}, "mass": {}}
    for nm in nf:
        if nf[nm]:
            out["profile"][nm] = prof[nm] / nf[nm]
            out["freq"][nm] = hits[nm] / nf[nm]
    for nm in ms:
        if mn[nm].sum():
            out["mass"][nm] = np.divide(ms[nm], mn[nm], out=np.full(N_LAYERS, np.nan), where=mn[nm] > 0)
    return out


def judge_by_pass(files: list, recs: dict, tau: float = 0.95, max_pass: int = 8):
    """Mean top-1 probability and share > tau of open slots at judge pass 1..8 (judge npz, else record fields)."""
    if files:
        sp, sc, sn, items = np.zeros(max_pass), np.zeros(max_pass), np.zeros(max_pass), np.zeros(max_pass)
        n = 0
        for f in files:
            try:
                z = np.load(f)
                ps, p, op = z["judge_pass"].astype(int), z["judge_p"].astype(np.float32), z["judge_open"].astype(bool)
            except Exception:                                             # noqa: BLE001
                continue
            n += 1
            for k in range(min(max_pass, int(ps.max()) + 1 if ps.size else 0)):
                v = p[ps == k][op[ps == k]]
                v = v[np.isfinite(v)]
                if v.size:
                    sp[k] += v.sum(); sc[k] += (v > tau).sum(); sn[k] += v.size; items[k] += 1
        if n and sn.any():
            m = sn > 0
            k = int(np.nonzero(m)[0].max()) + 1
            return {"src": f"judge npz, {n} items", "p": (sp / np.maximum(sn, 1))[:k].tolist(),
                    "c": (sc / np.maximum(sn, 1))[:k].tolist(), "items": items[:k].astype(int).tolist()}
    jp, jc = defaultdict(list), defaultdict(list)
    for r in recs.values():
        for i, v in enumerate(r.get("judge_p_mean_by_pass") or []):
            if v is not None:
                jp[i].append(v)
        for i, v in enumerate(r.get("judge_commit_frac_by_pass") or []):
            if v is not None:
                jc[i].append(v)
    if not jp:
        return None
    ks = range(max(jp) + 1)
    return {"src": f"records, {len(recs)} items", "p": [_mean(jp[i]) for i in ks], "c": [_mean(jc[i]) for i in ks],
            "items": [len(jp[i]) for i in ks]}


def reliability(files: list):
    """Drafter confidence vs (drafted token == the 30B's final token), 10 bins (as expert_report.py)."""
    cnt, sconf, sok = np.zeros(10), np.zeros(10), np.zeros(10)
    n = 0
    for f in files:
        try:
            z = np.load(f)
            if "p_draft" not in z.files:
                continue
            dt, pd, tok, va = z["draft_tokens"], z["p_draft"], z["tokens"], z["valid"].astype(bool)
        except Exception:                                                 # noqa: BLE001
            continue
        m = va & (dt >= 0) & np.isfinite(pd)
        if not m.any():
            continue
        c = pd[m].astype(np.float64)
        b = np.clip((c * 10).astype(int), 0, 9)
        np.add.at(cnt, b, 1); np.add.at(sconf, b, c); np.add.at(sok, b, (dt[m] == tok[m]).astype(np.float64))
        n += 1
    tot = cnt.sum()
    if not tot:
        return None
    return {"src": f"traces, {n} items", "items": n, "slots": int(tot), "ece": float(np.abs(sconf - sok).sum() / tot),
            "bins": [[i / 10, int(c), float(s / c) if c else None, float(o / c) if c else None]
                     for i, (c, s, o) in enumerate(zip(cnt, sconf, sok))]}


def redraft_summary(d: str):
    s = load_json(os.path.join(d, "summary.json"))
    if s and s.get("per_domain"):
        return s["per_domain"], "summary.json"
    agg_ = {}
    for f in sorted(glob.glob(os.path.join(d, "agreement.shard*of*.jsonl"))):
        for r in read_jsonl(f):
            for dom in (r.get("domain", "?"), "all"):
                a = agg_.setdefault(dom, {"items": 0, "slots": 0, "new_ok": 0, "prev_ok": 0})
                a["items"] += 1
                for k in ("slots", "new_ok", "prev_ok"):
                    a[k] += int(r.get(k) or 0)
    if not agg_:
        return None, None
    for a in agg_.values():
        a["agreement"] = a["new_ok"] / a["slots"] if a["slots"] else None
        a["prev_agreement"] = a["prev_ok"] / a["slots"] if a["slots"] else None
    return agg_, "agreement shards (not merged yet)"


def running_rows(d: str, split: str) -> list:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"*.{split}.shard*of*.running.jsonl"))):
        rows += [r for r in read_jsonl(f) if isinstance(r, dict) and r.get("items") is not None]
    by_n = {}
    for r in rows:
        by_n[int(r["items"])] = r                                        # every shard's row holds whole-cell metrics
    return [by_n[k] for k in sorted(by_n)]


# ---------------------------------------------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------------------------------------------
class Plotter:
    def __init__(self, out: str):
        self.out = out
        self.made, self.skipped = [], []

    def skip(self, scope: str, name: str, why: str):
        self.skipped.append((scope, name, why))
        print(f"[plot] skip {scope} {name}: {why}", flush=True)

    def save(self, fig, rel: str, title: str = None):
        if title:
            fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.tight_layout()
        fig.savefig(p + ".tmp.png", metadata={"Software": None})
        plt.close(fig)
        os.replace(p + ".tmp.png", p)
        self.made.append(rel)


def _note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])


def _bar_labels(ax, bars, vals, fmt="{:.2f}"):
    for b, v in zip(bars, vals):
        if v is not None and v == v:
            neg = b.get_height() < 0
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(4 if neg else 0, -2 if neg else 2),
                        textcoords="offset points", ha="left" if neg else "center", va="top" if neg else "bottom", fontsize=7, color=INK2)


def plot_running(pl, arm, split, rows, s0_acc, stock_acc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = [r["items"] for r in rows]
    acc = [r.get("accuracy") for r in rows]
    a1.plot(x, [np.nan if v is None else v for v in acc], color=ARM_COLOR.get(arm, INK), marker="o",
            ms=3 if len(x) > 40 else 5, label=arm)
    if all(v is None for v in acc):
        a1.text(0.5, 0.5, "no scored items yet", ha="center", va="center", transform=a1.transAxes, color=INK2)
    if s0_acc is not None and arm != "S0":
        a1.axhline(s0_acc, color=ARM_COLOR["S0"], lw=1, ls="--", label=f"S0 final {s0_acc:.3f}")
    if stock_acc is not None:
        a1.axhline(stock_acc, color=ARM_COLOR["stock"], lw=1, ls=":", label=f"stock {stock_acc:.3f}")
    a1.set(xlabel="items done", ylabel="accuracy", title="running accuracy")
    a1.legend(fontsize=7)
    d = [(r["items"], r["delta_acc_vs_s0"], r.get("delta_acc_ci95_lo"), r.get("delta_acc_ci95_hi")) for r in rows
         if r.get("delta_acc_vs_s0") is not None]
    if d:
        xs, dv, lo, hi = (np.array([np.nan if v is None else v for v in c], float) for c in zip(*d))
        a2.fill_between(xs, lo, hi, color=ARM_COLOR.get(arm, INK), alpha=0.18, lw=0, label="95% CI")
        a2.plot(xs, dv, color=ARM_COLOR.get(arm, INK), label=f"Δ = {dv[-1]:+.3f} [{lo[-1]:+.3f}, {hi[-1]:+.3f}]")
        a2.axhline(0, color=INK2, lw=0.8)
        a2.set(xlabel="items done", ylabel="Δ accuracy vs S0 (paired)", title="paired Δ vs S0")
        a2.legend(fontsize=7)
    else:
        _note(a2, "no paired Δ vs S0 yet")
    pl.save(fig, f"{arm}/{split}__running.png", f"{arm} · {split} · running metrics ({x[-1]} items)")


def plot_experts(pl, arm, split, st, s0st, stockst, ept):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.3, 1]})
    L = np.arange(N_LAYERS)
    for k in KINDS:
        if k in st["profile"]:
            a1.plot(L, st["profile"][k], color=KIND_COLOR[k], label=f"{arm} {k}")
    for ref, ls in ((s0st, "--"), (stockst, ":")):
        if ref:
            name, s = ref
            for k in ("judge", "denoise"):
                if k in s["profile"]:
                    a1.plot(L, s["profile"][k], color=KIND_COLOR[k], ls=ls, lw=1.2, label=f"{name} {k}")
    a1.set(xlabel="layer", ylabel="experts per layer, one forward", title=f"per-layer profile ({st['items']} items)")
    a1.legend(fontsize=7, ncol=2)
    names = [n for n in ("stock", "S0", arm) if ept.get(n) and ept[n].get("exp_tok")]
    names = list(dict.fromkeys(names))
    if names:
        xs = np.arange(len(names))
        v = [ept[n]["exp_tok"] for n in names]
        b = a2.bar(xs - 0.2, v, 0.38, color=[ARM_COLOR.get(n, INK) for n in names], label="loaded")
        _bar_labels(a2, b, v, "{:.0f}")
        c = [ept[n].get("charged_exp_tok") if n != "stock" else None for n in names]
        cb = a2.bar(xs + 0.2, [x or 0 for x in c], 0.38, color=[ARM_COLOR.get(n, INK) for n in names], alpha=0.45,
                    hatch="//", edgecolor="white", label="charged (+ drafter)")
        _bar_labels(a2, cb, c, "{:.0f}")
        a2.set_xticks(xs, names)
        a2.set(ylabel="expert-layer loads per token", title="experts per token")
        a2.legend(fontsize=7)
    else:
        _note(a2, "no records")
    pl.save(fig, f"{arm}/{split}__experts_profile.png", f"{arm} · {split} · experts that ran")


def plot_heatmap(pl, arm, split, st):
    ks = [k for k in KINDS if k in st["freq"]]
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks) + 0.8, 4.0), squeeze=False)
    im = None
    for ax, k in zip(axes[0], ks):
        im = ax.imshow(st["freq"][k], aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set(title=k, xlabel="expert", ylabel="layer")
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.01, label="share of forwards the expert ran")
    fig.suptitle(f"{arm} · {split} · expert activation frequency by stage ({st['items']} items)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    p = os.path.join(pl.out, f"{arm}/{split}__expert_heatmap.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p + ".tmp.png", metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)
    os.replace(p + ".tmp.png", p)
    pl.made.append(f"{arm}/{split}__expert_heatmap.png")


def plot_mass(pl, arm, split, st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2.3, 1]})
    ks = [k for k in KINDS if k in st["mass"]]
    for k in ks:
        a1.plot(np.arange(N_LAYERS), st["mass"][k], color=KIND_COLOR[k], label=k)
    a1.set(xlabel="layer", ylabel="router top-8 probability mass", title="per layer")
    a1.legend(fontsize=7)
    v = [float(np.nanmean(st["mass"][k])) for k in ks]
    b = a2.bar(ks, v, color=[KIND_COLOR[k] for k in ks])
    _bar_labels(a2, b, v, "{:.3f}")
    a2.set(title="mean over layers", ylim=(0, 1.0))
    pl.save(fig, f"{arm}/{split}__router_mass.png", f"{arm} · {split} · router top-8 mass by stage ({st['items']} items)")


def plot_judge(pl, arm, split, jb, s0jb):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, j, ls in ((arm, jb, "-"), ("S0", s0jb if arm != "S0" else None, "--")):
        if not j:
            continue
        x = np.arange(1, len(j["p"]) + 1)
        col = ARM_COLOR.get(name, INK)
        a1.plot(x, [np.nan if v is None else v for v in j["p"]], ls=ls, marker="o", ms=5, color=col,
                label=f"{name} ({j['src']})")
        a2.plot(x, [np.nan if v is None else v for v in j["c"]], ls=ls, marker="o", ms=5, color=col, label=name)
    a1.set(xlabel="judge pass within block", ylabel="mean top-1 probability (open slots)", title="judge confidence")
    a2.set(xlabel="judge pass within block", ylabel="share of slots > 0.95", title="commit share", ylim=(0, 1))
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    pl.save(fig, f"{arm}/{split}__judge_by_pass.png", f"{arm} · {split} · judge probabilities by pass")


def plot_reliability(pl, arm, split, rel):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    bins = rel["bins"]
    centers = [b[0] + 0.05 for b in bins]
    acc = [b[3] if b[3] is not None else np.nan for b in bins]
    a1.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--", label="perfect calibration")
    a1.bar(centers, acc, 0.09, color=ARM_COLOR.get(arm, INK), label="drafted token == final")
    conf = [b[2] if b[2] is not None else np.nan for b in bins]
    a1.plot(centers, conf, "o", ms=4, color=INK, label="mean confidence in bin")
    a1.set(ylabel="accuracy", ylim=(0, 1), title=f"ECE {rel['ece']:.3f} · {rel['slots']} slots · {rel['src']}")
    a1.legend(fontsize=7, loc="upper left")
    a2.bar(centers, [b[1] for b in bins], 0.09, color=ARM_COLOR.get(arm, INK), alpha=0.5)
    a2.set(xlabel="drafter confidence", ylabel="slots", xlim=(0, 1))
    pl.save(fig, f"{arm}/{split}__drafter_reliability.png", f"{arm} · {split} · drafter reliability")


def plot_passes_speedup(pl, arm, split, ppi, sp):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, v in ppi.items():
        if v:
            a1.hist(v, bins=np.arange(0, max(max(v), 2) + 1.0, 0.5), color=ARM_COLOR.get(name, INK), density=True,
                    histtype="step", lw=2, label=f"{name}: mean {np.mean(v):.2f} ({len(v)} items)")
    a1.set(xlabel="judge / denoise passes per block (per item)", ylabel="share of items (density)",
           title="passes per block")
    a1.legend(fontsize=7)
    names = [n for n in sp if sp[n] is not None]
    if names:
        b = a2.bar(names, [sp[n] for n in names], color=[ARM_COLOR.get(n.split()[0], INK) for n in names])
        _bar_labels(a2, b, [sp[n] for n in names], "{:.2f}×")
        a2.axhline(1, color=INK2, lw=0.8)
        a2.set(ylabel="stock experts/token ÷ charged experts/token", title="charged speedup vs stock")
    else:
        _note(a2, "no stock reference for this set")
    pl.save(fig, f"{arm}/{split}__passes_speedup.png", f"{arm} · {split} · passes and charged speedup")


def plot_redraft(pl, arm, pdom, src):
    doms = [d for d in sorted(pdom) if d != "all"] + (["all"] if "all" in pdom else [])
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(doms) + 2), 3.6))
    x = np.arange(len(doms))
    new = [pdom[d].get("agreement") for d in doms]
    prev = [pdom[d].get("prev_agreement") for d in doms]
    b1 = ax.bar(x - 0.2, [v or 0 for v in new], 0.38, color=ARM_COLOR.get(arm, INK), label=f"{arm}")
    b2 = ax.bar(x + 0.2, [v or 0 for v in prev], 0.38, color=ARM_COLOR["S0"], alpha=0.5,
                label="previous drafter (S0), same blocks")
    _bar_labels(ax, b1, new); _bar_labels(ax, b2, prev)
    ax.set_xticks(x, doms)
    ax.set(ylabel="draft agreement with the 30B's tokens", ylim=(0, 1.08))
    ax.legend(fontsize=7, loc="upper right")
    pl.save(fig, f"{arm}/redraft_agreement.png", f"{arm} · re-draft agreement per domain ({src})")


def plot_probe(pl, arm, pr):
    doms = sorted({k.split("/")[0] for k in pr if "/" in k and k.split("/")[0] not in ("all", "experts")})
    doms += ["all"]
    keys = (("keep_rate", "keep"), ("fix_rate", "fix"), ("conf_precision", "precision@0.95"), ("ece", "ECE"))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.4))
    for ax, (k, lab) in zip(axes, keys):
        v = [pr.get(f"{d}/{k}") for d in doms]
        v = [None if (x is None or x != x) else float(x) for x in v]
        b = ax.bar(doms, [x or 0 for x in v], color=[ARM_COLOR.get(arm, INK) if d != "all" else INK2 for d in doms])
        _bar_labels(ax, b, v, "{:.3f}")
        if k == "conf_precision":
            ax.axhline(0.99, color="#e34948", lw=1, ls="--")
            lab = "precision@0.95 (floor 0.99, dashed)"
            ax.set_ylim(min([x for x in v if x] + [0.9]) - 0.02, 1.005)
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=30)
    pl.save(fig, f"{arm}/probe.png", f"{arm} · calibration probe of the S0 judge on {arm}'s drafts")


def plot_compare(pl, split, rows):
    arms = [a for a in rows if rows[a].get("n")]
    keys = (("acc", "accuracy"), ("delta", "paired Δ acc vs S0 (95% CI)"), ("exp_tok", "experts per token"),
            ("speedup", "charged speedup vs stock"), ("draft_agreement", "draft agreement (decode)"))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.4))
    for ax, (k, lab) in zip(axes, keys):
        v = [rows[a].get(k) for a in arms]
        if not any(x is not None for x in v):
            _note(ax, "—"); ax.set_title(lab); continue
        b = ax.bar(arms, [x or 0 for x in v], color=[ARM_COLOR.get(a, INK) for a in arms])
        if k == "delta":
            lo = [rows[a].get("delta_lo") for a in arms]; hi = [rows[a].get("delta_hi") for a in arms]
            for i, (x, l_, h) in enumerate(zip(v, lo, hi)):
                if x is not None and l_ is not None:
                    ax.errorbar(i, x, yerr=[[x - l_], [h - x]], color=INK, capsize=3, lw=1)
            ax.axhline(0, color=INK2, lw=0.8)
        if k == "speedup":
            ax.axhline(1, color=INK2, lw=0.8)
        _bar_labels(ax, b, v, "{:.0f}" if k == "exp_tok" else "{:.3f}")
        ax.set_title(lab)
    ns = ", ".join(f"{a} {rows[a]['n']}" for a in arms)
    pl.save(fig, f"compare/{split}__arms.png", f"{split} · arms compared (items: {ns})")


def plot_compare_redraft(pl, per_arm):
    doms = sorted({d for p in per_arm.values() for d in p if d != "all"}) + ["all"]
    arms = list(per_arm)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(doms) + 2), 3.6))
    w = 0.8 / (len(arms) + 1)
    x = np.arange(len(doms))
    s0 = [_mean(per_arm[a].get(d, {}).get("prev_agreement") for a in arms) for d in doms]
    ax.bar(x - 0.4 + w / 2, [v or 0 for v in s0], w, color=ARM_COLOR["S0"], alpha=0.45, edgecolor="white",
           label="round-3 S0 drafts (previous)")
    for i, a in enumerate(arms):
        v = [per_arm[a].get(d, {}).get("agreement") for d in doms]
        b = ax.bar(x - 0.4 + w * (i + 1.5), [y or 0 for y in v], w, color=ARM_COLOR.get(a, INK), edgecolor="white",
                   label=f"{a} re-drafted")
        _bar_labels(ax, b, v)
    ax.set_xticks(x, doms)
    ax.set(ylabel="draft agreement", ylim=(0, 1))
    ax.legend(fontsize=7, ncol=len(arms) + 1)
    pl.save(fig, "compare/redraft_agreement.png", "re-draft agreement per domain, all arms")


# ---------------------------------------------------------------------------------------------------------------
# one pass
# ---------------------------------------------------------------------------------------------------------------
def run_pass(a) -> dict:
    out = out_path(a.out)
    os.makedirs(out, exist_ok=True)
    pl, src = Plotter(out), Sources(a)
    arms = [x for x in a.arms.split(",") if x] + [x for x in a.extra if x not in a.arms.split(",")]
    sets = [x for x in a.sets.split(",") if x]
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sets": {}, "arms": {}}
    s0_cache, stock_cache = {}, {}
    for split in sets:
        s0_cache[split] = src.s0(split)
        stock_cache[split] = src.stock(split)
    redraft_all = {}
    for arm in arms:
        # --- per arm: re-draft, probe
        if arm not in ("stock",):
            rd_dir = a.extra_redraft.get(arm) or os.path.join(rpath(a.redraft_root), arm)
            pdom, how = redraft_summary(rd_dir)
            if pdom:
                plot_redraft(pl, arm, pdom, how)
                redraft_all[arm] = pdom
                summary["arms"].setdefault(arm, {})["redraft_all"] = pdom.get("all")
            else:
                pl.skip(arm, "redraft_agreement", f"no summary.json / agreement shards in {rd_dir}")
            pr = load_json(os.path.join(rpath(a.probe_root), f"{arm}__r3_B", "probe.json"))
            if pr:
                plot_probe(pl, arm, pr)
            else:
                pl.skip(arm, "probe", f"no {a.probe_root}/{arm}__r3_B/probe.json")
        report = load_json(os.path.join(rpath(a.report_root), arm, "experts.json")) or {}
        any_set = False
        for split in sets:
            d = src.arm_dir(arm, split)
            scope = f"{arm}/{split}"
            if d is None:
                continue
            any_set = True
            recs = load_records(d, split)
            s0recs, s0tag = s0_cache[split]
            strecs, sttag = stock_cache[split]
            eq = cost_per_forward(arm, recs)
            ra, r0, rs = agg(recs, eq), agg(s0recs, cost_per_forward("S0", s0recs)), agg(strecs, 0.0)
            row = summary["sets"].setdefault(split, {})
            # running accuracy and paired delta
            rows = running_rows(d, split)
            if rows:
                plot_running(pl, arm, split, rows, r0.get("acc"), rs.get("acc"))
            else:
                pl.skip(scope, "running", "no *.running.jsonl rows")
            # experts
            efiles = npz_files(d, "experts", split, a.max_items)
            st = expert_stats(efiles) if efiles else None
            if st and st["profile"]:
                refs = []
                for name in ("S0", "stock"):
                    if name != arm:
                        rd = src.arm_dir(name, split)
                        f2 = npz_files(rd, "experts", split, a.max_items) if rd else []
                        refs.append((name, expert_stats(f2)) if f2 else None)
                    else:
                        refs.append(None)
                ept = {arm: ra, "S0": r0, "stock": rs}
                plot_experts(pl, arm, split, st, refs[0], refs[1], ept)
                plot_heatmap(pl, arm, split, st)
            else:
                pl.skip(scope, "experts_profile+heatmap", "no experts/<set>/*.npz")
            if st and st["mass"]:
                plot_mass(pl, arm, split, st)
            else:
                pl.skip(scope, "router_mass", "no mass_sum in the experts npz")
            # judge by pass
            jb = judge_by_pass(npz_files(d, "judge", split, a.max_items), recs)
            if jb:
                s0d = src.arm_dir("S0", split) if arm != "S0" else None
                s0jb = judge_by_pass(npz_files(s0d, "judge", split, a.max_items), s0recs) if s0d else \
                    judge_by_pass([], s0recs)
                plot_judge(pl, arm, split, jb, s0jb)
            else:
                pl.skip(scope, "judge_by_pass", "no judge npz and no judge_* record fields (stock or pre-Phase-2)")
            # drafter reliability
            rel = reliability(npz_files(d, "traces", split, a.max_items))
            if rel is None and (report.get("sets", {}).get(split) or {}).get("drafter_reliability"):
                r_ = report["sets"][split]["drafter_reliability"]
                rel = {**r_, "src": "report experts.json"}
            if rel:
                plot_reliability(pl, arm, split, rel)
                row.setdefault(arm, {})["drafter_ece"] = rel["ece"]
            else:
                pl.skip(scope, "drafter_reliability", "no traces with p_draft and no report experts.json")
            # passes per block, charged speedup
            if recs:
                ppi = {arm: passes_per_item(recs)}
                if arm != "S0" and s0recs:
                    ppi["S0"] = passes_per_item(s0recs)
                sp = {}
                if rs.get("exp_tok"):
                    for name, r in ((arm, ra), ("S0", r0)):
                        if r.get("charged_exp_tok") and name not in sp:
                            sp[name] = rs["exp_tok"] / r["charged_exp_tok"]
                plot_passes_speedup(pl, arm, split, ppi, sp)
            else:
                pl.skip(scope, "passes_speedup", "no finished records yet")
            row[arm] = {**row.get(arm, {}), **ra, "eq_per_forward": eq}
            row["S0_ref"] = {**r0, "src": s0tag}
            row["stock_ref"] = {**rs, "src": sttag}
        if not any_set:
            pl.skip(arm, "all per-set plots", f"no records under {os.path.join(a.decode_root, arm)}/<set>")
    # --- cross-arm per set
    for split in sets:
        s0recs, s0tag = s0_cache[split]
        strecs, _ = stock_cache[split]
        rs = agg(strecs, 0.0)
        rows = {}
        for arm in (["S0"] + [x for x in arms if x not in ("S0", "stock")]):
            if arm == "S0":
                recs = s0recs
            else:
                d = src.arm_dir(arm, split)
                recs = load_records(d, split) if d else {}
            if not recs:
                continue
            r = agg(recs, cost_per_forward(arm, recs))
            if arm != "S0" and s0recs:
                sh = sorted(i for i in recs if i in s0recs and recs[i].get("correct") is not None
                            and s0recs[i].get("correct") is not None)
                if len(sh) >= 2:
                    pdl = paired_delta([int(recs[i]["correct"]) for i in sh], [int(s0recs[i]["correct"]) for i in sh])
                    r.update(delta=pdl["delta"], delta_lo=pdl["lo"], delta_hi=pdl["hi"], paired_n=pdl["n"])
            if rs.get("exp_tok") and r.get("charged_exp_tok"):
                r["speedup"] = rs["exp_tok"] / r["charged_exp_tok"]
            rows[arm] = r
        if len(rows) >= 2:
            plot_compare(pl, split, rows)
            summary["sets"].setdefault(split, {})["compare"] = rows
        else:
            pl.skip(f"compare/{split}", "arms", f"fewer than two arms with records (have: {sorted(rows) or 'none'})")
    if redraft_all:
        plot_compare_redraft(pl, redraft_all)
    else:
        pl.skip("compare", "redraft_agreement", "no arm has re-draft outputs")
    summary["made"], summary["skipped"] = pl.made, [list(s) for s in pl.skipped]
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: None)
    print(f"[plot] {time.strftime('%H:%M:%S UTC', time.gmtime())}: {len(pl.made)} plots -> {out}, "
          f"{len(pl.skipped)} skipped", flush=True)
    return {"out": out, "made": pl.made, "skipped": pl.skipped}


# ---------------------------------------------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------------------------------------------
def load_env(path: str = os.path.join(ROOT, ".env")) -> None:
    """KEY=VALUE pairs of the repo's .env into os.environ (an existing value wins). Nothing is printed."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                if k and k not in os.environ:
                    os.environ[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass


def wandb_upload(res: dict, state_path: str) -> None:
    """Log every new or changed PNG as wandb.Image to the single 'phase2-plots' run. Never raises."""
    load_env()
    if os.environ.get("ENABLE_WANDB", "0").strip().lower() not in ("1", "true", "yes") or \
            not os.environ.get("WANDB_API_KEY"):
        print("[wandb] off (ENABLE_WANDB != 1 or no WANDB_API_KEY)", flush=True)
        return
    try:
        seen = load_json(state_path) or {}
        new = {}
        for rel in res["made"]:
            h = hashlib.sha256(open(os.path.join(res["out"], rel), "rb").read()).hexdigest()
            if seen.get(rel) != h:
                new[rel] = h
        if not new:
            print("[wandb] no changed plots", flush=True)
            return
        import wandb
        os.environ.pop("WANDB_ENTITY", None)                              # the key's default entity
        wdir = "/dev/shm/p2_plots_wandb"
        os.makedirs(wdir, exist_ok=True)
        run = wandb.init(project=os.environ.get("DA_WANDB_PROJECT") or "phase2-drafter-swap", id=WB_RUN_ID,
                         name=WB_RUN_ID, resume="allow", dir=wdir, job_type="plots")
        row = {}
        for rel in new:
            key = rel[:-4].replace("__", "/")
            row[key] = wandb.Image(os.path.join(res["out"], rel), caption=rel)
        run.log(row)
        url = getattr(run, "url", None)
        run.finish()
        seen.update(new)
        with open(state_path, "w") as fh:
            json.dump(seen, fh, indent=0)
        print(f"[wandb] {len(new)} plots -> {url}", flush=True)
    except Exception as e:                                                # noqa: BLE001
        print(f"[wandb] WARNING upload failed: {type(e).__name__}: {str(e)[:300]}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one pass (the default)")
    ap.add_argument("--loop", type=float, default=0, help="minutes between passes (0 = once)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--max-items", type=int, default=64, help="npz items per set and kind (evenly spaced)")
    ap.add_argument("--decode-root", default="logs/phase2/decode")
    ap.add_argument("--s0-ref-root", default="logs/phase2/ref/S0_by_set")
    ap.add_argument("--s0-flat", default="logs/phase2/ref/S0_flat")
    ap.add_argument("--stock-ref-root", default="logs/phase2/ref/stock_flat_by_set")
    ap.add_argument("--stock-flat", default="logs/phase2/ref/stock_flat")
    ap.add_argument("--redraft-root", default="logs/phase2/redraft")
    ap.add_argument("--probe-root", default="logs/phase2/probe")
    ap.add_argument("--report-root", default="logs/phase2/report")
    ap.add_argument("--extra-arm", action="append", default=[], metavar="NAME=DIR",
                    help="decode out dir of an arm (records <DIR>/<set>/ or directly in <DIR>), e.g. a smoke run")
    ap.add_argument("--extra-redraft", action="append", default=[], metavar="NAME=DIR",
                    help="re-draft out dir of an arm, overriding <redraft-root>/<NAME>")
    ap.add_argument("--out", default="logs/phase2/plots")
    ap.add_argument("--stop-file", default="logs/phase2/STOP")
    a = ap.parse_args(argv)
    a.extra = dict(x.split("=", 1) for x in a.extra_arm)
    a.extra = {k: rpath(v) for k, v in a.extra.items()}
    a.extra_redraft = {k: rpath(v) for k, v in (x.split("=", 1) for x in a.extra_redraft)}
    stop = rpath(a.stop_file)
    while True:
        if a.loop and os.path.exists(stop):
            print(f"[plot] {stop} exists: done", flush=True)
            return 0
        try:
            res = run_pass(a)
            if not a.no_wandb:
                wandb_upload(res, os.path.join(res["out"], ".wandb_uploaded.json"))
        except Exception as e:                                            # noqa: BLE001
            if not a.loop:
                raise
            print(f"[plot] pass failed: {type(e).__name__}: {e}", flush=True)
        if not a.loop:
            return 0
        t_end = time.time() + 60 * a.loop
        while time.time() < t_end:
            if os.path.exists(stop):
                print(f"[plot] {stop} exists: done", flush=True)
                return 0
            time.sleep(min(30, max(0.0, t_end - time.time())))


if __name__ == "__main__":
    sys.exit(main())
