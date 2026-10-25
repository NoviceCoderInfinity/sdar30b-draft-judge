"""Calibration diagnostics for the judge LoRA (round 3b; user request 2026-09-15).

Per open slot of a state the judge sees:
  conf    top-1 probability of the fp32 softmax with [MASK] suppressed (the full judge's commit confidence)
  correct that top-1 token == the target token
  nll     true-token NLL under the same MASK-suppressed softmax
  kind    0 mask (slot holds [MASK]) / 1 confirm (slot holds the draft token and it equals the target) /
          2 override (slot holds a draft token that is wrong)
  t       the state's mask ratio = #[MASK] slots / #valid slots of the active block

Reported (holdout, every eval):
  * equal-mass ECE (bootstrap CI) and top-label NLL of MASKED positions, bucketed by t, plus mean conf - acc
    (+ = overconfident, - = underconfident);
  * reliability diagrams at t ~ 0.1 / 0.5 / 0.9;
  * joint calibration: a parallel decoder commits several slots from ONE factorised forward, i.e. it assumes they are
    independent. For the slots committed together we compare the empirical P(all correct) with the product of their
    confidences, as a function of how many are committed at once (k). Two commit rules: the decoder's own (p > tau,
    at least one forced) and top-k by p.
Training steps get a cheap running version: an equal-width 20-bin confidence histogram with per-bin accuracy
(-> ECE, reliability) and CE / accuracy / confidence per slot kind.

"Correct" is judged against the training trajectory's token (the model's own verified-correct decode), which is not
the only valid continuation -- accuracy is a lower bound; calibration shape across t, k and steps is the signal.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

KINDS = ("mask", "confirm", "override")
T_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
T_TARGETS = (0.1, 0.5, 0.9)
T_WINDOW = 0.1
TOPK = (1, 2, 4, 8, 16, 32)
K_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32))
N_TRAIN_BINS = 20
NAN = float("nan")


# ------------------------------------------------------------------------------------------------------------------
# per-slot view (torch, on device)
# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def slot_view(logits_fp32: torch.Tensor, targets: torch.Tensor, block_ids: torch.Tensor, open_mask: torch.Tensor,
              valid: torch.Tensor, mask_id: int):
    """logits [n, V] of the open rows, targets [n], block_ids/open_mask/valid [block] (the active block, CPU ok).

    -> conf [n], correct [n] bool, nll [n], kind [n] long, t float."""
    dev = logits_fp32.device
    z = torch.cat([logits_fp32[:, :mask_id], logits_fp32[:, mask_id + 1:]], dim=1)
    logp = F.log_softmax(z, dim=-1)
    lp_top, i_top = logp.max(dim=-1)
    i_top = i_top + (i_top >= mask_id).long()                     # back to vocab ids
    tcol = (targets - (targets > mask_id).long()).clamp_min(0)
    nll = -logp.gather(1, tcol.unsqueeze(1)).squeeze(1)
    inp = block_ids[open_mask].to(dev)
    tg = targets.to(dev)
    kind = torch.where(inp == mask_id, 0, torch.where(inp == tg, 1, 2))
    t = float((block_ids[valid] == mask_id).sum()) / max(1, int(valid.sum()))
    return lp_top.exp(), i_top == tg, nll, kind, t


# ------------------------------------------------------------------------------------------------------------------
# training-step accumulator (fixed-size float64 vector: all_reduce(SUM)-able)
# ------------------------------------------------------------------------------------------------------------------
def train_accum_zeros(device) -> torch.Tensor:
    return torch.zeros(3 * N_TRAIN_BINS + 4 * len(KINDS), dtype=torch.float64, device=device)


@torch.no_grad()
def train_accum_add(acc: torch.Tensor, conf, correct, nll, kind) -> None:
    N, K = N_TRAIN_BINS, len(KINDS)
    c = conf.double(); y = correct.double(); b = (c * N).long().clamp(0, N - 1)
    acc[0:N].index_add_(0, b, torch.ones_like(c))
    acc[N:2 * N].index_add_(0, b, y)
    acc[2 * N:3 * N].index_add_(0, b, c)
    o = 3 * N
    for k in range(K):
        m = (kind == k).double()
        acc[o + k] += m.sum()
        acc[o + K + k] += (nll.double() * m).sum()
        acc[o + 2 * K + k] += (y * m).sum()
        acc[o + 3 * K + k] += (c * m).sum()


def train_accum_summary(acc: torch.Tensor) -> Dict[str, object]:
    """Scalars (ece_train, conf/acc/gap, per-kind ce/acc/conf/frac) + the histogram lists."""
    a = acc.detach().double().cpu().numpy()
    N, K = N_TRAIN_BINS, len(KINDS)
    n, cor, cs = a[0:N], a[N:2 * N], a[2 * N:3 * N]
    tot = n.sum()
    out: Dict[str, object] = {}
    if tot > 0:
        nz = n > 0
        out["ece_train"] = float((np.abs(cor[nz] / n[nz] - cs[nz] / n[nz]) * n[nz]).sum() / tot)
        out["conf_train"] = float(cs.sum() / tot); out["acc_argmax_nomask"] = float(cor.sum() / tot)
        out["overconf_gap"] = out["conf_train"] - out["acc_argmax_nomask"]
    o = 3 * N
    kn = a[o:o + K]
    for k, name in enumerate(KINDS):
        if kn[k] > 0:
            out[f"ce_{name}"] = float(a[o + K + k] / kn[k]); out[f"acc_{name}"] = float(a[o + 2 * K + k] / kn[k])
            out[f"conf_{name}"] = float(a[o + 3 * K + k] / kn[k])
        out[f"frac_{name}"] = float(kn[k] / kn.sum()) if kn.sum() > 0 else NAN
    if kn.sum() > 0:
        out["ce_unweighted"] = float(a[o + K:o + 2 * K].sum() / kn.sum())
    out["conf_hist_n"] = n.tolist(); out["conf_hist_correct"] = cor.tolist(); out["conf_hist_conf"] = cs.tolist()
    return out


# ------------------------------------------------------------------------------------------------------------------
# holdout: per-state records (numpy) for the calibration eval
# ------------------------------------------------------------------------------------------------------------------
def state_record(conf, correct, nll, kind, t: float, regime: str, tau: float = 0.95):
    c = conf.double().cpu().numpy(); y = correct.cpu().numpy().astype(bool)
    slots = {"conf": c.astype(np.float32), "correct": y, "nll": nll.double().cpu().numpy().astype(np.float32),
             "kind": kind.cpu().numpy().astype(np.int8), "t": np.full(c.size, t, np.float32),
             "regime": np.full(c.size, {"sample": 0, "draft_pass1": 1, "allmask_pass1": 2}[regime], np.int8)}
    joint = []                                     # (rule 0=tau 1=topk, k, all_correct, prod_conf, t)
    if c.size:
        lc = np.log(np.clip(c, 1e-12, 1.0))
        commit = np.nonzero(c > tau)[0]
        if commit.size == 0:
            commit = np.array([int(np.argmax(c))])
        joint.append((0, commit.size, bool(y[commit].all()), float(np.exp(lc[commit].sum())), t))
        order = np.argsort(-c, kind="stable")
        for k in TOPK:
            if k <= c.size:
                s = order[:k]
                joint.append((1, k, bool(y[s].all()), float(np.exp(lc[s].sum())), t))
    return slots, joint


def merge_payloads(payloads: List[dict]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    keys = ("conf", "correct", "nll", "kind", "t", "regime")
    slots = {k: np.concatenate([p["slots"][k] for p in payloads if p and p.get("slots") and len(p["slots"][k])] or
                               [np.zeros(0, np.float32)]) for k in keys}
    rows = [r for p in payloads if p for r in p.get("joint", [])]
    joint = np.array(rows, dtype=np.float64).reshape(-1, 5)
    return slots, joint


# ------------------------------------------------------------------------------------------------------------------
# metrics (numpy)
# ------------------------------------------------------------------------------------------------------------------
def equal_mass_ece(conf: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    n = conf.size
    if n == 0:
        return NAN
    o = np.argsort(conf, kind="stable"); cs = conf[o].astype(float); ys = y[o].astype(float)
    return float(sum(len(p) / n * abs(ys[p].mean() - cs[p].mean())
                     for p in np.array_split(np.arange(n), min(n_bins, n)) if len(p)))


def top_label_nll(conf: np.ndarray, y: np.ndarray) -> float:
    if conf.size == 0:
        return NAN
    p = np.clip(conf.astype(float), 1e-6, 1 - 1e-6); yy = y.astype(float)
    return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())


def bootstrap(fn, conf, y, n_boot: int = 200, seed: int = 0) -> List[float]:
    n = conf.size
    if n < 2:
        return [NAN, NAN]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(fn(conf[i], y[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [NAN, NAN]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def group(conf, y, nll=None, n_boot: int = 200, seed: int = 0) -> Dict[str, object]:
    n = int(conf.size)
    if n == 0:
        return {"n": 0}
    return {"n": n, "acc": float(y.mean()), "conf": float(conf.mean()), "gap": float(conf.mean() - y.mean()),
            "ece": equal_mass_ece(conf, y), "ece_ci": bootstrap(equal_mass_ece, conf, y, n_boot, seed),
            "nll_top": top_label_nll(conf, y), "nll_top_ci": bootstrap(top_label_nll, conf, y, n_boot, seed + 1),
            "nll_true": float(nll.mean()) if nll is not None and nll.size else NAN}


def reliability(conf, y, n_bins: int = 10) -> List[Dict[str, float]]:
    n = conf.size
    if n == 0:
        return []
    o = np.argsort(conf, kind="stable")
    return [{"conf": float(conf[o][p].mean()), "acc": float(y[o][p].astype(float).mean()), "n": int(len(p))}
            for p in np.array_split(np.arange(n), min(n_bins, n)) if len(p)]


def analyze(slots: Dict[str, np.ndarray], joint: np.ndarray, step: int, tau: float = 0.95) -> Dict[str, object]:
    conf, y, nll, kind, t = slots["conf"], slots["correct"].astype(bool), slots["nll"], slots["kind"], slots["t"]
    res: Dict[str, object] = {"step": int(step), "tau": tau, "n_slots": int(conf.size)}
    res["all_open"] = group(conf, y, nll)
    res["by_kind"] = {name: group(conf[kind == k], y[kind == k], nll[kind == k], seed=10 + k) for k, name in enumerate(KINDS)}
    m = kind == 0
    by_t = []
    for i, (lo, hi) in enumerate(zip(T_EDGES[:-1], T_EDGES[1:])):
        sel = m & (t >= lo) & ((t < hi) if hi < 1.0 else (t <= hi))
        g = group(conf[sel], y[sel], nll[sel], seed=100 + i)
        g.update({"t_lo": lo, "t_hi": hi, "t_mean": float(t[sel].mean()) if sel.any() else NAN})
        by_t.append(g)
    res["masked_by_t"] = by_t
    rel = {}
    for tt in T_TARGETS:
        sel = m & (np.abs(t - tt) <= T_WINDOW + 1e-6)
        rel[f"{tt:.1f}"] = {"n": int(sel.sum()), "ece": equal_mass_ece(conf[sel], y[sel]), "bins": reliability(conf[sel], y[sel])}
    res["reliability"] = rel
    jt = {"tau_rule": [], "top_k": []}
    if joint.size:
        rule, kk, allc, prod = joint[:, 0], joint[:, 1], joint[:, 2], joint[:, 3]
        for lo, hi in K_BUCKETS:
            sel = (rule == 0) & (kk >= lo) & (kk <= hi)
            n = int(sel.sum())
            jt["tau_rule"].append({"k_lo": lo, "k_hi": hi, "n_states": n,
                                   "mean_k": float(kk[sel].mean()) if n else NAN,
                                   "p_all_correct": float(allc[sel].mean()) if n else NAN,
                                   "p_all_correct_ci": wilson(int(allc[sel].sum()), n),
                                   "prod_conf": float(prod[sel].mean()) if n else NAN,
                                   "gap": float(prod[sel].mean() - allc[sel].mean()) if n else NAN})
        for k in TOPK:
            sel = (rule == 1) & (kk == k)
            n = int(sel.sum())
            jt["top_k"].append({"k": k, "n_states": n, "p_all_correct": float(allc[sel].mean()) if n else NAN,
                                "p_all_correct_ci": wilson(int(allc[sel].sum()), n),
                                "prod_conf": float(prod[sel].mean()) if n else NAN,
                                "gap": float(prod[sel].mean() - allc[sel].mean()) if n else NAN})
    res["joint"] = jt
    return res


def wandb_scalars(res: Dict[str, object]) -> Dict[str, float]:
    """Flat scalars for W&B / jsonl (prefix added by the caller)."""
    out: Dict[str, float] = {}
    for key, g in [("all", res["all_open"])] + [(k, v) for k, v in res["by_kind"].items()]:
        for f in ("ece", "nll_top", "nll_true", "gap", "acc", "conf"):
            if f in g:
                out[f"{key}_{f}"] = g[f]
    for g in res["masked_by_t"]:
        tag = f"t{g['t_lo']:.1f}-{g['t_hi']:.1f}"
        for f in ("ece", "nll_top", "gap", "n"):
            if f in g:
                out[f"masked_{tag}_{f}"] = g[f]
    for r in res["joint"]["tau_rule"]:
        tag = f"k{r['k_lo']}-{r['k_hi']}"
        out[f"joint_tau_{tag}_p_all"] = r["p_all_correct"]; out[f"joint_tau_{tag}_prod"] = r["prod_conf"]
        out[f"joint_tau_{tag}_gap"] = r["gap"]; out[f"joint_tau_{tag}_n"] = r["n_states"]
    for r in res["joint"]["top_k"]:
        out[f"joint_top{r['k']}_p_all"] = r["p_all_correct"]; out[f"joint_top{r['k']}_prod"] = r["prod_conf"]
        out[f"joint_top{r['k']}_gap"] = r["gap"]
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ------------------------------------------------------------------------------------------------------------------
# plots (matplotlib, Agg)
# ------------------------------------------------------------------------------------------------------------------
def plot_all(res: Dict[str, object], out_dir: str, train_hist: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    step = res["step"]; paths = {}

    def ci_err(vals, cis):          # clamp at 0: a point can sit outside its bootstrap/Wilson CI on small n
        lo = [max(0.0, v - c[0]) if np.isfinite(v) and np.isfinite(c[0]) else 0.0 for v, c in zip(vals, cis)]
        hi = [max(0.0, c[1] - v) if np.isfinite(v) and np.isfinite(c[1]) else 0.0 for v, c in zip(vals, cis)]
        return [lo, hi]

    def safe(name, fn):             # one broken figure must not cost the others
        try:
            fn()
        except Exception as e:                                               # noqa: BLE001
            plt.close("all"); paths[f"error_{name}"] = f"{type(e).__name__}: {e}"

    # 1. ECE / top-label NLL / conf-acc gap vs mask ratio t (masked positions)
    def _fig_1():
        bt = [g for g in res["masked_by_t"]]
        x = [0.5 * (g["t_lo"] + g["t_hi"]) for g in bt]
        ece = [g.get("ece", NAN) for g in bt]; nllt = [g.get("nll_top", NAN) for g in bt]; gap = [g.get("gap", NAN) for g in bt]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].errorbar(x, ece, yerr=ci_err(ece, [g.get("ece_ci", [NAN, NAN]) for g in bt]), marker="o", capsize=4)
        ax[0].set(title="equal-mass ECE (15 bins), 95% bootstrap CI", xlabel="mask ratio t", ylabel="ECE")
        ax[1].errorbar(x, nllt, yerr=ci_err(nllt, [g.get("nll_top_ci", [NAN, NAN]) for g in bt]), marker="o", capsize=4, label="top-label NLL")
        ax[1].plot(x, [g.get("nll_true", NAN) for g in bt], "s--", label="true-token NLL")
        ax[1].set(title="NLL vs t", xlabel="mask ratio t"); ax[1].legend()
        ax[2].bar(x, gap, width=0.15, color=["tab:red" if (np.isfinite(v) and v > 0) else "tab:blue" for v in gap])
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set(title="conf − acc  (red over / blue under)", xlabel="mask ratio t")
        for xi, g in zip(x, bt):
            ax[2].annotate(f"n={g.get('n', 0)}", (xi, 0), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
        fig.suptitle(f"step {step}: calibration of MASKED positions vs mask ratio (holdout)")
        fig.tight_layout(); p = os.path.join(out_dir, f"step{step:06d}_ece_nll_vs_t.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["ece_nll_vs_t"] = p

    safe('ece_nll_vs_t', _fig_1)

    # 2. reliability grid at t ~ 0.1 / 0.5 / 0.9
    def _fig_2():
        fig, ax = plt.subplots(1, len(T_TARGETS), figsize=(5 * len(T_TARGETS), 4.4))
        for a, (tt, r) in zip(np.atleast_1d(ax), res["reliability"].items()):
            a.plot([0, 1], [0, 1], "k--", lw=0.8)
            if r["bins"]:
                cs = [b["conf"] for b in r["bins"]]; ac = [b["acc"] for b in r["bins"]]
                a.plot(cs, ac, "o-")
                for b in r["bins"]:
                    a.plot([b["conf"], b["conf"]], [b["conf"], b["acc"]], color="tab:red" if b["conf"] > b["acc"] else "tab:blue", lw=1)
            a.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence (equal-mass bins)", ylabel="accuracy",
                  title=f"t ≈ {tt} (±{T_WINDOW}), n={r['n']}, ECE={r['ece']:.3f}" if r["n"] else f"t ≈ {tt}: no positions")
        fig.suptitle(f"step {step}: reliability diagrams of masked positions"); fig.tight_layout()
        p = os.path.join(out_dir, f"step{step:06d}_reliability_grid.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["reliability_grid"] = p

    safe('reliability_grid', _fig_2)

    # 3. joint calibration vs tokens committed at once
    def _fig_3():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        tr = [r for r in res["joint"]["tau_rule"] if r["n_states"]]
        if tr:
            lab = [f"{r['k_lo']}" if r["k_lo"] == r["k_hi"] else f"{r['k_lo']}–{r['k_hi']}" for r in tr]
            xs = np.arange(len(tr)); emp = [r["p_all_correct"] for r in tr]
            ax[0].errorbar(xs, emp, yerr=ci_err(emp, [r["p_all_correct_ci"] for r in tr]), marker="o", capsize=4, label="empirical P(all correct)")
            ax[0].plot(xs, [r["prod_conf"] for r in tr], "s--", label="product of confidences")
            ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{l}\n(n={r['n_states']})" for l, r in zip(lab, tr)], fontsize=8)
        ax[0].set(title=f"decoder rule: commit p > {res['tau']} (≥1 forced)", xlabel="tokens committed in the same pass", ylim=(0, 1.02)); ax[0].legend()
        tk = [r for r in res["joint"]["top_k"] if r["n_states"]]
        if tk:
            xs = np.arange(len(tk)); emp = [r["p_all_correct"] for r in tk]
            ax[1].errorbar(xs, emp, yerr=ci_err(emp, [r["p_all_correct_ci"] for r in tk]), marker="o", capsize=4, label="empirical P(all correct)")
            ax[1].plot(xs, [r["prod_conf"] for r in tk], "s--", label="product of confidences")
            ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{r['k']}\n(n={r['n_states']})" for r in tk], fontsize=8)
        ax[1].set(title="top-k most confident slots committed together", xlabel="k", ylim=(0, 1.02)); ax[1].legend()
        fig.suptitle(f"step {step}: joint calibration (gap = product − empirical; > 0 = independence overestimates)")
        fig.tight_layout(); p = os.path.join(out_dir, f"step{step:06d}_joint_calibration.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["joint_calibration"] = p

    safe('joint_calibration', _fig_3)

    # 4. training confidence histogram + per-bin accuracy since the previous eval
    def _fig_4():
        if train_hist and sum(train_hist.get("conf_hist_n", [])) > 0:
            n = np.array(train_hist["conf_hist_n"]); cor = np.array(train_hist["conf_hist_correct"]); cs = np.array(train_hist["conf_hist_conf"])
            edges = np.linspace(0, 1, N_TRAIN_BINS + 1); mid = 0.5 * (edges[:-1] + edges[1:]); nz = n > 0
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
            ax[0].bar(mid, n / n.sum(), width=1 / N_TRAIN_BINS * 0.9); ax[0].set_yscale("log")
            ax[0].set(title="top-1 confidence distribution (training slots)", xlabel="confidence", ylabel="share of slots")
            ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].plot(cs[nz] / n[nz], cor[nz] / n[nz], "o-")
            ece_tr = float((np.abs(cor[nz] / n[nz] - cs[nz] / n[nz]) * n[nz]).sum() / n.sum())
            ax[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence (20 equal-width bins)", ylabel="accuracy",
                      title=f"training reliability, ECE={ece_tr:.3f}")
            fig.suptitle(f"steps up to {step}: training slots since the previous eval"); fig.tight_layout()
            p = os.path.join(out_dir, f"step{step:06d}_train_confidence.png"); fig.savefig(p, dpi=110); plt.close(fig)
            paths["train_confidence"] = p

    safe('train_confidence', _fig_4)

    # 5. trends across every eval so far
    def _fig_5():
        hist = []
        for f in sorted(glob.glob(os.path.join(out_dir, "step*.json"))):
            try:
                hist.append(json.load(open(f)))
            except Exception:
                continue
        if len(hist) >= 2:
            steps = [h["step"] for h in hist]
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for i, (lo, hi) in enumerate(zip(T_EDGES[:-1], T_EDGES[1:])):
                ax[0].plot(steps, [h["masked_by_t"][i].get("ece", NAN) for h in hist], "o-", label=f"t {lo:.1f}–{hi:.1f}")
            ax[0].set(title="masked-position ECE by t", xlabel="step"); ax[0].legend(fontsize=7)
            for name in KINDS:
                ax[1].plot(steps, [h["by_kind"][name].get("gap", NAN) for h in hist], "o-", label=name)
            ax[1].axhline(0, color="k", lw=0.8); ax[1].set(title="conf − acc by slot kind", xlabel="step"); ax[1].legend(fontsize=7)
            for j, (lo, hi) in enumerate(K_BUCKETS):
                ax[2].plot(steps, [h["joint"]["tau_rule"][j]["gap"] if h["joint"]["tau_rule"] else NAN for h in hist], "o-",
                           label=f"k {lo}–{hi}")
            ax[2].axhline(0, color="k", lw=0.8); ax[2].set(title="joint gap (prod − empirical), decoder rule", xlabel="step"); ax[2].legend(fontsize=7)
            fig.tight_layout(); p = os.path.join(out_dir, "trend_calibration.png"); fig.savefig(p, dpi=110); plt.close(fig)
            paths["trend"] = p
    safe('trend', _fig_5)

    return paths
