#!/usr/bin/env python
"""Phase 2, NO-POOL drafter swap: accuracy, mechanism and speed of each new drafter vs Reference A (SDAR-1.7B drafter +
the 128M judge, no pool) with Reference B (stock 30B decode) for context. CPU only; rerun any time:

    CUDA_VISIBLE_DEVICES= /venv/p2/bin/python code/phase2/analyze_nopool_p2.py

Inputs (read only):
    Reference A  logs/phase2/ref/S0_128M_nopool_by_set/<set>/*.jsonl    (Phase 1 records: counts only, no prob logs)
    Reference B  logs/phase2/ref/stock_flat_by_set/<set>/<prefix>_*.jsonl
    new arms     logs/phase2/decode/{F2_nopool,S1_nopool}/<set>/*.shard*of8.jsonl (+ judge/, traces/ npz per item)
A new arm's set is analysed only once its records are complete (all items present); a set still running is shown
as progress with a provisional paired delta. Outputs (this directory): REPORT.md, analysis.json, *.png.

Definitions (as in code/phase2/gate_p2.py): experts/token = sum unique_expert_layer_loads / sum n_gen_to_eos;
charged = (expert loads + drafter forwards x drafter cost per forward) / tokens; passes/block = MoE forwards except
prefill and kv flush per block; agreement = share of a block's final tokens equal to the draft; kept/block = final
tokens equal to the draft per block; forced/block = judge passes in which no slot cleared 0.95 (the best slot was
committed anyway) per block. Paired delta: bootstrap over items (repo rule: 2,000 resamples, seed 0; pooled = resampled
within each set); McNemar = exact two-sided binomial test on the discordant pairs.

Stock-trajectory analysis ("on-trajectory" blocks). The 128M judge was trained to reproduce the stock 30B decode
(block 32, threshold 0.95) given a correct prefix. For every block whose prefix is token-identical to the stock
decode's, the stock decode's block is therefore exactly the judge's training target for that state. For those
blocks we measure, per arm, how often the final tokens equal the stock tokens (both arms, from records), and for the
new arms (judge + trace npz) the judge's calibration against that target: top-1 probability vs argmax == stock
token at every read, commit precision by commit type (confirm = commit of the draft token, override, forced), the
draft's correctness and the judge's probability of the draft token. Reference A has no per-read probabilities, so
these calibration numbers are new-arm only.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs/phase2/analysis/nopool")
REF_A = os.path.join(ROOT, "logs/phase2/ref/S0_128M_nopool_by_set")
REF_B = os.path.join(ROOT, "logs/phase2/ref/stock_flat_by_set")
DEC = os.path.join(ROOT, "logs/phase2/decode")
CODE = os.path.join(ROOT, "code")
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

# set, size, record-file prefix, domain
SETS = [("humaneval", 164, "humaneval", "code"), ("mbpp", 257, "mbpp", "code"), ("gsm8k500", 500, "gsm8k", "math"),
        ("ifeval541", 541, "ifeval", "if"), ("math500", 500, "math500", "math"), ("gpqa", 198, "gpqa", "science"),
        ("aime", 90, "aime", "math")]
SIZE = {s: n for s, n, _, _ in SETS}
PREFIX = {s: p for s, _, p, _ in SETS}
DOMAIN = {s: d for s, _, _, d in SETS}
NEW_ARMS = [("F2_nopool", "F2", "CoDA-1.7B"), ("S1_nopool", "S1", "Qwen3-0.6B bd3lm")]
EQ = {"S0": 364.64, "F2": 364.64, "S1": 126.32}       # charged expert-equivalents per drafter forward (drafters.py)
TAU = 0.95
BOOT = dict(resamples=2000, seed=0)
EOS_IDS = (151645, 151643)
COLORS = {"A": "#2a78d6", "F2_nopool": "#eb6834", "S1_nopool": "#1baf7a", "stock": "#898781"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
KEEP = ("idx", "correct", "score_error", "prompt_tokens", "pred", "gold", "gen_ids", "n_gen_nonmask", "n_gen_to_eos",
        "stopped_eos", "truncated", "n_blocks", "n_forward", "n_prefill", "n_flush", "n_judge", "n_denoise",
        "n_valid_per_block", "unique_expert_layer_loads", "seconds", "drafter_n_by_kind", "tentative_per_block",
        "remask_per_block", "full_commit_eq_draft_per_block", "full_fallback_per_block", "committed_judge_per_block",
        "forced_per_block", "full_passes_per_block", "draft_agreement_per_block", "commit_probe",
        "drafter_eq_per_forward", "gpu", "passes_per_block", "stock_passes_per_block")


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def load_records(d: str, s: str) -> dict:
    """{idx: slim record} from <d>/<s>/<prefix>_*.<s>.shard*of*.jsonl (running files and other cells skipped);
    a scored record wins over an unscored duplicate."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, s, f"{PREFIX[s]}_*.{s}.shard*of*.jsonl"))):
        if f.endswith(".running.jsonl"):
            continue
        with open(f) as fh:
            for line in fh:
                if not line.endswith("\n"):
                    break                                   # partly written last line
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                i = int(r["idx"])
                if i in recs and recs[i].get("correct") is not None and r.get("correct") is None:
                    continue
                recs[i] = {k: r.get(k) for k in KEEP}
    return recs


# ---------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------
def mcnemar(a, b):
    """exact two-sided McNemar on paired 0/1 lists: (wins of a, wins of b, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a), np.asarray(b)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def paired(x: dict, y: dict, idxs=None) -> dict:
    """delta = acc(x) - acc(y) over shared scored items, bootstrap CI, McNemar."""
    ids = sorted(i for i in (idxs if idxs is not None else x) if i in x and i in y
                 and x[i].get("correct") is not None and y[i].get("correct") is not None)
    a = np.array([int(x[i]["correct"]) for i in ids]); b = np.array([int(y[i]["correct"]) for i in ids])
    if not len(ids):
        return {"n": 0}
    d = a - b
    rng = np.random.default_rng(BOOT["seed"])
    boots = d[rng.integers(0, len(d), size=(BOOT["resamples"], len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    w, l, p = mcnemar(a, b)
    return {"n": len(ids), "acc_x": float(a.mean()), "acc_y": float(b.mean()), "delta": float(d.mean()),
            "lo": float(lo), "hi": float(hi), "x_only": w, "y_only": l, "mcnemar_p": p}


def paired_pooled(pairs: list) -> dict:
    """pairs: [(x_recs, y_recs)] per set -> micro (item-pooled) and macro (mean of set deltas) deltas, bootstrap
    resampled within each set, McNemar on all discordant pairs."""
    ds, A, B = [], [], []
    for x, y in pairs:
        ids = sorted(i for i in x if i in y and x[i].get("correct") is not None and y[i].get("correct") is not None)
        a = np.array([int(x[i]["correct"]) for i in ids]); b = np.array([int(y[i]["correct"]) for i in ids])
        if len(ids):
            ds.append(a - b); A.append(a); B.append(b)
    if not ds:
        return {"n": 0}
    N = sum(len(d) for d in ds)
    rng = np.random.default_rng(BOOT["seed"])
    R = BOOT["resamples"]
    sums = np.zeros(R); means = np.zeros(R)
    for d in ds:
        bs = d[rng.integers(0, len(d), size=(R, len(d)))]
        sums += bs.sum(1); means += bs.mean(1)
    micro, macro = sums / N, means / len(ds)
    w, l, p = mcnemar(np.concatenate(A), np.concatenate(B))
    return {"n": int(N), "n_sets": len(ds), "acc_x": float(np.concatenate(A).mean()), "acc_y": float(np.concatenate(B).mean()),
            "delta": float(np.concatenate(ds).mean()), "lo": float(np.quantile(micro, .025)), "hi": float(np.quantile(micro, .975)),
            "macro_delta": float(np.mean([d.mean() for d in ds])), "macro_lo": float(np.quantile(macro, .025)),
            "macro_hi": float(np.quantile(macro, .975)), "x_only": w, "y_only": l, "mcnemar_p": p}


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ---------------------------------------------------------------------------------------------------------------
# per-record features
# ---------------------------------------------------------------------------------------------------------------
_TOK = None
_ALPHA = None


def tokenizer():
    global _TOK, _ALPHA
    if _TOK is None:
        import research_protocol as P
        from transformers import AutoTokenizer
        m = P.MODELS["b32"]
        _TOK = AutoTokenizer.from_pretrained(m["repo"], trust_remote_code=True, local_files_only=True,
                                             revision=m["revision"], cache_dir=m["cache_dir"])
        toks = _TOK.convert_ids_to_tokens(list(range(len(_TOK))))
        _ALPHA = np.array([bool(re.search(r"[A-Za-z]{2,}", t or "")) for t in toks] + [False] * 1024)
    return _TOK


def gen_to_eos(r):
    g = r.get("gen_ids") or []
    n = int(r.get("n_gen_to_eos") or len(g))
    return g[:n]


def item_features(r: dict, s: str) -> dict:
    """degenerate-output and task-specific flags of one record."""
    tokenizer()
    g = gen_to_eos(r)
    ga = np.asarray(g, dtype=np.int64)
    stut = int(((ga[1:] == ga[:-1]) & _ALPHA[ga[1:]]).sum()) if len(ga) > 1 else 0
    rep = 0
    if len(g) >= 12:
        c = Counter(tuple(g[i:i + 12]) for i in range(len(g) - 11))
        rep = max(c.values())
    f = {"len": len(g), "no_eos": int(not r.get("stopped_eos")), "truncated": int(bool(r.get("truncated"))),
         "stutter": stut, "stutter_any": int(stut > 0), "repetitive": int(rep >= 4)}
    pred = r.get("pred") or ""
    if DOMAIN[s] == "code":
        from oc_datasets.mbpp import MBPPEvaluator
        code = MBPPEvaluator()._process_answer(str(pred))
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(code)
            f["syntax_error"] = 0
        except Exception:
            f["syntax_error"] = 1
    elif DOMAIN[s] == "math":
        f["no_boxed"] = int("\\boxed" not in pred)
    elif s == "gpqa":
        from analyze_s1b import _gpqa_letter
        f["no_letter"] = int(_gpqa_letter(pred) == "")
    return f


TASK_FLAG = {"code": "syntax_error", "math": "no_boxed", "science": "no_letter", "if": None}


# ---------------------------------------------------------------------------------------------------------------
# arm aggregates (records only)
# ---------------------------------------------------------------------------------------------------------------
def agg(R: dict, eq: float, feats: dict | None = None) -> dict:
    R = list(R.values())
    if not R:
        return {"n": 0}
    gen = sum(int(r["n_gen_to_eos"]) for r in R)
    uniq = sum(int(r["unique_expert_layer_loads"]) for r in R)
    n_dr = sum(sum((r.get("drafter_n_by_kind") or {}).values()) for r in R)
    blocks = sum(int(r["n_blocks"]) for r in R)
    passes = sum(int(r["n_forward"]) - int(r["n_prefill"]) - int(r["n_flush"]) for r in R)
    nv = sum(sum(r["n_valid_per_block"]) for r in R)
    sc = [int(r["correct"]) for r in R if r.get("correct") is not None]
    ag = [a for r in R for a in (r.get("draft_agreement_per_block") or []) if a is not None]
    kept = sum(a * v for r in R for a, v in zip(r.get("draft_agreement_per_block") or [], r["n_valid_per_block"])
               if a is not None)
    fp = [x for r in R for b in (r.get("full_passes_per_block") or []) for x in b]
    judged = bool(fp)
    out = {"n": len(R), "acc": float(np.mean(sc)) if sc else None, "n_scored": len(sc),
           "exp_tok": uniq / gen, "drafter_fwd_tok": n_dr / gen, "charged_exp_tok": (uniq + n_dr * eq) / gen,
           "passes_per_block": passes / blocks, "blocks_per_item": blocks / len(R), "tokens_per_item": gen / len(R),
           "ms_per_token": 1000.0 * sum(r["seconds"] for r in R) / gen}
    if judged:
        committed = sum(sum(r.get("committed_judge_per_block") or []) for r in R)
        ceq = sum(sum(r.get("full_commit_eq_draft_per_block") or []) for r in R)
        first = sum(b[0]["commit"] for r in R for b in r["full_passes_per_block"] if b)
        n_forced = sum(int(x["forced"]) for x in fp)
        a0 = [r["draft_agreement_per_block"][0] for r in R if r["draft_agreement_per_block"] and r["draft_agreement_per_block"][0] is not None]
        al = [a for r in R for a in (r["draft_agreement_per_block"] or [])[1:] if a is not None]
        p0 = [len(r["full_passes_per_block"][0]) for r in R if r["full_passes_per_block"]]
        pl = [len(b) for r in R for b in r["full_passes_per_block"][1:]]
        out.update({"agreement_block0": float(np.mean(a0)) if a0 else None, "agreement_later": float(np.mean(al)) if al else None,
                    "passes_block0": float(np.mean(p0)) if p0 else None, "passes_later": float(np.mean(pl)) if pl else None,
                    "block0_share_of_blocks": len(p0) / blocks, "n_block0": len(p0), "judge_passes_total": len(fp)})
        out.update({"agreement": float(np.mean(ag)) if ag else None, "agreement_tok": kept / nv,
                    "kept_per_block": kept / blocks, "forced_per_block": n_forced / blocks,
                    "blocks_with_forced": float(np.mean([f for r in R for f in r["forced_per_block"]])),
                    "forced_share_of_commits": n_forced / committed if committed else None,
                    "first_pass_commit_share": first / nv, "confirm_share": ceq / committed if committed else None,
                    "remask_per_block": sum(sum(r["remask_per_block"]) for r in R) / blocks,
                    "tentative_per_block": sum(sum(r["tentative_per_block"]) for r in R) / blocks,
                    "fallback_per_block": sum(sum(r["full_fallback_per_block"]) for r in R) / blocks})
    if feats:
        F = [feats[r["idx"]] for r in R if r["idx"] in feats]
        for k in ("no_eos", "truncated", "stutter_any", "repetitive", "syntax_error", "no_boxed", "no_letter"):
            v = [f[k] for f in F if k in f]
            if v:
                out[k] = float(np.mean(v))
        L = sum(f["len"] for f in F)
        out["stutter_per_1k"] = 1000.0 * sum(f["stutter"] for f in F) / L if L else None
    return out


def commit_probe_agg(R: dict) -> dict | None:
    """Reference A's per-item commit summaries (Phase 1 `commit_probe`) pooled."""
    cps = [r["commit_probe"] for r in R.values() if r.get("commit_probe")]
    if not cps:
        return None
    n = lambda k: sum(c.get(k, 0) for c in cps)
    wm = lambda pk, nk: (sum(c[pk] * c[nk] for c in cps if c.get(pk) is not None and c.get(nk))
                         / max(1, sum(c[nk] for c in cps if c.get(pk) is not None and c.get(nk))))
    hist = np.sum([c["hist"] for c in cps], 0)
    nf = n("n_forced")
    return {"n_commit": n("n_commit"), "confirm_share": n("n_confirm") / max(1, n("n_commit")),
            "forced_share": nf / max(1, n("n_commit")), "mean_p_confirm": wm("mean_p_confirm", "n_confirm"),
            "mean_p_override": wm("mean_p_override", "n_override"), "mean_p_forced": wm("mean_p_forced", "n_forced"),
            "hist_bins": cps[0]["hist_bins"], "hist_share": (hist / hist.sum()).round(5).tolist()}


# ---------------------------------------------------------------------------------------------------------------
# stock trajectory (records only; both arms)
# ---------------------------------------------------------------------------------------------------------------
def traj_item(r: dict, st: dict) -> dict:
    """Blocks of `r` whose prefix equals the stock decode's: per block (b, gen offset, n relevant slots, matches).
    Walks until the first block that differs from stock (it is included) or until stock's output ends."""
    G, V, S = r["gen_ids"], r["n_valid_per_block"], st["gen_ids"]
    Ls = int(st["n_gen_to_eos"]) if st.get("stopped_eos") else len(S)
    Vs = st["n_valid_per_block"]
    off, blocks, div = 0, [], None
    for b, v in enumerate(V):
        if off >= Ls or b >= len(Vs) or Vs[b] != v:
            break
        hi = min(off + v, Ls)
        m = sum(int(x == y) for x, y in zip(G[off:hi], S[off:hi]))
        blocks.append((b, off, hi - off, m))
        if m != hi - off:
            div = b
            break
        off += v
    return {"blocks": blocks, "div": div, "identical": div is None and off >= Ls}


def traj_agg(R: dict, ST: dict, T: dict) -> dict:
    """slot / block fidelity to the stock trajectory over on-trajectory blocks, split by forced / agreement."""
    ns = nm = nb = ne = 0
    b0 = [0, 0]; bl = [0, 0]
    by_forced = {True: [0, 0], False: [0, 0]}
    by_ag = defaultdict(lambda: [0, 0])
    ident, corr_ident, corr_div, first_div = [], [], [], []
    for i, t in T.items():
        r = R[i]
        for b, off, n, m in t["blocks"]:
            ns += n; nm += m; nb += 1; ne += int(m == n)
            x = b0 if b == 0 else bl
            x[0] += 1; x[1] += int(m != n)
            f = bool(r["forced_per_block"][b]) if r.get("forced_per_block") else False
            by_forced[f][0] += 1; by_forced[f][1] += int(m == n)
            a = (r.get("draft_agreement_per_block") or [None] * 99)[b]
            if a is not None:
                k = min(int(a * 5), 4)
                by_ag[k][0] += 1; by_ag[k][1] += int(m == n)
        ident.append(int(t["identical"]))
        if r.get("correct") is not None:
            (corr_ident if t["identical"] else corr_div).append(int(r["correct"]))
        if t["div"] is not None:
            first_div.append(t["div"])
    return {"items": len(T), "on_traj_blocks": nb, "slot_match": nm / ns if ns else None,
            "block_exact": ne / nb if nb else None, "divergence_hazard": 1 - ne / nb if nb else None,
            "items_identical_to_stock": float(np.mean(ident)) if ident else None,
            "acc_if_identical": float(np.mean(corr_ident)) if corr_ident else None,
            "acc_if_diverged": float(np.mean(corr_div)) if corr_div else None,
            "median_first_div_block": float(np.median(first_div)) if first_div else None,
            "hazard_block0": b0[1] / b0[0] if b0[0] else None, "hazard_later": bl[1] / bl[0] if bl[0] else None,
            "block_exact_forced": by_forced[True][1] / by_forced[True][0] if by_forced[True][0] else None,
            "block_exact_unforced": by_forced[False][1] / by_forced[False][0] if by_forced[False][0] else None,
            "n_forced_blocks": by_forced[True][0], "n_unforced_blocks": by_forced[False][0],
            "block_exact_by_agreement": {f"{k/5:.1f}-{(k+1)/5:.1f}": [v[0], _r(v[1] / v[0])] for k, v in sorted(by_ag.items())}}


def survival(T: dict, maxb: int = 40) -> list:
    """share of items whose output is still token-identical to stock after block b (b = 0..maxb-1); an item that
    finished identical to stock stays counted as identical."""
    return [float(np.mean([t["div"] is None or t["div"] > b for t in T.values()])) if T else None for b in range(maxb)]


# ---------------------------------------------------------------------------------------------------------------
# judge npz (new arms only)
# ---------------------------------------------------------------------------------------------------------------
def judge_item(args):
    """Per item: rows for every judge read at an open slot of an on-trajectory block, and one row per committed slot
    of every block. Returns numpy arrays (concatenated by the caller)."""
    arm_dir, s, r, tr, st_gen = args
    i = r["idx"]
    jf = os.path.join(arm_dir, s, "judge", s, f"{i:05d}.npz")
    tf = os.path.join(arm_dir, s, "traces", s, f"{i:05d}.npz")
    if not (os.path.exists(jf) and os.path.exists(tf)):
        return None
    j, t = np.load(jf), np.load(tf)
    jp = j["judge_pass"].astype(int)
    blk = np.cumsum(jp == 0) - 1
    nb = int(blk.max()) + 1 if len(blk) else 0
    G = r["gen_ids"]
    on = {b: (off, n) for b, off, n, m in tr["blocks"]}
    reads, commits = [], []
    gmis = 0
    off = 0
    think0, agp, ags, agn = 0, 0, 0, 0
    for b in range(nb):
        rows = np.where(blk == b)[0]
        valid = t["valid"][b].astype(bool)
        vpos = np.where(valid)[0]
        tok, D, pdr = t["tokens"][b], t["draft_tokens"][b], t["p_draft"][b]
        Dv, Tv = D[valid], tok[valid]
        Lv = len(Dv)
        if b == 0:
            think0 = int(THINK_OPEN in Dv[:4])
        agp += int((Dv == Tv).sum()); agn += Lv
        ags += max(int((Dv[k:] == Tv[:Lv - k]).sum()) if k >= 0 else int((Dv[:Lv + k] == Tv[-k:]).sum())
                   for k in range(-min(8, Lv - 1), min(9, Lv))) if Lv else 0
        gmis += int(list(tok[valid]) != list(G[off:off + len(vpos)]))
        tgt = np.full(32, -1, np.int64)
        if b in on:
            o, n = on[b]
            tgt[vpos[:n]] = np.asarray(st_gen[o:o + n])
        op = j["judge_open"][rows]; p = j["judge_p"][rows].astype(np.float32)
        am = j["judge_argmax"][rows]; q = j["judge_q"][rows].astype(np.float32)
        fps = r["full_passes_per_block"][b]
        held = valid.copy()                                     # slot holds the draft token at this read
        clean = True                                            # every slot committed so far in this block == stock
        for k in range(len(rows)):
            nxt = op[k + 1] if k + 1 < len(rows) else np.zeros(32, bool)
            com = op[k] & ~nxt
            forced = bool(fps[k]["forced"]) if k < len(fps) else False
            sl = np.where(op[k])[0]
            if b in on:
                has = tgt[sl] >= 0
                ss = sl[has]
                if len(ss):
                    reads.append(np.stack([np.full(len(ss), k), p[k, ss], (am[k, ss] == tgt[ss]), q[k, ss],
                                           (D[ss] == tgt[ss]), com[ss], np.full(len(ss), forced), (am[k, ss] == D[ss]),
                                           held[ss], pdr[ss], np.full(len(ss), clean)], 1).astype(np.float32))
            cs = np.where(com)[0]
            typ = np.where(np.full(len(cs), forced), 2, np.where(am[k, cs] == D[cs], 0, 1))
            commits.append(np.stack([np.full(len(cs), k), p[k, cs], typ, tgt[cs] >= 0,
                                     np.where(tgt[cs] >= 0, am[k, cs] == tgt[cs], -1), D[cs] == tgt[cs],
                                     np.full(len(cs), clean and b in on), am[k, cs] == D[cs]], 1).astype(np.float32))
            if b in on:
                cm = com & (tgt >= 0)
                if (am[k][cm] != tgt[cm]).any():
                    clean = False
            held = op[k] & ~com & (am[k] == D)
        off += len(vpos)
    R = np.concatenate(reads) if reads else np.zeros((0, 11), np.float32)
    C = np.concatenate(commits) if commits else np.zeros((0, 8), np.float32)
    # pass-0 rows per on-trajectory slot (draft correctness, drafter confidence, judge q at the first read)
    return {"idx": i, "reads": R, "commits": C, "gen_mismatch": gmis, "think0": think0, "ag_plain": agp,
            "ag_shift": ags, "ag_n": agn}


RCOL = {k: n for n, k in enumerate(["pass", "p", "ok", "q", "draft_ok", "commit", "forced", "confirm", "held", "pdraft", "clean"])}
CCOL = {k: n for n, k in enumerate(["pass", "p", "type", "on", "ok", "draft_ok", "clean", "is_draft"])}
THINK_OPEN = 151667                                        # Qwen3 <think>
PBINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 1.0001]
ECE_BINS = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0001]


def reliability(conf, ok, bins):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append({"bin": f"[{lo:g},{min(hi, 1):g}{')' if hi <= 1 else ']'}", "n": int(m.sum()),
                         "conf": _r(conf[m].mean()), "acc": _r(ok[m].mean())})
    return rows


def ece(conf, ok, bins=ECE_BINS):
    e, N = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            e += m.sum() / N * abs(conf[m].mean() - ok[m].mean())
    return e if N else None


def judge_summary(Rd: np.ndarray, Cm: np.ndarray) -> dict:
    """calibration of the judge against the stock target on on-trajectory reads; commit-type precision."""
    out = {"n_reads": int(len(Rd)), "n_commits": int(len(Cm))}
    if len(Cm):
        typ = Cm[:, CCOL["type"]]
        out["commit_type_share"] = {n: _r((typ == t).mean()) for t, n in ((0, "confirm"), (1, "override"), (2, "forced"))}
        out["commit_mean_p"] = {n: _r(Cm[typ == t, CCOL["p"]].mean()) if (typ == t).any() else None
                                for t, n in ((0, "confirm"), (1, "override"), (2, "forced"))}
        isd = Cm[:, CCOL["is_draft"]] == 1                   # Reference A's commit_probe convention (forced included)
        out["commit_probe_style"] = {"confirm_share": _r(isd.mean()), "forced_share": _r((typ == 2).mean()),
                                     "mean_p_confirm": _r(Cm[isd, CCOL["p"]].mean()) if isd.any() else None,
                                     "mean_p_override": _r(Cm[~isd, CCOL["p"]].mean()) if (~isd).any() else None,
                                     "mean_p_forced": _r(Cm[typ == 2, CCOL["p"]].mean()) if (typ == 2).any() else None}
        h, _ = np.histogram(np.clip(Cm[:, CCOL["p"]], 0, 0.99999), bins=[0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 1.0])
        out["commit_hist_share"] = (h / h.sum()).round(5).tolist()
        on = Cm[:, CCOL["on"]] == 1
        c = Cm[on]
        if len(c):
            prec = {}
            for t, n in ((0, "confirm"), (1, "override"), (2, "forced")):
                m = c[:, CCOL["type"]] == t
                prec[n] = {"n": int(m.sum()), "precision": _r(c[m, CCOL["ok"]].mean()) if m.any() else None,
                           "mean_p": _r(c[m, CCOL["p"]].mean()) if m.any() else None}
            wrong = c[:, CCOL["ok"]] == 0
            out["commit_precision_on_traj"] = prec
            # where errors ENTER: wrong commits made while the block was still identical to stock
            cl = c[:, CCOL["clean"]] == 1
            ent = cl & wrong
            out["error_entry"] = {"n": int(ent.sum()), "rate_per_clean_commit": _r(ent.sum() / max(1, cl.sum())),
                                  "by_type": {n: _r(((c[:, CCOL["type"]] == t) & ent).sum() / max(1, ent.sum()))
                                              for t, n in ((0, "confirm"), (1, "override"), (2, "forced"))},
                                  "at_first_read": _r((c[ent, CCOL["pass"]] == 0).mean()) if ent.any() else None,
                                  "confirm_of_wrong_draft_per_clean_commit": _r(((c[:, CCOL["type"]] == 0) & ent).sum() / max(1, cl.sum()))}
            for nm_, sel in (("commit_precision_clean_state", c[:, CCOL["clean"]] == 1), ("commit_precision_pass0", c[:, CCOL["pass"]] == 0)):
                cc = c[sel]
                out[nm_] = {"all": {"n": int(len(cc)), "precision": _r(cc[:, CCOL["ok"]].mean()) if len(cc) else None}}
                for t, n in ((0, "confirm"), (1, "override"), (2, "forced")):
                    m = cc[:, CCOL["type"]] == t
                    out[nm_][n] = {"n": int(m.sum()), "precision": _r(cc[m, CCOL["ok"]].mean()) if m.any() else None,
                                   "mean_p": _r(cc[m, CCOL["p"]].mean()) if m.any() else None}
            out["commit_precision_all_on_traj"] = _r(c[:, CCOL["ok"]].mean())
            out["wrong_commit_share_by_type"] = {n: _r(((c[:, CCOL["type"]] == t) & wrong).sum() / max(1, wrong.sum()))
                                                 for t, n in ((0, "confirm"), (1, "override"), (2, "forced"))}
            out["confident_wrong_confirm_rate"] = _r(((c[:, CCOL["type"]] == 0) & wrong).mean())
            out["wrong_commits_pass0_share"] = _r((c[wrong, CCOL["pass"]] == 0).mean()) if wrong.any() else None
    if len(Rd):
        p, ok = Rd[:, RCOL["p"]], Rd[:, RCOL["ok"]]
        out["reliability_p_all_reads"] = reliability(p, ok, PBINS)
        out["ece_p_all_reads"] = _r(ece(p, ok))
        out["argmax_acc_all_reads"] = _r(ok.mean())
        cr = Rd[:, RCOL["clean"]] == 1
        out["reliability_p"] = reliability(p[cr], ok[cr], PBINS)
        out["ece_p"] = _r(ece(p[cr], ok[cr]))
        out["n_clean_reads"] = int(cr.sum())
        p0 = Rd[:, RCOL["pass"]] == 0
        r0 = Rd[p0]
        q, dok, pd = r0[:, RCOL["q"]], r0[:, RCOL["draft_ok"]], r0[:, RCOL["pdraft"]]
        out["pass0"] = {"n_slots": int(len(r0)), "draft_acc_vs_stock": _r(dok.mean()),
                        "judge_argmax_acc": _r(r0[:, RCOL["ok"]].mean()),
                        "ece_p": _r(ece(r0[:, RCOL["p"]], r0[:, RCOL["ok"]])),
                        "reliability_p": reliability(r0[:, RCOL["p"]], r0[:, RCOL["ok"]], PBINS),
                        "reliability_q_draft": reliability(q, dok, PBINS), "ece_q_draft": _r(ece(q, dok)),
                        "reliability_drafter_conf": reliability(pd, dok, ECE_BINS), "ece_drafter_conf": _r(ece(pd, dok)),
                        "recall_correct_drafts": _r(r0[dok == 1, RCOL["commit"]].mean()) if (dok == 1).any() else None,
                        "commit_rate_wrong_drafts": _r((r0[dok == 0, RCOL["commit"]] * r0[dok == 0, RCOL["confirm"]]).mean()) if (dok == 0).any() else None,
                        "override_fixes_wrong_drafts": _r(((r0[:, RCOL["confirm"]] == 0) & (r0[:, RCOL["ok"]] == 1))[dok == 0].mean()) if (dok == 0).any() else None}
        hold = Rd[:, RCOL["pass"]] > 0
        if hold.any():
            out["later_passes"] = {"n": int(hold.sum()),
                                   "acc_tentative_slots": _r(Rd[hold & (Rd[:, RCOL["held"]] == 1), RCOL["ok"]].mean()) if (hold & (Rd[:, RCOL["held"]] == 1)).any() else None,
                                   "acc_masked_slots": _r(Rd[hold & (Rd[:, RCOL["held"]] == 0), RCOL["ok"]].mean()) if (hold & (Rd[:, RCOL["held"]] == 0)).any() else None}
    return out


# ---------------------------------------------------------------------------------------------------------------
# IFEval per instruction type
# ---------------------------------------------------------------------------------------------------------------
def agreement_split(arm_dir: str, s: str, ids) -> dict:
    """Draft-vs-final slot agreement from the trace npz, split by block index and by content slots (up to and including
    the first <|im_end|> of the item's output) vs padding slots (after it). Also the most common padding token of the
    draft and of the final output (the 30B pads with <|endoftext|>)."""
    IM = 151645
    eq_all, eq_c, eq_p, eq_cl, nblk, nb0 = [], [], [], [], 0, 0
    pd_d, pd_f = Counter(), Counter()
    items = 0
    for i in ids:
        tf = os.path.join(arm_dir, s, "traces", s, f"{i:05d}.npz")
        if not os.path.exists(tf):
            continue
        z = np.load(tf)
        nb = z["tokens"].shape[0]
        vm = [z["valid"][b].astype(bool) for b in range(nb)]
        T = np.concatenate([z["tokens"][b][vm[b]] for b in range(nb)])
        D = np.concatenate([z["draft_tokens"][b][vm[b]] for b in range(nb)])
        blk = np.concatenate([np.full(int(vm[b].sum()), b) for b in range(nb)])
        e = np.where(T == IM)[0]
        content = np.arange(len(T)) <= (e[0] if len(e) else len(T))
        eq = D == T
        eq_all.append(eq); eq_c.append(eq[content]); eq_p.append(eq[~content]); eq_cl.append(eq[content & (blk > 0)])
        pd_d.update(D[~content].tolist()); pd_f.update(T[~content].tolist())
        nblk += nb; nb0 += 1; items += 1
    if not items:
        return {}
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, bool)
    m = lambda x: _r(x.mean()) if len(x) else None
    a, c, pp, cl = cat(eq_all), cat(eq_c), cat(eq_p), cat(eq_cl)
    return {"items": items, "blocks_per_item": _r(nblk / items), "block0_share_of_blocks": _r(nb0 / nblk),
            "agree_all": m(a), "agree_content": m(c), "agree_padding": m(pp), "padding_share": _r(len(pp) / len(a)),
            "agree_content_later_blocks": m(cl),
            "pad_token_draft": tokenizer().decode([pd_d.most_common(1)[0][0]]) if pd_d else None,
            "pad_token_final": tokenizer().decode([pd_f.most_common(1)[0][0]]) if pd_f else None}


def ifeval_types(arms: dict) -> dict:
    """arms: {name: recs}. -> per instruction group: n constraints and pass rate per arm (strict, fail-closed)."""
    from draftalign import if_scorer as IF
    tokenizer()
    IF._TOK = _TOK
    res = {}
    for name, R in arms.items():
        per = {}
        for i, r in R.items():
            ids, kw = IF.parse_gold(r["gold"])
            if not IF.finished_at_eos(r):
                ok = [False] * len(ids)
            else:
                resp = IF.strip_reasoning(IF.response_text(r))
                ok = IF.check_constraints(resp, ids, kw) if resp and resp.strip() else [False] * len(ids)
            per[i] = list(zip(ids, ok))
        res[name] = per
    return res


def ifeval_table(per: dict, ref: str, new: str) -> list:
    g = defaultdict(lambda: defaultdict(list))
    from scipy.stats import binomtest
    pw = defaultdict(lambda: [0, 0])          # group -> [new-only passes, ref-only passes] over constraints
    for i, lst in per[new].items():
        if i in per[ref] and len(per[ref][i]) == len(lst):
            for (iid, okn), (_, okr) in zip(lst, per[ref][i]):
                pw[iid.split(":")[0]][0] += int(okn and not okr)
                pw[iid.split(":")[0]][1] += int(okr and not okn)
    for name in (ref, new, "stock"):
        for i, lst in per[name].items():
            if i not in per[new] or i not in per[ref]:
                continue
            for iid, ok in lst:
                g[iid.split(":")[0]][name].append(int(ok))
    rows = []
    for grp, d in g.items():
        n = len(d[ref])
        rows.append({"group": grp, "n": n, "stock": _r(np.mean(d["stock"])) if d["stock"] else None,
                     "A": _r(np.mean(d[ref])), "new": _r(np.mean(d[new])),
                     "delta": _r(np.mean(d[new]) - np.mean(d[ref])), "new_only": pw[grp][0], "A_only": pw[grp][1],
                     "p": float(binomtest(min(pw[grp]), sum(pw[grp]), 0.5).pvalue) if sum(pw[grp]) else 1.0})
    return sorted(rows, key=lambda x: x["delta"] * x["n"])


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    A, B = {}, {}
    for s, n, _, _ in SETS:
        A[s], B[s] = load_records(REF_A, s), load_records(REF_B, s)
    status, new = {}, {}
    for arm, key, label in NEW_ARMS:
        new[arm], status[arm] = {}, {}
        for s, n, _, _ in SETS:
            R = load_records(os.path.join(DEC, arm), s) if os.path.isdir(os.path.join(DEC, arm, s)) else {}
            new[arm][s] = R
            status[arm][s] = {"n": len(R), "size": n, "complete": len(R) >= n,
                              "scored": sum(r.get("correct") is not None for r in R.values())}
    res = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status,
           "definitions": __doc__.split("Definitions")[1].strip(), "sets": {}, "pooled": {}, "partial": {}}
    feats = {"A": {}, "stock": {}}
    for s, n, _, _ in SETS:
        res["sets"][s] = {"reference": {"A": None, "stock": None}, "arms": {}}
    complete_any = sorted({s for arm, _, _ in NEW_ARMS for s, st in status[arm].items() if st["complete"]},
                          key=[x[0] for x in SETS].index)
    # features of the references only for sets some arm finished (tokenizer work)
    for s in complete_any:
        feats["A"][s] = {i: item_features(r, s) for i, r in A[s].items()}
        feats["stock"][s] = {i: item_features(r, s) for i, r in B[s].items()}
        res["sets"][s]["reference"] = {"A": agg(A[s], EQ["S0"], feats["A"][s]), "stock": agg(B[s], 0.0, feats["stock"][s]),
                                       "A_vs_stock": paired(A[s], B[s]), "A_commit_probe": commit_probe_agg(A[s])}
        TA = {i: traj_item(A[s][i], B[s][i]) for i in A[s] if i in B[s]}
        res["sets"][s]["reference"]["A_trajectory"] = traj_agg(A[s], B[s], TA)
        res["sets"][s]["reference"]["_TA"] = TA
    plots = {"forest": [], "rel": {}, "pva": {}, "speed": {}, "surv": {}}
    pool_bins = {}
    for arm, key, label in NEW_ARMS:
        done = [s for s, _, _, _ in SETS if status[arm][s]["complete"]]
        for s, _, _, _ in SETS:                          # progress of a set still running
            R = new[arm][s]
            if R and not status[arm][s]["complete"]:
                res["partial"].setdefault(arm, {})[s] = {"n": len(R), "size": SIZE[s], "vs_A": paired(R, A[s]),
                                                         "vs_stock": paired(R, B[s])}
        if not done:
            continue
        reads_all, commits_all = [], []
        pooled_T, pooled_TA = {}, {}
        pvaN = defaultdict(lambda: [0, 0])
        drafts_all = []
        for s in done:
            R = new[arm][s]
            eq = float(next((r["drafter_eq_per_forward"] for r in R.values() if r.get("drafter_eq_per_forward")), EQ[key]))
            fe = {i: item_features(r, s) for i, r in R.items()}
            ar = {"label": label, "drafter": key, "agg": agg(R, eq, fe), "vs_A": paired(R, A[s]), "vs_stock": paired(R, B[s])}
            stock_e = res["sets"][s]["reference"]["stock"]["exp_tok"]
            for nm, g in (("arm", ar["agg"]), ("A", res["sets"][s]["reference"]["A"])):
                g["speedup_vs_stock_charged"] = stock_e / g["charged_exp_tok"]
                g["speedup_vs_stock_uncharged"] = stock_e / g["exp_tok"]
            T = {i: traj_item(R[i], B[s][i]) for i in R if i in B[s]}
            ar["trajectory"] = traj_agg(R, B[s], T)
            pooled_T.update({(s, i): (R[i], t) for i, t in T.items()})
            pooled_TA.update({(s, i): (A[s][i], t) for i, t in res["sets"][s]["reference"]["_TA"].items()})
            # judge npz
            args = [(os.path.join(DEC, arm), s, R[i], T.get(i, {"blocks": []}), B[s][i]["gen_ids"] if i in B[s] else [])
                    for i in sorted(R)]
            with Pool(min(32, os.cpu_count() or 4)) as pool:
                J = [x for x in pool.map(judge_item, args, chunksize=8) if x is not None]
            Rd = np.concatenate([x["reads"] for x in J]) if J else np.zeros((0, 11), np.float32)
            Cm = np.concatenate([x["commits"] for x in J]) if J else np.zeros((0, 8), np.float32)
            reads_all.append(Rd); commits_all.append(Cm)
            ar["judge"] = judge_summary(Rd, Cm)
            ar["judge"]["items_with_npz"] = len(J)
            ar["judge"]["gen_ids_mismatch_blocks"] = int(sum(x["gen_mismatch"] for x in J))
            ar["drafts"] = {"think_preamble_in_block0": _r(np.mean([x["think0"] for x in J])) if J else None,
                            "agreement_slots": _r(sum(x["ag_plain"] for x in J) / max(1, sum(x["ag_n"] for x in J))),
                            "agreement_shift_tolerant_8": _r(sum(x["ag_shift"] for x in J) / max(1, sum(x["ag_n"] for x in J)))}
            drafts_all.append([sum(x["think0"] for x in J), len(J), sum(x["ag_plain"] for x in J), sum(x["ag_shift"] for x in J),
                               sum(x["ag_n"] for x in J)])
            ar["agree_split"] = agreement_split(os.path.join(DEC, arm), s, sorted(R))
            # discordant items vs A
            disc = {"losses": [], "wins": []}
            for i in sorted(R):
                if i not in A[s] or R[i].get("correct") is None or A[s][i].get("correct") is None:
                    continue
                c, a = int(R[i]["correct"]), int(A[s][i]["correct"])
                if c != a:
                    disc["wins" if c > a else "losses"].append(i)
            ar["discordant"] = {}
            for kind, ids in disc.items():
                if not ids:
                    ar["discordant"][kind] = {"n": 0, "idx": []}
                    continue
                sub = {i: R[i] for i in ids}
                g = agg(sub, eq, fe)
                ga = agg({i: A[s][i] for i in ids}, EQ["S0"], feats["A"][s])
                ar["discordant"][kind] = {"n": len(ids), "idx": ids,
                                          "new": {k: _r(g.get(k)) for k in ("agreement", "passes_per_block", "forced_per_block",
                                                                            "no_eos", "repetitive", "stutter_per_1k",
                                                                            "syntax_error", "no_boxed", "no_letter", "tokens_per_item")},
                                          "A": {k: _r(ga.get(k)) for k in ("agreement", "passes_per_block", "forced_per_block",
                                                                           "no_eos", "repetitive", "stutter_per_1k",
                                                                           "syntax_error", "no_boxed", "no_letter", "tokens_per_item")},
                                          "new_identical_to_stock": _r(np.mean([T[i]["identical"] for i in ids if i in T])),
                                          "A_identical_to_stock": _r(np.mean([res["sets"][s]["reference"]["_TA"][i]["identical"]
                                                                              for i in ids if i in res["sets"][s]["reference"]["_TA"]])),
                                          "stock_correct": _r(np.mean([B[s][i]["correct"] for i in ids if i in B[s] and B[s][i].get("correct") is not None]))}
            if s == "ifeval541":
                per = ifeval_types({"A": A[s], "new": R, "stock": B[s]})
                ar["ifeval_types"] = ifeval_table(per, "A", "new")
                chk = [int(all(ok for _, ok in per["A"][i])) == int(A[s][i]["correct"]) for i in A[s] if A[s][i].get("correct") is not None]
                ar["ifeval_rescore_agrees_with_records_A"] = _r(np.mean(chk))
            # passes vs agreement
            pva = defaultdict(lambda: [0, 0])
            for r in R.values():
                for a, ps in zip(r["draft_agreement_per_block"], r["full_passes_per_block"]):
                    if a is not None:
                        k = min(int(a * 10), 9); pva[k][0] += 1; pva[k][1] += len(ps)
                        pvaN[k][0] += 1; pvaN[k][1] += len(ps)
            ar["passes_by_agreement"] = {f"{k/10:.1f}": [v[0], _r(v[1] / v[0], 3)] for k, v in sorted(pva.items())}
            res["sets"][s]["arms"][arm] = ar
            plots["forest"].append((arm, s, ar["vs_A"], res["sets"][s]["reference"]["A_vs_stock"]))
            plots["speed"].setdefault(s, {})[arm] = ar["agg"]["charged_exp_tok"]
        # A passes vs agreement on the same sets
        pvaA = defaultdict(lambda: [0, 0])
        for s in done:
            for r in A[s].values():
                for a, ps in zip(r["draft_agreement_per_block"], r["full_passes_per_block"]):
                    if a is not None:
                        k = min(int(a * 10), 9); pvaA[k][0] += 1; pvaA[k][1] += len(ps)
        # pooled
        Rd, Cm = np.concatenate(reads_all), np.concatenate(commits_all)
        allR = {(s, i): r for s in done for i, r in new[arm][s].items()}
        allA = {(s, i): r for s in done for i, r in A[s].items()}
        allB = {(s, i): r for s in done for i, r in B[s].items()}
        Tn = {k: v[1] for k, v in pooled_T.items()}
        TAp = {k: v[1] for k, v in pooled_TA.items()}
        po = {"sets": done,
              "vs_A": paired_pooled([(new[arm][s], A[s]) for s in done]),
              "vs_stock": paired_pooled([(new[arm][s], B[s]) for s in done]),
              "A_vs_stock": paired_pooled([(A[s], B[s]) for s in done]),
              "agg": {"new": agg(allR, EQ[key]), "A": agg(allA, EQ["S0"]), "stock": agg(allB, 0.0)},
              "trajectory": {"new": traj_agg(allR, allB, Tn), "A": traj_agg(allA, allB, TAp)},
              "judge": judge_summary(Rd, Cm),
              "passes_by_agreement": {"new": {f"{k/10:.1f}": [v[0], _r(v[1] / v[0], 3)] for k, v in sorted(pvaN.items())},
                                      "A": {f"{k/10:.1f}": [v[0], _r(v[1] / v[0], 3)] for k, v in sorted(pvaA.items())}}}
        stock_e = po["agg"]["stock"]["exp_tok"]
        for g in po["agg"].values():
            if g is not po["agg"]["stock"]:
                g["speedup_vs_stock_charged"] = stock_e / g["charged_exp_tok"]
        deg = {}
        for nm in ("new", "A", "stock"):
            tot = defaultdict(float); cnt = defaultdict(int)
            for s in done:
                g = res["sets"][s]["arms"][arm]["agg"] if nm == "new" else res["sets"][s]["reference"][nm]
                nn = g["n"]
                for k in ("no_eos", "truncated", "stutter_any", "repetitive", "syntax_error", "no_boxed", "no_letter"):
                    if g.get(k) is not None:
                        tot[k] += g[k] * nn; cnt[k] += nn
            deg[nm] = {k: _r(tot[k] / cnt[k]) for k in tot}
        po["degenerate"] = deg
        da = np.sum(drafts_all, 0)
        po["drafts"] = {"think_preamble_in_block0": _r(da[0] / max(1, da[1])), "agreement_slots": _r(da[2] / max(1, da[4])),
                        "agreement_shift_tolerant_8": _r(da[3] / max(1, da[4]))}
        gN, gA = po["agg"]["new"], po["agg"]["A"]
        po["block0_excess_pass_share"] = _r((gN["passes_block0"] - gA["passes_block0"]) * gN["n_block0"] / gN["judge_passes_total"])
        po["discordant"] = {k: sum(res["sets"][s]["arms"][arm]["discordant"][k]["n"] for s in done) for k in ("losses", "wins")}
        res["pooled"][arm] = po
        plots["rel"][arm] = po["judge"]
        plots["pva"][arm] = po["passes_by_agreement"]
        plots["surv"][arm] = (survival(Tn), survival(TAp))
        plots["forest"].append((arm, "pooled", po["vs_A"], po["A_vs_stock"]))
    for s in res["sets"]:
        res["sets"][s]["reference"].pop("_TA", None)
    res["answers"] = answers(res)
    res["seconds"] = round(time.time() - t0, 1)
    make_plots(res, plots)
    import content_analysis as CA
    extra_lines, verdict_lines, res["content_deep_dive"] = CA.run(sys.modules[__name__], res)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_report(res, extra_lines, verdict_lines)
    print(f"[analyze_nopool] done in {time.time() - t0:.1f}s; complete: "
          + "; ".join(f"{a}: {[s for s, st in status[a].items() if st['complete']]}" for a, _, _ in NEW_ARMS))


# ---------------------------------------------------------------------------------------------------------------
# verdicts (templated from the numbers)
# ---------------------------------------------------------------------------------------------------------------
def dip_word(d):
    if not d or not d.get("n"):
        return "n/a"
    if d["hi"] < 0:
        return "significant dip"
    if d["delta"] < 0 and d.get("mcnemar_p", 1) < 0.05:
        return "dip (p < 0.05)"
    if d["delta"] < 0:
        return "non-significant dip"
    if d["lo"] > 0:
        return "significant gain"
    return "no dip"


def answers(res):
    out = {}
    for arm, po in res["pooled"].items():
        per = {s: dip_word(res["sets"][s]["arms"][arm]["vs_A"]) for s in po["sets"]}
        n, a, j, t, dg = po["agg"]["new"], po["agg"]["A"], po["judge"], po["trajectory"], po["degenerate"]
        out[arm] = {"q1_per_set": per, "q1_pooled": dip_word(po["vs_A"]),
                    "indicators": {
                        "agreement_new_vs_A": [_r(n.get("agreement")), _r(a.get("agreement"))],
                        "passes_per_block_new_vs_A": [_r(n["passes_per_block"]), _r(a["passes_per_block"])],
                        "forced_per_block_new_vs_A": [_r(n.get("forced_per_block")), _r(a.get("forced_per_block"))],
                        "stock_divergence_hazard_new_vs_A": [_r(t["new"]["divergence_hazard"]), _r(t["A"]["divergence_hazard"])],
                        "slot_match_to_stock_new_vs_A": [_r(t["new"]["slot_match"]), _r(t["A"]["slot_match"])],
                        "commit_precision_clean_new": j.get("commit_precision_clean_state", {}).get("all", {}).get("precision"),
                        "error_entry_rate_new": j.get("error_entry", {}).get("rate_per_clean_commit"),
                        "error_entry_by_type_new": j.get("error_entry", {}).get("by_type"),
                        "wrong_draft_confirmed_first_read_new": j.get("pass0", {}).get("commit_rate_wrong_drafts"),
                        "correct_draft_recall_first_read_new": j.get("pass0", {}).get("recall_correct_drafts"),
                        "agreement_block0_new_vs_A": [_r(n.get("agreement_block0")), _r(a.get("agreement_block0"))],
                        "agreement_later_new_vs_A": [_r(n.get("agreement_later")), _r(a.get("agreement_later"))],
                        "think_preamble_block0_new": po.get("drafts", {}).get("think_preamble_in_block0"),
                        "block0_excess_pass_share": po.get("block0_excess_pass_share"),
                        "no_eos_new_A_stock": [dg["new"].get("no_eos"), dg["A"].get("no_eos"), dg["stock"].get("no_eos")],
                        "charged_speedup_new_vs_A": [_r(n.get("speedup_vs_stock_charged")), _r(a.get("speedup_vs_stock_charged"))]}}
    return out


# ---------------------------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------------------------
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def make_plots(res, plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK})
    arms = [a for a, _, _ in NEW_ARMS if a in res["pooled"]]
    if not arms:
        return
    lab = {a: l for a, _, l in NEW_ARMS}
    # 1. forest plot of paired deltas
    rows = [(a, s, d, ds) for a, s, d, ds in plots["forest"]]
    labels = []
    for a, s, d, ds in rows:
        if s not in labels:
            labels.append(s)
    labels = [x for x in labels if x != "pooled"] + ["pooled"]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(labels) + 1.4))
    _style(ax)
    off = {a: (k - (len(arms) - 1) / 2) * 0.22 for k, a in enumerate(arms)}
    for a, s, d, ds in rows:
        y = len(labels) - 1 - labels.index(s)
        ax.plot([100 * d["lo"], 100 * d["hi"]], [y + off[a]] * 2, color=COLORS[a], lw=2, solid_capstyle="round")
        ax.plot(100 * d["delta"], y + off[a], "o", ms=7, color=COLORS[a], mec="#fcfcfb", mew=2,
                label=f"{lab[a]} − A" if s == labels[0] else None)
        if a == arms[0]:
            ax.plot(-100 * ds["delta"], y - 0.33, "D", ms=5, color=COLORS["stock"], mec="#fcfcfb", mew=1.5,
                    label="stock − A (context)" if s == labels[0] else None)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1], fontsize=9, color=INK)
    ax.set_xlabel("paired accuracy difference vs Reference A (points, 95% bootstrap CI)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy_delta.png"), dpi=150); plt.close(fig)

    # 2. calibration on the new drafts (pooled): judge top-1 at pass 0, judge q of the draft token, drafter confidence
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5))
    titles = ["judge top-1 p → argmax = stock token\n(clean-state reads, on-trajectory slots)",
              "judge p(draft token) → draft = stock token\n(first read)", "drafter confidence → draft = stock token"]
    keys = ["reliability_p", "reliability_q_draft", "reliability_drafter_conf"]
    for ax, ttl, k in zip(axs, titles, keys):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        for a in arms:
            rel = (plots["rel"][a].get(k) if k == "reliability_p" else plots["rel"][a].get("pass0", {}).get(k)) or []
            xs = [r["conf"] for r in rel if r["n"] >= 30]; ys = [r["acc"] for r in rel if r["n"] >= 30]
            ax.plot(xs, ys, "-o", color=COLORS[a], lw=2, ms=5, mec="#fcfcfb", mew=1.5, label=lab[a])
        ax.set_title(ttl, fontsize=8.5); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("mean confidence in bin", fontsize=8); ax.set_ylabel("share correct", fontsize=8)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "calibration_new_drafts.png"), dpi=150); plt.close(fig)

    # 3. passes per block vs block agreement
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _style(ax)
    a0 = arms[0]
    pa = plots["pva"][a0]["A"]
    ax.plot([float(k) + 0.05 for k in pa], [v[1] for v in pa.values()], "-o", color=COLORS["A"], lw=2, ms=5,
            mec="#fcfcfb", mew=1.5, label="A: SDAR-1.7B drafter" + (f" (sets of {lab[a0]})" if len(arms) > 1 else ""))
    for a in arms:
        pn = plots["pva"][a]["new"]
        ax.plot([float(k) + 0.05 for k in pn], [v[1] for v in pn.values()], "-o", color=COLORS[a], lw=2, ms=5,
                mec="#fcfcfb", mew=1.5, label=lab[a])
    ax.set_xlabel("block agreement (share of final tokens equal to the draft)", fontsize=8.5)
    ax.set_ylabel("judge passes per block", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "passes_vs_agreement.png"), dpi=150); plt.close(fig)

    # 4. charged experts per token per set
    sets = [s for s, _, _, _ in SETS if s in plots["speed"]]
    series = [("stock", "stock (no drafter)"), ("A", "A: SDAR-1.7B")] + [(a, lab[a]) for a in arms]
    fig, ax = plt.subplots(figsize=(1.6 * len(sets) + 2.5, 3.6))
    _style(ax); ax.grid(False, axis="x")
    w = 0.8 / len(series)
    for k, (key, name) in enumerate(series):
        xs, ys = [], []
        for n_, s in enumerate(sets):
            if key == "stock":
                v = res["sets"][s]["reference"]["stock"]["exp_tok"]
            elif key == "A":
                v = res["sets"][s]["reference"]["A"]["charged_exp_tok"]
            else:
                v = plots["speed"][s].get(key)
            if v is not None:
                xs.append(n_ - 0.4 + w * (k + 0.5)); ys.append(v)
        ax.bar(xs, ys, width=w * 0.88, color=COLORS[key], label=name)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(sets, fontsize=9, color=INK)
    ax.set_ylabel("charged experts per token (lower = faster)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, ncol=len(series), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "experts_per_token.png"), dpi=150); plt.close(fig)

    # 5. survival on the stock trajectory
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _style(ax)
    sn, sa = plots["surv"][a0]
    ax.plot(range(len(sa)), sa, color=COLORS["A"], lw=2, label="A: SDAR-1.7B")
    for a in arms:
        ax.plot(range(len(plots["surv"][a][0])), plots["surv"][a][0], color=COLORS[a], lw=2, label=lab[a])
    ax.set_xlim(0, 20); ax.set_ylim(0, 1)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("generated block", fontsize=8.5); ax.set_ylabel("share of items still token-identical to stock", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stock_trajectory_survival.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def pct(x, k=1):
    return "–" if x is None else f"{100 * x:.{k}f}"


def num(x, k=2):
    return "–" if x is None else f"{x:.{k}f}"


def ci(d):
    if not d or not d.get("n"):
        return "–"
    return f"{100 * d['delta']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def write_report(res, extra=None, top=None):
    L = []
    w = L.append
    lab = {a: l for a, _, l in NEW_ARMS}
    w("# Phase 2 no-pool drafter swap: accuracy, mechanism, speed\n")
    w(f"Generated {res['generated_utc']} by `analyze_nopool.py` (CPU only, {res['seconds']} s). Each new drafter + the "
      "128M judge (35805974), no expert pool, is compared with **Reference A** (SDAR-1.7B drafter + the same judge, no "
      "pool, Phase 1 records) and, for context, **Reference B** (stock 30B decode). Paired over the same items. Δ = paired "
      "accuracy difference in points with a 95% bootstrap CI (2,000 resamples, seed 0); p value = McNemar exact test on the "
      "discordant items (new-only correct / A-only correct); it is a significance test, p < 0.05 = real difference.\n")
    if top:
        L.extend(top)
    w("## Status\n")
    w("| arm | " + " | ".join(s for s, _, _, _ in SETS) + " |")
    w("|---|" + "---:|" * len(SETS))
    for arm, _, _ in NEW_ARMS:
        st = res["status"][arm]
        w(f"| {arm} | " + " | ".join(("**done**" if st[s]["complete"] else f"{st[s]['n']}/{st[s]['size']}") if st[s]["n"] else "–"
                                    for s, _, _, _ in SETS) + " |")
    w("")
    for arm, po in res["pooled"].items():
        sets = po["sets"]
        S = res["sets"]
        COMB = "All benchmarks combined" + (f" (only {sets[0]} so far)" if len(sets) == 1 else "")
        w(f"## {arm}: {lab[arm]} drafter (sets: {', '.join(sets)})\n")
        # Q1
        w("### 1. Accuracy (paired per item)\n")
        w("| set | n | stock | A | new | Δ new − A [95% CI] | new-only / A-only | p value (significance test; < 0.05 = real difference) | Δ A − stock | Δ new − stock |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sets:
            ar = S[s]["arms"][arm]; d = ar["vs_A"]; ref = S[s]["reference"]
            w(f"| {s} | {d['n']} | {pct(ref['stock']['acc'])} | {pct(ref['A']['acc'])} | {pct(ar['agg']['acc'])} | {ci(d)} | "
              f"{d['x_only']} / {d['y_only']} | {d['mcnemar_p']:.3f} | {ci(ref['A_vs_stock'])} | {ci(ar['vs_stock'])} |")
        d = po["vs_A"]
        w(f"| **{COMB}** | {d['n']} | {pct(po['agg']['stock']['acc'])} | {pct(d['acc_y'])} | {pct(d['acc_x'])} | **{ci(d)}** | "
          f"{d['x_only']} / {d['y_only']} | {d['mcnemar_p']:.3f} | {ci(po['A_vs_stock'])} | {ci(po['vs_stock'])} |")
        w(f"\nAll benchmarks combined = items of all finished benchmarks (bootstrap within each benchmark). Macro mean of the set deltas: "
          f"{100 * d['macro_delta']:+.1f} [{100 * d['macro_lo']:+.1f}, {100 * d['macro_hi']:+.1f}]. Verdict per set: "
          + ", ".join(f"{s}: {v}" for s, v in res["answers"][arm]["q1_per_set"].items()) + f"; all benchmarks combined: **{res['answers'][arm]['q1_pooled']}**.\n")
        w("![accuracy](accuracy_delta.png)\n")
        # speed
        w("### Speed side (per set)\n")
        w("| set | arm | experts/token | charged experts/token | speedup vs stock (charged) | passes/block | agreement | drafter tokens kept/block | forced/block | blocks with a forced commit | first-pass commit share |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sets:
            ref = S[s]["reference"]; ar = S[s]["arms"][arm]["agg"]
            st = ref["stock"]
            w(f"| {s} | stock | {num(st['exp_tok'], 1)} | {num(st['exp_tok'], 1)} | 1.00 | {num(st['passes_per_block'])} | – | – | – | – | – |")
            for nm, g in (("A", ref["A"]), ("new", ar)):
                w(f"| | {nm} | {num(g['exp_tok'], 1)} | {num(g['charged_exp_tok'], 1)} | {num(g.get('speedup_vs_stock_charged'))} | "
                  f"{num(g['passes_per_block'])} | {pct(g.get('agreement'))}% | {num(g.get('kept_per_block'), 1)} | "
                  f"{num(g.get('forced_per_block'))} | {pct(g.get('blocks_with_forced'))}% | {pct(g.get('first_pass_commit_share'))}% |")
        g = po["agg"]
        w(f"| **{COMB}** | stock | {num(g['stock']['exp_tok'], 1)} | {num(g['stock']['exp_tok'], 1)} | 1.00 | {num(g['stock']['passes_per_block'])} | – | – | – | – | – |")
        for nm in ("A", "new"):
            x = g[nm]
            w(f"| | {nm} | {num(x['exp_tok'], 1)} | {num(x['charged_exp_tok'], 1)} | {num(x.get('speedup_vs_stock_charged'))} | "
              f"{num(x['passes_per_block'])} | {pct(x.get('agreement'))}% | {num(x.get('kept_per_block'), 1)} | "
              f"{num(x.get('forced_per_block'))} | {pct(x.get('blocks_with_forced'))}% | {pct(x.get('first_pass_commit_share'))}% |")
        w("\nCharged = expert loads + drafter forwards × the drafter's cost per forward (S0 and F2 364.64, S1 126.32 "
          "expert-equivalents). Speedup = stock experts/token ÷ charged experts/token. Experts/token is per generated token, "
          "so it also moves with output length.\n")
        w("![experts](experts_per_token.png)\n")
        w("**First block vs later blocks.** The first generated block follows the prompt directly; a drafter whose chat "
          "template opens the answer differently from SDAR-30B misaligns every slot there.\n")
        w("| set | arm | agreement, block 0 | agreement, later blocks | passes, block 0 | passes, later | block-0 draft has `<think>` | slot agreement / shift-tolerant (±8) |")
        w("|---|---|---:|---:|---:|---:|---:|---:|")
        for s in sets + ["pooled"]:
            for nm in ("A", "new"):
                if s == "pooled":
                    g = po["agg"][nm]; dr = po["drafts"] if nm == "new" else None
                else:
                    g = S[s]["reference"]["A"] if nm == "A" else S[s]["arms"][arm]["agg"]
                    dr = S[s]["arms"][arm].get("drafts") if nm == "new" else None
                w(f"| {(COMB if s == 'pooled' else s) if nm == 'A' else ''} | {nm} | {pct(g.get('agreement_block0'))}% | {pct(g.get('agreement_later'))}% | "
                  f"{num(g.get('passes_block0'))} | {num(g.get('passes_later'))} | "
                  + (f"{pct(dr['think_preamble_in_block0'])}% | {pct(dr['agreement_slots'])}% / {pct(dr['agreement_shift_tolerant_8'])}% |" if dr else "– | – |"))
        w(f"\nExtra block-0 passes of the new arm over A, as a share of all its judge passes (all benchmarks combined): "
          f"{pct(po.get('block0_excess_pass_share'))}%. Shift-tolerant agreement = best position-wise agreement over "
          "offsets of −8..+8 tokens (new-arm traces; A's drafts are not logged).\n")
        w("**Where the agreement goes** (new-arm traces; A's drafts are not logged, so no A column). Content slots = tokens up to "
          "and including the first end-of-turn token; padding slots = everything after it, where the 30B writes `<|endoftext|>`. "
          "Short outputs put many slots in block 0 (whose drafts open with a `<think>` header) and in padding. Agreement here is weighted by slot, so it differs from the per-block mean in the speed table.\n")
        w("| set | blocks / item | block 0 share of blocks | all slots | content slots | content slots, later blocks | padding slots: share / agreement | padding token: draft vs 30B |")
        w("|---|---:|---:|---:|---:|---:|---:|---|")
        for s in sets:
            x = S[s]["arms"][arm].get("agree_split") or {}
            if x:
                w(f"| {s} | {num(x['blocks_per_item'])} | {pct(x['block0_share_of_blocks'])}% | {pct(x['agree_all'])}% | "
                  f"{pct(x['agree_content'])}% | {pct(x['agree_content_later_blocks'])}% | {pct(x['padding_share'])}% / {pct(x['agree_padding'])}% | "
                  f"`{x['pad_token_draft']}` vs `{x['pad_token_final']}` |".replace("|>", "\\|>").replace("<|", "<\\|"))
        w("")
        # mechanism
        w("### 2. Mechanism\n")
        w("**Fidelity to the stock 30B trajectory** (records, both arms). On-trajectory block = its prefix is token-identical "
          "to the stock decode's, so the stock block is the judge's training target for that state. Hazard = share of "
          "on-trajectory blocks whose final tokens differ from stock.\n")
        w("| set | arm | on-traj blocks | slot match | divergence hazard / block | hazard, block 0 | hazard, later | hazard, forced blocks | hazard, unforced blocks | items identical to stock | acc if identical | acc if diverged |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sets + ["pooled"]:
            if s == "pooled":
                tA, tN = po["trajectory"]["A"], po["trajectory"]["new"]
            else:
                tA, tN = S[s]["reference"]["A_trajectory"], S[s]["arms"][arm]["trajectory"]
            for nm, t in (("A", tA), ("new", tN)):
                w(f"| {(COMB if s == 'pooled' else s) if nm == 'A' else ''} | {nm} | {t['on_traj_blocks']} | {pct(t['slot_match'])}% | {pct(t['divergence_hazard'])}% | "
                  f"{pct(t.get('hazard_block0'))}% | {pct(t.get('hazard_later'))}% | "
                  f"{pct(None if t['block_exact_forced'] is None else 1 - t['block_exact_forced'])}% | "
                  f"{pct(None if t['block_exact_unforced'] is None else 1 - t['block_exact_unforced'])}% | "
                  f"{pct(t['items_identical_to_stock'])}% | {pct(t['acc_if_identical'])} | {pct(t['acc_if_diverged'])} |")
        w("\n![survival](stock_trajectory_survival.png)\n")
        j = po["judge"]
        w("**Judge calibration on the new drafts** (new arm only: Reference A's Phase 1 records hold no per-read "
          "probabilities). Target = the stock token at on-trajectory slots.\n")
        p0 = j.get("pass0", {})
        w("| quantity (all benchmarks combined) | value |")
        w("|---|---:|")
        prec = j.get("commit_precision_on_traj", {})
        rows = [("draft tokens equal to the stock token (first read)", pct(p0.get("draft_acc_vs_stock")) + "%"),
                ("judge argmax equal to the stock token (first read)", pct(p0.get("judge_argmax_acc")) + "%"),
                ("commit precision, all commits (on-traj)", pct(j.get("commit_precision_all_on_traj")) + "%"),
                ("... confirms of the draft (p > 0.95): n / precision / mean p",
                 f"{prec.get('confirm', {}).get('n')} / {pct(prec.get('confirm', {}).get('precision'))}% / {num(prec.get('confirm', {}).get('mean_p'), 3)}"),
                ("... overrides of the draft (p > 0.95): n / precision / mean p",
                 f"{prec.get('override', {}).get('n')} / {pct(prec.get('override', {}).get('precision'))}% / {num(prec.get('override', {}).get('mean_p'), 3)}"),
                ("... forced commits: n / precision / mean p",
                 f"{prec.get('forced', {}).get('n')} / {pct(prec.get('forced', {}).get('precision'))}% / {num(prec.get('forced', {}).get('mean_p'), 3)}"),
                ("clean-state commits (all earlier commits in the block match stock): precision all / confirm / override / forced",
                 " / ".join(pct(j.get("commit_precision_clean_state", {}).get(k, {}).get("precision")) + "%" for k in ("all", "confirm", "override", "forced"))
                 + "  (n = " + " / ".join(str(j.get("commit_precision_clean_state", {}).get(k, {}).get("n")) for k in ("all", "confirm", "override", "forced")) + ")"),
                ("first-read commits: precision all / confirm / override / forced",
                 " / ".join(pct(j.get("commit_precision_pass0", {}).get(k, {}).get("precision")) + "%" for k in ("all", "confirm", "override", "forced"))
                 + "  (n = " + " / ".join(str(j.get("commit_precision_pass0", {}).get(k, {}).get("n")) for k in ("all", "confirm", "override", "forced")) + ")"),
                ("error entry (wrong commit while the block is still identical to stock): per clean-state commit",
                 pct(j.get("error_entry", {}).get("rate_per_clean_commit"), 2) + "%"),
                ("... error entries by type: confirm of a wrong draft / override / forced",
                 " / ".join(pct(j.get("error_entry", {}).get("by_type", {}).get(k)) + "%" for k in ("confirm", "override", "forced"))),
                ("... error entries at the first read", pct(j.get("error_entry", {}).get("at_first_read")) + "%"),
                ("confident wrong confirms of the draft, per clean-state commit",
                 pct(j.get("error_entry", {}).get("confirm_of_wrong_draft_per_clean_commit"), 2) + "%"),
                ("all wrong commits incl. within-block cascades: confirm / override / forced",
                 " / ".join(pct(j.get("wrong_commit_share_by_type", {}).get(k)) + "%" for k in ("confirm", "override", "forced"))),
                ("first read: correct drafts committed (recall)", pct(p0.get("recall_correct_drafts")) + "%"),
                ("first read: wrong drafts confirmed", pct(p0.get("commit_rate_wrong_drafts"), 2) + "%"),
                ("first read: wrong drafts where the judge's argmax is the stock token", pct(p0.get("override_fixes_wrong_drafts")) + "%"),
                ("ECE, judge top-1 (clean-state reads / first read / all on-traj reads)",
                 f"{num(j.get('ece_p'), 3)} / {num(p0.get('ece_p'), 3)} / {num(j.get('ece_p_all_reads'), 3)}"),
                ("ECE, judge p(draft) for draft = stock", num(p0.get("ece_q_draft"), 3)),
                ("ECE, drafter confidence for draft = stock", num(p0.get("ece_drafter_conf"), 3)),
                ("later reads: argmax acc at tentative (draft-held) / masked slots",
                 f"{pct(j.get('later_passes', {}).get('acc_tentative_slots'))}% / {pct(j.get('later_passes', {}).get('acc_masked_slots'))}%")]
        for k, v in rows:
            w(f"| {k} | {v} |")
        w("\nRaw commit precision against stock (first row above) mixes in within-block cascades: once one slot of a block differs "
          "from stock, the stock tokens stop being the target for the rest of that block. Clean-state numbers count only "
          "reads and commits made while every earlier commit in the block still equals stock (the judge's training "
          "distribution); they are the calibration numbers to read.\n")
        w("\nJudge top-1 reliability (clean-state reads at on-trajectory slots; precision = argmax equals the stock token):\n")
        w("| p bin | n | mean p | precision |")
        w("|---|---:|---:|---:|")
        for r in j.get("reliability_p", []):
            w(f"| {r['bin']} | {r['n']} | {num(r['conf'], 3)} | {num(r['acc'], 3)} |")
        w("\n![calibration](calibration_new_drafts.png)\n")
        w("**Commit mix, A vs new** (all blocks; A from its Phase 1 `commit_probe` summaries, new from the judge npz):\n")
        w("| set | arm | confirm share | forced share | mean p confirm | mean p override | mean p forced |")
        w("|---|---|---:|---:|---:|---:|---:|")
        for s in sets:
            cp = S[s]["reference"].get("A_commit_probe")
            jn = S[s]["arms"][arm]["judge"]
            if cp:
                w(f"| {s} | A | {pct(cp['confirm_share'])}% | {pct(cp['forced_share'])}% | {num(cp['mean_p_confirm'], 3)} | "
                  f"{num(cp['mean_p_override'], 3)} | {num(cp['mean_p_forced'], 3)} |")
            cq = jn.get("commit_probe_style", {})
            w(f"| {s if not cp else ''} | new | {pct(cq.get('confirm_share'))}% | {pct(cq.get('forced_share'))}% | "
              f"{num(cq.get('mean_p_confirm'), 3)} | {num(cq.get('mean_p_override'), 3)} | {num(cq.get('mean_p_forced'), 3)} |")
        w("\n(A's `commit_probe` convention for both arms: confirm = committed token equals the draft token, override = it "
          "does not; forced commits are counted in both and also on their own.)\n")
        w("**Passes per block against block agreement** (all benchmarks combined; same agreement → same passes means the judge handles the "
          "new drafts like S0's, and the cost gap comes from how often the drafter is right):\n")
        w("![pva](passes_vs_agreement.png)\n")
        pv = po["passes_by_agreement"]
        w("| agreement bin | A blocks | A passes | new blocks | new passes |")
        w("|---|---:|---:|---:|---:|")
        for k in sorted(set(pv["A"]) | set(pv["new"])):
            a_, n_ = pv["A"].get(k, [0, None]), pv["new"].get(k, [0, None])
            w(f"| {k}–{float(k) + 0.1:.1f} | {a_[0]} | {num(a_[1])} | {n_[0]} | {num(n_[1])} |")
        w("\n**Degenerate outputs** (share of items; stutter = an item with an immediately repeated word token; "
          "repetitive = some 12-token sequence occurs ≥ 4 times; task flag = code that does not parse / no \\boxed / no option letter):\n")
        w("| set | arm | no EOS | truncated | stutter | repetitive | task flag | tokens/item |")
        w("|---|---|---:|---:|---:|---:|---:|---:|")
        for s in sets:
            tf = TASK_FLAG[DOMAIN[s]]
            for nm, g in (("stock", S[s]["reference"]["stock"]), ("A", S[s]["reference"]["A"]), ("new", S[s]["arms"][arm]["agg"])):
                w(f"| {s if nm == 'stock' else ''} | {nm} | {pct(g.get('no_eos'))}% | {pct(g.get('truncated'))}% | {pct(g.get('stutter_any'))}% | "
                  f"{pct(g.get('repetitive'))}% | {pct(g.get(tf)) + '%' if tf else '–'} | {num(g['tokens_per_item'], 0)} |")
        w("\n**Discordant items** (new-only wrong = losses, new-only right = wins), features of the NEW arm's outputs on them "
          "(A's in brackets):\n")
        w("| set | kind | n | stock correct | identical to stock new [A] | agreement | passes/block | forced/block | no EOS | repetitive | task flag |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sets:
            tf = TASK_FLAG[DOMAIN[s]]
            for kind in ("losses", "wins"):
                dd = S[s]["arms"][arm]["discordant"][kind]
                if not dd["n"]:
                    w(f"| {s if kind == 'losses' else ''} | {kind} | 0 | | | | | | | | |")
                    continue
                nn, aa = dd["new"], dd["A"]
                w(f"| {s if kind == 'losses' else ''} | {kind} | {dd['n']} | {pct(dd['stock_correct'], 0)}% | "
                  f"{pct(dd['new_identical_to_stock'], 0)}% [{pct(dd['A_identical_to_stock'], 0)}%] | "
                  f"{pct(nn['agreement'], 0)}% [{pct(aa['agreement'], 0)}%] | {num(nn['passes_per_block'])} [{num(aa['passes_per_block'])}] | "
                  f"{num(nn['forced_per_block'])} [{num(aa['forced_per_block'])}] | {pct(nn['no_eos'], 0)}% [{pct(aa['no_eos'], 0)}%] | "
                  f"{pct(nn['repetitive'], 0)}% [{pct(aa['repetitive'], 0)}%] | "
                  + (f"{pct(nn.get(tf), 0)}% [{pct(aa.get(tf), 0)}%]" if tf else "–") + " |")
        for s in sets:
            if "ifeval_types" in S[s]["arms"][arm]:
                w(f"\n**IFEval by instruction group** (constraint-level strict pass rate; rescoring reproduces A's record "
                  f"scores on {pct(S[s]['arms'][arm]['ifeval_rescore_agrees_with_records_A'])}% of items):\n")
                w("| group | constraints | stock | A | new | Δ new − A | new-only / A-only constraints | p value (significance test; < 0.05 = real difference) |")
                w("|---|---:|---:|---:|---:|---:|---:|---:|")
                for r in S[s]["arms"][arm]["ifeval_types"]:
                    w(f"| {r['group']} | {r['n']} | {pct(r['stock'])} | {pct(r['A'])} | {pct(r['new'])} | {100 * r['delta']:+.1f} | {r['new_only']} / {r['A_only']} | {r['p']:.3f} |")
                w("\nThe p values treat each constraint as independent (constraints of one prompt are not), and nine groups are tested, so a single p < 0.05 here is only a hint.\n")
        w("")
        w("### 3–4. Answers (templated from the numbers above)\n")
        an = res["answers"][arm]["indicators"]
        w(f"- **Dip?** All benchmarks combined {ci(po['vs_A'])} points (p = {po['vs_A']['mcnemar_p']:.3f}; significance test, < 0.05 = real difference) → **{res['answers'][arm]['q1_pooled']}**; "
          f"per set: " + ", ".join(f"{s} {v}" for s, v in res["answers"][arm]["q1_per_set"].items()) + ".")
        w(f"- **Drafter strength:** agreement {pct(an['agreement_new_vs_A'][0])}% vs A {pct(an['agreement_new_vs_A'][1])}%; "
          f"passes/block {num(an['passes_per_block_new_vs_A'][0])} vs {num(an['passes_per_block_new_vs_A'][1])}; "
          f"forced/block {num(an['forced_per_block_new_vs_A'][0])} vs {num(an['forced_per_block_new_vs_A'][1])}.")
        w(f"- **Alignment / leakage:** divergence hazard from the stock trajectory {pct(an['stock_divergence_hazard_new_vs_A'][0])}% "
          f"vs A {pct(an['stock_divergence_hazard_new_vs_A'][1])}% per on-trajectory block; slot match "
          f"{pct(an['slot_match_to_stock_new_vs_A'][0])}% vs {pct(an['slot_match_to_stock_new_vs_A'][1])}%; "
          f"clean-state commit precision on the new drafts {pct(an['commit_precision_clean_new'])}%; errors enter at "
          f"{pct(an['error_entry_rate_new'], 2)}% of clean-state commits (confirm / override / forced = "
          + " / ".join(pct((an['error_entry_by_type_new'] or {}).get(k)) + "%" for k in ("confirm", "override", "forced"))
          + f"); wrong drafts confirmed at the first read {pct(an['wrong_draft_confirmed_first_read_new'], 2)}%, correct "
          f"drafts committed at the first read {pct(an['correct_draft_recall_first_read_new'])}%.")
        w(f"- **Template / position:** block-0 agreement {pct(an['agreement_block0_new_vs_A'][0])}% vs A "
          f"{pct(an['agreement_block0_new_vs_A'][1])}% (block-0 drafts opening with `<think>`: {pct(an['think_preamble_block0_new'])}%); "
          f"later blocks {pct(an['agreement_later_new_vs_A'][0])}% vs {pct(an['agreement_later_new_vs_A'][1])}%; extra block-0 "
          f"passes = {pct(an['block0_excess_pass_share'])}% of the new arm's judge passes.")
        w(f"- **Degenerate outputs:** no-EOS new / A / stock = " + " / ".join(pct(x) + "%" for x in an["no_eos_new_A_stock"]) + ".")
        w(f"- **Speed:** charged speedup vs stock {num(an['charged_speedup_new_vs_A'][0])}x vs A {num(an['charged_speedup_new_vs_A'][1])}x.")
        pv = po["passes_by_agreement"]
        same = [k for k in pv["A"] if k in pv["new"] and pv["A"][k][0] >= 50 and pv["new"][k][0] >= 50]
        cheaper = sum(pv["new"][k][1] <= pv["A"][k][1] for k in same)
        if res["answers"][arm]["q1_pooled"] in ("no dip", "non-significant dip", "significant gain"):
            w(f"- **Why no dip (Q3):** every committed token is the 30B's own argmax under the judge; wrong drafts are confirmed at "
              f"the first read only {pct(an['wrong_draft_confirmed_first_read_new'], 2)}% of the time and clean-state commits "
              f"match the stock target {pct(an['commit_precision_clean_new'])}% of the time (ECE {num(po['judge'].get('ece_p'), 3)}), "
              f"so a weaker draft moves the number of judge passes ({an['passes_per_block_new_vs_A'][0] - an['passes_per_block_new_vs_A'][1]:+.2f}/block vs A) "
              "rather than accuracy.")
        pnew, pA = an['passes_per_block_new_vs_A']
        if pnew <= pA:
            sp = (f"the new drafter agrees less with the final tokens than A but needs no more judge passes ({num(pnew)} vs {num(pA)} per block), "
                  "so drafter agreement is not what limits the passes here (not explained by this analysis; the agreement-matched bins are "
                  f"confounded by block mix, {cheaper}/{len(same)} bins with >= 50 blocks each favour the new arm)")
        else:
            sp = (f"at equal block agreement the new arm needs no more passes than A in {cheaper}/{len(same)} agreement bins "
                  "(>= 50 blocks each), so the extra passes follow from how often the drafter is right, not from the judge")
        w(f"- **Scope (Q4):** {sp}. Judge "
          f"retraining on the new drafts has little to fix on calibration; its only lever is first-read recall of correct drafts "
          f"({pct(an['correct_draft_recall_first_read_new'])}%), plus the wrong confirms/overrides behind the errors above. The block-0 template mismatch is a no-training fix worth "
          f"{pct(an['block0_excess_pass_share'])}% of passes. Assessment only; no training effect has been measured.\n")
    if extra:
        L.extend(extra)
    if res["partial"]:
        w("## Sets still running (provisional, not analysed)\n")
        w("| arm | set | done | Δ new − A so far [95% CI] | Δ new − stock so far |")
        w("|---|---|---:|---:|---:|")
        for arm, d in res["partial"].items():
            for s, x in d.items():
                w(f"| {arm} | {s} | {x['n']}/{x['size']} | {ci(x['vs_A'])} | {ci(x['vs_stock'])} |")
        w("")
    w("## Notes\n")
    w("- Reference A records come from Phase 1 (same 30B revision, same judge hash 35805974, no pool); their GPU differs "
      "(Server vs Workstation edition) only for the stock reference, so wall-clock ms/token is not compared.")
    w("- Calibration numbers use the stock decode as the target only where the prefix is identical to stock (the "
      "judge's training distribution). Off-trajectory blocks have no target and are excluded from them; the commit "
      "mix (confirm/forced shares, mean p) uses every block.")
    w("- Ignored by design: the earlier pool configuration (decode/F2, decode/S1, ref/S0_128M_by_set, analysis/logprob).")
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
