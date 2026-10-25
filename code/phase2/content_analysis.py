#!/usr/bin/env python
"""Content-only agreement, format-token counterfactual, pass decomposition and improvement estimates for the no-pool
drafter swap (CoDA-1.7B = F2_nopool, Qwen3-0.6B = S1_nopool) against Reference A (SDAR-1.7B + judge) and stock.
CPU only. Called from analyze_nopool.py (`compute`, then `render` / `write_plan`); helpers also usable alone.

Slot taxonomy. A slot is one valid position of a 32-token block; the *final* token is what the 30B + judge committed.
  header   block-0 slots covered by the drafter's `<think>` ... `</think>` + whitespace opening (new drafters only);
  tail     the first `<|im_end|>` of an item's output and everything after it (`\\n`, `<|endoftext|>` padding);
  content  everything else. A content slot is *kept* when the draft token equals the final token.
Header and tail slots are "format slots". Reference A has no logged drafts, only per-block counts (agreement, valid slots,
passes, forced, per-pass commit counts) plus its final tokens, so for A the tail is known exactly, there is no header (its
drafts open like the 30B), the content-only agreement is exact on blocks that contain no tail slot, and on the others it
uses an estimated tail agreement `ta` (A's agreement on blocks with >= 24 tail slots, a lower bound on its tail agreement).

Estimates (marked "est."): see the assumptions printed next to each table; none of them is a re-decode.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

IM, EOT, THINK, ETHINK = 151645, 151643, 151667, 151668
THETA = 0.05                  # judge first-read probability of the draft token: plausible alternative (>=) vs genuine miss (<)
CLS = {0: "content", 1: "header", 2: "tail"}


# ---------------------------------------------------------------------------------------------------------------
# slot tables (new drafters: traces + judge npz + records)
# ---------------------------------------------------------------------------------------------------------------
def ws_table(tok) -> np.ndarray:
    n = len(tok) + 1024
    ws = np.zeros(n + 10, bool)
    for i in range(n):
        try:
            d = tok.decode([i])
            ws[i] = d != "" and d.strip() == ""
        except Exception:
            pass
    return ws


def header_len(D: np.ndarray, ws: np.ndarray) -> int:
    """leading valid slots of block 0 forming `<think>` .. `</think>` + whitespace (0 if the draft does not open so)."""
    if len(D) == 0 or D[0] != THINK:
        return 0
    n = 1
    while n < len(D) and D[n - 1] != ETHINK:
        n += 1
    if n > len(D) or D[n - 1] != ETHINK:
        return 0
    while n < len(D) and ws[D[n]]:
        n += 1
    return n


def load_item(arm_dir: str, s: str, i: int, rec: dict, ws: np.ndarray):
    jf = os.path.join(arm_dir, s, "judge", s, f"{i:05d}.npz")
    tf = os.path.join(arm_dir, s, "traces", s, f"{i:05d}.npz")
    if not (os.path.exists(jf) and os.path.exists(tf)):
        return None
    j, t = np.load(jf), np.load(tf)
    jp = j["judge_pass"].astype(int)
    blk_of_pass = np.cumsum(jp == 0) - 1
    nb = t["tokens"].shape[0]
    fps = rec["full_passes_per_block"]
    vms = [t["valid"][b].astype(bool) for b in range(nb)]
    Tall = np.concatenate([t["tokens"][b][vms[b]] for b in range(nb)])
    e = np.where(Tall == IM)[0]
    first_im = int(e[0]) if len(e) else 10 ** 9
    cols = defaultdict(list)
    off = 0
    for b in range(nb):
        sl = np.where(vms[b])[0]
        n = len(sl)
        T, D, pdr = t["tokens"][b][sl], t["draft_tokens"][b][sl], t["p_draft"][b][sl]
        rows = np.where(blk_of_pass == b)[0]
        op = j["judge_open"][rows][:, sl]
        P = j["judge_p"][rows][:, sl].astype(np.float32)
        AM = j["judge_argmax"][rows][:, sl]
        Q = j["judge_q"][rows][:, sl].astype(np.float32)
        npass = len(rows)
        nxt = np.vstack([op[1:], np.zeros((1, n), bool)])
        com = op & ~nxt
        cpass = np.where(com.any(0), com.argmax(0), -1)
        ar = np.arange(n)
        ci = np.maximum(cpass, 0)
        cam = np.where(cpass >= 0, AM[ci, ar], -1)
        forced_pass = np.array([bool(x["forced"]) for x in fps[b]] + [False] * max(0, npass - len(fps[b])))[:npass]
        ctype = np.where(cpass < 0, -1, np.where(forced_pass[ci], 2, np.where(cam == D, 0, 1)))
        pos = off + np.arange(n)
        cls = np.zeros(n, np.int8)
        cls[pos >= first_im] = 2
        h = header_len(D, ws) if b == 0 else 0
        if h:
            cls[:h] = np.where(cls[:h] == 2, 2, 1)
        m = com.sum(1)
        for k, v in (("blk", np.full(n, b)), ("slot", sl), ("pos", pos), ("T", T), ("D", D), ("pdr", pdr), ("cls", cls),
                     ("p1", P[0]), ("a1", AM[0]), ("q1", Q[0]), ("cpass", cpass),
                     ("cp", np.where(cpass >= 0, P[ci, ar], np.nan)), ("ctype", ctype), ("npass", np.full(n, npass)),
                     ("m_in_pass", np.where(cpass >= 0, m[ci], 0)), ("hdr_len", np.full(n, h))):
            cols[k].append(v)
        off += n
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["item"] = np.full(len(out["T"]), i)
    return out


def load_arm(AN, arm: str, s: str, R: dict, ws: np.ndarray, B: dict) -> dict:
    """slot table of every item (concatenated). Adds `on_traj` (block prefix identical to stock's) and `stock` (the
    stock token at that position, -1 where the block is off the stock trajectory)."""
    cols = defaultdict(list)
    for i in sorted(R):
        x = load_item(os.path.join(AN.DEC, arm), s, i, R[i], ws)
        if x is None:
            continue
        on = np.zeros(len(x["T"]), bool)
        stock = np.full(len(x["T"]), -1, np.int64)
        if i in B:
            for b, off, n, m in AN.traj_item(R[i], B[i])["blocks"]:
                on[off:off + n] = True
                stock[off:off + n] = np.asarray(B[i]["gen_ids"][off:off + n])[:n]
        x["on_traj"], x["stock"] = on, stock
        for k, v in x.items():
            cols[k].append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


def block_keys(X: dict):
    key = X["item"].astype(np.int64) * 1000 + X["blk"]
    u, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(u), int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return u, inv, first


# ---------------------------------------------------------------------------------------------------------------
# block tables from records (all arms: A, new drafters, stock)
# ---------------------------------------------------------------------------------------------------------------
def block_table(R: dict, kind: str = "judge") -> dict:
    """arrays over all blocks of the records: item, b, n valid, tail slots, passes, kept draft tokens, forced passes."""
    it, bb, nn, tl, ps, kp, fo = [], [], [], [], [], [], []
    for i, r in R.items():
        G, V = r["gen_ids"], r["n_valid_per_block"]
        e = [k for k, x in enumerate(G) if x == IM]
        first_im = e[0] if e else 10 ** 9
        fps = r.get("full_passes_per_block") or []
        ag = r.get("draft_agreement_per_block") or []
        off = 0
        for b, v in enumerate(V):
            if kind == "stock":
                p = (r.get("passes_per_block") or [0] * len(V))[b]
                k_, f_ = np.nan, 0
            else:
                pr = fps[b] if b < len(fps) else []
                p = len(pr)
                a = ag[b] if b < len(ag) else None
                k_ = np.nan if a is None else float(a) * v
                f_ = sum(int(x["forced"]) for x in pr)
            it.append(i); bb.append(b); nn.append(v)
            tl.append(max(0, (off + v) - max(off, first_im)) if first_im < off + v else 0)
            ps.append(p); kp.append(k_); fo.append(f_)
            off += v
    arr = lambda x, d=float: np.asarray(x, d)
    return {"item": arr(it, int), "b": arr(bb, int), "n": arr(nn), "tail": arr(tl), "passes": arr(ps), "kept": arr(kp),
            "forced": arr(fo)}


def group_of(b: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """0 = first block, 1 = later block with tail slots, 2 = later block without any tail slot (interior)."""
    return np.where(b == 0, 0, np.where(tail > 0, 1, 2))


GROUPS = {0: "first block", 1: "last block (has padding)", 2: "interior blocks"}


# ---------------------------------------------------------------------------------------------------------------
# agreement tables
# ---------------------------------------------------------------------------------------------------------------
def a_tail_agreement(BA: dict) -> float:
    """A's draft agreement on blocks with >= 24 tail slots (a lower bound on its tail-slot agreement; ~ 0.94-0.98)."""
    m = (BA["tail"] >= 24) & ~np.isnan(BA["kept"])
    return float(BA["kept"][m].sum() / BA["n"][m].sum()) if m.sum() >= 10 else 1.0


def agree_new(X: dict) -> dict:
    cls, T, D, blk = X["cls"], X["T"], X["D"], X["blk"]
    kept = D == T
    N = len(T)
    u, inv, first = block_keys(X)
    fmt_in_block = np.bincount(inv, weights=(cls != 0).astype(float), minlength=len(u)) > 0
    pure = (~fmt_in_block)[inv] & (blk > 0)

    def r(m):
        return float(kept[m].mean()) if m.sum() else None
    out = {"slots": int(N), "header_share": float((cls == 1).mean()), "tail_share": float((cls == 2).mean()),
           "all": r(np.ones(N, bool)), "content": r(cls == 0), "content_later": r((cls == 0) & (blk > 0)),
           "b0_content": r((cls == 0) & (blk == 0)), "b0_nontail": r((blk == 0) & (cls != 2)), "tail": r(cls == 2),
           "pure": r(pure), "kept_tail": int((kept & (cls == 2)).sum()), "kept_b0nt": int((kept & (blk == 0) & (cls != 2)).sum()),
           "kept_later_nt": int((kept & (blk > 0) & (cls != 2)).sum()), "n_tail": int((cls == 2).sum()),
           "n_b0nt": int(((blk == 0) & (cls != 2)).sum()), "n_later_nt": int(((blk > 0) & (cls != 2)).sum()), "by_pos": {}}
    for nm, m in (("0", blk == 0), ("1", blk == 1), ("2", blk == 2), ("3+", blk >= 3)):
        out["by_pos"][nm] = {"n": int(((cls == 0) & m).sum()), "content": r((cls == 0) & m)}
    return out


def agree_A(BA: dict, ta: float) -> dict:
    ok = ~np.isnan(BA["kept"])
    n, tl, kp, b = BA["n"][ok], BA["tail"][ok], BA["kept"][ok], BA["b"][ok]
    N = n.sum()

    def content(m):
        den = n[m].sum() - tl[m].sum()
        return float((kp[m].sum() - ta * tl[m].sum()) / den) if den > 0 else None
    pure = (b > 0) & (tl == 0)
    allm = np.ones(len(n), bool)
    out = {"slots": int(N), "tail_share": float(tl.sum() / N), "all": float(kp.sum() / N), "content": content(allm),
           "content_later": content(b > 0), "b0_nontail": content(b == 0), "b0_content": content(b == 0), "tail": ta,
           "pure": float(kp[pure].sum() / n[pure].sum()) if pure.sum() else None, "ta": ta,
           "kept_tail": float(ta * tl.sum()), "kept_b0nt": float(kp[b == 0].sum() - ta * tl[b == 0].sum()),
           "kept_later_nt": float(kp[b > 0].sum() - ta * tl[b > 0].sum()), "n_tail": int(tl.sum()),
           "n_b0nt": int(n[b == 0].sum() - tl[b == 0].sum()), "n_later_nt": int(n[b > 0].sum() - tl[b > 0].sum()), "by_pos": {}}
    for nm, m in (("0", b == 0), ("1", b == 1), ("2", b == 2), ("3+", b >= 3)):
        out["by_pos"][nm] = {"n": int(n[m].sum() - tl[m].sum()), "content": content(m)}
    out["content_range"] = [float((kp.sum() - tl.sum()) / (N - tl.sum())), float(kp.sum() / (N - tl.sum()))]
    return out


def gap_decomposition(a: dict, nw: dict) -> dict:
    """gap in all-slot agreement (A - new, points) split into tail / first-block / later-block contributions."""
    NA, NN = a["slots"], nw["slots"]
    g = {}
    for k, ka, kn in (("tail", a["kept_tail"], nw["kept_tail"]), ("first block", a["kept_b0nt"], nw["kept_b0nt"]),
                      ("later blocks", a["kept_later_nt"], nw["kept_later_nt"])):
        g[k] = 100 * (ka / NA - kn / NN)
    g["total"] = 100 * (a["all"] - nw["all"])
    return g


# ---------------------------------------------------------------------------------------------------------------
# pass accounting
# ---------------------------------------------------------------------------------------------------------------
def group_table(BT: dict) -> dict:
    g = group_of(BT["b"], BT["tail"])
    out = {}
    for k in (0, 1, 2):
        m = g == k
        has = m.any() and not np.isnan(BT["kept"][m]).all()
        out[k] = {"share": float(m.mean()), "passes": float(BT["passes"][m].mean()) if m.any() else None, "n": int(m.sum()),
                  "agree": float(np.nansum(BT["kept"][m]) / BT["n"][m].sum()) if has else None}
    out["all"] = {"passes": float(BT["passes"].mean()), "n": int(len(g))}
    return out


def decompose_delta(gn: dict, ga: dict) -> dict:
    """exact: sum_g (w_n p_n - w_A p_A) = sum_g wbar (p_n - p_A) + sum_g pbar (w_n - w_A)."""
    within, mix = {}, {}
    for k in (0, 1, 2):
        wn, wa = gn[k]["share"], ga[k]["share"]
        pn, pa = gn[k]["passes"] or 0.0, ga[k]["passes"] or 0.0
        within[k] = 0.5 * (wn + wa) * (pn - pa)
        mix[k] = 0.5 * (pn + pa) * (wn - wa)
    return {"within": within, "mix": mix, "total": float(gn["all"]["passes"] - ga["all"]["passes"]),
            "mix_total": float(sum(mix.values()))}


def class_attribution(X: dict, theta: float = THETA) -> dict:
    """passes per block attributed to slot classes: every judge pass is split equally among the slots it commits (a forced
    pass commits one slot and is attributed to it entirely)."""
    u, inv, first = block_keys(X)
    nb = len(u)
    cls, D, T, q1, m = X["cls"], X["D"], X["T"], X["q1"], X["m_in_pass"]
    kept = (D == T) & (cls == 0)
    mis = (D != T) & (cls == 0)
    w = np.where(m > 0, 1.0 / np.maximum(m, 1), 0.0)
    classes = {"header": cls == 1, "padding (tail)": cls == 2, "content, kept": kept,
               "content, plausible alternative": mis & (q1 >= theta), "content, genuine miss": mis & (q1 < theta)}
    out = {k: float(w[v].sum() / nb) for k, v in classes.items()}
    out["total"] = float(sum(out.values()))
    out["passes_per_block"] = float(X["npass"][first].mean())
    out["slots_per_block"] = {k: float(v.sum() / nb) for k, v in classes.items()}
    ctype = X["ctype"]
    out["forced_pass_share_by_class"] = {k: float(((ctype == 2) & v).sum() / max(1, (ctype == 2).sum())) for k, v in classes.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------
# counterfactual building blocks
# ---------------------------------------------------------------------------------------------------------------
def b0_stripped_agreement(X: dict) -> dict:
    """block-0 agreement if the header were removed from the draft (draft shifted left by the header length, overlapping
    slots only), versus as decoded; over all block-0 slots."""
    items = np.unique(X["item"])
    k0 = n0 = ks = ns = 0
    for i in items:
        m = (X["blk"] == 0) & (X["item"] == i)
        T, D, h = X["T"][m], X["D"][m], int(X["hdr_len"][m][0])
        n = len(T)
        k0 += int((T == D).sum()); n0 += n
        if h and n > h:
            ks += int((D[h:] == T[:n - h]).sum()); ns += n - h
        else:
            ks += int((D == T).sum()); ns += n
    return {"as_decoded": k0 / max(1, n0), "stripped": ks / max(1, ns)}


def loads_per_pass(R: dict) -> float:
    X = np.array([[r["n_judge"], r["n_flush"] + r["n_prefill"]] for r in R.values()], float)
    y = np.array([r["unique_expert_layer_loads"] for r in R.values()], float)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def block_feature_table(X: dict, theta: float = THETA) -> dict:
    u, inv, first = block_keys(X)
    nb = len(u)
    cls, D, T, q1 = X["cls"], X["D"], X["T"], X["q1"]
    kept = D == T
    cont = cls == 0
    f = lambda m: np.bincount(inv, weights=m.astype(float), minlength=nb)
    return {"item": X["item"][first], "blk": X["blk"][first], "passes": X["npass"][first].astype(float),
            "alt": f(cont & ~kept & (q1 >= theta)), "gen": f(cont & ~kept & (q1 < theta)), "kept_c": f(cont & kept),
            "tail": f(cls == 2), "tail_kept": f((cls == 2) & kept), "nv": f(np.ones(len(cls)))}


def pad_coefficient(FT: dict, reps: int = 300, seed: int = 0) -> dict:
    """extra passes per MISMATCHED padding slot relative to a matched one, from blocks of two drafters on the same items:
    passes ~ const + alt + gen + kept_c + tail_mismatch + tail_kept (OLS), item-cluster bootstrap 95% CI."""
    Xd = np.column_stack([np.ones(len(FT["passes"])), FT["alt"], FT["gen"], FT["kept_c"], FT["tail"] - FT["tail_kept"], FT["tail_kept"]])
    y, it = FT["passes"], FT["item"]
    coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    ids = np.unique(it)
    idx_of = {i: np.where(it == i)[0] for i in ids}
    ds = []
    for _ in range(reps):
        ix = np.concatenate([idx_of[i] for i in rng.choice(ids, len(ids))])
        c = np.linalg.lstsq(Xd[ix], y[ix], rcond=None)[0]
        ds.append(c[4] - c[5])
    return {"per_slot": float(coef[4] - coef[5]), "lo": float(np.quantile(ds, .025)), "hi": float(np.quantile(ds, .975)),
            "coef": dict(zip(["const", "alt", "gen", "kept_c", "tail_mismatch", "tail_kept"], coef.tolist()))}


def oracle_ladder(X: dict, taus=(0.8, 0.5, 0.0)) -> dict:
    """passes per block if slots whose first-read argmax already equals the final token were committed at pass 0 whenever
    the judge's first-read p >= tau' (tau' = 0: perfect confidence on right answers). Slots whose first-read argmax is not
    the final token keep their logged commit pass. A static, optimistic bound (no compounding of earlier commits)."""
    u, inv, first = block_keys(X)
    nb = len(u)
    cp = X["cpass"]
    out = {"observed": float(X["npass"][first].mean())}
    for tp in taus:
        at0 = (cp == 0) | ((X["a1"] == X["T"]) & (X["p1"] >= tp))
        mm = np.zeros(nb)
        np.maximum.at(mm, inv[~at0], cp[~at0])
        late = np.bincount(inv, weights=(~at0).astype(float), minlength=nb) > 0
        out[f"tau{tp:g}"] = float(np.where(late, mm + 1, 1).mean())
    return out


def tail_fill_saving(X: dict) -> dict:
    """passes saved if, once `<|im_end|>` is committed, the rest of the block (`\\n` + `<|endoftext|>` padding) were filled
    by rule instead of by judge passes (the block then ends after its last content / `<|im_end|>` commit)."""
    u, inv, first = block_keys(X)
    nb = len(u)
    cls, cp = X["cls"], X["cpass"]
    isim = (X["T"] == IM) & (cls == 2)
    m1 = np.full(nb, -1)
    keep = (cls != 2) | isim
    np.maximum.at(m1, inv[keep], cp[keep])
    npass = X["npass"][first].astype(float)
    hastail = np.bincount(inv, weights=(cls == 2).astype(float), minlength=nb) > 0
    pr = np.where(hastail, m1 + 1, npass)
    return {"per_block": float((npass - pr).mean()), "share": float((npass - pr).sum() / npass.sum())}


# ---------------------------------------------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------------------------------------------
def compute(AN) -> dict:
    tok = AN.tokenizer()
    ws = ws_table(tok)
    res = {"theta": THETA, "arms": {}, "A_tail_agreement": {}, "sets": []}
    A, B = {}, {}
    new = {a: {} for a, _, _ in AN.NEW_ARMS}
    for arm in new:
        for s, n, _, _ in AN.SETS:
            if os.path.isdir(os.path.join(AN.DEC, arm, s)):
                R = AN.load_records(os.path.join(AN.DEC, arm), s)
                if len(R) >= n:
                    new[arm][s] = R
    sets = [s for s, _, _, _ in AN.SETS if any(s in new[a] for a in new)]
    for s in sets:
        A[s], B[s] = AN.load_records(AN.REF_A, s), AN.load_records(AN.REF_B, s)
    res["sets"] = sets
    SLOT = {}
    for arm, key, label in AN.NEW_ARMS:
        for s in sets:
            if s in new[arm]:
                SLOT[(arm, s)] = load_arm(AN, arm, s, new[arm][s], ws, B[s])
    for arm, key, label in AN.NEW_ARMS:
        res["arms"][arm] = {"label": label, "key": key, "sets": {}}
        for s in sets:
            if (arm, s) not in SLOT:
                continue
            X, R = SLOT[(arm, s)], new[arm][s]
            RA = {i: A[s][i] for i in R if i in A[s]}
            BA, BN = block_table(RA), block_table(R)
            BS = block_table({i: B[s][i] for i in R if i in B[s]}, "stock")
            ta = a_tail_agreement(BA)
            res["A_tail_agreement"][s] = ta
            ag_new, ag_A = agree_new(X), agree_A(BA, ta)
            gA, gN, gS = group_table(BA), group_table(BN), group_table(BS)
            dec = decompose_delta(gN, gA)
            attr = class_attribution(X)
            r = {"n_items": len(R), "blocks": int(len(BN["b"])), "agree_new": ag_new, "agree_A": ag_A,
                 "gap": gap_decomposition(ag_A, ag_new), "groups": {"A": gA, "new": gN, "stock": gS}, "delta": dec,
                 "attribution": attr, "b0": b0_stripped_agreement(X), "oracle": oracle_ladder(X), "tail_fill": tail_fill_saving(X),
                 "loads_per_pass": loads_per_pass(R), "ta": ta, "slope_w": interior_slope(X)}
            agr_n, agr_a = BN["kept"] / BN["n"], BA["kept"] / BA["n"]
            hi_ag = np.concatenate([BN["passes"][agr_n >= 0.9], BA["passes"][agr_a >= 0.9]])
            floor = float(hi_ag.mean()) if len(hi_ag) >= 20 else float("nan")
            r["ceiling"] = {"floor": floor, "n_floor_blocks": int(len(hi_ag)),
                            "save": float(np.maximum(0, BN["passes"] - floor).mean()) if np.isfinite(floor) else float("nan")}
            w = dec["within"]
            aa, gg = attr["content, plausible alternative"], attr["content, genuine miss"]
            miss = aa + gg
            r["abc"] = {"a_first_block": w[0], "a_padding": w[1], "content_interior": w[2], "mix": dec["mix_total"],
                        "b_plausible_alternative": w[2] * (aa / miss if miss else 0.0),
                        "c_genuine_miss": w[2] * (gg / miss if miss else 0.0), "total": dec["total"]}
            gen = sum(int(x["n_gen_to_eos"]) for x in R.values())
            uniq = sum(int(x["unique_expert_layer_loads"]) for x in R.values())
            n_dr = sum(sum((x.get("drafter_n_by_kind") or {}).values()) for x in R.values())
            eq = AN.EQ[key]
            gst = sum(int(x["n_gen_to_eos"]) for x in B[s].values() if x["idx"] in R)
            ust = sum(int(x["unique_expert_layer_loads"]) for x in B[s].values() if x["idx"] in R)
            r["speed"] = {"gen": gen, "uniq": uniq, "drafter_charge": n_dr * eq, "stock_exp_tok": ust / gst,
                          "charged_exp_tok": (uniq + n_dr * eq) / gen, "blocks": r["blocks"],
                          "passes_per_block": float(BN["passes"].mean()), "drafter_share": n_dr * eq / (uniq + n_dr * eq),
                          "A_passes_per_block": float(BA["passes"].mean())}
            res["arms"][arm]["sets"][s] = r
    for s in sets:
        have = [a for a in res["arms"] if s in res["arms"][a]["sets"]]
        pad = None
        if len(have) == 2:
            parts = [block_feature_table(SLOT[(a, s)]) for a in have]
            FT = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            pad = pad_coefficient(FT)
        for a in have:
            X = SLOT[(a, s)]
            kept = X["D"] == X["T"]
            cls = X["cls"]
            res["arms"][a]["sets"][s]["pad"] = {"obs_tail_agreement": float(kept[cls == 2].mean()) if (cls == 2).any() else None,
                                                "coef": pad}
    res["dist"], res["examples"] = distributions(SLOT, tok)
    res["_slot"] = SLOT
    return res


def distributions(SLOT: dict, tok) -> tuple:
    dist, ex = {}, {}
    rng = np.random.default_rng(0)
    for arm in sorted({a for a, _ in SLOT}):
        agg = defaultdict(list)
        for (a, s), X in SLOT.items():
            if a != arm:
                continue
            cls, D, T = X["cls"], X["D"], X["T"]
            kept = D == T
            mis = (cls == 0) & ~kept
            groups = {"content, kept": (cls == 0) & kept, "content, plausible alternative": mis & (X["q1"] >= THETA),
                      "content, genuine miss": mis & (X["q1"] < THETA), "header": cls == 1, "padding (tail)": cls == 2}
            on = X["on_traj"] & (X["stock"] >= 0)
            for g, m in groups.items():
                agg[g].append((X["q1"][m], X["pdr"][m], X["cpass"][m], X["ctype"][m], X["p1"][m], (X["a1"] == T)[m], on[m],
                               (D == X["stock"])[m], (T == X["stock"])[m]))
        dist[arm] = {}
        for g, L in agg.items():
            q1, pdr, cp, ct, p1, a1ok, on, dS, tS = (np.concatenate([x[k] for x in L]) for k in range(9))
            if not len(q1):
                continue
            dist[arm][g] = {"n": int(len(q1)), "q1_q": np.quantile(q1, [.1, .5, .9]).round(4).tolist(),
                            "drafter_conf_q": np.quantile(pdr, [.1, .5, .9]).round(3).tolist(),
                            "drafter_conf_gt90": float((pdr > .9).mean()), "commit_pass_mean": float(cp.mean()),
                            "commit_pass0": float((cp == 0).mean()), "forced": float((ct == 2).mean()),
                            "first_read_argmax_is_final": float(a1ok.mean()), "first_read_p_median": float(np.median(p1)),
                            "on_traj_share": float(on.mean()),
                            "on_traj_draft_eq_stock": float(dS[on].mean()) if on.any() else None,
                            "on_traj_final_eq_stock": float(tS[on].mean()) if on.any() else None}

    def pick(maskfn, k=4):
        pool = []
        for (arm, s), X in SLOT.items():
            idx = np.where(maskfn(X))[0]
            if len(idx):
                pool += [(arm, s, int(j)) for j in rng.choice(idx, min(len(idx), 40), replace=False)]
        out = []
        for ix in (rng.choice(len(pool), min(k, len(pool)), replace=False) if pool else []):
            arm, s, j = pool[int(ix)]
            X = SLOT[(arm, s)]
            lo = max(0, j - 7)
            ctx = [int(t) for t, it in zip(X["T"][lo:j], X["item"][lo:j]) if it == X["item"][j]]
            out.append({"arm": arm, "set": s, "item": int(X["item"][j]), "block": int(X["blk"][j]), "slot": int(X["slot"][j]),
                        "context": tok.decode(ctx), "draft": tok.decode([int(X["D"][j])]), "final": tok.decode([int(X["T"][j])]),
                        "q1": float(X["q1"][j]), "judge_top1": tok.decode([int(X["a1"][j])]), "p1": float(X["p1"][j]),
                        "drafter_conf": float(X["pdr"][j]), "commit_pass": int(X["cpass"][j]),
                        "commit": {0: "confirm", 1: "override", 2: "forced"}.get(int(X["ctype"][j]), "-"),
                        "stock": tok.decode([int(X["stock"][j])]) if X["stock"][j] >= 0 else None})
        return out
    ex["header"] = pick(lambda X: (X["cls"] == 1))
    ex["padding (tail)"] = pick(lambda X: (X["cls"] == 2) & (X["D"] != X["T"]))
    ex["content, kept (confirmed at the first read)"] = pick(lambda X: (X["cls"] == 0) & (X["D"] == X["T"]) & (X["cpass"] == 0))
    ex["content, plausible alternative"] = pick(lambda X: (X["cls"] == 0) & (X["D"] != X["T"]) & (X["q1"] >= THETA))
    ex["content, genuine miss"] = pick(lambda X: (X["cls"] == 0) & (X["D"] != X["T"]) & (X["q1"] < THETA))
    ex["draft equals the stock token, final differs"] = pick(
        lambda X: X["on_traj"] & (X["cls"] == 0) & (X["D"] == X["stock"]) & (X["T"] != X["stock"]))
    return dist, ex


# ---------------------------------------------------------------------------------------------------------------
# estimates (counterfactuals, bounds): all marked est., none is a re-decode
# ---------------------------------------------------------------------------------------------------------------
def interior_slope(X: dict) -> float:
    """within-drafter OLS slope: judge passes per mismatched content slot, interior blocks (later blocks without any format
    slot). Upper-end estimate of what an extra matched slot buys (hard blocks have both more misses and more passes)."""
    u, inv, first = block_keys(X)
    nb = len(u)
    cls = X["cls"]
    fmt = np.bincount(inv, weights=(cls != 0).astype(float), minlength=nb) > 0
    mis = np.bincount(inv, weights=((cls == 0) & (X["D"] != X["T"])).astype(float), minlength=nb)
    ps = X["npass"][first].astype(float)
    ok = (~fmt) & (X["blk"][first] > 0)
    if ok.sum() < 30:
        return float("nan")
    return float(np.polyfit(mis[ok], ps[ok], 1)[0])


def add_estimates(res: dict) -> None:
    """adds r['cf'] (format-fix estimates), r['bounds'] (judge / drafter / drafter-cost levers) to every (arm, set)."""
    # cross-arm slope (passes per interior block per agreement point) through the origin, all (arm, set) pairs
    num = den = 0.0
    for arm, d in res["arms"].items():
        for s, r in d["sets"].items():
            gn, ga = r["groups"]["new"][2], r["groups"]["A"][2]
            if gn["agree"] is None or ga["agree"] is None or gn["passes"] is None or ga["passes"] is None:
                continue
            dag = 100 * (ga["agree"] - gn["agree"])
            dp = gn["passes"] - ga["passes"]
            num += dp * dag
            den += dag * dag
    res["slope_cross_arm"] = num / den if den else float("nan")
    for arm, d in res["arms"].items():
        for s, r in d["sets"].items():
            sp = r["speed"]
            nb, gen, cpp = r["blocks"], sp["gen"], r["loads_per_pass"]
            P0 = sp["passes_per_block"]
            g = r["groups"]
            # --- header strip: reference-class interpolation on the first block ---------------------------------------
            obs, strip = r["b0"]["as_decoded"], r["b0"]["stripped"]
            a_b0 = g["A"][0]["agree"]
            rec = 0.0
            if a_b0 is not None and a_b0 > obs:
                rec = float(np.clip((strip - obs) / (a_b0 - obs), 0, 1))
            full = max(0.0, (g["new"][0]["passes"] or 0) - (g["A"][0]["passes"] or 0)) * g["new"][0]["share"]
            hdr = {"agree_obs": obs, "agree_stripped": strip, "agree_A": a_b0, "recovery": rec, "central": rec * full,
                   "hi": full, "lo": 0.0}
            # --- padding remap -------------------------------------------------------------------------------------------
            pad = r["pad"]
            ta_obs = pad["obs_tail_agreement"]
            tail_slots_pb = r["agree_new"]["n_tail"] / nb
            gain_slots = max(0.0, r["ta"] - (ta_obs or 0)) * tail_slots_pb
            if pad["coef"] is not None:
                reg_c = max(0.0, pad["coef"]["per_slot"]) * gain_slots
                hi = max(0.0, pad["coef"]["hi"]) * gain_slots
                ref_c = max(0.0, r["abc"]["a_padding"])          # measured last-block pass gap to A (A pads like the 30B)
                c = min(reg_c, ref_c)
                basis = "min(two-drafter regression, measured last-block pass gap to A)"
            else:
                c, hi = 0.0, max(0.0, r["abc"]["a_padding"])
                basis = "no second drafter on this set: only the reference-class bound"
            padd = {"agree_obs": ta_obs, "agree_target": r["ta"], "central": c, "hi": max(hi, c), "lo": 0.0, "basis": basis,
                    "per_slot": None if pad["coef"] is None else pad["coef"]["per_slot"]}
            tf = r["tail_fill"]["per_block"]
            fmt_c, fmt_hi = hdr["central"] + padd["central"], hdr["hi"] + padd["hi"]

            def spd(save):          # charged speedup if `save` passes/block disappear (estimate: loads per judge pass constant)
                ch = sp["charged_exp_tok"] - cpp * save * nb / gen
                return sp["stock_exp_tok"] / ch
            r["cf"] = {"P0": P0, "header": hdr, "padding": padd, "tail_fill": tf, "format_central": fmt_c, "format_hi": fmt_hi,
                       "P_format_central": P0 - fmt_c, "P_format_hi": P0 - fmt_hi, "speedup_obs": spd(0.0),
                       "speedup_format_central": spd(fmt_c), "speedup_format_hi": spd(fmt_hi),
                       "speedup_format_tail_central": spd(fmt_c + tf), "speedup_tail": spd(tf)}
            # --- levers -------------------------------------------------------------------------------------------------
            sh_int = g["new"][2]["share"]
            orc = r["oracle"]
            gap_to_A = max(0.0, r["delta"]["total"])
            interior_gap = max(0.0, r["delta"]["within"][2])
            lo10 = abs(res["slope_cross_arm"]) * 10 * sh_int if np.isfinite(res["slope_cross_arm"]) else float("nan")
            hi10 = max(0.0, r.get("slope_w", float("nan"))) * 3.2 * sh_int if np.isfinite(r.get("slope_w", float("nan"))) else float("nan")
            r["levers"] = {
                "judge": {"observed": orc["observed"], "tau0.8": orc["tau0.8"], "tau0.5": orc["tau0.5"], "tau0": orc["tau0"],
                          "central": orc["observed"] - orc["tau0.5"], "hi": orc["observed"] - orc["tau0"]},
                "drafter_to_A": {"central": interior_gap, "hi": gap_to_A},
                "drafter_plus10": {"lo": lo10, "hi": hi10},
                "drafter_ceiling": {"floor": r["ceiling"]["floor"], "hi": r["ceiling"]["save"]},
                "drafter_steps_half": {"central": 0.0, "hi_loads": 0.5 * sp["drafter_charge"], "drafter_share": sp["drafter_share"]},
                "loads_per_pass": cpp,
            }
            r["levers"]["speedups"] = {
                "judge_central": spd(r["levers"]["judge"]["central"]), "judge_hi": spd(r["levers"]["judge"]["hi"]),
                "drafter_to_A_central": spd(interior_gap), "drafter_to_A_hi": spd(gap_to_A),
                "drafter_plus10_lo": spd(lo10) if np.isfinite(lo10) else None, "drafter_plus10_hi": spd(hi10) if np.isfinite(hi10) else None,
                "drafter_ceiling": spd(r["ceiling"]["save"]) if np.isfinite(r["ceiling"]["save"]) else None,
                "drafter_steps_hi": sp["stock_exp_tok"] / (sp["charged_exp_tok"] - 0.5 * sp["drafter_charge"] / gen)}


def pooled_levers(res: dict, main: dict | None = None) -> dict:
    """per drafter: % reduction of charged expert loads per lever, pooled over its complete sets (token weights) and for the
    best single set, plus the implied charged speedup. Levers with a central value are ranked; range-only ones are scenarios."""
    nan0 = lambda x: 0.0 if x is None or not np.isfinite(x) else x
    fns = {
        "Header strip / prefill": (lambda r: r["cf"]["header"]["central"], lambda r: r["cf"]["header"]["hi"]),
        "Padding-token remap": (lambda r: r["cf"]["padding"]["central"], lambda r: r["cf"]["padding"]["hi"]),
        "Rule-fill of the post-EOS tail": (lambda r: r["cf"]["tail_fill"], lambda r: r["cf"]["tail_fill"]),
        "Judge retrained on this drafter (confidence)": (lambda r: r["levers"]["judge"]["central"], lambda r: r["levers"]["judge"]["hi"]),
        "Drafter aligned to SDAR-1.7B's agreement": (lambda r: r["levers"]["drafter_to_A"]["central"], lambda r: r["levers"]["drafter_to_A"]["hi"]),
    }
    out = {}
    for arm, d in res["arms"].items():
        S = d["sets"]
        if not S:
            continue
        tot = sum(r["speed"]["charged_exp_tok"] * r["speed"]["gen"] for r in S.values())
        stock = sum(r["speed"]["stock_exp_tok"] * r["speed"]["gen"] for r in S.values())

        def red(f, rs=S):
            return 100 * sum(r["loads_per_pass"] * f(r) * r["blocks"] for r in rs.values()) / sum(
                r["speed"]["charged_exp_tok"] * r["speed"]["gen"] for r in rs.values())
        lv = {}
        for k, (fc, fh) in fns.items():
            per = {s: red(fc, {s: r}) for s, r in S.items()}
            best = max(per, key=per.get)
            lv[k] = {"central": red(fc), "lo": 0.0, "hi": red(fh), "best_set": best, "best_set_cut": per[best], "ranked": True}
        # scenarios / upper bounds without a central estimate
        hi_steps = 100 * sum(0.5 * r["speed"]["drafter_charge"] for r in S.values()) / tot
        per = {s: 100 * 0.5 * r["speed"]["drafter_charge"] / (r["speed"]["charged_exp_tok"] * r["speed"]["gen"]) for s, r in S.items()}
        lv["Fewer drafter steps (8 -> 4)"] = {"central": None, "lo": 0.0, "hi": hi_steps, "best_set": max(per, key=per.get),
                                              "best_set_cut": max(per.values()), "ranked": False}
        flo = lambda r: nan0(r["levers"]["drafter_plus10"]["lo"])
        fhi = lambda r: nan0(r["levers"]["drafter_plus10"]["hi"])
        per = {s: red(fhi, {s: r}) for s, r in S.items()}
        lv["Drafter +10 points content agreement"] = {"central": None, "lo": red(flo), "hi": red(fhi), "best_set": max(per, key=per.get),
                                                      "best_set_cut": max(per.values()), "ranked": False}
        fce = lambda r: nan0(r["levers"]["drafter_ceiling"]["hi"])
        per = {s: red(fce, {s: r}) for s, r in S.items()}
        lv["Ceiling: every block at >= 90% draft agreement"] = {"central": None, "lo": red(fce), "hi": red(fce), "best_set": max(per, key=per.get),
                                                                "best_set_cut": max(per.values()), "ranked": False}
        sp_now = main["pooled"][arm]["agg"]["new"]["speedup_vs_stock_charged"] if main and arm in main.get("pooled", {}) else stock / tot
        out[arm] = {"levers": lv, "speedup_now": sp_now, "gen": sum(r["speed"]["gen"] for r in S.values()), "sets": list(S)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# rendering (REPORT.md section), figures, IMPROVEMENT_PLAN.md
# ---------------------------------------------------------------------------------------------------------------
def _p(x, k=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{k}f}%"


def _n(x, k=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def _sg(x, k=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{k}f}"


def _rng(c, lo, hi, k=2):
    return f"{_n(c, k)} [{_n(lo, k)}–{_n(hi, k)}]"


ARM_ORDER = ["F2_nopool", "S1_nopool"]
SHORT = {"F2_nopool": "CoDA-1.7B", "S1_nopool": "Qwen3-0.6B", "A": "SDAR-1.7B (A)"}


def _sets_of(res):
    return [s for s in res["sets"] if any(s in res["arms"][a]["sets"] for a in res["arms"])]


def _any_r(res, s):
    for a in ARM_ORDER:
        if a in res["arms"] and s in res["arms"][a]["sets"]:
            return res["arms"][a]["sets"][s]
    return None


def render(res: dict, plan_name: str = "IMPROVEMENT_PLAN.md") -> list:
    L = []
    w = L.append
    sets = _sets_of(res)
    arms = [a for a in ARM_ORDER if a in res["arms"] and res["arms"][a]["sets"]]
    done = ", ".join(f"{SHORT[a]}: {', '.join(res['arms'][a]['sets'])}" for a in arms)
    w("## Deep dive: format tokens versus real content (both new drafters against Reference A)\n")
    w(f"Complete sets used: {done}. Everything here is CPU-only and offline: the logged drafts (`traces/`), the judge's per-read "
      "probabilities (`judge/`) and the records; nothing is re-decoded. **Format slots** are (i) *header*: the first block-0 slots "
      "of a new drafter's draft that spell `<think>` ... `</think>` plus the blank line after it (about 3.6 slots, present in 100% "
      "of block-0 drafts) and (ii) *padding*: the first `<|im_end|>` of an item's output and everything after it (`\\n`, then "
      "`<|endoftext|>` fill, which the 30B always writes; the new drafters' padding differs, see below). Every other slot is "
      "**content**. A slot is *kept* when the draft token equals the final token. Reference A has no logged drafts, only per-block "
      "counts, so for A the numbers marked *exact* use blocks without padding and the numbers marked *est.* assume A's padding "
      "agrees at the rate it shows on blocks that are almost all padding (the 'ta' value, "
      + ", ".join(f"{s} {_p(res['A_tail_agreement'][s], 0)}" for s in sets) + "; a lower bound), with the full bracket shown "
      "where it matters.\n")

    # ---------------------------------------------------------------- D1
    w("### D1. Content-only agreement\n")
    w("Slot-weighted agreement between the draft and the final tokens (not the per-block mean used in the speed tables). "
      "*Interior* = blocks after the first that contain no padding slot: no format slot of any kind, so the number is exact for all "
      "three drafters.\n")
    w("| set | drafter | header slots | padding slots | all slots | content-only, all blocks | content-only, later blocks | interior blocks (exact) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        r0 = _any_r(res, s)
        a = r0["agree_A"]
        w(f"| {s} | SDAR-1.7B (A) | – | {_p(a['tail_share'])} | {_p(a['all'])} | {_p(a['content'])} est. "
          f"[{_p(a['content_range'][0], 0)}–{_p(a['content_range'][1], 0)}] | {_p(a['content_later'])} est. | {_p(a['pure'])} |")
        for arm in arms:
            if s in res["arms"][arm]["sets"]:
                n = res["arms"][arm]["sets"][s]["agree_new"]
                w(f"| | {SHORT[arm]} | {_p(n['header_share'])} | {_p(n['tail_share'])} | {_p(n['all'])} | {_p(n['content'])} | "
                  f"{_p(n['content_later'])} | {_p(n['pure'])} |")
    w("\n![content agreement](content_agreement.png)\n")
    w("**By block position, content-only** (block 0 for the new drafters excludes the header slots but not the rest of the "
      "first block, whose draft still opens with a different preface; blocks 1, 2, 3+ contain only content slots except for "
      "their padding, which is excluded). A is est. wherever a block holds padding.\n")
    w("| set | drafter | block 0 | block 1 | block 2 | blocks 3+ |")
    w("|---|---|---:|---:|---:|---:|")
    for s in sets:
        r0 = _any_r(res, s)
        rows = [("SDAR-1.7B (A) est.", r0["agree_A"])] + [(SHORT[a], res["arms"][a]["sets"][s]["agree_new"]) for a in arms if s in res["arms"][a]["sets"]]
        for k, (nm, ag) in enumerate(rows):
            w(f"| {s if k == 0 else ''} | {nm} | " + " | ".join(_p(ag["by_pos"][p]["content"]) for p in ("0", "1", "2", "3+")) + " |")
    w("\n**How much of the all-slot agreement gap to A remains on real content.** Gap = A − drafter in agreement points over all "
      "slots, split into the padding slots, the first block's non-padding slots, and the later blocks' non-padding slots (real content; "
      "A's padding agreement is est.). The first block is a template effect for the new drafters (header plus preface), so it is "
      "not counted as real content.\n")
    w("| set | drafter | total gap (points) | padding | first block | later blocks (real content) | real-content share of the gap | content-only gap in later blocks (A − drafter, points) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        for k, arm in enumerate(arms):
            if s not in res["arms"][arm]["sets"]:
                continue
            r = res["arms"][arm]["sets"][s]
            g = r["gap"]
            share = g["later blocks"] / g["total"] if g["total"] > 1e-9 else None
            sh = "none" if share is not None and share <= 0.02 else _p(share, 0)
            w(f"| {s if arm == [a for a in arms if s in res['arms'][a]['sets']][0] else ''} | {SHORT[arm]} | {g['total']:+.1f} | "
              f"{g['tail']:+.1f} | {g['first block']:+.1f} | {g['later blocks']:+.1f} | {sh} | "
              f"{100 * ((r['agree_A']['content_later'] or 0) - (r['agree_new']['content_later'] or 0)):+.1f} |")
    w("")

    # ---------------------------------------------------------------- D2
    w("### D2. Counterfactual: strip or pre-fill the header, remap the padding token (estimates)\n")
    w("What the logs show first. The judge overrides the header slots early (share committed at the very first pass: "
      + ", ".join(f"{SHORT[a]} {_p(res['dist'][a]['header']['commit_pass0'], 0)}" for a in arms if a in res['dist'] and 'header' in res['dist'][a])
      + "; they cost little), and the padding slots are committed late (mean commit pass "
      + ", ".join(f"{SHORT[a]} {_n(res['dist'][a]['padding (tail)']['commit_pass_mean'], 1)}" for a in arms if a in res['dist'] and 'padding (tail)' in res['dist'][a])
      + ") for every drafter, matched padding or not (consistent with the judge waiting for the end-of-turn token to be fixed first). "
      "So both fixes are expected to move agreement a lot and passes only a little.\n")
    w("**Assumptions (all est.).** (1) *Header strip / pre-fill*: the draft is shifted left by the header length (the drafter would "
      "write the same words, minus the header); agreement over the overlapping slots is taken as the new block-0 agreement. "
      "Passes: the gain is the share of the gap to A's block-0 agreement that this closes, times the block-0 pass gap to A, times the "
      "block-0 share of blocks (reference-class interpolation; range = nothing .. A-like block 0). (2) *Padding remap*: after the "
      "first `<|im_end|>` the draft's `<|im_end|>` repeats are replaced by `<|endoftext|>` so padding agrees at A's rate. Passes: "
      "two estimators. (i) Regression over both drafters' blocks on the same items, passes ~ content misses + padding mismatches + "
      "padding matches (item-cluster bootstrap 95% interval for the extra passes per mismatched padding slot). (ii) The measured "
      "pass gap between the drafter's last blocks (those with padding) and A's, whose padding is already right (exact, from the "
      "block-type table below). The central value is the smaller of the two, clipped at 0, because the regression can be confounded "
      "by content differences between drafters; the range's upper end is the regression interval's upper end. (3) Expert loads move in proportion to judge passes (loads per judge pass fitted per drafter and set); the "
      "drafter's own charge and the accuracy are unchanged. (4) The judge would behave on the modified drafts as it does on the "
      "logged ones; only a GPU decode can check this.\n")
    w("| set | drafter | block-0 agreement: now → header stripped (A) | est. passes saved/block: header | tail agreement now → target | extra passes per mismatched padding slot [95% CI] | est. passes saved/block: padding | passes/block now → est. after both | charged speedup now → est. after both |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        for k, arm in enumerate(arms):
            if s not in res["arms"][arm]["sets"]:
                continue
            r = res["arms"][arm]["sets"][s]
            cf = r["cf"]
            h, p = cf["header"], cf["padding"]
            pc = r["pad"]["coef"]
            pcs = "n/a (one drafter)" if pc is None else f"{pc['per_slot']:+.3f} [{pc['lo']:+.3f}, {pc['hi']:+.3f}]"
            w(f"| {s if arm == [a for a in arms if s in res['arms'][a]['sets']][0] else ''} | {SHORT[arm]} | "
              f"{_p(h['agree_obs'])} → {_p(h['agree_stripped'])} ({_p(h['agree_A'])}) | {_rng(h['central'], h['lo'], h['hi'], 3)} | "
              f"{_p(p['agree_obs'])} → {_p(p['agree_target'])} | {pcs} | {_rng(p['central'], p['lo'], p['hi'], 3)} | "
              f"{_n(cf['P0'])} → {_n(cf['P_format_central'])} ({_n(cf['P_format_hi'])} best case) | "
              f"{_n(cf['speedup_obs'], 3)}x → {_n(cf['speedup_format_central'], 3)}x ({_n(cf['speedup_format_hi'], 3)}x best case) |")
    w("\nThe padding regression is only an identification strategy, not a proof: its coefficient is positive on MBPP and "
      "IFEval, near zero on HumanEval and *negative* on GSM8K, so it cannot be read causally on its own; the measured last-block "
      "gap to A caps the central estimate (it is zero or negative where the drafter's last blocks are not slower than A's).")
    w("\nWhy the header fix is set-specific: after the header is removed the first block's draft is the drafter's own first words. On "
      "HumanEval (the prompt is a function stub) both drafters then write the code straight away and agree with the 30B on "
      "about two thirds of the block; on MBPP the Qwen3 draft starts with a sentence of prose (`Here is a Python function ...`) "
      "while the 30B opens with a code fence, so stripping the header alone recovers nothing for Qwen3 there (1% → 1%) and about half "
      "for CoDA (2% → 30%).\n")
    w("Separate, applies to every drafter including A, not a drafter fix: **rule-filling the tail.** The 30B's post-EOS text is "
      "`<|im_end|>`, `\\n`, then `<|endoftext|>` for every item (98-100% of outputs of every arm). If, once `<|im_end|>` is committed, the "
      "rest of the block were filled by rule, the block would end after its last content or `<|im_end|>` commit. Est. passes saved per "
      "block: " + "; ".join(f"{s}: " + ", ".join(f"{SHORT[a]} {_n(res['arms'][a]['sets'][s]['cf']['tail_fill'], 3)}" for a in arms if s in res['arms'][a]['sets'])
                            for s in sets) + ". It is small because the content slots, not the padding, decide when a block ends.\n")
    w("![format-fix estimates](format_fix_estimates.png)\n")

    # ---------------------------------------------------------------- D3
    w("### D3. Where the extra passes come from (new drafter against Reference A)\n")
    w("**Exact accounting by block type (records only).** *First block* = block 0; *last block* = a later block that holds padding "
      "slots; *interior* = later blocks without padding. Passes are judge passes per block.\n")
    w("| set | block type | share of blocks (A / CoDA / Qwen3) | stock | A | CoDA | Qwen3 |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for s in sets:
        ref = None
        have = {a: res["arms"][a]["sets"][s] for a in arms if s in res["arms"][a]["sets"]}
        r0 = _any_r(res, s)
        for gk in (0, 1, 2):
            shares = [r0["groups"]["A"][gk]["share"]] + [have[a]["groups"]["new"][gk]["share"] if a in have else None for a in ARM_ORDER]
            cells = [r0["groups"]["stock"][gk]["passes"], r0["groups"]["A"][gk]["passes"]] + [have[a]["groups"]["new"][gk]["passes"] if a in have else None for a in ARM_ORDER]
            w(f"| {s if gk == 0 else ''} | {GROUPS[gk]} | " + " / ".join(_p(x, 0) if x is not None else "–" for x in shares) + " | "
              + " | ".join(_n(x) for x in cells) + " |")
        cells = [r0["groups"]["stock"]["all"]["passes"], r0["groups"]["A"]["all"]["passes"]] + [have[a]["groups"]["new"]["all"]["passes"] if a in have else None for a in ARM_ORDER]
        w(f"| | **all blocks** | | " + " | ".join(f"**{_n(x)}**" for x in cells) + " |")
    w("\n**Decomposition of the pass difference, new drafter − A (passes per block).** Exact split of the difference into the "
      "within-block-type terms and a mix term (different output lengths change how many blocks of each type there are): "
      "(a) first block = header + preface, (a) last block = padding, then the interior content term split into "
      "(b) misses the judge rated plausible (first-read probability of the draft token ≥ " + f"{res['theta']:g}" + ") and "
      "(c) genuine misses (< " + f"{res['theta']:g}" + "), pro rata to the pass attribution of those two classes in the new drafter's own "
      "interior-and-other blocks (see the attribution table below; an assumption, because A's drafts are not logged). "
      "Negative = the new drafter needs fewer passes than A.\n")
    w("| set | drafter | passes/block new − A | (a) first block | (a) padding | content, interior | of which (b) plausible alternative | of which (c) genuine miss | mix |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        for k, arm in enumerate(arms):
            if s not in res["arms"][arm]["sets"]:
                continue
            x = res["arms"][arm]["sets"][s]["abc"]
            w(f"| {s if arm == [a for a in arms if s in res['arms'][a]['sets']][0] else ''} | {SHORT[arm]} | {_sg(x['total'])} | {_sg(x['a_first_block'])} | "
              f"{_sg(x['a_padding'])} | {_sg(x['content_interior'])} | {_sg(x['b_plausible_alternative'])} | {_sg(x['c_genuine_miss'])} | {_sg(x['mix'])} |")
    w("\n**Slot-class attribution of the new drafters' passes.** Every judge pass is split equally among the slots it commits; a "
      "forced pass commits one slot and is charged to it whole. Cells are passes per block (slots per block). Plausible alternative / "
      "genuine miss = content slots where the draft token ≠ the final token and the judge's first-read probability of the draft "
      "token was ≥ / < " + f"{res['theta']:g}" + ". An attribution convention, not a causal effect.\n")
    w("| set | drafter | header | padding | content, kept | content, plausible alternative | content, genuine miss | = passes/block | A passes/block |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        for k, arm in enumerate(arms):
            if s not in res["arms"][arm]["sets"]:
                continue
            r = res["arms"][arm]["sets"][s]
            at = r["attribution"]
            sl = at["slots_per_block"]
            cell = lambda key: f"{_n(at[key])} ({_n(sl[key], 1)})"
            w(f"| {s if arm == [a for a in arms if s in res['arms'][a]['sets']][0] else ''} | {SHORT[arm]} | {cell('header')} | {cell('padding (tail)')} | {cell('content, kept')} | "
              f"{cell('content, plausible alternative')} | {cell('content, genuine miss')} | {_n(at['total'])} | {_n(r['speed']['A_passes_per_block'])} |")
    w("\n![pass attribution](pass_attribution.png)\n")
    w("**Distributions per slot class** (all complete benchmarks of each drafter combined). `p(draft)` = the judge's probability of the "
      "draft token at the first read (all valid slots then hold the draft); drafter conf = the drafter's own confidence in that "
      "token; `argmax = final` = the judge's first-read top-1 already equals the token finally committed; on-trajectory columns use "
      "only blocks whose prefix equals the stock decode's.\n")
    w("| drafter | slot class | n | p(draft), median [10%, 90%] | drafter conf, median | drafter conf > 0.9 | committed at first pass | forced commit | mean commit pass | argmax = final at first read | on-traj: draft = stock / final = stock |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = ["header", "padding (tail)", "content, kept", "content, plausible alternative", "content, genuine miss"]
    for arm in arms:
        for k, g in enumerate(order):
            d = res["dist"].get(arm, {}).get(g)
            if not d:
                continue
            q = d["q1_q"]
            w(f"| {SHORT[arm] if k == 0 else ''} | {g} | {d['n']} | {q[1]:.3f} [{q[0]:.4f}, {q[2]:.3f}] | {d['drafter_conf_q'][1]:.2f} | "
              f"{_p(d['drafter_conf_gt90'], 0)} | {_p(d['commit_pass0'], 0)} | {_p(d['forced'], 0)} | {_n(d['commit_pass_mean'], 1)} | "
              f"{_p(d['first_read_argmax_is_final'], 0)} | {_p(d['on_traj_draft_eq_stock'], 0)} / {_p(d['on_traj_final_eq_stock'], 0)} |")
    w("\n![distributions](content_distributions.png)\n")
    w("**Concrete examples** (seeded random draws from the complete sets; `ctx` = the 7 preceding final tokens; p(draft) = judge "
      "probability of the draft token at the first read; top-1 = the judge's first-read argmax; conf = drafter confidence; "
      "commit = how and at which judge pass the slot was finally committed; stock = the stock decode's token where the block is "
      "on the stock trajectory).\n")
    for cat, items in res["examples"].items():
        w(f"*{cat}*\n")
        for e in items:
            stk = f", stock {e['stock']!r}" if e["stock"] is not None else ""
            w(f"- {SHORT[e['arm']]} {e['set']} item {e['item']} block {e['block']} slot {e['slot']}: ctx `{e['context']!r}` → draft {e['draft']!r}, "
              f"final {e['final']!r}{stk}; p(draft) {e['q1']:.3f}, top-1 {e['judge_top1']!r} ({e['p1']:.2f}), conf {e['drafter_conf']:.2f}, "
              f"{e['commit']} at pass {e['commit_pass']}")
        w("")
    w(f"The ranked improvement plan with the estimates above, and the training bounds, is in `{plan_name}`.\n")
    return L


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
def plots(res: dict, AN) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": AN.INK, "axes.labelcolor": AN.INK2, "axes.titlecolor": AN.INK})
    sets = _sets_of(res)
    arms = [a for a in ARM_ORDER if a in res["arms"] and res["arms"][a]["sets"]]
    col = {"A": AN.COLORS["A"], "F2_nopool": AN.COLORS["F2_nopool"], "S1_nopool": AN.COLORS["S1_nopool"]}
    names = {"A": "SDAR-1.7B (A)", "F2_nopool": "CoDA-1.7B", "S1_nopool": "Qwen3-0.6B"}
    ns = len(sets)
    # ---- F1: agreement ---------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, ns, figsize=(2.9 * ns + 0.6, 3.6), sharey=True, squeeze=False)
    metrics = [("all slots", "all"), ("content only\n(all blocks)", "content"), ("content only\n(later blocks)", "content_later")]
    for ax, s in zip(axs[0], sets):
        AN._style(ax)
        r0 = _any_r(res, s)
        series = [("A", r0["agree_A"])] + [(a, res["arms"][a]["sets"][s]["agree_new"]) for a in arms if s in res["arms"][a]["sets"]]
        wd = 0.8 / len(series)
        for j, (a, ag) in enumerate(series):
            xs = np.arange(len(metrics)) + (j - (len(series) - 1) / 2) * wd
            vals = [100 * (ag[k] or 0) for _, k in metrics]
            ax.bar(xs, vals, wd * 0.88, color=col[a], label=names[a] if s == sets[0] else None, edgecolor="#fcfcfb", linewidth=1.5)
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=AN.INK2)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m for m, _ in metrics], fontsize=7)
        ax.set_title(s, fontsize=9); ax.set_ylim(0, 100)
    axs[0][0].set_ylabel("draft = final token (% of slots)", fontsize=8)
    axs[0][0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.text(0.01, 0.005, "A's content-only values are est. (padding agreement taken from near-pure-padding blocks).", fontsize=6.5, color=AN.INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(AN.OUT, "content_agreement.png"), dpi=150); plt.close(fig)

    # ---- F2: pass attribution -------------------------------------------------------------------------------------
    cls_cols = {"header": "#eda100", "padding (tail)": "#e87ba4", "content, kept": "#008300",
                "content, plausible alternative": "#4a3aa7", "content, genuine miss": "#e34948"}
    fig, axs = plt.subplots(1, ns, figsize=(2.9 * ns + 0.6, 3.8), squeeze=False)
    for ax, s in zip(axs[0], sets):
        AN._style(ax)
        have = [a for a in arms if s in res["arms"][a]["sets"]]
        xs = np.arange(len(have))
        bottom = np.zeros(len(have))
        ymax = max(sum(res["arms"][a]["sets"][s]["attribution"][k] for k in cls_cols) for a in have)
        for k, c in cls_cols.items():
            vals = np.array([res["arms"][a]["sets"][s]["attribution"][k] for a in have])
            ax.bar(xs, vals, 0.55, bottom=bottom, color=c, edgecolor="#fcfcfb", linewidth=2, label=k if s == sets[0] else None)
            for x, v, b0 in zip(xs, vals, bottom):
                if v > 0.045 * ymax:
                    ax.text(x, b0 + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="#ffffff" if k != "header" and k != "padding (tail)" else AN.INK)
            bottom += vals
        r0 = _any_r(res, s)
        ax.hlines(r0["speed"]["A_passes_per_block"], -0.5, len(have) - 0.5, color=AN.INK, lw=1.6, linestyles="--")
        ax.text(len(have) - 0.5, r0["speed"]["A_passes_per_block"], f" A {r0['speed']['A_passes_per_block']:.1f}", va="bottom", ha="right", fontsize=7, color=AN.INK)
        ax.set_xticks(xs); ax.set_xticklabels([names[a] for a in have], fontsize=7.5)
        ax.set_title(s, fontsize=9); ax.set_xlim(-0.6, len(have) - 0.4)
    axs[0][0].set_ylabel("judge passes per block (attributed)", fontsize=8)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(AN.OUT, "pass_attribution.png"), dpi=150); plt.close(fig)

    # ---- F3: distributions ----------------------------------------------------------------------------------------
    SL = res["_slot"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5))
    bins_q = np.array([0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.95, 1.0001])
    lab_q = ["<1e-4", "1e-3", "1e-2", "0.05", "0.1", "0.2", "0.5", "0.95", ">0.95"]
    for arm in arms:
        Xs = [X for (a, s), X in SL.items() if a == arm]
        cat = lambda k: np.concatenate([X[k] for X in Xs])
        cls, D, T, q1, pdr, cp = cat("cls"), cat("D"), cat("T"), cat("q1"), cat("pdr"), cat("cpass")
        kept, mis = (cls == 0) & (D == T), (cls == 0) & (D != T)
        for m, ls, nm in ((kept, "-", "kept"), (mis, "--", "missed")):
            h, _ = np.histogram(q1[m], bins_q)
            axs[0].plot(range(len(h)), 100 * h / h.sum(), ls, color=col[arm], lw=2, label=f"{names[arm]}, {nm}")
            h, _ = np.histogram(pdr[m], np.linspace(0, 1.0001, 11))
            axs[1].plot(np.linspace(0.05, 0.95, 10), 100 * h / h.sum(), ls, color=col[arm], lw=2)
        classes = {"header": cls == 1, "padding": cls == 2, "kept": kept, "alt": mis & (q1 >= THETA), "genuine": mis & (q1 < THETA)}
        xs = np.arange(len(classes)) + (arms.index(arm) - (len(arms) - 1) / 2) * 0.36
        axs[2].bar(xs, [cp[m].mean() for m in classes.values()], 0.34, color=col[arm], edgecolor="#fcfcfb", linewidth=1.5)
    for ax in axs:
        AN._style(ax)
    axs[0].set_xticks(range(len(lab_q))); axs[0].set_xticklabels(lab_q, fontsize=6.5, rotation=45)
    axs[0].set_xlabel("judge p(draft token) at the first read", fontsize=8); axs[0].set_ylabel("% of content slots", fontsize=8)
    axs[0].set_title("how the judge rates the draft token", fontsize=8.5); axs[0].legend(frameon=False, fontsize=6.5)
    axs[1].set_xlabel("drafter confidence in its token", fontsize=8); axs[1].set_title("drafter confidence, kept vs missed", fontsize=8.5)
    axs[2].set_xticks(range(5)); axs[2].set_xticklabels(["header", "padding", "kept", "plausible\nalt.", "genuine\nmiss"], fontsize=7)
    axs[2].set_ylabel("mean judge pass of the commit", fontsize=8); axs[2].set_title("when a slot is committed", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(AN.OUT, "content_distributions.png"), dpi=150); plt.close(fig)

    # ---- F4: format-fix speedup estimates -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6.5, 1.15 * ns * len(arms) + 1.5), 3.8))
    AN._style(ax)
    xs, k = [], 0
    ticks, tl = [], []
    for s in sets:
        for a in arms:
            if s not in res["arms"][a]["sets"]:
                continue
            cf = res["arms"][a]["sets"][s]["cf"]
            for j, (val, hi, nm, c) in enumerate(((cf["speedup_obs"], None, "now", "#898781"),
                                                  (cf["speedup_format_central"], cf["speedup_format_hi"], "header + padding fix (est.)", col[a]),
                                                  (cf["speedup_format_tail_central"], None, "+ tail rule-fill (est.)", "#52514e"))):
                x = k + (j - 1) * 0.27
                ax.bar(x, val, 0.25, color=c, edgecolor="#fcfcfb", linewidth=1.5, label=nm if (k == 0 and a == arms[0]) else None, alpha=1 if j else 0.9)
                if hi:
                    ax.plot([x, x], [val, hi], color=AN.INK, lw=1.2)
                    ax.plot([x - 0.05, x + 0.05], [hi, hi], color=AN.INK, lw=1.2)
            ticks.append(k); tl.append(f"{s}\n{names[a]}")
            k += 1
    ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=6.5)
    lo = min(res["arms"][a]["sets"][s]["cf"]["speedup_obs"] for a in arms for s in res["arms"][a]["sets"]) - 0.1
    ax.set_ylim(max(1.0, lo), None)
    ax.set_ylabel("charged speedup vs stock (x)", fontsize=8)
    ax.set_title("format fixes: estimated charged speedup vs stock (bars start at 1.0x = stock; whisker = best case)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(AN.OUT, "format_fix_estimates.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# IMPROVEMENT_PLAN.md
# ---------------------------------------------------------------------------------------------------------------
LEVER_META = {
    "Header strip / prefill": ("no training", "drafter prompt/template change (pre-fill the empty `<think>` block in the drafter's "
                               "prompt, or drop the header and shift the draft); one GPU decode to verify",
                               "gain is set-specific: it needs the drafter's first words to match the 30B's"),
    "Padding-token remap": ("no training", "one-line draft post-process: after the first `<|im_end|>` write `<|endoftext|>` (only Qwen3 "
                            "needs it; CoDA already pads like the 30B)", "central = the smaller of a two-drafter regression and the measured last-block pass gap to A; the regression alone is larger and can be confounded by content differences"),
    "Rule-fill of the post-EOS tail": ("no training", "decode-loop rule: after `<|im_end|>` is committed fill `\\n` + `<|endoftext|>` without "
                                       "judge passes; applies to every drafter incl. A", "tokens after EOS never reach the answer; verify token-identical output on GPU"),
    "Fewer drafter steps (8 -> 4)": ("no training", "config only (`s4` or `dyn90`); halves the drafter's charge", "UPPER BOUND ONLY: assumes agreement "
                                     "does not fall, which is not measured and usually does"),
    "Judge retrained on this drafter (confidence)": ("training (LoRA)", "retrain the judge on this drafter's drafts (the 128M judge was trained on SDAR-1.7B drafts)",
                                                     "static bound (no compounding of earlier commits); range = nothing .. perfect confidence on already-right first reads"),
    "Drafter aligned to SDAR-1.7B's agreement": ("training (drafter)", "fine-tune or distil the drafter toward the 30B's outputs (out of scope in this no-training phase)",
                                                 "central = the measured interior-block pass gap to A, best case = the whole pass gap to A"),
    "Ceiling: every block at >= 90% draft agreement": ("training (drafter), theoretical", "a drafter that matches the 30B almost everywhere",
                                                       "optimistic ceiling: each block's passes capped at the mean of this drafter's own blocks with >= 90% agreement (est.)"),
    "Drafter +10 points content agreement": ("training (drafter)", "as above", "range from the cross-drafter slope (low) to the within-drafter slope (high); the two disagree because hard blocks have both more misses and more passes"),
}


def write_plan(res: dict, AN, pooled: dict, verdict: list | None = None) -> None:
    KA = "Drafter aligned to SDAR-1.7B's agreement"
    sets = _sets_of(res)
    arms = [a for a in ARM_ORDER if a in res["arms"] and res["arms"][a]["sets"]]
    L = []
    w = L.append
    w("# Improvement plan: what could still speed up the no-pool drafter swap\n")
    w(f"Generated {AN.time.strftime('%Y-%m-%dT%H:%M:%SZ', AN.time.gmtime())} by `analyze_nopool.py` + `content_analysis.py` (CPU only, offline). "
      "Complete sets: " + "; ".join(f"{SHORT[a]}: {', '.join(res['arms'][a]['sets'])}" for a in arms) + ". "
      "Reference A = SDAR-1.7B drafter + the same judge, no pool. **Every number below that is not a measurement is an estimate (est.)**; "
      "the method and assumptions are in the REPORT.md section *Deep dive*. Speed here means the charged expert loads per token "
      "(expert loads + drafter forwards × drafter cost), the repo's definition; accuracy is not touched by any lever except where noted.\n")
    w("**Key (plain words).** A *pass* is one forward pass of the big 30B judge over a 32-token block; fewer passes per block means faster "
      "decoding. *Draft agreement* is the share of positions where the small drafter's token is the token that was finally kept. "
      "*Header* is the empty `<think></think>` text the new drafters write first; *padding* is the filler after the end-of-turn token. "
      "*Charged speedup* is the stock decode's expert cost divided by this method's cost, counting the drafter's own work. "
      "*est.* means an offline estimate from the logs, not a measurement; a range in brackets is low – high.\n")
    if verdict:
        L.extend(verdict)
    # ------------------------------------------------------------------------------------------ bottom line
    w("## Bottom line\n")
    fmt_short, fmt_hi_short = [], []
    for a in arms:
        for s in ("humaneval", "mbpp"):
            if s in res["arms"][a]["sets"]:
                cf = res["arms"][a]["sets"][s]["cf"]
                fmt_short.append((a, s, cf["format_central"] / cf["P0"], cf["format_hi"] / cf["P0"]))
    fmt_long = [(a, s, res["arms"][a]["sets"][s]["cf"]["format_central"] / res["arms"][a]["sets"][s]["cf"]["P0"],
                 res["arms"][a]["sets"][s]["cf"]["format_hi"] / res["arms"][a]["sets"][s]["cf"]["P0"])
                for a in arms for s in res["arms"][a]["sets"] if s not in ("humaneval", "mbpp")]
    q = [res["arms"]["S1_nopool"]["sets"][s] for s in res["arms"].get("S1_nopool", {}).get("sets", {})]
    h0 = [res["arms"][a]["sets"]["humaneval"]["cf"]["header"] for a in arms if "humaneval" in res["arms"][a]["sets"]]
    agree_example = ((f"Qwen3 padding agreement {_p(min(r['pad']['obs_tail_agreement'] for r in q), 0)}–{_p(max(r['pad']['obs_tail_agreement'] for r in q), 0)} → about "
                      f"{_p(float(np.mean([r['ta'] for r in q])), 0)}; " if q else "")
                     + (f"HumanEval block-0 agreement {_p(h0[0]['agree_obs'], 0)} → {_p(h0[0]['agree_stripped'], 0)} (A {_p(h0[0]['agree_A'], 0)})" if h0 else ""))
    rng_s = lambda L: f"{100 * min(x[2] for x in L):.1f}–{100 * max(x[2] for x in L):.1f}% (best case up to {100 * max(x[3] for x in L):.1f}%)" if L else "n/a"
    w(f"- **The format fixes (header, padding) are not enough on their own.** Est. share of judge passes removed on the short-output sets "
      f"(HumanEval, MBPP): {rng_s(fmt_short)}; on the long-output sets: {rng_s(fmt_long)}. They move the *agreement* number a great deal "
      "(" + agree_example + ") but the judge already overrides the header slots early and commits the padding late whether or not the draft "
      "matched it, so passes barely change.")
    parts = []
    for a in arms:
        pl = pooled[a]["levers"]
        fc = sum(pl[k]["central"] for k in ("Header strip / prefill", "Padding-token remap"))
        fh = sum(pl[k]["hi"] for k in ("Header strip / prefill", "Padding-token remap"))
        parts.append(f"{SHORT[a]} {fc:.1f}% (best case {fh:.1f}%)")
    w("- Over all of each drafter's complete benchmarks combined (weighted by generated tokens, so the long GSM8K / IFEval outputs dominate), header + padding together "
      "cut the charged expert loads by est. " + "; ".join(parts) + ". The no-training levers combined stay in the low single digits.")
    sents = []
    for s in sets:
        bits = []
        for a in arms:
            if s in res["arms"][a]["sets"]:
                x = res["arms"][a]["sets"][s]["abc"]
                bits.append(f"{SHORT[a]} {x['total']:+.2f} (first block {x['a_first_block']:+.2f}, padding {x['a_padding']:+.2f}, interior content {x['content_interior']:+.2f})")
        sents.append(f"{s}: " + "; ".join(bits))
    w("- **Where the extra passes vs SDAR-1.7B really are** (measured, exact split, passes per block; negative = fewer than A): "
      + " | ".join(sents) + ". So MBPP's gap is mostly first block + padding (format), GSM8K's is real content in interior blocks "
      "(interior agreement " + ", ".join(
          f"{SHORT[a]} {_p(res['arms'][a]['sets']['gsm8k500']['agree_new']['pure'], 0)}" for a in arms if "gsm8k500" in res["arms"][a]["sets"]) + " vs A "
      + (_p(_any_r(res, "gsm8k500")["agree_A"]["pure"], 0) if "gsm8k500" in sets else "–") + "). Interior-block agreement more than 3 points below A "
      "(a content gap): " + (", ".join(f"{SHORT[a]} {s} ({_p(res['arms'][a]['sets'][s]['agree_new']['pure'], 0)} vs {_p(res['arms'][a]['sets'][s]['agree_A']['pure'], 0)})"
                                      for s in sets for a in arms if s in res["arms"][a]["sets"]
                                      and (res["arms"][a]["sets"][s]["agree_A"]["pure"] or 0) - (res["arms"][a]["sets"][s]["agree_new"]["pure"] or 0) > 0.03) or "none") + ".")
    w("- **Judge retraining is not where the speed is** (est.): even a judge that were perfectly confident on every first-read argmax that "
      "is already right would save only the 'judge' row below (static bound). **Drafter alignment is the largest lever** (est.: reaching A's content agreement cuts charged loads by up to "
      + ", ".join(f"{pooled[a]['levers'][KA]['best_set_cut']:.1f}% on {pooled[a]['levers'][KA]['best_set']} for {SHORT[a]}" for a in arms)
      + "; more beyond A: see the ceiling column), and it is training (out of scope here). Accuracy already holds without any of this, so none of it is needed for correctness.\n")
    # ------------------------------------------------------------------------------------------ ranked tables
    w("## Ranked levers (all complete benchmarks of each drafter combined)\n")
    w("Rank = estimated cut in charged expert loads (central value) over all complete benchmarks combined, weighted by generated tokens, so the long GSM8K / IFEval outputs "
      "dominate; the *best single set* column shows where each lever matters most. `now` = the drafter's combined charged speedup vs "
      "stock; `after` uses the central value (the best case in brackets). Levers without a central estimate are listed after the "
      "ranking as scenarios or upper bounds.\n")
    for a in arms:
        pl = pooled[a]
        lvs = pl["levers"]
        ranked = sorted([(v["central"], k, v) for k, v in lvs.items() if v["ranked"]], key=lambda x: -x[0])
        other = [(k, v) for k, v in lvs.items() if not v["ranked"]]
        spd = lambda x: pl["speedup_now"] / (1 - x / 100)
        w(f"### {SHORT[a]} (sets: {', '.join(pl['sets'])}; combined charged speedup now {pl['speedup_now']:.3f}x)\n")
        w("| rank | lever | needs | est. cut in charged expert loads, central [low – high] | best single set | combined charged speedup after (best case) | how | caveat |")
        w("|---:|---|---|---:|---:|---:|---|---|")
        for rk, (c, k, v) in enumerate(ranked, 1):
            kind, how, cav = LEVER_META[k]
            w(f"| {rk} | {k} | {kind} | {v['central']:.2f}% [{v['lo']:.2f}% – {v['hi']:.2f}%] | {v['best_set']} {v['best_set_cut']:.1f}% | "
              f"{spd(c):.3f}x ({spd(v['hi']):.3f}x) | {how} | {cav} |")
        w("")
        w("Not ranked (no central estimate): " + "; ".join(
            f"**{k}**: [{v['lo']:.2f}% – {v['hi']:.2f}%] (best single set {v['best_set']} {v['best_set_cut']:.1f}%; {LEVER_META[k][2]})" for k, v in other) + ".\n")
    # ------------------------------------------------------------------------------------------ per set
    w("## Per set: the no-training fixes (est.)\n")
    w("Short outputs (HumanEval, MBPP) have more padding and more first blocks per token, so the same fix is worth more there.\n")
    w("| set | drafter | passes/block now | header fix | padding fix | rule-fill tail | header + padding: passes/block after (best case) | charged speedup now → after (best case) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in sets:
        for a in arms:
            if s not in res["arms"][a]["sets"]:
                continue
            cf = res["arms"][a]["sets"][s]["cf"]
            w(f"| {s} | {SHORT[a]} | {_n(cf['P0'])} | −{_n(cf['header']['central'], 3)} [0–{_n(cf['header']['hi'], 3)}] | "
              f"−{_n(cf['padding']['central'], 3)} [0–{_n(cf['padding']['hi'], 3)}] | −{_n(cf['tail_fill'], 3)} | "
              f"{_n(cf['P_format_central'])} ({_n(cf['P_format_hi'])}) | {_n(cf['speedup_obs'], 3)}x → {_n(cf['speedup_format_central'], 3)}x ({_n(cf['speedup_format_hi'], 3)}x) |")
    w("\n## Training options (assessment only; nothing trained)\n")
    w("### Retrain the judge LoRA on the new drafter's drafts\n")
    w("What it could plausibly fix: the first-read recall of correct drafts (the judge commits only about 70% of correct first-read "
      "drafts at the first pass; REPORT.md calibration table) and the forced commits (about 30-65% of all judge passes, one slot each). "
      "Upper bound from the probability logs "
      "(est., static): if every slot whose first-read argmax already equals the final token were committed at the first pass whenever the "
      "judge's first-read probability is at least τ', and everything else kept its logged commit pass:\n")
    w("| set | drafter | passes/block now | τ' = 0.8 | τ' = 0.5 | τ' = 0 (perfect confidence) | charged speedup now → τ' = 0 |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for s in sets:
        for a in arms:
            if s not in res["arms"][a]["sets"]:
                continue
            r = res["arms"][a]["sets"][s]
            o = r["oracle"]
            w(f"| {s} | {SHORT[a]} | {_n(o['observed'])} | {_n(o['tau0.8'])} | {_n(o['tau0.5'])} | {_n(o['tau0'])} | "
              f"{_n(r['cf']['speedup_obs'], 3)}x → {_n(r['levers']['speedups']['judge_hi'], 3)}x |")
    w("\nThe bound is small (a few percent of passes) because the slots that make a block long are not under-confident right answers: "
      "they are slots whose first-read argmax is *not* the final token, which only become decidable after other slots are committed. "
      "A retrained judge could compound (earlier commits make later reads better), which this static bound cannot show; there is no "
      "evidence for a large compounding gain, because the existing judge, trained on SDAR-1.7B drafts only, already needs *no more* passes "
      "than A at equal agreement on both new drafters (REPORT.md, passes against agreement). **Expectation: low priority (est. ≤ a few percent).**\n")
    w("### Fine-tune or align the drafter itself (out of scope for this no-training phase; estimate of what it would buy)\n")
    gaps = []
    for s in sets:
        for a in arms:
            if s in res["arms"][a]["sets"]:
                r = res["arms"][a]["sets"][s]
                g = 100 * ((r["agree_A"]["pure"] or 0) - (r["agree_new"]["pure"] or 0))
                if g > 3:
                    gaps.append(f"{s} {SHORT[a]}: interior agreement {_p(r['agree_new']['pure'], 0)} vs A {_p(r['agree_A']['pure'], 0)}, "
                                f"{_sg(r['groups']['new'][2]['passes'] - r['groups']['A'][2]['passes'])} passes per interior block vs A")
    w("Measured: the real-content gap to SDAR-1.7B (interior blocks, more than 3 agreement points below A) is: "
      + ("; ".join(gaps) if gaps else "none") + ". Est. benefit of closing exactly the measured interior-block gap to A (central), of "
      "the whole pass gap to A (best case), and of +10 points of content agreement (range from the cross-drafter slope "
      + f"{abs(res['slope_cross_arm']):.3f} passes per interior block per agreement point to the within-drafter slope):\n")
    w("| set | drafter | passes/block now | close the gap to A (central / best case) | +10 points content agreement (low – high) | ceiling: every block ≥ 90% agreement (passes/block floor) | charged speedup now → after close-gap (best case) → ceiling |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for s in sets:
        for a in arms:
            if s not in res["arms"][a]["sets"]:
                continue
            r = res["arms"][a]["sets"][s]
            lv = r["levers"]
            w(f"| {s} | {SHORT[a]} | {_n(r['cf']['P0'])} | −{_n(lv['drafter_to_A']['central'])} / −{_n(lv['drafter_to_A']['hi'])} | "
              f"−{_n(lv['drafter_plus10']['lo'])} – −{_n(lv['drafter_plus10']['hi'])} | −{_n(lv['drafter_ceiling']['hi'])} (floor {_n(lv['drafter_ceiling']['floor'])}) | "
              f"{_n(r['cf']['speedup_obs'], 3)}x → {_n(lv['speedups']['drafter_to_A_central'], 3)}x ({_n(lv['speedups']['drafter_to_A_hi'], 3)}x) → {_n(lv['speedups']['drafter_ceiling'], 3)}x |")
    w("\nWhere the interior-block gap to A is zero there is nothing to close and the first column is 0; the +10-point and ceiling "
      "columns show what agreement beyond A's level would be worth. The ceiling is a theoretical bound (perfect drafts, floor = the "
      "mean passes of A's and this drafter's own blocks at ≥ 90% agreement), not an attainable target: it only shows that the "
      "scheme's passes are dominated by draft misses, not by the judge. **Expectation: the largest lever, but it costs a drafter "
      "training run, and the realistic part is the 'close the gap' / +10 points columns.**\n")
    w("### Other\n")
    w("- *Fewer drafter denoising steps.* The drafter's charge is "
      + ", ".join(f"{SHORT[a]} {_p(float(np.mean([res['arms'][a]['sets'][s]['speed']['drafter_share'] for s in res['arms'][a]['sets']])), 0)}" for a in arms)
      + " of the charged loads (mean over sets); halving its steps would cut at most half of that (upper bound above) and costs agreement by an unmeasured amount.")
    w("- *Threshold τ (0.95).* About a third to two thirds of all judge passes are forced (no slot cleared τ). A lower τ would cut them but "
      "also commit wrong tokens (judge precision in the 0.8-0.95 band is 0.86-0.92 against the stock token); it trades accuracy for speed and needs a GPU sweep, not an offline estimate.\n")
    w("## What this plan does not claim\n")
    w("- None of the estimates is a re-decode. Every counterfactual assumes the judge treats modified drafts as it treats the logged ones.")
    w("- The loads-per-pass conversion assumes the unique experts loaded per judge pass stay constant (fitted per drafter and set, R² 0.95-0.99).")
    w("- The cross-drafter slope is fitted on 7-8 (drafter, set) points; the within-drafter slope is confounded by block difficulty.")
    w("- Qwen3-0.6B results exist only for the sets listed above; CoDA IFEval has no Qwen3 counterpart yet, so the padding regression is "
      "unavailable there (reference-class bound only).")
    with open(os.path.join(AN.OUT, "IMPROVEMENT_PLAN.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def run(AN, main: dict) -> tuple:
    """compute, write figures and IMPROVEMENT_PLAN.md, return (REPORT lines, verdict lines, json-able summary)."""
    res = compute(AN)
    add_estimates(res)
    pooled = pooled_levers(res, main)
    plots(res, AN)
    verdict = final_verdict(main, res, pooled, AN)
    write_plan(res, AN, pooled, verdict)
    lines = render(res)
    slim = json.loads(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    slim["pooled_levers"] = pooled
    return lines, verdict, slim


# ---------------------------------------------------------------------------------------------------------------
# final verdict (top of REPORT.md and IMPROVEMENT_PLAN.md)
# ---------------------------------------------------------------------------------------------------------------
QUESTION = {"F2_nopool": "corpus question", "S1_nopool": "size question"}


def final_verdict(main: dict, res: dict, pooled: dict, AN) -> list:
    L = []
    w = L.append
    arms = [a for a in ARM_ORDER if a in main["pooled"]]
    all_sets = [s for s, _, _, _ in AN.SETS]
    clean_prec = {a: main["answers"][a]["indicators"]["commit_precision_clean_new"] for a in arms}
    done = {a: main["pooled"][a]["sets"] for a in arms}
    complete = all(len(done[a]) == len(all_sets) for a in arms)
    w("## Final verdict" + ("" if complete else " (provisional: some benchmarks still running)") + "\n")
    w("Frozen SDAR-30B + the Phase 1 LoRA judge (128M tokens), no expert pool, **no training anywhere in Phase 2**; only the drafter is "
      "swapped. Reference A = SDAR-1.7B drafter + the same judge. Benchmarks: " + ", ".join(all_sets) + ". "
      "*p value (significance test; < 0.05 = real difference).* *Full-pipeline speedup* = stock experts per token ÷ this method's "
      "experts per token with the drafter's own compute charged in; *all benchmarks combined* weights every item once.\n")
    w("| drafter (question) | all benchmarks combined: accuracy Δ vs A [95% CI] | p value (significance test; < 0.05 = real difference) | Δ vs stock [95% CI] | combined full-pipeline speedup: drafter vs A | experts/token vs A (break-even) |")
    w("|---|---:|---:|---:|---:|---:|")
    A = main["pooled"][arms[0]]["agg"]["A"]
    for a in arms:
        po = main["pooled"][a]
        d, ds = po["vs_A"], po["vs_stock"]
        g, gA = po["agg"]["new"], po["agg"]["A"]
        drafter_new = g["charged_exp_tok"] - g["exp_tok"]
        drafter_A = gA["charged_exp_tok"] - gA["exp_tok"]
        raise_ = g["exp_tok"] / gA["exp_tok"] - 1
        be = (drafter_A - drafter_new) / gA["exp_tok"]
        w(f"| {SHORT[a]} ({QUESTION[a]}) | {100 * d['delta']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] points | {d['mcnemar_p']:.3f} | "
          f"{100 * ds['delta']:+.1f} [{100 * ds['lo']:+.1f}, {100 * ds['hi']:+.1f}] | **{g['speedup_vs_stock_charged']:.3f}x** vs {gA['speedup_vs_stock_charged']:.3f}x | "
          f"{100 * raise_:+.1f}% (break-even {100 * be:+.1f}%) |")
    w("")
    w("| set | accuracy Δ vs A: CoDA | Qwen3 | full-pipeline speedup: SDAR-1.7B (A) | CoDA | Qwen3 | interior-block content agreement: A / CoDA / Qwen3 |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for s in all_sets:
        cells = [f"{100 * main['sets'][s]['arms'][a]['vs_A']['delta']:+.1f}" if a in main["sets"][s]["arms"] and main["sets"][s]["arms"][a].get("vs_A") else "–" for a in arms]
        sp = [main["sets"][s]["arms"][a]["agg"]["speedup_vs_stock_charged"] if a in main["sets"][s]["arms"] and main["sets"][s]["arms"][a].get("agg") else None for a in arms]
        spA = main["sets"][s]["reference"]["A"]["speedup_vs_stock_charged"] if main["sets"][s]["reference"].get("A") else None
        ag = []
        r0 = _any_r(res, s)
        ag.append(_p(r0["agree_A"]["pure"], 0) if r0 else "–")
        for a in arms:
            ag.append(_p(res["arms"][a]["sets"][s]["agree_new"]["pure"], 0) if s in res["arms"][a]["sets"] else "–")
        w(f"| {s} | " + " | ".join(cells) + f" | {_n(spA)}x | " + " | ".join(f"{_n(x)}x" if x else "–" for x in sp) + " | " + " / ".join(ag) + " |")
    w("")
    for a in arms:
        po = main["pooled"][a]
        d = po["vs_A"]
        g, gA = po["agg"]["new"], po["agg"]["A"]
        pl = pooled[a]["levers"]
        pmin = min(main["sets"][s]["arms"][a]["vs_A"]["mcnemar_p"] for s in po["sets"])
        KA = "Drafter aligned to SDAR-1.7B's agreement"
        gaps = [s for s in res["arms"][a]["sets"] if (res["arms"][a]["sets"][s]["agree_A"]["pure"] or 0) - (res["arms"][a]["sets"][s]["agree_new"]["pure"] or 0) > 0.03]
        nogap = [s for s in res["arms"][a]["sets"] if s not in gaps]
        fmt = pl["Header strip / prefill"]["central"] + pl["Padding-token remap"]["central"] + pl["Rule-fill of the post-EOS tail"]["central"]
        fmt_hi = pl["Header strip / prefill"]["hi"] + pl["Padding-token remap"]["hi"] + pl["Rule-fill of the post-EOS tail"]["hi"]
        sp_after = g["speedup_vs_stock_charged"] / (1 - pl[KA]["central"] / 100)
        w(f"### {SHORT[a]} ({QUESTION[a]})\n")
        verdict_word = "No dip" if d["lo"] <= 0 <= d["hi"] else ("Dip" if d["hi"] < 0 else "Gain")
        w(f"1. **Dip?** {verdict_word}: {100 * d['delta']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] points vs A over all benchmarks combined (p = {d['mcnemar_p']:.3f}; "
          "significance test, < 0.05 = real difference); " + (f"no single benchmark is significant (smallest p = {pmin:.2f}). " if pmin >= 0.05 else f"smallest per-benchmark p = {pmin:.3f}. ")
          + f"The interval excludes an overall loss larger than {abs(100 * min(d['lo'], 0)):.1f} points; single benchmarks have wider intervals (about ±1.4 to ±6.5 points), so small per-benchmark losses are not excluded.")
        w(f"2. **Why (what is different)?** The drafter is weaker: draft agreement {100 * g['agreement']:.1f}% vs A's {100 * gA['agreement']:.1f}%, judge passes per block {g['passes_per_block']:.2f} vs {gA['passes_per_block']:.2f}. "
          f"That is a real content gap (interior-block agreement more than 3 points below A) on {', '.join(gaps) if gaps else 'none'}"
          + (f"; none on {', '.join(nogap)}" if nogap else "")
          + ". The first-block `<think>` header" + (" and the padding token are" if a == "S1_nopool" else " is") + " a format difference worth under 0.2% of charged loads"
          + (" each" if a == "S1_nopool" else "") + ", not the cause.")
        w(f"3. **Why no dip?** The judge decides every commit: wrong drafts are confirmed at the first read only {100 * main['pooled'][a]['judge']['pass0']['commit_rate_wrong_drafts']:.2f}% of the time, "
          f"clean-state commits match the stock token {100 * clean_prec[a]:.1f}%, and the output follows the stock trajectory more often than A's does. "
          "A weaker draft therefore costs judge passes, not accuracy.")
        w(f"4. **Scope / could training help (assessment, nothing trained)?** Combined full-pipeline speedup is {g['speedup_vs_stock_charged']:.3f}x vs A's {gA['speedup_vs_stock_charged']:.3f}x. "
          f"No-training format fixes (header, padding, tail rule-fill) are est. {fmt:.1f}% of charged loads in total (best case {fmt_hi:.1f}%): not enough. "
          f"Judge retraining est. {pl['Judge retrained on this drafter (confidence)']['central']:.1f}%. Aligning the drafter to A's content agreement est. {pl[KA]['central']:.1f}% "
          f"(up to {pl[KA]['best_set_cut']:.1f}% on {pl[KA]['best_set']}), i.e. {g['speedup_vs_stock_charged']:.3f}x → about {sp_after:.3f}x. Only drafter training has real headroom (out of scope here).\n")
    cod, qw = main["pooled"].get("F2_nopool"), main["pooled"].get("S1_nopool")
    if cod and qw:
        gc, gq, gA = cod["agg"]["new"], qw["agg"]["new"], cod["agg"]["A"]
        be = (( gA["charged_exp_tok"] - gA["exp_tok"]) - (gq["charged_exp_tok"] - gq["exp_tok"])) / gA["exp_tok"]
        w("**Overall conclusion for the two Phase 2 questions.**")
        w(f"- *Corpus (CoDA-1.7B, code-heavy corpus, same base, size and tokenizer as SDAR-1.7B; note it is also a full-sequence model, so read it as a pattern across domains):* accuracy holds, "
          f"speed does not improve ({gc['speedup_vs_stock_charged']:.3f}x vs {gA['speedup_vs_stock_charged']:.3f}x): interior-block content agreement vs SDAR-1.7B in points: " + ", ".join(f"{s} {100 * ((res['arms']['F2_nopool']['sets'][s]['agree_new']['pure'] or 0) - (res['arms']['F2_nopool']['sets'][s]['agree_A']['pure'] or 0)):+.0f}" for s in res["arms"]["F2_nopool"]["sets"])
          + " (about equal on HumanEval, small gaps on MBPP and IFEval, large on the reasoning and science sets), so the corpus decides how fast, not how accurate.")
        w(f"- *Size (Qwen3-0.6B, which also changes the corpus, so read it as \"this 0.6B drafter\"):* accuracy holds and it is faster ({gq['speedup_vs_stock_charged']:.3f}x vs {gA['speedup_vs_stock_charged']:.3f}x): "
          f"its weaker drafts raise experts per token by {100 * (gq['exp_tok'] / gA['exp_tok'] - 1):.1f}%, below the {100 * be:.1f}% break-even set by its 3x cheaper drafter. In both cases the judge, not the drafter, carries accuracy; the drafter only moves speed.\n")
    return L
