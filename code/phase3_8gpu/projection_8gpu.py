"""Forward-level cost model for the 8-GPU Phase 3 re-run: same-machine speedup, a frozen prediction of the base pipeline
on this machine (tested against the measured base run), and a projection of rd1 / rd2 onto the Phase 1 machine.

Why: Phase 1's base records come from a machine that runs identical work 2.17x faster, so rd (here) / base (Phase 1)
is not the method's speedup. Every record already carries the time of every 30B forward ([kind, ms, unique expert-layer
loads, layers, tokens, pool]) and the drafter time by kind, so each machine's cost can be modelled per forward:

    30B forward ms  = a[kind] + b[kind] * unique expert-layer loads        (judge, kvflush, rejudge, denoise; least squares)
    30B prefill     = the measured prefill of the same item on that machine (identical work in every arm)
    drafter         = drafter forwards by kind * that machine's mean ms per forward of that kind
    other host time = that machine's (seconds - moe_seconds - drafter_seconds) per 30B forward
    GPU-active s    = g0 * 30B forwards + g1 * unique expert-layer loads + g2 * drafter forwards   (this machine only)

The base pipeline's outputs are token-identical across machines (setup check, 16/16), so its forward list (kinds and
expert loads) in the Phase 1 records is exactly what base executes here. Hence:
  PREDICTED base here      = this machine's model applied to the Phase 1 base forward lists (fit on rd only, out of sample)
  PROJECTED rd on Phase 1  = the Phase 1 model applied to the rd forward lists measured here
The prediction is frozen (results/phase3_8gpu/projection_frozen.json) the first time it is computed and never rewritten,
so the measured base run tests it. MEASURED, PREDICTED and PROJECTED numbers are labelled as such everywhere.

Run (cheap, niced): nice -n 19 /venv/p2/bin/python code/phase3_8gpu/projection_8gpu.py
Writes results/phase3_8gpu/PROJECTION.md and results/phase3_8gpu/projection.json.
"""
import datetime as dt
import glob
import json
import os

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
L = "logs/phase3_8gpu"
P1 = "logs/phase2/ref/S0_128M_nopool_by_set"
OUT = "results/phase3_8gpu"
FROZEN = f"{OUT}/projection_frozen.json"
SETS = [("ifeval541", "IFEval-541"), ("math500", "MATH-500"), ("gpqa", "GPQA-Diamond"), ("aime", "AIME"),
        ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500")]
ARMS = ["rd1", "rd2"]
KC = {0: "prefill", 1: "denoise", 2: "kvflush", 3: "judge", 4: "rejudge"}
DK = ("prefill", "draft", "redraft", "flush")


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[int(r["idx"])] = r
    return out


def fit_machine(recs):
    """recs: list of record dicts from one machine. Returns the cost model and its in-sample fit quality."""
    m = {"kinds": {}}
    for kc, kn in KC.items():
        if kn == "prefill":
            continue
        x, y = [], []
        for r in recs:
            for fw in r["forwards"]:
                if fw[0] == kc:
                    x.append(fw[2]); y.append(fw[1])
        if len(y) < 20:
            continue
        x, y = np.array(x, float), np.array(y, float)
        X = np.stack([np.ones_like(x), x], 1)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        m["kinds"][kn] = {"a_ms": float(c[0]), "b_us_per_load": float(c[1] * 1000), "r2": float(r2), "n": int(len(y)),
                          "mean_ms": float(y.mean())}
    dn = {k: sum(r["drafter_n_by_kind"].get(k, 0) for r in recs) for k in DK}
    dm = {k: sum(r["drafter_ms_by_kind"].get(k, 0.0) for r in recs) for k in DK}
    m["drafter_ms"] = {k: (dm[k] / dn[k] if dn[k] else None) for k in DK}
    if m["drafter_ms"]["redraft"] is None:       # Phase 1 base never re-drafts; a re-draft is one drafter forward
        m["drafter_ms"]["redraft"] = m["drafter_ms"]["draft"]
    other = sum(r["seconds"] - r["moe_seconds"] - r["drafter_seconds"] for r in recs)
    m["other_s_per_fwd"] = other / sum(r["n_forward"] for r in recs)
    ga = [r for r in recs if r.get("gpu_active_s") is not None]
    if len(ga) >= 20:
        X = np.array([[r["n_forward"], r["unique_expert_layer_loads"], sum(r["drafter_n_by_kind"].values())] for r in ga], float)
        y = np.array([r["gpu_active_s"] for r in ga])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        m["gpu"] = {"s_per_fwd": float(c[0]), "s_per_load": float(c[1]), "s_per_drafter_fwd": float(c[2]),
                    "r2": float(1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "n": len(ga)}
    return m


def predict_item(m, r, prefill_ms):
    """Seconds of item r (its forward list and drafter counts) on the machine of model m; prefill_ms from that machine."""
    ms = prefill_ms
    for fw in r["forwards"]:
        kn = KC[fw[0]]
        if kn == "prefill":
            continue
        k = m["kinds"].get(kn) or m["kinds"]["judge"]
        ms += k["a_ms"] + k["b_us_per_load"] / 1000 * fw[2]
    ms += sum(r["drafter_n_by_kind"].get(k, 0) * (m["drafter_ms"][k] or 0.0) for k in DK)
    return ms / 1000 + m["other_s_per_fwd"] * r["n_forward"]


def predict_gpu(m, r):
    g = m.get("gpu")
    if not g:
        return None
    return (g["s_per_fwd"] * r["n_forward"] + g["s_per_load"] * r["unique_expert_layer_loads"]
            + g["s_per_drafter_fwd"] * sum(r["drafter_n_by_kind"].values()))


def tot(R, ids, key="seconds"):
    v = [R[i].get(key) for i in ids]
    return None if any(x is None for x in v) else float(sum(v))


def toks(R, ids):
    return float(sum(R[i]["n_gen_to_eos"] for i in ids))


def f(x, d=2, suf=""):
    return "·" if x is None else f"{x:.{d}f}{suf}"


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    D = {}
    for s, _ in SETS:
        D[s] = {"p1": load(f"{P1}/{s}/*.jsonl"), "base": load(f"{L}/decode/base/{s}/*.jsonl"),
                "complete_base": os.path.exists(f"{L}/decode/base/{s}/.complete")}
        for a in ARMS:
            D[s][a] = load(f"{L}/decode/{a}/{s}/*.jsonl")
    # one model per machine, fit on every benchmark: Phase 1 on its base records, this machine on rd1 + rd2 ONLY
    m_p1 = fit_machine([r for s, _ in SETS for r in D[s]["p1"].values()])
    m_here = fit_machine([r for s, _ in SETS for a in ARMS for r in D[s][a].values()])
    m_here_with_base = None
    if any(D[s]["base"] for s, _ in SETS):     # for reference only; the frozen prediction never uses base here
        m_here_with_base = fit_machine([r for s, _ in SETS for a in ARMS + ["base"] for r in D[s][a].values()])

    frozen = json.load(open(FROZEN)) if os.path.exists(FROZEN) else None
    res = {"generated_utc": now, "model_phase1": m_p1, "model_here_rd_only": m_here,
           "model_here_with_base": m_here_with_base, "sets": {}}
    new_frozen = {"frozen_utc": now, "note": "this machine's model fit on rd1 + rd2 records only, applied to the Phase 1 "
                  "base forward lists; written once, before the base pipeline ran here", "sets": {}}
    for s, name in SETS:
        d = D[s]; P = d["p1"]; e = {"name": name, "arms": {}}
        ids = sorted(P)
        # in-sample check on Phase 1 itself
        e["p1_measured_s"] = tot(P, ids)
        e["p1_model_s"] = sum(predict_item(m_p1, P[i], P[i]["ms_by_kind"]["prefill"]) for i in ids)
        # this machine's prefill of each item (identical in every arm): from any rd record of it
        pre_here = {}
        for a in ARMS + ["base"]:
            for i, r in d[a].items():
                pre_here.setdefault(i, r["ms_by_kind"]["prefill"])
        cov = [i for i in ids if i in pre_here]
        e["pred_base_here_items"] = len(cov)
        if cov:
            ps = sum(predict_item(m_here, P[i], pre_here[i]) for i in cov)
            pg = sum(predict_gpu(m_here, P[i]) for i in cov) if m_here.get("gpu") else None
            e["pred_base_here_s"], e["pred_base_here_gpu_s"] = ps, pg
            e["pred_base_here_tok_s"] = toks(P, cov) / ps
            new_frozen["sets"][s] = {"items": len(cov), "pred_base_here_s": ps, "pred_base_here_gpu_s": pg,
                                     "pred_base_here_tok_s": toks(P, cov) / ps}
        fz = (frozen or {}).get("sets", {}).get(s)
        e["frozen"] = fz
        B = d["base"]
        if B:
            bids = sorted(B)
            e["base_here"] = {"items": len(bids), "complete": d["complete_base"], "measured_s": tot(B, bids),
                              "measured_gpu_s": tot(B, bids, "gpu_active_s"), "tok_s": toks(B, bids) / tot(B, bids),
                              "pred_s_same_items": sum(predict_item(m_here, P[i], pre_here[i]) for i in bids
                                                       if i in P and i in pre_here),
                              "identical_tokens": sum(B[i]["gen_ids"] == P[i]["gen_ids"] for i in bids if i in P)}
            if m_here.get("gpu"):
                e["base_here"]["pred_gpu_s_same_items"] = sum(predict_gpu(m_here, P[i]) for i in bids if i in P)
        for a in ARMS:
            R = d[a]
            if not R:
                continue
            rids = sorted(i for i in R if i in P)
            x = {"items": len(rids), "rd_here_s": tot(R, rids), "rd_here_gpu_s": tot(R, rids, "gpu_active_s"),
                 "rd_here_tok_s": toks(R, rids) / tot(R, rids)}
            x["rd_here_model_s"] = sum(predict_item(m_here, R[i], R[i]["ms_by_kind"]["prefill"]) for i in rids)
            # projection onto the Phase 1 machine: the Phase 1 model on the rd forward lists, Phase 1's prefill per item
            x["proj_rd_p1_s"] = sum(predict_item(m_p1, R[i], P[i]["ms_by_kind"]["prefill"]) for i in rids)
            x["p1_base_s"] = tot(P, rids)
            x["proj_speedup_p1_time"] = x["p1_base_s"] / x["proj_rd_p1_s"]
            x["proj_speedup_p1_tok_s"] = (toks(R, rids) / x["proj_rd_p1_s"]) / (toks(P, rids) / x["p1_base_s"])
            # same machine, base predicted from the forward model (fit on rd only)
            pc = [i for i in rids if i in pre_here]
            pb = sum(predict_item(m_here, P[i], pre_here[i]) for i in pc)
            x["pred_speedup_here_time"] = pb / tot(R, pc)
            x["pred_speedup_here_tok_s"] = (toks(R, pc) / tot(R, pc)) / (toks(P, pc) / pb)
            if m_here.get("gpu") and x["rd_here_gpu_s"]:
                pg = sum(predict_gpu(m_here, P[i]) for i in pc)
                x["pred_speedup_here_gpu_time"] = pg / tot(R, pc, "gpu_active_s")
                x["pred_speedup_here_gpu_tok_s"] = (toks(R, pc) / tot(R, pc, "gpu_active_s")) / (toks(P, pc) / pg)
            # same machine, MEASURED (paired items)
            if B:
                hid = sorted(i for i in rids if i in B)
                if hid:
                    x["meas_paired"] = len(hid)
                    x["meas_speedup_here_time"] = tot(B, hid) / tot(R, hid)
                    x["meas_speedup_here_tok_s"] = (toks(R, hid) / tot(R, hid)) / (toks(B, hid) / tot(B, hid))
                    bg, rg = tot(B, hid, "gpu_active_s"), tot(R, hid, "gpu_active_s")
                    if bg and rg:
                        x["meas_speedup_here_gpu_time"] = bg / rg
                        x["meas_speedup_here_gpu_tok_s"] = (toks(R, hid) / rg) / (toks(B, hid) / bg)
            e["arms"][a] = x
        res["sets"][s] = e
    if frozen is None and not any(D[s]["base"] for s, _ in SETS) and all(D[s]["rd1"] and D[s]["rd2"] for s, _ in SETS):
        json.dump(new_frozen, open(FROZEN, "w"), indent=1)
        frozen = new_frozen
    res["frozen"] = frozen
    json.dump(res, open(f"{OUT}/projection.json", "w"), indent=1)
    write_md(res)


def write_md(res):
    o = []; a = o.append
    mp, mh = res["model_phase1"], res["model_here_rd_only"]
    a("# Phase 3 8-GPU: forward-level cost model, same-machine speedup and Phase 1 projection")
    a("")
    a(f"Generated {res['generated_utc']} by code/phase3_8gpu/projection_8gpu.py from the raw per-item records. "
      "**MEASURED** = timed here; **PREDICTED** = this machine's cost model (fit on rd1 + rd2 only) applied to the base "
      "pipeline's forward lists, frozen before base ran here; **PROJECTED** = the Phase 1 machine's cost model applied to "
      "the rd forward lists (rd was never timed on that machine).")
    a("")
    a("## 1. Cost model per machine (one 30B forward = a + b x unique expert-layer loads)")
    a("")
    a("| Machine | Forward kind | Forwards | Mean ms | a (ms) | b (us / expert-layer load) | R² |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for lab, m in (("Phase 1 (base records)", mp), ("this machine (rd1 + rd2)", mh)):
        for kn, k in m["kinds"].items():
            a(f"| {lab} | {kn} | {k['n']} | {k['mean_ms']:.1f} | {k['a_ms']:.1f} | {k['b_us_per_load']:.2f} | {k['r2']:.4f} |")
    a("")
    dr = lambda m: ", ".join(f"{k} {v:.2f}" for k, v in m["drafter_ms"].items() if v)
    a(f"Drafter ms per forward: Phase 1 {dr(mp)}; here {dr(mh)}. Other host time per 30B forward: Phase 1 "
      f"{1000 * mp['other_s_per_fwd']:.2f} ms, here {1000 * mh['other_s_per_fwd']:.2f} ms.")
    if mh.get("gpu"):
        g = mh["gpu"]
        a(f"GPU-active time here: {1000 * g['s_per_fwd']:.2f} ms per 30B forward + {1e6 * g['s_per_load']:.2f} us per "
          f"expert-layer load + {1000 * g['s_per_drafter_fwd']:.2f} ms per drafter forward (R² {g['r2']:.4f}, {g['n']} items).")
    if "judge" in mp["kinds"] and "judge" in mh["kinds"]:
        a(f"Judge-pass cost ratio here / Phase 1: a {mh['kinds']['judge']['a_ms'] / mp['kinds']['judge']['a_ms']:.2f}x, "
          f"b {mh['kinds']['judge']['b_us_per_load'] / mp['kinds']['judge']['b_us_per_load']:.2f}x: the per-expert "
          "cost dominates, as expected for a CPU-launch-bound batch-1 MoE.")
    a("")
    a("## 2. Speedup of rd over base")
    a("")
    a("| Benchmark | Arm | Items | rd tok/s here (MEASURED) | Speedup here, time: MEASURED | PREDICTED (frozen model) | "
      "GPU-only speedup, GPU time: MEASURED | PREDICTED | PROJECTED speedup on the Phase 1 machine, time | "
      "PROJECTED rd tok/s on Phase 1 | Phase 1 base tok/s (MEASURED) |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, e in res["sets"].items():
        for arm, x in e["arms"].items():
            p1_tok = None; pr_tok = None
            if x.get("p1_base_s") and x.get("proj_rd_p1_s"):
                pr_tok = x["rd_here_tok_s"] * x["rd_here_s"] / x["proj_rd_p1_s"]
                p1_tok = pr_tok / x["proj_speedup_p1_tok_s"]
            a(f"| {e['name']} | {arm} | {x['items']} | {f(x['rd_here_tok_s'])} | {f(x.get('meas_speedup_here_time'), 3, 'x')} | "
              f"{f(x.get('pred_speedup_here_time'), 3, 'x')} | {f(x.get('meas_speedup_here_gpu_time'), 3, 'x')} | "
              f"{f(x.get('pred_speedup_here_gpu_time'), 3, 'x')} | {f(x.get('proj_speedup_p1_time'), 3, 'x')} | "
              f"{f(pr_tok)} | {f(p1_tok)} |")
    a("")
    a("Speedup (time) = base seconds / rd seconds on the same items; GPU-only = the same with GPU-active seconds "
      "(NVML GPM). The PROJECTED column uses the Phase 1 base time MEASURED there, so its only modelled part is the rd time.")
    a("")
    a("## 3. Model checks")
    a("")
    a("| Benchmark | Phase 1 base: model / measured | rd1 here: model / measured | Base here: PREDICTED (frozen) / MEASURED | "
      "Base here GPU time: PREDICTED / MEASURED | Base here items done | Base tokens identical to Phase 1 |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for s, e in res["sets"].items():
        r1 = e["arms"].get("rd1") or {}
        bh = e.get("base_here") or {}
        fz = e.get("frozen") or {}
        bp = (bh.get("pred_s_same_items") / bh["measured_s"]) if bh.get("measured_s") else None
        if fz and bh.get("complete") and fz.get("items") == bh.get("items"):
            bp = fz["pred_base_here_s"] / bh["measured_s"]
        gp = (bh.get("pred_gpu_s_same_items") / bh["measured_gpu_s"]) if bh.get("measured_gpu_s") and bh.get("pred_gpu_s_same_items") else None
        a(f"| {e['name']} | {f(e['p1_model_s'] / e['p1_measured_s'], 3)} | "
          f"{f(r1['rd_here_model_s'] / r1['rd_here_s'], 3) if r1 else '·'} | {f(bp, 3)} | {f(gp, 3)} | "
          f"{bh.get('items', 0)}{' (complete)' if bh.get('complete') else ''} | "
          f"{bh.get('identical_tokens', '·')}{(' / ' + str(bh['items'])) if bh else ''} |")
    a("")
    fz = res.get("frozen")
    if fz:
        a(f"Frozen prediction written {fz['frozen_utc']} (before any base item ran here): "
          + "; ".join(f"{s} {v['pred_base_here_s'] / 3600:.2f} h / {v['pred_base_here_tok_s']:.2f} tok/s"
                      for s, v in fz["sets"].items()) + ".")
    open(f"{OUT}/PROJECTION.md", "w").write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
