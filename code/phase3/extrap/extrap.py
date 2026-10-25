"""Phase 3 -> old setup (8x RTX PRO 6000 Server, the Phase 1/2 box) extrapolation.
Usage: /venv/p2/bin/python -I extrap.py ROOT P2_METRICS_JSON OUT_PREFIX
Method
 1. Box ratio per component (30B prefill / judge / kvflush / rejudge, drafter, python overhead) from the 16 HumanEval
    items both boxes decoded with the identical config (setup check, token-identical outputs).
 2. Old-box seconds of each rd item = sum over components of (this box's ms for that component / ratio).
 3. Old-box tokens/s = sum tokens / sum predicted seconds (the artifact's definition: whole decode, scoring excluded).
    Cross-check: the artifact's own cost model (30B expert loads/token x old-box ms per load + drafter fwd x ms/fwd).
 4. Old-box wall per cell = 8 shards, items[k::8] in benchmark order, max shard sum + model load; this box: 4 shards.
 5. Cells not yet run: per-item seconds = old-box S0 seconds of the same item x the arm's measured rd/S0 time factor.
"""
import glob, json, os, re, sys, datetime as dt
ROOT, P2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [f"{ROOT}/code/compat", f"{ROOT}/code"]; os.chdir(ROOT)
import research_protocol as P
SETS = ["humaneval", "mbpp", "gsm8k500", "ifeval541", "math500", "gpqa", "aime"]
ARMS = ["rd1", "rd2"]
p2 = json.load(open(P2))
def load(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if l.strip():
                try: r = json.loads(l)
                except ValueError: continue
                out[int(r["idx"])] = r
    return out
def comp(r):
    c = {f"moe_{k}": v for k, v in r["ms_by_kind"].items()}
    c.update({f"dr_{k}": v for k, v in r["drafter_ms_by_kind"].items()})
    c["other"] = 1000 * r["seconds"] - sum(v for k, v in c.items())
    return c
# ---- 1. box ratios -------------------------------------------------------------------------------------------------
old16 = load("logs/phase2/ref/S0_128M_nopool_by_set/humaneval/*.jsonl"); new16 = load("logs/phase3/setup/check_judge/*.jsonl")
ids16 = sorted(new16); assert all(old16[i]["gen_ids"] == new16[i]["gen_ids"] for i in ids16)
def ratio(keys):
    n = sum(comp(new16[i]).get(k, 0) for i in ids16 for k in keys); o = sum(comp(old16[i]).get(k, 0) for i in ids16 for k in keys)
    return n / o
RAT = {"moe_prefill": ratio(["moe_prefill"]), "moe_judge": ratio(["moe_judge"]), "moe_kvflush": ratio(["moe_kvflush"]),
       "dr": ratio(["dr_prefill", "dr_draft", "dr_flush"]), "other": ratio(["other"])}
RAT["moe_rejudge"] = RAT["moe_judge"]; RAT["moe_denoise"] = RAT["moe_judge"]
tot_ratio = sum(new16[i]["seconds"] for i in ids16) / sum(old16[i]["seconds"] for i in ids16)
# leave-one-out check of the component model on the 16 paired items
loo = []
for j in ids16:
    rest = [i for i in ids16 if i != j]
    def rr(keys):
        return sum(comp(new16[i]).get(k, 0) for i in rest for k in keys) / sum(comp(old16[i]).get(k, 0) for i in rest for k in keys)
    R2 = {"moe_prefill": rr(["moe_prefill"]), "moe_judge": rr(["moe_judge"]), "moe_kvflush": rr(["moe_kvflush"]),
          "dr": rr(["dr_prefill", "dr_draft", "dr_flush"]), "other": rr(["other"])}
    c = comp(new16[j]); pred = sum(v / (R2["dr"] if k.startswith("dr_") else R2.get(k, R2["moe_judge"])) for k, v in c.items()) / 1000
    loo.append(pred / old16[j]["seconds"] - 1)
def old_seconds(r):
    return sum(v / (RAT["dr"] if k.startswith("dr_") else RAT.get(k, RAT["moe_judge"])) for k, v in comp(r).items()) / 1000
# ---- 2/3. per cell metrics -----------------------------------------------------------------------------------------
def order(s): return [int(it["idx"]) for it in P.items_for_split(s)]
def stats(Rs):
    g = sum(r["n_gen_to_eos"] for r in Rs); u = sum(r["unique_expert_layer_loads"] for r in Rs)
    nd = sum(sum(r["drafter_n_by_kind"].values()) for r in Rs)
    return {"n": len(Rs), "acc": 100 * sum(bool(r["correct"]) for r in Rs) / len(Rs),
            "passes_blk": sum(r["n_forward"] - r["n_prefill"] - r["n_flush"] for r in Rs) / sum(r["n_blocks"] for r in Rs),
            "exp_tok": u / g, "drafter_fwd_tok": nd / g, "tokens": g,
            "box_sec": sum(r["seconds"] for r in Rs), "old_sec": sum(old_seconds(r) for r in Rs)}
def run_start(arm, s):
    for l in open("logs/phase3/run.log"):
        m = re.match(rf"\[p3 (\d\d:\d\d:\d\d) UTC\] start {arm} {s} ", l)
        if m: t = m.group(1)
    return t if 'm' in dir() else None
# model load overhead on this box: run.log start -> (first record utc - its seconds), max over shards
def load_overhead(arm, s):
    st = None
    for l in open("logs/phase3/run.log"):
        m = re.match(rf"\[p3 (\d\d:\d\d:\d\d) UTC\] start {arm} {s} \(attempt 1\)", l)
        if m: st = m.group(1)
    if not st: return None
    ov = []
    for f in glob.glob(f"logs/phase3/decode/{arm}/{s}/*.shard*.jsonl"):
        rs = [json.loads(l) for l in open(f) if l.strip()]
        if not rs: continue
        t0 = dt.datetime.strptime(rs[0]["utc"], "%Y-%m-%dT%H:%M:%SZ")
        st_dt = t0.replace(hour=int(st[:2]), minute=int(st[3:5]), second=int(st[6:]))
        if st_dt > t0: st_dt -= dt.timedelta(days=1)          # run.log has time of day only: start was before midnight UTC
        ov.append((t0 - st_dt).total_seconds() - rs[0]["seconds"])
    return max(ov) if ov else None
def wall(sec_by_idx, s, nsh, overhead):
    o = [i for i in order(s) if i in sec_by_idx]
    return max(sum(sec_by_idx[i] for i in o[k::nsh]) for k in range(nsh)) + overhead
S0old = {s: load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl") for s in SETS}
res = {"box_ratio_components": RAT, "box_ratio_total": tot_ratio, "loo_rel_err": loo, "cells": {}}
k_arm = {}
for a in ARMS:
    num = den = 0
    for s in SETS:
        Rr = load(f"logs/phase3/decode/{a}/{s}/*.jsonl")
        if not Rr: continue
        st = stats(list(Rr.values())); ref = p2[s]["S0_nopool"]; stock = p2[s]["stock"]
        msl = (ref["ms_tok"] - ref["drafter_ms_tok"]) / ref["exp_tok"]; msf = ref["drafter_ms_tok"] / ref["drafter_fwd_tok"]
        st["tps_box"] = st["tokens"] / st["box_sec"]; st["tps_old"] = st["tokens"] / st["old_sec"]
        st["tps_old_costmodel"] = 1000 / (st["exp_tok"] * msl + st["drafter_fwd_tok"] * msf)
        st["old_sec_cost"] = st["tokens"] * (st["exp_tok"] * msl + st["drafter_fwd_tok"] * msf) / 1000
        st["f_cost"] = st["old_sec_cost"] / st["old_sec"]   # cost model time / timing model time
        st["tps_S0_old"] = 1000 / ref["ms_tok"]; st["tps_stock_table1"] = 1000 / (stock["exp_tok"] * msl)
        st["acc_S0"] = ref["acc"]; st["acc_stock"] = stock["acc"]; st["passes_S0"] = ref["passes_blk"]; st["passes_stock"] = stock["passes_blk"]
        st["exp_tok_S0"] = ref["exp_tok"]
        st["complete"] = os.path.exists(f"logs/phase3/decode/{a}/{s}/.complete")
        # paired accuracy delta vs today's judge (Phase 1 records), 95% bootstrap CI, 2000 resamples, seed 0 (as the Phase 2 page)
        import numpy as np
        cm = sorted(i for i in Rr if i in S0old[s])
        if cm:
            dd = np.array([int(bool(Rr[i]["correct"])) - int(bool(S0old[s][i]["correct"])) for i in cm])
            rng = np.random.default_rng(0); bs = [rng.choice(dd, len(dd)).mean() for _ in range(2000)]
            st["delta_vs_S0"] = [100 * dd.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]
        # SAME-BOX baseline: today's judge (SDAR-1.7B + 128M judge, no re-draft) decoded on THIS box under the same
        # 4-GPU load (full set for humaneval/mbpp, first 12 items elsewhere); paired on identical items
        # (a) full baseline on this box (humaneval, mbpp): rd vs judge on every item, both here, same 4-GPU load
        # (b) 12-item baseline (other sets): this box's slowdown for the judge on those items (same work on both boxes,
        #     validated on humaneval/mbpp: -2.4% / +0.4% vs the full answer) applied to rd over ALL its items here
        Jf = load(f"logs/phase3/decode/judge/{s}/*.jsonl")
        if Jf and not os.path.exists(f"logs/phase3/decode/judge/{s}/.complete"): Jf = {}   # a base run still in progress is not a full baseline
        Js = {} if Jf else load(f"logs/phase3/decode/judge_sample/{s}/*.jsonl")
        if Jf:
            com = [i for i in Rr if i in Jf]
            tr = sum(Rr[i]["n_gen_to_eos"] for i in com) / sum(Rr[i]["seconds"] for i in com)
            tj = sum(Jf[i]["n_gen_to_eos"] for i in com) / sum(Jf[i]["seconds"] for i in com)
            st["samebox_items"] = len(com); st["samebox_speedup_vs_judge"] = tr / tj; st["samebox_method"] = "full baseline"
            st["samebox_tps_rd"] = tr; st["samebox_tps_judge"] = tj
            st["tps_old_samebox"] = (1000 / ref["ms_tok"]) * tr / tj
        elif Js and len(glob.glob(f"logs/phase3/decode/judge_sample/{s}/*.jsonl.done")) >= 4:
            ids = [i for i in Js if i in S0old[s]]
            r12 = sum(Js[i]["seconds"] for i in ids) / sum(S0old[s][i]["seconds"] for i in ids)
            st["box_ratio_judge_12"] = r12; st["samebox_items"] = len(ids); st["samebox_method"] = "12-item box ratio"
            st["tps_old_samebox"] = st["tps_box"] * r12
            st["samebox_speedup_vs_judge"] = st["tps_old_samebox"] / (1000 / ref["ms_tok"])
        st["tps_headline"] = st.get("tps_old_samebox") or st["tps_old_costmodel"]
        st["headline_src"] = "same-box" if st.get("tps_old_samebox") else "cost model (est.)"
        common = [i for i in Rr if i in S0old[s]]
        num += st["f_cost"] * sum(old_seconds(Rr[i]) for i in common); den += sum(S0old[s][i]["seconds"] for i in common)
        ov = load_overhead(a, s); st["load_overhead_box_s"] = ov
        if st["complete"]:
            st["wall_box_4x_min"] = wall({i: r["seconds"] for i, r in Rr.items()}, s, 4, ov or 0) / 60
            st["wall_old_8x_min"] = wall({i: old_seconds(r) * st["f_cost"] for i, r in Rr.items()}, s, 8, (ov or 0)) / 60
        res["cells"][f"{a}/{s}"] = st
    k_arm[a] = num / den if den else None
res["k_arm_vs_S0_oldbox"] = k_arm
# this box's seconds per old-box (cost model) second, per arm, from finished cells
box_k = {}
for a in ARMS:
    cs = [c for k, c in res["cells"].items() if k.startswith(a + "/") and c.get("complete")]
    if cs: box_k[a] = sum(c["box_sec"] for c in cs) / sum(c["old_sec_cost"] for c in cs)
res["box_sec_per_old_cost_sec"] = box_k
# ---- 5. projection for every cell (measured where complete) ---------------------------------------------------------
ovs = [c["load_overhead_box_s"] for c in res["cells"].values() if c.get("load_overhead_box_s")]
ov = max(ovs) if ovs else 90.0
proj = {}
for a in ARMS:
    for s in SETS:
        c = res["cells"].get(f"{a}/{s}")
        if c and c.get("complete"):
            proj[f"{a}/{s}"] = {"box4": c["wall_box_4x_min"], "old8": c["wall_old_8x_min"], "src": "measured"}
        else:
            k = k_arm.get(a) or k_arm.get("rd1") or 1.0
            so = {i: r["seconds"] * k for i, r in S0old[s].items()}
            fb = box_k.get(a) or tot_ratio
            proj[f"{a}/{s}"] = {"box4": wall({i: v * fb for i, v in so.items()}, s, 4, ov) / 60,
                                "old8": wall(so, s, 8, ov) / 60, "src": f"S0 old-box item times x {k:.3f}"}
res["projection_min"] = proj
json.dump(res, open(OUT + ".json", "w"), indent=1)
# ---- report ---------------------------------------------------------------------------------------------------------
L = []
L.append(f"Box ratio (this 4x box / old 8x box, same 16 items, identical outputs): total {tot_ratio:.3f}; components " +
         ", ".join(f"{k} {v:.3f}" for k, v in RAT.items() if k not in ('moe_rejudge', 'moe_denoise')))
import statistics as stt
L.append(f"Leave-one-out per-item error of the old-box time model: mean {100*stt.mean(loo):+.1f}%, sd {100*stt.pstdev(loo):.1f}%, max |{100*max(abs(x) for x in loo):.1f}|%")
L.append(f"rd time vs S0 on the old box, same items (pooled over finished sets): " + ", ".join(f"{a} x{v:.3f}" for a, v in k_arm.items() if v))
L.append("")
L.append("SAME-BOX: rd vs today's judge on identical items, both decoded on this box under 4-GPU load")
for key, c in res["cells"].items():
    if c.get("samebox_items"):
        L.append(f"  {key:16s} items {c['samebox_items']:4d}  rd {c.get('samebox_tps_rd', float('nan')):.2f} tok/s vs judge {c.get('samebox_tps_judge', float('nan')):.2f} [{c.get('samebox_method','')}] -> {c['samebox_speedup_vs_judge']:.3f}x  => old box {c['tps_old_samebox']:.2f} tok/s ({c['tps_old_samebox']/c['tps_stock_table1']:.2f}x stock)")
L.append("")
L.append("| cell | n | acc (S0 / stock) | passes/blk (S0) | exp/tok (S0) | tok/s this box | tok/s OLD 8x box: HEADLINE [source] / cost model / single-ratio (S0, stock) | wall this box 4x | wall OLD 8x |")
L.append("|---|---|---|---|---|---|---|---|---|")
for key, c in res["cells"].items():
    p = proj[key]
    L.append(f"| {key}{'' if c['complete'] else ' (partial)'} | {c['n']} | {c['acc']:.1f} ({c['acc_S0']:.1f} / {c['acc_stock']:.1f}) | {c['passes_blk']:.2f} ({c['passes_S0']:.2f}) | "
             f"{c['exp_tok']:.0f} ({c['exp_tok_S0']:.0f}) | {c['tps_box']:.2f} | **{c['tps_headline']:.2f}** [{c['headline_src']}] / cost {c['tps_old_costmodel']:.2f} / ratio {c['tps_old']:.2f} ({c['tps_S0_old']:.1f}, {c['tps_stock_table1']:.1f}) | "
             f"{p['box4']:.1f} min | {p['old8']:.1f} min |")
t4 = sum(p["box4"] for p in proj.values()); t8 = sum(p["old8"] for p in proj.values())
L.append("")
L.append(f"Projected decode wall, all 14 cells: this box 4x {t4/60:.2f} h; OLD 8x box {t8/60:.2f} h (excl. setup/backup; measured where complete, projected elsewhere)")
for k, p in proj.items(): L.append(f"  {k:16s} 4x {p['box4']:6.1f} min   old8x {p['old8']:6.1f} min   [{p['src']}]")
open(OUT + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
