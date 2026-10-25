#!/usr/bin/env python
"""8-GPU Phase 3 re-run: every number recomputed from the raw per-item records. Nothing is extrapolated or adjusted.

    /venv/p2/bin/python code/phase3_8gpu/report_8gpu.py        (from the repository root; also run by bg_loop.sh)

Inputs
  logs/phase3_8gpu/decode/{rd1,rd2}/<bench>/*.jsonl   rd1 / rd2 on this machine (8 shards items[k::8], one per GPU)
  logs/phase3_8gpu/decode/base/<bench>/*.jsonl        base pipeline on this machine, IF it is ever run here
  logs/phase2/ref/S0_128M_nopool_by_set/<bench>/      base pipeline on the Phase 1 machine (Phase 1 records)
  logs/phase3_8gpu/events.tsv, canary/summary.json, bg_jobs.tsv, hostmon/, setup/
Outputs
  results/phase3_8gpu/final_numbers.json, results/phase3_8gpu/REPORT.md

Definitions (as results/phase3/REPORT.md section 5 / code/phase3/extrap/final_report.py):
  tok/s = sum(n_gen_to_eos) / sum(seconds); ms/token = 1000 sum(seconds) / sum(tokens); total generation time = sum(seconds)
  seconds per item = torch.cuda.synchronize() + time.perf_counter() around the whole decode (da_decode.decode_item)
  passes / block = sum(n_forward - n_prefill - n_flush) / sum(n_blocks)
  30B expert loads / token = sum(unique_expert_layer_loads) / sum(n_gen_to_eos)
  drafter forwards / token = sum(all drafter forwards) / sum(tokens)
  delta accuracy = paired rd - base on the same items, 95% CI from 2,000 bootstrap resamples, numpy default_rng(0)
  speedup (tok/s) = tok/s(rd) / tok/s(base), same items; speedup (time) = sum(seconds base) / sum(seconds rd), same items
  host index H = sum(30B prompt-prefill ms, rd) / sum(the same, base), same items (identical work in both)
Added here: experts per 30B forward = sum(unique_expert_layer_loads) / sum(n_forward); per judge pass =
  sum(uniq_by_kind.judge) / sum(n_judge); per layer and forward = sum(unique_expert_layer_loads) / sum(layer_forwards).
"""
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path[:0] = ["code/compat", "code"]
import research_protocol as P  # noqa: E402

L = "logs/phase3_8gpu"
OUT = "results/phase3_8gpu"
BENCH = [("ifeval541", "IFEval-541"), ("math500", "MATH-500"), ("gpqa", "GPQA-Diamond"), ("aime", "AIME"),
         ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500")]
ARMS = ["rd1", "rd2"]
NSH = 8


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        if ".running." in f:
            continue
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                    out[int(r["idx"])] = r
                except ValueError:
                    pass
    return out


def agg(R, ids):
    if not ids:
        return None
    tok = sum(R[i]["n_gen_to_eos"] for i in ids); sec = sum(R[i]["seconds"] for i in ids)
    loads = sum(R[i]["unique_expert_layer_loads"] for i in ids)
    nfw = sum(R[i]["n_forward"] for i in ids); nj = sum(R[i]["n_judge"] for i in ids)
    lf = sum(R[i]["layer_forwards"] for i in ids)
    uj = sum((R[i].get("uniq_by_kind") or {}).get("judge", 0) for i in ids)
    ga = [R[i].get("gpu_active_s") for i in ids]
    gpu = sum(ga) if all(g is not None for g in ga) else None       # exact GPU-active time (GPM counters), every item
    return {"gpu_active_s": gpu, "gpu_idle_s": (sec - gpu) if gpu is not None else None,
            "gpu_share_pct": (100 * gpu / sec) if gpu is not None else None,
            "gpu_tok_s": (tok / gpu) if gpu else None, "gpu_ms_tok": (1000 * gpu / tok) if gpu is not None else None,
            "n": len(ids), "correct": int(sum(bool(R[i]["correct"]) for i in ids)),
            "acc": 100 * float(np.mean([bool(R[i]["correct"]) for i in ids])),
            "passes_blk": sum(R[i]["n_forward"] - R[i]["n_prefill"] - R[i]["n_flush"] for i in ids) / sum(R[i]["n_blocks"] for i in ids),
            "loads_tok": loads / tok,
            "drafter_fwd_tok": sum(sum(R[i]["drafter_n_by_kind"].values()) for i in ids) / tok,
            "experts_per_30b_forward": loads / nfw, "experts_per_judge_pass": uj / nj if nj else None,
            "experts_per_layer_forward": loads / lf,
            "n_forward_30b": nfw, "n_judge": nj, "loads": loads,
            "tokens": tok, "seconds": sec, "tok_s": tok / sec, "ms_tok": 1000 * sec / tok}


def paired_delta(A, B, ids):
    d = np.array([int(bool(A[i]["correct"])) - int(bool(B[i]["correct"])) for i in ids])
    rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def hidx(R, B, ids, key="ms_by_kind"):
    num = sum(R[i][key]["prefill"] for i in ids); den = sum(B[i][key]["prefill"] for i in ids)
    return num / den if den else None


def ts(u):
    return dt.datetime.strptime(u, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def events():
    ev = []
    if os.path.exists(f"{L}/events.tsv"):
        for line in open(f"{L}/events.tsv"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                ev.append((float(p[0]), p[1], p[2], p[3]))
    return ev


def walls(ev):
    st, out = {}, {}
    for t, iso, kind, label in ev:
        parts = label.split()
        if kind in ("start_run", "end_run") and len(parts) >= 2:
            k = (parts[0], parts[1])
            if kind == "start_run":
                st[k] = (t, iso)
            elif k in st:
                w = out.setdefault(k, {"wall_min": 0.0, "attempts": 0})
                w["wall_min"] += (t - st[k][0]) / 60; w["attempts"] += 1
                w["start_utc"] = w.get("start_utc", st[k][1]); w["end_utc"] = iso
    return out


def bg_intervals():
    out = []
    if os.path.exists(f"{L}/bg_jobs.tsv"):
        for line in open(f"{L}/bg_jobs.tsv"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                try:
                    out.append((float(p[0]), float(p[1]), p[2]))
                except ValueError:
                    pass
    return out


def overlap_check(R, B, ids, bg):
    """Items whose decode overlapped a background job vs the others: per-item ratio of mean judge-pass time to the
    same item's Phase 1 base record (normalises for prompt and context length); also the prefill ratio."""
    if not bg:
        return {"background_jobs": 0}
    ov, other = [], []
    for i in ids:
        end = ts(R[i]["utc"]); start = end - R[i]["seconds"] - 1.0
        hit = any(a < end and b > start for a, b, _ in bg)
        rj = (R[i]["ms_by_kind"]["judge"] / R[i]["n_judge"]) / (B[i]["ms_by_kind"]["judge"] / B[i]["n_judge"])
        (ov if hit else other).append(rj)
    f = lambda v: float(np.mean(v)) if v else None
    return {"background_jobs": len(bg), "items_overlapping": len(ov), "items_other": len(other),
            "judge_pass_ratio_overlapping": f(ov), "judge_pass_ratio_other": f(other),
            "difference_pct": (100 * (f(ov) / f(other) - 1)) if ov and other else None}


def windows(R, B, ids, minutes=10):
    if not ids:
        return []
    t0 = min(ts(R[i]["utc"]) for i in ids)
    buck = {}
    for i in ids:
        buck.setdefault(int((ts(R[i]["utc"]) - t0) // (60 * minutes)), []).append(i)
    return [{"window": k, "from_min": k * minutes, "items": len(v), "H_30b_prefill": hidx(R, B, v)}
            for k, v in sorted(buck.items())]


def hostmon_delta(arm, s):
    fs = sorted(glob.glob(f"{L}/hostmon/*_before_{arm}_{s}.txt")), sorted(glob.glob(f"{L}/hostmon/*_after_{arm}_{s}.txt"))
    if not fs[0] or not fs[1]:
        return None

    def parse(f):
        d = {"gpu": {}}
        for line in open(f):
            line = line.strip()
            if line.startswith("nr_throttled") or line.startswith("throttled_usec"):
                k, v = line.split(); d[k] = int(v)
            elif line[:1].isdigit() and line.count(",") >= 6:
                c = [x.strip() for x in line.split(",")]
                try:
                    d["gpu"][int(c[0])] = {"active": c[1], "sw_power_cap_us": int(c[2].split()[0]),
                                           "sw_thermal_us": int(c[3].split()[0]), "hw_thermal_us": int(c[4].split()[0]),
                                           "hw_power_brake_us": int(c[5].split()[0])}
                except (ValueError, IndexError):
                    pass
        return d
    a, b = parse(fs[0][-1]), parse(fs[1][-1])
    out = {"cpu_nr_throttled": b.get("nr_throttled", 0) - a.get("nr_throttled", 0),
           "cpu_throttled_usec": b.get("throttled_usec", 0) - a.get("throttled_usec", 0), "gpu": {}}
    for g in sorted(set(a["gpu"]) & set(b["gpu"])):
        out["gpu"][g] = {k: (b["gpu"][g][k] - a["gpu"][g][k]) / 1e6 for k in
                         ("sw_power_cap_us", "sw_thermal_us", "hw_thermal_us", "hw_power_brake_us")}
    out["gpu_throttle_seconds_max"] = max((max(v.values()) for v in out["gpu"].values()), default=0.0)
    return out


P2_SETUPS = [("stock", "Stock 30B (no drafter)"), ("S0_nopool", "SDAR-1.7B + judge (base)"),
             ("F2_nopool", "CoDA-1.7B + judge"), ("S1_nopool", "Qwen3-0.6B + judge")]


def phase2_drafters():
    """Phase 2 drafter swap (page "Phase 2 Drafter Swap", Table 1), from its own metrics (code/phase3/extrap/p2_metrics.json):
    accuracy, passes per block and 30B expert loads per token compare directly; tokens per second are the page's figures for
    the Phase 1 machine -- SDAR-1.7B + judge measured there, every other setup computed by the page's method (30B expert
    loads per token x the Phase 1 machine's ms per expert load from the SDAR rows, plus the setup's drafter ms per token)."""
    p = "code/phase3/extrap/p2_metrics.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for s, _ in BENCH:
        if s not in d:
            continue
        S0 = d[s]["S0_nopool"]
        msl = (S0["ms_tok"] - S0["drafter_ms_tok"]) / S0["exp_tok"]
        rows = {}
        for k, lab in P2_SETUPS:
            m = d[s][k]
            tps = 1000 / S0["ms_tok"] if k == "S0_nopool" else 1000 / (m["exp_tok"] * msl + (m.get("drafter_ms_tok") or 0.0))
            rows[k] = {"label": lab, "n": m.get("n"), "acc": m["acc"], "passes_blk": m["passes_blk"], "exp_tok": m["exp_tok"],
                       "tok_s_phase1_machine": tps}
        base = rows["S0_nopool"]["tok_s_phase1_machine"]
        for k in rows:
            rows[k]["speedup_vs_base"] = rows[k]["tok_s_phase1_machine"] / base
        out[s] = {"ms_per_expert_load_phase1_machine": msl, "setups": rows}
    return out


STOCK_PREFIX = {"ifeval541": "ifeval", "math500": "math500", "gpqa": "gpqa", "aime": "aime", "humaneval": "humaneval",
                "mbpp": "mbpp", "gsm8k500": "gsm8k"}


def stock_compare(res):
    """Against the plain 30B MoE (stock, no drafter). Exact (machine-independent): accuracy (paired with the stock records),
    passes per block and 30B expert loads per token. Wall-clock speedup: CHAINED = (base over stock, the Phase 2 page's
    figure for the Phase 1 machine) x (rd over base, measured on this machine). Stock was timed on other machines only."""
    p2 = res.get("phase2_drafters") or {}
    out = {"benchmarks": {}}
    pooled = {k: 0.0 for k in ("st_tok", "st_sec_p1", "st_loads", "st_n", "st_correct",
                               "b_tok", "b_sec_p1", "b_tok_here", "b_sec_here")}
    for arm in ("rd1", "rd2"):
        for k in ("tok", "sec", "loads", "n", "correct"):
            pooled[f"{arm}_{k}"] = 0.0
    for s, name in BENCH:
        e = res["benchmarks"].get(s); q = (p2.get(s) or {}).get("setups") or {}
        if not e or not q:
            continue
        ST = load(f"logs/phase2/ref/stock_flat/{STOCK_PREFIX[s]}_B32_t95_stock.{s}.shard*.jsonl")
        if not ST:
            continue
        bh = e.get("base_here") or {}
        st_tok = sum(r["n_gen_to_eos"] for r in ST.values())
        st_tps_p1 = q["stock"]["tok_s_phase1_machine"]; b_tps_p1 = q["S0_nopool"]["tok_s_phase1_machine"]
        x = {"name": name, "stock_items": len(ST), "stock_acc": 100 * float(np.mean([bool(r["correct"]) for r in ST.values()])),
             "stock_passes_blk": q["stock"]["passes_blk"], "stock_loads_tok": q["stock"]["exp_tok"],
             "stock_tok_s_phase1_machine": st_tps_p1, "base_over_stock_phase2": b_tps_p1 / st_tps_p1, "arms": {}}
        pooled["st_tok"] += st_tok; pooled["st_sec_p1"] += st_tok / st_tps_p1; pooled["st_loads"] += q["stock"]["exp_tok"] * st_tok
        pooled["st_n"] += len(ST); pooled["st_correct"] += sum(bool(r["correct"]) for r in ST.values())
        B1 = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
        pooled["b_tok"] += sum(r["n_gen_to_eos"] for r in B1.values()); pooled["b_sec_p1"] += sum(r["seconds"] for r in B1.values())
        if bh.get("all"):
            pooled["b_tok_here"] += bh["all"]["tokens"]; pooled["b_sec_here"] += bh["all"]["seconds"]
        for arm in ("rd1", "rd2"):
            a = e["arms"].get(arm)
            if not a:
                continue
            R = load(f"{L}/decode/{arm}/{s}/*.jsonl"); m = a["rd"]; v = a.get("vs_base_here")
            ids = sorted(i for i in R if i in ST)
            y = {"acc": m["acc"], "delta_acc_vs_stock": paired_delta(R, ST, ids) if ids else None, "paired_items": len(ids),
                 "work_ratio_loads": q["stock"]["exp_tok"] / m["loads_tok"], "passes_ratio": q["stock"]["passes_blk"] / m["passes_blk"]}
            if v:
                y["rd_over_base_here"] = v["speedup_tok_s"]
                y["rd_over_stock_chained"] = x["base_over_stock_phase2"] * v["speedup_tok_s"]
                y["rd_over_stock_gpu_only_chained"] = x["base_over_stock_phase2"] * (v.get("speedup_gpu_tok_s") or float("nan"))
            x["arms"][arm] = y
            pooled[f"{arm}_tok"] += m["tokens"]; pooled[f"{arm}_sec"] += m["seconds"]; pooled[f"{arm}_loads"] += m["loads_tok"] * m["tokens"]
            pooled[f"{arm}_n"] += m["n"]; pooled[f"{arm}_correct"] += m["correct"]
        out["benchmarks"][s] = x
    P = pooled
    if P["st_tok"] and P["b_sec_here"]:
        base_over_stock = (P["b_tok"] / P["b_sec_p1"]) / (P["st_tok"] / P["st_sec_p1"])
        out["all"] = {"stock_acc": 100 * P["st_correct"] / P["st_n"], "stock_loads_tok": P["st_loads"] / P["st_tok"],
                      "base_over_stock_phase2": base_over_stock, "arms": {}}
        for arm in ("rd1", "rd2"):
            if P[f"{arm}_sec"]:
                rb = (P[f"{arm}_tok"] / P[f"{arm}_sec"]) / (P["b_tok_here"] / P["b_sec_here"])
                out["all"]["arms"][arm] = {"acc": 100 * P[f"{arm}_correct"] / P[f"{arm}_n"], "rd_over_base_here": rb,
                                           "rd_over_stock_chained": base_over_stock * rb,
                                           "work_ratio_loads": (P["st_loads"] / P["st_tok"]) / (P[f"{arm}_loads"] / P[f"{arm}_tok"])}
    # the one same-machine check: stock decoded on THIS machine in setup (13 GSM8K items), against base / rd on those items
    S = load(f"{L}/setup/check_stock/*.jsonl"); Bh = load(f"{L}/decode/base/gsm8k500/*.jsonl")
    ids = sorted(i for i in S if i in Bh)
    if ids:
        tps = lambda D: sum(D[i]["n_gen_to_eos"] for i in ids) / sum(D[i]["seconds"] for i in ids)
        chk = {"items": len(ids), "stock_tok_s": tps(S), "base_over_stock": tps(Bh) / tps(S)}
        for arm in ("rd1", "rd2"):
            Rr = load(f"{L}/decode/{arm}/gsm8k500/*.jsonl")
            if all(i in Rr for i in ids):
                chk[f"{arm}_over_stock"] = tps(Rr) / tps(S)
        out["same_machine_check_gsm8k"] = chk
    return out


def diag_cost_us():
    vals = []
    for f in glob.glob(f"{L}/setup/judge_diag_overhead/gpu*.json"):
        try:
            vals.append(json.load(open(f))["max_us"])
        except (ValueError, KeyError):
            pass
    return max(vals) if vals else None


def main():
    ev = events(); W = walls(ev); bg = bg_intervals(); tdiag = diag_cost_us()
    res = {"generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "label": "MEASURED",
           "machine": "8x NVIDIA RTX PRO 6000 Blackwell Server Edition, Intel Xeon Gold 6430 (vast.ai, gpu_frac 1.0)",
           "judge_diag_cost_us_per_pass": tdiag, "benchmarks": {}}
    res["phase2_drafters"] = phase2_drafters()
    for f, k in ((f"{L}/setup/checks.json", "setup_checks"), (f"{L}/canary/summary.json", "canary")):
        if os.path.exists(f):
            res[k] = json.load(open(f))
    # this machine vs the Phase 1 machine on identical work: the setup's 16 HumanEval base items
    C = load(f"{L}/setup/check_judge/*.jsonl"); O = load("logs/phase2/ref/S0_128M_nopool_by_set/humaneval/*.jsonl")
    ci = sorted(i for i in C if i in O)
    if ci:
        res["machine_vs_phase1"] = {"items": len(ci), "token_identical": sum(C[i]["gen_ids"] == O[i]["gen_ids"] for i in ci),
                                    "seconds_here": sum(C[i]["seconds"] for i in ci), "seconds_phase1": sum(O[i]["seconds"] for i in ci),
                                    "time_ratio": sum(C[i]["seconds"] for i in ci) / sum(O[i]["seconds"] for i in ci),
                                    "prefill_ratio": hidx(C, O, ci)}
    for s, name in BENCH:
        B1 = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
        BH = load(f"{L}/decode/base/{s}/*.jsonl")
        n_items = len(P.items_for_split(s))
        e = {"name": name, "items_total": n_items, "base_phase1_all": agg(B1, sorted(B1)), "arms": {}}
        if BH:
            e["base_here"] = {"items_done": len(BH), "complete": os.path.exists(f"{L}/decode/base/{s}/.complete"),
                              "all": agg(BH, sorted(BH)), "wall": W.get(("base", s))}
            idb = [i for i in BH if i in B1]
            e["base_here"]["identical_to_phase1"] = [sum(BH[i]["gen_ids"] == B1[i]["gen_ids"] for i in idb), len(idb)]
        for a in ARMS:
            R = load(f"{L}/decode/{a}/{s}/*.jsonl")
            if not R:
                continue
            ids = sorted(i for i in R if i in B1)
            m = agg(R, sorted(R))
            x = {"items_done": len(R), "complete": os.path.exists(f"{L}/decode/{a}/{s}/.complete"), "rd": m,
                 "wall": W.get((a, s))}
            order = [int(it["idx"]) for it in P.items_for_split(s)]
            x["slowest_shard_seconds"] = max(sum(R[i]["seconds"] for i in order[k::NSH] if i in R) for k in range(NSH))
            if ids:
                bp, rp = agg(B1, ids), agg(R, ids)
                half = len(ids) // 2
                x["vs_phase1_base"] = {
                    "paired_items": len(ids), "base": bp, "rd": rp, "delta_acc": paired_delta(R, B1, ids),
                    "speedup_tok_s": rp["tok_s"] / bp["tok_s"], "speedup_time": bp["seconds"] / rp["seconds"],
                    "H_30b_prefill": hidx(R, B1, ids), "H_drafter_prefill": hidx(R, B1, ids, "drafter_ms_by_kind"),
                    "H_first_half": hidx(R, B1, ids[:half]) if half else None, "H_second_half": hidx(R, B1, ids[half:]),
                    "H_10min_windows": windows(R, B1, ids), "background_overlap": overlap_check(R, B1, ids, bg)}
            if BH:
                hid = sorted(i for i in R if i in BH)
                if hid:
                    bh, rh = agg(BH, hid), agg(R, hid)
                    x["vs_base_here"] = {"paired_items": len(hid), "base": bh, "rd": rh, "delta_acc": paired_delta(R, BH, hid),
                                         "speedup_tok_s": rh["tok_s"] / bh["tok_s"], "speedup_time": bh["seconds"] / rh["seconds"],
                                         "speedup_gpu_time": (bh["gpu_active_s"] / rh["gpu_active_s"])
                                         if bh["gpu_active_s"] and rh["gpu_active_s"] else None,
                                         "speedup_gpu_tok_s": (rh["gpu_tok_s"] / bh["gpu_tok_s"])
                                         if bh["gpu_tok_s"] and rh["gpu_tok_s"] else None,
                                         "H_30b_prefill": hidx(R, BH, hid)}
            if tdiag and m:
                x["judge_diag_cost_share_pct"] = 100 * m["n_judge"] * tdiag / 1e6 / m["seconds"]
            x["host_during_run"] = hostmon_delta(a, s)
            e["arms"][a] = x
        res["benchmarks"][s] = e
    try:
        res["vs_stock"] = stock_compare(res)
    except Exception as ex:                          # noqa: BLE001 (never lose the report over this section)
        res["vs_stock"] = {"error": f"{type(ex).__name__}: {ex}"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/final_numbers.json", "w"), indent=1)
    write_md(res)
    print(f"[report {res['generated_utc']}] wrote {OUT}/final_numbers.json and REPORT.md")


def f(v, p=2, suf=""):
    return "·" if v is None else f"{v:.{p}f}{suf}"


def sd(v, p=1):
    return "·" if v is None else ("+" if v > 0 else ("−" if v < 0 else "")) + f"{abs(v):.{p}f}"


def write_md(res):
    A = []
    a = A.append
    a("# Phase 3 re-draft on the 8-GPU machine: MEASURED")
    a("")
    a(f"Generated {res['generated_utc']} from the raw per-item records. Every number is MEASURED; nothing is extrapolated or adjusted.")
    a("")
    a(f"Machine: {res['machine']}. rd1 / rd2: `--drafter d32 --draft s8 --judge full --adapter <128M> --judge-diag "
      "--full-redraft-every 1|2`, block 32, threshold 0.95, no expert pool, batch 1, 8 shards (one process per GPU).")
    a("")
    mv = res.get("machine_vs_phase1")
    if mv:
        a(f"**This machine vs the Phase 1 machine on identical work** (setup check, base pipeline, {mv['items']} HumanEval items, "
          f"{mv['token_identical']}/{mv['items']} token-identical): {mv['seconds_here']:.1f} s here vs {mv['seconds_phase1']:.1f} s "
          f"in the Phase 1 records = **{mv['time_ratio']:.3f}x the time** (prompt prefill {mv['prefill_ratio']:.3f}x). "
          "Speedups against Phase 1 base records therefore compare two machines of different speed; they are listed "
          "only as raw ratios, not as the method's speedup.")
        a("")
    if res.get("judge_diag_cost_us_per_pass"):
        a(f"`--judge-diag` cost, measured: {res['judge_diag_cost_us_per_pass']:.1f} us per judge pass (max over 8 GPUs); "
          "its share of each run's decode time is in the table.")
        a("")
    a("## Table 1: rd1 and rd2, every item decoded on this machine")
    a("")
    a("| Benchmark | Setup | Items | Accuracy % | Δ acc vs Phase 1 base, pts [95% CI] | Passes / block | 30B expert loads / token | "
      "Experts / 30B forward | Experts / judge pass | Drafter fwd / token | Tokens / s | ms / token | Total generation time | Wall (8 GPUs) | judge-diag share |")
    a("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, e in res["benchmarks"].items():
        b = e["base_phase1_all"]
        if b:
            a(f"| {e['name']} | base, Phase 1 records | {b['n']} | {f(b['acc'], 1)} | reference | {f(b['passes_blk'])} | {f(b['loads_tok'], 0)} | "
              f"{f(b['experts_per_30b_forward'], 0)} | {f(b['experts_per_judge_pass'], 0)} | {f(b['drafter_fwd_tok'])} | "
              f"{f(b['tok_s'])} (Phase 1 machine) | {f(b['ms_tok'], 1)} | {f(b['seconds'] / 3600, 2)} h | · | · |")
        for arm, x in e["arms"].items():
            m = x["rd"]; v = x.get("vs_phase1_base") or {}
            dlt = v.get("delta_acc")
            w = x.get("wall") or {}
            state = "" if x["complete"] else f" (running: {x['items_done']}/{e['items_total']})"
            a(f"| {e['name']} | {arm}{state} | {m['n']} | {f(m['acc'], 1)} | "
              f"{(sd(dlt[0]) + ' [' + sd(dlt[1]) + ', ' + sd(dlt[2]) + ']') if dlt else '·'} | {f(m['passes_blk'])} | {f(m['loads_tok'], 0)} | "
              f"{f(m['experts_per_30b_forward'], 0)} | {f(m['experts_per_judge_pass'], 0)} | {f(m['drafter_fwd_tok'])} | {f(m['tok_s'])} | "
              f"{f(m['ms_tok'], 1)} | {f(m['seconds'] / 3600, 2)} h | {f(w.get('wall_min'), 1, ' min')} | {f(x.get('judge_diag_cost_share_pct'), 3, '%')} |")
    a("")
    a("## Table 2: raw ratios against the base pipeline (same items)")
    a("")
    a("| Benchmark | Setup | Paired items | Actual speedup, tok/s | Actual speedup, total time | GPU-only speedup, tok/s | GPU-only speedup, GPU time | H (30B prefill) | H drafter prefill | H 1st half | H 2nd half | Base reference |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s, e in res["benchmarks"].items():
        for arm, x in e["arms"].items():
            for key, lab in (("vs_base_here", "base on THIS machine (same-machine speedup)"),
                             ("vs_phase1_base", "Phase 1 records (OTHER machine: raw cross-machine ratio)")):
                v = x.get(key)
                if v:
                    a(f"| {e['name']} | {arm} | {v['paired_items']} | {f(v['speedup_tok_s'], 3)}x | {f(v['speedup_time'], 3)}x | "
                      f"{f(v.get('speedup_gpu_tok_s'), 3, 'x')} | {f(v.get('speedup_gpu_time'), 3, 'x')} | "
                      f"{f(v.get('H_30b_prefill'), 3)} | {f(v.get('H_drafter_prefill'), 3)} | {f(v.get('H_first_half'), 3)} | "
                      f"{f(v.get('H_second_half'), 3)} | {lab} |")
    a("")
    p2 = res.get("phase2_drafters") or {}
    if p2:
        a("## Re-drafting vs swapping the drafter (Phase 2)")
        a("")
        a("rd1 and rd2 keep the base drafter (SDAR-1.7B-Chat-b32) and add re-drafting; Phase 2 swapped the drafter instead. "
          "Speedups are over the SDAR-1.7B + judge base: rd1 / rd2 measured on this machine against the base run here; "
          "CoDA / Qwen3 from the Phase 2 page (Phase 1 machine; computed by that page's method from expert loads). "
          "Accuracy, passes per block and expert loads per token do not depend on the machine.")
        a("")
        a("| Benchmark | CoDA-1.7B speedup | Qwen3-0.6B speedup | rd1 speedup (actual / GPU-only) | rd2 speedup (actual / GPU-only) | Expert loads / token: base / CoDA / Qwen3 / rd1 / rd2 | Accuracy %: base / CoDA / Qwen3 / rd1 / rd2 |")
        a("|---|---:|---:|---:|---:|---|---|")
        for s, e in res["benchmarks"].items():
            q = (p2.get(s) or {}).get("setups") or {}
            if not q:
                continue
            sp = []
            for arm in ("rd1", "rd2"):
                v = (e["arms"].get(arm) or {}).get("vs_base_here")
                sp.append(f"{f(v['speedup_tok_s'], 3)}x / {f(v.get('speedup_gpu_tok_s'), 3)}x" if v else "after the base run")
            r1 = (e["arms"].get("rd1") or {}).get("rd") or {}; r2 = (e["arms"].get("rd2") or {}).get("rd") or {}
            a(f"| {e['name']} | {f(q['F2_nopool']['speedup_vs_base'], 3)}x | {f(q['S1_nopool']['speedup_vs_base'], 3)}x | {sp[0]} | {sp[1]} | "
              f"{f(q['S0_nopool']['exp_tok'], 0)} / {f(q['F2_nopool']['exp_tok'], 0)} / {f(q['S1_nopool']['exp_tok'], 0)} / {f(r1.get('loads_tok'), 0)} / {f(r2.get('loads_tok'), 0)} | "
              f"{f(q['S0_nopool']['acc'], 1)} / {f(q['F2_nopool']['acc'], 1)} / {f(q['S1_nopool']['acc'], 1)} / {f(r1.get('acc'), 1)} / {f(r2.get('acc'), 1)} |")
        a("")
    vs = res.get("vs_stock") or {}
    if vs.get("benchmarks"):
        a("## Against the plain 30B MoE (stock)")
        a("")
        a("Exact (machine-independent): accuracy paired with the stock records, and 30B expert loads per token (stock / setup = times less work). "
          "Wall-clock speedup over stock is CHAINED: base over stock from the Phase 2 page (Phase 1 machine; stock timed there on IFEval only, computed "
          "from expert loads elsewhere) x rd over base measured on this machine. It is an estimate, not a single-machine measurement.")
        a("")
        a("| Benchmark | Stock acc | rd1 acc (Δ vs stock [95% CI]) | rd2 acc (Δ vs stock [95% CI]) | Less 30B work than stock, rd1 / rd2 | Base over stock (Phase 2) | rd1 over stock (chained) | rd2 over stock (chained) |")
        a("|---|---:|---|---|---:|---:|---:|---:|")
        for s_, x in vs["benchmarks"].items():
            r1, r2 = x["arms"].get("rd1", {}), x["arms"].get("rd2", {})
            dd = lambda y: (f"{f(y['acc'], 1)} ({sd(y['delta_acc_vs_stock'][0])} [{sd(y['delta_acc_vs_stock'][1])}, {sd(y['delta_acc_vs_stock'][2])}])"
                            if y.get("delta_acc_vs_stock") else "·")
            a(f"| {x['name']} | {f(x['stock_acc'], 1)} | {dd(r1)} | {dd(r2)} | {f(r1.get('work_ratio_loads'), 2)}x / {f(r2.get('work_ratio_loads'), 2)}x | "
              f"{f(x['base_over_stock_phase2'], 3)}x | {f(r1.get('rd_over_stock_chained'), 3)}x | {f(r2.get('rd_over_stock_chained'), 3)}x |")
        al = vs.get("all") or {}
        if al:
            r1, r2 = al["arms"].get("rd1", {}), al["arms"].get("rd2", {})
            a(f"| All 7 (pooled) | {f(al['stock_acc'], 1)} | {f(r1.get('acc'), 1)} | {f(r2.get('acc'), 1)} | {f(r1.get('work_ratio_loads'), 2)}x / {f(r2.get('work_ratio_loads'), 2)}x | "
              f"{f(al['base_over_stock_phase2'], 3)}x | {f(r1.get('rd_over_stock_chained'), 3)}x | {f(r2.get('rd_over_stock_chained'), 3)}x |")
        ck = vs.get("same_machine_check_gsm8k")
        if ck:
            a("")
            a(f"Same-machine check: stock was decoded on this machine for {ck['items']} GSM8K items during setup; on those items base is "
              f"{f(ck['base_over_stock'], 3)}x, rd1 {f(ck.get('rd1_over_stock'), 3)}x and rd2 {f(ck.get('rd2_over_stock'), 3)}x faster than stock (measured).")
        a("")
    cn = res.get("canary")
    if cn:
        a("## Canary (HumanEval item 0, base settings, all 8 GPUs at once)")
        a("")
        a(f"Phase 1 record of the same item: {f(cn.get('phase1_record_seconds'), 3)} s.")
        a("")
        a("| Canary | Last record (UTC) | Mean s (8 GPUs) | Min | Max | vs first canary | vs Phase 1 record | Identical to Phase 1 |")
        a("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in cn["canaries"]:
            a(f"| {r['canary']} | {r['utc_last']} | {f(r['mean_s'], 3)} | {f(r['min_s'], 3)} | {f(r['max_s'], 3)} | "
              f"{f(r['vs_first'], 4)} | {f(r.get('vs_phase1_record'), 4)} | {r['identical_to_phase1']}/{r['gpus']} |")
        a("")
    a("## Host during each run, and the background-job check")
    a("")
    a("| Benchmark | Setup | CPU throttled periods | Max GPU throttle time (s) | Items overlapping a backup / page job | Judge-pass time, overlapping vs other |")
    a("|---|---|---:|---:|---:|---:|")
    for s, e in res["benchmarks"].items():
        for arm, x in e["arms"].items():
            h = x.get("host_during_run") or {}
            o = (x.get("vs_phase1_base") or {}).get("background_overlap") or {}
            a(f"| {e['name']} | {arm} | {h.get('cpu_nr_throttled', '·')} | {f(h.get('gpu_throttle_seconds_max'), 1)} | "
              f"{o.get('items_overlapping', '·')} of {(o.get('items_overlapping') or 0) + (o.get('items_other') or 0)} | "
              f"{sd(o.get('difference_pct'), 2) + '%' if o.get('difference_pct') is not None else '·'} |")
    a("")
    open(f"{OUT}/REPORT.md", "w").write("\n".join(A) + "\n")


if __name__ == "__main__":
    main()
