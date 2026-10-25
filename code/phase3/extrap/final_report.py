"""Phase 3 final report: every number recomputed from the raw per-item decode records.
Run from the repository root:  /venv/p2/bin/python -I logs/phase3/extrap/final_report.py
Writes results/phase3/REPORT.md and results/phase3/final_numbers.json.

Inputs (all in the repo / the HF dataset Arushhh/phase3-redraft):
  logs/phase3/decode/{rd1,rd2}/<bench>/*.jsonl       rd1 / rd2, this box (4x RTX PRO 6000 Server, 4 shards)
  logs/phase3/decode/judge/<bench>/*.jsonl           base pipeline (SDAR-1.7B drafter + trained 128M judge), this box
  logs/phase3/decode/judge_sample/<bench>/*.jsonl    base pipeline, first 12 items only, this box
  logs/phase2/ref/S0_128M_nopool_by_set/<bench>/     base pipeline, Phase 1 box (8x RTX PRO 6000 Server), Phase 1 records
  logs/phase3/extrap/p2_metrics.json                 Phase 2 page metrics (stock 30B)
  logs/phase3/run.log                                start / done times
"""
import glob, json, os, re, sys, datetime as dt
import numpy as np
sys.path[:0] = ["code/compat", "code"]
import research_protocol as P

BENCH = [("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500")]
SAMPLE_ONLY = [("math500", "MATH-500"), ("aime", "AIME"), ("gpqa", "GPQA-Diamond"), ("ifeval541", "IFEval-541")]
p2 = json.load(open("logs/phase3/extrap/p2_metrics.json"))


def load(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"):
            continue
        for l in open(f):
            if l.strip():
                try:
                    r = json.loads(l); out[int(r["idx"])] = r
                except ValueError:
                    pass
    return out


def agg(R, ids):
    tok = sum(R[i]["n_gen_to_eos"] for i in ids); sec = sum(R[i]["seconds"] for i in ids)
    return {"n": len(ids), "acc": 100 * np.mean([bool(R[i]["correct"]) for i in ids]),
            "passes_blk": sum(R[i]["n_forward"] - R[i]["n_prefill"] - R[i]["n_flush"] for i in ids) / sum(R[i]["n_blocks"] for i in ids),
            "loads_tok": sum(R[i]["unique_expert_layer_loads"] for i in ids) / tok,
            "drafter_fwd_tok": sum(sum(R[i]["drafter_n_by_kind"].values()) for i in ids) / tok,
            "tokens": tok, "seconds": sec, "tok_s": tok / sec, "ms_tok": 1000 * sec / tok}


def paired_delta(A, B, ids):
    d = np.array([int(bool(A[i]["correct"])) - int(bool(B[i]["correct"])) for i in ids])
    rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]


def walls():
    st, out = {}, {}
    for l in open("logs/phase3/run.log"):
        m = re.match(r"\[p3 (\d\d:\d\d:\d\d) UTC\] (start|done) (rd\d|judge) (\w+)", l)
        if not m:
            continue
        t = dt.datetime.combine(dt.date(2026, 10, 6), dt.time.fromisoformat(m.group(1)))
        k = (m.group(3), m.group(4))
        if m.group(2) == "start":
            st[k] = t
        elif k in st:
            w = (t - st[k]).total_seconds()
            out[k] = {"wall_min": (w + 86400 if w < 0 else w) / 60, "done_utc": t.strftime("%H:%M:%S")}
    return out


W = walls()
res = {"benchmarks": {}, "samples": {}}
for s, name in BENCH:
    B = load(f"logs/phase3/decode/judge/{s}/*.jsonl"); O = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
    full_base = os.path.exists(f"logs/phase3/decode/judge/{s}/.complete")
    order = [int(it["idx"]) for it in P.items_for_split(s)]
    st = p2[s]["stock"]; s0 = p2[s]["S0_nopool"]
    msl = (s0["ms_tok"] - s0["drafter_ms_tok"]) / s0["exp_tok"]
    stock_tps = 1000 / (st["exp_tok"] * msl)                      # the Phase 2 page's stock tok/s (Phase 1 box)
    b_here = agg(B, sorted(B)); b_old = agg(O, sorted(O))
    same = sum(B[i]["gen_ids"] == O[i]["gen_ids"] for i in B if i in O)
    box_ratio = sum(B[i]["seconds"] for i in B) / sum(O[i]["seconds"] for i in B)   # this box / Phase 1 box, same work
    e = {"name": name, "full_base_run": full_base, "base_identical_to_phase1": [same, len([i for i in B if i in O])],
         "base_here": b_here, "base_phase1_box": b_old, "stock_tps_phase1_box": stock_tps, "box_ratio_base": box_ratio,
         "wall_base_min": W.get(("judge", s), {}).get("wall_min"), "arms": {}}
    for a in ("rd1", "rd2"):
        R = load(f"logs/phase3/decode/{a}/{s}/*.jsonl"); ids = sorted(i for i in R if i in B)
        m = agg(R, sorted(R)); mp = agg(R, ids); bp = agg(B, ids)
        raw = mp["tok_s"] / bp["tok_s"]
        h30 = sum(R[i]["ms_by_kind"]["prefill"] for i in ids) / sum(B[i]["ms_by_kind"]["prefill"] for i in ids)
        hdr = sum(R[i]["drafter_ms_by_kind"]["prefill"] for i in ids) / sum(B[i]["drafter_ms_by_kind"]["prefill"] for i in ids)
        half = len(ids) // 2
        h1 = sum(R[i]["ms_by_kind"]["prefill"] for i in ids[:half]) / sum(B[i]["ms_by_kind"]["prefill"] for i in ids[:half])
        h2 = sum(R[i]["ms_by_kind"]["prefill"] for i in ids[half:]) / sum(B[i]["ms_by_kind"]["prefill"] for i in ids[half:])
        adj = raw * h30
        # 8-GPU wall on the Phase 1 box: per-item seconds here x (Phase 1 / here, base pipeline, same items), 8 shards items[k::8]
        f = sum(O[i]["seconds"] for i in B if i in O) / sum(B[i]["seconds"] for i in B if i in O)
        sh = [i for i in order if i in R]
        w8 = max(sum(R[i]["seconds"] * f for i in sh[k::8]) for k in range(8)) / 60
        e["arms"][a] = {"all_items": m, "paired_items": len(ids), "rd_paired": mp, "base_paired": bp,
                        "delta_acc_vs_base": paired_delta(R, B, ids),
                        "speedup_raw": raw, "host_index_30b_prefill": h30, "host_index_drafter_prefill": hdr,
                        "host_index_first_half": h1, "host_index_second_half": h2, "speedup_drift_adjusted": adj,
                        "wall_here_min": W.get((a, s), {}).get("wall_min"),
                        "extrap_tps_raw": b_old["tok_s"] * raw, "extrap_tps_adjusted": b_old["tok_s"] * adj,
                        "extrap_x_stock_raw": b_old["tok_s"] * raw / stock_tps, "extrap_x_stock_adjusted": b_old["tok_s"] * adj / stock_tps,
                        "extrap_wall_8gpu_min": w8}
    res["benchmarks"][s] = e
for s, name in SAMPLE_ONLY:
    J = load(f"logs/phase3/decode/judge_sample/{s}/*.jsonl"); O = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
    ids = sorted(J)
    res["samples"][s] = {"name": name, "items": len(ids), "identical_to_phase1": sum(J[i]["gen_ids"] == O[i]["gen_ids"] for i in ids if i in O),
                         "base_tps_here": agg(J, ids)["tok_s"],
                         "box_ratio": sum(J[i]["seconds"] for i in ids) / sum(O[i]["seconds"] for i in ids)}
os.makedirs("results/phase3", exist_ok=True)
json.dump(res, open("results/phase3/final_numbers.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ REPORT.md
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
L = []
A = L.append
A("# Phase 3: re-draft between judge passes — final report")
A("")
A(f"Generated {dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M IST')} by `logs/phase3/extrap/final_report.py` from the raw per-item records.")
A("")
A("## 1. What ran, what did not")
A("")
A("- **Box:** vast.ai, 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, one decode process per GPU (4 shards, `items[k::4]`), batch size 1.")
A("- **Models (unchanged from every earlier run):** JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc37d240aa6871f167e8f92d694281b0c22; drafter JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05d84be90a01c72443b2123ebf0376f6fd4; trained judge = LoRA Arushhh/c1b-judge-lora-sdar30b-b32 `main_128m/tok_000128000000` (weights hash 35805974, copy in this dataset under `weights/c1b_128M/`). Block 32, threshold 0.95, s8 first draft, `--judge full`, no expert pool.")
A("- **Setups:** base pipeline = `--drafter d32 --draft s8 --judge full --adapter <128M> --judge-diag` (run_p3.sh `judge` arm); rd1 = base + `--full-redraft-every 1`; rd2 = base + `--full-redraft-every 2`.")
A("- **Finished, every item, all three setups on this box:** HumanEval (164), MBPP (257), GSM8K-500 (500).")
A("- **Not run (stopped by the project lead after GSM8K):** rd1/rd2 on MATH-500, AIME, GPQA-Diamond, IFEval-541. The base pipeline ran only on their first 12 items (section 6).")
A("- **Base pipeline correctness on this box:** its outputs are token-identical to the Phase 1 records: " +
  ", ".join(f"{e['name']} {e['base_identical_to_phase1'][0]}/{e['base_identical_to_phase1'][1]}" for e in res["benchmarks"].values()) +
  "; " + ", ".join(f"{v['name']} {v['identical_to_phase1']}/{v['items']}" for v in res["samples"].values()) + ".")
A("")
A("## 2. MEASURED on this 4-GPU box (no extrapolation)")
A("")
A("All numbers below are measured on this box. Speedup = rd tok/s ÷ base tok/s on the same items (both decoded here, 4 GPUs busy).")
A("")
A("| Benchmark | Setup | Items | Accuracy % | Δ acc vs base, pts [95% CI] | Passes / block | 30B expert loads / token | Drafter forwards / token | Tok/s | ms / token | Wall (4 GPUs) | Speedup vs base (raw, measured) |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s, e in res["benchmarks"].items():
    b = e["base_here"]
    A(f"| {e['name']} | base | {b['n']} | {f1(b['acc'])} | reference | {f2(b['passes_blk'])} | {b['loads_tok']:.0f} | {f2(b['drafter_fwd_tok'])} | {f2(b['tok_s'])} | {f1(b['ms_tok'])} | {f1(e['wall_base_min']) + ' min' if e['wall_base_min'] else '–'} | 1.00 (reference) |")
    for a, x in e["arms"].items():
        m = x["all_items"]; d = x["delta_acc_vs_base"]
        A(f"| {e['name']} | {a} | {m['n']} | {f1(m['acc'])} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] | {f2(m['passes_blk'])} | {m['loads_tok']:.0f} | {f2(m['drafter_fwd_tok'])} | {f2(m['tok_s'])} | {f1(m['ms_tok'])} | {f1(x['wall_here_min']) + ' min' if x['wall_here_min'] else '–'} | **{f3(x['speedup_raw'])}x** ({x['paired_items']} paired items) |")
A("")
A("## 3. Host-speed drift on this box (why the measured speedup is not a clean number)")
A("")
A("**Evidence the host is shared:** `vastai show instance` reports `gpu_frac` 0.5 (we rented 4 of the machine's 8 GPUs; another tenant rents the other 4), `cpu_cores_effective` 128 of 256 (Xeon 6767P); the host load average was 11–13 while our 4 decode processes account for ~4; no CPU quota throttling and no GPU clock throttling were active. The other tenant's work shares the CPUs, memory bandwidth and the CPU turbo budget.")
A("")
A("This is a shared, rented server and the decode loop is CPU-bound (batch 1: each GPU is busy ~30–40% of the time, the rest is the CPU launching small kernels), so the box's speed changes with other tenants' load. rd and the base pipeline ran hours apart, so the ratio of their speeds contains the change in host speed between the two runs.")
A("")
A("**How it is measured:** for one item, the first forward of each model over the prompt (`ms_by_kind['prefill']` for the 30B, `drafter_ms_by_kind['prefill']` for the drafter) is exactly the same computation in rd1, rd2 and the base pipeline (same prompt, same weights, same code; re-drafting starts later). Host index H = Σ prefill time in the rd run ÷ Σ prefill time in the base run, same items. H = 1: same host speed; H < 1: the host was faster during rd (speedup overstated); H > 1: slower during rd (speedup understated).")
A("")
A("| Benchmark | Setup | H, 30B prompt prefill | H, drafter prompt prefill | H, first half / second half of items | Raw speedup | Speedup × H (drift-adjusted, NOT a measurement) |")
A("|---|---|---|---|---|---|---|")
for s, e in res["benchmarks"].items():
    for a, x in e["arms"].items():
        A(f"| {e['name']} | {a} | {f3(x['host_index_30b_prefill'])} | {f3(x['host_index_drafter_prefill'])} | {f3(x['host_index_first_half'])} / {f3(x['host_index_second_half'])} | {f3(x['speedup_raw'])}x | {f3(x['speedup_drift_adjusted'])}x |")
A("")
A("**Reading:** MBPP's runs saw the same host speed (H ≈ 1.00), so its raw speedup is clean. HumanEval's rd runs ran on a faster host (raw speedup overstated by ~2–5%); GSM8K's on a slower host (raw speedup understated by ~9–11%). H also moves within a run (first vs second half differ by up to ~5%), so the adjusted speedup carries roughly ±3–4% uncertainty. **The project lead does not accept the adjusted numbers as results; they are shown only to size the problem.** The clean fix is to decode rd and the base pipeline at the same time on the same box (e.g. 2 GPUs each, same items), so a host slowdown hits both equally.")
A("")
A("## 4. EXTRAPOLATED to the Phase 1 box (8x RTX PRO 6000 Server)")
A("")
A("**EXTRAPOLATED — based on the extrapolation using the measured numbers on 4× GPUs for this dataset for this model, using reference from the runs from this model (the base pipeline measured on the Phase 1 8× box).**")
A("")
A("Formula: tok/s(rd, Phase 1 box) = tok/s(base, MEASURED on the Phase 1 box, all items) × speedup(rd vs base, MEASURED here on the same items). × stock = that ÷ stock tok/s of the Phase 2 page. 8-GPU wall = rd's per-item seconds here × (Σ base seconds on the Phase 1 box ÷ Σ base seconds here, same items), replayed over 8 shards `items[k::8]`, slowest shard. The drift in section 3 passes 1:1 into these numbers.")
A("")
A("| Benchmark | Setup | Base tok/s, Phase 1 box (MEASURED) | rd tok/s, Phase 1 box (**extrapolated**, raw) | × stock (**extrapolated**) | rd tok/s (**extrapolated**, drift-adjusted) | 8-GPU wall (**extrapolated**) |")
A("|---|---|---|---|---|---|---|")
for s, e in res["benchmarks"].items():
    for a, x in e["arms"].items():
        A(f"| {e['name']} | {a} | {f1(e['base_phase1_box']['tok_s'])} (stock {f1(e['stock_tps_phase1_box'])}) | **{f1(x['extrap_tps_raw'])}** | {f2(x['extrap_x_stock_raw'])}x | {f1(x['extrap_tps_adjusted'])} ({f2(x['extrap_x_stock_adjusted'])}x stock) | {f1(x['extrap_wall_8gpu_min'])} min |")
A("")
A("**How to avoid this next time:** rent the whole machine for timing runs (`gpu_frac` = 1); decode the baseline and the new method at the same time (split GPUs or alternate items) and compare on paired items; run a fixed canary workload every ~10 min per GPU to detect host changes; pin each process to the CPU cores near its GPU; always report speed-independent costs (passes/block, expert loads/token) next to wall-clock.")
A("")
A("## 5. Formulas")
A("")
A("- tok/s = Σ n_gen_to_eos ÷ Σ seconds over items (never a mean of per-item rates); ms/token = 1000 · Σ seconds ÷ Σ tokens.")
A("- `seconds` per item: `torch.cuda.synchronize()` + `time.perf_counter()` around the whole decode (prompt prefill, every judge / drafter / re-draft / KV-flush forward); excludes tokenisation, scoring and record writing (`code/draftalign/da_decode.py` decode_item). No busy polling.")
A("- Passes / block = Σ (n_forward − n_prefill − n_flush) ÷ Σ n_blocks. 30B expert loads / token = Σ unique_expert_layer_loads ÷ Σ n_gen_to_eos. Drafter forwards / token = Σ drafter forwards (all kinds) ÷ Σ tokens.")
A("- Speedup vs base = tok/s(rd) ÷ tok/s(base) on the items both decoded (all items for the three benchmarks).")
A("- Δ accuracy = paired difference rd − base on the same items, 95% CI from 2,000 bootstrap resamples (numpy default_rng(0)), as the Phase 2 page.")
A("- Host index H and the drift-adjusted speedup: section 3. Extrapolation: section 4.")
A("")
A("## 6. Base pipeline on the other benchmarks (first 12 items only, this box)")
A("")
A("| Benchmark | Items | Identical to Phase 1 | Base tok/s here | This box ÷ Phase 1 box (time, same items) |")
A("|---|---|---|---|---|")
for v in res["samples"].values():
    A(f"| {v['name']} | {v['items']} | {v['identical_to_phase1']}/{v['items']} | {f2(v['base_tps_here'])} | {f3(v['box_ratio'])} |")
A("")
A("## 7. How to fetch everything")
A("")
A("- **Hugging Face (public dataset):** `hf download Arushhh/phase3-redraft --repo-type dataset --local-dir phase3-redraft` — `logs/phase3/decode/**` (every per-item record), `logs/phase3/jobs/**` (shard logs), `logs/phase3/run.log`, `logs/phase3/setup/**` (setup checks), `logs/phase3/extrap/**` (all extrapolation scripts and outputs), `results/phase3/**` (this report, final_numbers.json, tables), `weights/c1b_128M/` (the trained judge adapter), `ref/phase2/S0_128M_nopool_by_set/**` (Phase 1 box base records used as reference), `code/phase3/**`.")
A("- **GitHub:** `git clone -b phase3-redraft https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git` — code (`code/draftalign`, `code/phase3`, `code/phase3/extrap`), `results/phase3`, and all of `logs/phase3`.")
A("- **Public base models:** not re-uploaded; fetch at the exact revisions in section 1.")
A("")
A("## 8. How to run the numbers yourself")
A("")
A("```bash")
A("git clone -b phase3-redraft https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git dmge && cd dmge")
A("# Phase 1 reference records (if not already in logs/phase2/ref): copy them from the HF dataset")
A("hf download Arushhh/phase3-redraft --repo-type dataset --include 'ref/phase2/*' --local-dir /tmp/p3 && \\")
A("  mkdir -p logs/phase2/ref && cp -r /tmp/p3/ref/phase2/S0_128M_nopool_by_set logs/phase2/ref/")
A("python logs/phase3/extrap/final_report.py        # writes results/phase3/REPORT.md + final_numbers.json")
A("python logs/phase3/extrap/extrap.py . logs/phase3/extrap/p2_metrics.json logs/phase3/extrap/latest   # per-run detail")
A("```")
A("(`final_report.py` needs numpy and the repo's `research_protocol` for the benchmark item order; the run box used `/venv/p2`.)")
open("results/phase3/REPORT.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
