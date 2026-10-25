# Phase 3: re-draft between judge passes — final report

Generated 2026-10-06 23:36 IST by `logs/phase3/extrap/final_report.py` from the raw per-item records.

## 1. What ran, what did not

- **Box:** vast.ai, 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, one decode process per GPU (4 shards, `items[k::4]`), batch size 1.
- **Models (unchanged from every earlier run):** JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc37d240aa6871f167e8f92d694281b0c22; drafter JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05d84be90a01c72443b2123ebf0376f6fd4; trained judge = LoRA Arushhh/c1b-judge-lora-sdar30b-b32 `main_128m/tok_000128000000` (weights hash 35805974, copy in this dataset under `weights/c1b_128M/`). Block 32, threshold 0.95, s8 first draft, `--judge full`, no expert pool.
- **Setups:** base pipeline = `--drafter d32 --draft s8 --judge full --adapter <128M> --judge-diag` (run_p3.sh `judge` arm); rd1 = base + `--full-redraft-every 1`; rd2 = base + `--full-redraft-every 2`.
- **Finished, every item, all three setups on this box:** HumanEval (164), MBPP (257), GSM8K-500 (500).
- **Not run (stopped by the project lead after GSM8K):** rd1/rd2 on MATH-500, AIME, GPQA-Diamond, IFEval-541. The base pipeline ran only on their first 12 items (section 6).
- **Base pipeline correctness on this box:** its outputs are token-identical to the Phase 1 records: HumanEval 164/164, MBPP 257/257, GSM8K-500 500/500; MATH-500 12/12, AIME 12/12, GPQA-Diamond 12/12, IFEval-541 12/12.

## 2. MEASURED on this 4-GPU box (no extrapolation)

All numbers below are measured on this box. Speedup = rd tok/s ÷ base tok/s on the same items (both decoded here, 4 GPUs busy).

| Benchmark | Setup | Items | Accuracy % | Δ acc vs base, pts [95% CI] | Passes / block | 30B expert loads / token | Drafter forwards / token | Tok/s | ms / token | Wall (4 GPUs) | Speedup vs base (raw, measured) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HumanEval | base | 164 | 74.4 | reference | 5.91 | 739 | 0.36 | 7.23 | 138.3 | 11.4 min | 1.00 (reference) |
| HumanEval | rd1 | 164 | 75.6 | +1.2 [-4.9, +7.3] | 4.58 | 578 | 0.70 | 8.61 | 116.2 | 11.7 min | **1.191x** (164 paired items) |
| HumanEval | rd2 | 164 | 79.9 | +5.5 [+0.0, +11.0] | 4.65 | 627 | 0.51 | 8.58 | 116.5 | 9.9 min | **1.187x** (164 paired items) |
| MBPP | base | 257 | 66.1 | reference | 8.75 | 1285 | 0.44 | 4.70 | 212.6 | 13.9 min | 1.00 (reference) |
| MBPP | rd1 | 257 | 71.2 | +5.1 [+0.0, +10.1] | 7.17 | 1062 | 1.32 | 5.03 | 199.0 | 13.6 min | **1.069x** (257 paired items) |
| MBPP | rd2 | 257 | 69.3 | +3.1 [-1.6, +7.8] | 7.30 | 1122 | 0.82 | 5.12 | 195.3 | 12.3 min | **1.089x** (257 paired items) |
| GSM8K-500 | base | 500 | 91.6 | reference | 7.12 | 889 | 0.33 | 6.91 | 144.7 | 61.9 min | 1.00 (reference) |
| GSM8K-500 | rd1 | 500 | 91.6 | +0.0 [-1.4, +1.4] | 5.93 | 755 | 0.77 | 6.83 | 146.3 | 64.4 min | **0.989x** (500 paired items) |
| GSM8K-500 | rd2 | 500 | 92.0 | +0.4 [-1.2, +1.8] | 6.03 | 770 | 0.55 | 6.85 | 146.0 | 62.6 min | **0.991x** (500 paired items) |

## 3. Host-speed drift on this box (why the measured speedup is not a clean number)

**Evidence the host is shared:** `vastai show instance` reports `gpu_frac` 0.5 (we rented 4 of the machine's 8 GPUs; another tenant rents the other 4), `cpu_cores_effective` 128 of 256 (Xeon 6767P); the host load average was 11–13 while our 4 decode processes account for ~4; no CPU quota throttling and no GPU clock throttling were active. The other tenant's work shares the CPUs, memory bandwidth and the CPU turbo budget.

This is a shared, rented server and the decode loop is CPU-bound (batch 1: each GPU is busy ~30–40% of the time, the rest is the CPU launching small kernels), so the box's speed changes with other tenants' load. rd and the base pipeline ran hours apart, so the ratio of their speeds contains the change in host speed between the two runs.

**How it is measured:** for one item, the first forward of each model over the prompt (`ms_by_kind['prefill']` for the 30B, `drafter_ms_by_kind['prefill']` for the drafter) is exactly the same computation in rd1, rd2 and the base pipeline (same prompt, same weights, same code; re-drafting starts later). Host index H = Σ prefill time in the rd run ÷ Σ prefill time in the base run, same items. H = 1: same host speed; H < 1: the host was faster during rd (speedup overstated); H > 1: slower during rd (speedup understated).

| Benchmark | Setup | H, 30B prompt prefill | H, drafter prompt prefill | H, first half / second half of items | Raw speedup | Speedup × H (drift-adjusted, NOT a measurement) |
|---|---|---|---|---|---|---|
| HumanEval | rd1 | 0.979 | 0.965 | 0.982 / 0.975 | 1.191x | 1.165x |
| HumanEval | rd2 | 0.947 | 0.926 | 0.929 / 0.965 | 1.187x | 1.124x |
| MBPP | rd1 | 1.012 | 1.000 | 1.019 / 1.005 | 1.069x | 1.081x |
| MBPP | rd2 | 0.995 | 0.991 | 1.000 / 0.991 | 1.089x | 1.084x |
| GSM8K-500 | rd1 | 1.094 | 1.086 | 1.102 / 1.085 | 0.989x | 1.081x |
| GSM8K-500 | rd2 | 1.114 | 1.111 | 1.105 / 1.123 | 0.991x | 1.104x |

**Reading:** MBPP's runs saw the same host speed (H ≈ 1.00), so its raw speedup is clean. HumanEval's rd runs ran on a faster host (raw speedup overstated by ~2–5%); GSM8K's on a slower host (raw speedup understated by ~9–11%). H also moves within a run (first vs second half differ by up to ~5%), so the adjusted speedup carries roughly ±3–4% uncertainty. **The project lead does not accept the adjusted numbers as results; they are shown only to size the problem.** The clean fix is to decode rd and the base pipeline at the same time on the same box (e.g. 2 GPUs each, same items), so a host slowdown hits both equally.

## 4. EXTRAPOLATED to the Phase 1 box (8x RTX PRO 6000 Server)

**EXTRAPOLATED — based on the extrapolation using the measured numbers on 4× GPUs for this dataset for this model, using reference from the runs from this model (the base pipeline measured on the Phase 1 8× box).**

Formula: tok/s(rd, Phase 1 box) = tok/s(base, MEASURED on the Phase 1 box, all items) × speedup(rd vs base, MEASURED here on the same items). × stock = that ÷ stock tok/s of the Phase 2 page. 8-GPU wall = rd's per-item seconds here × (Σ base seconds on the Phase 1 box ÷ Σ base seconds here, same items), replayed over 8 shards `items[k::8]`, slowest shard. The drift in section 3 passes 1:1 into these numbers.

| Benchmark | Setup | Base tok/s, Phase 1 box (MEASURED) | rd tok/s, Phase 1 box (**extrapolated**, raw) | × stock (**extrapolated**) | rd tok/s (**extrapolated**, drift-adjusted) | 8-GPU wall (**extrapolated**) |
|---|---|---|---|---|---|---|
| HumanEval | rd1 | 14.6 (stock 11.6) | **17.4** | 1.50x | 17.0 (1.47x stock) | 3.0 min |
| HumanEval | rd2 | 14.6 (stock 11.6) | **17.3** | 1.50x | 16.4 (1.42x stock) | 2.7 min |
| MBPP | rd1 | 8.5 (stock 6.6) | **9.1** | 1.37x | 9.2 (1.39x stock) | 3.6 min |
| MBPP | rd2 | 8.5 (stock 6.6) | **9.3** | 1.40x | 9.3 (1.39x stock) | 3.3 min |
| GSM8K-500 | rd1 | 12.6 (stock 10.0) | **12.4** | 1.25x | 13.6 (1.36x stock) | 17.6 min |
| GSM8K-500 | rd2 | 12.6 (stock 10.0) | **12.5** | 1.25x | 13.9 (1.39x stock) | 17.6 min |

**How to avoid this next time:** rent the whole machine for timing runs (`gpu_frac` = 1); decode the baseline and the new method at the same time (split GPUs or alternate items) and compare on paired items; run a fixed canary workload every ~10 min per GPU to detect host changes; pin each process to the CPU cores near its GPU; always report speed-independent costs (passes/block, expert loads/token) next to wall-clock.

## 5. Formulas

- tok/s = Σ n_gen_to_eos ÷ Σ seconds over items (never a mean of per-item rates); ms/token = 1000 · Σ seconds ÷ Σ tokens.
- `seconds` per item: `torch.cuda.synchronize()` + `time.perf_counter()` around the whole decode (prompt prefill, every judge / drafter / re-draft / KV-flush forward); excludes tokenisation, scoring and record writing (`code/draftalign/da_decode.py` decode_item). No busy polling.
- Passes / block = Σ (n_forward − n_prefill − n_flush) ÷ Σ n_blocks. 30B expert loads / token = Σ unique_expert_layer_loads ÷ Σ n_gen_to_eos. Drafter forwards / token = Σ drafter forwards (all kinds) ÷ Σ tokens.
- Speedup vs base = tok/s(rd) ÷ tok/s(base) on the items both decoded (all items for the three benchmarks).
- Δ accuracy = paired difference rd − base on the same items, 95% CI from 2,000 bootstrap resamples (numpy default_rng(0)), as the Phase 2 page.
- Host index H and the drift-adjusted speedup: section 3. Extrapolation: section 4.

## 6. Base pipeline on the other benchmarks (first 12 items only, this box)

| Benchmark | Items | Identical to Phase 1 | Base tok/s here | This box ÷ Phase 1 box (time, same items) |
|---|---|---|---|---|
| MATH-500 | 12 | 12/12 | 8.93 | 1.779 |
| AIME | 12 | 12/12 | 8.58 | 1.823 |
| GPQA-Diamond | 12 | 12/12 | 6.18 | 1.802 |
| IFEval-541 | 12 | 12/12 | 4.21 | 1.881 |

## 7. How to fetch everything

- **Hugging Face (public dataset):** `hf download Arushhh/phase3-redraft --repo-type dataset --local-dir phase3-redraft` — `logs/phase3/decode/**` (every per-item record), `logs/phase3/jobs/**` (shard logs), `logs/phase3/run.log`, `logs/phase3/setup/**` (setup checks), `logs/phase3/extrap/**` (all extrapolation scripts and outputs), `results/phase3/**` (this report, final_numbers.json, tables), `weights/c1b_128M/` (the trained judge adapter), `ref/phase2/S0_128M_nopool_by_set/**` (Phase 1 box base records used as reference), `code/phase3/**`.
- **GitHub:** `git clone -b phase3-redraft https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git` — code (`code/draftalign`, `code/phase3`, `code/phase3/extrap`), `results/phase3`, and all of `logs/phase3`.
- **Public base models:** not re-uploaded; fetch at the exact revisions in section 1.

## 8. How to run the numbers yourself

```bash
git clone -b phase3-redraft https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git dmge && cd dmge
# Phase 1 reference records (if not already in logs/phase2/ref): copy them from the HF dataset
hf download Arushhh/phase3-redraft --repo-type dataset --include 'ref/phase2/*' --local-dir /tmp/p3 && \
  mkdir -p logs/phase2/ref && cp -r /tmp/p3/ref/phase2/S0_128M_nopool_by_set logs/phase2/ref/
python logs/phase3/extrap/final_report.py        # writes results/phase3/REPORT.md + final_numbers.json
python logs/phase3/extrap/extrap.py . logs/phase3/extrap/p2_metrics.json logs/phase3/extrap/latest   # per-run detail
```
(`final_report.py` needs numpy and the repo's `research_protocol` for the benchmark item order; the run box used `/venv/p2`.)
