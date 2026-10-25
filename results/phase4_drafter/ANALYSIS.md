<!-- Saved 13 Oct 2026 (operator: back up everything). Written read-only by the Phase 4 results-analysis agent (v2,
10:00 IST) and saved verbatim from its final report, except corrections made on saving: the judge-LoRA control numbers
of section 4.4 and the change log now quote the FINAL judge_lora_cost.json (all 541 IFEval questions; the agent had
quoted an earlier partial run). The Phase 3 on-this-machine numbers of section 4.7 are reproduced by
code/drafter_lora/phase3_same_box.py -> results/phase4_drafter/phase3_same_box.json; speed vs vanilla by
vanilla_same_box.json. Session scripts it cites (analysis_tools/, diag/) are in logs/phase4/session_scratchpad/. -->

# Phase 4 (drafter LoRA): results analysis

**Version 2 (final), 13 Oct 2026, 10:00 IST.** This is a read-only analysis of the finished Phase 4 run. It includes
the vanilla 30B re-decoded on this box: 06:54-09:56 IST, all 2,250 questions. Sections 4 and 5 were added at the
operator's request (about 07:15 IST).

**Conventions.**
* Times are IST.
* **P3** is Phase 3: the untrained SDAR-1.7B drafter, no ban.
* **P4** is Phase 4: the trained, merged drafter plus the ban.
* **vanilla** is the plain SDAR-30B-A3B, with no drafter and no judge.
* **base** is drafter plus judge without re-drafting (the Phase 1 pipeline).
* Δ means P4 − P3 unless stated otherwise.
* "×" means time per answer, baseline ÷ P4, unless stated otherwise.
* Labels on numbers:
  * **derived**: arithmetic on cited numbers, done for this analysis.
  * **estimate**: a model-based number, with its method stated.
  * **projection**: a forecast quoted from a source.

**Source keys.**

| key | file |
|---|---|
| R | `results/phase4_drafter/REPORT.md` (§ = section) |
| FT | `FINAL_TABLE.md` |
| BT | `baseline_tables.md` |
| SC | `speedup_chain.json` |
| G1-G4 | the gate JSONs |
| M1, M2, M3 | `measured_rd1.json`, `measured_rd2.json`, `measured_rd1_gate3.json` |
| VS | `vanilla_same_box.json` / `.md` (regenerated 09:56 IST, all 7 sets) |
| JLC | `judge_lora_cost.json` / `.md` (commit 43c82a1) |
| C25, TD | `check_25pct.json`, `train_done.json` |
| ST | `STATUS.md` |
| DOC | `docs/PHASE4_DRAFTER_LORA.md` |
| TL, CFG | `logs/phase4/train/main/{train.rank0.jsonl,config.json}` |
| DA, DB, DC | the diagnosis agents' outputs (session scratchpad `diag/`) |
| PLAN | the operator's plan artifact |
| P1, P2, P3p | the Phase 1, Phase 2 and Phase 3 report pages |
| P3R | `results/phase3_8gpu/REPORT.md` |
| CORP | `docs/CORPUS_1B_BUILD.md`, `code/corpus1b/{checkers,common}.py`, `logs/c1b/trainset/wave*/index.jsonl` |
| DEC | `code/draftalign/{da_decode,adapter,seqcache,run_da_decode}.py` |
| REC | per-item decode records under `logs/phase4/decode/` and `logs/phase4/ref/`, read by this analysis's read-only scripts (scratchpad `analysis_tools/`) |

## 1. Bottom line

**What Phase 4 did.** It trained a LoRA (r 32, 34.9M parameters) on the SDAR-1.7B drafter to predict the 30B's own
answers, then decoded all 2,250 benchmark questions in rd1 and rd2. Decoding used the merged drafter and a new ban: a
re-draft may not re-propose the token the judge just rejected.

**The drafter learned something real but small.**
* Held-out first-draft agreement went from 38.94% to 41.40% (+2.45 points), and non-final blocks gained +3.1 points in
  real decoding.
* Re-drafts no longer repeat the rejected token (27.3% / 15.0% → 0%).
* Drafter forwards per block fell 7.6% in rd1 and 5.3% in rd2.

**It did not deliver what the plan wanted.**
* Judge passes per block moved +0.6% (rd1) and −2.1% (rd2); the target was at least 5% fewer.
* Accuracy is unchanged from Phase 3: pooled +0.1 / −0.7 points.
* IFEval stays 6.5-6.7 points below vanilla, with 104 / 107 looping answers against a target of 60 or fewer.
* Gates 1 and 3 failed. The run continued only because gates were 5-minute checkpoints.

**Why: the judge's 0.95 commit rule, not a bug.** A slot is committed only when the judge is more than 95% confident.
The drafter's extra correct tokens land where the 30B itself is unsure, so they stay *tentative*:
* On matched blocks, agreement rose +5.0 points but pass-1 commits rose only +0.2.
* Perfect first drafts still need about 2 passes.
* Blocks that need more than 10 passes hold 58% of all passes.

**Speed, measured on this box.** The vanilla 30B was re-decoded here and is token-identical to the earlier records on
2,250/2,250 questions. P4 is **1.246× (rd1) and 1.262× (rd2) faster per answer**, 34.2 s against 27.5 / 27.1 s. By
benchmark it ranges from 1.09× (IFEval) to 1.45× (AIME) [VS]. The chained estimate (1.33× / 1.34×) was about 6% too
high. P4 needs 37% fewer 30B forwards per token, but gives back:
* 8-9% to longer answers (mostly IFEval loops);
* 5.7% to dearer judge forwards (the judge LoRA, about 9 ms per forward [JLC]);
* 8-11% to drafter and host time.

Against Phase 3 on the same box, P4 is about 1.04× (rd1) and 1.00× (rd2) (estimate).

## 2. Contributions

### 2.1 Delivered

| Deliverable | Details | Where |
|---|---|---|
| Trained adapter | LoRA r 32 / α 64 on q, k, v, o, gate, up, down × 28 layers; 34,865,152 parameters, 1.7% of the drafter. One pass: 1,285 steps, 328,818 items, 133.2M corpus tokens, 9,137 s [R §6, TD] | `logs/phase4/train/main/final_adapter`, `ckpt/` |
| Merged drafter | bf16, 2,031,739,904 parameters, tag `dr3d174340`. 3/3 load checks pass. Merge fidelity −0.11 points (42.33% → 42.22%) [check_merged.log, R §1] | `.../merged` |
| Public copies | HF `Arushhh/phase4-drafter-lora` (public); GitHub branches `phase4-drafter-lora` and `phase4-logs` | |
| Code | `code/drafter_lora/`, including the same-box tools (`run_vanilla_here.sh`, `vanilla_same_box.py`, `judge_lora_cost.py`) and the `--redraft-ban-rejected` flag. With the ban off, decoding is byte-identical to Phase 3 [DOC, R §8b] | repo |
| Tests | 29 + 60 CPU tests pass [ST] | |
| Measured decodes | Gate 3 (579 questions); Gate 4 rd1 and rd2 (2,250 each); vanilla on this box (2,250) | `logs/phase4/decode/` |
| Diagnosis | DA (code identity), DB (training vs inference), DC (decode records) [R §8b] | scratchpad |
| Reporting fixes | Vanilla glob (`93fd175`); `p4.sh` quoting (`108c126`) | |

### 2.2 Established

1. **A drafter can be trained toward the 30B, modestly.**
   * Held-out agreement +2.45 points, with every domain gaining: chat +2.02, code +4.85, if +2.39, maths +1.87, science
     +3.99 [G1].
   * In decoding, non-final blocks +3.1 points; matched blocks +5.0 [DC].
2. **The gain does not cut judge passes.**
   * Gate 3: −0.33% per block [G3]. Gate 4: +0.6% / −2.1% [G4].
   * On 907 matched blocks, wrong first-draft slots fell 13.62 → 12.78 per block while passes went 6.546 → 6.537
     (−0.13%, CI [−0.95, +0.64]) [R §8b; DB].
3. **Drafter work falls.** Drafter forwards per block −7.6% / −5.3% [FT]. Draft plus re-draft forwards per answer
   −10.6% / −2.9% (derived from M1, M2).
4. **The ban works as designed.**
   * Repeats fall from 27.3% / 15.0% to 0% [M1, M2].
   * 13% fewer re-draft events, and fewer re-drafted slots rejected: 41% vs 58% [R §8b; DC].
5. **Accuracy is unchanged from Phase 3.** Pooled +0.1 [−1.2, +1.4] and −0.7 [−2.1, +0.7] [FT]. The right/wrong
   flips are below Phase 3's own noise floor [DC].
6. **The pipeline is clean, deterministic and reproducible across machines.**
   * 76 CPU scenarios are byte-identical to Phase 3 with the ban off.
   * 8/8 config hashes rebuild.
   * 136/136 IFEval questions are identical between Gate 3 and Gate 4 [DA, R §8b].
   * The vanilla re-decode is token-identical to the earlier records on 2,250/2,250 questions [VS].
7. **Measured on this box, P4 is 1.246× / 1.262× vanilla per answer** [VS].
   * Phase 3 on this box is about 1.20× / 1.26× (estimate, 4.7).
   * The plan's headline of 1.37× / 1.35× was tokens per second [P3R, PLAN].
8. **The judge LoRA costs about 9 ms per judge forward**, measured within run [JLC].

## 3. Results against the plan

### 3.1 Gates and checks

| Check | Plan's bar [PLAN] | Measured | Verdict |
|---|---|---|---|
| Smoke test | params; KL 0; loss falls; parity ≥95%; pass ≤6 h | Skipped. The trainer asserted the parameter count; KL was 0 at step 1; CE 5.54 → ~2.9; projection 2.41 h. Parity 81.07% (3,000 items) [G1, TL, R §6] | Not run; the parity bar would have failed (5.3) |
| Selfcheck | 6 checks | 3/6 PASS. The failures trace to bf16 [ST] | Accepted |
| 25% check | ≥ +1.0 point | +1.84 [C25] | PASS |
| Merged drafter | loads; fidelity | 3/3; −0.11 points | PASS |
| Gate 1 | ≥ +3.0 points (target +7); retention ≥97% (fail <95%) | +2.45; 89.11% [G1] | **FAIL** |
| Gate 2 | precision drop ≤0.5 points; commit share rises | −0.48; 30.74% → 31.21% [G2] | PASS (0.02 to spare) |
| Gate 3 | ≥5% fewer passes/block (fail <2%); accuracy CI; IFEval loops | 0.33%; +1.04 [−1.21, +3.11]; 22 vs 25 [G3] | **FAIL** (passes) |
| Judge retraining (plan §8) | triggered if passes drop <5% | Trigger met; not run (needs approval) | Open |

The decision windows after Gates 1 and 3 drew no answer, and the run continued [R §3, ST].

### 3.2 Gate 4 success criteria (rd1 | rd2)

| Criterion | rd1 | rd2 | Verdict |
|---|---|---|---|
| IFEval loops ≤ 60 | 104 | 107 | not met |
| IFEval CI vs vanilla covers 0 | −6.5 [−10.4, −2.4] | −6.7 [−10.5, −2.6] | not met |
| Pooled CI vs vanilla covers 0 or better | −1.0 [−2.6, +0.5] | −1.3 [−2.9, +0.2] | met |
| Faster per answer than P3, tok/s not lower | P3 not re-decoded here. Same-box estimate 1.039× (rd1) | 1.002× (rd2) | **not established** (estimate only; 5.5) |
| ≥ 5% fewer judge passes per block | +0.6% | −2.1% | not met |

**The plan's goal** (IFEval gap closed, loops down to base level, faster) **was not achieved.**

### 3.3 Headline (2,250 questions per mode)

| Pooled | P3 rd1 | P4 rd1 | P3 rd2 | P4 rd2 | vanilla |
|---|---|---|---|---|---|
| Accuracy % [FT] | 64.4 | 64.5 | 64.9 | 64.2 | 65.5 |
| Δ vs vanilla [BT] | −1.1 [−2.8, +0.7] | −1.0 [−2.6, +0.5] | −0.6 [−2.1, +0.9] | −1.3 [−2.9, +0.2] | |
| Judge passes / block [BT] | 7.64 | 7.68 | 7.99 | 7.83 | 12.79 |
| Drafter forwards / block [FT] | 25.8 | 23.9 | 17.2 | 16.3 | 0 |
| Repeats of the rejected token | 27.3% | 0% | 15.0% | 0% | |
| Loops, all (IFEval) [BT] | 142 (107) | 123 (104) | 115 (94) | 127 (107) | 58 (44) |
| Time per answer on this box [VS] | not decoded here | **27.5 s** | not decoded here | **27.1 s** | **34.2 s** |
| Speed vs vanilla, **measured here** [VS] | ~1.20× (estimate, 4.7) | **1.246×** [1.226, 1.267] | ~1.26× (estimate) | **1.262×** [1.241, 1.284] | 1.00× |
| Speed vs vanilla, chained [SC] | 1.22× | 1.33× | 1.28× | 1.34× | |

Per benchmark, P4 − P3 [FT]:

| Benchmark | Passes/block Δ rd1 | rd2 | Drafter forwards/block Δ rd1 | rd2 | Accuracy Δ rd1 (95% range) | rd2 (95% range) |
|---|---|---|---|---|---|---|
| IFEval | +0.0% | −5.7% | −12.1% | −10.2% | −0.4 (−3.3, +2.8) | −2.0 (−5.5, +1.3) |
| MATH-500 | −0.1% | −3.2% | −4.3% | −4.0% | +0.8 (−1.8, +3.2) | +0.4 (−2.8, +3.4) |
| GPQA | +3.0% | +0.6% | −6.1% | −5.1% | −0.5 (−6.6, +5.1) | −1.0 (−7.1, +5.1) |
| AIME | −2.8% | −1.6% | −7.2% | −3.7% | +2.2 (+0.0, +5.6) | +2.2 (−3.3, +7.8) |
| HumanEval | +1.8% | −0.5% | −6.2% | +2.7% | +1.2 (−3.7, +6.1) | −4.9 (−9.8, +0.0) |
| MBPP | −0.7% | +0.1% | −4.5% | +3.4% | −0.8 (−5.4, +3.9) | +1.2 (−3.5, +6.2) |
| GSM8K | −2.7% | −2.2% | −5.0% | −3.3% | −0.2 (−1.8, +1.4) | −0.4 (−1.6, +0.8) |
| **Pooled** | **+0.6%** | **−2.1%** | **−7.6%** | **−5.3%** | **+0.1 (−1.2, +1.4)** | **−0.7 (−2.1, +0.7)** |

IFEval rd2 is the only cell that clears 5% fewer passes per block. It wrote 10.1% more tokens than P3 rd2, so its
judge passes per answer *rose* 3.6% (derived from G4).

### 3.4 How the run departed from the plan [R §4, ST]

* No smoke test.
* Setup checks capped at 5 minutes.
* Selfcheck 3/6 accepted.
* Gates turned into 5-minute checkpoints.
* Same-box baselines deferred. Vanilla was re-decoded here after Gate 4; P3 never was.
* W&B on during decodes (Phase 3 had it off).
* HF dataset public, uploaded every 20 minutes.
* Logs on branch `phase4-logs`.
* Judge retraining not run.
* Server-Edition box with AMD EPYC 9555 instead of the plan's Workstation-Edition offer.

## 4. Speed against vanilla, measured on this box

### 4.1 Method

`run_vanilla_here.sh` re-decodes the plain 30B (`--stock`, block 32, threshold 0.95) on this box: all 2,250 questions,
8 shards, W&B on as in P4's decodes. Every answer is token-identical to the earlier vanilla records, and correct =
ref_correct on 2,250/2,250 [VS]. So only vanilla's timing is new; its accuracy and counts are unchanged.

Speed is S_w = Σ w_V / Σ w_P, where w is `seconds` (synchronize + perf_counter around the whole decode), paired
bootstrap over the same questions (2,000 resamples, default_rng(0)). S_gpu is the same ratio with `gpu_active_s`
[VS, `vanilla_same_box.py`].

### 4.2 Results

| Benchmark (n) | S_w rd1 [95%] | S_w rd2 [95%] | S_gpu rd1 / rd2 | tok/s ratio | Chain [SC] | Chain ÷ measured |
|---|---|---|---|---|---|---|
| IFEval (541) | 1.102 [1.058, 1.144] | 1.087 [1.044, 1.130] | 1.067 / 1.073 | 1.51 / 1.52 | 1.157 / 1.166 | 1.051 / 1.073 |
| MATH-500 (500) | 1.275 [1.248, 1.303] | 1.308 [1.281, 1.334] | 1.245 / 1.272 | 1.28 / 1.33 | 1.335 / 1.317 | 1.046 / 1.007 |
| GPQA (198) | 1.385 [1.332, 1.443] | 1.442 [1.385, 1.505] | 1.355 / 1.422 | 1.34 / 1.39 | 1.516 / 1.556 | 1.095 / 1.079 |
| AIME (90) | 1.430 [1.357, 1.505] | 1.446 [1.376, 1.519] | 1.384 / 1.419 | 1.40 / 1.38 | 1.507 / 1.561 | 1.053 / 1.080 |
| HumanEval (164) | 1.256 [1.151, 1.371] | 1.255 [1.168, 1.357] | 1.233 / 1.254 | 1.35 / 1.38 | 1.378 / 1.394 | 1.097 / 1.110 |
| MBPP (257) | 1.222 [1.177, 1.268] | 1.290 [1.241, 1.342] | 1.221 / 1.288 | 1.30 / 1.35 | 1.377 / 1.433 | 1.127 / 1.111 |
| GSM8K (500) | 1.313 [1.282, 1.342] | 1.343 [1.318, 1.368] | 1.296 / 1.333 | 1.34 / 1.36 | 1.436 / 1.467 | 1.094 / 1.092 |
| **Pooled (2,250)** | **1.246 [1.226, 1.267]** | **1.262 [1.241, 1.284]** | 1.215 / 1.242 | 1.35 / 1.37 | 1.327 / 1.339 | **1.065 / 1.061** |

Sources: VS (rounded there) and REC (same formulas). Tokens per second over-states the speed wherever P4 writes more
tokens: IFEval 1.51 vs 1.10. Where P4's answers are shorter (GPQA), tok/s under-states it.

### 4.3 Decomposition

The identity is exact. N is tokens (`n_gen_to_eos`), F is 30B forwards, t = 1000·MoE seconds / F, and W/M is
wall-clock time over MoE time:

```
S_w = (N_V/N_P) x [(F/N)_V / (F/N)_P] x (t_V/t_P) x [(W/M)_V / (W/M)_P]
```

| Benchmark | Mode | length | forwards/token | ms/forward | outside-30B | = S_w |
|---|---|---|---|---|---|---|
| IFEval | rd1 / rd2 | 0.731 / 0.715 | 1.794 / 1.769 | 0.951 / 0.935 | 0.883 / 0.919 | 1.102 / 1.087 |
| MATH-500 | rd1 / rd2 | 0.998 / 0.987 | 1.525 / 1.514 | 0.938 / 0.953 | 0.893 / 0.919 | 1.275 / 1.308 |
| GPQA | rd1 / rd2 | 1.035 / 1.041 | 1.620 / 1.603 | 0.930 / 0.936 | 0.888 / 0.923 | 1.385 / 1.442 |
| AIME | rd1 / rd2 | 1.022 / 1.048 | 1.629 / 1.576 | 0.965 / 0.951 | 0.890 / 0.920 | 1.430 / 1.446 |
| HumanEval | rd1 / rd2 | 0.929 / 0.910 | 1.652 / 1.666 | 0.917 / 0.904 | 0.893 / 0.916 | 1.256 / 1.255 |
| MBPP | rd1 / rd2 | 0.937 / 0.955 | 1.612 / 1.584 | 0.914 / 0.925 | 0.885 / 0.922 | 1.222 / 1.290 |
| GSM8K | rd1 / rd2 | 0.981 / 0.987 | 1.623 / 1.593 | 0.915 / 0.921 | 0.901 / 0.928 | 1.313 / 1.343 |
| **Pooled** | rd1 / rd2 | 0.925 / 0.919 | 1.601 / 1.577 | 0.946 / 0.946 | 0.889 / 0.921 | 1.246 / 1.262 |

Pooled inputs: 319 vs 345 / 347 tokens per answer; 0.476 vs 0.297 / 0.302 30B forwards per token; 224.6 vs 237.4 /
237.6 ms per forward; the drafter is 10.5% / 7.5% of P4's wall-clock, other host time 0.9% / 0.8% (vanilla 0.4%)
[VS, REC].

**Reading.**
* P4's core advantage, 37% fewer 30B forwards per token, is about the same on every benchmark (1.51-1.79×).
* The spread in S_w comes mostly from length. IFEval's loops make P4 write 37-40% more tokens. Elsewhere length is
  within −5% to +10%.
* Two taxes apply everywhere: judge forwards 4-11% dearer, and the drafter at 7-11% of time.

### 4.4 Why a judge forward is dearer: the judge LoRA

**Within-run test (records only).** In rd2, 4-5% of judge passes have no draft-flagged slot. Flagged count: pass 0 =
n_valid; pass j = tentative(j−1) + redraft(j−1). On those passes `adapter.py` skips the LoRA and returns the frozen base
output. At matched experts per layer, flagged forwards cost:

| Benchmark | LoRA cost, flagged − unflagged | Fit d |
|---|---|---|
| IFEval | **+8.9 ms** [+6.6, +11.2], 27 bins | |
| MATH-500 | +9.0 ms | |
| GPQA | +9.2 ms | |
| AIME | +8.6 ms | |
| GSM8K | +8.4 ms | |
| HumanEval | (too few unflagged forwards to match) | +10.3 ms |
| MBPP | (too few unflagged forwards to match) | +8.6 ms |

Source: JLC. rd1 has no unflagged passes (re-drafts refill every hole), so it pays this on every judge pass.

**What else it could be, and why it is not.**
* *The fp32 readout:* ruled out. The per-forward timer (`seqcache._timed`: sync → model → sync) excludes `blind_read`
  and the `inputs_embeds` build [DEC].
* *Kernel count:* consistent. A flagged forward adds about 8 small kernels × 192 LoRA modules ≈ 1,536 launches, i.e.
  about 6 µs each (estimate).
* *Context length:* not the cause. Cost is flat in context, and in equal-context buckets the gap persists [REC].
* *Tokens per forward:* not the cause; it is 32 in both arms.
* *More experts per forward:* worth about 3-4 ms on IFEval (45.1-45.3 vs 44.2 experts per layer, at about 81 µs per
  (layer, expert) load) [REC].

**The residual.** Unflagged rd2 judge forwards still cost +2.1 ms (IFEval, bins +0.1 to +9.3; -2.3 to +5.3 ms across benchmarks) more than vanilla denoise forwards at
matched experts [JLC]. But cross-run comparisons carry about 3 ms of run-to-run offset: flagged − vanilla is +7.7 in
rd1 and +10.8 in rd2 [JLC]. The KV-flush forwards are a control: P4's go through the same 192 wrappers with flags off,
vanilla's have none. At matched experts they differ by −0.7 ms (rd1) and +1.3 ms (rd2), while P4 rd2 − rd1 is +2.1 ms
[REC]. So the hypothesis that the LoRA wrappers' Python overhead costs time even with flags off is at most about 1 ms
beyond run noise, and is **untested at this precision**. It would take an interleaved same-run A/B (wrappers installed
but off vs removed) or a profiler run on a GPU.

### 4.5 The decoder is launch-bound

**Evidence.**
* GPU-active time is about 53% of wall-clock in both arms [VS, REC].
* In the records inspected, SM occupancy is about 1% and SM utilisation 13-14%.
* `cpu_thread_s` ≈ `seconds`, but that includes CUDA spin-waits [R §10], so on its own it proves nothing.
* A forward costs about 24-35 ms fixed plus about 81 µs per (layer, expert) load [REC].
* The same fits on the Phase 3 box give 60-74 ms fixed, about 165 µs per load (7.6-8.1 ms per expert per layer), and
  an 18.5-20.6 ms LoRA cost. That is about 2× on the same GPU model, i.e. the CPU sets kernel throughput [REC].

**Implications (inference):**
* Fewer forwards convert almost one for one into wall-clock time.
* Fewer FLOPs hardly do.
* Extra kernels cost time even when they do almost no arithmetic: the LoRA path, casts, Python logic per pass.
* GPU-active time is not portable between boxes (4.7).
* Engineering changes would shift every ratio: CUDA graphs, fused MoE, pre-cast LoRA weights, batching.

### 4.6 Loops and length (diagnostic subsets)

**IFEval** [REC]:
* Loops: vanilla 44, P4 104 / 107. Caps: vanilla 46, P4 106 / 113.
* 72% / 79% of P4's extra tokens are on questions that loop in either arm. On the other questions P4 is 16.6% / 13.5%
  longer.
* S_w with no loop and no cap in either arm (n = 409 / 406): 1.087 [1.043, 1.129] / 1.119 [1.077, 1.161].
* S_w on the looping questions only (n = 115 / 119): 1.168 / 1.056.

**Other sets.** The same subset gives S_w close to the headline: MATH 1.278 / 1.309; GPQA 1.369 / 1.436; AIME
1.435 / 1.441; HumanEval rd1 1.308. MBPP and GSM8K have no loops.

**Reading.** Loops inflate tokens and tokens per second, and they cost accuracy. They do not set the time ratio. P4's
looping answers are cheap: about 6 30B forwards per block against about 14 for vanilla's.

### 4.7 Chain versus measurement, and P3 on this box (estimate)

**Method.** A cost model is fitted on P4's per-forward records on this box:
* judge forward: a + d·[flagged] + b·context + c·experts;
* KV-flush forward: the same form without the flag term;
* per-kind drafter ms and host time per forward.

**Validation.** It reproduces P4's measured time within ±1.0% on every set and mode. Out of sample, it predicts
vanilla's measured times within −0.7% to +2.5% (worst on the short code sets) [REC].

**Use.** Pricing P3's recorded traces at this box's costs gives the same-box estimates in the table [REC]:

| Benchmark | A in chain (P4/P3, cross-machine GPU-active) | Same-box estimate P4/P3 | A high by | Chain B×C (P3 vs vanilla) | Same-box estimate P3 vs vanilla | B×C high by |
|---|---|---|---|---|---|---|
| IFEval | 1.081 / 1.010 | 1.036 / 0.965 | 4.3% / 4.7% | 1.071 / 1.155 | 1.063 / 1.126 | 0.7% / 2.5% |
| MATH-500 | 1.112 / 1.078 | 1.056 / 1.034 | 5.3% / 4.3% | 1.200 / 1.222 | 1.208 / 1.264 | −0.7% / −3.3% |
| GPQA | 1.065 / 1.056 | 1.014 / 1.010 | 5.0% / 4.5% | 1.424 / 1.474 | 1.365 / 1.428 | 4.3% / 3.2% |
| AIME | 1.099 / 1.076 | 1.045 / 1.019 | 5.2% / 5.6% | 1.370 / 1.450 | 1.368 / 1.418 | 0.2% / 2.3% |
| HumanEval | 1.221 / 1.001 | 1.167 / 0.953 | 4.6% / 5.0% | 1.129 / 1.392 | 1.077 / 1.317 | 4.8% / 5.7% |
| MBPP | 1.066 / 1.048 | 1.006 / 1.000 | 6.0% / 4.8% | 1.291 / 1.367 | 1.214 / 1.290 | 6.4% / 6.0% |
| GSM8K | 1.076 / 1.082 | 1.028 / 1.035 | 4.7% / 4.5% | 1.334 / 1.356 | 1.277 / 1.298 | 4.5% / 4.5% |
| **Pooled** | 1.089 / 1.048 | **1.039 / 1.002** | 4.8% / 4.6% | 1.218 / 1.278 | **1.199 / 1.259** | 1.6% / 1.5% |

On the short code sets, read the B×C column as about ±2%, since the model over-predicts vanilla by 2-2.5% there.

**Reading.**
* The chain's cross-machine factor is 4-6% optimistic on **every** set.
* The Phase 2/3 factors err by −3% to +6% depending on the set. The computed C is pessimistic on MATH and optimistic on
  GPQA and the code sets.
* Pooled, the chain over-states by about 6%.
* P4 vs P3 on this box is about **1.04× (rd1) and 1.00× (rd2)**.

## 5. Stage-by-stage review from first principles

### 5.1 Data

| We assumed | The data shows | What could be wrong | How to check |
|---|---|---|---|
| Targets T are "the 30B's answers", so predicting T means predicting the judge's commits | T are the **plain 30B's stock-rule answers** (SGLang, block 32, threshold 0.95). They were kept only if they ended (EOS within the cap) and passed the domain checker (correctness; chat: ended, no loop, English). All 331,818 items have correct = 1 [CORP]. In decoding the committed text is the judge's, on another path: only 8% of IFEval answers are identical between P3 and P4 [DC] | Gate 1 scores agreement with vanilla's path, but decoding needs agreement with the judge's commits | Score agreement against judge-decoded text |
| The corpus covers decode-time situations | **No capped answers** (no-EOS answers were dropped) and no looping chat answers [CORP]. Decoding produced 187-233 capped and 115-142 looping answers per mode [M1, M2] | The drafter never trained in the regime where IFEval loses accuracy, so loops could not be trained away | Phase 3's untried "block loop continuations" rule [P3p Table 15] |
| The training mix suits the benchmarks | By answer tokens: maths 37.0%, chat 25.7%, IF 19.8%, science 9.5%, code 7.9% (73.8M answer tokens; prompts 60.6M of the 134.4M "corpus tokens"). The loss saw 39.2M hidden slots [CORP, TL, derived] | Code is 8% of targets but 2/7 benchmarks; chat is 26% with no benchmark | Re-weighting |
| The held-out split measures generalisation | 3,000 items stratified by domain, from the same waves; the judge also trained on those waves [CFG, DOC §5] | Gate 1 is in-distribution. Gate 2's absolute precision is optimistic (its delta is paired) | Gates on prompts neither model saw |
| First-draft states copy decode | They reveal the stored **SGLang untrained** draft's top-k slots. HF/SGLang parity is 81.07% at equal quality (38.94% vs 38.93%) [G1, DOC] | Off-policy: after training, the drafter's own s8 path visits unseen states | Regenerate states on-policy |
| Re-draft states copy rd1/rd2 | They reveal the **vanilla** commit order (about 2.5 slots per pass). The judge commits about 13 slots at pass 1, then about 5 falling to 1-2 per pass. Decode-time holes are the judge's disagreements, with tentative slots hidden [DOC, DC, DEC] | The hole patterns differ | Build states from judge-decode records |
| Post-EOS slots need no training | They were shown as truth and never loss rows [CFG]. Padding drafting fell 85.3% → 1.4% [R §8b]. Last-block agreement 0.728 → 0.363, at no pass cost (5.83 → 5.29) [DC] | Behaviour on slots without loss rows is unconstrained | Post-EOS loss rows / `--post-eos mask` |

### 5.2 Training

| We assumed | The data shows | What could be wrong | How to check |
|---|---|---|---|
| CE + 0.1 KL keeps what works ("the 30B's token stays on top") [PLAN] | The weighted KL was **0.26% of the loss** (0.25-0.29% per window); about 0.21 nats per KL row. Retention dropped to 0.874 by steps 21-100 and **no step exceeded 0.925 after step 20**; held-out 89.11%. The drafter kept 34.70, gained 6.70 and lost 4.24 points [TL, R §6-7, G1] | The per-slot optimum argument ignores shared weights: CE pressure on the 63.6% of slots the original gets wrong moves its right slots | KL-weight sweep |
| Lower loss means better drafts | CE fell 5.54 → 3.15 nats in 100 steps while argmax accuracy **fell** (−0.85 points vs the original). The rest of the pass moved the gap to +1.65 points for −0.25 nats [TL, R §6] | Mostly **recalibration**, which the judge never sees | Train-time commit-share probe |
| r 32 × 7 projections, lr 8e-5, one pass is enough | The curve flattens (+0.72 → +1.19 → +1.47 → +1.65); held-out +1.84 at 25%, +2.45 at the end. September fine-tunes moved agreement only 0.406 → 0.410 [PLAN]. No sweep was run | Capacity, learning rate, or an intrinsic ceiling (5.4) | Rank / placement / lr sweeps (attention-only = 12,845,056 parameters) |

### 5.3 Gates

| Gate | What it measures | Was the bar sensible? | What the number implies |
|---|---|---|---|
| Smoke (skipped) | Parity ≥95% etc. | The parity bar was unreachable for numerical reasons (81% parity, equal quality) | The off-policy issue went unexamined |
| 25% check | Offline agreement at step 322 | Fine | +1.84 on a flattening curve already pointed below +3.0 (hindsight) |
| Gate 1 | Offline agreement with **vanilla** tokens | **Not tied to the pass target.** By the plan's slope (0.30-0.38 passes per 10 points), +3 points ≈ 1.2-1.5% fewer passes and +7 ≈ 2.8-3.5% (estimate), so even the target fell short of 5%. The 97% retention bar was out of reach for this loss | +2.45 ≈ 1% by that slope |
| Gate 2 | Judge commit share and precision on pass-1 states | **The most direct predictor**, but the bar only asked that commit share rise | +0.47 points forecast decode pass-1 commits of +0.5 (41.0% → 41.5%) [G2, DC]: known at 01:03 IST |
| Gate 3 | Passes per block on 579 questions | **Per block rewards loops and length.** Looping capped IFEval blocks need 4.50 passes vs 17.82 [DC] | −0.33% per block; −3.7% per answer |
| Gate 4 | Full criteria | Loops were never targeted. P3 timing was impossible without a same-box re-run | 1 of 4 met (+1 not established) |

### 5.4 Decode

| We assumed | The data shows | What could be wrong | How to check |
|---|---|---|---|
| Better drafts → more commits → fewer passes | The rule: commit the judge argmax if p > 0.95; else keep an agreeing draft tentative, or re-mask; at least one forced commit per pass [DEC]. Matched blocks: +5.0 agreement → +0.2 pass-1 commits, +5.6 tentative; 70-82% of blocks take the same number of passes [DC] | **The new correct tokens sit where the 30B is unsure**; a correct draft does not make the judge confident | Oracle-draft probe (D = T), by difficulty |
| A perfect drafter → 1-2 passes ("unmeasured") | Perfect first drafts take 1.95 / 1.94 passes (IFEval 1.62 / 1.58); 48% finish in one pass; about 26.5/32 commit at pass 1 [DC, R §8b] | Hard blocks are untested | Oracle probe on hard blocks |
| Passes are spread evenly | Blocks with >10 passes are 25% of blocks but 58% of passes (IFEval 47% / 87%). Agreement on matched hard blocks 27% vs 24%. 31-51% of later passes commit only the forced slot [DC, P3p] | The cost sits in the slowest slots | Multi-slot forcing; eased confirmations |
| The ban is free | Repeats → 0%; re-drafted slots rejected 41% vs 58% [R §8b]. Offline, the ban adds +8.7 / +10.1 points of re-draft agreement while training adds −0.6 / +0.8; without the ban the trained drafter repeats 37.31% vs 26.43% [G1] | Confounded with the drafter; may block correct re-proposals [P3p §8] | No-ban and untrained + ban arms |
| Hiding tentative tokens from the re-drafter costs nothing | By design [DEC, P3p §7] | Discards agreed context | Arm with tentative tokens visible (untested idea) |
| Judge calibration is unaffected | The judge was trained on untrained-drafter drafts. Precision fell 0.48 points, and 4.9 points on wrong-draft rows [G2] | Calibration tied to the old drafts | Judge retraining (plan §8) |
| The asymmetric rule is unused | `--tau-override` exists [DEC] | A ready "eased rule" test | `--tau-commit 0.9 --tau-override 0.95` |

### 5.5 Measurement

| Quantity | Status | Rests on | Finding |
|---|---|---|---|
| Accuracy, passes, forwards, experts | Measured, machine-independent | Determinism across boxes | Verified: 2,250/2,250 vanilla answers identical here; P3 base reproduced Phase 1 |
| P4 vs vanilla time | **Measured on the same box** [VS] | Same code and flags, different run windows, uploads in both | Run-to-run offsets of about 1-3 ms per forward exist [JLC, REC] |
| P4 vs P3 time | Cross-machine; same-box **estimate** only | The cost model (±1% validation) | GPU-active ratio about 5% optimistic |
| Chain A×B×C | Estimate | A portable; B transfers; C correct | 6% high pooled; A 4-6% high everywhere; B×C −3% to +6% |
| tok/s | Misleading | — | Rewards long and looping answers |
| GPU-active | Not portable | NVML graphics-active | Launch-bound kernels |

## 6. New findings

1. **Extra agreement becomes tentative, not committed**, on matched blocks. This extends Phase 3's observation from
   re-drafts to trained first drafts.
2. **Perfect drafts need about 2 passes, and hard blocks hold most passes.**
3. **Across blocks, agreement predicts passes; within a block it barely moves them.** About 0.42 passes per wrong slot
   across blocks vs −0.13% within [DC, R §8b]. By the plan's own slope, the gate bars could not reach 5% (estimate).
4. **Gate 2 predicted the decode outcome** about 5.6 h before Gate 4 confirmed it.
5. **Training made un-banned re-drafts more repetitive** (26.4% → 37.3%). Offline, the ban carries the re-draft
   gains.
6. **Trained-drafter mistakes are more convincing to the judge** (wrong-draft commit precision 92.7% → 87.8%; maths
   86.6% → 70.9%).
7. **Most of the loss reduction was recalibration**, and the **KL was 0.26% of the loss**.
8. **Post-EOS training gap**, at no pass cost.
9. **The corpus excludes the failure regime** (no capped or looping answers).
10. **The IFEval accuracy loss is mainly re-drafting (Phase 3), not the trained drafter, and only on IFEval** (7).
11. **Same-box speed vs vanilla is 1.246× / 1.262×**, not 1.33× / 1.34×. The 37% forward saving is taxed by length,
    a dearer judge forward and the drafter (4.3).
12. **The judge LoRA costs about 9 ms per forward**, measured within run [JLC]. It is launch overhead: about 2× on the
    slower-CPU Phase 3 box.
13. **The decoder is launch-bound.** About 24-35 ms fixed plus about 81 µs per expert load; GPU active about 53%.
14. **GPU-active time is about 5% biased across machines**, so the chain's A is 4-6% high everywhere.
15. **Phase 3's headline over-stated its speed.** The tok/s chain said 1.37× / 1.35×. The time-per-answer chain says
    1.22× / 1.28×, and the same-box estimate is 1.20× / 1.26×.
16. **Loops cost accuracy, not speed.** They inflate tok/s; clean-answer S_w is close to the headline everywhere.
17. **IFEval flips track loops, and loop membership churns:** 76 questions loop in both phases, 31 only in P3, 28 only
    in P4 [DC].
18. **Two reporting bugs were caught before the final tables:** the stray vanilla AIME cell (`93fd175`, which moved
    only one vanilla AIME time on Gate 3) and the `p4.sh` brace default (`108c126`).

## 7. What we lost or regressed

| What | Measured | Against | Source |
|---|---|---|---|
| Retention | 89.11% (chat 83.26%, IF 82.84%) | bar 97% | G1 |
| Judge precision | −0.48 points; wrong-draft rows −4.9 (maths −15.7) | bar 0.5 | G2 |
| IFEval vs vanilla | −6.5 [−10.4, −2.4] / −6.7 [−10.5, −2.6] | P3 −6.1 / −4.6 | BT |
| Loops | IFEval 104 / 107; all sets 123 / 127 | vanilla 44 / 58 (about 2.4× / 2.1×); base 60 / 77 | BT, P3p |
| Length caps, rd2 | 197 | P3 177; vanilla 126 | M2 |
| HumanEval rd2 | −4.9 (−9.8, +0.0) vs P3 | P3 rd2's 79.9 was itself +5.5 over base (regression to the mean, interpretation) | FT, P3R |
| Small negatives vs P3 | GPQA −0.5 / −1.0, MBPP rd1 −0.8, GSM8K −0.2 / −0.4, IFEval rd1 −0.4 (all CIs include 0) | | FT |
| More passes | GPQA rd1 +3.0%, HumanEval rd1 +1.8%, pooled rd1 +0.6% | | FT |
| Speed vs P3 on this box | about 1.04× (rd1) and 1.00× (rd2), estimate | plan: faster | 4.7 |
| Post-EOS drafting | 85.3% → 1.4% | | R §8b |

**Accuracy attribution, full sets, paired** (vanilla → base → P3 → P4; 95% ranges [REC]):

| Set | vanilla → base | base → P3 rd1 / rd2 | P3 → P4 rd1 / rd2 | total rd1 / rd2 |
|---|---|---|---|---|
| IFEval | −1.7 [−5.2, +2.0] | **−4.4 [−8.1, −0.4]** / −3.0 [−6.5, +0.7] | −0.4 / −2.0 | −6.5 / −6.7 |
| MATH-500 | −0.8 | −0.8 / −1.4 | +0.8 / +0.4 | −0.8 / −1.8 |
| GPQA | −0.5 | +0.5 / +1.0 | −0.5 / −1.0 | −0.5 / −0.5 |
| AIME | −1.1 | 0.0 / 0.0 | +2.2 / +2.2 | +1.1 / +1.1 |
| HumanEval | −3.7 | +1.2 / **+5.5 [0.0, +11.0]** | +1.2 / −4.9 | −1.2 / −3.0 |
| MBPP | +0.8 | **+5.1 [0.0, +10.1]** / +3.1 | −0.8 / +1.2 | +5.1 / +5.1 |
| GSM8K | +1.4 | 0.0 / +0.4 | −0.2 / −0.4 | +1.2 / +1.4 |
| Pooled | −0.5 [−2.0, +1.0] | −0.5 / −0.1 | +0.1 / −0.7 | −1.0 / −1.3 |

Two cautions: "base" includes both the untrained drafts and the judge LoRA, and the records cannot separate them.
Re-drafting is the main loss **only on IFEval**; on code it helped.

**Time and compute.**
* The box has been up since 20:45:41 IST on 12 Oct: 10.1 h by 06:50 [ST], about 13.2 h by 09:56 (derived).
* Stages: training 2.54 h; Gate 1 about 28 min; Gate 3 41.5 min; Gate 4 4 h 51 min; the vanilla re-decode
  3 h 02 min; two 5-minute idle windows.
* The plan's stop rule would have ended GPU use at Gate 1 (plan: about $65). The override spent about 5.6 h of box
  time on Gates 2-4.
* The plan budgeted about 20.8 box-hours and about $314 (projection); this box's price is not in the sources.

## 8. Caveats and threats to validity

1. P4 vs P3 time is still cross-machine; the same-box value is an estimate (cost model, ±1% on P4, −0.7% to +2.5% on
   vanilla).
2. The chained figures (SC) are superseded for P4 vs vanilla by the measurement [VS]. For P3 vs vanilla they remain
   estimates.
3. Run-to-run offsets of about 1-3 ms per forward (about 1%) exist between same-box runs. Only within-run comparisons
   are clean at that level [JLC].
4. The drafter and the ban are confounded.
5. The held-out set shares waves with the judge's training data.
6. Single training seed; deterministic decoding means the intervals cover question sampling only.
7. Small benchmarks: AIME (90; +2.2 points = 2 items), GPQA (198), HumanEval (164). With 28+ interval comparisons, a
   couple of edge intervals are expected by chance.
8. Gate 4 contains Gate 3's questions.
9. The smoke test was skipped (its parity rule would have fired on numerical grounds), and the selfcheck failures
   were accepted (bf16; low risk).
10. W&B was on in the Phase 4 and vanilla re-decodes and off in Phase 3.
11. Matched-block analyses cover 907 / 1,336 of 6,711 Gate 3 blocks, skewed easy: 16% hard vs about 25% overall.
12. DC's invariant flags are a checker assumption (prompts shorter than one block have no prefill forward), not a
    decode fault [REC].
13. Speeds belong to this launch-bound implementation and box. Engineering changes would move them.

## 9. What would settle the open questions

| Question | Experiment | Cost in the sources |
|---|---|---|
| P4 vs P3 per answer, measured | Re-decode a P3 slice (or all of rd1/rd2) here; also calibrates GPU-time transfer | About 15 min for a slice (projection [ST]); for scale, Gate 4 took 4 h 51 min (measured) |
| Drafter vs ban | No-ban arm (trained) and untrained + ban arm, at least on the Gate 3 quarter | For scale, Gate 3 took 41.5 min |
| Ceiling under the 0.95 rule | Oracle-draft judge probe (D = T), by block difficulty | For scale, Gate 2's probe ran 48.5 s |
| Is 0.95 the binding constraint? | `--tau-commit 0.9 --tau-override 0.95`, gated on accuracy, loops, caps | none |
| The judge LoRA's ~9 ms; wrapper overhead | Profile / same-run A/B with the LoRA on vs off and wrappers removed; pre-cast A/B to bf16 | none |
| Retention with gains | KL sweep, attention-only LoRA, rank / lr sweeps | none (a pass took 2.54 h, measured) |
| Post-EOS gap | Post-EOS loss rows / `--post-eos mask` | none |
| Judge tuned to the new drafts | Plan §8 judge retraining (needs approval) | About 19.5 box-hours, about $294 (projection [PLAN]) |
| Loops / IFEval | Phase 3's "block loop continuations" rule | none |
| Training noise | Second seed | none |

## 10. Glossary

| Term | Meaning |
|---|---|
| Block, slot | Text is written in blocks of 32 slots |
| Drafter | SDAR-1.7B; proposes tokens |
| Judge | Frozen SDAR-30B-A3B + frozen 128M LoRA (flag-gated: active only when some slot holds a draft) |
| Judge pass | One 30B forward over the block |
| Commit | The judge's top-1 > 0.95 |
| Forced commit | No slot clears 0.95, so the most confident one is committed |
| Tentative | The judge agrees with the draft at ≤ 0.95 |
| Re-mask | The judge disagrees, so the slot is cleared |
| First draft (s8) | 8 steps × 4 slots |
| Re-draft (rd1 / rd2) | Refill re-masked slots after every / every 2nd pass |
| Ban | A re-draft may not repeat the just-rejected token |
| Base | Drafter + judge, no re-draft |
| Vanilla | The plain 30B |
| Agreement (offline / decode) | First draft = 30B token (Gate 1 replay on corpus items / inside real decoding); different scales |
| Retention | Share of the original drafter's right slots still right |
| Recalibration | Changing probabilities without changing the top token |
| Hard block | More than 10 passes |
| Matched block | Same prefix and same final text in P3 and P4 |
| Oracle draft | Draft = the 30B's tokens |
| Loop | More than half of the tokens inside a repeated 12-gram |
| Length cap | 1,024 tokens (512 for code) |
| Expert loads | Distinct (layer, expert) pairs per forward |
| S_w / S_gpu | Same-box time-per-answer / GPU-active ratio, vanilla ÷ P4 |
| Launch-bound | Speed set by kernel launches and latency, not arithmetic |
| Chain A×B×C | P4/P3 (GPU-active, cross-machine) × P3/base (Phase 3 box) × base/vanilla (Phase 2, partly computed) |
| Paired 95% range | Bootstrap over questions, 2,000 resamples, seed 0 |

## Change log

* **v1 (07:20 IST).** Partial IFEval same-box numbers.
* **v2 (10:00 IST).** All 7 sets measured on the same box, plus:
  * the decomposition per set;
  * judge LoRA cost from JLC;
  * the KV-flush control;
  * the chain error split (cost model);
  * the accuracy attribution per set;
  * a correction: unflagged judge forwards ≠ vanilla forwards (+2.1 ms on IFEval, -2.3 to +5.3 ms across benchmarks; within cross-run noise).