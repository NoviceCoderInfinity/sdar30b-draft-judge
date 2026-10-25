# Phase 3 8-GPU: forward-level cost model, same-machine speedup and Phase 1 projection

Generated 2026-10-08T00:01:44Z by code/phase3_8gpu/projection_8gpu.py from the raw per-item records. **MEASURED** = timed here; **PREDICTED** = this machine's cost model (fit on rd1 + rd2 only) applied to the base pipeline's forward lists, frozen before base ran here; **PROJECTED** = the Phase 1 machine's cost model applied to the rd forward lists (rd was never timed on that machine).

## 1. Cost model per machine (one 30B forward = a + b x unique expert-layer loads)

| Machine | Forward kind | Forwards | Mean ms | a (ms) | b (us / expert-layer load) | R² |
|---|---|---:|---:|---:|---:|---:|
| Phase 1 (base records) | kvflush | 24995 | 233.9 | 21.6 | 77.57 | 0.9211 |
| Phase 1 (base records) | judge | 243069 | 218.6 | 29.8 | 75.66 | 0.9161 |
| this machine (rd1 + rd2) | kvflush | 52947 | 514.5 | 58.8 | 168.80 | 0.9676 |
| this machine (rd1 + rd2) | judge | 413535 | 494.1 | 79.3 | 167.82 | 0.9678 |

Drafter ms per forward: Phase 1 prefill 11.93, draft 9.32, redraft 9.32, flush 10.03; here prefill 25.75, draft 21.14, redraft 21.15, flush 21.74. Other host time per 30B forward: Phase 1 1.55 ms, here 4.28 ms.
GPU-active time here: 18.02 ms per 30B forward + 45.08 us per expert-layer load + 7.50 ms per drafter forward (R² 0.9998, 4500 items).
Judge-pass cost ratio here / Phase 1: a 2.66x, b 2.22x: the per-expert cost dominates, as expected for a CPU-launch-bound batch-1 MoE.

## 2. Speedup of rd over base

| Benchmark | Arm | Items | rd tok/s here (MEASURED) | Speedup here, time: MEASURED | PREDICTED (frozen model) | GPU-only speedup, GPU time: MEASURED | PREDICTED | PROJECTED speedup on the Phase 1 machine, time | PROJECTED rd tok/s on Phase 1 | Phase 1 base tok/s (MEASURED) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IFEval-541 | rd1 | 541 | 4.73 | 0.986x | 1.012x | 0.970x | 0.982x | 1.005x | 10.85 | 8.93 |
| IFEval-541 | rd2 | 541 | 4.65 | 1.063x | 1.091x | 1.045x | 1.058x | 1.066x | 10.51 | 8.93 |
| MATH-500 | rd1 | 500 | 7.91 | 1.015x | 1.052x | 1.023x | 1.031x | 1.071x | 17.95 | 15.73 |
| MATH-500 | rd2 | 500 | 7.63 | 1.033x | 1.071x | 1.079x | 1.086x | 1.117x | 17.72 | 15.73 |
| GPQA-Diamond | rd1 | 198 | 5.21 | 1.119x | 1.132x | 1.075x | 1.086x | 1.137x | 11.74 | 10.00 |
| GPQA-Diamond | rd2 | 198 | 5.22 | 1.158x | 1.172x | 1.138x | 1.149x | 1.186x | 11.86 | 10.00 |
| AIME | rd1 | 90 | 6.98 | 1.051x | 1.074x | 1.035x | 1.042x | 1.039x | 15.80 | 14.42 |
| AIME | rd2 | 90 | 7.21 | 1.112x | 1.137x | 1.084x | 1.091x | 1.076x | 15.96 | 14.42 |
| HumanEval | rd1 | 164 | 7.77 | 0.955x | 0.968x | 0.930x | 0.938x | 0.957x | 17.40 | 14.59 |
| HumanEval | rd2 | 164 | 7.48 | 1.178x | 1.194x | 1.153x | 1.163x | 1.172x | 16.63 | 14.59 |
| MBPP | rd1 | 257 | 4.18 | 1.059x | 1.079x | 1.032x | 1.043x | 1.064x | 9.39 | 8.54 |
| MBPP | rd2 | 257 | 4.18 | 1.121x | 1.143x | 1.107x | 1.119x | 1.124x | 9.36 | 8.54 |
| GSM8K-500 | rd1 | 500 | 6.10 | 1.075x | 1.087x | 1.056x | 1.066x | 1.079x | 13.81 | 12.58 |
| GSM8K-500 | rd2 | 500 | 6.18 | 1.093x | 1.104x | 1.081x | 1.091x | 1.094x | 13.95 | 12.58 |

Speedup (time) = base seconds / rd seconds on the same items; GPU-only = the same with GPU-active seconds (NVML GPM). The PROJECTED column uses the Phase 1 base time MEASURED there, so its only modelled part is the rd time.

## 3. Model checks

| Benchmark | Phase 1 base: model / measured | rd1 here: model / measured | Base here: PREDICTED (frozen) / MEASURED | Base here GPU time: PREDICTED / MEASURED | Base here items done | Base tokens identical to Phase 1 |
|---|---:|---:|---:|---:|---:|---:|
| IFEval-541 | 1.011 | 0.994 | 1.026 | 1.012 | 541 (complete) | 541 / 541 |
| MATH-500 | 0.983 | 0.999 | 1.037 | 1.007 | 500 (complete) | 500 / 500 |
| GPQA-Diamond | 0.987 | 1.009 | 1.012 | 1.009 | 198 (complete) | 198 / 198 |
| AIME | 1.030 | 1.003 | 1.022 | 1.007 | 90 (complete) | 90 / 90 |
| HumanEval | 1.001 | 1.010 | 1.014 | 1.008 | 164 (complete) | 164 / 164 |
| MBPP | 1.005 | 1.009 | 1.019 | 1.011 | 257 (complete) | 257 / 257 |
| GSM8K-500 | 1.006 | 1.001 | 1.011 | 1.009 | 500 (complete) | 500 / 500 |

Frozen prediction written 2026-10-07T18:08:38Z (before any base item ran here): ifeval541 13.33 h / 3.87 tok/s; math500 10.22 h / 7.06 tok/s; gpqa 6.64 h / 4.45 tok/s; aime 3.10 h / 6.16 tok/s; humaneval 0.75 h / 6.44 tok/s; mbpp 1.03 h / 3.75 tok/s; gsm8k500 4.28 h / 5.52 tok/s.
