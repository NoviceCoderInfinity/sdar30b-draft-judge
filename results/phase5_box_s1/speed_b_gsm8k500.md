# Stage B speed check, box session 1 (instance 57321867), vanilla GSM8K-500 on 8 GPUs

Run: `SETS=gsm8k500 bash code/drafter_lora/run_vanilla_here.sh` via `code/judge_round/speed_check_b.sh` (quiet box).
Wall-clock 0:40:11 (23 Oct 11:13:27 -> 11:53:38 IST). Records: logs/phase4/decode/vanilla_here/gsm8k500 (this box);
Phase 4 box's records moved to logs/phase4/decode/vanilla_p4box/gsm8k500. Summary: logs/phase5/speed_b/gsm8k500_full/summary.json.
Seconds per answer = the decoder's `seconds` field; GPU-active = `gpu_active_s` (nvidia-ml-py 13.615.71). Same 500 items on both boxes.

| quantity | this box | Phase 4 box | ratio |
|---|---|---|---|
| mean seconds per answer | 35.31 | 17.40 | 2.029 |
| median seconds per answer | 32.58 | 15.88 | 2.05 |
| mean GPU-active seconds per answer | 9.88 | 9.23 | 1.070 |
| token identity (ref_match gen_ids) | 500/500 | 500/500 | |
| gen_ids identical to the Phase 4 records | 500/500 | | |
| correct | 451/500 | 451/500 | |
| per-GPU mean seconds, min / max | 33.32 / 37.78 (max/min 1.134) | 15.93 / 18.64 (1.170) | |

Per GPU (shard k on GPU k), mean seconds: here 34.27 35.57 37.78 36.10 34.10 33.32 35.71 35.61; Phase 4 18.15 17.98 17.77 18.64 16.33 15.93 17.42 16.97.

r = 35.31 / 17.4 = 2.03: more than 10% slower -> STOP (operator's proposed limit, not pre-registered). GPU-active time is within 7%,
so the gap is CPU-side (kernel launch, sync) overhead of the batch-1 decoder. Box: AMD EPYC 9554 (Zen 4) vs Phase 4's 9555 (Zen 5),
idle cores at 1.5 GHz under schedutil, driver 580.105.08 vs 615.71.09, no PIX GPU pairs.

Setup-check timings on this box (decoder `seconds` summed, both checks in parallel, box not quiet): stock 16 GSM8K items 584.6 s
(Phase 4 285.9 s, 2.04x); judge 16 HumanEval items 228.6 s (Phase 4 131.7 s, 1.74x; Phase 3 box 233.9 s).

Quiet-box evidence (logs/phase5/speed_b/gsm8k500_full/quiet_box.txt): before: load 7.47, GPUs 0 MiB, no non-system process; after:
load 1.73, GPUs 0 MiB, no non-system process (GPU 2 showed a 40% utilisation sample at the instant the last shard exited, memory 0 MiB).
During the run only the 8 shards and the 15-minute backup loop ran. Not run: the 100-question drift slice (operator: nothing else after B).
