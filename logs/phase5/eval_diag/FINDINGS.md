# Phase 5 — why the retrained judge degenerates in live decode (CPU investigation 04:30–04:40 IST, 24 Oct)

**Trigger:** the rd1 eval with `rec_main_ep2/final` (155.66M) produced degenerate loops on ifeval541 (acc ~1.7% vs Phase 4 48.8%; outputs like "the the the…", "oliolioli…"). Operator stopped it by PID (records kept) and launched the `diag` bisect (ifeval541 first 16 items, exact eval flags, 2 GPUs each for 128M / 136M / 147.61M / 155.66M).

## Bisect result (diag, partial but saturated — run_da_decode, exact rd1 flags)
| adapter | acc | stop@EOS | truncated | pass/blk | gen→EOS |
|---|---|---|---|---|---|
| start_128M (Phase 4) | 0.56 | 89% | 11% | 15.23 | 343 |
| rec 136M (+8M tokens) | 0.14 | 14% | 86% | 8.65 | 892 |
| rec final 147.61M | 0.14 | 14% | 86% | 9.40 | 891 |
| rec_main_ep2 final 155.66M | 0.12 | 12% | 88% | 8.53 | 910 |

**Degeneration is immediate and saturating:** full-blown at the FIRST retraining checkpoint (136M = only +8M tokens) and flat thereafter. It is a property of the retraining SETUP, not the dose → no retrained checkpoint is salvageable, and picking an earlier one does not help. The telltale is termination: 128M stops at EOS 89% of the time (gen≈343); every retrained checkpoint truncates 86–88% of the time (gen≈900 = the length cap, never emits EOS) → repetition loops.

## Root cause: off-policy + teacher-forced training, invisible to the probe
The judge is retrained on the **128M judge's RECORDED per-pass states**, and those states are **teacher-forced**:
- `sampler_rec.py:16,124` — a training state's prefix is "the clean prefix (prompt + **vanilla's earlier blocks**, which is what teacher forcing put in the KV cache)". The recorder forces each block to vanilla's tokens (`c1b_judge_loop.py:236-237`, `t[:start]==blk[:start]`; "closing record mode: the teacher block (teacher forcing)").
- Live decode (`run_da_decode.py`) instead conditions each block on the **judge's OWN committed earlier blocks**.

So the judge is trained only on states reachable from vanilla's correct context, but at inference it rides its own trajectory. The loss (`sampler_rec.py:16-17`) targets **vanilla's tokens T at every open slot** (committed slots are context); it raises the judge's probability on T at open slots → raises commit propensity broadly. That is the intended "fewer passes" signal in-distribution, but it transfers as **over-commit on the judge's own OOD trajectory**: once it commits a slightly-wrong token, the drafter proposes repetitions, the judge commits those too (high prob), EOS is never reached → loop to the cap. (On some items it instead reject-loops — the batched eval showed items at ~24 pass/blk — same OOD failure, opposite sub-mode.)

### Why the probe and step-0 did not catch it
1. **Held-out probe** (conf_precision 0.996) runs on the SAME teacher-forced recorded states, teacher-forced, never autoregressive — over-committing looks like high precision there. It cannot see live loops.
2. **step-0 gap** (`train_c1b_rec.py:62-89`) compares the HF-trainer forward argmax vs the **SGLang-recorded live-judge argmax for the identical adapter**: 0.8978 at 128M = a real ~10% cross-engine/view mismatch in the training TARGET. Training optimizes through that mismatch, and the gap grew to **0.7256** at 155.66M = the adapter moving *away* from live-faithful decisions. That drift is the measurable fingerprint of the degeneration.

## Operator's four questions
1. **post_eos / flag conventions differ from decode?** Yes. `post_eos="truth"` (default) → the judge is never trained on post-EOS slots (they carry T, no flag, no loss row; `sampler_rec.py:20-24`), yet in decode it decides them live. Bigger: training prefix = vanilla (teacher-forced) vs live prefix = judge's own commits — the exposure-bias core.
2. **Does the loss teach committing re-drafted tokens?** It targets vanilla's T at all open slots (incl. redrafted ones); it raises commit propensity and does not separate "commit because right" from "commit because confident" → over-commit transfers to OOD trajectories.
3. **Does the 0.8978 step-0 gap point to a train/decode mismatch?** Yes, directly (see above): HF-train forward vs SGLang-record forward disagree 10% for the exact recording adapter; training into that drifts the adapter off the manifold where live decode terminates.
4. **loop-guard interaction?** loop-guard 12 bounds per-block redraft work but cannot rescue a judge that over-commits repetitions / never emits EOS; the sequence still truncates at the gen cap. Symptom-limiter, not cause.

## Bottom line
Phase 5 off-policy retraining degenerates live decode at **any dose**; no checkpoint beats the 128M baseline. On the original success criterion this is a **KILL** (accuracy collapses far past the <2% kill line; the lower pass/blk on retrained checkpoints is an artifact of degenerate non-terminating loops, not a real speedup). A valid retrain would need on-policy data (re-record with the candidate judge / DAgger) or a decode-faithful training forward on the judge's-own-prefix states — neither feasible before credit death ~08:00. **Recommend keeping 128M (Phase 4) as the production judge.**
