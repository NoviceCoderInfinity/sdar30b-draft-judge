# Phase 5 judge retraining — FAILURE ANALYSIS (root cause)

**24 Oct 2026, ~04:50 IST. CPU investigation; no GPU job started (the `chain128` Phase-4 eval owns the GPUs).**

## Verdict
Phase 5 (retrain only the flag-gated 128M judge on fresh wave-C recorded states) is a **FAIL**. Every retrained checkpoint degenerates in live HF decode into non-terminating repetition loops; accuracy collapses far outside noise. The 128M Phase-4 judge is unaffected. Keep **128M (Phase 4)** as the production judge.

## Evidence — diag bisect (logs/phase5/eval_diag, ifeval541 first 16 items, exact rd1 flags, plain HF harness run_da_decode, 2 GPUs/adapter)
| adapter | tokens | acc | looping | stop@EOS | truncated | gen→EOS |
|---|---|---|---|---|---|---|
| start_128m (Phase 4) | 128M | **56%** | 1/16 | 89% | 11% | 343 |
| rec tok_136M | +8M | 9% | 10/11 | 14% | 86% | 892 |
| rec final 147.61M | +19.6M | 7% | 14/15 | 14% | 86% | 891 |
| rec_main_ep2 final 155.66M | +27.66M | 8% | 11/12 | 12% | 88% | 910 |

128M reproduces Phase-4 accuracy (56% on these items) and terminates at EOS 89% of the time → **harness and adapter loading are fine**. Every retrained checkpoint truncates 86–88% of the time (gen≈900 = the length cap, never emits EOS), output is literal repetition (`"...the the the the..."`, `"oliolioli..."`). **Breakage is full-blown at the FIRST checkpoint (+8M tokens) and flat after → systematic in the training setup, not over-training. No earlier checkpoint is salvageable.**

## The recorded-state training construction is CORRECT (ruling out the "wrong targets/loss rows" hypotheses)
Loaded a real `RecStateSampler` ('sample' regime) over `logs/phase5/judge_states/rec_main` + the filtered index and dumped states (scratchpad/state_compare.py). For every state:
- **target = vanilla T, NOT the recorded argmax.** At a late pass, target==recorded-argmax on only 2/24 loss rows (`sampler_rec.py:142,153` keeps `base.targets`). ✔ correct.
- **committed slots are never loss rows** (`committed & open_mask == 0` in every state; `open_mask = recorded open & valid`). ✔ correct.
- **re-drafted / draft tokens are INPUTS, not targets** (at draft-filled slots target≠input on 20/24). ✔ correct.
- **flag sits exactly on draft-filled open slots** (`open & valid & input!=MASK`), matching `train_fast.block_hidden` and the A0 `flag == open & (input!=MASK)` check. Recorded passes have no MASK holes, so the flag is armed on every loss row. ✔ consistent.
- **post-EOS / EOS slots:** with `post_eos="truth"` the EOS slot is trainset-invalid for 49/60 items, so EOS is not a training target — but this is the **long-standing** trainset convention (128M trained the same way and still terminates 89%), so it is **not** the new cause. Termination is a flag-OFF base-model behavior; the flag-gated adapter cannot change it directly.

So the failure is **not** mislabeled targets, leaked committed/post-EOS rows, or redraft-as-target.

## Root cause — a train/decode ENGINE mismatch introduced by Phase 5's SGLang recording
Phase 5 is the **first** time the judge states were recorded in **SGLang** (`judge_sglang.py --mode record`, the newly ported `c1b_judge_loop.py`); earlier rounds' judge loop "lived only" in the HF harness (`docs/PHASE5_SGLANG_JUDGE.md`). The standing RULE keeps **all training (`train_c1b`) and all eval (`run_da_decode`) in the HF/torch harness.** So:

- **record = SGLang**, **train-forward = HF**, **eval-decode = HF.**
- SGLang and HF disagree substantially:
  - **step-0 check = 0.8978** argmax agreement between the HF train-forward and the SGLang-recorded live-judge argmax **for the identical 128M adapter** (`train_c1b_rec.py:62-89`): ~10% of judge decisions differ by engine alone.
  - **CP3 parity = 0.596** vanilla / **0.633** judge-side token agreement between SGLang and the HF harness: ~37–40% of tokens differ at the base-model level. CP3 was accepted on *aggregate* passes/block (±0.6%) and accuracy (−2.0), with token/argmax agreement "reported only, not blocking."

Consequence: the recorded states trace the **SGLang** judge's trajectory, but the HF judge at train- and decode-time visits a **different** trajectory. Training the HF judge to fit SGLang-recorded states is **off-distribution** — it optimizes commit decisions on states the HF judge never actually reaches. Even +8M tokens pushes the flag-gated commit behavior off the narrow manifold where HF decode stays coherent: the judge **over-commits** on its own (HF) trajectory, the committed context fills with wrong/repeated tokens, and the frozen base — now conditioned on garbage — never reaches an EOS state → repetition loops, 86–88% truncation, accuracy collapse. Teacher-forced prefixes in the recorded states (each block's prefix forced to vanilla, `sampler_rec.py:16,124`, `c1b_judge_loop.py:236-237`) compound the shift, since live decode conditions on the judge's own commits.

### Why the probe and step-0 could not catch it
Both are computed on the **same SGLang-recorded states** (teacher-forced, in-distribution for the metric), never on an HF autoregressive rollout. The held-out probe even improved (conf_precision 0.996, all/acc 0.50 at 155.66M). The **only** visible fingerprint was the step-0 gap **growing 0.8978 → 0.7256** as training pulled the adapter further from the (SGLang) recording judge — i.e. away from faithful decode behavior.

## Operator's five questions — answers
1. **flag on committed vs draft slots?** Correct: flag only on draft-filled open slots; committed slots carry no flag and are context, not loss rows.
2. **loss rows include committed or post-EOS slots?** No. Committed excluded (verified 0); post-EOS excluded by `post_eos="truth"` (the standard convention, not the cause).
3. **re-drafted tokens as targets?** No — redrafts are the input embedded at draft-filled slots; the target is vanilla T.
4. **target = recorded argmax instead of vanilla T?** No — target is vanilla T (argmax matches T on only a minority of rows).
5. **the 0.8978 step-0 gap?** This is the crux: a 10% HF-vs-SGLang judge-argmax mismatch for the identical adapter. Training through it (toward SGLang-recorded decisions) drifts the HF adapter off the live-decode manifold; the gap grows to 0.7256 at 155.66M.

## Lesson / what a valid retrain would need
- Record the training states **in the same HF harness** that trains and evaluates (no cross-engine gap), **or** make SGLang↔HF per-decision agreement ≫ 0.9 before trusting SGLang-recorded states for training (CP3's ~0.6 token / 0.898 argmax agreement was sufficient for *aggregate* parity reporting but far too low to train the judge on).
- Prefer **on-policy** states (re-record with the candidate judge in the eval engine; DAgger) over off-policy imitation of the predecessor's teacher-forced trajectory.
- Add a **live autoregressive** acceptance gate (a few decode items, check EOS/termination + accuracy) to training checkpoints — the teacher-forced probe is blind to this failure.

Neither fix is feasible before credit death (~08:00 IST), so Phase 5 ends here: FAIL, 128M retained.
