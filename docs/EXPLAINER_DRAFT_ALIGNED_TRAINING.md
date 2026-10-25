# Draft-aligned training of SDAR-30B-A3B at block 32: the plain explanation

Written 2026-09-06 for anyone who joins this project without the session context. Everything here is either a
measured number (marked with the run it came from) or a stated assumption. Companion documents:
`DRAFT_ALIGNED_DESIGN_2026-09-06.md` (the design and all measured results), `TRAINER_SPEC_v0.md` (the trainer),
`HANDOFF_2026-09-06_DRAFT_ALIGNED.md` (jobs and commands).

## 0. Status and current recipe (updated 2026-09-07 14:15 IST; read this before the sections below)

> **Note added 2026-09-14.** "Round 3" in this document means the drafter fine-tune (section 6 of the
> design doc). That fine-tune ran as `train_drafter/d1` and `d2_cleanprefix` and was rejected at its gates
> (`HUGGINGFACE.md` §4). The project's current round 3 is a judge re-train from the stock base on a
> code / maths / IF mix: `docs/TRAINING_RECIPE_2026-09-09.md` and its *Amendments 2026-09-14*.

Sections 1-7 were written before round 1 ran and describe the ROUND-1 recipe. What is current:
- Round 1 (adapter with no draft marker, active at every position, router gates trained) is SUPERSEDED: 38% accuracy,
  19.6 passes/block (section 15).
- Round 2a is the current mechanism (section 14): a learned "this slot is a draft" vector added to the input
  embedding of every tentative draft slot; the LoRA delta applied ONLY at those slots (gate g_i in {0,1}); router
  gates frozen and the expert pool off; 30% of draft states are the exact pass-1 inference state and there are no
  all-mask states. Same loss (section 7), same data and budget as round 1. Trainable: 26.7 M LoRA + 2,048 = 26.74 M.
- Round-2a results (sections 17-20): held-out draft-slot accuracy 0.58 (copy baseline 0.56), confident commits right
  96%; paired on 100 train prompts: 91.0% at 6.9 passes/block (commit .95) or 94.0% at 7.93 (confirm .90 / override
  .99), versus stock 95.0% at 9.31 and the training-free rule 95.0% at 7.93. The judge ties the free rule at equal
  accuracy; it does not yet beat it. Cause: overrides (judge replaces the draft) are 17% wrong when confident;
  confirmations are 99% right but the judge confirms only 62% of the correct drafts confidently.
- Running now: round-2 data (remaining GSM8K, MATH, MBPP; 2.5-3x round 1); then automatically the drafter
  fine-tuning (section 6 of the design doc, round 3 lever) on 1 GPU and round 2b (resume 2a, +2,000 steps) on 7.
- Where things are: code/draftalign/ (probe_ab, da_decode, run_da_decode, adapter, train_forward, losses, dataset,
  train, train_drafter, analyze_*), logs/draftalign/ (records, traces, training logs), results/metrics/draftalign/.

## 1. How the big model writes, and why it is slow

The 30B model (Mixture-of-Experts: 128 experts per layer, 8 used per token, 48 layers) does not write one word at a
time. It takes a block of 32 empty slots, looks at all 32 at once, and for each slot outputs a probability over every
token in the vocabulary. It fills only the slots where the top token has probability above 0.95, then looks again.
Each look is one full forward pass, and each forward loads about 58 different experts per layer. Measured on the
block-32 checkpoint (100 GSM8K train prompts, stock rule): 9.3 forwards per block, 1,091 expert-layer executions per
generated token, 95.0% accuracy.

Why 9 passes. Take a slot in the middle of an empty block where the text could continue "So the total is", "Therefore
the total is", or "The total is therefore". All three are fine. The model splits its probability, say 0.4 / 0.35 / 0.25,
so nothing reaches 0.95 and nothing commits, although any choice would have been right. A slot whose token is forced
by the prefix (the digit after "10 x 2 = 2") gets 0.99 and commits at once. So pass 1 commits only the forced slots
(7.5 of 29.4 generated slots on average), and each committed slot forces about 3 more in the next pass. That chain
is the 9 passes.
Measured (100 frozen train prompts, 620 blocks): at pass 1 the model's top guess already equals the final token at
58.4% of slots, but only 25.6% commit; a third of all tokens (33.6%) are "right at pass 1 but uncommitted".
(The 87-item probe subset gives 59.1%; earlier text rounded this to "60%".)

## 2. What a small drafter offers

A 1.7B dense model (SDAR-1.7B-Chat-b32, same tokenizer) writes the whole block first. The big model then only has to
judge one concrete candidate per slot, which is a peaked decision where choosing was flat. Measured: the drafter's
block agrees with the big model's final tokens on 56% of slots (8 drafter steps), and a drafter forward costs 18 ms
against 370-520 ms for a big-model forward in our loop, i.e. eight drafter steps cost about 0.4 big-model forwards.

## 3. Why the stock big model cannot judge the draft (measured, code/draftalign/probe_ab.py)

- SDAR's training loss is computed only at empty ([MASK]) slots. At a filled slot the model was never told what to
  output. Measured: with the block filled by its own final tokens, the output at slot i equals the token at slot i+1
  in 93.6% of the slots that HAVE a successor inside the block (90.4% if each block's last slot, which has none, is
  counted as a miss) and the slot's own token in 10.0%. The autoregressive next-token head of the Qwen3 base survives
  there. Reading a filled slot on the stock model gives a shifted copy of the input, not a judgement.
- The in-distribution workaround (leave the judged slot empty, put draft tokens around it) makes the model confident
  but sycophantic: it agrees with the draft at 92% of slots, confirms 76% of the drafter's WRONG tokens at > 0.95, and
  corrects 1.2%. Decoding that way end to end: 67.0% accuracy vs 95.0%.
- The one uncontaminated signal is the model's own first pass with the block still empty. Accepting draft tokens to
  which that pass already gives probability > 0.5 keeps accuracy exactly (95.0, every item unchanged) and saves 15% of
  passes and 12% of expert work. (Precision OF THE ACCEPTED SLOTS at gamma .5 is 0.908, not the 0.963 quoted in the
  design doc: that figure is the precision of the whole pass-1 committed set, stock commits included. See section 21.)
  Accepting at > 0.3 loses 7 points. This is the ceiling without training.

Conclusion: the ability "judge a draft token given its neighbours" does not exist in the frozen model; training puts
it there. No threshold or vote can.

## 4. What is trained (round-1 table; in round 2a the gates are FROZEN and a 2,048-number draft vector is added)

Only an adapter inside the 30B model. Experts, embeddings, output head and the drafter are frozen.

| trained part | where | parameters |
|---|---|---|
| LoRA rank 32 (alpha 64) on the 4 attention projections q, k, v, o | all 48 layers | 26.7 M |
| the router gate (the 128-way "which experts" matrix) | all 48 layers | 12.6 M |
| total | | 39.3 M = 0.13% of 30.5 B |

LoRA: for a weight W the model computes W x + (alpha/r) B A x with thin trainable A (r x in) and B (out x r), B
zero-initialised so training starts exactly at the stock model (verified: max logit difference 0.0). The attention
adapter is what lets the model change how it reads a filled slot. The gates are trained so the model keeps working
when each forward is restricted to a pool of 24-48 experts.

## 5. The training data: one row per block

Generated by decoding train prompts (3,000 GSM8K train, 2,500 MATH train across all subjects, 120 MBPP train; the
100 frozen train prompts and all test sets excluded) with the stock model + drafter under the > 0.5 acceptance rule of
section 3, keeping items whose final answer is correct. Per block the row holds:
- FINAL: the 32 tokens the block ended with (the targets);
- DRAFT: the 32 tokens the 1.7B drafter wrote for the same prefix;
- committed[pass]: which slots the stock rule filled at each pass;
- accepted: which draft tokens the > 0.5 rule took at pass 1;
- the prefix (prompt + earlier blocks), and item correctness.

Real example (GSM8K train item 9, block 3; prefix "...overtime, which is paid by her hourly wage + "):

```
slot | FINAL (target) | DRAFT (1.7B) | same | pass committed by the stock rule
  0  | '1'            | '1'          | yes  | 0
  3  | ' her'         | ' her'       | yes  | 0
  7  | ' This'        | ' This'      | yes  | 1
  8  | ' means'       | ' means'     | yes  | 2
 11  | ' rate'        | ' rate'      | yes  | 15
 17  | ' ($'          | ' $'         | NO   | 10
 18  | '1'            | '0'          | NO   | 8
 20  | '/'            | ' ='         | NO   | 13
 ... slots 21-31: draft "$22. hour.\nIf she works 1" vs final ") = $18 + $9 = $"   (all NO)
agreement 0.56, stock rule needed 16 passes
```

## 6. What one training example is (round-1 mix; round 2a: 30% all-draft states, no all-mask states)

From one row, freshly randomised every time it is drawn: with probability 0.4 no slot is committed (the pass-1
state); otherwise the committed set of one recorded pass. Every open slot holds the DRAFT token with probability 1-m
and [MASK] with probability m, where m is drawn uniformly in [0, 1]; with probability 0.2 all open slots are [MASK]
(the stock regime, so the model does not forget normal decoding). The prompt and earlier blocks are the prefix, with
the same block-causal attention the inference cache uses.

```
INPUT : [C '1'] [C '/'] ... [C ' overtime'] [D ' rate'] [C ' is'] [D ' $'] [C '1'] [C '8'] [C ' +'] [D ' $'] [C '1'] [D '8'] [D ' ='] [D ' $'] [D '2'] ...
TARGET:  -      -          -                 ' rate'      -        ' $'     -       -      -       ' ($'    -       '8'   '/'    '2'    ')'  ...
```
[C x] committed (no loss), [D x] draft token in an open slot, [MASK] empty open slot. The loss is computed on open
slots only. The draft token is INPUT only; it is never a target. Where the draft was right the target equals it
("confirm"); where it was wrong the target differs ("override").

## 7. The loss (code/draftalign/losses.py)

O = open slots of the example, T_i the target at slot i, z_i the model's logits at slot i, p_i = softmax(z_i).

```
L = (1/|O|) Σ_{i in O} [ -log p_i(T_i) ]                      cross-entropy on every open slot
  + beta * (1/|C|) Σ_{i in C} H( softmax(z_i / tau) )          sharpening, only where already right
C = { i in O : argmax p_i = T_i },  beta = 2.0,  tau = 0.5,  H = entropy with the [MASK] column removed
```
The first term teaches what to output. The second is dParallel's "certainty forcing" (arXiv 2509.26488): where the top
guess is already the target, push its probability toward 1 so the slot clears the 0.95 commit threshold in one pass.
It is never applied to a wrong slot. The [MASK] column is removed so entropy cannot be lowered by piling probability on
[MASK], which is never a legal output. In a random share of steps the forward runs with an expert pool of 24, 32 or 48
experts voted from the block; the same loss then trains the gates to route well inside a pool.
Not in the loss: the drafter; any pull toward the original model (the targets are its own outputs); committed slots;
prefix tokens. Optimiser: AdamW, lr 2e-5 (LoRA) and 1e-5 (gates), cosine, global batch 32 states, 2 epochs.
Measured on one real step before training: cross-entropy 1.38, sharpening term 0.004, top guess right on 78% of open
slots, 5.3 s per step, 57.7 GiB per GPU.

## 8. Can a 39 M-parameter adapter undo a habit learned on trillions of tokens?

The habit is not deep, it is unsupervised. Qwen3 was pretrained to predict the next token; SDAR then changed its
behaviour at EMPTY slots completely with 50 B tokens of continued training, about 0.1% of the pretraining budget, and
never touched filled slots, which is why the next-token output survives there. Both pieces the judge needs already
exist in the network: "copy the token at my own slot" is an identity attention pattern, and "predict a slot from its
neighbours" is the masked-slot machinery SDAR trained. The adapter has to make the second fire at filled slots instead
of the first, which is a change of where attention looks, exactly what a low-rank update to q, k, v, o expresses.
Published precedents for teaching a diffusion LM to correct filled tokens with modest fine-tuning: Corrective DLMs
(arXiv 2512.15596), ProSeCo (2602.11590), and dParallel for the confidence part. The uncertain quantity is not
whether the behaviour changes but how confident the judge becomes, which is what the held-out metrics track.

## 9. How much data, compared with dParallel

dParallel: about 92,000 sequences (256 tokens, one active block of 32 with half its tokens masked), 3-6 epochs, LoRA
rank 32-128 on LLaDA-8B: roughly 1.5 M supervised tokens per epoch, 4-9 M in total.
Ours, round 1 (GSM8K only): about 15,000 blocks with on average ~20 open slots: ~300,000 supervised slots per epoch,
~600,000 over 2 epochs, i.e. 7-15x less than dParallel. Round 2 (GSM8K + MATH + MBPP): about 40,000 blocks, ~1.6 M
over 2 epochs, 3-6x less. Each of our slots carries more information (a draft token to judge), but the count is
smaller, so under-confidence is the first risk to watch. More prompts are available (7,273 GSM8K, 7,500 MATH) and the
generation pipeline is automated.

## 10. What success looks like (pre-registered, design doc section 4)

Primary, on the frozen GSM8K-500, every metric paired against the stock decode on the same items:
- accuracy within 1.5 points of stock (90.2%), confidence interval lower bound above -3;
- passes per block <= 5 (stock 10.06), i.e. big-model forwards per token <= ~0.22 (stock 0.415);
- expert-layer executions per token, big model plus the drafter's byte-equivalent, <= 582 (stock 1,164);
- with the pool: <= 32 unique experts per forward at <= 1 further point of accuracy;
- the draft arm (T1) beats the same recipe without draft (T0) by >= 20% in passes per block at equal accuracy.
If all hold, the result reads: a 1.7B drafter plus a 39 M-parameter adapter lets a 30B MoE diffusion model decode a
32-token block in about 4 big-model forwards instead of 10, each touching about 32 experts instead of 58, at unchanged
accuracy: roughly 2.5x fewer big-model forwards and 4x fewer expert loads per token, beating TEAM's 7.56 experts per
token on this exact model.
Partial outcomes and what they mean: passes fall but the pool costs accuracy -> keep the step reduction alone (still
~2x forwards). T1 ~ T0 -> the draft adds nothing beyond certainty forcing; the result is a dParallel-style replication
on an MoE block-diffusion model, publishable but not the drafter claim. Neither -> the mechanism analysis (next-token
residue, sycophancy, blind-acceptance ceiling) is the paper, a weaker one.
Failure: accuracy drops more than 3 points at any pass saving.

## 11. Timeline (from 2026-09-06 14:00 UTC)

Data generation done ~19:00 UTC; training T1 starts automatically; first held-out numbers ~1 h later; T1 finished
~00:30 UTC on 2026-09-07; evaluation on the 100 train prompts ~01:00 (go/no-go); GSM8K-500 ~02:30; T0 control ~06:00.

## 12. How this differs from speculative decoding

Same shape (small model drafts, big model judges), different judge.

Classic speculative decoding: the big model is autoregressive. Its output at position i is, by training, the
probability of the token at i+1 given positions 1..i. Feed prefix + k drafted tokens in one forward and the output at
the slot BEFORE d_j is a judgement of d_j produced without seeing d_j (causal mask). The verifier exists for free,
verification is exact, a rejection-sampling rule makes the output distribution identical to the big model's, and
acceptance is a PREFIX (once d_3 is rejected, d_4.. are discarded because they assumed d_3). With a drafter that
agrees on 56% of tokens the accepted prefix averages about 2 tokens: the "one-token prefix" found earlier on this
branch.

Our big model is block-diffusion with bidirectional attention inside the block. Consequences:
1. The free verifier does not exist. The output at a filled slot was never trained (loss on masked slots only), what
   survives there is the next-token head, and it has already SEEN the token it would judge. Measured: 90% next-token
   copying. The adapter creates the verifier: the output at a filled slot is trained to mean "the correct token for
   THIS slot given everything around it" (keep = output equals input; replace = output differs).
2. Acceptance is PER SLOT, not a prefix. The judge reads the whole block, so its verdict on slot 20 can be right when
   slot 17 was wrong, and the diffusion decoder fills slots in any order. One pass can accept any subset of the 32
   slots and re-mask the rest: at 56% agreement, ~18 slots instead of ~2. This is why it can pay off at block 32, and
   exactly what a causal verifier cannot do. A causal-within-block readout (SimSD's layout for SDAR at small block widths) is
   possible but gives prefix acceptance, worthless at block 32.
3. Losslessness is empirical, not proven. The judge is a new model, so quality is protected by the targets being the
   base model's own outputs and by paired evaluation on frozen items, not by an exactness argument.
The loop also differs: rejected slots are re-masked and judged again with the accepted slots as context, a few rounds
per block; speculative decoding has no second round.

Literature placement: classic speculative decoding on a diffusion target exists (SimSD, SDAR-1.7B -> SDAR-8B,
blocks 4/8, prefix acceptance, training-free; works because whole 4-token blocks are often accepted; does not scale to
32). Per-slot acceptance WITHOUT draft context exists (APD, PSD, and our blind-acceptance baseline: accept a draft
token the big model's empty-block marginal already gives > 0.5; 15% fewer passes, accuracy unchanged). Per-slot
acceptance WITH draft context needs a judge that can read a filled slot; no stock diffusion model has one; that is
what the adapter adds. The drafter is stock here; speculative-decoding practice (EAGLE, Medusa) trains the drafter on
the target's outputs, which is the next cheap lever if the judge works.

## 13. Training log, round 1 (T1, started 2026-09-06 23:09 IST, 8 GPUs, 1,057 steps)
- Data: 2,843 correct GSM8K items of 3,000 (157 dropped), 16,905 blocks, 16,334 train / 571 held-out.
- Steps 10-100: loss 9.5 -> 3.7, cross-entropy 9.2 -> 2.8, training top-1 confidence 0.79 -> 0.58: the confident
  next-token copying at draft-filled slots is unlearned first (it dominates the initial loss).
- Held-out at step 100 (64 states per regime): all-mask states accuracy 0.49 (stock 0.59), share above .95 0.08
  (stock 0.26); draft-filled states accuracy 0.14 (stock 0.16), share above .95 0.23 with precision 0.38. Early;
  masked decoding disturbed, draft reading not yet learned.
- Steps 100-160: cross-entropy 2.8 -> 1.9, training accuracy 0.42 -> 0.50, confidence recovering to 0.61-0.66.
- Decision rule set: by step 300, draft-filled accuracy must be rising clearly (> 0.3) and all-mask accuracy
  recovering toward 0.59; otherwise stop, halve the learning rate, raise the all-mask share, restart from the
  step-200 checkpoint.
- Held-out at step 200: all-mask acc 0.48, share > .95 0.19 (precision 0.98); draft-filled acc 0.22, share > .95 0.17
  (precision 0.50). Stock reference at draft-filled slots: acc 0.16, share > .95 0.78, precision 0.19 (the confident
  next-token copies).
- Held-out at step 300 (decision point): all-mask acc 0.50, share > .95 0.23 (precision 0.95); draft-filled acc 0.32,
  share > .95 0.15 (precision 0.67). Draft reading is being learned (0.14 -> 0.22 -> 0.32); masked decoding stable at
  0.48-0.50 (stock 0.59) with confidence back to near stock. Rule satisfied on the draft side, marginal on the mask
  side: training continues; next checks at steps 500 and 1,057, then the paired decode evaluation on the 100 train
  prompts (the real go/no-go).
- Held-out at steps 400 / 500: all-mask acc 0.51 / 0.51 (stock 0.59), share > .95 0.27 / 0.28 (stock 0.26) at
  precision 0.94 / 0.95 (stock 1.00); draft-filled acc 0.35 / 0.36 (stock 0.16), share > .95 0.17 / 0.18 at
  precision 0.71 / 0.71. Gains are slowing as the learning rate decays (1.4e-5 at step 500). Two open concerns for
  the end-to-end test: confident commits at draft-filled slots are wrong 29% of the time against the target, and at
  masked slots 5% (stock 0%); whether those "wrong" tokens are harmless paraphrases or real errors is only visible in
  the paired decode evaluation after step 1,057.

## 14. Round-1 verdict and the round-2a redesign (decided 2026-09-07 02:00 IST)
Round 1 stopped at step 600 (held-out gains slowing: draft-filled acc 0.349 -> 0.357 between steps 400 and 500 while
the learning rate decays; it then rose to 0.393 by step 600, so the curve was slowing, not flat). Its step-600 checkpoint gets the paired decode evaluation.
Diagnosis. At draft-filled slots the adapter (0.393 at the step-600 checkpoint that was actually evaluated; 0.36 is the
step-500 value) is below the trivial "copy the input" policy (0.556 = drafter agreement). Two structural causes: (1) a tentative draft token and a committed token are identical in the input, so
the judge must first infer which slots to judge, which is impossible when the drafter's errors are plausible; at
inference we know which slots are drafts and were withholding it (speculative-decoding verifiers always know; diffusion
models are always told the noise level). (2) One LoRA delta at every position lets the judge's training bleed into
masked and committed slots: normal decoding fell 0.59 -> 0.51.
Redesign (round 2a, from scratch, same data and budget as round 1 so gains are attributable):
- draft-flag embedding: a learned vector added to the input embedding of every tentative draft slot (zero-init);
- flag-gated LoRA: the delta is applied only at flagged slots; committed, masked and prefix positions compute exactly
  as stock (bit-identical), so all-mask decoding cannot drift;
- 30% of draft states are the exact pass-1 inference state; no all-mask states (zero gradient under gating);
- router gates frozen, expert pool off (judge mechanism isolated); rank 32; lr 2e-5, cosine to 50%; 1,057 steps.
Prior art checked: LLaDA2.1's T2T stream and Self-Generated Error Training (2606.17175) train correction of filled
tokens WITHOUT telling the model which positions are corrupted (uniform or self-generated noise, full fine-tuning);
token-gated LoRA exists as a general technique (LoRA-Flow, gated LoRA). Flagging the known draft positions and gating
the adapter on them is the new combination here.
Correction (2026-09-07 02:35 IST, from the round-2a sanity run): at all-mask held-out states the STOCK model scores
about 0.49 against the training targets, not 0.59 -- the targets are the blind-accept decode's final tokens, which
include accepted draft tokens the stock pass-1 argmax does not always match. So round 1's masked-slot accuracy
(0.49 -> 0.51) was not a degradation; only its confident-slot precision (1.00 -> 0.95) is real drift. The main
round-1 problem stands: draft-slot accuracy 0.36 below the 0.56 copy baseline. Round 2a keeps the flag/gating design
(it protects masked decoding by construction and gives the judge the draft positions), with this reading corrected.

## 15. Round-1 result (ACTUAL, T1 at step 600, paired decode on the 100 train prompts, 2026-09-07 02:56 IST)
Full-fill judge with the round-1 adapter (commit > .95, keep draft where the judge agrees, else re-mask):
accuracy 38.0% (stock 95.0, training-free blind-accept 95.0); 19.6 passes/block (stock 9.31, blind 7.93);
1,909 MoE expert-layer executions/token (stock 1,091); 346 ms/token (stock 187); 94.7% of blocks advanced only by the
forced single commit. Verdict: recipe v1 fails on both axes. Mechanism: the judge is rarely confident (so passes
explode) and when it is confident it is wrong 29% of the time on held-out slots, which the paired accuracy shows are
real errors, not paraphrases. Records: logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c95_ad5d9add10/.
Round 2a (flagged draft slots, gated LoRA; section 14) launched 02:56 IST on 8 GPUs, 1,057 steps.

## 16. Round 2a training log (flagged draft slots + gated LoRA; started 2026-09-07 02:56 IST)
- Step-0 identity check passed; loss 16.5 at step 10 (30% fully-draft states, where the stock model is confidently
  wrong by ~16 nats per slot), 6.7 at step 100, cross-entropy 1.25 at step 190 (round 1: 2.0 at the same step).
- Held-out step 100: all-mask = stock exactly (acc 0.53, 25% confident, precision 1.00; this fixes the true stock
  reference against the training targets); draft-filled acc 0.14, 41% confident at precision 0.23 (unlearning phase).
- Held-out step 200: draft-filled acc 0.49 (round 1 @200: 0.22; @600: 0.36), confident share 0.13 at precision 0.99
  (round 1: 0.50 / 0.71); all-mask unchanged = stock. The flag is doing its job; the remaining task is confidence.
- Held-out step 300: draft-filled acc 0.53, confident share 0.30 at precision 0.94; all-mask = stock. Decision: continue
  (precision met; accuracy 3 points under the copy baseline but rising fast, unlike round 1). After step 1,057 the
  adapter is evaluated on the 100 train prompts four ways (commit .95 / .90, keep-agreeing-drafts / re-mask-only),
  then round-2 data generation starts on all 8 GPUs (code/draftalign/orchestrate_round2a_eval.sh).

## 17. Round-2a result (ACTUAL, adapter at step 1,057, paired decode on the 100 train prompts, 2026-09-07 09:05 IST)
Held-out at the end: draft-slot accuracy 0.58 (copy baseline 0.56; round 1: 0.36), confident share 0.39 at
precision 0.96; all-mask states exactly stock.

| decode | accuracy | passes/block | MoE forwards/token | expert-layer exec/token | ms/token |
|---|---|---|---|---|---|
| stock | 95.0 | 9.31 | 0.392 | 1,091 | 187 |
| training-free blind-accept .5 | 95.0 | 7.93 | 0.340 | 963 | 172 |
| round 1 judge (step 600; held-out draft-slot acc 0.393) | 38.0 | 19.6 | 0.758 | 1,909 | 346 |
| round 2a judge, commit .95 | 91.0 (-4.0 [-9, 0]) | 6.90 | 0.301 | 878 | 164 |
| round 2a judge, commit .90 | 89.0 (-6.0 [-12, -1]) | 5.75 | 0.257 | 747 | 139 |

The mechanism works (38 -> 91 accuracy, 19.6 -> 6.9 passes at the same budget as round 1) and beats stock on every
cost axis (-26% passes, -20% expert work), but it pays 4 accuracy points, outside the 1.5-point gate. Per block at
.95: 17.0 commits keep the draft token, 15.3 slots kept tentative, 56 re-masks (churn), 66.6% of blocks needed the
forced commit at some pass. Next: a held-out analysis of WHICH confident commits are wrong (confirm vs override) and
the threshold that gives ~99% precision; then round 2b with more data and a calibration fix.

## 18. Where round 2a's wrong commits come from (held-out analysis, 300 states, results/metrics/draftalign/holdout_precision_t2a.json)
At pass-1 all-draft states (8,687 slots; the draft is right at 55.5%): the judge's argmax is right at 63.0%.
Commits above 0.95: 42% of slots at precision 0.963 -- split into CONFIRMATIONS (argmax equals the draft token):
34.7% of slots at precision 0.991, and OVERRIDES (argmax differs from the draft): 7.2% of slots at precision 0.825.
So almost all wrong confident commits are overrides. Raising only the override threshold: at 0.99 overrides are
3.7% of slots at 0.953; at 0.995, 2.7% at 0.983. Expected wrong commits per block (29 slots): current rule 0.46;
confirm .95 / override .99: 0.14; confirm .90 / override .99: 0.19 with the same total commit share as today.
Mixed states show the same pattern (overrides 0.937 at .95; masked slots 0.984). Decision: add an asymmetric commit
rule (separate override threshold) to the decoder and evaluate confirm .95 / override .99 and confirm .90 / override
.99; keep the loss unchanged for round 2b and let the extra data improve override accuracy.

## 19. Asymmetric commit rule on round 2a (ACTUAL, 100 train prompts, paired)
| rule | accuracy | passes/block | exec/token | ms/token |
|---|---|---|---|---|
| stock | 95.0 | 9.31 | 1,091 | 187 |
| training-free blind-accept .5 | 95.0 | 7.93 | 963 | 172 |
| judge: confirm .95 / override .95 | 91.0 | 6.90 | 878 | 164 |
| judge: confirm .90 / override .90, re-mask disagreements | 91.0 | 6.51 | 826 | 149 |
| judge: confirm .95 / override .99 | 93.0 (-2) | 8.26 | 1,032 | 192 |
| judge: confirm .90 / override .99 | 94.0 (-1 [-5, +2]) | 7.93 | 995 | 182 |
Reading: making overrides stricter recovers accuracy but returns the pass savings; at -1 point the judge sits exactly
on the training-free rule's point (7.93 passes). Round 2a is therefore a large qualitative step (38 -> 94) but not yet
a better accuracy/pass frontier than the free baseline. Why: confirmations are capped by drafter agreement (55%) and
the judge confirms only 62% of the correct drafts at .95 (35% of slots); the productive overrides are exactly the
ones that are 17% wrong. Levers for round 2b: (i) confirmation recall (confidence at correct drafts; the held-out
confident share was still rising when the learning rate decayed), via more data and longer training on the union
(GSM8K 7.3k + MATH 2.5k + MBPP), resumed from the 2a adapter; (ii) override precision, same route; (iii) later, the
drafter's agreement itself (train the drafter on the MoE's outputs).
Remaining variants: confirm .90 / override .99 with re-mask-only = 92.0 at 8.75 passes; confirm .95 / override .995 =
93.0 at 8.87 passes. All four asymmetric variants sit on or below the stock/blind frontier. Round 2b (union data,
resumed from 2a, +2,000 steps) is queued to run automatically after data generation; round 3 = train the drafter on
the MoE's outputs (raises the 55% agreement cap), regenerate data with it, retrain the judge.

## 20. Coherence check and where the round-2a errors are (2026-09-07 13:45 IST, mentor's question)
The trained judge's answers on the 100 train prompts read like the stock model's: same reasoning steps, slightly
different wording (e.g. "her monthly phone bill would be" -> "is"). Stock and the free rule never break the answer
format; the judge at commit .95 breaks it in 2-3 of its 9 wrong answers: correct reasoning ending in `\boxed{180$.`
(missing brace) or `boxed{64}$` (missing backslash) -- a wrong override of one token in the answer wrapper, where a
single token flips correctness. The other 6 wrong answers are real reasoning errors. At confirm .90 / override .99 no
format failures occur (6 wrong, all reasoning). Records: logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_*/.
Queued: the judge (confirm .90 / override .99) and stock on MATH-500 for a second-dataset coherence check.

## 21. Corrections applied 2026-09-07 (documentation pass; every one re-derived on CPU from the records)
Re-derivation script: `results/figures/draftalign/verify_probe_numbers.py` (reads logs/draftalign/ab_probe/ and the
recorded stock traces; no GPU, no model).
1. Section 1: pass 1 commits **7.5 of 29.4** generated slots, not 7.7 of 29 (stock traces, 620 blocks: 7.53/29.39;
   probe subset: 7.52/29.39).
2. Section 1: the pass-1 argmax already equals the final token at **58.4%** of slots on the 100 frozen train prompts
   (59.1% on the 87-item probe subset), not 60%; commit share 25.6%; correct-but-uncommitted 33.6%.
3. Sections 3 and design 7.2: the blind-marginal "precision" column (.914 / .963 / .988 at gamma .3 / .5 / .7) is the
   precision of the WHOLE pass-1 committed set (stock commits, which are precise by construction, plus the accepted
   draft slots). The precision of the **accepted draft slots alone** is **0.826 / 0.908 / 0.960**. The wrong-per-block
   column (1.22 / 0.43 / 0.12) was always the accept-only figure and is unchanged; it is arithmetically inconsistent
   with the higher precision numbers, which is how this was caught.
4. Sections 3 and 12: the next-token copy rate at filled slots is **93.6%** over the slots that have a successor inside
   the block; 90.4% is the same quantity with each block's last slot (no successor) counted as a miss. The draft-filled
   figure 84.8% has the same convention (87.8% over slots with a successor).
5. Sections 14 and 17: round 1's held-out draft-slot accuracy at the **step-600 checkpoint that was evaluated** was
   **0.393**, not 0.36 (0.36 = step 500). The curve was slowing, not flat (0.349 -> 0.357 -> 0.393 at steps 400 / 500 /
   600). The verdict is unchanged: 0.393 is still below the 0.556 copy baseline.
Not corrected, but worth stating explicitly: the training targets are the **blind-accept (free-rule) decode's** final
tokens, not the stock decode's, so at accepted slots the target is a drafter token the stock marginal endorsed. This is
also why the stock model scores only ~0.49-0.53 against those targets at all-mask held-out states.
