# Draft-aligned decoding for SDAR-30B-A3B at block 32: design plan v0

Date: 2026-09-06. Author: Claude (senior-scientist role) for Arush. Status: v0, written before the block-32 probe
(`code/draftalign/probe_ab.py`) has produced numbers; sections marked [AB] will be filled from it.

## 0. Ground truth (the user's instruction, 2026-09-06)

Use the dense drafter `JetLM/SDAR-1.7B-Chat-b32` (rev 2c83c05) -- its tokens, probabilities, hidden states -- on the
MoE `JetLM/SDAR-30B-A3B-Chat-b32` (rev c351bbc), both at block 32, to reduce (i) the number of unique experts the MoE
executes per forward and/or (ii) the number of denoising passes per block, at the MoE's accuracy. Training a LoRA /
adapter / small controller is wanted. Everything done before on this branch is an observation, not a constraint.

Both block-32 checkpoints exist on the Hub and are cached in `hf_cache/hub` (the drafter was downloaded today). Every
an earlier drafter experiment on this branch used a mismatched drafter checkpoint; those numbers are not the
block-32 operating point of the drafter.

## 1. First principles: why the stock model cannot use a draft, and what training must add

1.1 Where the passes go (REPLAY, CPU, the 100 frozen train prompts, b32 model, block 32, thr .95, 625 blocks):
- 9.45 passes per block (median 9, p90 17). Pass 1 commits 7.5 of ~29.5 generated slots; every later pass ~3.
- At pass 1 the MoE's argmax already equals the final token at 60% of slots; only 25% are committed. One third of all
  tokens are "correct but not committed" at pass 1 (median confidence 0.59). The rest of the trajectory is a chain:
  each pass resolves ~3 slots, which makes ~3 more slots confident.
- Mechanism: a masked slot's distribution is a MARGINAL over all completions consistent with the committed context.
  Wherever phrasing is free, the marginal is flat even though any single completion would be fine. Commit-by-threshold
  therefore serialises the block.

1.2 What a draft changes. A block filled with a coherent draft turns every marginal into a CONDITIONAL on one specific
completion: the model only has to judge "is this token right given these neighbours". Judging is peaked where
generating is flat. This is the mechanism behind Speculative Correction (arXiv 2608.02625, Aug 2026): a small dLLM
drafts, a large dLLM refines the full draft; refining a fully masked span under the same budget scores 0.004 vs 0.899
with the draft on GSM8K.

1.3 Why it fails on stock SDAR. SDAR's training loss is computed only at [MASK] positions (official training code,
`forward_add_noise_packed` + `logits_to_keep`; paper: "loss only on tokens replaced by the absorbing state"). The logit
at a FILLED position is never supervised. So when a draft token is placed in the block, the stock model's output at that
slot is an untrained quantity; in practice it echoes the input (Stage 1 "self-confirmation": decoding from the draft
failed). This is the precise reason the mentor is right that the model must be aligned: no threshold or vote on the
stock model can read a draft-filled slot, because nothing was ever trained there.

Training-free escape hatch that stays in-distribution: fill half the slots with draft tokens and mask the other half,
twice with the halves swapped ("complementary halves"). Every judged slot is a masked slot (trained regime) whose
context contains draft tokens at the other half. Two forwards give the stock model's conditional judgement of every
draft token. This is (a) the probe of the mechanism, (b) a training-free baseline, and (c) the generator of
draft-anchored training targets (section 3.3).

1.4 Why unique experts per forward are what they are. At block 32 each forward routes 32 positions x top-8 = 256 draws
into 128 experts; the union is ~58 (measured 58.4; TEAM reports 59.1). Committed slots are recomputed every pass.
Two levers exist: fewer positions routed per forward (TEAM's delayed cache) and a per-forward pool that all positions
must route within (DES-Vote, dMoE). The draft matters for the pool: a pool voted from masked slots reflects what MASKED
tokens want (TEAM: masked-token routing is concentrated, 0.86 cosine similarity between masked tokens), not what the
final tokens want; Stage 0 measured recall of the finished block's expert set 0.37 from masks vs 0.77-0.87 from a
draft-filled block (mismatched drafter checkpoint; to be redone in [AB]). A draft-informed pool can therefore be
smaller at the same coverage. And every pass saved saves ~58 expert loads per layer per block.

1.5 Cost of the drafter, first principles. Weight bytes per forward: MoE with 58 unique experts/layer
= 58 x 48 x 3 x 768 x 2048 x 2 B = 26 GB + ~3 GB dense; drafter 1.7B = 3.4 GB. One drafter forward ~ 1/8.5 of an MoE
forward in memory traffic; 4 drafter steps ~ 0.47 MoE forwards, 8 steps ~ 0.94. So a draft is worth it only if it
removes more than ~1 MoE pass per block (of 9.45) -- the target is to remove 5-6. [AB] will replace this with measured ms.

## 2. Literature map (what exists, what is new)

| Piece | Closest prior work | What they do | Gap we fill |
|---|---|---|---|
| Draft as initialisation for a dLLM | Speculative Correction (2608.02625), PSD (2605.15609), APD (2506.00413) | training-free; full-response refinement; LLaDA2.1 | trained block-level alignment on a model whose loss never trained filled slots; MoE |
| Trained step reduction | dParallel (2509.26488, ICLR26): certainty-forcing distillation, LoRA, own trajectories; Learn2PD (2509.25188): learned commit filter | no draft, dense models | draft-conditioned states; the control arm T0 reproduces dParallel on SDAR-b32 |
| Correcting filled tokens | GIDD (2503.04482), Corrective DLMs (2512.15596), ProSeCo (2602.11590), Remask-don't-replace (2604.18738) | uniform or self-sampled corruption | corruption = the real drafter's errors (train/inference aligned, as Spec-AUF 2607.01893 argues for drafters) |
| Fewer experts per forward in MoE dLLMs | DES (2602.00879), TEAM (2602.08404), dMoE (2605.30876), REFLEX (2608.01784), EC-DLM (2604.01622) | DES/TEAM training-free; dMoE trains routers for a block-level top-p pool (69.5 -> 16.5 experts, 99.55%) | pool voted from a draft-filled block; pool restriction trained jointly with draft alignment on SDAR |
| Draft model predicts target experts | SP-MoE (2510.10302), SPICE (2608.21240), APEX, ExpertFlow, MoE-SpeQ | AR MoE, prefetch/offload | 32 simultaneous positions, restriction not prefetch |

Strongest published baseline on our exact model and regime: TEAM (training-free, A100): GSM8K 90.60 -> 90.30,
experts/step 59.11 -> 36.20, tokens/step 3.16 -> 4.79, experts per token 18.71 -> 7.56, 1.83x (1.94x average).

## 3. The method: Draft-Aligned SDAR (DA-SDAR)

3.1 Inference (DA-decode), per block, after the prompt prefill:
1. Drafter (1.7B-b32) generates the block given the committed prefix (its own KV cache is kept in sync by flushing the
   MoE's committed blocks into it): K denoising steps (K = 4 or 8, or dynamic .90), outputs D[32] and p_draft[32].
2. MoE judge pass: input = prefix cache + block with D at all open slots (no masks). Output per slot i: argmax a_i,
   p_i. Commit slot i to a_i if p_i > tau. Uncommitted slots: keep D_i as tentative context if a_i == D_i, else re-mask.
3. Repeat the judge pass on the new state (committed / tentative / masked) until all slots are committed; at least one
   commit per pass (highest p among open slots), as in SDAR.
4. Flush forward (store_kv) as in SDAR; drafter cache flushed with the same tokens.
Expert pool (K2): in every judge pass, per layer, pool = top-M experts by the sum over block positions of the router
softmax (dMoE rule; DES-Vote's top-8-weight rule as the alternative); every position routes top-8 within the pool.
Optional controller (K3): a small MLP on [p_i, p_i(D_i), a_i == D_i, p_draft_i, drafter entropy, slot index, pass index]
replaces the fixed threshold for the commit decision (Learn2PD-style, trained on DA-decode states with the label
"a_i equals the target").

3.2 What is trained (MoE side). LoRA (rank 32) on q,k,v,o of all 48 layers; the 48 router gates (`mlp.gate`,
128 x 2048 each, 12.6M parameters) fully; experts frozen; embeddings and lm_head frozen. No peft dependency needed
(manual LoRA wrappers); `labels=` of the remote class must NOT be used (it shifts by one); loss is computed from the
hidden states of the active block only, lm_head applied to those rows.

3.3 Training states and targets.
- Prompts: GSM8K train minus the 100 frozen train prompts; MATH train subset; MBPP train. Never GSM8K test, MBPP test,
  MATH-500.
- Trajectories: the training-free DA-decode (complementary halves, section 1.3) run on the stock MoE with the real
  drafter at the inference setting. It yields per block: the drafter's D, the accepted set, the final tokens T (the
  "draft-anchored targets": draft tokens the stock MoE accepted plus stock-denoised tokens elsewhere) and every
  intermediate state. Filter items to correct final answers where gold exists (dParallel does this).
  Fallback if [AB] shows the halves judgement is poor: T = the stock 32/.95 decode's own tokens (pure self-distillation).
- State sampling per training example (one active block, prefix clean): u ~ U(0,1) fraction of slots committed (drawn
  preferentially from slots where D == T, mirroring real trajectories); of the open slots, each holds D_i with
  probability 1 - m and [MASK] with probability m, m ~ U(0,1); with probability 0.2 the block is fully masked (stock
  regime, regularisation). Attention: prefix block-causal over clean tokens (exactly what the inference cache holds),
  active block bidirectional over prefix + itself. Single copy, no BD3-LM two-copy trick: the loss is only on the active
  block.
- Loss: CE(T_i) at every open slot (draft-filled and masked) + beta x entropy at open slots whose argmax already equals
  T_i (certainty forcing, dParallel: temperature 0.5, beta 2). Expert pool active during training with M sampled from
  {24, 32, 48, none} per step so one adapter serves several budgets (router gradients flow through the renormalised
  top-8 weights; selection itself is non-differentiable as in all MoE training).
- Optimiser AdamW, lr 2e-5 (LoRA) / 1e-5 (gates), cosine, batch 1 per GPU x 4 GPUs x grad-accum 8 = 32; 2 epochs over
  ~20k (item, block) examples; gradient checkpointing per layer implemented by us (remote code's is broken).
- Compute: MoE forward+backward on ~600-1100 tokens ~2-3 s in the per-expert loop -> ~10k optimiser steps on 4 GPUs
  ~ 8-10 h. Data generation: DA-decode on ~6k prompts ~ 55-60 GPU-h (14 h on GPUs 4-7; 7 h if 0-3 are granted).

3.4 Drafter side (Phase 2, optional): fine-tune the 1.7B-b32 drafter on the MoE's accepted outputs (knowledge
distillation on trajectories, as EAGLE/Medusa/Spec-AUF train drafters on the target's outputs) to raise per-slot
agreement; cheap (1.7B). Only after the MoE-side result exists, so that the two contributions are separable.

## 4. Experiments (pre-registered)

Baselines (ACTUAL, block 32, b32 model, frozen GSM8K-500 + MBPP + MATH-500): B0 stock 32/.95 (GSM8K-500 90.2%, 10.06
passes/block, 58.4 experts/layer-forward, 1164 expert-layer executions/token: done); B1 TEAM reproduced from its repo
on our items (else paper numbers); B2 faithful DES-Vote M=32 on b32 (never run on the correct checkpoint); B3
training-free DA-decode (halves) with the real drafter.
Arms (trained, identical data/steps/LoRA budget): T0 certainty forcing WITHOUT draft (dParallel on SDAR-b32; the
control that isolates the draft); T1 draft-aligned (states of 3.3, no pool); T1+K2 (pool in the loop, M in {24,32,48});
T1+K2+K3 (controller).
Metrics, every one paired per item with 95% bootstrap CI: accuracy; passes/block; MoE forwards/token (prefill +
denoise + flush); drafter forwards/token; unique experts per layer-forward; expert-layer executions per token; total
weight bytes per token (MoE + drafter, the hardware-neutral cost); wall ms/token in the reference loop for all arms.
Gates: (G1) accuracy within 1.5 points of B0 (CI lower bound > -3); (G2) MoE expert-layer executions per token
+ drafter bytes-equivalent <= 0.5 x B0 (<= 582); (G3) passes/block <= 5; (G4) T1 beats T0 by >= 20% in passes/block at
equal accuracy (otherwise the honest conclusion is "certainty forcing alone suffices on SDAR-b32", still publishable);
(G5) experts per token (TEAM's definition) < 7.56 including the drafter.

## 5. Where the adapter can and cannot help (honest)

Can: passes/block (the dominant lever at block 32: 9.45 -> target 3-5); the pass-1 pool (draft-informed votes); the
commit policy (controller). Cannot: the flush forward (1 per block, ~10%); unique experts below the pool size M
without quality loss; the drafter's own cost (must be < 1 MoE pass per block: use K = 4 if [AB] shows 4-step drafts
are nearly as good as 8-step); hard tokens where the drafter is wrong and the correction depends on other corrections
(those slots are re-masked and fall back to stock behaviour: no loss, no gain); wall-clock in the launch-bound reference
loop (expert-count savings do not translate 1:1 to ms there; a fused MoE kernel is needed for ms claims).

## 6. Decision points

D1 [AB probe, today]: draft-halves judgement must beat mask-init on argmax==final by a large margin (say +15 points)
and accept most correct draft tokens; draft-voted pools must cover R_final markedly better than mask-voted pools at
M = 24/32. If both hold, build DA-decode (training-free) and generate data. If only the first holds, drop K2. If
neither, the drafter at block 32 is too weak: report and stop this line.
D2 [B3 on the 100 train prompts]: training-free DA-decode accuracy vs stock and passes/block. Sets the anchored-target
quality and the training-free baseline.
D3 [T0 vs T1]: the draft's contribution.

## 7. [AB] Probe results (ACTUAL, 2026-09-06, 87 clean train items, 537 blocks, 15,782 generated slots; code/draftalign/probe_ab.py, logs/draftalign/ab_probe/)

Integrity: the mask-init forward reproduces the recorded stock pass-1 argmax at 15,782/15,782 slots. 13 of the 100
train items were excluded because their trace files had been overwritten by test items with the same dataset index
(defect in the earlier trace naming; the new runners namespace traces by split).

7.1 Drafter (1.7B-b32, block 32, teacher-forced prefix), agreement of its block with the MoE's final tokens:
4 static steps 0.497, 8 steps 0.558, dynamic .90 (10.6 steps) 0.586 (slot rates; item-mean CI for 8 steps
[0.485, 0.536]). Cost in the reference loop: 18 ms per drafter forward vs 369 ms per mask-init MoE forward and
~520 ms per draft-filled MoE forward. When the draft is wrong, the drafter gives the final token p ~ 0.12.

7.2 What the stock MoE does with a draft in the block (new mechanism, explains every earlier failure):
- Block filled with the FINAL tokens (P5): the output at slot i equals the token at slot i+1 in 90.4% of slots and the
  slot's own token in 10.0%. Filled with the draft (P1): 84.8% next-draft-token. The block-diffusion training never
  supervised filled slots, so the Qwen3 autoregressive next-token head survives there. A draft-in-place readout on the
  stock model is a next-token copy, not a judgement.
- Complementary halves (judged slot masked, draft at the other half; P2): confidence jumps (commit rate at .95: 0.26
  -> 0.90) but the argmax equals the DRAFT token in 92% of slots; the model confirms a WRONG draft token at > .95 in
  0.334 of all slots = 76% of the drafter's errors; it corrects a wrong draft in 1.2% of slots (mask-init: 14%).
  With the model's own incoherent pass-1 argmaxes as the fill (P4) confirmation is 0.066: it is the coherence of the
  draft that the model trusts. With the final tokens as fill (P6) it reproduces them at 0.997. So the stock masked
  conditional is a sycophantic verifier of any coherent context. Training-free draft-context judging is out.
- Blind marginal (the stock pass-1 forward itself, no draft in the input, zero extra forwards): probability q_i that
  the MoE's own marginal gives the drafter's token. Accepting draft tokens with q_i > gamma beyond the stock commits:
  gamma .3: +7.0 slots/block, 1.2 wrong per block; gamma .5: +4.7, 0.43 wrong per block; gamma .7: +3.1, 0.12.
  Precision of the ACCEPTED extra slots (draft == final), slot-level over the 87 items: gamma .3 0.83, .5 0.908, .7 0.96
  (the earlier 0.914/0.963/0.988 were over ALL slots above the cut including the stock's own commits; corrected
  2026-09-07 from the probe records, script: results/figures/ or analyze_ab.py derivation). Stock pass 1 commits 7.7 of 29.4 slots. This is a non-sycophantic acceptor (the
  marginal never saw the draft) and is the training-free baseline B3 ("DV-blind"; APD-style rule: the small model's
  coherent joint picks among tokens the large model's marginal already finds plausible).
7.3 Experts. Unique experts per layer-forward: mask-init 43.1, draft-filled 64.5, final-filled 64.4. Coverage of the
final tokens' (position, expert) picks by a top-M pool voted (DES-Vote rule) from: masks / draft-filled / final-filled:
M16 .32/.54/.59, M24 .43/.66/.72, M32 .51/.75/.82, M48 .63/.88/.94; paired draft-vs-mask at M32 +0.24 [+0.23, +0.25].
The draft recovers ~80% of the coverage gap between mask votes and the oracle. Unique-expert recall of the final
forward's set: masks 0.49, draft 0.89.

7.4 Decision D1: GO on both mechanisms, with two changes to section 3:
(a) targets: the halves-anchored targets are withdrawn (sycophancy). Targets come from the DV-blind decode (accepted
    draft tokens where the MoE's own marginal agrees, stock elsewhere), filtered to correct answers; fallback = stock.
(b) the trained judge must repurpose filled slots from "next token" to "corrected current token" (LoRA on attention +
    the corruption process of 3.3); the certainty term only applies once the judge is right.
Prior-art update: SimSD (2606.02544) uses SDAR-1.7B to draft for SDAR-8B, training-free, with a paired data/mask
layout and temporal-causal attention, prefix speculative sampling at small block widths, 7.5x reported. PSD (2605.15609)
self-drafts with per-position confidence acceptance at block 32. Neither trains the target, treats an MoE, or uses
per-slot (non-prefix) acceptance of an external draft at block 32; our B3 is the closest training-free point.

7.5 [ACTUAL] Training-free draft-verify decodes on the 100 train prompts (code/draftalign/run_da_decode.py; stock twin
reproduces the recorded stock cell on 100/100 items: identical generations, passes, commits and forward counts):
- Halves judge (draft context, 2 judge forwards per block, commit > .95, accept draft if agree & p > .70 or q > .50):
  accuracy 67.0% vs stock 95.0%, paired -28.0 [-37.0, -20.0]; 2.01 passes/block (stock 9.31); 107 ms/token (stock
  187); the output agrees with the drafter's tokens at 0.917. The stock model adopts the draft wholesale. Closed.
- Blind-marginal judge (accept draft tokens the MoE's own pass-1 marginal gives q > gamma; zero extra MoE forwards):
  ACTUAL (100 train prompts, paired vs stock 95.0%): gamma .50: accuracy 95.0 (every item's correctness unchanged),
  7.93 passes/block (stock 9.31, -15%), MoE expert-layer executions/token 963 (stock 1091, -12%), drafter 0.338
  forwards/token (6 ms/token), wall 172 vs 187 ms/token (1.09x in the launch-bound loop); accepts 4.49 draft
  slots/block (15.3% of tokens). gamma .30: 88.0 (-7.0 [-12, -3]), 6.98 passes, 852 exec/token: loses, as the probe's
  1.2 wrong accepts/block predicted. Decision D2: gamma .50 is the accuracy-neutral training-free point (B3) and the
  target generator. Refinements (ACTUAL): gamma .40 = 94.0 (-1.0 [-3, 0]), 7.74 passes: no gain; gamma .50 with
  re-judging at every later pass = 91.0 (-4.0 [-9, 0]), 6.62 passes: later-pass acceptance hurts. FINAL training-free
  point B3 = s8 draft, blind gamma .50, no re-judge (frozen GSM8K-500 cell running).
7.6 Trainer sanity (ACTUAL, GPU, results/audits/draftalign/sanity_2026-09-06.json): the custom training forward
reproduces the stock forward on a real item at S=640 with max |dlogit| 0.0 and identical argmax, with and without
layer checkpointing; one AdamW step: loss 1.389 (CE 1.381, certainty 0.004, open-slot argmax accuracy 0.78), only
LoRA-B and gate tensors move (LoRA-A has zero gradient at init by construction), no base tensor changes or receives a
gradient; 5.3 s per step, peak 57.7 GiB on one GPU.

7.7 [ACTUAL] Training-free DV-blind (s8, gamma .50) on the frozen GSM8K-500 (paired vs the recorded stock cell):
accuracy 90.4 vs 90.2 (+0.2 [-1.4, +1.8], n = 500); passes/block 8.49 vs 10.06 (-16%); MoE forwards/token 0.357 vs
0.415; MoE expert-layer executions/token 1,017 vs 1,164 (-13%); drafter 0.333 forwards/token (6 ms/token);
wall 184 vs 198 ms/token (1.08x, launch-bound loop); 4.47 draft slots accepted per block (15% of tokens). Held-out
confirmation of the train-prompt result: the training-free baseline B3 is accuracy-neutral and modest.
