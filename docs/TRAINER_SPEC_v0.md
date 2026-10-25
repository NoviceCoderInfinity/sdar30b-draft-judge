# Trainer spec v0: draft-aligned LoRA for SDAR-30B-A3B-Chat-b32 (block 32)

## Model surgery (code/draftalign/adapter.py)
- Manual LoRA (no peft): wrap q_proj,k_proj,v_proj,o_proj of all 48 layers: y = W x + (alpha/r) B(A x), r=32, alpha=64, A kaiming, B zero; params fp32, matmul under bf16 autocast. Save/load as safetensors: `save_adapter(model, path)`, `load_adapter(model, path)`; `merge=False` (keep wrappers at inference; or merge into W for speed - both).
- Router gates `mlp.gate` (Linear 2048->128, 48 layers): trainable copies in fp32, written back to bf16 for inference; saved alongside.
- Everything else frozen (`requires_grad=False`), model in bf16.

## Training forward (code/draftalign/train_forward.py)
- Input: one sequence [prompt ; finished blocks ; active block] on the absolute 32-grid, S = hi of the active block. Attention = block_tril over the grid (bool [1,1,S,S], or the [1,S,S] 0/1 form decode.py uses) => every block attends bidirectionally to itself and all earlier blocks = exactly the inference cache semantics (prefill block-tril; block forwards over cache+block). position_ids = arange(S). use_cache=False, store_kv irrelevant.
- Gradient checkpointing: our own wrapper around each decoder layer (remote code's GC is broken: undefined causal_mask, wrong positional args). Re-implement `SDARMoeModel.forward` minimal (embed -> rotary -> layers with checkpoint -> norm) mirroring the remote file; assert parity with the stock forward on a CPU fake / on GPU with a tiny input.
- MoE layer forward for training: differentiable hit-only expert loop (unique experts of the rows; per expert gather rows, run gate/up/down, scatter_add weighted). Router: fp32 softmax -> [optional POOL restriction on the active-block rows only] -> top-8 -> renormalise (norm_topk_prob=True). Pool: votes = sum over active-block rows (valid slots) of the fp32 softmax (dMoE rule; DES top-8-weight rule as option); top-M pool; block rows route only within the pool (logits outside -> -inf before top-k). Prefix rows route unrestricted. M sampled per step from {24, 32, 48, None} (config).
- Loss rows: active-block open slots (draft-filled + masked). hidden -> lm_head (fp32 logits on those rows only) -> CE(T_i) mean over rows + beta * mean entropy(softmax(logits/temp)) over rows where argmax == T_i (certainty forcing; beta=2, temp=0.5, dParallel). MASK logit suppressed in the entropy term. Optional label smoothing 0.

## Data (code/draftalign/build_train_set.py)
- Source records: DA-decode / stock trajectories (jsonl + trace npz: tokens[b,32], valid[b,32], committed[b,p,32], draft_tokens[b,32]); prompts from GSM8K train (exclude the 100 frozen), MATH train subset, MBPP train. Filter to correct final answers where gold exists.
- Example = (item, block b). State sampler at train time: with prob 0.4 pass-1 state (nothing committed); else pick a recorded pass p uniformly and use committed[p] (realistic committed sets); open slots: each holds D_i w.p. (1-m) and MASK w.p. m, m ~ U(0,1); with prob 0.2 the whole block is masked (stock regime). Targets T = tokens[b]. Prefix = prompt + tokens of blocks < b (clean).
- Also emit the K3 controller dataset (per slot features + label argmax==T) from the same forwards later.

## Optimiser / schedule (code/draftalign/train.py)
- torchrun 4 procs (GPUs 4-7), one full model per GPU (61 GB bf16 + activations; check peak), batch 1 per step per GPU, grad-accum 8 => effective 32 examples/step; AdamW lr 2e-5 (LoRA) / 1e-5 (gates), wd 0, cosine to 10%, warmup 100 steps; 2 epochs over ~20k examples => ~1250 optimiser steps; log loss / certainty / pass-1 argmax accuracy on a held-out 1k examples every 100 steps; save adapter every 250 steps.
- Sequence cap 1200 tokens (prompt + generated); drop longer.
- Sanity gates before the real run (CPU/GPU, no pilot on the science): (1) custom forward == stock forward logits on the same input (bit-close in bf16) with the pool disabled and LoRA at B=0; (2) loss finite; (3) one optimiser step changes only LoRA/gate params; (4) throughput measured.

## Inference with the adapter (extend code/draftalign/da_decode.py)
- `--adapter path`: load LoRA + gates; `--judge full`: one forward with every open slot draft-filled (the trained regime) instead of halves; commit a_i if p_i > tau; open slots with a_i == D_i keep D_i as tentative context, else MASK; repeat judge passes (each pass is a full-fill forward over the current state) until all committed; >= 1 commit per pass; flush. `--pool M`: pool restriction on every block forward (judge, denoise; not flush/prefill - or also flush? measure both).
- Report the same metrics as the training-free runner + drafter cost.
