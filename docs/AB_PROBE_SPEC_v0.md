# AB probe spec v0 (inference ablation on the stock models, block 32, b32 checkpoints)

Purpose: measure, on the CORRECT checkpoints, the two mechanisms the trained design relies on, before any training:
 (M1) a draft-filled block turns flat masked marginals into peaked conditionals -> more positions judged correctly / committable per pass;
 (M2) a coreset voted from a draft-filled block covers the FINAL tokens' expert picks far better than one voted from masks -> a small per-forward expert pool is viable.
Also produces the drafter cost model and the drafter's per-position agreement at block 32.

Data: the 100 frozen train prompts (calib 50 + valid 50), and the b32 MoE's recorded stock 32/.95 trajectories
(logs/research_v1/baselines_b32_traces/gsm8k_B32_t95_stock/<idx>.npz with keys proposed[b,p,32], confidence, committed, valid[b,32], tokens[b,32], n_pass, block_start, prompt_length; jsonl records have gen_ids and prompt ids via research_protocol.encode_prompt). Never the 500 test items.

Models: MoE JetLM/SDAR-30B-A3B-Chat-b32 rev c351bbc (research_protocol.MODELS["b32"]); drafter JetLM/SDAR-1.7B-Chat-b32 rev 2c83c05d84be90a01c72443b2123ebf0376f6fd4 in hf_cache/hub (add MODELS["d32"]). bf16, sdpa, batch 1.

Per block b of each item (prefix = prompt + MoE's finished blocks < b, teacher-forced from the trace; final tokens F = tokens[b]; valid slots V):
 Drafter: block-32 generation of this block given the prefix, three settings: static 4 steps (8 tokens/step), static 8 steps, dynamic thr .90 (>=1/step, cap 32). Record draft tokens D_K, per-position prob at commit time, number of drafter forwards, ms per forward.
 MoE forwards (all with the same prefix cache; record per position: argmax, top-1 prob, and per layer the top-8 expert ids of every block position):
  P0 mask-init (all 32 [MASK]) -- must reproduce the recorded pass-1 proposals (assert equality on a sample: integrity check);
  P1 draft-full: slots = D_8;                       P3 self-draft-full: slots = recorded pass-1 proposals;
  P2 draft-halves: 2 forwards (even slots masked + odd = D_8; odd masked + even = D_8);  P4 self-draft-halves likewise;
  P5 oracle-full: slots = F (also defines the FINAL routing sets R_final per position/layer);  P6 oracle-halves.
 Metrics per forward kind, over valid slots: argmax==F rate; mean top-1 prob; commit rate at .95 and .90; for halves: read only masked slots; draft acceptance = (argmax == D) rate and p(D) distribution; self-confirmation = conf>.95 & argmax==D & D!=F (for full-fill kinds); correction = D!=F & argmax==F.
 Expert metrics per layer: unique experts in the forward; coreset coverage: vote v_e = sum over block positions of router softmax prob for e (dMoE/DES-Vote style); top-M pool for M in {16,24,32,48}; coverage = fraction of (position, expert) picks in R_final that lie in the pool; report for init in {mask, draft-full, draft-halves union, self-draft-full, oracle-full}. Also the plain recall of unique-expert sets vs R_final's union.
 Timing: ms per MoE forward and per drafter forward (CUDA-synchronised), for the cost model.

Outputs: per-block JSONL (all scalars + per-position arrays as lists) under logs/draftalign/ab_probe/<shard>.jsonl; analysis script prints the tables with item-paired bootstrap CIs over the 100 items.
Run: 4 shards on GPUs 4-7, one process each; launch logged through research_protocol.LaunchLog.
