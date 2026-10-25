#!/bin/bash
cd /workspace/diffusion-moe-general-expert
export HF_HOME=/workspace/diffusion-moe-general-expert/hf_home HF_HUB_OFFLINE=1
/venv/p2/bin/torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_rec.py \
  --states logs/phase5/judge_states/rec_main --index logs/c1b/trainset/fresh/train_index.jsonl \
  --out logs/phase5/judge/rec_main --continue-from logs/c1b/train/main_128m/tok_000128000000 \
  --max-tokens 160e6 --save-at-tokens 136e6,144e6,152e6,160e6 --step0-states 256 --step0-min-agree 0.85
