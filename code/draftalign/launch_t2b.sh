#!/bin/bash
# Round 2b: resume the round-2a adapter on the union set (GSM8K + MATH + MBPP), same mechanism, longer. 8 GPUs.
# STEPS_TOTAL = 1057 (already done) + new steps; pass the new-step count as $1 (default 2000).
set -eu; cd "$(dirname "$0")/../.."
NEW=${1:-2000}; TOTAL=$((1057 + NEW))
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup /venv/main/bin/torchrun --standalone --nproc_per_node=8 code/draftalign/train.py \
  --index logs/draftalign/trainset_union_r2/index.jsonl --out logs/draftalign/train/t2b_union_flag \
  --resume logs/draftalign/train/t2a_gsm8k_flag/step001057 --steps $TOTAL --grad-accum 4 --pool-Ms 128 --no-train-gates \
  --draft-flag --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 --max-seq 1200 --holdout-items 150 \
  --eval-every 100 --save-every 200 --log-every 10 --seed 1 --keep-checkpoints 2 > logs/draftalign/train/t2b_union_flag.log 2>&1 &
echo "$(date -u +%FT%TZ) T2b launched (resume 2a@1057, +$NEW steps -> $TOTAL)" >> logs/draftalign/orchestrate_round1.log
