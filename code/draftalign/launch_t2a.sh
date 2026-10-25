#!/bin/bash
# Round 2a: redesigned judge (draft flag + gated LoRA), from scratch, same data/budget as round 1. 8 GPUs.
set -eu; cd "$(dirname "$0")/../.."
mkdir -p logs/draftalign/train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup /venv/main/bin/torchrun --standalone --nproc_per_node=8 code/draftalign/train.py \
  --index logs/draftalign/trainset_gsm8k_r1/index.jsonl --out logs/draftalign/train/t2a_gsm8k_flag \
  --steps 1057 --grad-accum 4 --pool-Ms 128 --no-train-gates --draft-flag --p-draftfull 0.3 --p-allmask 0.0 \
  --min-lr-frac 0.5 --max-seq 1200 --holdout-items 100 --eval-every 100 --save-every 200 --log-every 10 --seed 0 \
  --keep-checkpoints 2 > logs/draftalign/train/t2a_gsm8k_flag.log 2>&1 &
echo "T2a launched: $(date -u +%FT%TZ)" >> logs/draftalign/orchestrate_round1.log
