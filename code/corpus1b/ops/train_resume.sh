#!/usr/bin/env bash
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home TOKENIZERS_PARALLELISM=false
/venv/train/bin/torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
  --index logs/c1b/trainset/waveA/index.jsonl --out logs/c1b/train/main --lr 8e-5 --max-tokens 74759690 --epochs 1 \
  --stop-at 1790669700 --stop-file logs/c1b/train/main/STOP --save-every-tokens 2e6 --keep-rolling 1000 --resume \
  >> logs/c1b/train/main.log 2>&1
echo "[train_resume] exit $? $(date -u +%H:%M:%S)" >> logs/c1b/train/main.log
