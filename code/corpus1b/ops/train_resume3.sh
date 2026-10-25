#!/usr/bin/env bash
# after the 13:45 IST stop: the same run resumes from its newest checkpoint (the domain-weighted plan continues at
# its own step) and trains to 14:45 IST; logs/c1b/train/main/STOP_AT (unix time, re-read every step) moves the stop
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home TOKENIZERS_PARALLELISM=false
/venv/train/bin/torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
  --index logs/c1b/trainset/waveA/index.jsonl --out logs/c1b/train/main --lr 8e-5 --max-tokens 74759690 --epochs 1 \
  --stop-at 1790673300 --stop-at-file logs/c1b/train/main/STOP_AT --stop-file logs/c1b/train/main/STOP \
  --save-every-tokens 1.8e6 --keep-rolling 1000 --resume \
  --domain-weights maths=25,code=25,science=15,if=15,chat=20 --repeat-domains science,if --max-passes 4 \
  >> logs/c1b/train/main.log 2>&1
echo "[train_resume3] exit $? $(date -u +%H:%M:%S)" >> logs/c1b/train/main.log
