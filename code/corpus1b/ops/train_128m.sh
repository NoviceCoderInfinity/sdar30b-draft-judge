#!/usr/bin/env bash
# Continue the Phase-1 judge run 28.5M -> 128M (user 2026-09-29, same recipe): starts main_128m from main/final with
# --continue-from; relaunches after a crash with --resume (newest checkpoint in main_128m). Safety stop 02:20 IST 1 Oct.
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home TOKENIZERS_PARALLELISM=false
RES=${1:-}   # "resume" on relaunch
FROM="--continue-from logs/c1b/train/main/final"; [ "$RES" = resume ] && FROM="--resume"
/venv/train/bin/torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
  --index logs/c1b/trainset/index_128m.jsonl --holdout-pids logs/c1b/trainset/holdout_pids_waveA.txt \
  --out logs/c1b/train/main_128m $FROM --lr 8e-5 --max-tokens 128e6 --epochs 1 \
  --stop-at 1790801400 --stop-at-file logs/c1b/train/main_128m/STOP_AT --stop-file logs/c1b/train/main_128m/STOP \
  --save-every-tokens 2e6 --keep-rolling 1000 \
  --domain-weights maths=25,code=25,science=15,if=15,chat=20 --repeat-domains science,if --max-passes 4 \
  >> logs/c1b/train/main_128m.log 2>&1
echo "[train_128m] exit $? $(date -u +%H:%M:%S)" >> logs/c1b/train/main_128m.log
