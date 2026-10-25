#!/bin/bash
# Round 1: wait for the GSM8K data shards -> build the train set -> T1 (draft-aligned) on 8 GPUs -> T0 (no-draft
# control) on 8 GPUs -> start the round-2 GPU queue. Logs to logs/draftalign/orchestrate_round1.log.
set -u
cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log
DG=logs/draftalign/datagen/gsm8k_pool_0_3000_dv-s8_blind_q50
PY=/venv/main/bin/python; TR=/venv/main/bin/torchrun
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator started (pid $$)"
until [ "$(ls $DG/*.shard*of8.jsonl.done 2>/dev/null | wc -l)" -ge 8 ]; do sleep 120; done
log "all 8 GSM8K data shards done"
# wait until every GPU is free (other cells finish on their own; the live queue only held GSM8K shards)
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 60; done
log "all GPUs free"
IDX=logs/draftalign/trainset_gsm8k_r1
$PY code/draftalign/build_train_set.py --records $DG --traces $DG/traces --layout da --split-name "gsm8k_pool:0:3000" \
    --out $IDX --only-correct --max-seq 1200 --exclude-frozen >> "$LOG" 2>&1 || { log "build_train_set FAILED"; exit 1; }
NB=$($PY - <<PY
import json; s=json.load(open("$IDX/build_summary.json")); print(int(s["summary"]["usable_blocks"]))
PY
)
STEPS=$(( (2 * NB + 31) / 32 )); [ "$STEPS" -lt 100 ] && STEPS=100
log "train set built: usable blocks $NB -> $STEPS optimiser steps (2 epochs, global batch 32)"
COMMON="--index $IDX/index.jsonl --steps $STEPS --grad-accum 4 --pool-Ms 24,32,48,128 --max-seq 1200 --holdout-items 100 --eval-every 100 --save-every 200 --log-every 10 --seed 0"
log "launch T1"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $TR --standalone --nproc_per_node=8 code/draftalign/train.py $COMMON \
    --out logs/draftalign/train/t1_gsm8k_r1 > logs/draftalign/train/t1_gsm8k_r1.log 2>&1
log "T1 finished (exit $?)"
log "launch T0 (no draft: --p-allmask 1.0)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $TR --standalone --nproc_per_node=8 code/draftalign/train.py $COMMON --p-allmask 1.0 \
    --out logs/draftalign/train/t0_gsm8k_r1 > logs/draftalign/train/t0_gsm8k_r1.log 2>&1
log "T0 finished (exit $?)"
cp logs/draftalign/queue_round2.txt logs/draftalign/queue.txt
nohup code/draftalign/gpu_queue.sh logs/draftalign/queue.txt logs/draftalign/queue.log > /dev/null 2>&1 &
log "round-2 queue started"
