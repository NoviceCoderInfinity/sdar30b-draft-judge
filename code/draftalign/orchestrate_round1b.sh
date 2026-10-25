#!/bin/bash
# Round 1, part b (replaces the tail of orchestrate_round1.sh while T1 is running):
# wait for T1 -> evaluate T1 on the 100 train prompts (full judge, with and without the expert pool, 4 GPUs each)
# -> T0 control on 8 GPUs -> start the round-2 GPU queue. Log: logs/draftalign/orchestrate_round1.log
set -u
cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log
PY=/venv/main/bin/python; TR=/venv/main/bin/torchrun
T1=logs/draftalign/train/t1_gsm8k_r1; IDX=logs/draftalign/trainset_gsm8k_r1
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator 1b started (pid $$): waiting for T1"
while true; do
  [ -d $T1/step001057 ] && [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && break
  if [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T1/step001057 ]; then sleep 120; [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T1/step001057 ] && { log "T1 FAILED: trainer gone without the final checkpoint"; exit 1; }; fi
  sleep 60
done
log "T1 finished (final checkpoint present)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
log "launch T1 evals on calibvalid: full judge (GPUs 0-3) and full judge + pool 32 (GPUs 4-7)"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $T1 --judge full --draft s8 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t1_full.shard${k}of4.log 2>&1 &
  g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $T1 --judge full --draft s8 --pool 32 --pool-rule full --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t1_full_p32.shard${k}of4.log 2>&1 &
done
sleep 120
until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 30; done
log "T1 evals finished"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
COMMON="--index $IDX/index.jsonl --steps 1057 --grad-accum 4 --pool-Ms 24,32,48,128 --max-seq 1200 --holdout-items 100 --eval-every 100 --save-every 200 --log-every 10 --seed 0"
log "launch T0 (no draft: --p-allmask 1.0)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $TR --standalone --nproc_per_node=8 code/draftalign/train.py $COMMON --p-allmask 1.0 \
    --out logs/draftalign/train/t0_gsm8k_r1 > logs/draftalign/train/t0_gsm8k_r1.log 2>&1
log "T0 finished (exit $?)"
cp logs/draftalign/queue_round2.txt logs/draftalign/queue.txt
nohup code/draftalign/gpu_queue.sh logs/draftalign/queue.txt logs/draftalign/queue.log > /dev/null 2>&1 &
log "round-2 queue started"
