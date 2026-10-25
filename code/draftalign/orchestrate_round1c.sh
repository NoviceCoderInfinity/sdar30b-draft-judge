#!/bin/bash
# Round 1, part c (replaces 1b): wait for T1 -> evaluate T1 on calibvalid (full judge, with/without pool) ->
# start the round-2 GPU queue (data generation: remaining GSM8K, MATH, MBPP; baselines) IMMEDIATELY, T0 control later.
set -u
cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log
PY=/venv/main/bin/python
T1=logs/draftalign/train/t1_gsm8k_r1
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator 1c started (pid $$): waiting for T1; after its evals the round-2 queue starts (T0 deferred)"
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
cp logs/draftalign/queue_round2.txt logs/draftalign/queue.txt
nohup code/draftalign/gpu_queue.sh logs/draftalign/queue.txt logs/draftalign/queue.log > /dev/null 2>&1 &
log "round-2 queue started (data generation first; T0 control to be launched manually after round-2 data)"
