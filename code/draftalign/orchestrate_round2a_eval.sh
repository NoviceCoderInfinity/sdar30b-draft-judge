#!/bin/bash
# After round 2a: evaluate the final adapter on the 100 train prompts (full judge; commit .90 ablation; re-mask-only
# ablation), then start the round-2 data-generation queue on all 8 GPUs. Log: logs/draftalign/orchestrate_round1.log
set -u
cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log; PY=/venv/main/bin/python
T2=logs/draftalign/train/t2a_gsm8k_flag
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator 2a-eval started (pid $$): waiting for T2a step001057"
while true; do
  [ -d $T2/step001057 ] && [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && break
  if [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step001057 ]; then sleep 120; [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step001057 ] && { log "T2a FAILED: trainer gone without the final checkpoint"; exit 1; }; fi
  sleep 60
done
log "T2a finished"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
CK=$T2/step001057
run_pair(){ # $1 = name, $2 = extra args for GPUs 0-3 variant, $3 = name2, $4 = extra args for GPUs 4-7 variant
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK --judge full --draft s8 $2 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t2a_$1.shard${k}of4.log 2>&1 &
    g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK --judge full --draft s8 $4 --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t2a_$3.shard${k}of4.log 2>&1 &
  done
  sleep 120
  until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 20; done
}
log "T2a evals: full judge (GPUs 0-3) and commit .90 (GPUs 4-7)"
run_pair full "" c90 "--tau-commit .90"
log "T2a evals: re-mask-only (GPUs 0-3) and commit .90 + re-mask-only (GPUs 4-7)"
run_pair remask "--full-remask-only" c90remask "--tau-commit .90 --full-remask-only"
log "T2a evals finished"
cp logs/draftalign/queue_round2.txt logs/draftalign/queue.txt
nohup code/draftalign/gpu_queue.sh logs/draftalign/queue.txt logs/draftalign/queue.log > /dev/null 2>&1 &
log "round-2 data queue started"
