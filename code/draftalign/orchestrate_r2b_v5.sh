#!/bin/bash
# Round 2b, v5 = the tail of v4 after the container restart of 2026-09-08 (low balance): wait for T2b (relaunched by the
# monitoring session from step002400) to reach step003057, pause the queue, run the four paired calibvalid evals on 8 GPUs, resume the queue.
# agreement (GPU 0) when it finishes; evaluate T2b four ways when it finishes. Log: logs/draftalign/orchestrate_round1.log
set -u; cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log; PY=/venv/main/bin/python; TR=/venv/main/bin/torchrun
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator r2b-v5 started (pid $$): waiting for T2b step003057 (resumed from step002400 after the 2026-09-08 container restart)"
# T2b: wait, then the four evals on 8 GPUs
T2=logs/draftalign/train/t2b_union_flag
while true; do
  [ -d $T2/step003057 ] && [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && break
  if [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ]; then sleep 120; [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ] && { log "T2b FAILED: trainer gone without the final checkpoint"; exit 1; }; fi
  sleep 60
done
log "T2b finished; pausing the queue for the T2b evals"
[ -f logs/draftalign/queue.txt ] && mv logs/draftalign/queue.txt logs/draftalign/queue.paused
until [ "$(pgrep -fc 'gpu_q[u]eue.sh')" -eq 0 ]; do sleep 10; done
pkill -f "run_da_decode.py --model b32" 2>/dev/null; sleep 20
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
CK2=$T2/step003057
wave(){ for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK2 --judge full --draft s8 $2 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t2b_$1.shard${k}of4.log 2>&1 &
    g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK2 --judge full --draft s8 $4 --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t2b_$3.shard${k}of4.log 2>&1 &
  done; sleep 120; until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 20; done; }
log "T2b evals wave 1: c95 and c90/o99"; wave c95 "" c90o99 "--tau-commit .90 --tau-override .99"
log "T2b evals wave 2: c95/o99 and c90"; wave c95o99 "--tau-override .99" c90 "--tau-commit .90"
log "T2b evals finished (READY)"
if [ -f logs/draftalign/queue.paused ]; then mv logs/draftalign/queue.paused logs/draftalign/queue.txt; nohup code/draftalign/gpu_queue.sh logs/draftalign/queue.txt logs/draftalign/queue.log > /dev/null 2>&1 & log "queue resumed on all GPUs"; fi
