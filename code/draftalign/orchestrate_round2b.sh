#!/bin/bash
# Round 2b: wait for the round-2 data (MATH 8, MBPP 2, GSM8K-rest 8 shards) and for the GPU queue to drain, build the
# union index, launch T2b (resume 2a) on 8 GPUs, then evaluate it four ways on the 100 train prompts.
set -u; cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log; PY=/venv/main/bin/python
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
D=logs/draftalign/datagen
log "orchestrator 2b started (pid $$): waiting for round-2 data and an empty queue"
until [ "$(ls $D/math_pool_0_7500_3_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 8 ] && [ "$(ls $D/mbpp_pool_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 2 ] && [ "$(ls $D/gsm8k_pool_3000_7273_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 8 ]; do sleep 300; done
log "round-2 data complete"
until [ ! -f logs/draftalign/queue.txt ] && [ "$(pgrep -fc 'run_da_[d]ecode.py')" -eq 0 ]; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
log "queue drained; building the union index"
code/draftalign/build_union_index.sh >> "$LOG" 2>&1 || { log "union index FAILED"; exit 1; }
code/draftalign/launch_t2b.sh 2000
T2=logs/draftalign/train/t2b_union_flag; sleep 300
while true; do
  [ -d $T2/step003057 ] && [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && break
  if [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ]; then sleep 120; [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ] && { log "T2b FAILED: trainer gone without the final checkpoint"; exit 1; }; fi
  sleep 60
done
log "T2b finished"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
CK=$T2/step003057
wave(){ for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK --judge full --draft s8 $2 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t2b_$1.shard${k}of4.log 2>&1 &
    g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK --judge full --draft s8 $4 --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t2b_$3.shard${k}of4.log 2>&1 &
  done; sleep 120; until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 20; done; }
log "T2b evals wave 1: c95 and c90/o99"; wave c95 "" c90o99 "--tau-commit .90 --tau-override .99"
log "T2b evals wave 2: c95/o99 and c90"; wave c95o99 "--tau-override .99" c90 "--tau-commit .90"
log "T2b evals finished (READY: analyse, then frozen test + T0 control)"
