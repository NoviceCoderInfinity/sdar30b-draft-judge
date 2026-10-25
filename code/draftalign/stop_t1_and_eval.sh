#!/bin/bash
# Stop T1 at its step-600 checkpoint, keep only the last two checkpoints, evaluate T1@600 on the 100 train prompts
# (full judge on GPUs 0-3, full judge + pool 32 on GPUs 4-7) from the frozen worktree /workspace/dlm-exp1-eval.
set -u
MAIN=/workspace/dlm-exp1-voting; EV=/workspace/dlm-exp1-eval; LOG=$MAIN/logs/draftalign/orchestrate_round1.log
T1=$MAIN/logs/draftalign/train/t1_gsm8k_r1
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "stop_t1_and_eval started (pid $$): waiting for step000600"
until [ -f $T1/step000600/trainer_state.pt ] && [ "$(cat $T1/latest 2>/dev/null)" = "step000600" ]; do sleep 20; done
sleep 30
TR=$(pgrep -f "^/venv/main/bin/python3 /venv/main/bin/torchrun" | head -1)
log "step000600 saved; stopping T1 (torchrun pid $TR)"
[ -n "$TR" ] && kill $TR; sleep 15; pkill -f "train.py --index" 2>/dev/null; sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 10; done
rm -rf $T1/step000200; log "T1 stopped; checkpoints kept: $(ls $T1 | grep step | tr '\n' ' ')"
cd $EV
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup /venv/main/bin/python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $T1 --judge full --draft s8 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t1_600_full.shard${k}of4.log 2>&1 &
  g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup /venv/main/bin/python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $T1 --judge full --draft s8 --pool 32 --pool-rule full --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t1_600_full_p32.shard${k}of4.log 2>&1 &
done
log "T1@600 evals launched from $EV"
sleep 120
until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 20; done
log "T1@600 evals finished; results under $EV/logs/draftalign/da_decode/"
