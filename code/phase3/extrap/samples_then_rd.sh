#!/usr/bin/env bash
# Remaining 12-item baselines (today's judge = rd flags minus re-draft, 4 GPUs busy like rd), then restart rd in the
# short-first order. Same box, same load.
cd /workspace/dmge; ROOT=$PWD; ORDER=$(cat logs/phase3/extrap/order.txt); PY=/venv/p2/bin/python
AD=logs/phase2/ref/c1b_128M/tok_000128000000
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*" | tee -a logs/phase3/run.log >> logs/phase3/switch.log; }
say "rd paused (user: finish the baseline samples first); no rd AIME records had been written"
for set in gsm8k500 gpqa math500 ifeval541; do
  out=logs/phase3/decode/judge_sample/$set; jl=logs/phase3/jobs/judge_sample/$set; mkdir -p $out $jl
  say "start judge_sample $set (first 12 items, 4 GPUs)"
  pids=()
  for k in 0 1 2 3; do
    mkdir -p /dev/shm/p3_sandbox/judge_sample/$set/shard$k
    ( cd /dev/shm/p3_sandbox/judge_sample/$set/shard$k && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py \
        --model b32 --items $set --drafter d32 --draft s8 --judge full --adapter $AD --judge-diag --limit 12 \
        --shard $k --nshards 4 --gpu $k --score --no-trace --out $ROOT/$out ) > $jl/shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  say "done judge_sample $set: $(ls $out/*.jsonl.done 2>/dev/null | wc -l)/4 shards"
done
say "restart with benchmark order: $ORDER (shortest first, IFEval last; same settings)"
tmux new -d -s p3 "NSH=4 P3_SETS='$ORDER' bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log"
