#!/usr/bin/env bash
# At the first GSM8K cell start (MBPP done + backed up; models still loading, no records yet):
#  A. stop the rd run; B. today's judge (run_p3.sh's own `judge` arm = rd flags minus --full-redraft-every) on ALL of
#  HumanEval + MBPP; C. the same judge on the first 12 items of the other 5 benchmarks (3 per GPU, 4 GPUs busy like rd);
#  D. restart the rd run with the short-first order. Same box, same load, same flags otherwise.
cd /workspace/dmge; ROOT=$PWD; LOG=logs/phase3/switch.log; ORDER=$(cat logs/phase3/extrap/order.txt)
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*" | tee -a logs/phase3/run.log >> $LOG; }
n0=$(grep -n "start rd1 mbpp" logs/phase3/run.log | tail -1 | cut -d: -f1)
while true; do
  new=$(tail -n +$n0 logs/phase3/run.log)
  echo "$new" | grep -qE "FAILED: |DONE:" && { echo "$(date -u) run failed/done before switch; nothing done" >> $LOG; exit 0; }
  echo "$new" | grep -qE "start rd[12] gsm8k500 " && break
  sleep 1
done
tmux kill-session -t p3 2>/dev/null; pkill -f "run_da_decod[e].py"
for i in $(seq 60); do pgrep -f "run_da_decod[e].py" >/dev/null || break; sleep 1; done
pkill -9 -f "run_da_decod[e].py" 2>/dev/null; sleep 3
say "paused rd run before GSM8K (no GSM8K records written); baseline: today's judge on this box, same load"
NSH=4 P3_ARMS=judge P3_SETS="humaneval mbpp" bash code/phase3/run_p3.sh 2>&1 | grep -v "^DONE" | sed 's/DONE: /baseline finished: /' | tee -a logs/phase3/run.log
PY=/venv/p2/bin/python; AD=logs/phase2/ref/c1b_128M/tok_000128000000
for set in aime gsm8k500 gpqa math500 ifeval541; do
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
echo "$(date -u) rd run restarted" >> $LOG
