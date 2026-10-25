#!/usr/bin/env bash
# Lead (06 Oct ~22:22 IST): for a fair rd1 / rd2 / base comparison on GSM8K-500, run the BASE PIPELINE (SDAR-1.7B drafter +
# trained 128M judge = run_p3.sh's `judge` arm: the rd flags minus --full-redraft-every) on ALL of GSM8K-500 on this box,
# 4 GPUs, same load, right after rd2 GSM8K; then resume rd in the same order (MATH-500 next). Applied when run_p3.sh starts
# the first MATH-500 cell (models still loading, no MATH records written yet).
cd /workspace/dmge; ORDER=$(cat logs/phase3/extrap/order.txt)
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*" | tee -a logs/phase3/run.log >> logs/phase3/switch.log; }
n0=$(wc -l < logs/phase3/run.log)
while true; do
  new=$(tail -n +$((n0+1)) logs/phase3/run.log)
  echo "$new" | grep -qE "FAILED: |DONE:" && { echo "$(date -u) run failed/done before the switch; nothing done" >> logs/phase3/switch.log; exit 0; }
  echo "$new" | grep -qE "start rd[12] math500 " && break
  sleep 1
done
sleep 2
tmux kill-session -t p3 2>/dev/null
pkill -f "code/draftalign/run_da_decode[.]py"
for i in $(seq 60); do pgrep -f "code/draftalign/run_da_decode[.]py" >/dev/null || break; sleep 1; done
pkill -9 -f "code/draftalign/run_da_decode[.]py" 2>/dev/null; sleep 3
for a in rd1 rd2; do d=logs/phase3/decode/$a/math500; [ -d $d ] && [ ! -f $d/.complete ] && [ -z "$(cat $d/*.jsonl 2>/dev/null)" ] && rm -rf $d; done
say "rd paused before MATH-500 (no MATH records written); base pipeline (today's judge) on all of GSM8K-500, same box and load"
NSH=4 P3_ARMS=judge P3_SETS="gsm8k500" bash code/phase3/run_p3.sh 2>&1 | sed 's/DONE: /baseline finished: /' | tee -a logs/phase3/run.log
say "restart with benchmark order: $ORDER (after the GSM8K base-pipeline run; same settings)"
tmux new -d -s p3 "NSH=4 P3_SETS='$ORDER' bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log"
echo "$(date -u) rd restarted" >> logs/phase3/switch.log
