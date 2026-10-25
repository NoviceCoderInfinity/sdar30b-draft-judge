#!/usr/bin/env bash
# User (06 Oct ~20:20 IST): GSM8K and MATH-500 first, the rest only if credit is topped up.
# samples_then_rd.sh launches rd with the order it read at start; as soon as it does (models still loading, no records),
# stop that launch and relaunch with the new order. Same settings otherwise.
cd /workspace/dmge; ORDER=$(cat logs/phase3/extrap/order.txt)
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*" | tee -a logs/phase3/run.log >> logs/phase3/switch.log; }
n0=$(grep -n "rd paused (user: finish the baseline samples first)" logs/phase3/run.log | tail -1 | cut -d: -f1)
while true; do
  tail -n +$n0 logs/phase3/run.log | grep -q "restart with benchmark order" && break
  sleep 1
done
sleep 2
tmux kill-session -t p3 2>/dev/null
pkill -f "code/draftalign/run_da_decode[.]py"
for i in $(seq 60); do pgrep -f "code/draftalign/run_da_decode[.]py" >/dev/null || break; sleep 1; done
pkill -9 -f "code/draftalign/run_da_decode[.]py" 2>/dev/null; sleep 3
for a in rd1 rd2; do for s in aime gsm8k500 gpqa math500 ifeval541; do
  d=logs/phase3/decode/$a/$s; [ -d $d ] && [ ! -f $d/.complete ] && [ -z "$(cat $d/*.jsonl 2>/dev/null)" ] && rm -rf $d
done; done
say "restart with benchmark order: $ORDER (user: GSM8K and MATH-500 first; same settings)"
tmux new -d -s p3 "NSH=4 P3_SETS='$ORDER' bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log"
