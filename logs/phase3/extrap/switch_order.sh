#!/usr/bin/env bash
# One-shot: when run_p3.sh starts the first GSM8K cell (models still loading, no records yet), stop it and restart
# with the benchmark order in order.txt (shortest first, IFEval last). Same command otherwise; finished cells skip.
cd /workspace/dmge; LOG=logs/phase3/switch.log; ORDER=$(cat logs/phase3/extrap/order.txt)
n0=$(wc -l < logs/phase3/run.log)
while true; do
  new=$(tail -n +$((n0+1)) logs/phase3/run.log)
  echo "$new" | grep -qE "FAILED: |DONE:" && { echo "$(date -u) run failed/done before switch; nothing done" >> $LOG; exit 0; }
  echo "$new" | grep -qE "start rd[12] gsm8k500 " && break
  sleep 1
done
echo "$(date -u) GSM8K cell started; switching order to: $ORDER" >> $LOG
tmux kill-session -t p3 2>/dev/null
pkill -f "run_da_decod[e].py"
for i in $(seq 60); do pgrep -f "run_da_decod[e].py" >/dev/null || break; sleep 1; done
pkill -9 -f "run_da_decod[e].py" 2>/dev/null; sleep 3
echo "$(date -u) old shards gone: $(pgrep -fc 'run_da_decod[e].py') left; GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >> $LOG
echo "[p3 $(date -u +%H:%M:%S) UTC] restart with benchmark order: $ORDER (shortest first, IFEval last; same settings)" >> logs/phase3/run.log
tmux new -d -s p3 "NSH=4 P3_SETS='$ORDER' bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log"
echo "$(date -u) restarted" >> $LOG
