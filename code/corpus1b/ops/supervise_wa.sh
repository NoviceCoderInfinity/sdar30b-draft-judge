#!/usr/bin/env bash
# restart the wave-A orchestrator if it exits after a failed GPU step (every step resumes); never after a verify /
# post-processing failure (those need a person), never after WAVE A DONE, at most 6 restarts
cd /workspace/dmge
n=0
while true; do
  while [ "$(ps -eo args --no-headers | grep -c 'bash code/corpus1b/run_wave_a[.]sh')" -gt 0 ]; do sleep 30; done
  tail -3 logs/c1b/waveA.log | grep -q "WAVE A DONE" && { echo "[supervisor] wave A done"; exit 0; }
  tail -3 logs/c1b/waveA.log | grep -qE "VERIFY FAILED|post-processing of .* FAILED" && { echo "[supervisor] verify failure: not restarting"; exit 1; }
  n=$((n+1)); [ $n -gt 6 ] && { echo "[supervisor] too many restarts"; exit 1; }
  sleep 20
  echo "[supervisor $(date -u +%H:%M:%S) UTC] orchestrator exited: restart $n" | tee -a logs/c1b/waveA.log
  tmux kill-session -t wa 2>/dev/null; sleep 2
  tmux new -d -s wa 'bash code/corpus1b/run_wave_a.sh 2>&1 | tee -a logs/c1b/waveA.log'
  sleep 60
done
