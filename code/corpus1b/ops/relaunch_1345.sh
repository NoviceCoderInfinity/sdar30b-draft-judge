#!/usr/bin/env bash
# when the current run (tmux train3, --stop-at 13:45 IST) exits, relaunch it to 14:45 IST (train_resume3.sh, tmux
# train4). A crash is relaunched the same way (it resumes from the newest checkpoint): at most 3 times, never after
# 14:38 IST. The run is matched by its torchrun command line only.
cd /workspace/dmge
alive(){ pgrep -f "^/venv/train/bin/python /venv/train/bin/torchrun .*train_c1b.py" > /dev/null; }
n=0
while [ "$(date +%s)" -lt 1790672880 ]; do
  if alive; then sleep 10; continue; fi
  sleep 5; alive && continue
  n=$((n+1)); [ $n -gt 3 ] && { echo "[relaunch] too many relaunches: stopping"; exit 1; }
  echo "[relaunch $(TZ=Asia/Kolkata date +%H:%M:%S) IST] run exited ($(grep -E '^\[(stop-at|done)\]|Error' logs/c1b/train/main.log | tail -1 | cut -c1-120)): relaunch $n"
  tmux kill-session -t train3 2>/dev/null; tmux kill-session -t train4 2>/dev/null
  tmux new -d -s train4 'bash logs/c1b/dl/train_resume3.sh'
  sleep 150
done
echo "[relaunch] past 14:38 IST: watcher done"
