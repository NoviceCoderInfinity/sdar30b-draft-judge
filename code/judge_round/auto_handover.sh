#!/usr/bin/env bash
# Phase 5 Plan-A handover v2 (operator 23:30 IST, no human step): start training as soon as RUN-2 has finished on all 7
# GPUs (its supervisor tmux `record2` exits when shards 0-4,6,7 are done) OR at 01:10 IST latest. At that instant: stop
# run-3 (GPU5) and any remaining record shards BY PID, keep their finished items, do NOT start a 4th slice; then run
# train_cp6.sh (index over all record roots + step-0 check + 8-GPU train_c1b_rec) in tmux. 32M-token target, done by 04:30.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
DEADLINE=$(date -d '2026-10-23 19:30:00 UTC' +%s)   # 01:00 IST latest start (run-2 ends ~00:58; its last items are maths, the scarcest domain)
log(){ echo "[handover $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
kill_by_pid(){ local pat=$1 p; for p in $(ps -eo pid,args --no-headers | grep -E "$pat" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null && log "killed pid $p"; done; }

# wait until run-2 (the 7-GPU supervisor) finishes, or the 01:10 latest-start deadline
while tmux has-session -t record2 2>/dev/null && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 30; done
if tmux has-session -t record2 2>/dev/null; then log "01:10 latest-start reached (run-2 not yet done) — proceeding"; else log "run-2 finished on all 7 GPUs"; fi

# stop run-3 (GPU5) and any remaining record shards BY PID (finished items are kept on disk); no 4th slice
kill_by_pid 'judge_sglang[.]py --mode record'
kill_by_pid 'sglang.launch_server'
sleep 8
log "recorded total=$(cat logs/phase5/judge_sglang/record*/gpu*/done.txt 2>/dev/null | grep -c .) items across all roots"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
tmux new -d -s train 'bash code/judge_round/train_cp6.sh 2>&1 | tee -a logs/phase5/judge/cp6_launch.log'
log "TRAINING LAUNCHED (tmux train) — train_cp6.sh"
# arm the 04:15 IST hard training-stop (keep the latest saved checkpoint for eval), kill by PID
tmux new -d -s trainstop 'DL=$(date -d "2026-10-23 22:45:00 UTC" +%s)
  while [ "$(date +%s)" -lt "$DL" ]; do tmux has-session -t train 2>/dev/null || exit 0; sleep 60; done
  echo "[trainstop $(TZ=Asia/Kolkata date "+%H:%M:%S IST")] 04:15 reached: stopping training by PID (keep latest checkpoint)"
  for p in $(ps -eo pid,args --no-headers | grep -E "train_c1b_rec|torchrun|train_cp6.sh" | grep -v grep | awk "{print \$1}"); do kill -9 "$p" 2>/dev/null; done'
log "train-stop watchdog armed for 04:15 IST"
