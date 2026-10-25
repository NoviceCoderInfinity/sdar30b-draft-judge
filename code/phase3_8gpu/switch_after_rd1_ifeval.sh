#!/usr/bin/env bash
# One-shot (2026-10-07): the launcher started 07:47 UTC still had a base arm queued. When rd1 ifeval541 completes (all
# 8 decode processes have exited), stop that launcher before it starts anything else, take the host snapshot, and
# continue with the corrected launcher: rd1 and rd2 only.
cd /workspace/dmge
L=logs/phase3_8gpu
until [ -f $L/decode/rd1/ifeval541/.complete ]; do sleep 0.5; done
tmux kill-session -t p3g 2>/dev/null
pkill -f "run_da_decod[e].py --model b32 --items humaneval" 2>/dev/null
bash code/phase3_8gpu/hostmon.sh after_rd1_ifeval541
echo "[p3 $(date -u +%H:%M:%S) UTC] done rd1 ifeval541; previous launcher stopped (its queued base arm removed); continuing with rd1 and rd2 only" >> $L/run.log
sleep 1
tmux new -d -s p3g "cd /workspace/dmge && P3G_SKIP_SETUP=1 bash code/phase3_8gpu/run_p3_8gpu.sh 2>&1 | tee -a logs/phase3_8gpu/run.log"
echo "[switch $(date -u +%H:%M:%S) UTC] new launcher started"
