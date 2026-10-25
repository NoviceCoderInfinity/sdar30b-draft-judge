#!/bin/bash
# Mac-session decision 04:41 IST: Phase 5 judge degenerates (all trained checkpoints loop); use the remaining GPU time for
# same-box PLAIN-harness numbers of the Phase 4 judge (128M, start_128m) + the vanilla multiplier. Hard stop 07:10 IST.
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
L=$ROOT/logs/phase5/eval/chain128.log
say(){ echo "[chain $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*" | tee -a $L; }
STOP=$(date -d '2026-10-24 01:40:00 UTC' +%s)   # 07:10 IST
( while [ "$(date +%s)" -lt "$STOP" ]; do sleep 20; done; say "07:10 hard stop: killing decodes by PID"; for p in $(ps -eo pid,args --no-headers | grep "[r]un_da_decode" | awk '{print $1}'); do kill $p; done ) &
WD=$!
A=$ROOT/logs/phase5/judge/start_128m
say "step 1: rd1 eval, Phase 4 judge (start_128m), core sets gsm8k500 math500 humaneval mbpp"
ADAPTER=$A SETS="gsm8k500 math500 humaneval mbpp" bash code/judge_round/eval_rd1.sh >> $L 2>&1
[ "$(date +%s)" -lt "$STOP" ] || exit 0
say "step 2: vanilla multiplier check (N=28 per set, 7 sets)"
bash code/judge_round/vanilla_check.sh >> $ROOT/logs/phase5/vanilla/vanilla.log 2>&1
[ "$(date +%s)" -lt "$STOP" ] || exit 0
say "step 3: rd1 eval, Phase 4 judge, gpqa aime (then ifeval541 if time)"
ADAPTER=$A SETS="gpqa aime" bash code/judge_round/eval_rd1.sh >> $L 2>&1
NOW=$(date +%s); if [ $((STOP-NOW)) -gt 2900 ]; then ADAPTER=$A SETS="ifeval541" bash code/judge_round/eval_rd1.sh >> $L 2>&1; else say "skip ifeval541 (not enough time)"; fi
say "CHAIN DONE"; kill $WD 2>/dev/null
