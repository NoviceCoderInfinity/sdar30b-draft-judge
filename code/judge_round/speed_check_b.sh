#!/usr/bin/env bash
# Stage B (box session 1, 23 Oct 2026): vanilla GSM8K-500 on 8 GPUs under the quiet-box rule, then the comparison with
# the Phase 4 box (mean seconds per answer 17.4, mean GPU-active seconds 9.2, token identity 500/500).
#   bash code/judge_round/speed_check_b.sh [gsm8k500|gsm8k500:100]      (the second form = the 100-question drift slice)
# Quiet-box evidence: /proc/loadavg, nvidia-smi and a ps snapshot before and after, to logs/phase5/speed_b/.
# The Phase 4 box's own records of this set are moved aside first (run_vanilla_here.sh resumes on records in its out dir).
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd $ROOT
SET=${1:-gsm8k500}; TAG=${TAG:-$(date -u +%H%M)}
B=logs/phase5/speed_b/$TAG; mkdir -p $B
EV=logs/phase5/events.tsv
ev(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> $EV; }
snap(){ { echo "== $1 $(date -u +%FT%TZ)"; echo "loadavg: $(cat /proc/loadavg)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader;
  echo "-- processes (non-system, > 1% cpu or holding a GPU):"; ps -eo pid,pcpu,etime,args --no-headers | grep -v -E 'shell-snapshots|^ *[0-9]+ +0\.[0-9] ' | grep -v -E 'claude$|supervisord|caddy|tensorboard|portal|tunnel|syncthing|cloudflared|sshd|tmux|sleep|bash -c|grep' | cut -c1-200; } >> $B/quiet_box.txt; }
OUT=logs/phase4/decode/vanilla_here/$SET
if [ -d $OUT ] && [ ! -f $OUT/.box_s1 ]; then ASIDE=logs/phase4/decode/vanilla_p4box/$SET; mkdir -p $(dirname $ASIDE); [ -e $ASIDE ] && ASIDE=$ASIDE.$(date +%s); mv $OUT $ASIDE; echo "moved Phase 4 records aside -> $ASIDE" | tee $B/aside.txt; fi
mkdir -p $OUT; touch $OUT/.box_s1
snap before
ev "speed_b_$SET" start
SETS=$SET bash code/drafter_lora/run_vanilla_here.sh 2>&1 | tee $B/run.log | grep -E 'vanilla-here' | cut -c1-200
ev "speed_b_$SET" "end rc=${PIPESTATUS[0]}"
snap after
/venv/p2/bin/python code/judge_round/speed_check_b.py --set $SET --out $B/summary.json | tee $B/summary.txt
