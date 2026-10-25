#!/usr/bin/env bash
# Measured facts for a STATUS block. Times are IST only (operator, 23 Oct 2026); box t0 = 23 Oct 10:09:25 IST (04:39:25Z),
# the 9-hour figure was planning only: no stop time (operator decision 23 Oct 15:03 IST). events.tsv stays UTC; quoted event times are converted to IST here.
T0=${T0:-2026-10-23T04:39:25Z}; ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}; cd $ROOT 2>/dev/null
ist(){ TZ=Asia/Kolkata date -d "$1" '+%d %b %H:%M IST'; }
now=$(date -u +%s); t0=$(date -u -d "$T0" +%s); used=$((now - t0))
printf 'time   %s\n' "$(ist @$now)"
printf 'box    %d:%02d:%02d used since t0 (23 Oct 10:09:25 IST); no stop time (operator, 23 Oct 15:03 IST)\n' $((used/3600)) $((used%3600/60)) $((used%60))
printf 'gpu    %s\n' "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '{u=u" "$1"%"; m=m" "int($2/1024)"G"} END{print "util"u" | mem"m}')"
printf 'disk   %s\n' "$(df -h / | awk 'NR==2{print "used "$3" free "$4" ("$5")"}'); repo $(du -sh $ROOT 2>/dev/null | cut -f1)"
printf 'load   %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
ev=logs/phase5/events.tsv
last(){ local l; l=$(grep -P "\t$1\t$2" $ev 2>/dev/null | tail -1); [ -n "$l" ] && printf '%s (%s)' "$(ist "$(echo "$l" | cut -f1)")" "$(echo "$l" | cut -f3)" || echo none; }
printf 'push   %s\n' "$(last git_push end)"
printf 'upload %s\n' "$(last hf_upload 'end rc=0')"
printf 'dl     %s\n' "$(grep -P '\tdl_' /workspace/dl/events.tsv | awk -F'\t' '{a[$2]=$3} END{for(k in a) printf "%s=%s ", k, a[k]}')"
