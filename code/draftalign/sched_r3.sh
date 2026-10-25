#!/usr/bin/env bash
# Two-level GPU scheduler for round-3 datagen (keeps every card busy).
#   nohup bash code/draftalign/sched_r3.sh <hi_queue> <lo_queue> <log> &
# Each queue line is one shell command containing {GPU}. Every 20 s, every GPU that holds < 1000 MiB and has
# no live job launched by this scheduler takes the next HI line (code / IF shards); only if HI is empty does it
# take a LO line (filler). Lines can be appended to either queue while it runs. Queue files are NOT deleted
# (gpu_queue.sh footgun); a line is moved to <queue>.launched when started. Exits when both queues are empty
# and no job it launched is alive. Stop: touch <log>.stop
set -u
HI=${1:?hi queue}; LO=${2:?lo queue}; LOG=${3:?log}
cd "$(dirname "$0")/../.."
touch "$HI" "$LO"
declare -A pid
pop() {   # pop first non-empty line of $1 into $line
  line=$(grep -m1 -v '^[[:space:]]*$' "$1" || true)
  [ -n "$line" ] || return 1
  grep -v -x -F -- "$line" "$1" > "$1.tmp" || true; mv "$1.tmp" "$1"
  echo "$line" >> "$1.launched"; return 0
}
echo "$(date -u +%FT%TZ) scheduler start hi=$HI lo=$LO" >> "$LOG"
while [ ! -e "$LOG.stop" ]; do
  alive=0
  for g in "${!pid[@]}"; do kill -0 "${pid[$g]}" 2>/dev/null && alive=$((alive+1)) || unset "pid[$g]"; done
  if [ $alive -eq 0 ] && ! grep -q -v '^[[:space:]]*$' "$HI" && ! grep -q -v '^[[:space:]]*$' "$LO"; then
    echo "$(date -u +%FT%TZ) queues empty, all jobs exited" >> "$LOG"; break
  fi
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000 {print $1}'); do
    [ -n "${pid[$g]:-}" ] && continue
    if pop "$HI"; then q=HI; elif pop "$LO"; then q=LO; else break; fi
    full=${line//\{GPU\}/$g}
    bash -c "$full" &
    pid[$g]=$!
    echo "$(date -u +%FT%TZ) GPU $g [$q] pid ${pid[$g]}: $full" >> "$LOG"
  done
  sleep 20
done
