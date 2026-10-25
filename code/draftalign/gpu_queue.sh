#!/bin/bash
# Minimal GPU job queue: pops one command line per free GPU from a queue file every 60 s.
# Queue lines contain {GPU}; a GPU counts as free when it holds < 1000 MiB and was not used by us in the last 150 s.
# Usage: nohup code/draftalign/gpu_queue.sh <queue_file> <log> &   (stop: remove the queue file)
set -u
Q=${1:?queue file}; LOG=${2:?log}; ALLOW=${3:-}   # ALLOW="0 1" restricts to those GPUs; empty = any
cd "$(dirname "$0")/../.."
declare -A last
while [ -f "$Q" ]; do
  if [ ! -s "$Q" ]; then echo "$(date -u +%FT%TZ) queue empty, exiting" >> "$LOG"; rm -f "$Q"; break; fi
  now=$(date +%s)
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000 {print $1}'); do
    [ -s "$Q" ] || break
    if [ -n "$ALLOW" ]; then case " $ALLOW " in *" $g "*) ;; *) continue ;; esac; fi
    t=${last[$g]:-0}; [ $((now - t)) -lt 150 ] && continue
    cmd=$(head -n 1 "$Q"); tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    full=${cmd//\{GPU\}/$g}
    echo "$(date -u +%FT%TZ) GPU $g: $full" >> "$LOG"
    bash -c "$full" &
    last[$g]=$now
  done
  sleep 60
done
