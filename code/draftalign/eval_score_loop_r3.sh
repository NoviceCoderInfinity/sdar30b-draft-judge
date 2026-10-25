#!/usr/bin/env bash
# Sequentially score finished code-set shards of the eval wave (one scorer at a time: run_score_r3.sh reuses a sandbox).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"; PR=$ROOT/logs/draftalign/probes_r3/r3_A
pass() {
  for d in $PR/*/humaneval/* $PR/*/mbpp/*; do
    [ -d "$d" ] || continue
    n1=$(ls "$d"/*.jsonl.done 2>/dev/null | wc -l); n2=$(ls "$d"/scored/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n1" -gt "$n2" ] && bash code/draftalign/run_score_r3.sh "$d" once "$d/scored" >> "$d/score.log" 2>&1
  done
}
while :; do
  pass
  pgrep -f "sched_r3[.]sh .*queue_eval_r3" >/dev/null || { pass; echo "$(date -u +%FT%TZ) wave over, final scoring pass done"; break; }
  sleep 120
done
