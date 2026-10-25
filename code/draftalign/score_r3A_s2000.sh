#!/usr/bin/env bash
# Score r3_A step002000 HumanEval/MBPP judge+DES48 shards as they finish (sandboxed, same path as the eval wave). Until 17:50 IST.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"; END=$(TZ=Asia/Kolkata date -d "today 17:50" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  for d in logs/draftalign/probes_r3/r3_A/step002000/humaneval/judge_des48 logs/draftalign/probes_r3/r3_A/step002000/mbpp/judge_des48; do
    [ -d "$d" ] || continue
    n1=$(ls "$d"/*.jsonl.done 2>/dev/null | wc -l); n2=$(ls "$d"/scored/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n1" -gt "$n2" ] && bash code/draftalign/run_score_r3.sh "$ROOT/$d" once "$ROOT/$d/scored" >> "$d/score.log" 2>&1
  done
  sleep 60
done
