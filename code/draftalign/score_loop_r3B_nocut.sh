#!/usr/bin/env bash
# No-cutoff replacement for the tail of eval_after_r3B.sh (user, 2026-09-15 16:57 IST: "don't stop at the cutoff, let the
# judge-only run continue"). Scores finished HumanEval/MBPP shards of r3_B step000710 (judge and judge+DES48) and of r3_A
# step002000 as they land, refreshes the r3_B summary, and never kills a decode. Runs until the server goes down.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PRB=$ROOT/logs/draftalign/probes_r3/r3_B/step000710; PRA=$ROOT/logs/draftalign/probes_r3/r3_A/step002000
while :; do
  for d in "$PRB"/humaneval/* "$PRB"/mbpp/* "$PRA"/humaneval/* "$PRA"/mbpp/*; do
    [ -d "$d" ] || continue
    n1=$(ls "$d"/*.jsonl.done 2>/dev/null | wc -l); n2=$(ls "$d"/scored/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n1" -gt "$n2" ] && bash code/draftalign/run_score_r3.sh "$d" once "$d/scored" >> "$d/score.log" 2>&1
  done
  /venv/r3/bin/python code/draftalign/eval_summary_r3B.py --step step000710 >> logs/draftalign/eval_summary_r3B.log 2>&1
  sleep 60
done
