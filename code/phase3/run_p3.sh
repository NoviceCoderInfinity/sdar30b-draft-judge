#!/usr/bin/env bash
# Phase 3: SDAR-1.7B drafter + Phase 1's 128M judge, re-drafting the [MASK] slots between judge passes.
# Two setups on all 7 benchmarks, one 8x RTX PRO 6000 box, fully automatic (run it inside tmux):
#   bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log
#   rd1  re-draft after every judge pass      (--full-redraft-every 1)
#   rd2  re-draft after every 2 judge passes  (--full-redraft-every 2)
# Everything else is the earlier runs' setting: run_da_decode.py (PyTorch, no SGLang), JetLM/SDAR-30B-A3B-Chat-b32 @
# c351bbc, SDAR-1.7B-Chat-b32 @ 2c83c05, block 32, threshold 0.95, s8 first draft, --judge full, the 128M judge
# (hash 35805974), NO expert pool (no DES), 8 shards per benchmark (one per GPU), --score, --no-trace.
# Steps: setup_p3.sh -> for each benchmark: rd1 and rd2 (alternating order) -> backup after every benchmark ->
#        analysis (analyze_p3.py) -> final backup (public HF dataset + GitHub branch).
# Resume: run the same command again (finished shards and items are skipped). Stop: touch logs/phase3/STOP
# (stops before the next benchmark); to stop at once also: pkill -f "run_da_decod[e].py".
# Env: P3_ARMS (default "rd1 rd2"), P3_SETS (default all 7), NSH (default 8).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
PY=/venv/p2/bin/python; NSH=${NSH:-8}
mkdir -p logs/phase3
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*"; }
backup(){ $PY code/phase3/backup_p3.py "$@" >> logs/phase3/backup.log 2>&1 && say "backup ok" || say "backup FAILED (logs/phase3/backup.log)"; }

bash code/phase3/setup_p3.sh || { say "SETUP FAILED -- see logs/phase3/setup/"; backup; exit 1; }

AD=logs/phase2/ref/c1b_128M/tok_000128000000
SETS=${P3_SETS:-"humaneval mbpp gsm8k500 ifeval541 math500 gpqa aime"}
read -r -a ARMS <<< "${P3_ARMS:-rd1 rd2}"
JUDGE="--drafter d32 --draft s8 --judge full --adapter $AD --judge-diag"
flags(){ case $1 in
  rd1) echo "$JUDGE --full-redraft-every 1" ;;
  rd2) echo "$JUDGE --full-redraft-every 2" ;;
  judge) echo "$JUDGE" ;;
  stock) echo "--stock" ;;
  *) return 1 ;;
esac; }
for a in "${ARMS[@]}"; do flags $a > /dev/null || { say "unknown arm $a"; exit 1; }; done

run_cell(){   # $1 arm, $2 benchmark: 8 shards, one per GPU, then wait; one retry (records resume)
  local arm=$1 set=$2 out=logs/phase3/decode/$1/$2 jl=logs/phase3/jobs/$1/$2 attempt k p n
  mkdir -p $out $jl
  [ -f $out/.complete ] && { say "skip $arm $set (complete)"; return 0; }
  for attempt in 1 2; do
    say "start $arm $set (attempt $attempt)"
    local pids=()
    for k in $(seq 0 $((NSH - 1))); do
      mkdir -p /dev/shm/p3_sandbox/$arm/$set/shard$k
      ( cd /dev/shm/p3_sandbox/$arm/$set/shard$k && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py \
          --model b32 --items $set $(flags $arm) --shard $k --nshards $NSH --gpu $k --score --no-trace --out $out \
        ) > $jl/shard$k.attempt$attempt.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    n=$(ls $out/*.jsonl.done 2>/dev/null | wc -l)
    if [ "$n" -ge "$NSH" ]; then touch $out/.complete; say "done $arm $set"; return 0; fi
    say "$arm $set: $n/$NSH shards finished"
  done
  return 1
}

i=0
for set in $SETS; do
  [ -f logs/phase3/STOP ] && { say "STOP file: stopping before $set"; break; }
  for j in $(seq 0 $(( ${#ARMS[@]} - 1 ))); do            # alternate which arm runs first on each benchmark
    arm=${ARMS[$(( (i + j) % ${#ARMS[@]} ))]}
    run_cell $arm $set || { say "FAILED: $arm $set after 2 attempts (logs/phase3/jobs/$arm/$set)"; \
      $PY code/phase3/analyze_p3.py > logs/phase3/analysis.log 2>&1; backup --final; exit 1; }
  done
  $PY code/phase3/analyze_p3.py > logs/phase3/analysis.log 2>&1 || say "analysis failed (logs/phase3/analysis.log)"
  backup
  i=$((i + 1))
done
$PY code/phase3/analyze_p3.py > logs/phase3/analysis.log 2>&1 || say "analysis failed (logs/phase3/analysis.log)"
backup --final
say "DONE: logs/phase3/analysis/TABLE.md (also results/phase3/TABLE.md, the HF dataset and the GitHub branch)"
