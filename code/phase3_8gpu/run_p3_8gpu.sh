#!/usr/bin/env bash
# Phase 3 re-draft on the 8x RTX PRO 6000 Server box: rd1 and rd2 only. The base pipeline is NOT re-run here; its
# reference is Phase 1's records (logs/phase2/ref/S0_128M_nopool_by_set).
# Copy of code/phase3/run_p3.sh. The run_da_decode.py command line of every rd decode is IDENTICAL to run_p3.sh's
# (same flags, --judge-diag included, 8 shards items[k::8], one process per GPU, --score --no-trace). Differences:
#   - outputs under logs/phase3_8gpu/ (run.log, decode/, jobs/, STOP) instead of logs/phase3/; setup = setup_p3_8gpu.sh
#   - fixed order: each benchmark rd1 then rd2 (run_p3.sh alternated); the four never-run benchmarks first
#   - ENABLE_WANDB=0, as in every earlier run (their shard logs: "[wandb] disabled by ENABLE_WANDB")
#   - between runs only, never during one: a canary (HumanEval item 0 with the setup judge check's settings, all 8 GPUs
#     at once) after setup and after every run; host snapshots (GPU clock-event counters, cgroup CPU throttling, load,
#     GPU processes) before and after every run; then report + page + backup
# Run:  tmux new -d -s p3g 'bash code/phase3_8gpu/run_p3_8gpu.sh 2>&1 | tee -a logs/phase3_8gpu/run.log'
# Stop: touch logs/phase3_8gpu/STOP (stops before the next run); at once also: pkill -f "run_da_decod[e].py"
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false ENABLE_WANDB=0
PY=/venv/p2/bin/python; NSH=${NSH:-8}
L=logs/phase3_8gpu
mkdir -p $L/hostmon $L/canary
say(){ echo "[p3 $(date -u +%H:%M:%S) UTC] $*"; }
ev(){ printf '%s\t%s\t%s\t%s\n' "$(date -u +%s.%N)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> $L/events.tsv; }

if [ "${P3G_SKIP_SETUP:-0}" = 1 ]; then
  say "setup skipped (P3G_SKIP_SETUP=1, the project lead's instruction): see $L/setup/ for what ran"
else
bash code/phase3_8gpu/setup_p3_8gpu.sh || { say "SETUP FAILED -- see $L/setup/"; exit 1; }
# the project lead's bar is stricter than the script's (>= 12/16): base outputs token-identical to Phase 1, 16/16
$PY - <<'EOF' || { say "SETUP CHECK below 16/16 token-identical -- stopping"; exit 1; }
import json
c = json.load(open("logs/phase3_8gpu/setup/checks.json"))
assert c["stock_items"] == 16 and c["stock_identical"] == 16, c
assert c["judge_with_reference"] == 16 and c["judge_identical"] == 16, c
print("[p3g] setup checks: stock 16/16, base (SDAR-1.7B + 128M judge) 16/16 token-identical to Phase 1:", c)
EOF
fi

AD=logs/phase2/ref/c1b_128M/tok_000128000000
REFJ=logs/phase2/ref/S0_128M_nopool_by_set
SETS=${P3_SETS:-"ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"}
read -r -a ARMS <<< "${P3G_ARMS:-rd1 rd2}"   # rd1 rd2; "base" (the base pipeline) queued after them by the project lead
JUDGE="--drafter d32 --draft s8 --judge full --adapter $AD --judge-diag"
flags(){ case $1 in
  rd1) echo "$JUDGE --full-redraft-every 1" ;;
  rd2) echo "$JUDGE --full-redraft-every 2" ;;
  base) echo "$JUDGE" ;;
  *) return 1 ;;
esac; }

run_cell(){   # $1 arm, $2 benchmark: 8 shards, one per GPU, then wait; one retry (records resume)
  local arm=$1 set=$2 out=$L/decode/$1/$2 jl=$L/jobs/$1/$2 attempt k p n
  mkdir -p $out $jl
  [ -f $out/.complete ] && { say "skip $arm $set (complete)"; return 2; }
  for attempt in 1 2; do
    say "start $arm $set (attempt $attempt)"; ev start_run "$arm $set attempt$attempt"
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
    ev end_run "$arm $set attempt$attempt shards_done=$n/$NSH"
    if [ "$n" -ge "$NSH" ]; then touch $out/.complete; say "done $arm $set"; return 0; fi
    say "$arm $set: $n/$NSH shards finished"
  done
  return 1
}

CSEQ=$(ls -d $L/canary/[0-9][0-9]_* 2>/dev/null | wc -l)
canary(){    # $1 label: the same item on all 8 GPUs at once, the setup judge check's settings, --limit 1 (= item 0)
  CSEQ=$((CSEQ + 1)); local d=$L/canary/$(printf %02d $CSEQ)_$1 k p pids=()
  mkdir -p $d; say "canary $1"; ev start_canary "$1"
  for k in $(seq 0 $((NSH - 1))); do
    mkdir -p /dev/shm/p3_sandbox/canary/gpu$k
    ( cd /dev/shm/p3_sandbox/canary/gpu$k && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py \
        --model b32 --items humaneval --drafter d32 --draft s8 --judge full --adapter $AD --limit 1 \
        --ref-dir $REFJ/humaneval --p2-integrity --no-trace --score --gpu $k --out $d/gpu$k ) > $d/gpu$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  ev end_canary "$1"
  $PY code/phase3_8gpu/canary_check.py $L/canary; local rc=$?
  [ $rc = 3 ] && { touch $L/STOP; say "CANARY: more than 3% slower than the first canary -- STOP written (stop rule)"; }
  return 0
}

hostmon(){   # $1 label: GPU clock-event counters, cgroup CPU throttling, load, GPU processes (never during a run)
  local f=$L/hostmon/$(date -u +%Y%m%dT%H%M%SZ)_$1.txt
  { echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "loadavg $(cat /proc/loadavg)"
    grep -E "^(nr_periods|nr_throttled|throttled_usec)" /sys/fs/cgroup/cpu.stat
    nvidia-smi --query-gpu=index,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.hw_power_brake_slowdown,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,memory.used --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $f 2>&1
}

post_run(){  # report + page + backup, between runs (nothing timed is running)
  local s
  for s in report_8gpu.py build_page.py; do
    [ -f code/phase3_8gpu/$s ] && { $PY code/phase3_8gpu/$s >> $L/analysis.log 2>&1 || say "$s failed ($L/analysis.log)"; }
  done
  # backup in the background (niced, logged in bg_jobs.tsv) so Hugging Face / GitHub retries never delay the next run
  [ -f code/phase3_8gpu/backup_8gpu.py ] && { ( nice -n 19 ionice -c3 $PY code/phase3_8gpu/backup_8gpu.py --tag "after $1" \
    >> $L/backup.log 2>&1 && say "backup ok ($1)" || say "backup FAILED ($1; $L/backup.log)" ) & }
  return 0
}

say "8-GPU Phase 3 re-run: arms ${ARMS[*]}; benchmarks $SETS; NSH=$NSH"
hostmon start
post_run resume
for set in $SETS; do
  for arm in "${ARMS[@]}"; do
    [ -f $L/STOP ] && { say "STOP file: stopping before $arm $set"; post_run stop; exit 0; }
    [ -f $L/decode/$arm/$set/.complete ] && { say "skip $arm $set (complete)"; continue; }
    hostmon "before_${arm}_${set}"
    run_cell $arm $set || { say "FAILED: $arm $set after 2 attempts ($L/jobs/$arm/$set)"; hostmon "failed_${arm}_${set}"; \
      post_run failure; exit 1; }
    hostmon "after_${arm}_${set}"
    post_run "$arm $set"
  done
done
wait   # the last backup
touch $L/DONE_$(IFS=_; echo "${ARMS[*]}")
say "DONE (arms ${ARMS[*]}): results/phase3_8gpu/REPORT.md"
