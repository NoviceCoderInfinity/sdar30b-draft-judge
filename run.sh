#!/usr/bin/env bash
# Decode + score one setup on one benchmark (or all seven), sharded over the GPUs, with the plain HF harness
# (code/draftalign/run_da_decode.py, block 32) that produced every number in RESULTS.md.
#
#   ./run.sh phase4-rd1 gsm8k500                 # all visible GPUs, all 500 items
#   ./run.sh phase4-rd1 all --gpus "0 1 2 3"     # the seven benchmarks on GPUs 0-3
#   ./run.sh vanilla math500 --limit 16          # first 16 items of each shard
#
# Benchmarks: gsm8k500 math500 humaneval mbpp gpqa aime ifeval541.
# Output: runs/<setup>/<benchmark>/ (one record per item, per-shard logs), then a summary table (scripts/summarize.py).
# Re-running resumes: finished items are skipped. Each shard runs from its own scratch directory, because code
# benchmarks execute the generated programs in the working directory.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
[ -f env.sh ] || { echo "run 'bash setup.sh <setup>' first"; exit 2; }
. ./env.sh
SETUP=${1:?usage: ./run.sh <setup> <benchmark|all> [--gpus "0 1"] [--limit N]}; SET=${2:?benchmark or all}; shift 2
GPUS=""; LIMIT=""
while [ $# -gt 0 ]; do
  case $1 in --gpus) GPUS=$2; shift 2;; --limit) LIMIT="--limit $2"; shift 2;; *) echo "unknown option $1"; exit 2;; esac
done
[ -f setups/$SETUP.sh ] || { echo "unknown setup $SETUP (bash setup.sh --list)"; exit 2; }
. setups/$SETUP.sh
[ -n "$GPUS" ] || GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
read -r -a G <<< "$GPUS"; N=${#G[@]}
[ "$SET" = all ] && SET="gsm8k500 math500 humaneval mbpp gpqa aime ifeval541"
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/sdar30b_runs}
for S in $SET; do
  OUT=$ROOT/runs/$SETUP/$S; mkdir -p "$OUT"
  echo "[run] $SETUP on $S: $N shards on GPUs ${G[*]} -> runs/$SETUP/$S"
  pids=()
  for k in $(seq 0 $((N - 1))); do
    SB=$SCRATCH/$SETUP/$S/shard$k; mkdir -p "$SB"
    ( cd "$SB" && CUDA_VISIBLE_DEVICES=${G[$k]} exec python "$ROOT/code/draftalign/run_da_decode.py" --model b32 \
        --items "$S" $FLAGS --shard $k --nshards $N --gpu ${G[$k]} --score --no-trace $LIMIT --out "$OUT" \
    ) > "$OUT/shard$k.log" 2>&1 &
    pids+=($!)
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad + 1)); done
  [ $bad = 0 ] || echo "[run] $S: $bad shard(s) failed, see runs/$SETUP/$S/shard*.log"
done
python scripts/summarize.py "runs/$SETUP"
