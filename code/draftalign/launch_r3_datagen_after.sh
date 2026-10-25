#!/usr/bin/env bash
# Round-3 datagen chained per GPU: shard k of <items> starts on GPU k as soon as GPU k's previous
# shard writes its .done marker, so no card idles between pools. Same decode command as launch_r3_datagen.sh.
#   nohup bash code/draftalign/launch_r3_datagen_after.sh <prev_datagen_dir> <items> <out_name> &
# SAFETY: cwd = /dev/shm/dlm/sandbox/<name>/shard<k> per shard (generated code executes in the cwd).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
export HF_HOME=/root/.cache/huggingface
PY=${PY:-/venv/r3/bin/python}
PREV="$1"; ITEMS="$2"; NAME="$3"; OUT=$ROOT/logs/draftalign/datagen/${NAME}_dv-s8_blind_q50_r3
case "$PREV" in /*) ;; *) PREV=$ROOT/$PREV ;; esac
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
mkdir -p "$OUT"
for k in $(seq 0 $((N-1))); do
  (
    g=${GPUS[$k]}; SB=/dev/shm/dlm/sandbox/${NAME}/shard${k}of${N}; mkdir -p "$SB"; cd "$SB"
    until ls "$PREV"/*.shard${k}of${N}.jsonl.done >/dev/null 2>&1; do sleep 30; done
    CUDA_VISIBLE_DEVICES=$g "$PY" "$ROOT/code/draftalign/run_da_decode.py" \
      --model b32 --drafter d32 --items "$ITEMS" --draft s8 --judge blind --tau-acc-q .50 \
      --shard $k --nshards $N --gpu $g --score --out "$OUT" > "$OUT/shard${k}of${N}.log" 2>&1
  ) &
done
wait
echo "all $N shards of $ITEMS exited -> $OUT"
