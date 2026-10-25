#!/usr/bin/env bash
# Round-3 datagen, generic: blind-rule decode (stock 30B + stock 1.7B drafter, block 32) of one pool slice.
# Command identical to DATA.md §4 Stage 1.
#   bash code/draftalign/launch_r3_datagen.sh opencode_pool:0:4000 opencode_pool_0_4000   [GPUS="0 1 2 3"]
# PY defaults to /venv/r3 (team freeze, no NGC flash_attn) -- see launch_r3_datagen_gsm8k.sh.
# SAFETY: each shard runs with cwd = /dev/shm/dlm/sandbox/<name>/shard<k>: --score executes generated code
# whose file writes/deletes land in the cwd. Never run a code/IF decode from the repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
export HF_HOME=/root/.cache/huggingface
PY=${PY:-/venv/r3/bin/python}
ITEMS="$1"; NAME="$2"
OUT=$ROOT/logs/draftalign/datagen/${NAME}_dv-s8_blind_q50_r3
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
N=${#GPUS[@]}
mkdir -p "$OUT"
for k in $(seq 0 $((N-1))); do
  g=${GPUS[$k]}; SB=/dev/shm/dlm/sandbox/${NAME}/shard${k}of${N}; mkdir -p "$SB"
  (cd "$SB" && CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$ROOT/code/draftalign/run_da_decode.py" \
    --model b32 --drafter d32 --items "$ITEMS" --draft s8 --judge blind --tau-acc-q .50 \
    --shard $k --nshards $N --gpu $g --score --out "$OUT" > "$OUT/shard${k}of${N}.log" 2>&1 &)
done
echo "launched $N shards of $ITEMS -> $OUT"
