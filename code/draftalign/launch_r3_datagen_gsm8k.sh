#!/usr/bin/env bash
# Round-3 datagen, maths slice (GSM8K only, per 2026-09-14 instruction): blind-rule decode of the
# GSM8K train pool with the stock 30B + stock 1.7B drafter, block 32. Command identical to DATA.md §4 Stage 1.
# Target 11,200 blocks; 2,400 prompts x ~5.95 blocks x ~0.947 survival ~= 13.5k blocks (margin).
# PY defaults to /venv/r3: the team's exact pip freeze (torch 2.11.0+cu128, transformers 4.52.4) with NO
# system site-packages -- the NGC image's real flash_attn makes transformers import flash_attn.layers,
# which the code/compat shim shadows (ModuleNotFoundError at model load).
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME=/root/.cache/huggingface
PY=${PY:-/venv/r3/bin/python}
ITEMS="gsm8k_pool:0:2400"
OUT=logs/draftalign/datagen/gsm8k_pool_0_2400_dv-s8_blind_q50_r3
N=${NSHARDS:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$OUT"
for k in $(seq 0 $((N-1))); do
  g=${GPUS[$k]}
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" code/draftalign/run_da_decode.py \
    --model b32 --drafter d32 --items "$ITEMS" --draft s8 --judge blind --tau-acc-q .50 \
    --shard $k --nshards $N --gpu $g --score --out "$OUT" > "$OUT/shard${k}of${N}.log" 2>&1 &
done
echo "launched $N shards -> $OUT"
