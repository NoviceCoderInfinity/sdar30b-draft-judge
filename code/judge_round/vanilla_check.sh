#!/usr/bin/env bash
# Phase 5 VANILLA MULTIPLIER CHECK: ~N items/set, PLAIN harness VANILLA (--stock), this box, paired to the Phase 4
# vanilla records (logs/phase4/decode/vanilla_p4box/<set>) by item idx. Reports per-set + pooled median seconds ratio
# (this box / Phase 4 box) and token identity. Speedup is DERIVED elsewhere = our rd1 seconds / Phase4 vanilla seconds
# x this multiplier. SGLang is never used here.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1
N=${N:-28}; SETS=${SETS:-"ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"}
abs(){ case $1 in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
say(){ echo "[vanilla $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
for S in $SETS; do
  OUT=logs/phase5/vanilla/$S; mkdir -p "$OUT"; pids=()
  say "$S: vanilla (--stock) first $N items, 8-way sharded"
  for k in 0 1 2 3 4 5 6 7; do
    SB=/dev/shm/p5van/$S/gpu$k; mkdir -p "$SB"
    ( cd "$SB" && CUDA_VISIBLE_DEVICES=$k /venv/p2/bin/python "$ROOT/code/draftalign/run_da_decode.py" \
        --model b32 --items "$S" --stock --shard "$k" --nshards 8 --gpu "$k" --limit "$N" \
        --score --no-trace --out "$(abs $OUT)" ) > "$OUT/gpu$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  say "$S: done"
done
say "VANILLA DECODES DONE — pairing to Phase 4 vanilla_p4box"
/venv/p2/bin/python code/judge_round/vanilla_pair.py 2>&1 | tee logs/phase5/vanilla/multiplier.txt
