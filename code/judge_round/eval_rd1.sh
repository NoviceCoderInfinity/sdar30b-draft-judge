#!/usr/bin/env bash
# Phase 5 EVAL: rd1, 7 sets, PLAIN HF harness (run_da_decode.py, Phase-4 rd1 flags), judge adapter = latest rec_main
# milestone (never merged). Each set sharded 8-way, one process per GPU (per-item `seconds`/`gpu_active_s` stay clean,
# one process per GPU). Sandbox cwd under /dev/shm (AGENTS.md §9). MAXITEMS caps per set if time is short (0 = full).
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1
SETS=${SETS:-"ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"}
# Adapter: honor an explicit $ADAPTER (auto_eval passes the newest COMPLETE checkpoint, which may be rec_main_ep2/final);
# otherwise fall back to the newest rec_main_ep2 milestone, then the newest rec_main milestone.
ADAPTER=${ADAPTER:-$(ls -d logs/phase5/judge/rec_main_ep2/tok_* logs/phase5/judge/rec_main/tok_* 2>/dev/null | sort | tail -1)}
MERGED=logs/phase4/train/main/merged
MAXITEMS=${MAXITEMS:-0}
abs(){ case $1 in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
say(){ echo "[eval $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
[ -n "$ADAPTER" ] && [ -f "$ADAPTER/adapter.safetensors" ] || { say "no judge checkpoint found (ADAPTER=$ADAPTER)"; exit 1; }
say "judge adapter = $ADAPTER (never merged); sets: $SETS"
LIM=""; [ "$MAXITEMS" != 0 ] && LIM="--limit $MAXITEMS"
for S in $SETS; do
  OUT=logs/phase5/eval/$S; mkdir -p "$OUT"; pids=(); bad=""
  say "$S: 8-way sharded decode starting"
  for k in 0 1 2 3 4 5 6 7; do
    SB=/dev/shm/p5eval/$S/gpu$k; mkdir -p "$SB"
    ( cd "$SB" && CUDA_VISIBLE_DEVICES=$k /venv/p2/bin/python "$ROOT/code/draftalign/run_da_decode.py" \
        --model b32 --items "$S" --drafter "$(abs $MERGED)" --draft s8 --judge full --adapter "$(abs $ADAPTER)" \
        --full-redraft-every 1 --redraft-ban-rejected --loop-guard 12 --shard "$k" --nshards 8 --gpu "$k" \
        --score --no-trace $LIM --out "$(abs $OUT)" ) > "$OUT/gpu$k.log" 2>&1 &
    pids+=($!)
  done
  k=0; for p in "${pids[@]}"; do wait "$p" || bad="$bad $k"; k=$((k+1)); done
  [ -z "$bad" ] && say "$S: done" || say "$S: FAILED on gpu(s)$bad (see $OUT/gpu*.log)"
done
say "ALL EVAL SETS DONE"
