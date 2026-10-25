#!/usr/bin/env bash
# Phase 5 decode runner (all stages). Defaults = stage 1: the loop guard on IFEval-541, rd1 and rd2, Phase 4 drafter +
# ban + guard, frozen 128M judge, 8 shards.
#   bash code/loop_guard/run_lg.sh decode    # arms in LG_ARMS (rd1 rd2), sets in SETS, NSH shards (only shards 0..NGPU-1 run)
#   bash code/loop_guard/run_lg.sh report    # vs Phase 4 same arm (REF_ROOT/<arm>) and vs vanilla on the Phase 4 box
# Knobs: SETS, NSH, OUTDIR, AD (judge adapter), EXTRA (decoder flags; default ban + guard), LG, REF_ROOT, LG_ARMS.
# Decodes run from per-shard sandbox directories under /dev/shm (AGENTS.md footgun p).
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}
PY=${PY:-/venv/p2/bin/python}
NGPU=${NGPU:-8}
SETS=${SETS:-ifeval541}
NSH=${NSH:-8}
LG=${LG:-12}
TRAIN=${TRAIN:-$ROOT/logs/phase4/train/main}
AD=${AD:-$ROOT/logs/phase2/ref/c1b_128M/tok_000128000000}
EXTRA=${EXTRA:---redraft-ban-rejected --loop-guard $LG}
L=$ROOT/logs/phase5; R=$ROOT/results/phase5_loop_guard; mkdir -p $L $R
OUTDIR=${OUTDIR:-$L/decode}
REF_ROOT=${REF_ROOT:-$ROOT/logs/phase4/decode}
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_lg.log; }
cd $ROOT

decode_cell(){   # $1 arm (rd1|rd2), $2 set, $3 out dir, $4 nshards; shards 0..NGPU-1, one retry (records resume)
  local arm=$1 set=$2 out=$3 nsh=$4 every attempt k n
  case $arm in rd1) every=1 ;; rd2) every=2 ;; *) say "bad arm $arm"; return 1 ;; esac
  [ -f $out/.complete ] && { say "skip $arm $set (complete)"; return 0; }
  mkdir -p $out
  for attempt in 1 2; do
    local pids=()
    for k in $(seq 0 $((NGPU - 1))); do
      local sb=/dev/shm/p5_sandbox/$(echo $out | tr '/' '_')/shard$k; mkdir -p $sb
      ( cd $sb && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --items $set \
          --drafter $TRAIN/merged --draft s8 --judge full --adapter $AD --judge-diag --full-redraft-every $every \
          $EXTRA --shard $k --nshards $nsh --gpu $k --score --no-trace --out $out \
      ) > $out/shard$k.attempt$attempt.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    n=$(ls $out/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n" -ge "$NGPU" ] && { touch $out/.complete; say "done $arm $set ($out)"; return 0; }
    say "$arm $set: $n/$NGPU shards finished (attempt $attempt)"
  done
  return 1
}

case ${1:-help} in
decode)
  for arm in ${LG_ARMS:-rd1 rd2}; do
    for set in $SETS; do
      say "decode $arm $set [$EXTRA] adapter $(basename $AD) -> $OUTDIR/${arm}_lg$LG/$set"
      decode_cell $arm $set $OUTDIR/${arm}_lg$LG/$set $NSH || { say "FAILED: $arm $set"; exit 1; }
    done
  done ;;
report)
  for arm in ${LG_ARMS:-rd1 rd2}; do
    $PY code/drafter_lora/report_p4.py --arm ${arm}_lg$LG --new $OUTDIR/${arm}_lg$LG --sets $(echo $SETS | tr ' ' ',') \
      --phase3 $REF_ROOT/$arm \
      --vanilla "$ROOT/logs/phase4/decode/vanilla_here/{set}/*_B32_t95_stock.{set}.shard*.jsonl" \
      --out $R/lg${LG}_${arm}.json || { say "report failed for $arm"; exit 1; }
    $PY code/loop_guard/verdict_lg.py --arm $arm --report $R/lg${LG}_${arm}.json --out $R/verdict_${arm}.json || exit 1
  done ;;
*) echo "usage: $0 decode|report"; exit 2 ;;
esac
