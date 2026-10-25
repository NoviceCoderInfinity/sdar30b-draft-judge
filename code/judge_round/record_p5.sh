#!/usr/bin/env bash
# Phase 5 CP5: record the judge's per-pass states over the fresh index, one SGLang process per GPU (resumable via done.txt).
# SGLang is data generation here (allowed); every eval/speed number is a separate plain HF-harness run.
#   tmux new -d -s record 'bash code/judge_round/record_p5.sh 2>&1 | tee -a logs/phase5/judge_sglang/record/record.log'
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}; cd "$ROOT" || exit 1
NGPU=${NGPU:-8}
INDEX=${INDEX:-logs/c1b/trainset/fresh/index.jsonl}
PIDS=${PIDS:-logs/phase5/record_pids.json}
OUT=${OUT:-logs/phase5/judge_sglang/record}
MEMF=${MEMF:-0.70}; DMEMF=${DMEMF:-0.10}
# --max-running 1 (serial per GPU) avoids the SGLang judge-port's batched-scheduler crashes (scattered re-presented
# blocks; ragged mixed prefill+decode batches), both proven absent at max-running 1. Block-level parallelism is kept.
MAXRUN=${MAXRUN:-1}
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$ROOT/code
. logs/phase5/sgl_setup/sgl_env.sh 2>/dev/null
mkdir -p "$OUT"
say(){ echo "[record $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
pids=(); for k in $(seq 0 $((NGPU-1))); do
  ( export CUDA_VISIBLE_DEVICES=$k SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296
    /venv/sgl/bin/python code/judge_round/judge_sglang.py --mode record --index "$INDEX" --pids "$PIDS" \
      --shard $k --nshards $NGPU --gpu $k --redraft-every 1 --redraft-ban --loop-guard 12 \
      --max-running $MAXRUN --out "$OUT" --mem-fraction $MEMF --drafter-mem-fraction $DMEMF ) > "$OUT/record_gpu$k.log" 2>&1 &
  pids+=($!); say "launched gpu $k (pid ${pids[-1]})"
done
bad=""; k=0; for p in "${pids[@]}"; do wait $p || bad="$bad $k"; k=$((k+1)); done
[ -z "$bad" ] && say "ALL $NGPU record processes finished rc=0" || { say "record FAILED on GPU(s)$bad"; exit 1; }
