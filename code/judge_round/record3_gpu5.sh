#!/usr/bin/env bash
# Phase 5 CP5: third disjoint subset on GPU 5 alone (all pids, one process), batched; on crash relaunch at mr1. Resumable.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
INDEX=logs/c1b/trainset/fresh/index.jsonl; PIDS=${PIDS:-logs/phase5/record_pids_3rd.json}
OUT=${OUT:-logs/phase5/judge_sglang/record3}; GPU=${GPU:-5}
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$ROOT/code
. logs/phase5/sgl_setup/sgl_env.sh 2>/dev/null
mkdir -p "$OUT"
say(){ echo "[rec3 $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
NEED=$(/venv/p2/bin/python -c "import json;print(len(json.load(open('$PIDS'))['pids']))")
launch(){ local mr=$1
  for q in $(ps -eo pid,args --no-headers | grep -E "launch_server .*--port 3010$GPU( |$)" | grep -v grep | awk '{print $1}'); do kill -9 "$q" 2>/dev/null; done
  ( export CUDA_VISIBLE_DEVICES=$GPU SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296
    /venv/sgl/bin/python code/judge_round/judge_sglang.py --mode record --index "$INDEX" --pids "$PIDS" \
      --shard 0 --nshards 1 --gpu $GPU --redraft-every 1 --redraft-ban --loop-guard 12 \
      --max-running "$mr" --out "$OUT" --mem-fraction 0.70 --drafter-mem-fraction 0.10 ) > "$OUT/record_gpu${GPU}.log" 2>&1; }
mr=64
while true; do
  launch $mr
  ndone=$(grep -c . "$OUT/gpu$GPU/done.txt" 2>/dev/null || echo 0)
  [ "$ndone" -ge "$NEED" ] && { say "DONE ($ndone/$NEED)"; break; }
  if [ "$mr" = 64 ]; then say "crashed at mr64 ($ndone/$NEED); relaunch at mr1"; mr=1; else say "exited at mr1 ($ndone/$NEED); relaunch"; fi
done
