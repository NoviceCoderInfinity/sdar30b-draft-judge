#!/usr/bin/env bash
# Phase 5 CP5 recording supervisor: one SGLang record process per GPU, batched (--max-running 64) by default; if a shard
# crashes (the port's batched-scheduler bugs), relaunch THAT shard at --max-running 1, resuming via done.txt. Resumable.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
NGPU=${NGPU:-8}; INDEX=logs/c1b/trainset/fresh/index.jsonl; PIDS=${PIDS:-logs/phase5/record_pids.json}
OUT=${OUT:-logs/phase5/judge_sglang/record}
SHARDS=${SHARDS:-$(seq 0 $((NGPU-1)))}   # which shard indices (= GPU ids) to launch/supervise; pids split NGPU-way
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$ROOT/code
. logs/phase5/sgl_setup/sgl_env.sh 2>/dev/null
mkdir -p "$OUT"
say(){ echo "[rec-sup $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
mapfile -t NEED < <(/venv/p2/bin/python - "$PIDS" "$NGPU" <<'PY'
import json, sys
p = sorted(json.load(open(sys.argv[1]))["pids"]); n = int(sys.argv[2])
for k in range(n): print(len(p[k::n]))
PY
)
launch(){ local k=$1 mr=$2
  # free the shard's drafter port from a crashed predecessor, then start the record process
  for q in $(ps -eo pid,args --no-headers | grep -E "launch_server .*--port 3010$k( |$)" | grep -v grep | awk '{print $1}'); do kill -9 "$q" 2>/dev/null; done
  ( export CUDA_VISIBLE_DEVICES=$k SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296
    /venv/sgl/bin/python code/judge_round/judge_sglang.py --mode record --index "$INDEX" --pids "$PIDS" \
      --shard "$k" --nshards "$NGPU" --gpu "$k" --redraft-every 1 --redraft-ban --loop-guard 12 \
      --max-running "$mr" --out "$OUT" --mem-fraction 0.70 --drafter-mem-fraction 0.10 ) > "$OUT/record_gpu${k}.log" 2>&1
  echo $? > "$OUT/gpu${k}.rc"; }
declare -A PID MR
for k in $SHARDS; do MR[$k]=64; launch "$k" 64 & PID[$k]=$!; say "shard $k launched (max-running 64)"; done
while true; do
  alldone=1
  for k in $SHARDS; do
    [ "${MR[$k]}" = done ] && continue
    if ! kill -0 "${PID[$k]}" 2>/dev/null; then
      ndone=$(grep -c . "$OUT/gpu$k/done.txt" 2>/dev/null || echo 0)
      if [ "$ndone" -ge "${NEED[$k]}" ]; then MR[$k]=done; say "shard $k DONE ($ndone/${NEED[$k]})"; continue; fi
      if [ "${MR[$k]}" = 64 ]; then say "shard $k crashed at mr64 ($ndone/${NEED[$k]}); relaunch at mr1"; MR[$k]=1; launch "$k" 1 & PID[$k]=$!
      else say "shard $k exited at mr1 ($ndone/${NEED[$k]}); relaunch at mr1"; launch "$k" 1 & PID[$k]=$!; fi
    fi
    [ "${MR[$k]}" = done ] || alldone=0
  done
  [ "$alldone" = 1 ] && { say "ALL shards done"; break; }
  sleep 20
done
