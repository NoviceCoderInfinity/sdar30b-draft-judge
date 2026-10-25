#!/usr/bin/env bash
# Asynchronous checkpoint probes on the EVAL GPUs (training never waits). One worker per GPU; each takes the
# oldest un-probed step<N> dir of <run> (FIFO) and runs, for that adapter:
#   1. judge decode (setting 3: --judge full --draft s8, c95) on 100-item uniform subsets of GSM8K-500, MBPP,
#      HumanEval -- WITHOUT --score, cwd = sandbox (code-execution footgun), then scored as user r3score;
#   2. per-domain holdout diagnostics (eval_holdout_r3.py);
#   3. probe_summary_r3.py -> summary.jsonl + best.json (read by ckpt_sync_r3.sh).
#   nohup bash code/draftalign/probe_r3.sh <run_dir> <index> "6 7" > logs/draftalign/probe_r3.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN="$1"; INDEX="$2"; GPUS=(${3:-6 7}); NAME=$(basename "$RUN"); PR=logs/draftalign/probes_r3/$NAME
set -a; . ./.env; set +a; export HF_HOME=/root/.cache/huggingface; PY=/venv/r3/bin/python
mkdir -p "$PR"
probe_one() {   # $1 = step dir, $2 = gpu
  local sd="$1" g="$2" st; st=$(basename "$sd"); local out="$PR/$st"; mkdir -p "$out"
  for ds in gsm8k500 mbpp humaneval; do for arm in judge judge_des48; do
    local d="$ROOT/$out/$ds/$arm" sb="/dev/shm/dlm/sandbox/probe_${NAME}_${st}_${ds}_$arm"; mkdir -p "$d" "$sb"
    local pool=""; [ "$arm" = judge_des48 ] && pool="--pool 48 --pool-rule des --pool-flush"
    [ -e "$d/scored" ] && [ -n "$(ls "$d/scored"/*.jsonl.done 2>/dev/null)" ] && continue
    (cd "$sb" && CUDA_VISIBLE_DEVICES=$g "$PY" "$ROOT/code/draftalign/run_da_decode.py" --model b32 --drafter d32 \
       --items "$ds:0:100" --adapter "$ROOT/$sd" --judge full --draft s8 $pool --shard 0 --nshards 1 --gpu "$g" \
       --no-trace --out "$d" > "$d/decode.log" 2>&1)
    bash code/draftalign/run_score_r3.sh "$d" once "$d/scored" >> "$d/score.log" 2>&1
  done; done
  CUDA_VISIBLE_DEVICES=$g "$PY" code/draftalign/eval_holdout_r3.py --index "$INDEX" --adapter "$sd" \
     --out "$PR/holdout.jsonl" > "$out/holdout.log" 2>&1
  "$PY" code/draftalign/probe_summary_r3.py "$PR" > "$PR/summary.txt" 2>&1
  echo "$(date -u +%FT%TZ) GPU $g probed $st" >> "$PR/probe.log"
}
worker() {
  local g="$1"
  while :; do
    local sd=""
    for c in $(ls -d "$RUN"/step* 2>/dev/null | sort); do
      [ -f "$c/adapter.safetensors" ] || continue; [ $(( $(date +%s) - $(stat -c %Y "$c/adapter.safetensors") )) -lt 90 ] && continue
      mkdir "$PR/.claim_$(basename "$c")" 2>/dev/null && { sd="$c"; break; }
    done
    if [ -n "$sd" ]; then probe_one "$sd" "$g"; else [ -e "$PR/stop" ] && return; sleep 120; fi
  done
}
for g in "${GPUS[@]}"; do worker "$g" & done; wait
