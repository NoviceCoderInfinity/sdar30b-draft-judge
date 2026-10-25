#!/usr/bin/env bash
# Evaluation queues for the judge checkpoints, on the existing harness (run_da_decode.py + gpu_queue.sh).
#   bash code/corpus1b/eval_ckpt.sh checkpointA     # after wave A
#   bash code/corpus1b/eval_ckpt.sh tau32M          # the 32M drafter-confidence test (extra work, only if time)
#   bash code/corpus1b/eval_ckpt.sh final           # after wave B
# checkpointA, launched in this order (the queue is FIFO over free GPUs):
#   1. gsm8k500 (the gate), sharded over all 8 GPUs so it finishes first;
#   2. humaneval, mbpp, math500, aime;  3. gpqa, ifeval541, finqa500, chartqa300   (judge + DES48 at checkpoint A:
#      main/tok_000128000000, or main/final when the index held fewer than 128M tokens);
#   4. the released model (stock) on ifeval541, finqa500, chartqa300.
# tau32M: main tok_32M vs conf32 tok_32M, judge alone, tau 0.90/0.95/0.98 on mbpp + humaneval (into checkpointA/).
# final:  judge(1B) with and without DES48 on every set + chat800; the released model (+DES48) on the new sets.
# Deadline (DEADLINE_FILE, default /workspace/deadline.env): nothing is launched after DEADLINE - 45 min (the queue
# file is removed; its .bak stays) and whatever still runs at DEADLINE is killed; the summary then covers what
# finished (partial cells keep their n) and gate_report.py lists what was cut or never launched.
# Code sets run in /dev/shm sandboxes (AGENTS.md 9(p)); records land in logs/c1b/eval/<tag>/<cell>/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home}
DEADLINE_FILE=${DEADLINE_FILE:-/workspace/deadline.env}
DEADLINE=""; [ -f "$DEADLINE_FILE" ] && . "$DEADLINE_FILE"
TAG=$1; PYT=/venv/train/bin/python
case $TAG in tau32M) OUT=logs/c1b/eval/checkpointA ;; *) OUT=logs/c1b/eval/$TAG ;; esac
mkdir -p $OUT/logs; Q=$OUT/queue_$TAG.txt; : > $Q
MAIN=$ROOT/logs/c1b/train/main; FINAL=$ROOT/logs/c1b/train/main_b/final; CONF=$ROOT/logs/c1b/train/conf32
say(){ echo "[eval $TAG $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }

# shards per set: the gate uses the whole box; the rest roughly by per-item cost, so a cut queue still leaves
# stride-sharded (uniform) partial samples
declare -A NS=([gsm8k500]=8 [math500]=8 [aime]=4 [humaneval]=4 [mbpp]=4 [gpqa]=8 [ifeval541]=8 [finqa500]=8 \
               [chartqa300]=4 [chat800]=8)
[ "$TAG" = "tau32M" ] && NS=([humaneval]=1 [mbpp]=1)
shard_line(){  # $1 name, $2 items, $3 shard, $4.. run_da_decode args
  local name=$1 items=$2 k=$3; shift 3; local n=${NS[$items]:-2}
  local box=/dev/shm/c1b_sandbox/eval/$TAG/${name}_$k
  echo "mkdir -p $box && cd $box && CUDA_VISIBLE_DEVICES={GPU} $PYT $ROOT/code/draftalign/run_da_decode.py --model b32 --items $items $* --shard $k --nshards $n --gpu {GPU} --score --no-trace --out $OUT/$name > $ROOT/$OUT/logs/${name}_$k.log 2>&1" >> $Q
}
line(){  # $1 name, $2 items, $3.. run_da_decode args: every shard of the set
  local name=$1 items=$2; shift 2; local n=${NS[$items]:-2}
  for k in $(seq 0 $((n-1))); do shard_line $name $items $k "$@"; done
}
JUDGE="--drafter d32 --judge full --draft s8"
DES="--pool 48 --pool-rule des --pool-flush"

if [ "$TAG" = "checkpointA" ]; then
  CK=$MAIN/tok_000128000000; [ -d $CK ] || CK=$MAIN/final
  [ -d $CK ] || { say "no checkpoint A ($MAIN/tok_000128000000 or final)"; exit 1; }
  echo "$CK" > $OUT/checkpoint.txt
  line "gsm8k500_judge128M_des48" gsm8k500 $JUDGE --adapter $CK $DES          # 1. the gate, on the whole box
  # 2-4 breadth-first: shard k of every set, in the group order (2: humaneval mbpp math500 aime; 3: gpqa ifeval541
  # finqa500 chartqa300; 4: the released model on the three new sets), before shard k+1 of any set, so a deadline
  # cut still leaves a uniform (stride-sharded) sample of EVERY set
  for k in 0 1 2 3 4 5 6 7; do
    for s in humaneval mbpp math500 aime gpqa ifeval541 finqa500 chartqa300; do
      [ $k -lt ${NS[$s]} ] && shard_line "${s}_judge128M_des48" $s $k $JUDGE --adapter $CK $DES
    done
    for s in ifeval541 finqa500 chartqa300; do [ $k -lt ${NS[$s]} ] && shard_line "${s}_released" $s $k --stock; done
  done
elif [ "$TAG" = "tau32M" ]; then
  for t in 0.90 0.95 0.98; do
    for s in mbpp humaneval; do
      line "${s}_main32M_c${t}" $s $JUDGE --adapter $MAIN/tok_000032000000 --tau-commit $t
      line "${s}_conf32M_c${t}" $s $JUDGE --adapter $CONF/tok_000032000000 --tau-commit $t
    done
  done
elif [ "$TAG" = "final" ]; then
  for s in gsm8k500 math500 aime humaneval mbpp gpqa ifeval541 finqa500 chartqa300 chat800; do
    line "${s}_judge1B_des48" $s $JUDGE --adapter $FINAL $DES
    line "${s}_judge1B" $s $JUDGE --adapter $FINAL
  done
  for s in ifeval541 finqa500 chartqa300 chat800; do
    line "${s}_released" $s --stock
    line "${s}_released_des48" $s --stock $DES
  done
else
  echo "unknown tag $TAG"; exit 1
fi
cp $Q $Q.bak                                     # gpu_queue.sh deletes its queue when it drains (AGENTS.md 9(c))
say "$(wc -l < $Q) jobs"

# deadline guard: stop launching at DEADLINE - 45 min, kill this queue's jobs at DEADLINE
if [ -n "$DEADLINE" ]; then
  if [ "$(date +%s)" -ge $(( DEADLINE - 2700 )) ]; then
    say "past DEADLINE - 45 min: launching nothing"; rm -f $Q
  fi
  ( while [ "$(date +%s)" -lt $(( DEADLINE - 2700 )) ]; do sleep 30; done
    [ -f $Q ] && { echo "$(date -u +%FT%TZ) DEADLINE - 45 min: no further launches; not launched: $(wc -l < $Q) lines" >> $OUT/gpu_queue.log; cp $Q $OUT/not_launched_$TAG.txt; rm -f $Q; }
    while [ "$(date +%s)" -lt $DEADLINE ]; do sleep 20; done
    pkill -f "[r]un_da_decode.py.*$OUT/" && echo "$(date -u +%FT%TZ) DEADLINE: killed the running eval jobs" >> $OUT/gpu_queue.log || true
  ) &
  GUARD=$!
fi
[ -f $Q ] && bash code/draftalign/gpu_queue.sh $Q $OUT/gpu_queue.log "0 1 2 3 4 5 6 7"
# gpu_queue exits when the queue is empty (or removed); wait for the last jobs (bracketed pattern: AGENTS.md 9(d))
while [ "$(ps -eo args | grep -c "[r]un_da_decode.py.*$OUT/")" -gt 0 ]; do sleep 30; done
[ -n "${GUARD:-}" ] && kill $GUARD 2>/dev/null || true
$PYT code/corpus1b/eval_summary.py --dir $OUT
say "summary -> $OUT/summary.md"
