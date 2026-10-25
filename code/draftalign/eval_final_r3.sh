#!/usr/bin/env bash
# Final-checkpoint eval after the deadline stop (2026-09-15): HumanEval-164 + MBPP-257 x judge+DES48 / judge / judge+DES32
# on all 8 GPUs, identical decode + sandboxed scoring path to eval_wave_r3.sh. Hard cutoff CUT (IST, default 11:55):
# decodes still running then are killed so the final HF/GitHub push can include whatever finished.
#   nohup bash code/draftalign/eval_final_r3.sh step001900 [11:55] > logs/draftalign/eval_final_r3.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
ST="$1"; CUT=$(TZ=Asia/Kolkata date -d "today ${2:-11:55}" +%s)
PY=/venv/r3/bin/python; PR=$ROOT/logs/draftalign/probes_r3/r3_A; CK=$ROOT/logs/draftalign/eval_ckpts
Q=$ROOT/logs/draftalign/queue_eval_final_r3.txt; LOG=$ROOT/logs/draftalign/eval_final_r3.log; : > "$Q"; : > "$Q.lo"; rm -f "$LOG.stop"
mkdir -p "$CK/$ST" && cp "$ROOT/logs/draftalign/train/r3_A/$ST/adapter.safetensors" "$ROOT/logs/draftalign/train/r3_A/$ST/adapter_config.json" "$CK/$ST/" || { echo "no $ST adapter"; exit 1; }
line() {  # out_dir items nshards extra_args
  local d="$1" items="$2" n="$3" extra="$4"; mkdir -p "$d"
  for k in $(seq 0 $((n-1))); do
    local sb="/dev/shm/dlm/sandbox/evalf_$(echo "${d#$PR/}" | tr / _)_s$k"
    echo "mkdir -p $sb && cd $sb && set -a && . $ROOT/.env && set +a && export HF_HOME=/root/.cache/huggingface && CUDA_VISIBLE_DEVICES={GPU} $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --drafter d32 --items $items $extra --shard $k --nshards $n --gpu {GPU} --no-trace --out $d > $d/shard${k}of$n.log 2>&1" >> "$Q"
  done
}
pool() { case "$1" in *des32) echo "--pool 32 --pool-rule des --pool-flush";; *des48) echo "--pool 48 --pool-rule des --pool-flush";; esac; }
for arm in judge_des48 judge judge_des32; do
  line "$PR/$ST/humaneval/$arm" humaneval 2 "--adapter $CK/$ST --judge full --draft s8 $(pool $arm)"
  line "$PR/$ST/mbpp/$arm" mbpp 3 "--adapter $CK/$ST --judge full --draft s8 $(pool $arm)"
done
echo "$(TZ=Asia/Kolkata date '+%H:%M:%S IST') queued $(grep -c . "$Q") jobs for $ST, cutoff $(TZ=Asia/Kolkata date -d @$CUT '+%H:%M IST')"
bash code/draftalign/sched_r3.sh "$Q" "$Q.lo" "$LOG" &
SCHED=$!
score_pass() {
  for d in $PR/$ST/*/*; do
    [ -d "$d" ] || continue
    n1=$(ls "$d"/*.jsonl.done 2>/dev/null | wc -l); n2=$(ls "$d"/scored/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n1" -gt "$n2" ] && bash code/draftalign/run_score_r3.sh "$d" once "$d/scored" >> "$d/score.log" 2>&1
  done
}
while kill -0 $SCHED 2>/dev/null; do
  if [ "$(date +%s)" -ge "$CUT" ]; then
    echo "$(TZ=Asia/Kolkata date '+%H:%M:%S IST') cutoff: stopping scheduler and remaining $ST decodes"
    touch "$LOG.stop"; pkill -f "run_da_decode[.]py.*eval_ckpts/$ST" ; sleep 5; break
  fi
  score_pass; sleep 45
done
wait $SCHED 2>/dev/null
score_pass
set -a; . ./.env; set +a; $PY code/draftalign/eval_summary_r3.py > /dev/null 2>&1
echo "$(TZ=Asia/Kolkata date '+%H:%M:%S IST') final eval finished; scored shards: $(ls $PR/$ST/*/*/scored/*.jsonl.done 2>/dev/null | wc -l)/15"
