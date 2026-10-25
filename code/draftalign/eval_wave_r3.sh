#!/usr/bin/env bash
# Downstream eval wave after r3_A was stopped at step 1560 (2026-09-15 08:16 IST): all 8 GPUs, one decode per GPU,
# queue lines in PRIORITY order (HumanEval/MBPP first, epoch-2 then epoch-1, stock DES gaps, GSM8K-100, MATH-50,
# then latest step1500). Code sets decode WITHOUT --score from a sandbox cwd and are scored by score_loop below
# as the unprivileged scorer user; GSM8K/MATH score inline (no code exec).
#   nohup bash code/draftalign/eval_wave_r3.sh > logs/draftalign/eval_wave_r3.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PY=/venv/r3/bin/python; PR=$ROOT/logs/draftalign/probes_r3/r3_A; CK=$ROOT/logs/draftalign/eval_ckpts
Q=$ROOT/logs/draftalign/queue_eval_r3.txt; LOG=$ROOT/logs/draftalign/eval_wave_r3.log; : > "$Q"; : > "$Q.lo"; rm -f "$LOG.stop"
line() {  # out_dir items nshards extra_args
  local d="$1" items="$2" n="$3" extra="$4" score=""; case "$items" in gsm8k*|math*) score="--score";; esac
  mkdir -p "$d"
  for k in $(seq 0 $((n-1))); do
    local sb="/dev/shm/dlm/sandbox/eval_$(echo "${d#$PR/}" | tr / _)_s$k"
    echo "mkdir -p $sb && cd $sb && set -a && . $ROOT/.env && set +a && export HF_HOME=/root/.cache/huggingface && CUDA_VISIBLE_DEVICES={GPU} $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --drafter d32 --items $items $extra --shard $k --nshards $n --gpu {GPU} --no-trace $score --out $d > $d/shard${k}of$n.log 2>&1" >> "$Q"
  done
}
pool() { case "$1" in *des32) echo "--pool 32 --pool-rule des --pool-flush";; *des48) echo "--pool 48 --pool-rule des --pool-flush";; esac; }
judge() { for arm in judge judge_des32 judge_des48; do line "$PR/$1/$2/$arm" "$3" "$4" "--adapter $CK/$1 --judge full --draft s8 $(pool $arm)"; done; }
stock() { line "$PR/stock/$1/$2" "$3" "$4" "--stock $(pool $2)"; }
judge step001400 humaneval humaneval 2;  judge step001400 mbpp mbpp 3          # epoch 2
judge step000700 humaneval humaneval 2;  judge step000700 mbpp mbpp 3          # epoch 1
stock humaneval stock_des32 humaneval 2; stock humaneval stock_des48 humaneval 2   # team mirror lacks these
judge step001400 gsm8k500 gsm8k500:0:100 2; judge step000700 gsm8k500 gsm8k500:0:100 2; stock gsm8k500 stock gsm8k500:0:100 2
judge step001400 math500 math500:0:50 2;  judge step000700 math500 math500:0:50 2
judge step001500 humaneval humaneval 2;  judge step001500 mbpp mbpp 3          # latest (lowest priority)
echo "$(date -u +%FT%TZ) queued $(grep -c . "$Q") decode jobs"
bash code/draftalign/sched_r3.sh "$Q" "$Q.lo" "$LOG"
echo "$(date -u +%FT%TZ) scheduler exited"
