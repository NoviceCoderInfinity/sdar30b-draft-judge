#!/usr/bin/env bash
# One-shot probe wave on ALL GPUs after training is stopped (2026-09-15 deadline plan): for each given step dir,
# judge / judge+DES32 / judge+DES48 on N-item uniform subsets of GSM8K-500, MBPP, HumanEval, decoded without --score
# from a sandbox cwd, then scored as user r3score, then probe_summary_r3.py (6-arm table vs the team's stock records).
#   bash code/draftalign/probe_wave_r3.sh r3_A 60 step000900 step001000
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
set -a; . ./.env; set +a; export HF_HOME=/root/.cache/huggingface; PY=/venv/r3/bin/python
RUN="$1"; N="$2"; shift 2; PR=logs/draftalign/probes_r3/$RUN; Q=logs/draftalign/queue_probe_wave.txt; : > "$Q"
job() {  # step ds:N arm  -> one queue line
  local st="$1" dsn="$2" arm="$3" ds="${2%%:*}" pool=""
  [ "$arm" = judge_des32 ] && pool="--pool 32 --pool-rule des --pool-flush"; [ "$arm" = judge_des48 ] && pool="--pool 48 --pool-rule des --pool-flush"
  local d="$ROOT/$PR/$st/$ds/$arm" sb="/dev/shm/dlm/sandbox/probe_${RUN}_${st}_${ds}_$arm"
  echo "mkdir -p $d $sb && cd $sb && set -a && . $ROOT/.env && set +a && export HF_HOME=/root/.cache/huggingface && CUDA_VISIBLE_DEVICES={GPU} $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --drafter d32 --items $ds:0:${dsn#*:} --adapter $ROOT/logs/draftalign/train/$RUN/$st --judge full --draft s8 $pool --shard 0 --nshards 1 --gpu {GPU} --no-trace --out $d > $d/decode.log 2>&1 && bash $ROOT/code/draftalign/run_score_r3.sh $d once $d/scored >> $d/score.log 2>&1" >> "$Q"
}
# long jobs first (final checkpoint only): AIME all 90 (judge, judge+DES48), GPQA-Diamond 60-item subset (judge+DES48)
FINAL="$1"; job "$FINAL" aime:90 judge_des48; job "$FINAL" aime:90 judge; job "$FINAL" gpqa:60 judge_des48
for st in "$@"; do for ds in gsm8k500 mbpp humaneval; do for arm in judge judge_des32 judge_des48; do
  job "$st" "$ds:$N" "$arm"
done; done; done
: > "$Q.lo"; rm -f logs/draftalign/probe_wave.log.stop
bash code/draftalign/sched_r3.sh "$Q" "$Q.lo" logs/draftalign/probe_wave.log
"$PY" code/draftalign/probe_summary_r3.py "$PR" | tee "$PR/summary.txt"
