#!/usr/bin/env bash
# Print queue lines for sched_r3.sh: one decode per shard of <items>, S shards (stride-sharded, so any finished
# subset of shards is a uniform subsample of the slice). Same decode command as launch_r3_datagen.sh.
#   bash code/draftalign/queue_r3_lines.sh gsm8k_pool:2400:4400 gsm8k_pool_2400_4400 32          >> <queue>
#   bash code/draftalign/queue_r3_lines.sh opencode_pool:0:6000 opencode_pool_0_6000 32 noscore  >> <queue>
# SAFETY: --score executes generated code (oc_datasets/mbpp.py: child process + exec, no chdir, same user).
#   - every shard runs with cwd = its own throwaway dir under /dev/shm/dlm/sandbox/ (never the repo root)
#   - code pools pass "noscore": the GPU process never executes generated code; scoring happens afterwards
#     as an unprivileged user (code/draftalign/run_score_r3.sh), with the identical per_item_correct call.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ITEMS="$1"; NAME="$2"; S="$3"; MODE="${4:-score}"
SCORE="--score"; [ "$MODE" = "noscore" ] && SCORE=""
OUT=$ROOT/logs/draftalign/datagen/${NAME}_dv-s8_blind_q50_r3
for k in $(seq 0 $((S-1))); do
  SB=/dev/shm/dlm/sandbox/${NAME}/shard${k}of${S}
  echo "mkdir -p $SB $OUT && cd $SB && set -a && . $ROOT/.env && set +a && export HF_HOME=/root/.cache/huggingface && CUDA_VISIBLE_DEVICES={GPU} /venv/r3/bin/python $ROOT/code/draftalign/run_da_decode.py --model b32 --drafter d32 --items $ITEMS --draft s8 --judge blind --tau-acc-q .50 --shard $k --nshards $S --gpu {GPU} ${SCORE} --out $OUT > $OUT/shard${k}of${S}.log 2>&1"
done
