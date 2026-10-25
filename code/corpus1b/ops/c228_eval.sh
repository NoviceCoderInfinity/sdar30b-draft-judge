#!/usr/bin/env bash
# Evaluate the newest main_228m checkpoint with the BASE decode (SDAR-1.7B drafter + the judge, --judge full --draft s8,
# tau 0.95, NO expert pool) -- the setup Phase 1's 128M no-pool numbers and Phase 3's "base" used -- then compare per item
# with the 128M judge and with stock. Called by c228_run.sh:  bash code/corpus1b/ops/c228_eval.sh <full|quick> <cutoff unix>
#   full : IFEval-541, MATH-500, GPQA-Diamond, AIME, GSM8K-500, MBPP, HumanEval (44 shards, longest first, ~2.5 h)
#   quick: GSM8K-500, HumanEval, MBPP (16 shards, ~0.5 h)
# At the cutoff, unfinished shards are killed and the summary covers the finished items (each shard is a stride sample).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
set -a; . ./.env; set +a
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
MODE=${1:-full}; CUT=${2:-0}
say(){ echo "[c228-eval $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
OUT=logs/c1b/eval/checkpoint228M_nopool; CK=logs/c1b/train/main_228m/eval_checkpoint; mkdir -p $OUT/logs
SRC=$(python3 -c "import glob, json, os; c = [(json.load(open(f))['tokens'], os.path.dirname(f)) for f in glob.glob('logs/c1b/train/main_228m/final/trainer_state.json') + glob.glob('logs/c1b/train/main_228m/tok_*/trainer_state.json') + glob.glob('logs/c1b/train/main_228m/roll_*/trainer_state.json')]; print(max(c)[1] if c else '')")
[ -n "$SRC" ] || { say "no main_228m checkpoint to evaluate"; exit 1; }
rm -rf $CK.tmp && cp -r "$SRC" $CK.tmp && rm -rf $CK && mv $CK.tmp $CK
echo "$CK (copy of $(basename $SRC)) adapter sha256 $(sha256sum $CK/adapter.safetensors | cut -c1-16), $(python3 -c "import json; s=json.load(open('$CK/trainer_state.json')); print(f\"{s['tokens']:.0f} tokens, step {s['step']}\")")" > $OUT/checkpoint.txt
say "evaluating $(cat $OUT/checkpoint.txt) [$MODE]"
if [ "$MODE" = quick ]; then SPECS="gsm8k500:8 mbpp:4 humaneval:4"; else SPECS="ifeval541:8 math500:8 gpqa:8 aime:4 gsm8k500:8 mbpp:4 humaneval:4"; fi
Q=$OUT/queue.txt; : > $Q
J="--drafter d32 --judge full --draft s8 --adapter $ROOT/$CK"
for spec in $SPECS; do s=${spec%%:*}; n=${spec##*:}
  for k in $(seq 0 $((n-1))); do c=${s}_judge
    echo "mkdir -p /dev/shm/c1b_sandbox/eval/c228/${c}_$k && cd /dev/shm/c1b_sandbox/eval/c228/${c}_$k && CUDA_VISIBLE_DEVICES={GPU} /venv/train/bin/python $ROOT/code/draftalign/run_da_decode.py --model b32 --items $s $J --shard $k --nshards $n --gpu {GPU} --score --no-trace --out $ROOT/$OUT/$c >> $ROOT/$OUT/logs/${c}_$k.log 2>&1" >> $Q
  done; done
cp $Q $Q.bak; say "$(wc -l < $Q) jobs"
nohup bash code/draftalign/gpu_queue.sh $Q $OUT/gpu_queue.log "0 1 2 3 4 5 6 7" > /dev/null 2>&1 &
sleep 90
while [ -f $Q ] || [ "$(ps -eo args | grep -c "[r]un_da_decode.py.*$OUT/")" -gt 0 ]; do
  if [ "$CUT" -gt 0 ] && [ "$(date +%s)" -ge "$CUT" ]; then
    say "cutoff reached: no more launches, killing unfinished shards (summary covers finished items)"
    cp $Q $OUT/not_launched.txt 2>/dev/null; rm -f $Q
    pkill -f "[r]un_da_decode.py.*$OUT/"; sleep 20; pkill -9 -f "[r]un_da_decode.py.*$OUT/"; break
  fi
  sleep 30
done
/venv/train/bin/python code/corpus1b/eval_summary.py --dir $OUT && say "summary -> $OUT/summary.md"
/venv/train/bin/python code/corpus1b/ops/c228_compare.py --dir $OUT && say "comparison -> $OUT/compare.md"
say "EVAL DONE"
