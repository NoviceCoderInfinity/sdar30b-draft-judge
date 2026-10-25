#!/usr/bin/env bash
# After TRAIN 128M DONE: freeze the newest main_128m checkpoint, evaluate the judge on the checkpointB protocol
# (GSM8K-500, MATH-500, MBPP, HumanEval, GPQA-Diamond, AIME at the standard 1,024 tokens, IFEval-541; DES48, s8, tau .95),
# 44 shards longest first on 8 GPUs; stock/blind are reused from RESULTS.md and checkpointB. Then summary + HF push.
cd /workspace/dmge; set -a; . ./.env; set +a; export HF_HOME=/workspace/dmge/hf_home
say(){ echo "[eval-128m $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*"; }
until grep -q 'TRAIN 128M DONE' logs/c1b/after_gen_128m.log 2>/dev/null; do sleep 30; done
OUT=logs/c1b/eval/checkpoint128M; CK=logs/c1b/train/main_128m/eval_checkpoint128M; mkdir -p $OUT/logs
SRC=$(python3 -c "import glob, json, os; c = [(json.load(open(f))['tokens'], os.path.dirname(f)) for f in glob.glob('logs/c1b/train/main_128m/final/trainer_state.json') + glob.glob('logs/c1b/train/main_128m/tok_*/trainer_state.json') + glob.glob('logs/c1b/train/main_128m/roll_*/trainer_state.json')]; print(max(c)[1] if c else '')")
[ -n "$SRC" ] || { say "no checkpoint"; exit 1; }
rm -rf $CK.tmp && cp -r $SRC $CK.tmp && rm -rf $CK && mv $CK.tmp $CK
echo "$CK (copy of $(basename $SRC)) adapter sha256 $(sha256sum $CK/adapter.safetensors | cut -c1-16), $(python3 -c "import json; s=json.load(open('$CK/trainer_state.json')); print(f\"{s['tokens']:.0f} tokens, step {s['step']}\")")" > $OUT/checkpoint.txt
say "evaluating $(cat $OUT/checkpoint.txt)"
Q=$OUT/queue.txt; : > $Q
J="--drafter d32 --judge full --draft s8 --adapter /workspace/dmge/$CK --pool 48 --pool-rule des --pool-flush"
for spec in ifeval541:8 math500:8 gpqa:8 aime:4 gsm8k500:8 mbpp:4 humaneval:4; do s=${spec%%:*}; n=${spec##*:}
  for k in $(seq 0 $((n-1))); do c=${s}_judge_des48
    echo "mkdir -p /dev/shm/c1b_sandbox/eval/c128/${c}_$k && cd /dev/shm/c1b_sandbox/eval/c128/${c}_$k && CUDA_VISIBLE_DEVICES={GPU} /venv/train/bin/python /workspace/dmge/code/draftalign/run_da_decode.py --model b32 --items $s $J --shard $k --nshards $n --gpu {GPU} --score --no-trace --out $OUT/$c > /workspace/dmge/$OUT/logs/${c}_$k.log 2>&1" >> $Q
  done; done
cp $Q $Q.bak; say "$(wc -l < $Q) jobs"
bash code/draftalign/gpu_queue.sh $Q $OUT/gpu_queue.log "0 1 2 3 4 5 6 7"
while [ "$(ps -eo args | grep -c "[r]un_da_decode.py.*$OUT/")" -gt 0 ]; do sleep 30; done
/venv/train/bin/python code/corpus1b/eval_summary.py --dir $OUT && say "summary -> $OUT/summary.md"
DATA_TIMEOUT=1200 bash code/corpus1b/backup_hf.sh once && say "HF push done"
say "EVAL 128M DONE"
