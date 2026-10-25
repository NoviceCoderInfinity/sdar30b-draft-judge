#!/usr/bin/env bash
# Restart r3_B right after its next checkpoint save and resume from it with calibration logging on (user, 2026-09-15
# ~12:30 IST, "strategy 3": no wasted steps). Training recipe, data order, optimizer and lr schedule are unchanged --
# --resume restores the adapter, AdamW state and step; only --calib-log / --calib-eval-* are added.
#   nohup bash code/draftalign/restart_r3B_calib.sh [step000300] > logs/draftalign/restart_r3B_calib.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN=logs/draftalign/train/r3_B; TARGET=${1:-step000300}
ist() { TZ=Asia/Kolkata date '+%H:%M:%S IST'; }

# 1. pause the post-training eval watcher: it would otherwise take the restart for the end of training
W=$(ps -eo pid,args | awk '/[e]val_after_r3B[.]sh/ {print $1}')
[ -n "$W" ] && kill $W && echo "$(ist) paused eval watcher pid(s) $W"

# 2. wait for the checkpoint save to be COMPLETE (the [save] line is printed after adapter + trainer_state + latest)
echo "$(ist) waiting for $TARGET to be saved"
until grep -q "^\[save\] .*/r3_B/$TARGET" "$RUN/train.log"; do sleep 3; done
[ -f "$RUN/$TARGET/adapter.safetensors" ] && [ -f "$RUN/$TARGET/trainer_state.pt" ] || { echo "$(ist) $TARGET incomplete -- abort, training untouched"; exit 1; }
echo "$(ist) $TARGET saved; stopping training"
TR=$(ps -eo pid,args | awk '/[t]orchrun .*--out logs\/draftalign\/train\/r3_B / {print $1}')
[ -n "$TR" ] && kill -INT $TR
for i in $(seq 1 90); do ps -eo args | grep -q "[d]raftalign/train[.]py.*r3_B" || break; sleep 2; done
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" = 0 ] && break; sleep 2; done
echo "$(ist) training stopped (last logged: $(grep '^\[step' "$RUN/train.log" | tail -1 | cut -c1-20)); GPUs free"

# 3. resume from TARGET with the same recipe + calibration logging
setsid nohup /venv/r3/bin/torchrun --standalone --nproc_per_node=8 code/draftalign/train.py \
  --index logs/draftalign/trainset_r3/index.jsonl --out "$RUN" --resume "$RUN/$TARGET" --steps 710 --grad-accum 4 \
  --pool-Ms 128 --no-train-gates --draft-flag --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 \
  --max-seq 1200 --holdout-items 150 --eval-every 100 --save-every 100 --log-every 10 --seed 1 --keep-checkpoints 0 \
  --beta 0.5 --calib 1.0 --override-weight 2.0 --code-override-weight 3.0 \
  --calib-log --calib-eval-states 64 --calib-eval-extra 16 --calib-eval-at-start \
  >> "$RUN/train.log" 2>&1 < /dev/null &
for i in $(seq 1 90); do [ "$(ps -eo args | grep -c '[d]raftalign/train[.]py.*r3_B')" -ge 8 ] && break; sleep 2; done
echo "$(ist) relaunched: $(ps -eo args | grep -c '[d]raftalign/train[.]py.*r3_B') rank processes"

# 4. re-arm the post-training eval watcher now that training is running again
S=eval_after_r3B
setsid nohup bash "code/draftalign/${S}.sh" 17:30 >> "logs/draftalign/${S}.out" 2>&1 < /dev/null &
sleep 2; echo "$(ist) eval watcher re-armed: $(ps -eo pid,args | awk '/[e]val_after_r3B[.]sh/ {print $1}')"

# 5. report the first resumed lines
for i in $(seq 1 150); do grep -q "^\[calib-eval $((10#${TARGET#step}))\]\|Traceback" <(sed -n '/\[resume\] from step/,$p' "$RUN/train.log") && break; sleep 4; done
sed -n '/\[resume\] from step/,$p' "$RUN/train.log" | grep -E "^\[resume|^\[init|^\[draft-flag|^\[calib-eval|^\[step|Traceback|Error" | tail -8
