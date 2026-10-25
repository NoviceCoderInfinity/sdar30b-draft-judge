#!/usr/bin/env bash
# Round-3 arm A: judge LoRA, fresh from the stock base, on the 24k-block 35/35/30 index (option B), 4 epochs.
# Recipe §4 (TRAINING_RECIPE_2026-09-09.md) with the 2026-09-14 amendments: 6 train GPUs x grad-accum 5 = 30
# states/step (recipe: 8x4=32), 96,000 states / 30 = 3,200 steps, checkpoint + holdout every 100, keep all
# checkpoints on disk (ckpt_sync_r3.sh pushes each to HF and prunes to latest-2 + best). LoRA byte-identical to
# t2a/t2b (r 32, alpha 64, dropout 0, q/k/v/o), lr 2e-5, warmup 50, cosine to 0.5, pool_Ms 128, draft flag on.
# 2026-09-15: ALL 8 GPUs train (server dies 09:20 IST); probes run as a wave after training is stopped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
set -a; . ./.env; set +a; export HF_HOME=/root/.cache/huggingface
RUN=${RUN:-r3_A}; OUT=logs/draftalign/train/$RUN; INDEX=logs/draftalign/trainset_r3/index.jsonl
STEPS=${STEPS:-2175}; ACCUM=${ACCUM:-4}   # 8 ranks x 4 = 32 states/step (recipe-exact); 2,175 steps = 3 epochs of the 23.6k-block index (user, 02:40 IST)
[ -f "$INDEX" ] || { echo "no index at $INDEX"; exit 1; }
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 1; }
# free the training GPUs: stop the datagen scheduler and any arm-C filler decodes (resumable later)
touch logs/draftalign/sched_r3.log.stop; sleep 2
pkill -f "run_da_decode[.]py.*gsm8k_pool:2400:4400" || true; sleep 20
for g in 0 1 2 3 4 5 6 7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$m" -lt 2000 ] || { echo "GPU $g busy ($m MiB)"; exit 1; }
done
# index validation only (loads no model): the sampler must enumerate the examples and the holdout split
/venv/r3/bin/python code/draftalign/train.py --index "$INDEX" --out "$OUT" --dry-run --holdout-items 150 --seed 1 > /dev/shm/dlm/train_dryrun.log 2>&1 \
  || { echo "index validation failed"; tail -5 /dev/shm/dlm/train_dryrun.log; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup /venv/r3/bin/torchrun --standalone --nproc_per_node=8 \
  code/draftalign/train.py --index "$INDEX" --out "$OUT" \
  --steps "$STEPS" --grad-accum "$ACCUM" --pool-Ms 128 --no-train-gates --draft-flag \
  --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 --max-seq 1200 \
  --holdout-items 150 --eval-every 200 --save-every 100 --log-every 10 --seed 1 --keep-checkpoints 0 \
  > "$OUT/train.log" 2>&1 &
echo "$(date -u +%FT%TZ) training launched pid $! -> $OUT"
# probes: no free GPUs while all 8 train (2026-09-15 deadline plan) -- run probe wave after training stops
nohup bash code/draftalign/ckpt_sync_r3.sh "$OUT" > logs/draftalign/ckpt_sync_r3_$RUN.log 2>&1 &
echo "ckpt_sync started"
