#!/bin/bash
# Training-data generation with the training-free DV-blind decoder (targets = accepted draft tokens where the MoE's
# own marginal agrees, stock elsewhere). Usage: launch_datagen.sh <gamma e.g. 50> <items e.g. gsm8k_pool:0:3000> <gpus "0 1 2 3 4 5 6 7">
# One shard per GPU; records + traces under logs/draftalign/datagen/<cell>/ ; scored inline (--score).
set -euo pipefail
Q=${1:?gamma, e.g. 50}; ITEMS=${2:?items split, e.g. gsm8k_pool:0:3000}; GPUS=${3:-"0 1 2 3 4 5 6 7"}
cd "$(dirname "$0")/../.."
read -ra G <<< "$GPUS"; N=${#G[@]}
TAG=$(echo "$ITEMS" | tr ':' '_')
OUT=logs/draftalign/datagen/${TAG}_dv-s8_blind_q$Q
mkdir -p "$OUT"
for k in $(seq 0 $((N-1))); do
  g=${G[$k]}
  CUDA_VISIBLE_DEVICES=$g nohup /venv/main/bin/python code/draftalign/run_da_decode.py \
    --model b32 --drafter d32 --items "$ITEMS" --draft s8 --judge blind --tau-acc-q .$Q \
    --shard $k --nshards $N --gpu $g --score --out "$OUT" > "$OUT/shard${k}of${N}.log" 2>&1 &
done
echo "launched $N shards -> $OUT"
