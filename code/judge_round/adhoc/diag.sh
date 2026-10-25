#!/bin/bash
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1
MERGED=$ROOT/logs/phase4/train/main/merged
declare -A AD=( [a128]=$ROOT/logs/phase5/judge/start_128m [b136]=$ROOT/logs/phase5/judge/rec_main/tok_000136000000 [c147]=$ROOT/logs/phase5/judge/rec_main/final [d155]=$ROOT/logs/phase5/judge/rec_main_ep2/final )
G=0
for T in a128 b136 c147 d155; do
  OUT=$ROOT/logs/phase5/eval_diag/$T; mkdir -p $OUT
  for k in 0 1; do
    SB=/dev/shm/p5diag/$T/gpu$k; mkdir -p $SB
    ( cd $SB && CUDA_VISIBLE_DEVICES=$G /venv/p2/bin/python $ROOT/code/draftalign/run_da_decode.py \
      --model b32 --items ifeval541 --drafter $MERGED --draft s8 --judge full --adapter ${AD[$T]} \
      --full-redraft-every 1 --redraft-ban-rejected --loop-guard 12 --shard $k --nshards 2 --gpu $G \
      --score --no-trace --limit 16 --out $OUT ) > $OUT/gpu$k.log 2>&1 &
    G=$((G+1))
  done
done
wait
echo "DIAG DONE $(TZ=Asia/Kolkata date +%H:%M:%S)" >> $ROOT/logs/phase5/eval_diag/done.txt
