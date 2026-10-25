#!/bin/bash
# Round 2b + round-3 drafter, v2: wait for the round-2 data and an empty queue; build the judge union index and the
# drafter set; train the DRAFTER on GPU 0 (1 GPU) while T2b (resume 2a) trains on GPUs 1-7; evaluate drafter
# agreement (GPU 0) when it finishes; evaluate T2b four ways when it finishes. Log: logs/draftalign/orchestrate_round1.log
set -u; cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log; PY=/venv/main/bin/python; TR=/venv/main/bin/torchrun
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
D=logs/draftalign/datagen
log "orchestrator r2b-v2 started (pid $$): waiting for round-2 data and an empty queue"
until [ "$(ls $D/math_pool_0_7500_3_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 8 ] && [ "$(ls $D/mbpp_pool_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 2 ] && [ "$(ls $D/gsm8k_pool_3000_7273_dv-s8_blind_q50/*.done 2>/dev/null | wc -l)" -ge 8 ]; do sleep 300; done
log "round-2 data complete"
until [ ! -f logs/draftalign/queue.txt ] && [ "$(pgrep -fc 'run_da_[d]ecode.py')" -eq 0 ]; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
log "queue drained; building the judge union index and the drafter set"
code/draftalign/build_union_index.sh >> "$LOG" 2>&1 || { log "union index FAILED"; exit 1; }
$PY code/draftalign/build_drafter_set.py --source "$D/gsm8k_pool_0_3000_dv-s8_blind_q50=gsm8k_pool:0:3000" --source "$D/gsm8k_pool_3000_7273_dv-s8_blind_q50=gsm8k_pool:3000:7273" --source "$D/math_pool_0_7500_3_dv-s8_blind_q50=math_pool:0:7500:3" --source "$D/mbpp_pool_dv-s8_blind_q50=mbpp_pool" --out logs/draftalign/drafterset_union >> "$LOG" 2>&1 || { log "drafter set FAILED"; exit 1; }
NIT=$(wc -l < logs/draftalign/drafterset_union/index.jsonl); DSTEPS=$(( (2 * NIT + 15) / 16 )); [ "$DSTEPS" -lt 200 ] && DSTEPS=200
log "drafter set: $NIT items -> $DSTEPS steps (2 epochs, batch 16); launching drafter training on GPU 0 and T2b on GPUs 1-7"
mkdir -p logs/draftalign/train_drafter logs/draftalign/train
CUDA_VISIBLE_DEVICES=0 nohup $TR --standalone --nproc_per_node=1 --master_port 29601 code/draftalign/train_drafter.py --index logs/draftalign/drafterset_union/index.jsonl --out logs/draftalign/train_drafter/d1 --steps $DSTEPS --grad-accum 16 --lr 1e-5 --seed 0 > logs/draftalign/train_drafter/d1.log 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup $TR --standalone --nproc_per_node=7 --master_port 29602 code/draftalign/train.py --index logs/draftalign/trainset_union_r2/index.jsonl --out logs/draftalign/train/t2b_union_flag --resume logs/draftalign/train/t2a_gsm8k_flag/step001057 --steps 3057 --grad-accum 5 --pool-Ms 128 --no-train-gates --draft-flag --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 --max-seq 1200 --holdout-items 150 --eval-every 100 --save-every 200 --log-every 10 --seed 1 --keep-checkpoints 2 > logs/draftalign/train/t2b_union_flag.log 2>&1 &
log "T2b launched (7 GPUs, global batch 35, resume 2a@1057 -> 3057)"
# drafter: wait, then agreement eval on GPU 0
until [ "$(pgrep -fc 'train_[d]rafter.py')" -eq 0 ]; do sleep 60; done
CK=$(ls -d logs/draftalign/train_drafter/d1/step* 2>/dev/null | sort | tail -1)
log "drafter training finished (ckpt $CK); evaluating agreement vs stock on GPU 0"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 1000 ]; do sleep 10; done
mkdir -p results/metrics/draftalign
CUDA_VISIBLE_DEVICES=0 $PY code/draftalign/eval_drafter_agreement.py --index logs/draftalign/drafterset_union/index.jsonl --ckpt "$CK" --baseline d32 --n-items 100 --seed 0 --out results/metrics/draftalign/drafter_agreement_d1.json > logs/draftalign/train_drafter/d1_agreement.log 2>&1
log "drafter agreement eval exit $? (results/metrics/draftalign/drafter_agreement_d1.json)"
# T2b: wait, then the four evals on 8 GPUs
T2=logs/draftalign/train/t2b_union_flag
while true; do
  [ -d $T2/step003057 ] && [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && break
  if [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ]; then sleep 120; [ "$(pgrep -fc 'train.py --index')" -eq 0 ] && [ ! -d $T2/step003057 ] && { log "T2b FAILED: trainer gone without the final checkpoint"; exit 1; }; fi
  sleep 60
done
log "T2b finished"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)" -eq 0 ]; do sleep 30; done
CK2=$T2/step003057
wave(){ for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK2 --judge full --draft s8 $2 --shard $k --nshards 4 --gpu $k --score > logs/draftalign/train/eval_t2b_$1.shard${k}of4.log 2>&1 &
    g=$((k+4)); CUDA_VISIBLE_DEVICES=$g nohup $PY code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter $CK2 --judge full --draft s8 $4 --shard $k --nshards 4 --gpu $g --score > logs/draftalign/train/eval_t2b_$3.shard${k}of4.log 2>&1 &
  done; sleep 120; until [ "$(pgrep -fc 'run_da_decode.py --model b32 --drafter d32 --items calibvalid --adapter')" -eq 0 ]; do sleep 20; done; }
log "T2b evals wave 1: c95 and c90/o99"; wave c95 "" c90o99 "--tau-commit .90 --tau-override .99"
log "T2b evals wave 2: c95/o99 and c90"; wave c95o99 "--tau-override .99" c90 "--tau-commit .90"
log "T2b evals finished (READY)"
