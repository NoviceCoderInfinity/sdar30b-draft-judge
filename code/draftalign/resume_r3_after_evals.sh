#!/usr/bin/env bash
# 2026-09-15 (user): pause evals -> plateau probe (one checkpoint per GPU, <=15 min) -> resume r3_A from step001500
# on all 8 GPUs with the original flags (logs appended) -> ckpt_sync to Anupam-Rawat-IITB HF (Nemotron-split-moe-train).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
set -a; . ./.env; set +a; export HF_HOME=/root/.cache/huggingface
LOG=logs/draftalign/resume_r3.log; OUT=logs/draftalign/train/r3_A; P=logs/draftalign/plateau_probe_r3
log() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S IST') $*" | tee -a "$LOG"; }
log "waiting for the eval wave to finish"
while pgrep -f "sched_r3[.]sh .*queue_eval_r3" >/dev/null || pgrep -f "run_da_decode[.]py" >/dev/null; do sleep 20; done
for i in $(seq 1 60); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l); [ "$busy" -eq 0 ] && break; sleep 10; done
log "GPUs free (busy=$busy); plateau probe on 7 checkpoints"
mkdir -p "$P"; g=0
for ck in stock step000200 step000400 step000700 step001000 step001400 step001500; do
  a="$ROOT/logs/draftalign/eval_ckpts/$ck"; [ "$ck" = stock ] && a=stock
  CUDA_VISIBLE_DEVICES=$g timeout 900 /venv/r3/bin/python code/draftalign/probe_plateau_r3.py --adapter "$a" --n 160 --out "$P/$ck.json" > "$P/$ck.log" 2>&1 &
  g=$((g+1))
done
wait
log "probe finished: $(ls "$P"/*.json 2>/dev/null | wc -l)/7 json"
sleep 15
rm -f "$OUT/done"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup /venv/r3/bin/torchrun --standalone --nproc_per_node=8 \
  code/draftalign/train.py --index logs/draftalign/trainset_r3/index.jsonl --out "$OUT" --resume "$OUT/step001500" \
  --steps 2175 --grad-accum 4 --pool-Ms 128 --no-train-gates --draft-flag \
  --p-draftfull 0.3 --p-allmask 0.0 --min-lr-frac 0.5 --warmup 50 --max-seq 1200 \
  --holdout-items 150 --eval-every 200 --save-every 100 --log-every 10 --seed 1 --keep-checkpoints 0 \
  >> "$OUT/train.log" 2>&1 &
log "training resumed from step001500, torchrun pid $!"
nohup bash code/draftalign/ckpt_sync_r3.sh "$OUT" >> logs/draftalign/ckpt_sync_r3_r3_A.log 2>&1 &
log "ckpt_sync (Anupam-Rawat-IITB / Nemotron-split-moe-train) pid $!"
