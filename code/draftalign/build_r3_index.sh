#!/usr/bin/env bash
# Round-3 Stage 2 (DATA.md §4 / recipe §3.1): decode records -> 32-token training blocks, one pool at a time.
#   bash code/draftalign/build_r3_index.sh gsm8k      # after the GSM8K datagen finishes
#   bash code/draftalign/build_r3_index.sh opencode N # after opencode_pool:0:N finishes
#   bash code/draftalign/build_r3_index.sh if N       # after if_pool:0:N finishes (verifier-gated)
# Filters are the recipe's, unchanged: --only-correct --max-seq 1200 --exclude-frozen.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/root/.cache/huggingface
PY=${PY:-/venv/r3/bin/python}
ROOT=logs/draftalign/trainset_r3
case "$1" in
  gsm8k)    SPLIT="gsm8k_pool:0:2400"; REC=logs/draftalign/datagen/gsm8k_pool_0_2400_dv-s8_blind_q50_r3; SRC=gsm8k ;;
  opencode) SPLIT="opencode_pool:0:$2"; REC=logs/draftalign/datagen/opencode_pool_0_$2_dv-s8_blind_q50_r3; SRC=opencode; SCORED=${REC}_scored ;;
  if)       SPLIT="if_pool:0:$2";       REC=logs/draftalign/datagen/if_pool_0_$2_dv-s8_blind_q50_r3; SRC=if; SCORED=${REC}_scored ;;
  *) echo "usage: $0 gsm8k | opencode N | if N"; exit 2 ;;
esac
OUT=$ROOT/$SRC
RECORDS=${SCORED:-$REC}   # code pools are decoded without --score and scored by run_score_r3.sh into <cell>_scored/
# IF: re-score every shard with the current if_scorer (the 2026-09-14 EOS-guard fix post-dates the running decodes);
# no generated code is executed for IF, so this runs as root from a throwaway cwd.
if [ "$SRC" = if ]; then
  [ "$(ls "$REC"/*.jsonl | wc -l)" -eq "$(ls "$REC"/*.jsonl.done 2>/dev/null | wc -l)" ] || { echo "IF decode not finished"; exit 1; }
  mkdir -p /dev/shm/dlm/sandbox/rescore_if && (cd /dev/shm/dlm/sandbox/rescore_if && "$PY" "$OLDPWD/code/draftalign/score_r3_records.py" --records "$OLDPWD/$REC" --out "$OLDPWD/$SCORED" --code "$OLDPWD/code" --workers 32)
fi
# every shard must have finished (run_da_decode writes <record>.done on completion)
nrec=$(ls "$REC"/*.jsonl | wc -l); ndone=$(ls "$RECORDS"/*.jsonl.done 2>/dev/null | wc -l)
[ "$nrec" -gt 0 ] && [ "$nrec" -eq "$ndone" ] || { echo "not finished: $ndone/$nrec shards done in $REC"; exit 1; }
[ -e "$OUT/index.jsonl" ] && { echo "refusing to append to existing $OUT/index.jsonl"; exit 1; }
"$PY" code/draftalign/build_train_set.py --records "$RECORDS" --traces "$REC/traces" --layout da \
  --split-name "$SPLIT" --source "$SRC" --out "$OUT" --only-correct --max-seq 1200 --exclude-frozen
