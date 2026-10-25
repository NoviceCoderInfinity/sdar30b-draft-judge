#!/usr/bin/env bash
# Vanilla 30B (stock decode: no drafter, no judge adapter) on THIS box, all 7 benchmarks, every item: the same-machine
# baseline for Phase 4's speed against vanilla (operator, 13 Oct 2026).  Same decode as the vanilla records
# (run_da_decode.py --stock: block 32, threshold 0.95), 8 shards items[k::8] (one process per GPU), each shard in its own
# sandbox cwd (AGENTS.md footgun p).  Every item is compared token by token with the vanilla records (--ref-dir ->
# ref_match in the record).  Records: logs/phase4/decode/vanilla_here/<set>/ ; one retry per set (records resume).
#   bash code/drafter_lora/run_vanilla_here.sh        (stop between sets: touch logs/phase4/STOP)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
PY=${PY:-/venv/p2/bin/python}; NGPU=${NGPU:-8}; L=logs/phase4
SETS=${SETS:-"ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"}
REF=$L/ref/phase2/assets/ref_stock_30B
say(){ echo "[p4 $(date -u +%Y-%m-%dT%H:%M:%SZ)] vanilla-here: $*" | tee -a $L/p4.log; }
ev(){ printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" >> $L/events.tsv; }
for set in $SETS; do
  [ -f $L/STOP ] && { say "STOP file present: not starting $set"; exit 0; }
  out=$L/decode/vanilla_here/$set
  [ -f $out/.complete ] && { say "skip $set (complete)"; continue; }
  mkdir -p $out
  for attempt in 1 2; do
    ev start "vanilla_here $set attempt$attempt"
    pids=()
    for k in $(seq 0 $((NGPU - 1))); do
      sb=/dev/shm/p4_sandbox/vanilla_here/$set/shard$k; mkdir -p $sb
      ( cd $sb && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --items $set --stock \
          --ref-dir $REF --shard $k --nshards $NGPU --gpu $k --score --no-trace --out $out \
      ) > $out/shard$k.attempt$attempt.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    n=$(ls $out/*.jsonl.done 2>/dev/null | wc -l)
    ev end "vanilla_here $set attempt$attempt shards_done=$n/$NGPU"
    if [ "$n" -ge "$NGPU" ]; then touch $out/.complete; say "done $set ($out)"; break; fi
    say "$set: $n/$NGPU shards finished (attempt $attempt)"
    [ $attempt = 2 ] && say "FAILED: $set"
  done
done
say "all sets processed"
