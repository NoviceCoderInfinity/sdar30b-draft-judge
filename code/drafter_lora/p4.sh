#!/usr/bin/env bash
# Phase 4, the drafter LoRA: one command per runbook stage. Every stage stops (exit 3) at its gate's failure; nothing
# goes on to the next stage by itself -- the operator reads the verdict and runs the next one.
#
#   bash code/drafter_lora/p4.sh setup    # setup_p3_8gpu.sh (env, models, judge, 16/16 checks) + fetch_p4.py + tests + selfcheck.py
#   bash code/drafter_lora/p4.sh smoke    # 200 steps on 2,000 items, replay parity on 200 held-out items, speed probe
#   bash code/drafter_lora/p4.sh train    # the one pass (resumes from logs/phase4/train/main/ckpt/latest)
#   bash code/drafter_lora/p4.sh gate1    # offline: first-draft agreement, retention, re-draft probe (+ drafts dump)
#   bash code/drafter_lora/p4.sh gate2    # judge probe: commit precision on trained vs stored drafts
#   bash code/drafter_lora/p4.sh gate3    # rd1 + ban on items[k::32], k = 0..7, of all 7 benchmarks vs Phase 3 rd1
#   bash code/drafter_lora/p4.sh gate4    # rd1 and rd2 + ban, all items, all 7 benchmarks (GATE4_ARMS="rd1 rd2")
#   bash code/drafter_lora/p4.sh backup   # secret scan, then upload to the HF dataset $P4_HF_REPO
#
# Stop between runs: touch logs/phase4/STOP.  Env overrides: PY, TORCHRUN, NGPU, SETS, TRAIN, VAN_TPL.
# Decodes run from per-shard sandbox working directories under /dev/shm (AGENTS.md footgun p); every output path is
# resolved against the repository root by research_protocol.safe_out.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
PY=${PY:-/venv/p2/bin/python}; TORCHRUN=${TORCHRUN:-/venv/p2/bin/torchrun}; NGPU=${NGPU:-8}
L=logs/phase4; R=results/phase4_drafter; mkdir -p $L $R
IDX="--index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl"
TRAIN=${TRAIN:-$L/train/main}
AD=logs/phase2/ref/c1b_128M/tok_000128000000
P3=$L/ref/phase3/logs/phase3_8gpu/decode
VAN_DEFAULT="$L/ref/phase2/assets/ref_stock_30B/*_B32_t95_stock.{set}.shard*.jsonl"   # flat bundle; no pooled cells
VAN_TPL=${VAN_TPL:-$VAN_DEFAULT}       # (in its own variable: the '}' of {set} inside ${..:-..} would end the expansion)
SETS=${SETS:-"ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"}
say(){ echo "[p4 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $L/p4.log; }
ev(){ printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" >> $L/events.tsv; }
stopped(){ [ -f $L/STOP ] && { say "STOP file present ($L/STOP): not starting $1"; exit 0; }; }
need(){ [ -e "$1" ] || { say "missing $1 -- run the stage that makes it first"; exit 1; }; }

# one shard per GPU, all in the background; returns the number of failed shards
shards(){   # $1 log dir, then the command template with {k}
  local jl=$1; shift; local k pids=() fail=0
  mkdir -p $jl
  for k in $(seq 0 $((NGPU - 1))); do
    ( export CUDA_VISIBLE_DEVICES=$k; eval "${*//\{k\}/$k}" ) > $jl/shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  return $fail
}

decode_cell(){   # $1 arm (rd1|rd2), $2 set, $3 out dir, $4 nshards; shards 0..NGPU-1, one retry (records resume)
  local arm=$1 set=$2 out=$3 nsh=$4 every attempt k n
  case $arm in rd1) every=1 ;; rd2) every=2 ;; *) say "bad arm $arm"; return 1 ;; esac
  [ -f $out/.complete ] && { say "skip $arm $set (complete)"; return 0; }
  mkdir -p $out
  for attempt in 1 2; do
    ev start "$arm $set $out attempt$attempt"
    local pids=()
    for k in $(seq 0 $((NGPU - 1))); do
      local sb=/dev/shm/p4_sandbox/$(basename $out)/$set/shard$k; mkdir -p $sb
      ( cd $sb && CUDA_VISIBLE_DEVICES=$k exec $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --items $set \
          --drafter $TRAIN/merged --draft s8 --judge full --adapter $AD --judge-diag --full-redraft-every $every \
          --redraft-ban-rejected --shard $k --nshards $nsh --gpu $k --score --no-trace --out $out \
      ) > $out/shard$k.attempt$attempt.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    n=$(ls $out/*.jsonl.done 2>/dev/null | wc -l)
    ev end "$arm $set $out attempt$attempt shards_done=$n/$NGPU"
    [ "$n" -ge "$NGPU" ] && { touch $out/.complete; say "done $arm $set ($out)"; return 0; }
    say "$arm $set: $n/$NGPU shards finished (attempt $attempt)"
  done
  return 1
}

stage=${1:-help}
case $stage in
setup)
  stopped setup
  say "setup: env + pinned models (30B ~61 GB, 1.7B ~3.4 GB) + 128M judge + 16/16 checks; progress: tail -f $L/setup.log"
  bash code/phase3_8gpu/setup_p3_8gpu.sh 2>&1 | tee -a $L/setup.log; [ ${PIPESTATUS[0]} = 0 ] || { say "setup_p3_8gpu.sh FAILED ($L/setup.log)"; exit 1; }
  $PY - <<'EOF' || { say "setup checks below 16/16 -- stop"; exit 3; }
import json
c = json.load(open("logs/phase3_8gpu/setup/checks.json"))
assert c["stock_items"] == 16 and c["stock_identical"] == 16, c
assert c["judge_with_reference"] == 16 and c["judge_identical"] == 16, c
print("[p4] vanilla and base decoding 16/16 token-identical to Phase 1:", c)
EOF
  say "fetch: corpus waves A+B (2.3 GB, 512 tars -> 331,818 items), Phase 3 + vanilla records (0.2 GB)"
  HF_HUB_OFFLINE=0 $PY code/drafter_lora/fetch_p4.py --what all 2>&1 | tee -a $L/setup.log; [ ${PIPESTATUS[0]} = 0 ] || { say "fetch FAILED ($L/setup.log)"; exit 1; }
  $PY -c "import peft" 2>/dev/null && say "note: peft is installed but not used (drafter_lora/lora.py is self-contained)"
  $PY -m pytest -q code/tests/test_drafter_lora.py code/tests/test_da_decode_redraft.py || { say "tests FAILED"; exit 1; }
  say "self-check on the real SDAR-1.7B + real corpus items (GPU 0): count, forward == remote code, KL 0, step, parity, merge"
  CUDA_VISIBLE_DEVICES=0 $PY code/drafter_lora/selfcheck.py --index logs/c1b/trainset/waveA/index.jsonl --device cuda:0 \
    --n-items 8 --parity-items 8 2>&1 | tee -a $L/setup.log; [ ${PIPESTATUS[0]} = 0 ] || { say "self-check FAILED ($L/selfcheck.json)"; exit 3; }
  $PY -c "import torch; a = torch.cuda.get_arch_list(); print('arch', a); assert 'sm_120' in a, a" \
    || { say "torch has no sm_120 (Blackwell) kernels"; exit 1; }
  say "setup done"; ev setup_done ;;

smoke)
  stopped smoke
  S=$L/smoke; mkdir -p $S
  say "smoke 1/3: 200 steps on a 2,000-item slice"
  $TORCHRUN --standalone --nproc_per_node=$NGPU code/drafter_lora/train_drafter_lora.py $IDX --out $S/train \
    --limit-items 2000 --items-per-step 10 --check-at 0 --no-merge --max-hours 0 --run-name phase4_smoke \
    > $S/train.log 2>&1 || { say "smoke training FAILED ($S/train.log)"; exit 1; }
  say "smoke 2/3: untrained replay vs stored drafts on 200 held-out items"
  rm -rf $S/parity; shards $S/parity_logs "$PY code/drafter_lora/eval_agree.py $IDX --holdout-pids $S/train/holdout_pids.json \
    --parity-only --n-items 200 --shard {k} --nshards $NGPU --device cuda:0 --out $S/parity/shard_{k}.json" \
    || { say "parity shards FAILED ($S/parity_logs)"; exit 1; }
  $PY code/drafter_lora/eval_agree.py --aggregate $S/parity --out $S/parity.json
  say "smoke 3/3: speed probe (full configuration, 30 steps)"
  $TORCHRUN --standalone --nproc_per_node=$NGPU code/drafter_lora/train_drafter_lora.py $IDX --out $S/speedprobe \
    --max-steps 30 --project-at 20 --check-at 0 --no-merge --run-name phase4_speedprobe \
    > $S/speedprobe.log 2>&1 || { say "speed probe FAILED ($S/speedprobe.log)"; exit 1; }
  $PY code/drafter_lora/smoke_check.py --train $S/train --parity $S/parity.json --speed $S/speedprobe/speed.json \
    --out $S/smoke.json; rc=$?
  ev smoke "rc=$rc"; [ $rc = 0 ] && say "smoke PASS" || { say "smoke: STOP AND ASK ($S/smoke.json)"; exit 3; }
  rm -rf $S/speedprobe/ckpt ;;

train)
  stopped train
  say "training: one pass -> $TRAIN"
  $TORCHRUN --standalone --nproc_per_node=$NGPU code/drafter_lora/train_drafter_lora.py $IDX --out $TRAIN --resume \
    --run-name phase4_main 2>&1 | tee -a $TRAIN.log; rc=${PIPESTATUS[0]}
  ev train "rc=$rc"
  [ $rc = 3 ] && { say "training STOPPED at a check: $TRAIN/STOP.json -- ask"; exit 3; }
  [ $rc = 0 ] || { say "training FAILED (rc $rc)"; exit 1; }
  cp $TRAIN/check_25pct.json $R/ 2>/dev/null; cp $TRAIN/speed.json $R/train_speed.json 2>/dev/null
  # runbook 3a: the merged dir loads with AutoModelForCausalLM(trust_remote_code, local_files_only) and --drafter takes it
  CUDA_VISIBLE_DEVICES=0 $PY code/drafter_lora/check_merged.py $TRAIN/merged --device cuda:0 2>&1 | tee $R/check_merged.log
  [ ${PIPESTATUS[0]} = 0 ] || { say "merged drafter check FAILED ($R/check_merged.log)"; exit 1; }
  say "training done: $TRAIN/final_adapter, $TRAIN/merged (merged check PASS)" ;;

gate1)
  stopped gate1; need $TRAIN/final_adapter
  rm -rf $L/gate1/shards
  shards $L/gate1/logs "$PY code/drafter_lora/eval_agree.py $IDX --holdout-pids $TRAIN/holdout_pids.json \
    --trained $TRAIN/final_adapter --redraft-probe --dump-drafts $L/gate1/drafts --shard {k} --nshards $NGPU \
    --device cuda:0 --out $L/gate1/shards/shard_{k}.json" || { say "gate1 shards FAILED ($L/gate1/logs)"; exit 1; }
  $PY code/drafter_lora/eval_agree.py --aggregate $L/gate1/shards --out $R/gate1.json; rc=$?
  # merge fidelity: Gates 3-4 decode with the bf16-MERGED drafter; it must agree as the un-merged adapter does
  for kind in final_adapter merged; do
    rm -rf $L/gate1/merge_check/$kind
    shards $L/gate1/merge_check/logs_$kind "$PY code/drafter_lora/eval_agree.py $IDX --holdout-pids $TRAIN/holdout_pids.json \
      --trained $TRAIN/$kind --n-items 300 --shard {k} --nshards $NGPU --device cuda:0 \
      --out $L/gate1/merge_check/$kind/shard_{k}.json" || { say "merge check shards FAILED"; exit 1; }
    $PY code/drafter_lora/eval_agree.py --aggregate $L/gate1/merge_check/$kind --out $R/gate1_merge_check_$kind.json >/dev/null
  done
  $PY - <<EOF
import json
a = json.load(open("$R/gate1_merge_check_final_adapter.json"))["overall"]["agreement_trained"]
m = json.load(open("$R/gate1_merge_check_merged.json"))["overall"]["agreement_trained"]
print(f"[p4] merge fidelity on 300 held-out items: adapter {a:.4f}, bf16-merged {m:.4f}, diff {100 * (m - a):+.2f} points")
EOF
  ev gate1 "rc=$rc"; [ $rc = 0 ] && say "Gate 1 PASS ($R/gate1.json)" || { say "Gate 1: STOP AND ASK ($R/gate1.json)"; exit 3; } ;;

gate2)
  stopped gate2; need $L/gate1/drafts
  $TORCHRUN --standalone --nproc_per_node=$NGPU code/drafter_lora/judge_probe.py $IDX \
    --holdout-pids $TRAIN/holdout_pids.json --drafts $L/gate1/drafts --adapter $AD --out $R/gate2.json \
    > $L/gate2.log 2>&1; rc=$?
  ev gate2 "rc=$rc"; [ $rc = 0 ] && say "Gate 2 PASS ($R/gate2.json)" || { say "Gate 2: STOP AND ASK ($R/gate2.json, $L/gate2.log)"; exit 3; } ;;

gate3)
  need $TRAIN/merged
  for set in $SETS; do
    stopped "gate3 $set"
    decode_cell rd1 $set $L/decode/rd1_gate3/$set 32 || { say "FAILED: gate3 rd1 $set"; exit 1; }
  done
  $PY code/drafter_lora/compare_gates.py --gate 3 --arm rd1 --new $L/decode/rd1_gate3 --phase3 $P3/rd1 \
    --out $R/gate3.json; rc=$?
  ev gate3 "rc=$rc"; [ $rc = 0 ] && say "Gate 3 PASS ($R/gate3.json)" || { say "Gate 3: STOP AND ASK ($R/gate3.json)"; exit 3; } ;;

gate4)
  need $TRAIN/merged
  for set in $SETS; do
    for arm in ${GATE4_ARMS:-rd1 rd2}; do
      stopped "gate4 $arm $set"
      decode_cell $arm $set $L/decode/$arm/$set 8 || { say "FAILED: gate4 $arm $set"; exit 1; }
    done
  done
  for arm in ${GATE4_ARMS:-rd1 rd2}; do
    $PY code/drafter_lora/compare_gates.py --gate 4 --arm $arm --new $L/decode/$arm --phase3 $P3/$arm \
      --vanilla "$VAN_TPL" --out $R/gate4_$arm.json
  done
  ev gate4 done; say "Gate 4 reports: $R/gate4_*.json" ;;

backup)
  $PY code/drafter_lora/backup_p4.py ;;

*)
  sed -n 2,16p "$0"; exit 1 ;;
esac
