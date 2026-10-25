#!/usr/bin/env bash
# Phase 5, stage 2: three probes, no training. Each needs the stage-0 setup.
#   bash code/loop_guard/run_probes.sh bins    # A: judge confidence bins on held-out pass-1 states (Gate-2 states), ~5 min
#   bash code/loop_guard/run_probes.sh oracle  # B: rd1 with ORACLE drafts (= vanilla's tokens) on the Gate-3 quarter, ~40 min
#   bash code/loop_guard/run_probes.sh tau     # C: rd1 + ban + guard, agreed drafts commit at 0.90 (overrides at 0.95), quarter, ~40 min
#   bash code/loop_guard/run_probes.sh report
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}
PY=${PY:-/venv/p2/bin/python}
TR=${TR:-/venv/p2/bin/torchrun}
NGPU=${NGPU:-8}
L=$ROOT/logs/phase5; R=$ROOT/results/phase5_loop_guard; mkdir -p $L $R
SETS7="ifeval541 math500 gpqa aime humaneval mbpp gsm8k500"
IDX="--index $ROOT/logs/c1b/trainset/waveA/index.jsonl --index $ROOT/logs/c1b/trainset/waveB/index.jsonl"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_probes.log; }
cd $ROOT
case ${1:-help} in
bins)
  for f in logs/phase4/train/main/holdout_pids.json; do [ -f $f ] || { say "missing $f"; exit 1; }; done
  [ "$(ls logs/phase4/gate1/drafts/drafts.shard*.npz 2>/dev/null | wc -l)" -ge 8 ] || { say "missing logs/phase4/gate1/drafts (HF phase4-drafter-lora)"; exit 1; }
  $TR --standalone --nproc_per_node=$NGPU code/loop_guard/probe_bins.py $IDX --holdout-pids logs/phase4/train/main/holdout_pids.json \
    --drafts logs/phase4/gate1/drafts --adapter logs/phase2/ref/c1b_128M/tok_000128000000 --out $R/probe_bins.json 2>&1 | tee -a $L/probe_bins.log
  exit ${PIPESTATUS[0]} ;;
oracle)   # no ban, no guard: both would mask the oracle's only token
  for set in $SETS7; do
    EXTRA="--oracle-drafter $ROOT/logs/phase4/decode/vanilla_here/$set" SETS=$set NSH=32 OUTDIR=$L/probes/oracle LG_ARMS=rd1 LG=0 \
      bash code/loop_guard/run_lg.sh decode || exit 1
  done ;;
tau)
  EXTRA="--redraft-ban-rejected --loop-guard 12 --tau-commit 0.90 --tau-override 0.95" SETS="$SETS7" NSH=32 OUTDIR=$L/probes/tau90 LG_ARMS=rd1 \
    bash code/loop_guard/run_lg.sh decode ;;
report)
  for arm in oracle tau90; do
    lg=12; [ $arm = oracle ] && lg=0
    $PY code/drafter_lora/report_p4.py --arm probe_$arm --new $L/probes/$arm/rd1_lg$lg --phase3 $ROOT/logs/phase4/decode/rd1_gate3 \
      --vanilla "$ROOT/logs/phase4/decode/vanilla_here/{set}/*_B32_t95_stock.{set}.shard*.jsonl" --out $R/probe_$arm.json || exit 1
  done
  $PY - <<'PYEOF'
import json
R="results/phase5_loop_guard"
for arm in ("oracle","tau90"):
    d=json.load(open(f"{R}/probe_{arm}.json")); p=d["vs_phase3"]["pooled"]; v=d.get("vs_vanilla",{}).get("pooled",{})
    n,r=p["new"],p["phase3"]
    print(f"[{arm}] quarter: judge passes/block {r['judge_passes_per_block']:.2f} (Phase 4 rd1) -> {n['judge_passes_per_block']:.2f} "
          f"({p['judge_passes_reduction_pct']:+.1f}% fewer); vanilla {v.get('vanilla',{}).get('judge_passes_per_block',float('nan')):.2f}; "
          f"accuracy {r['accuracy_pct']:.1f} -> {n['accuracy_pct']:.1f} (d {p['delta_accuracy_pts_ci95'][0]:+.1f} "
          f"[{p['delta_accuracy_pts_ci95'][1]:+.1f},{p['delta_accuracy_pts_ci95'][2]:+.1f}]); loops {r['loops']} -> {n['loops']}")
PYEOF
  ;;
*) echo "usage: $0 bins|oracle|tau|report"; exit 2 ;;
esac
