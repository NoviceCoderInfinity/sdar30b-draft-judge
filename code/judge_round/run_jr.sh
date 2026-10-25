#!/usr/bin/env bash
# Phase 5, stage 3: the judge round (20-hour plan). Continue the 128M judge for 50M corpus tokens on states that carry
# the Phase 4 drafter's FIRST drafts (K1), rd2-shaped masked-hole states (K2) and rd1-shaped RE-DRAFT states (K3),
# then gate on the Gate-3 quarter in rd1 (+ ban + loop guard) and, on a pass, evaluate all 7 sets in rd1 and rd2.
# Rules in docs/PHASE5_JUDGE_ROUND.md.  SGLang is used ONLY to write first drafts; everything else is the HF harness.
#   bash code/judge_round/run_jr.sh subset | drafts | redrafts | swap | train | gate | full | report
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}
PY=${PY:-/venv/p2/bin/python}
PYS=${PYS:-/venv/sgl/bin/python}
TR=${TR:-/venv/p2/bin/torchrun}
NGPU=${NGPU:-8}
L=$ROOT/logs/phase5/judge; R=$ROOT/results/phase5_loop_guard; mkdir -p $L $R
IDX="--index $ROOT/logs/c1b/trainset/waveA/index.jsonl --index $ROOT/logs/c1b/trainset/waveB/index.jsonl"
MERGED=${MERGED:-$ROOT/logs/phase4/train/main/merged}
HOLD=${HOLD:-$ROOT/logs/phase4/train/main/holdout_pids.json}
FROM=${FROM:-$ROOT/logs/c1b/train/main_128m/tok_000128000000}       # adapter + optimizer + token count (HF main_128m)
TOKENS=${TOKENS:-52e6}
KMIX=${KMIX:-0.15,0.25,0.60}                                         # K1 pass-1 / K2 masked holes / K3 re-draft states
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/run_jr.log; }
cd $ROOT
case ${1:-help} in
subset)
  $PY code/judge_round/select_subset.py $IDX --exclude $HOLD --tokens $TOKENS --out $L/subset_pids.json ;;
drafts)   # FIRST drafts of the Phase 4 drafter, in SGLang (data generation only), one engine per GPU; Phase 1 measured 68-94 blocks/s/GPU
  [ -x $PYS ] || { say "no /venv/sgl: run code/judge_round/setup_sgl.sh first"; exit 1; }
  pids=()
  for k in $(seq 0 $((NGPU - 1))); do
    ( [ -f $ROOT/logs/phase5/sgl_setup/sgl_env.sh ] && . $ROOT/logs/phase5/sgl_setup/sgl_env.sh;   # CUDA_HOME/PATH for FlashInfer JIT on sm_120 (setup_sgl.sh)
      CUDA_VISIBLE_DEVICES=$k SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-4294967296} \
        $PYS code/judge_round/draft_sglang_items.py $IDX --pids $L/subset_pids.json --drafter-dir $MERGED \
        --max-running ${SGL_MAX_RUNNING:-256} --mem-fraction ${SGL_MEM_FRACTION:-0.70} \
        --gpu $k --num-gpus $NGPU --out $L/drafts ) > $L/drafts_gpu$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { say "a draft shard failed (see $L/drafts_gpu*.log)"; exit 1; }; done
  n=$(ls $L/drafts/drafts.shard*of$NGPU.npz 2>/dev/null | wc -l); say "drafts: $n/$NGPU shard files"; [ "$n" -ge "$NGPU" ] || exit 1 ;;
redrafts) # RE-drafts on the holes (D0 != vanilla token) from the committed context, ban on; HF harness, 8 shards, batch 1
  pids=()
  for k in $(seq 0 $((NGPU - 1))); do
    ( CUDA_VISIBLE_DEVICES=$k $PY code/judge_round/redraft_items.py $IDX --pids $L/subset_pids.json --drafts $L/drafts \
        --trained $MERGED --shard $k --nshards $NGPU --device cuda:0 --seed 0 --out $L/redrafts ) > $L/redrafts_shard$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { say "a re-draft shard failed (see $L/redrafts_shard*.log)"; exit 1; }; done
  n=$(ls $L/redrafts/redrafts.shard*of$NGPU.npz 2>/dev/null | wc -l); say "redrafts: $n/$NGPU shard files"; [ "$n" -ge "$NGPU" ] || exit 1 ;;
swap)
  $PY code/judge_round/swap_drafts.py $IDX --pids $L/subset_pids.json --drafts $L/drafts --redrafts $L/redrafts --out $ROOT/logs/c1b/trainset/p4drafts ;;
train)    # 50M more corpus tokens from 128M -> 178M on K1/K2/K3 states; Phase 1 measured ~9.5M tok/h
  [ -f $FROM/trainer_state.pt ] || { say "missing $FROM/trainer_state.pt (download main_128m/tok_000128000000 from HF Arushhh/c1b-judge-lora-sdar30b-b32)"; exit 1; }
  if ls -d $L/main/roll_* $L/main/tok_* >/dev/null 2>&1; then CONT="--resume"; else CONT="--continue-from $FROM"; fi
  $TR --standalone --nproc_per_node=$NGPU code/judge_round/train_c1b_k3.py --k-mix $KMIX --index $ROOT/logs/c1b/trainset/p4drafts/index.jsonl \
    --out $L/main $CONT --max-tokens 178e6 --save-at-tokens 153e6,178e6 --eval-every-tokens 25e6 2>&1 | tee -a $L/train.log
  [ ${PIPESTATUS[0]} = 0 ] || { say "training exited with an error"; exit 1; }
  ls -d $L/main/tok_000178000000 >/dev/null || { say "no tok_000178000000 checkpoint"; exit 1; } ;;
gate)     # rd1 + ban + guard with the NEW judge on the Gate-3 quarter (items[k::32], k = 0..7), vs Phase 4 rd1_gate3
  AD=$L/main/tok_000178000000 SETS="ifeval541 math500 gpqa aime humaneval mbpp gsm8k500" NSH=32 OUTDIR=$L/decode/rd1_gate3 \
    LG_ARMS=rd1 bash code/loop_guard/run_lg.sh decode || exit 1
  $PY code/drafter_lora/compare_gates.py --gate 3 --arm rd1 --new $L/decode/rd1_gate3/rd1_lg12 --phase3 $ROOT/logs/phase4/decode/rd1_gate3 \
    --min-reduction-pct 5 --fail-reduction-pct 2 --out $R/judge_gate3.json; rc=$?
  say "judge round gate: rc=$rc ($R/judge_gate3.json); 0 = PASS, 3 = kill rule"; exit $rc ;;
full)     # both modes, all 7 sets, new judge, vs Phase 4 Gate 4 records (same questions)
  AD=$L/main/tok_000178000000 SETS="ifeval541 math500 gpqa aime humaneval mbpp gsm8k500" NSH=8 OUTDIR=$L/decode LG_ARMS="rd1 rd2" \
    bash code/loop_guard/run_lg.sh decode ;;
report)
  for arm in rd1 rd2; do
    $PY code/drafter_lora/report_p4.py --arm judge_$arm --new $L/decode/${arm}_lg12 --phase3 $ROOT/logs/phase4/decode/$arm \
      --vanilla "$ROOT/logs/phase4/decode/vanilla_here/{set}/*_B32_t95_stock.{set}.shard*.jsonl" --out $R/judge_$arm.json || exit 1
  done ;;
*) echo "usage: $0 subset|drafts|redrafts|swap|train|gate|full|report"; exit 2 ;;
esac
