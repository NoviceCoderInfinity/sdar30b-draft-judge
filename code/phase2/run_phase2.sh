#!/usr/bin/env bash
# Phase 2 pipeline for ONE drafter arm (Notion "Phase 2: Drafter swap ablations", steps 3-6). Resumable: every
# finished stage leaves logs/phase2/<ARM>/.done_<stage>; a stage stopped by a rule leaves .stopped_<stage>.
#
#   P2_SETS=ifeval541 bash code/phase2/run_phase2.sh S0 decode report     # S0 judge+DES48 on sets round 3 lacks
#   P2_SETS=<set> bash code/phase2/run_phase2.sh stock decode report  # stock 30B on a set with no stock reference
#                                    (not needed for IFEval-541: setup_p2.sh 3c brings Phase 1's stock run)
#   bash code/phase2/run_phase2.sh F1          # an arm: redraft -> probe -> decode -> report -> gate [-> retrain ...]
#   bash code/phase2/run_phase2.sh F1 decode   # one stage
#   bash code/phase2/run_phase2.sh S0ctl       # the retrain CONTROL: same trainer + recipe on S0's own drafts
#
# Stages for an arm A (run in order; a later stage needs the earlier ones):
#   redraft    re-draft the round-3 training index with A (8 shards) -> logs/phase2/redraft/A   [step 3]
#              stop rule drafter_misfit / tokenizer_or_remap_bug -> the arm ends here ("does not fit")
#   probe      the S0 judge (r3_B step 710) on A's drafts, round 3's held-out items          [step 4]
#   decode     zero-shot: A + S0 judge + DES48 on the 7 sets x 8 shards (in-domain first), expert + judge-probability
#              logs, stop rules                                                              [step 5]
#   report     experts / router / judge-probability report of every decoded set -> logs/phase2/report/A
#   gate       verdict vs S0 (rules.json pass_fail) -> logs/phase2/gate/A_zeroshot (P2_REGATE=1 recomputes it)
#   retrain    ONLY if the gate says retrain_needed (or P2_FORCE_RETRAIN=1): LoRA judge adapter from stock
#              on A's re-drafted index, round-3 calibrated recipe, 2 epochs                  [step 6]
#   probe_rt, decode_rt, gate_rt   the same with the retrained adapter; gate_rt pairs against S0ctl
# Env: P2_SETS (default all 7: humaneval mbpp gsm8k500 math500 gpqa ifeval541 aime), NSH (shards per set, 8), GPUS,
#      PY, TORCHRUN, P2_EXPERT_LOG (full, default: every row's top-8 expert ids + router probabilities | bits),
#      P2_S0_REF  hf (default): per set, a local S0 run (logs/phase2/decode/S0/<set>) if one exists, else round 3's
#                 S0 records (r3_B step 710 + DES48, downloaded by setup_p2.sh) -- valid because setup's stock check
#                 proves this stack reproduces round 3 token for token; S0: always the local S0 run
#      P2_NO_RETRAIN  1 (default): stop after the gate; step 6 runs only with P2_NO_RETRAIN=0 (or P2_FORCE_RETRAIN=1)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
ARM=${1:?arm: S0 | S0ctl | F1 | F2 | S1 | S2 | S6}; shift
STAGES=${*:-all}
PY=${PY:-/venv/p2/bin/python}; TORCHRUN=${TORCHRUN:-/venv/p2/bin/torchrun}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NSH=${NSH:-8}
INDEX=logs/phase2/data/trainset_r3/index.jsonl
S0_ADAPTER=${P2_ADAPTER:-logs/phase2/ref/r3_B/step000710}   # the judge every arm decodes with (P2_ADAPTER overrides)
S0_REF_ROOT=${P2_S0_REF_ROOT:-logs/phase2/ref/S0_by_set}      # S0 records with that judge, one dir per set
# P2_POOL=none decodes with no expert pool (default des48: the plan's DES48); P2_TAG names that variant's output dirs
OUT_ARM=$ARM${P2_TAG:-}
case ${P2_POOL:-des48} in des48) POOLFLAGS="--pool 48 --pool-rule des --pool-flush" ;; none) POOLFLAGS="" ;; *) echo "bad P2_POOL"; exit 1 ;; esac
D=logs/phase2/$OUT_ARM; mkdir -p $D logs/phase2/jobs
say(){ echo "[phase2 $ARM $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*" | tee -a $D/pipeline.log; }
done_(){ [ -f "$D/.done_$1" ]; }; mark(){ touch "$D/.done_$1"; }
want(){ [ "$STAGES" = all ] || [[ " $STAGES " == *" $1 "* ]]; }
[ -e logs/phase2/STOP ] && { say "logs/phase2/STOP present: remove it to run"; exit 3; }

# the drafter key the decode runner gets (S0 = the round-3 d32 cells, so its records pair with round 3's), and sets
case $ARM in
  S0|S0ctl) DKEY=d32; RKEY=S0 ;;
  stock) DKEY=""; RKEY=S0 ;;                      # the stock 30B (no drafter, no judge, no pool): speedup reference
  *) DKEY=$ARM; RKEY=$ARM ;;
esac
ALL_SETS="humaneval mbpp gsm8k500 math500 gpqa ifeval541 aime"
SETS=${P2_SETS:-$($PY -c "import sys; sys.path[:0]=['code/compat','code']; from phase2 import drafters as D; \
print(' '.join(D.spec('$RKEY').get('sets', '$ALL_SETS'.split())))")}
EXPLOG=${P2_EXPERT_LOG:-full}                     # full: every row's top-8 expert ids + router probabilities
ADAPTER=$S0_ADAPTER; RD=logs/phase2/redraft/$RKEY
[ "$ARM" = S0ctl ] && ADAPTER=logs/phase2/train/S0ctl/best

run_jobs(){   # $1 = job-file stem; stdin = job lines (json); returns the runner's exit code
  local J=logs/phase2/jobs/$1.jsonl; cat > $J
  $PY code/phase2/p2_jobs.py --jobs $J --log logs/phase2/jobs/$1/run.log --gpus $GPUS
}

# the S0 reference of a set: a local S0 run (logs/phase2/decode/S0/<set>) when one exists -- IFEval has no round-3
# S0 records, so the launcher runs S0 on it first -- else round 3's records (downloaded; hf). P2_S0_REF=S0 forces local.
s0_dir_for(){ local set=$1
  if [ "${P2_S0_REF:-hf}" = S0 ] || ls logs/phase2/decode/S0/$set/*.$set.shard*of*.jsonl > /dev/null 2>&1; then
    echo logs/phase2/decode/S0/$set; else echo $S0_REF_ROOT/$set; fi; }
S0_ROOTS=logs/phase2/decode/S0,$S0_REF_ROOT                 # for the gate: local first
STOCK_ROOTS=logs/phase2/decode/stock,logs/phase2/ref/stock_flat_by_set
# priority order: in-domain first (code, maths, instruction following), then near / out of domain, AIME last -- so
# a deadline cut lands on the least informative sets; the job runner still keeps all 8 GPUs busy
ORDERED=$(for s in humaneval mbpp gsm8k500 ifeval541 math500 gpqa aime; do [[ " $SETS " == *" $s "* ]] && echo -n "$s "; done)
for s in $SETS; do [[ " $ORDERED " == *" $s "* ]] || ORDERED="$ORDERED$s "; done   # any other set, last

decode_jobs(){   # $1 arm-dir name, $2 adapter, $3 S0 reference ("" none | auto), $4 extra flags
  for set in $ORDERED; do for k in $(seq 0 $((NSH - 1))); do     # sets in priority order (see ORDERED)
    ref=""
    [ "$3" = auto ] && ref="--p2-s0-dir $(s0_dir_for $set)"
    [ -n "$3" ] && [ "$3" != auto ] && ref="--p2-s0-dir logs/phase2/decode/$3/$set"
    if [ "$1" = stock ]; then
      body="--model b32 --stock --items $set"
    else
      body="--model b32 --drafter $DKEY --items $set --adapter $2 --judge full --draft s8 $POOLFLAGS"
    fi
    $PY -c "import json,sys; print(json.dumps({'name': '$1.$set.$k', 'group': '$1.$set', 'cmd': sys.argv[1]}))" \
"$PY $ROOT/code/draftalign/run_da_decode.py $body --shard $k --nshards $NSH --gpu {GPU} --score \
--out logs/phase2/decode/$1/$set --expert-log $EXPLOG --stop-rules default --p2-stock-dir logs/phase2/ref/stock_flat $ref $4"
  done; done
}

# the per-set reference dirs the decode hooks and the gate read (idempotent)
for set in humaneval mbpp gsm8k500 math500 gpqa aime ifeval541; do   # gate_p2 reads <root>/<cell dir>/<records>: one dir per set
  mkdir -p logs/phase2/ref/stock_flat_by_set/$set logs/phase2/ref/S0_by_set/$set
  ln -sf $ROOT/logs/phase2/ref/stock_flat/*_B32_t95_stock.$set.shard*.jsonl logs/phase2/ref/stock_flat_by_set/$set/ 2>/dev/null || true
  ln -sf $ROOT/logs/phase2/ref/S0_flat/*.$set.shard*.jsonl logs/phase2/ref/S0_by_set/$set/ 2>/dev/null || true
done

# an arm whose re-draft was stopped by a rule ("does not fit") stays stopped: that is its result
if [ -f $D/.stopped_redraft ] && [ "${P2_RETRY_STOPPED:-0}" != 1 ]; then
  say "$ARM was stopped at its re-draft ($RD/STOPPED.json): the drafter does not fit; nothing more runs for it"
  exit 3
fi

# ---- step 3: re-draft ----------------------------------------------------------------------------------------------
if want redraft && ! done_ redraft && [ "$ARM" != S0ctl ] && [ "$ARM" != stock ]; then
  say "redraft $INDEX with $RKEY -> $RD"
  for k in $(seq 0 7); do
    $PY -c "import json,sys; print(json.dumps({'name': 'redraft.$RKEY.$k', 'group': 'redraft.$RKEY', 'cmd': sys.argv[1]}))" \
"$PY $ROOT/code/phase2/redraft.py --index $INDEX --drafter $RKEY --out $RD --shard $k --nshards 8 --device cuda:0"
  done | run_jobs redraft_$RKEY; rc=$?
  if [ $rc = 3 ]; then
    touch $D/.stopped_redraft
    say "STOPPED at redraft: $(cat $RD/STOPPED.json 2>/dev/null | head -c 400). Flowchart: the drafter does not fit this MoE."
    exit 3
  fi
  [ $rc = 0 ] || { say "redraft FAILED (logs/phase2/jobs/redraft_$RKEY/)"; exit 1; }
  $PY code/phase2/redraft.py --index $INDEX --drafter $RKEY --out $RD --merge --nshards 8 | tee -a $D/pipeline.log \
    && mark redraft || { say "redraft merge failed"; exit 1; }
fi

probe_done(){ [ -z "$probe_pid" ] && return 0
  wait $probe_pid; local rc=$?; probe_pid=""
  [ $rc = 0 ] || [ $rc = 3 ] || { say "probe FAILED ($D/probe.log)"; exit 1; }
  [ $rc = 3 ] && touch $D/.stopped_probe
  mark probe; }
# ---- step 4: probe -------------------------------------------------------------------------------------------------
# The probe needs nothing from the decode and the decode nothing from the probe, so when both are due the probe runs
# in the background on the first GPU and the decode starts as soon as the probe holds that GPU's memory: the job
# runner treats a GPU holding >= 1000 MiB as busy, so it decodes on the other GPUs and takes the probe's GPU when the
# probe exits. P2_PROBE_OVERLAP=0 restores the sequential order.
probe_pid=""
if want probe && ! done_ probe && [ "$ARM" != S0ctl ] && [ "$ARM" != stock ]; then
  say "probe: S0 judge on $RKEY drafts"
  box=/dev/shm/p2_sandbox/probe_$RKEY; mkdir -p $box; PG=${GPUS%%,*}
  ( cd $box && CUDA_VISIBLE_DEVICES=$PG $PY $ROOT/code/phase2/probe_p2.py --index $RD/index.jsonl \
      --adapter $S0_ADAPTER --out logs/phase2/probe/${RKEY}__r3_B ) >> $D/probe.log 2>&1 & probe_pid=$!
  if [ "${P2_PROBE_OVERLAP:-1}" = 1 ] && want decode && ! done_ decode; then
    while kill -0 $probe_pid 2> /dev/null && [ "$(nvidia-smi -i $PG --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do
      sleep 5; done
    say "probe running on GPU $PG; the decode starts beside it"
  else
    probe_done
  fi
fi
# ---- step 5: zero-shot decode ----------------------------------------------------------------------------------------
# the S0 and stock reference arms run chosen sets only: their done-marker names the sets
DEC=decode; { [ "$ARM" = S0 ] || [ "$ARM" = stock ]; } && DEC="decode_${SETS// /_}"
if want decode && ! done_ $DEC; then
  case $ARM in
    S0)    ref=""; extra="--p2-integrity --ref-dir logs/phase2/ref/S0_flat" ;;   # the rerun vs round 3's records
    S0ctl|stock) ref=""; extra="" ;;
    *)     ref=auto; extra="" ;;
  esac
  say "decode: $ORDERED x $NSH shards (adapter $ADAPTER, expert log $EXPLOG)"
  decode_jobs $OUT_ARM $ADAPTER "$ref" "$extra" | run_jobs decode_$OUT_ARM; rc=$?
  [ $rc = 3 ] && { touch $D/.stopped_decode; say "decode: some sets STOPPED BY RULE (logs/phase2/jobs/decode_$OUT_ARM/run.log.summary.json)"; }
  [ $rc = 0 ] || [ $rc = 3 ] || { say "decode had crashed jobs (logs/phase2/jobs/decode_$OUT_ARM/)"; exit 1; }
  mark $DEC
fi
probe_done
if want report; then                                   # expert / judge / probability report of every decoded set
  $PY code/phase2/expert_report.py --arm $OUT_ARM --out logs/phase2/report/$OUT_ARM | tail -3 | tee -a $D/pipeline.log
fi

{ [ "$ARM" = S0 ] || [ "$ARM" = stock ]; } && { say "$ARM reference decode done ($SETS)"; exit 0; }

# ---- the gate ----------------------------------------------------------------------------------------------------
gate(){   # $1 arm dir, $2 S0 reference roots (comma list), $3 probe json, $4 out
  $PY code/phase2/gate_p2.py --arm $RKEY --arm-dir logs/phase2/decode/$1 --s0-dir $2 \
    --stock-dir $STOCK_ROOTS --redraft $RD/summary.json --probe $3 --out $4 --sets "${SETS// /,}" \
    | tee -a $D/pipeline.log
}
[ "$ARM" = S0ctl ] && { say "S0ctl decode done (the retrain control)"; exit 0; }
if want gate && { ! done_ gate || [ "${P2_REGATE:-0}" = 1 ]; }; then
  gate $ARM $S0_ROOTS logs/phase2/probe/${RKEY}__r3_B/probe.json logs/phase2/gate/${ARM}_zeroshot && mark gate
fi

# ---- step 6: retrain the LoRA judge adapter only if the zero-shot decode lost in-domain accuracy -----------------
need=$($PY -c "import json; print(int(json.load(open('logs/phase2/gate/${ARM}_zeroshot/gate.json'))['verdict']['retrain_needed']))" 2>/dev/null || echo 0)
[ "${P2_FORCE_RETRAIN:-0}" = 1 ] && need=1
if [ "$need" = 1 ] && [ "${P2_NO_RETRAIN:-1}" = 1 ] && [ "${P2_FORCE_RETRAIN:-0}" != 1 ]; then
  say "gate says RETRAIN NEEDED, but P2_NO_RETRAIN=1: stopping after the zero-shot gate (step 6 on a later run)"
  exit 0
fi
if [ "$need" != 1 ]; then say "no retrain: the zero-shot decode holds in-domain accuracy (gate: logs/phase2/gate/${ARM}_zeroshot/gate.md)"; exit 0; fi
TR=logs/phase2/train/$ARM
train(){   # $1 index, $2 out, $3 drafter key
  $TORCHRUN --standalone --nproc_per_node=8 code/phase2/train_p2.py --index $1 --out $2 --drafter-key $3 \
    --schedule cosine --steps 1400 --warmup-steps 50 --states-per-step 0:32 --lr 2e-5 --min-lr-frac 0.5 \
    --beta 0.5 --calib 1.0 --override-weight 2.0 --code-override-weight 3.0 --p-pass1 0.4 --p-draftfull 0.3 \
    --p-allmask 0.0 --max-seq 1200 --seed 1 --holdout-mode r3 --holdout-items 150 --probe-every-steps 100 \
    --save-every-steps 100 --log-every 10 $( [ -d $2 ] && echo --resume ) >> $2.log 2>&1
}
if want retrain && ! done_ retrain; then
  mkdir -p logs/phase2/train
  if [ ! -f logs/phase2/S0ctl/.done_train ]; then
    say "retrain control S0ctl: the same trainer and recipe on S0's own drafts ($INDEX re-drafted by S0)"
    train logs/phase2/redraft/S0/index.jsonl logs/phase2/train/S0ctl S0; rc=$?
    [ $rc = 0 ] || [ $rc = 3 ] || { say "S0ctl training FAILED (logs/phase2/train/S0ctl.log)"; exit 1; }
    mkdir -p logs/phase2/S0ctl; touch logs/phase2/S0ctl/.done_train
    bash code/phase2/run_phase2.sh S0ctl decode || true
  fi
  say "retrain the LoRA judge adapter on $RKEY drafts -> $TR"
  train $RD/index.jsonl $TR $RKEY; rc=$?
  [ $rc = 3 ] && { touch $D/.stopped_retrain; say "retrain STOPPED BY RULE ($TR/STOPPED.json): continuing with $TR/best"; }
  [ $rc = 0 ] || [ $rc = 3 ] || { say "retrain FAILED ($TR.log)"; exit 1; }
  mark retrain
fi
RT_ADAPTER=$TR/best; [ -d $RT_ADAPTER ] || RT_ADAPTER=$TR/final
if want probe_rt && ! done_ probe_rt; then
  box=/dev/shm/p2_sandbox/probe_rt_$RKEY; mkdir -p $box
  ( cd $box && CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PY $ROOT/code/phase2/probe_p2.py --index $RD/index.jsonl \
      --adapter $RT_ADAPTER --out logs/phase2/probe/${RKEY}__retrained ) >> $D/probe.log 2>&1; rc=$?
  [ $rc = 0 ] || [ $rc = 3 ] || { say "probe_rt FAILED"; exit 1; }
  mark probe_rt
fi
if want decode_rt && ! done_ decode_rt; then
  say "decode with the retrained adapter $RT_ADAPTER"
  decode_jobs ${ARM}_rt $RT_ADAPTER S0ctl "" | run_jobs decode_${ARM}_rt; rc=$?
  [ $rc = 0 ] || [ $rc = 3 ] || { say "decode_rt had crashed jobs"; exit 1; }
  mark decode_rt
fi
if want gate_rt && ! done_ gate_rt; then
  gate ${ARM}_rt logs/phase2/decode/S0ctl logs/phase2/probe/${RKEY}__retrained/probe.json logs/phase2/gate/${ARM}_retrained && mark gate_rt
fi
say "arm $ARM done: logs/phase2/gate/${ARM}_zeroshot/gate.md, logs/phase2/gate/${ARM}_retrained/gate.md"
