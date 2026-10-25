#!/usr/bin/env bash
# 128M -> 228M continuation, step 2 of 2 (after c228_setup.sh): convert new data with the Phase 1 pipeline (stock 30B
# answers via SGLang -> 1.7B drafts -> training items), continue the judge from main_128m/tok_000128000000 with the same
# recipe, evaluate, push everything. Resumable: every stage leaves logs/c1b/c228_state/.done_<stage>; rerun after any
# failure. Run from the repo root inside tmux:
#   tmux new -d -s c228 'bash code/corpus1b/ops/c228_run.sh 2>&1 | tee -a logs/c1b/c228.log'
# Needs /workspace/c228_deadline.env:   END=<unix time the rental ends>   MODE=full|box   (see c228_plan.py)
# Exit 3 = stop and ask the user (the plan says the target does not fit).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
set -a; . ./.env; set +a
export HF_HOME=$ROOT/hf_home TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
S=logs/c1b/c228_state; mkdir -p $S logs/c1b/gen/waveB logs/c1b/draft/waveB logs/c1b/train
for f in hfdata items smoke; do [ -f $S/.done_$f ] || { echo "setup not finished (.done_$f missing): run c228_setup.sh"; exit 1; }; done
C129="CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
GEN_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"
DRAFT_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
PYS=/venv/sgl/bin/python; PYT=/venv/train/bin/python; TR=/venv/train/bin/torchrun
RUN=logs/c1b/train/main_228m; FROM_CKPT=logs/c1b/train/main_128m/tok_000128000000
say(){ echo "[c228 $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
done_(){ [ -f "$S/.done_$1" ]; }; mark(){ touch "$S/.done_$1"; }
ist(){ TZ=Asia/Kolkata date -d @"$1" '+%d %b %H:%M IST'; }
count(){ $PYT code/corpus1b/ops/c228_count.py "$@" 2>/dev/null | tail -1; }
per_gpu(){  # name, command with {K}: all 8 GPUs; a failed GPU is relaunched once after 30 s; logs append
  local name=$1; shift; local pids=()
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1 \
      || { sleep 30; CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1; } ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}
draft_all(){ per_gpu "draft_c228_$1" "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave B --gpu {K} --num-gpus 8 --max-running 256 --mem-fraction 0.70" \
  && say "drafts done ($1)" || say "drafts: a GPU failed twice (logs/c1b/draft_c228_$1.gpu*.log); rerun resumes"; }
TS_PID=""
convert_bg(){  # to_trainset for one domain in the background; never two at once (they append to one index)
  [ -n "$TS_PID" ] && wait $TS_PID
  ( $PYT code/corpus1b/to_trainset.py --wave B --domains "$1" >> logs/c1b/trainset_c228_$1.log 2>&1 \
      && { touch "$S/.done_conv_$1"; say "items $1 converted: $(count $1)"; push_data; } \
      || say "to_trainset $1 FAILED (logs/c1b/trainset_c228_$1.log); rerun resumes" ) & TS_PID=$!
}
verify(){  # index entries >= 95% of the domain's kept records, no missing drafts (the Phase 1 rule)
  $PYT - "$1" <<'EOF'
import glob, json, sys
sys.path.insert(0, "code/corpus1b"); import common as C
d = sys.argv[1]
kept = sum(1 for f in glob.glob("logs/c1b/gen/waveB/gpu*/records.*.jsonl.gz") for r in C.read_jsonl(f) if r.get("kept") and r.get("domain") == d)
idx = sum(1 for e in C.read_jsonl("logs/c1b/trainset/waveB/index.jsonl") if e.get("domain") == d)
bs = json.load(open("logs/c1b/trainset/waveB/build_summary.json"))["counts"]
ok = kept > 0 and idx >= 0.95 * kept
print(json.dumps({"verify": d, "kept_records": kept, "index_entries": idx, "no_draft_last_run": bs["no_draft"], "ok": ok}), flush=True)
sys.exit(0 if ok else 1)
EOF
}
push_data(){ ( DATA_TIMEOUT=${1:-1500} bash code/corpus1b/backup_hf.sh data >> logs/c1b/backup_c228_explicit.log 2>&1 \
               && say "HF data push ok" || say "HF data push failed (the 20-min loop retries)" ) & }

# ---- 0. plan (once; frozen in logs/c1b/c228_state/plan.env) --------------------------------------------------------
[ -f $S/start.txt ] || date +%s > $S/start.txt
if [ ! -f $S/plan.env ]; then
  python3 code/corpus1b/ops/c228_plan.py; rc=$?
  if [ $rc -eq 3 ]; then
    say "SHORTFALL: MODE=full (100M converted + 100M trained + eval) does not fit before END. STOP AND ASK THE USER."
    mv $S/plan.env $S/plan_shortfall.env; exit 3
  fi
  [ $rc -eq 0 ] || { say "planner failed"; exit 1; }
fi
. $S/plan.env
say "plan: MODE=$MODE, eval $EVAL_MODE; training stops by $(ist $TRAIN_END); rental ends $(ist $END)"; cat $S/plan.md

# ---- 1. backups run the whole time: data every 20 min, checkpoints every 5 min (verified file by file) -------------
bg(){ pgrep -f "[b]ackup_hf.sh $1" >/dev/null || { setsid nohup env BACKUP_EVERY=1200 CKPT_EVERY=300 \
        bash code/corpus1b/backup_hf.sh $1 >> logs/c1b/backup_$1.log 2>&1 < /dev/null & disown; say "started backup $1"; }; }
bg loop; bg ckptloop

# ---- 2. the wave-B code generated on 30 Sep and never converted (~24.3M tokens): drafts + items -------------------
if ! done_ pending; then
  say "pending wave-B code: drafts + items"
  draft_all pending
  $PYT code/corpus1b/to_trainset.py --wave B --domains code >> logs/c1b/trainset_c228_code.log 2>&1
  verify code && { mark pending; say "pending code converted: $(count code)"; push_data; } \
    || say "pending code: verify failed; continuing (rerun later resumes)"
fi

# ---- 3. generate maths, chat, code (continuing wave B in the same hash order; done.txt skips what exists) ------------
GEN_CAP=$(python3 -c "print(int($TRAIN_END - 0.9 * float('$TRAIN_H') * 3600 - 1800))")   # generation never eats training
gen_box(){  # domain, hours -> one time-boxed generation pass on all 8 GPUs
  local d=$1 h=$2 stop
  stop=$(python3 -c "import time; print(int(min(time.time() + float('$h') * 3600, $GEN_CAP)))")
  [ "$stop" -le "$(( $(date +%s) + 300 ))" ] && { say "gen $d: no time left in the plan, skipped"; return 0; }
  say "gen $d until $(ist $stop) (planned ${h} h)"
  per_gpu gen_c228_$d "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave B --domains $d --gpu {K} --num-gpus 8 --stop-at $stop" \
    && say "gen $d box done: $(count $d)" || say "gen $d: a GPU failed twice (logs/c1b/gen_c228_$d.gpu*.log)"
}
for d in maths chat code; do
  D=${d^^}; h_var="GEN_H_$D"; t_var="TARGET_M_$D"; H=${!h_var}; TGT=${!t_var}
  if done_ gen_$d; then :; else
  t0=$(date +%s); gen_box $d "$H"
  if [ "$MODE" = full ]; then  # top up once toward 95% of the target if the measured rate leaves room
    extra=$($PYT - "$d" "$TGT" "$t0" <<'EOF'
import json, subprocess, sys, time
d, tgt, t0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
got = json.loads(subprocess.check_output(["/venv/train/bin/python", "code/corpus1b/ops/c228_count.py", d]).decode().splitlines()[-1])["generated_M"].get(d, 0.0)
el = max(0.05, (time.time() - t0) / 3600.0)
print(round(min(2.0, (tgt * 0.95 - got) / (got / el) * 1.05), 3) if 0 < got < 0.95 * tgt else 0)
EOF
)
    if python3 -c "import sys; sys.exit(0 if float('${extra:-0}') > 0.05 else 1)"; then
      say "gen $d below 95% of ${TGT}M: top-up box ${extra} h"; gen_box $d "$extra"
    fi
  fi
  mark gen_$d
  fi
  done_ draft_$d || { draft_all $d; mark draft_$d; }
  done_ conv_$d || convert_bg $d
done
[ -n "$TS_PID" ] && wait $TS_PID
if ! done_ convert; then
  for d in maths chat code; do
    done_ conv_$d || { $PYT code/corpus1b/to_trainset.py --wave B --domains $d >> logs/c1b/trainset_c228_$d.log 2>&1 && mark conv_$d; }
    verify $d || say "WARNING: $d conversion below 95% of kept records"
  done
  mark convert; say "conversion done: $(count)"; push_data
fi

# ---- 4. the training index (new items + 128M remainder + science/IF passes + pinned probe) --------------------------
if ! done_ index; then
  $PYT code/corpus1b/build_index_228m.py > $S/index.txt 2>&1; rc=$?
  grep -vE 'Warning|warn' $S/index.txt
  [ $rc -eq 0 ] && [ -s logs/c1b/trainset/index_228m.jsonl ] && { mark index; push_data; } \
    || { say "index build FAILED (rc=$rc, $S/index.txt)"; exit 1; }
fi

# ---- 5. train 128M -> MAX_TOKENS (stops at TRAIN_END at the latest), same recipe as main_128m ------------------------
if ! done_ train; then
  if [ "$MODE" = full ]; then
    left_h=$(python3 -c "import time; print(round(($TRAIN_END - time.time()) / 3600, 2))")
    python3 -c "import sys; sys.exit(0 if $left_h * 9.5 >= 0.95 * 100 else 1)" || {
      say "SHORTFALL: only ${left_h} h left for training (needs ~10.5 h for +100M). STOP AND ASK THE USER."
      say "(to proceed with fewer tokens: set MODE=box in /workspace/c228_deadline.env and rerun)"; exit 3; }
  fi
  mkdir -p $RUN; n=0
  while :; do
    if ls -d $RUN/roll_* $RUN/tok_* $RUN/final >/dev/null 2>&1; then FROM="--resume"; else FROM="--continue-from $FROM_CKPT"; fi
    say "training ($FROM), stops at 228M tokens or $(ist $TRAIN_END)"
    mark_line=$(wc -l < $RUN.log 2>/dev/null || echo 0)
    $TR --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
      --index logs/c1b/trainset/index_228m.jsonl --holdout-pids logs/c1b/trainset/holdout_pids_waveA.txt \
      --out $RUN $FROM --lr 8e-5 --max-tokens $MAX_TOKENS --epochs 1 \
      --stop-at $TRAIN_END --stop-at-file $RUN/STOP_AT --stop-file $RUN/STOP \
      --save-every-tokens 2e6 --keep-rolling 1000 --save-at-tokens 160e6,192e6,228e6 \
      --domain-weights maths=25,code=25,science=15,if=15,chat=20 --repeat-domains science,if --max-passes 4 \
      >> $RUN.log 2>&1
    tail -n +$((mark_line + 1)) $RUN.log | grep -q '^\[done\]' && break
    n=$((n + 1)); [ $n -gt 3 ] && { say "training crashed 4 times: STOP AND ASK THE USER ($RUN.log)"; exit 1; }
    [ "$(date +%s)" -ge "$TRAIN_END" ] && { say "training crashed after its stop time; moving on"; break; }
    say "training exited without [done]: relaunch $n in 60 s"; sleep 60
  done
  mark train; say "training finished: $(grep '^\[done\]' $RUN.log | tail -1)"
  bash code/corpus1b/backup_hf.sh ckpt >> logs/c1b/backup_c228_explicit.log 2>&1 && say "checkpoints verified on HF"
fi

# ---- 6. evaluation (no expert pool, base decode = SDAR-1.7B drafter + the new judge), compare with 128M and stock -----
if ! done_ eval && [ "$EVAL_MODE" != none ]; then
  bash code/corpus1b/ops/c228_eval.sh "$EVAL_MODE" "$EVAL_END" && mark eval
fi

# ---- 7. final push, verification, report ----------------------------------------------------------------------------
DATA_TIMEOUT=1200 bash code/corpus1b/backup_hf.sh once >> logs/c1b/backup_c228_explicit.log 2>&1
$PYT code/corpus1b/ops/c228_verify.py | tee $S/verify.txt
if tail -1 $S/verify.txt | grep -q "ALL PASS"; then date +%s > logs/c1b/READY_TO_KILL; say "ALL PASS: everything is on HF/GitHub. READY_TO_KILL written."
else say "verification FAILED: do not stop the box yet (see $S/verify.txt)"; fi
say "C228 DONE"
