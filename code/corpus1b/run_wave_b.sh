#!/usr/bin/env bash
# Wave B (~74 h), only after gate_A.md says CONTINUE: generate the other 872M tokens -> drafts -> training items ->
# continue the SAME run from its checkpoint A to 1B (the WSD decay happens over the last 15%) -> final evaluation.
# Generation runs one domain at a time (maths -> science -> code -> if -> chat; a domain whose list is not ready never
# blocks a later one), each time-boxed so a slow engine shortens every domain in proportion instead of dropping
# the last one: T_d = GEN_START + (GEN_END - GEN_START) * cumulative share up to d (0.25, 0.40, 0.65, 0.80, 1.00),
# GEN_END = DEADLINE - R, R = 1 h + the hours to train 872M (R_HOURS; else wave A's measured speed; else 12 h).
# Per domain: gen (--domains d --stop-at T_d) -> drafts -> items -> verify (index >= 95% of d's kept records,
# no_draft == 0) -> clean the source caches only finished domains use. The corpus is pushed to HF before training.
# Resumable like run_wave_a.sh. CONF=1 trains with the confidence embedding (if the 32M test said so); when
# logs/c1b/train/lr_choice.txt names another LR than the main run's, 0 -> checkpoint A is retrained at it first.
# Env knobs: SGL_FLAGS, DEADLINE_FILE (START/END/DEADLINE, default /workspace/deadline.env), R_HOURS, CONF, FORCE.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
DEADLINE_FILE=${DEADLINE_FILE:-/workspace/deadline.env}
. "$DEADLINE_FILE"                                               # START END DEADLINE (unix seconds)
PYT=/venv/train/bin/python; PYS=/venv/sgl/bin/python; TR=/venv/train/bin/torchrun
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then                    # SGLang on sm_120: see run_wave_a.sh
  SGL_FLAGS=${SGL_FLAGS:-}; SGL_ENV="env CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
else
  SGL_FLAGS=${SGL_FLAGS:---attention-backend triton --disable-cuda-graph}; SGL_ENV="env"
fi
GEN_ENV="$SGL_ENV SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"; DRAFT_ENV="$SGL_ENV SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
W=logs/c1b/waveB_state; S1=logs/c1b/stage1; mkdir -p $W logs/c1b/gen/waveB logs/c1b/draft/waveB logs/c1b/train
done_(){ [ -f "$W/.done_$1" ]; }; mark(){ touch "$W/.done_$1"; }
say(){ echo "[waveB $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
ist(){ TZ=Asia/Kolkata date -d @"$1" '+%d %b %H:%M IST'; }
grep -q "CONTINUE" logs/c1b/gate_A.md || [ "${FORCE:-0}" = "1" ] || { say "gate_A.md does not say CONTINUE"; exit 1; }
per_gpu(){ local name=$1; shift; local pids=()
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" > logs/c1b/$name.gpu$k.log 2>&1 & pids+=($!); done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc; }

# R: 1 h + hours to train 872M; default from wave A's benchmark (R_A = 1 h + hours for 128M)
if [ -z "${R_HOURS:-}" ]; then
  if [ -f logs/c1b/waveA_state/R_hours ]; then
    R_HOURS=$(awk -v a="$(cat logs/c1b/waveA_state/R_hours)" 'BEGIN{printf "%.2f", 1 + (a - 1) * 872 / 128}')
  else R_HOURS=12; fi
fi
R_S=$(awk -v r="$R_HOURS" 'BEGIN{printf "%d", r*3600}'); GEN_END=$(( DEADLINE - R_S ))
DOMS=(maths science code if chat)
declare -A SHARE=([maths]=0.25 [science]=0.15 [code]=0.25 [if]=0.15 [chat]=0.20)
pick(){
  local d
  for d in "${DOMS[@]}"; do done_ gen_$d && ! done_ clean_$d && { echo $d; return; }; done
  for d in "${DOMS[@]}"; do done_ gen_$d && continue; [ -f $S1/.done_prompts_$d ] && { echo $d; return; }; done
  echo ""
}
while true; do
  left=0; for x in "${DOMS[@]}"; do done_ clean_$x || left=1; done; [ $left = 0 ] && break
  d=$(pick)
  [ -z "$d" ] && { say "waiting for a prompt list"; sleep 60; continue; }
  if ! done_ gen_$d; then
    [ -f $W/gen_start ] || date +%s > $W/gen_start
    GEN_START=$(cat $W/gen_start)
    if [ -f $W/T_$d ]; then T=$(cat $W/T_$d); else
      cum=${SHARE[$d]}; for x in "${DOMS[@]}"; do [ $x != $d ] && [ -f $W/T_$x ] && cum=$(awk -v a=$cum -v b=${SHARE[$x]} 'BEGIN{print a+b}'); done
      T=$(awk -v s="$GEN_START" -v e="$GEN_END" -v c="$cum" 'BEGIN{printf "%d", s + (e - s) * c}'); echo "$T" > $W/T_$d
    fi
    say "gen $d until $(ist $T) (GEN_END $(ist $GEN_END), R ${R_HOURS} h)"
    per_gpu gen_waveB_$d "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave B --domains $d --stop-at $T --gpu {K} --num-gpus 8 $SGL_FLAGS" \
      || { say "a $d generation process failed: logs/c1b/gen_waveB_$d.gpu*.log; rerun resumes"; exit 2; }
    mark gen_$d
  fi
  if ! done_ draft_$d; then
    per_gpu draft_waveB_$d "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave B --gpu {K} --num-gpus 8 $SGL_FLAGS" \
      || { say "a draft process failed: logs/c1b/draft_waveB_$d.gpu*.log; rerun resumes"; exit 2; }
    mark draft_$d
  fi
  if ! done_ trainset_$d; then
    $PYT code/corpus1b/to_trainset.py --wave B > logs/c1b/trainset_waveB_$d.log 2>&1 || { say "to_trainset failed"; exit 2; }
    mark trainset_$d
  fi
  if ! done_ verify_$d; then
    if ! $PYT - "$d" <<'EOF' | tee -a $W/verify.jsonl
import glob, json, sys
sys.path.insert(0, "code/corpus1b")
import common as C
d = sys.argv[1]
kept = sum(1 for f in glob.glob("logs/c1b/gen/waveB/gpu*/records.*.jsonl.gz") for r in C.read_jsonl(f)
           if r.get("kept") and r.get("domain") == d)
idx = sum(1 for e in C.read_jsonl("logs/c1b/trainset/waveB/index.jsonl") if e.get("domain") == d)
nd = json.load(open("logs/c1b/trainset/waveB/build_summary.json"))["counts"]["no_draft"]
ok = kept > 0 and idx >= 0.95 * kept and nd == 0
print(json.dumps({"domain": d, "kept_records": kept, "index_entries": idx, "no_draft": nd, "ok": ok}), flush=True)
sys.exit(0 if ok else 1)
EOF
    then say "VERIFY FAILED for $d ($W/verify.jsonl): stopping"; exit 3; fi
    mark verify_$d
  fi
  if ! done_ clean_$d; then
    $PYT - $(for x in "${DOMS[@]}"; do done_ verify_$x && echo $x; done) <<'EOF'
import glob, os, shutil, sys
sys.path.insert(0, "code/corpus1b")
import sources as S
finished = set(sys.argv[1:])
keep = {"openai/gsm8k", "google-research-datasets/mbpp", "openai/openai_humaneval", "openai_humaneval"}
keep |= {e["repo"] for e in S.EVAL_SETS} | {h["repo"] for h in S.HELD_OUT} | {h["join"] for h in S.HELD_OUT if "join" in h}
by_repo = {}
for s in S.SOURCES:
    by_repo.setdefault(s["repo"], set()).add(s["domain"])
hf = os.environ["HF_HOME"]
for repo, doms in sorted(by_repo.items()):
    if repo in keep or not doms <= finished:
        continue
    org, name = repo.split("/", 1)
    for p in [os.path.join(hf, "hub", f"datasets--{org}--{name}")] + glob.glob(os.path.join(hf, "datasets", f"{org}___{name}*")):
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"[clean] removed {p}", flush=True)
for t in glob.glob("logs/c1b/gen/waveB/gpu*/trace"):
    shutil.rmtree(t, ignore_errors=True)
EOF
    mark clean_$d
  fi
  say "$d done: $(tail -1 $W/verify.jsonl)"
done

if ! done_ push_data; then
  say "pushing the wave-B corpus to Hugging Face before training"
  bash code/corpus1b/backup_hf.sh data || say "WARNING: data push failed (the backup loop retries)"
  mark push_data
fi

MAIN_LR=$($PYT -c "import json; print(json.load(open('logs/c1b/train/main/config.json'))['lr'])")
LR=$(cat logs/c1b/train/lr_choice.txt 2>/dev/null || echo "$MAIN_LR")
START=logs/c1b/train/main/tok_000128000000; [ -d $START ] || START=logs/c1b/train/main/final
if ! done_ train; then
  say "continue the run to 1B at LR $LR"
  EXTRA=""; [ "${CONF:-0}" = "1" ] && EXTRA="--conf-embed"
  RETRAIN=$($PYT -c "print(int(${CONF:-0} == 1 or abs(float('$LR') - float('$MAIN_LR')) > 1e-12))")
  if [ "$RETRAIN" = "1" ]; then       # CONF=1 and/or the LR test picked another LR: retrain 0 -> checkpoint A first
    A_TOK=$($PYT -c "import json; print(int(json.load(open('$START/trainer_state.json'))['tokens']))")
    say "retrain 0 -> $A_TOK tokens at LR $LR ${EXTRA} before continuing"
    $TR --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py --index logs/c1b/trainset/waveA/index.jsonl \
      --out logs/c1b/train/main_re --lr $LR $EXTRA --max-tokens $A_TOK --epochs 1 \
      $( [ -d logs/c1b/train/main_re ] && echo --resume ) > logs/c1b/train/main_re.log 2>&1
    START=logs/c1b/train/main_re/tok_000128000000; [ -d $START ] || START=logs/c1b/train/main_re/final
  fi
  $TR --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py --index logs/c1b/trainset/waveB/index.jsonl \
    --out logs/c1b/train/main_b --lr $LR $EXTRA --continue-from $START --max-tokens 1e9 \
    $( [ -d logs/c1b/train/main_b ] && echo --resume ) > logs/c1b/train/main_b.log 2>&1 && mark train
fi
done_ evalF || { say "final evaluation"; bash code/corpus1b/eval_ckpt.sh final && mark evalF; }
say "WAVE B DONE -> logs/c1b/eval/final/summary.md"
