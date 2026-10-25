#!/usr/bin/env bash
# Wave A on the 8-GPU box, one domain at a time (maths -> science -> code -> if -> chat), each time-boxed so a slow
# engine shortens every domain in proportion instead of dropping the last one:
#   per domain d: wait for its prompt list -> gen (--domains d --stop-at T_d) -> drafts -> training items ->
#                 verify (index >= 95% of d's kept records, build_summary no_draft == 0) -> clean source caches
#   T_d = GEN_START + (GEN_END - GEN_START) * cumulative share up to d (0.25, 0.40, 0.65, 0.80, 1.00),
#   GEN_END = DEADLINE - R, R = 1 h + the hours to train 128M (R_HOURS; 3 h if unknown).
# Then: push the corpus to HF (blocking) -> 3-way LR test only if DEADLINE - now >= R + 1 h (else LR 8e-5) ->
# main run to min(128M, the index's corpus tokens), one epoch -> checkpoint-A evaluations -> logs/c1b/gate_A.md.
# (The 32M drafter-confidence test is not run on the 17 h rental.) Every step is resumable: rerun after a failure.
#   tmux new -d -s wa 'bash code/corpus1b/run_wave_a.sh 2>&1 | tee -a logs/c1b/waveA.log'
# Env knobs: SGL_FLAGS, DEADLINE_FILE (START/END/DEADLINE, default /workspace/deadline.env), R_HOURS, LR (skip the
#            LR test), FORCE=1 (skip the stage-1 smoke requirement).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
DEADLINE_FILE=${DEADLINE_FILE:-/workspace/deadline.env}
. "$DEADLINE_FILE"                                               # START END DEADLINE (unix seconds)
PYT=/venv/train/bin/python; PYS=/venv/sgl/bin/python; TR=/venv/train/bin/torchrun
# SGLang on sm_120 (RTX PRO 6000 Blackwell): FlashInfer's JIT needs a CUDA >= 12.9 toolkit (installed beside the
# 12.8 one; /usr/local/cuda stays 12.8) and CUDA-graph capture needs a bigger FlashInfer workspace (gen: 128 running,
# drafts: 512). Without that toolkit: triton attention without CUDA graphs (about half the decode speed).
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
  SGL_FLAGS=${SGL_FLAGS:-}
  SGL_ENV="env CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
else
  SGL_FLAGS=${SGL_FLAGS:---attention-backend triton --disable-cuda-graph}
  SGL_ENV="env"
fi
GEN_ENV="$SGL_ENV SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"; DRAFT_ENV="$SGL_ENV SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
# drafts: 512 running x 32 slots x 152K-vocab fp32 logits (~10 GB per forward) OOMs at mem-fraction 0.80 on 96 GB
DRAFT_FLAGS=${DRAFT_FLAGS:---max-running 256 --mem-fraction 0.70}
W=logs/c1b/waveA_state; S1=logs/c1b/stage1; mkdir -p $W logs/c1b/gen/waveA logs/c1b/draft/waveA logs/c1b/train
done_(){ [ -f "$W/.done_$1" ]; }; mark(){ touch "$W/.done_$1"; }
say(){ echo "[waveA $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
ist(){ TZ=Asia/Kolkata date -d @"$1" '+%H:%M IST'; }

if [ "${FORCE:-0}" != "1" ]; then
  [ -f $S1/.done_smoke ] || { say "stage-1 smoke test has not passed ($S1/report.jsonl)"; exit 1; }
fi

per_gpu(){  # $1 = name, rest = command with {K}; one process per GPU; a failed one is relaunched once after 30 s
  local name=$1; shift; local pids=()                 # (e.g. SGLang's random internal port was still taken: EADDRINUSE)
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" > logs/c1b/$name.gpu$k.log 2>&1 \
      || { sleep 30; CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1; } ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}

IDX=logs/c1b/trainset/waveA/index.jsonl
T8(){ $TR --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py --index $IDX "$@"; }
# R = 1 h + the hours to train 128M. R_HOURS overrides; else the measured value ($W/R_hours, from a 1.5M-token
# benchmark on the first domain's items, below); else 3 h. GEN_END (and every later T_d) follows it.
set_R(){
  R_HOURS=${R_HOURS_OVERRIDE:-$( [ -f $W/R_hours ] && cat $W/R_hours || echo 3 )}
  R_S=$(awk -v r="$R_HOURS" 'BEGIN{printf "%d", r*3600}')
  GEN_END=$(( DEADLINE - R_S ))
}
R_HOURS_OVERRIDE=${R_HOURS:-}
set_R
EVAL_HOURS=${EVAL_HOURS:-4}; EVAL_S=$(awk -v h="$EVAL_HOURS" 'BEGIN{printf "%d", h*3600}')
DOMS=(maths science code if chat)
declare -A SHARE=([maths]=0.25 [science]=0.15 [code]=0.25 [if]=0.15 [chat]=0.20)   # cumulative 0.25 .40 .65 .80 1

adopt(){  # a step still running from an earlier orchestrator (after a restart): wait for it, then redo the step
  local n                                                     # (every step is resumable, so the redo is quick)
  while n=$(ps -eo args --no-headers | grep -c -- "$1") && [ "$n" -gt 0 ]; do
    say "waiting for $n running '$1' processes (started before this orchestrator)"; sleep 60
  done
}

verify_py(){  # $1 = domain: index entries >= 95% of its kept records, and this run's build_summary no_draft == 0
  $PYT - "$1" <<'EOF' | tee -a $W/verify.jsonl
import glob, json, sys
sys.path.insert(0, "code/corpus1b")
import common as C
d = sys.argv[1]
kept = sum(1 for f in glob.glob("logs/c1b/gen/waveA/gpu*/records.*.jsonl.gz") for r in C.read_jsonl(f)
           if r.get("kept") and r.get("domain") == d)
idx = sum(1 for e in C.read_jsonl("logs/c1b/trainset/waveA/index.jsonl") if e.get("domain") == d)
bs = json.load(open("logs/c1b/trainset/waveA/build_summary.json"))
nd = bs["counts"]["no_draft"]
late = bs["counts"].get("excluded_late_decontam", 0)
ok = kept > 0 and idx >= 0.95 * kept and nd == 0
print(json.dumps({"domain": d, "kept_records": kept, "index_entries": idx, "no_draft": nd,
                  "excluded_late_decontam": late, "ok": ok}), flush=True)
sys.exit(0 if ok else 1)
EOF
}

clean_py(){  # source caches of repos that only verified domains use (never eval / held-out repos, gsm8k, mbpp,
  # humaneval); the gen traces only while no generation runs (each gen wipes its own trace at startup); never
  # prompt lists, records, drafts, items or hf_cache/ (the models)
  $PYT - $(for x in "${DOMS[@]}"; do done_ verify_$x && echo $x; done) <<'EOF'
import glob, os, shutil, subprocess, sys
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
gen_running = subprocess.run("ps -eo args | grep -c '[g]en_sglang.py --wave'", shell=True,
                             capture_output=True, text=True).stdout.strip() not in ("", "0")
if not gen_running:
    for t in glob.glob("logs/c1b/gen/waveA/gpu*/trace"):
        shutil.rmtree(t, ignore_errors=True)
print(f"[clean] traces {'kept (a generation is running)' if gen_running else 'removed'}; "
      f"free disk {shutil.disk_usage('.').free / 1e9:.0f} GB", flush=True)
EOF
}

post(){  # CPU steps of a drafted domain: training items -> verify -> clean; a failure leaves $W/.failed_post_<d>
  local d=$1
  adopt "[t]o_trainset.py --wave A"
  if ! done_ trainset_$d; then
    say "training items for $d"
    $PYT code/corpus1b/to_trainset.py --wave A --domains $d > logs/c1b/trainset_waveA_$d.log 2>&1 \
      || { say "to_trainset failed for $d (logs/c1b/trainset_waveA_$d.log)"; touch $W/.failed_post_$d; return 1; }
    mark trainset_$d
  fi
  if ! done_ verify_$d; then
    verify_py $d || { say "VERIFY FAILED for $d ($W/verify.jsonl)"; touch $W/.failed_post_$d; return 1; }
    mark verify_$d
  fi
  if ! done_ clean_$d; then clean_py; mark clean_$d; fi
  say "$d post-processed: $(tail -1 $W/verify.jsonl)"
}
POST_PID=""
check_post(){ local x; for x in "${DOMS[@]}"; do [ -f $W/.failed_post_$x ] && { say "post-processing of $x FAILED: stopping"; exit 3; }; done; return 0; }
wait_post(){ if [ -n "$POST_PID" ]; then wait $POST_PID || true; POST_PID=""; fi; check_post; }

# The next GPU step: a domain whose drafts are pending (resume) first, else the first domain in DOMS order whose
# prompt list is ready (a later ready list never waits behind an earlier one). Its time box keeps its own share:
# T_d = GEN_START + window * (shares of the domains already scheduled + its own). While domain d+1 generates on
# the GPUs, domain d's CPU steps (items -> verify -> clean) run in the background, one domain at a time.
pick(){
  local d
  for d in "${DOMS[@]}"; do done_ gen_$d && ! done_ draft_$d && { echo $d; return; }; done
  for d in "${DOMS[@]}"; do
    done_ gen_$d && continue
    [ -f $S1/.done_prompts_$d ] && { echo $d; return; }      # a failed list never blocks the others (fix + rebuild)
  done
  echo ""
}
while true; do
  check_post
  if [ -n "$POST_PID" ] && ! kill -0 $POST_PID 2>/dev/null; then wait_post; fi
  if [ -z "$POST_PID" ] && [ -f $W/R_hours -o -n "$R_HOURS_OVERRIDE" ]; then   # a post-processing interrupted by
    for x in "${DOMS[@]}"; do                                                  # a restart resumes in the background
      if done_ draft_$x && ! done_ clean_$x; then ( post $x ) & POST_PID=$!; break; fi
    done
  fi
  left=0; for x in "${DOMS[@]}"; do done_ draft_$x || left=1; done; [ $left = 0 ] && break
  d=$(pick)
  if [ -z "$d" ]; then
    say "waiting for a prompt list$(for x in "${DOMS[@]}"; do [ -f $S1/.failed_prompts_$x ] && [ ! -f $S1/.done_prompts_$x ] && printf ' (%s failed: rebuilding)' $x; done)"
    sleep 60; continue
  fi
  if ! done_ gen_$d; then
    [ -f $W/gen_start ] || date +%s > $W/gen_start
    GEN_START=$(cat $W/gen_start)
    if [ -f $W/T_$d ]; then T=$(cat $W/T_$d); else
      # the time left until GEN_END, split over the domains not generated yet in proportion to their shares (for
      # the first domain this is GEN_START + window * share; when R moves GEN_END, the rest shrink in proportion)
      rest=0; for x in "${DOMS[@]}"; do done_ gen_$x || rest=$(awk -v a=$rest -v b=${SHARE[$x]} 'BEGIN{print a+b}'); done
      now=$(date +%s)
      T=$(awk -v n="$now" -v e="$GEN_END" -v c="${SHARE[$d]}" -v r="$rest" 'BEGIN{t = n + (e - n) * c / r; printf "%d", (t > n ? t : n)}')
      echo "$T" > $W/T_$d
    fi
    adopt "[g]en_sglang.py --wave A"
    say "gen $d until $(ist $T) (GEN_START $(ist $GEN_START), GEN_END $(ist $GEN_END), R ${R_HOURS} h)"
    per_gpu gen_waveA_$d "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave A --domains $d --stop-at $T --gpu {K} --num-gpus 8 $SGL_FLAGS" \
      || { say "a $d generation process failed: logs/c1b/gen_waveA_$d.gpu*.log; fix, rerun (it resumes)"; exit 2; }
    mark gen_$d
  fi
  if ! done_ draft_$d; then
    adopt "[d]raft_sglang.py --wave A"
    say "drafts after $d (SGLang, 8 GPUs)"
    per_gpu draft_waveA_$d "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave A --gpu {K} --num-gpus 8 $DRAFT_FLAGS $SGL_FLAGS" \
      || { say "a draft process failed: logs/c1b/draft_waveA_$d.gpu*.log; rerun resumes"; exit 2; }
    mark draft_$d
  fi
  if [ ! -f $W/R_hours ] && [ -z "$R_HOURS_OVERRIDE" ]; then
    # the first domain's items feed the training-speed benchmark, so it is post-processed before the next gen
    wait_post; post $d || check_post
    adopt "[t]rain_c1b.py --index"
    say "training-speed benchmark (1.5M tokens, 8 GPUs) to size R"
    T8 --out logs/c1b/train/bench --lr 8e-5 --max-tokens 1.5e6 --save-at-tokens 1e12 --save-every-tokens 1e12 \
      --eval-every-tokens 1e12 > logs/c1b/train/bench.log 2>&1 || say "benchmark failed (logs/c1b/train/bench.log): R stays 3 h"
    $PYT - <<'EOF' > $W/R_hours.tmp && mv $W/R_hours.tmp $W/R_hours || rm -f $W/R_hours.tmp
import json, sys
try:
    rows = [json.loads(l) for l in open("logs/c1b/train/bench/train.rank0.jsonl")]
except OSError:
    sys.exit(1)
rows = [r for r in rows if r.get("tokens", 0) > 3e5 and r.get("tokens_per_s")]      # past compile / warm-up
if not rows:
    sys.exit(1)
tps = sorted(r["tokens_per_s"] for r in rows)[len(rows) // 2]
print(f"{min(8.0, 1.0 + 128e6 / tps / 3600):.2f}")
EOF
    set_R
    say "measured R = ${R_HOURS} h (1 h + 128M at the benchmark's median tokens/s); GEN_END $(ist $GEN_END)"
  else
    wait_post                                                 # one to_trainset at a time: they append to one index
    ( post $d ) &
    POST_PID=$!
  fi
done
wait_post
for x in "${DOMS[@]}"; do done_ draft_$x && ! done_ clean_$x && { post $x || check_post; }; done
check_post

# the generated corpus goes to Hugging Face before any training starts
if ! done_ push_data; then
  say "pushing the wave-A corpus to Hugging Face before training"
  bash code/corpus1b/backup_hf.sh data || say "WARNING: data push failed (the backup loop retries)"
  mark push_data
fi

# learning rate: the 3-way test at 8M only if there is time for it
if [ -z "${LR:-}" ]; then
  if [ -f logs/c1b/train/lr_choice.txt ]; then
    LR=$(cat logs/c1b/train/lr_choice.txt)
  elif [ $(( DEADLINE - $(date +%s) )) -ge $(( R_S + 3600 )) ]; then
    for lr in 4e-5 8e-5 1.6e-4; do
      [ -f logs/c1b/train/lr_$lr/final/trainer_state.json ] && continue
      say "LR test $lr to 8M"
      T8 --out logs/c1b/train/lr_$lr --lr $lr --max-tokens 8e6 --save-at-tokens 8e6 --eval-every-tokens 1e12 \
        $( [ -d logs/c1b/train/lr_$lr ] && echo --resume ) > logs/c1b/train/lr_$lr.log 2>&1
    done
    $PYT - <<'EOF' > logs/c1b/train/lr_choice.txt
import json
best = None
for lr in ("4e-5", "8e-5", "1.6e-4"):
    rows = [json.loads(l) for l in open(f"logs/c1b/train/lr_{lr}/probe.jsonl")]
    ce = rows[-1]["all/ce"]
    best = (ce, lr) if best is None or ce < best[0] else best
print(best[1])
EOF
    LR=$(cat logs/c1b/train/lr_choice.txt)
  else
    say "no time for the LR test (DEADLINE - now < R + 1 h): LR 8e-5"
    LR=8e-5; echo "skipped" > logs/c1b/train/lr_test_skipped.txt
  fi
fi
say "learning rate $LR"

# the main run: one epoch of the index, at most 128M corpus tokens (never repeats data)
if ! done_ main; then
  MAXTOK=$($PYT -c "
import json; t = sum(json.loads(l)['corpus_tokens'] for l in open('$IDX')); print(int(min(128e6, t)))")
  echo "$MAXTOK" > $W/main_max_tokens
  say "main judge run to $MAXTOK tokens -> logs/c1b/train/main"
  TRAIN_STOP=$(( DEADLINE - EVAL_S ))                    # the evaluations keep EVAL_HOURS before DEADLINE
  say "main run stops by $(ist $TRAIN_STOP) (DEADLINE - ${EVAL_HOURS} h for the evaluations)"
  # the usual recipe (layer checkpointing as default); a checkpoint every 1M tokens (~880 steps, ~55 min) besides the
  # 1.84M milestone and `final` (~5-6 in all, ~0.3 GB each), all kept on the local disk while training runs; once it
  # finishes they are all pushed to the Hub and verified (hf_sync.py ckpt), then the 5 newest stay on disk
  T8 --out logs/c1b/train/main --lr $LR --max-tokens $MAXTOK --epochs 1 --stop-at $TRAIN_STOP \
    --save-every-tokens 5e5 --keep-rolling 1000 \
    $( [ -d logs/c1b/train/main ] && echo --resume ) >> logs/c1b/train/main.log 2>&1 && mark main
  say "training finished: pushing every checkpoint to the Hub (background) while the evaluations run"
  ( bash code/corpus1b/backup_hf.sh ckpt > logs/c1b/backup_ckpt_after_training.log 2>&1 ) &
fi
# checkpoint-A evaluations (ordered, deadline-bounded) and the gate
if ! done_ evalA; then
  say "checkpoint-A evaluations"
  bash code/corpus1b/eval_ckpt.sh checkpointA && mark evalA
fi
$PYT code/corpus1b/gate_report.py --out logs/c1b/gate_A.md
say "WAVE A DONE -> logs/c1b/gate_A.md"
