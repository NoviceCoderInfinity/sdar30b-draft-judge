#!/usr/bin/env bash
# Phase 5, FIRST GPU session of the SGLang judge decoder: does code/judge_round/judge_sglang.py reproduce the HF harness decode
# (SDAR-30B + flag-gated judge LoRA, SDAR-1.7B drafter, full judge, re-draft every pass "rd1", --redraft-ban-rejected,
# --loop-guard 12) well enough to generate training data?  One 8-GPU box, but the smoke uses only GPU 0 (SGLang) and GPU 1 (harness).
#   install  the SGLang algorithms (corpus1b/sglang_plugin/install.py, then --check): proves the plugin files are in this
#            SGLang's package and BOTH JudgeLoopTraced and DraftTracedV2 are discoverable (fails when either is not listed)
#   decode   about 8 items of gsm8k500 (items[0::60]) through the SGLang decoder on GPU 0 and, at the same time, through the HF
#            harness (run_da_decode.py, --score) on GPU 1: proves the engine starts, the loop runs to EOS, records have the
#            promised fields, and gives both sides' wall time (planning only: a speed measured in SGLang is never a result)
#   parity   judge_parity.py on the two record dirs: token agreement, judge passes per block, drafter forwards per block, accuracy
#            and loops; PASS / FAIL (exit 0 / 3) on token agreement >= 0.95, |accuracy gap| <= 3.0 points, |passes/block gap| <= 5%
#   record   judge_sglang.py --mode record on the first 4 usable rows of logs/c1b/trainset/waveA/index.jsonl: proves the
#            training-state recorder runs and writes its states file
#   report   elapsed time per stage, both decode times, the parity verdict and numbers, the states-file summary
#   all      install decode parity record report   (install / decode failures stop it; parity FAIL is reported at the end)
# RULE: SGLang is for data generation only; every benchmark number and all training run on the HF harness.
#
# How to run (from anywhere; ROOT = the repo root, default: this file's repo):
#   bash code/judge_round/smoke_judge_sgl.sh install
#   tmux new -d -s smoke 'bash code/judge_round/smoke_judge_sgl.sh all 2>&1 | tee -a logs/phase5/judge_sgl/all.log'
# Every stage writes a marker logs/phase5/judge_sgl/.done_<stage> (start / end epoch seconds) ONLY when it succeeds; a finished
# stage is skipped on a rerun, FORCE=1 reruns every stage asked for (an existing decode output dir is moved to <dir>.bak.<epoch>,
# never deleted).  decode also keeps one marker per side (.done_decode_sgl / .done_decode_harness): after one side fails, a rerun
# repeats only that side.  parity is cheap and reads whatever the decode dirs hold, so it is recomputed on every call.
#
# 8 items are too few to judge accuracy: one flipped item is a 11-point gap (the parity report says so).  Read the token and
# pass numbers of the smoke; for the accuracy gate run NSHARDS=4 (125 items) or more, or relax it: PARITY_FLAGS="--max-acc-gap 12".
#
# Env knobs (defaults):
#   ITEMS=gsm8k500 SHARD=0 NSHARDS=60   the same items[SHARD::NSHARDS] on both sides
#   SGL_GPU=0 HF_GPU=1                  physical GPUs of the SGLang decode and of the harness decode
#   ADAPTER=logs/c1b/train/main_128m/tok_000128000000   the flag-gated judge LoRA (the harness loads it; the SGLang side takes its own default)
#   MERGED=logs/phase4/train/main/merged                the Phase 4 drafter
#   LG=12                               loop-guard n-gram, both sides
#   SGL_FLAGS, HF_FLAGS, PARITY_FLAGS, RECORD_FLAGS   extra flags appended to judge_sglang.py --mode decode, run_da_decode.py,
#                                       judge_parity.py, judge_sglang.py --mode record
#   SERIAL=1                            run the two decodes one after the other (default: both at once, they use different GPUs)
#   RECORD_INDEX=logs/c1b/trainset/waveA/index.jsonl  RECORD_N=4  STATES_GLOB='<record out>/**/states*'
#   OUT=logs/phase5/judge_sgl           stage markers, logs, decode/ harness/ record/ parity.json
#   SANDBOX=/dev/shm/p5_sandbox/judge_sgl   cwd of the harness (AGENTS.md 9 (p): never run a scoring decode from the repo root)
#   PYT=/venv/p2/bin/python (harness, parity, helpers)   PYS=/venv/sgl/bin/python (SGLang)   SGL_ENV_FILE=logs/phase5/sgl_setup/sgl_env.sh
#   SGLANG_FLASHINFER_WORKSPACE_SIZE (4 GiB)   FORCE
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PYT=${PYT:-/venv/p2/bin/python}
PYS=${PYS:-/venv/sgl/bin/python}
SGL_ENV_FILE=${SGL_ENV_FILE:-logs/phase5/sgl_setup/sgl_env.sh}      # CUDA_HOME / PATH for FlashInfer's JIT on sm_120 (setup_sgl.sh)
ITEMS=${ITEMS:-gsm8k500}; SHARD=${SHARD:-0}; NSHARDS=${NSHARDS:-60}
SGL_GPU=${SGL_GPU:-0}; HF_GPU=${HF_GPU:-1}
ADAPTER=${ADAPTER:-logs/c1b/train/main_128m/tok_000128000000}
MERGED=${MERGED:-logs/phase4/train/main/merged}
LG=${LG:-12}
SGL_FLAGS=${SGL_FLAGS:-}; HF_FLAGS=${HF_FLAGS:-}; PARITY_FLAGS=${PARITY_FLAGS:-}; RECORD_FLAGS=${RECORD_FLAGS:-}
SERIAL=${SERIAL:-0}
RECORD_INDEX=${RECORD_INDEX:-logs/c1b/trainset/waveA/index.jsonl}; RECORD_N=${RECORD_N:-4}
OUT=${OUT:-logs/phase5/judge_sgl}
SANDBOX=${SANDBOX:-/dev/shm/p5_sandbox/judge_sgl}
FORCE=${FORCE:-0}
STAGES="install decode parity record"
cd "$ROOT" || exit 1
L=$OUT; D=$L/decode; H=$L/harness; R=$L/record
mkdir -p "$L"
say(){ echo "[smoke $(date +%H:%M:%S)] $*" | tee -a "$L/smoke.log"; }
abs(){ case $1 in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }
export C1B_CODE_DIR=$ROOT/code      # the algorithm plugins import draftalign.loop_guard / draft_core from the repo's code/

sgl_env(){   # the SGLang environment of one process (call inside a subshell)
  [ -f "$SGL_ENV_FILE" ] && . "$SGL_ENV_FILE"
  export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-4294967296}
}
need_sgl(){ [ -x "$PYS" ] || { say "no $PYS: run code/judge_round/setup_sgl.sh first"; return 1; }; }
mark(){   # mark NAME T0: write $L/.done_NAME with start / end epoch seconds
  printf 'start=%s\nend=%s\nat=%s\n' "$2" "$(date +%s)" "$(date '+%Y-%m-%d %H:%M:%S')" > "$L/.done_$1"; }
aside(){   # FORCE: an existing output dir is moved aside, never deleted
  if [ "$FORCE" = 1 ] && [ -d "$1" ]; then mv "$1" "$1.bak.$(date +%s)"; fi; mkdir -p "$1"; }
nrec(){ cat "$@" 2>/dev/null | grep -c . ; }

# ---------------------------------------------------------------------------
stage_install(){
  need_sgl || return 1
  ( sgl_env; $PYS code/corpus1b/sglang_plugin/install.py ) 2>&1 | tee -a "$L/install.log"
  [ ${PIPESTATUS[0]} = 0 ] || { say "install.py failed (see $L/install.log)"; return 1; }
  local chk; chk=$( ( sgl_env; $PYS code/corpus1b/sglang_plugin/install.py --check ) 2>&1 ); local rc=$?
  echo "$chk" | tee -a "$L/install.log"
  local n bad=""
  for n in JudgeLoopTraced DraftTracedV2; do echo "$chk" | grep -q "$n" || bad="$bad $n"; done
  [ -z "$bad" ] || { say "install --check does not list:$bad"; return 1; }
  [ $rc = 0 ] || { say "install.py --check exited $rc"; return 1; }
  say "install: JudgeLoopTraced and DraftTracedV2 are discoverable"
}

dec_sgl(){   # the SGLang decode on SGL_GPU; one marker for this side, its wall seconds in the marker
  local t0; t0=$(date +%s)
  ( sgl_env; export CUDA_VISIBLE_DEVICES=$SGL_GPU
    $PYS code/judge_round/judge_sglang.py --mode decode --items $ITEMS --shard $SHARD --nshards $NSHARDS --gpu $SGL_GPU \
      --redraft-every 1 --redraft-ban --loop-guard $LG --out "$D" $SGL_FLAGS ) > "$L/decode_sgl.log" 2>&1 || return 1
  mark decode_sgl $t0
}
dec_harness(){   # the HF harness on HF_GPU, scored; cwd = a throwaway sandbox, absolute repo paths
  local t0; t0=$(date +%s)
  mkdir -p "$SANDBOX"
  ( cd "$SANDBOX" && CUDA_VISIBLE_DEVICES=$HF_GPU $PYT "$ROOT/code/draftalign/run_da_decode.py" --model b32 --items $ITEMS \
      --drafter "$(abs "$MERGED")" --draft s8 --judge full --adapter "$(abs "$ADAPTER")" --full-redraft-every 1 \
      --redraft-ban-rejected --loop-guard $LG --shard $SHARD --nshards $NSHARDS --gpu $HF_GPU --score --no-trace \
      --out "$(abs "$H")" $HF_FLAGS ) > "$L/decode_harness.log" 2>&1 || return 1
  mark decode_harness $t0
}
stage_decode(){
  need_sgl || return 1
  [ -f "$L/.done_install" ] || say "warning: the install stage has not run (no $L/.done_install)"
  [ -d "$(abs "$ADAPTER")" ] || { say "no judge adapter at $ADAPTER (ADAPTER=)"; return 1; }
  [ -d "$(abs "$MERGED")" ] || { say "no drafter at $MERGED (MERGED=)"; return 1; }
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ng; ng=$(nvidia-smi -L | grep -c GPU)
    [ "$ng" -gt "$SGL_GPU" ] && [ "$ng" -gt "$HF_GPU" ] || { say "need GPUs $SGL_GPU and $HF_GPU, nvidia-smi lists $ng"; return 1; }
  fi
  local want_sgl=1 want_h=1 psg ph rsg=0 rh=0
  if [ "$FORCE" = 1 ]; then rm -f "$L/.done_decode_sgl" "$L/.done_decode_harness"; fi
  [ -f "$L/.done_decode_sgl" ] && { want_sgl=0; say "decode: SGLang side already done"; }
  [ -f "$L/.done_decode_harness" ] && { want_h=0; say "decode: harness side already done"; }
  [ $want_sgl = 1 ] && aside "$D"
  [ $want_h = 1 ] && aside "$H"
  if [ "$SERIAL" = 1 ]; then
    [ $want_sgl = 1 ] && { say "decode: SGLang on GPU $SGL_GPU ($ITEMS items[$SHARD::$NSHARDS])"; dec_sgl || rsg=1; }
    [ $want_h = 1 ] && { say "decode: harness on GPU $HF_GPU"; dec_harness || rh=1; }
  else
    [ $want_sgl = 1 ] && { say "decode: SGLang on GPU $SGL_GPU ($ITEMS items[$SHARD::$NSHARDS]) -> $D, log $L/decode_sgl.log"; dec_sgl & psg=$!; }
    [ $want_h = 1 ] && { say "decode: harness on GPU $HF_GPU -> $H, log $L/decode_harness.log"; dec_harness & ph=$!; }
    [ $want_sgl = 1 ] && { wait $psg || rsg=1; }
    [ $want_h = 1 ] && { wait $ph || rh=1; }
  fi
  [ $rsg = 0 ] || say "decode: the SGLang side FAILED (see $L/decode_sgl.log)"
  [ $rh = 0 ] || say "decode: the harness side FAILED (see $L/decode_harness.log)"
  [ $rsg = 0 ] && [ $rh = 0 ] || return 1
  local nsg nh; nsg=$(nrec "$D"/gpu*/records.jsonl); nh=$(nrec "$H"/*.jsonl)
  say "decode: $nsg SGLang records, $nh harness records"
  [ "$nsg" -gt 0 ] && [ "$nh" -gt 0 ] || { say "decode: a side wrote no records"; return 1; }
  [ "$nsg" = "$nh" ] || say "warning: the two sides decoded different numbers of items (parity joins on (split, idx))"
}

stage_parity(){
  [ -d "$D" ] && [ -d "$H" ] || { say "no $D or $H: run the decode stage first"; return 1; }
  $PYT code/judge_round/judge_parity.py --sglang "$D" --harness "$H" --out "$L/parity.json" $PARITY_FLAGS 2>&1 | tee -a "$L/parity.log"
  local rc=${PIPESTATUS[0]}
  say "parity: exit $rc (0 = PASS, 3 = FAIL; report $L/parity.json)"
  return $rc
}

stage_record(){
  need_sgl || return 1
  [ -f "$RECORD_INDEX" ] || { say "no $RECORD_INDEX (RECORD_INDEX=)"; return 1; }
  aside "$R"
  $PYT - "$RECORD_INDEX" "$L/record_pids.json" "$RECORD_N" <<'PY' || return 1    # the first N usable rows of the index -> {"pids": [...]}
import json, sys
idx, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pids = []
for line in open(idx):
    if not line.strip(): continue
    e = json.loads(line)
    if int(e.get("usable", 1)): pids.append(str(e["pid"]))
    if len(pids) == n: break
if len(pids) < n: sys.exit(f"{idx}: only {len(pids)} usable rows")
json.dump({"pids": pids}, open(out, "w")); print(f"record pids ({n}): {pids}")
PY
  say "record: judge_sglang.py --mode record on $RECORD_N items of $RECORD_INDEX -> $R"
  ( sgl_env; export CUDA_VISIBLE_DEVICES=$SGL_GPU
    $PYS code/judge_round/judge_sglang.py --mode record --index "$RECORD_INDEX" --pids "$L/record_pids.json" --gpu $SGL_GPU \
      --redraft-every 1 --redraft-ban --loop-guard $LG --out "$R" $RECORD_FLAGS ) > "$L/record.log" 2>&1 \
    || { say "record: judge_sglang.py failed (see $L/record.log)"; return 1; }
  states_summary
}
states_summary(){   # the states file(s) the recorder wrote: list them with size and, for .npz / .jsonl, their content shape
  STATES_GLOB=${STATES_GLOB:-$R/**/states*} $PYT - <<'PY'
import glob, os, sys
files = sorted(f for f in glob.glob(os.environ["STATES_GLOB"], recursive=True) if os.path.isfile(f))
if not files:
    print(f"ERROR: no states file matches {os.environ['STATES_GLOB']}"); sys.exit(1)
empty = 0
for f in files:
    size = os.path.getsize(f); empty += size == 0
    line = f"  {f}  {size / 1e6:.2f} MB"
    try:
        if f.endswith(".npz"):
            import numpy as np
            z = np.load(f, allow_pickle=False)
            line += "  " + ", ".join(f"{k}{list(z[k].shape)}" for k in z.files[:8]) + (" ..." if len(z.files) > 8 else "")
        elif f.endswith((".jsonl", ".json")):
            line += f"  {sum(1 for l in open(f) if l.strip())} lines"
    except Exception as e:
        line += f"  (content not summarised: {type(e).__name__}: {e})"
    print(line)
if empty: print(f"ERROR: {empty} empty states file(s)"); sys.exit(1)
print(f"states: {len(files)} file(s)")
PY
}

stage_report(){
  OUT_DIR=$L STAGES="$STAGES" SGL_GPU=$SGL_GPU HF_GPU=$HF_GPU $PYT - <<'PY'
import datetime, json, os
L, stages = os.environ["OUT_DIR"], os.environ["STAGES"].split()
def mk(name):
    p = os.path.join(L, ".done_" + name)
    if not os.path.exists(p): return None
    m = dict(l.strip().split("=", 1) for l in open(p) if "=" in l)
    return float(m["start"]), float(m["end"])
print(f"stage timing (markers in {L}):")
tot = 0.0
for s in stages:
    t = mk(s)
    if t is None: print(f"  {s:8s} not done"); continue
    tot += t[1] - t[0]
    print(f"  {s:8s} {datetime.datetime.fromtimestamp(t[0]):%m-%d %H:%M} -> {datetime.datetime.fromtimestamp(t[1]):%H:%M}  {t[1] - t[0]:7.0f} s")
for name, label in (("decode_sgl", f"SGLang on GPU {os.environ['SGL_GPU']}"), ("decode_harness", f"harness on GPU {os.environ['HF_GPU']}")):
    t = mk(name)
    print(f"    decode {label}: " + ("not done" if t is None else f"{t[1] - t[0]:.0f} s (includes model load; planning only, not a result)"))
print(f"  {'total':8s} {tot:44.0f} s")
p = os.path.join(L, "parity.json")
if not os.path.exists(p):
    print("parity: no report yet")
else:
    r = json.load(open(p)); o, v = r["overall"], r["verdict"]
    print(f"parity: {v['status']}" + (f"  ({v['message']})" if v["message"] else "") + f"   [{p}]")
    if o.get("n"):
        a = o["accuracy"]
        print(f"  items {o['n']}  exact {o['exact_rate']:.3f}  tokagr {o['token_agreement']:.4f}  prefix {o['prefix_agreement']:.3f}  |dlen| {o['mean_abs_len_diff']:.1f}")
        for k in ("passes_per_block", "drafter_fwd_per_block"):
            x = o[k]; f = lambda y, s: "-" if y is None else format(y, s)
            print(f"  {k:22s} sglang {f(x['sglang'], '.3f')}  harness {f(x['harness'], '.3f')}  gap {f(x['rel_gap'], '+.1%')}")
        print(f"  accuracy  sglang {a['sglang']}  harness {a['harness']}  gap {a['gap_points']} points  (n {a['n_scored']})  loops {o['loops']}")
PY
  if [ -f "$L/.done_record" ]; then echo "record: done; states files:"; find "$R" -type f -name 'states*' 2>/dev/null | head -5 | sed 's/^/  /'
  else echo "record: not done"; fi
}

# ---------------------------------------------------------------------------
run_stage(){   # run_stage NAME -> the stage's exit code; a marker only on success
  local s=$1 t0 rc
  if [ -f "$L/.done_$s" ] && [ "$FORCE" != 1 ] && [ "$s" != parity ]; then say "$s: done ($(grep '^at=' "$L/.done_$s" | cut -d= -f2)); FORCE=1 reruns"; return 0; fi
  rm -f "$L/.done_$s"; t0=$(date +%s); say "$s: start"
  stage_$s; rc=$?
  if [ $rc = 0 ]; then mark "$s" "$t0"; say "$s: done in $(( $(date +%s) - t0 )) s"; else say "$s: FAILED (exit $rc)"; fi
  return $rc
}
case ${1:-help} in
install|decode|parity|record) run_stage "$1"; exit $? ;;
report) stage_report ;;
all)
  run_stage install || exit 1
  run_stage decode || exit 1
  rc=0
  run_stage parity || rc=$?
  run_stage record || { r=$?; [ $rc = 0 ] && rc=$r; }
  stage_report
  exit $rc ;;
*) echo "usage: $0 install|decode|parity|record|report|all   (see the header for the env knobs)"; exit 2 ;;
esac
