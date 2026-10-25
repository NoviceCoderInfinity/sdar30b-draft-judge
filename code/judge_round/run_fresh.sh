#!/usr/bin/env bash
# Phase 5, fresh judge round: generate a FRESH corpus wave "C" (prompts never decoded before) for the next judge training
# round, on the 8-GPU box, and pack it with a repeated slice of waves A/B into ONE training index.
#   targets  how many prompts per domain, projected GPU-hours -> logs/c1b/fresh_targets.json          (CPU)
#   lists    corpus1b/fresh_lists.py: wave-C prompt lists + manifest.json in $OUT_LISTS; IF shortfall > 20% = error  (CPU)
#   gen      corpus1b/gen_sglang.py --wave C, one SGLang engine per GPU (/venv/sgl); prints kept counts per domain
#   drafts   corpus1b/draft_sglang.py --wave C --drafter-dir $MERGED, one SGLang engine per GPU
#   pack     corpus1b/to_trainset.py --wave C -> logs/c1b/trainset/waveC/index.jsonl + items/                (CPU)
#   index    judge_round/build_fresh_index.py: wave C (fresh) + science repeated from A/B -> trainset/fresh/index.jsonl
#   report   build_summary.json as a table + elapsed time per stage (from the stage markers)
#   all      targets lists gen drafts pack index report
# RULE: SGLang is for data generation only; every benchmark number and all training run on the HF harness.
#
# How to run (from anywhere; ROOT = the repo root, default: this file's repo):
#   bash code/judge_round/run_fresh.sh targets                 # look at the numbers first
#   tmux new -d -s fresh 'bash code/judge_round/run_fresh.sh all 2>&1 | tee -a logs/c1b/fresh/all.log'
# Every stage writes a marker logs/c1b/fresh/.done_<stage> (start / end epoch seconds); a finished stage is skipped on a
# rerun, FORCE=1 reruns every stage asked for.  gen and drafts are resumable (done.txt / draft chunks), so a rerun after a
# crash continues.  `lists` refuses to run once wave C has started decoding (its done.txt would shrink the lists), unless
# LISTS_AFTER_GEN=1.  `targets` is read back by lists / gen / report from logs/c1b/fresh_targets.json: after changing a
# knob, rerun `FORCE=1 ... targets`.
#
# Env knobs (defaults):
#   TOKENS=52e6          corpus tokens (prompt + answer) of the whole round, repeated science included
#   MARGIN=1.2           safety factor on the number of prompts (keep rates vary)
#   NGPU=8               GPUs, one gen / draft process each
#   MIX                  token shares: maths=0.241,code=0.225,chat=0.201,if=0.179,science=0.153 (the 128M judge index)
#   REPEAT_SCIENCE       corpus tokens of science repeated from waves A/B (default: TOKENS * the science share of MIX)
#   MERGED=logs/phase4/train/main/merged   the Phase 4 drafter (merged weights) that writes the drafts
#   OUT_LISTS=data/c1b/prompt_lists_fresh  where fresh_lists.py writes the wave-C prompt lists
#   KEEP_RATE, TOK_PER_KEPT, MTOK_PER_GPUH   the measured Phase 1 constants (kept answers / decoded prompt; corpus tokens
#                        per kept item; kept-token Mtok per GPU-hour), same "domain=value,..." form as MIX
#   HOLD=logs/phase4/train/main/holdout_pids.json, HOLD_A=logs/c1b/trainset/holdout_pids_waveA.txt   never repeated
#   GEN_FLAGS, DRAFT_FLAGS (default "--max-running 256 --mem-fraction 0.70", run_wave_a.sh: 512 running OOMs the 1.7B's
#   logits at 0.80), SGLANG_FLASHINFER_WORKSPACE_SIZE (4 GiB), PYT=/venv/p2/bin/python (HF / CPU stages),
#   PYS=/venv/sgl/bin/python (SGLang stages), FORCE, LISTS_AFTER_GEN.
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PYT=${PYT:-/venv/p2/bin/python}
PYS=${PYS:-/venv/sgl/bin/python}
NGPU=${NGPU:-8}
TOKENS=${TOKENS:-52e6}
MARGIN=${MARGIN:-1.2}
MIX=${MIX:-maths=0.241,code=0.225,chat=0.201,if=0.179,science=0.153}
KEEP_RATE=${KEEP_RATE:-maths=0.444,code=0.425,chat=0.903,if=0.449}
TOK_PER_KEPT=${TOK_PER_KEPT:-maths=494,code=510,chat=361,if=315,science=342}
MTOK_PER_GPUH=${MTOK_PER_GPUH:-maths=1.28,code=7.0,chat=3.0,if=1.42}
REPEAT_SCIENCE=${REPEAT_SCIENCE:-}
MERGED=${MERGED:-logs/phase4/train/main/merged}
OUT_LISTS=${OUT_LISTS:-data/c1b/prompt_lists_fresh}
HOLD=${HOLD:-logs/phase4/train/main/holdout_pids.json}
HOLD_A=${HOLD_A:-logs/c1b/trainset/holdout_pids_waveA.txt}
GEN_FLAGS=${GEN_FLAGS:-}
DRAFT_FLAGS=${DRAFT_FLAGS:---max-running 256 --mem-fraction 0.70}
SGL_ENV_FILE=${SGL_ENV_FILE:-logs/phase5/sgl_setup/sgl_env.sh}      # CUDA_HOME / PATH for FlashInfer's JIT on sm_120 (setup_sgl.sh)
FORCE=${FORCE:-0}
export TOKENS MARGIN NGPU MIX KEEP_RATE TOK_PER_KEPT MTOK_PER_GPUH REPEAT_SCIENCE OUT_LISTS
FRESH_DOMS=maths,code,chat,if
STAGES="targets lists gen drafts pack index"
L=logs/c1b/fresh; TJ=logs/c1b/fresh_targets.json; WC=logs/c1b/gen/waveC; TC=logs/c1b/trainset/waveC; FI=logs/c1b/trainset/fresh
cd "$ROOT" || exit 1
mkdir -p $L
say(){ echo "[fresh $(date +%H:%M:%S)] $*" | tee -a $L/run_fresh.log; }
export FRESH_L=$L FRESH_TJ=$TJ FRESH_WC=$WC FRESH_STAGES="$STAGES"

# python helpers: pyrun [args] <<'PY' ... PY  runs the heredoc under $PYT after PYLIB
PYLIB='
import json, os, sys, math, glob
def kv(s): return {a.strip(): float(b) for a, b in (p.split("=") for p in s.split(",") if p.strip())}
'
pyrun(){ { printf '%s\n' "$PYLIB"; cat; } | $PYT - "$@"; }
repeat_science(){   # corpus tokens of repeated science: REPEAT_SCIENCE, else TOKENS * the science share of MIX
  if [ -n "$REPEAT_SCIENCE" ]; then echo "$REPEAT_SCIENCE"; else
    pyrun <<'PY'
print(f"{float(os.environ['TOKENS']) * kv(os.environ['MIX'])['science']:.0f}")
PY
  fi; }
need_targets(){ [ -f $TJ ] || { say "no $TJ: run the targets stage first"; exit 1; }; }

# per-GPU SGLang processes (data generation only): the SGLang env, a 4 GiB FlashInfer workspace, one retry after 30 s
# (SGLang's random internal port may be taken); both stages are resumable, so a retry continues where the run stopped
sgl_run(){   # sgl_run K cmd...
  local k=$1; shift
  ( [ -f "$SGL_ENV_FILE" ] && . "$SGL_ENV_FILE"
    export CUDA_VISIBLE_DEVICES=$k SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-4294967296}
    "$@" --gpu $k --num-gpus $NGPU || { echo "[retry after 30 s]"; sleep 30; "$@" --gpu $k --num-gpus $NGPU; } ); }
per_gpu(){   # per_gpu NAME cmd... : one process per GPU in the background, wait for all; log $L/NAME_gpu<k>.log
  local name=$1; shift; local pids=() k bad=""
  [ -x "$PYS" ] || { say "no $PYS: run code/judge_round/setup_sgl.sh first"; exit 1; }
  for k in $(seq 0 $((NGPU - 1))); do sgl_run $k "$@" > $L/${name}_gpu$k.log 2>&1 & pids+=($!); done
  k=0; for p in "${pids[@]}"; do wait $p || bad="$bad $k"; k=$((k + 1)); done
  [ -z "$bad" ] || { say "$name: failed on GPU(s)$bad (see $L/${name}_gpu*.log)"; exit 1; }; }

# ---------------------------------------------------------------------------
stage_targets(){   # prompts per fresh domain = TOKENS * share / tok_per_kept / keep_rate * MARGIN; GPU-hours = TOKENS * share / Mtok_per_GPU_h
  pyrun "$TJ" <<'PY'
tok, margin, ngpu = float(os.environ["TOKENS"]), float(os.environ["MARGIN"]), int(os.environ["NGPU"])
mix, keep, tpk, rate = (kv(os.environ[k]) for k in ("MIX", "KEEP_RATE", "TOK_PER_KEPT", "MTOK_PER_GPUH"))
rep = float(os.environ["REPEAT_SCIENCE"] or tok * mix["science"])
doms = [d for d in ("maths", "code", "chat", "if") if d in mix]
if abs(sum(mix.values()) - 1) > 0.01: print(f"WARNING: MIX sums to {sum(mix.values()):.3f}, not 1")
out = {"tokens": tok, "margin": margin, "ngpu": ngpu, "mix": mix, "keep_rate": keep, "tok_per_kept": tpk,
       "mtok_per_gpu_h": rate, "repeat_science_tokens": rep, "domains": {}}
print(f"fresh wave C targets: TOKENS={tok / 1e6:.1f}M margin={margin} NGPU={ngpu} (science {rep / 1e6:.2f}M tokens repeated from waves A/B, no generation)")
print(f"{'domain':7s} {'share':>6s} {'fresh Mtok':>10s} {'tok/kept':>8s} {'keep':>6s} {'kept items':>10s} {'PROMPTS':>9s} {'Mtok/GPU-h':>10s} {'GPU-h':>7s} {'GPU-h x margin':>14s}")
tg, tot_h = [], 0.0
for d in doms:
    t = tok * mix[d]; kept = t / tpk[d]; prompts = math.ceil(kept / keep[d] * margin); h = t / 1e6 / rate[d]
    out["domains"][d] = {"share": mix[d], "fresh_tokens": t, "tok_per_kept": tpk[d], "keep_rate": keep[d], "kept_items": math.ceil(kept),
                         "prompts": prompts, "mtok_per_gpu_h": rate[d], "gpu_hours": h, "gpu_hours_with_margin": h * margin}
    tg.append(f"{d}={prompts}"); tot_h += h
    print(f"{d:7s} {mix[d]:6.3f} {t / 1e6:10.2f} {tpk[d]:8.0f} {keep[d]:6.3f} {math.ceil(kept):10d} {prompts:9d} {rate[d]:10.2f} {h:7.2f} {h * margin:14.2f}")
ft = sum(v["fresh_tokens"] for v in out["domains"].values())
print(f"{'fresh':7s} {sum(mix[d] for d in doms):6.3f} {ft / 1e6:10.2f} {'':8s} {'':6s} {sum(v['kept_items'] for v in out['domains'].values()):10d} {sum(v['prompts'] for v in out['domains'].values()):9d} {'':10s} {tot_h:7.2f} {tot_h * margin:14.2f}")
print(f"projected generation: {tot_h:.1f} GPU-hours = {tot_h / ngpu:.2f} h on {ngpu} GPUs ({tot_h * margin / ngpu:.2f} h if every margin prompt is decoded); drafts and packing come on top")
out["targets"] = ",".join(tg); out["gpu_hours"] = tot_h; out["hours_on_ngpu"] = tot_h / ngpu
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"--targets \"{out['targets']}\"  ->  {sys.argv[1]}")
PY
}

stage_lists(){
  need_targets
  if [ "${LISTS_AFTER_GEN:-0}" != 1 ] && ls $WC/gpu*/done.txt >/dev/null 2>&1 && [ -n "$(cat $WC/gpu*/done.txt 2>/dev/null | head -c 1)" ]; then
    say "wave C has already started decoding ($WC/gpu*/done.txt): rebuilding the lists would drop those prompts; LISTS_AFTER_GEN=1 to override"; exit 1; fi
  T=$($PYT -c "import json; print(json.load(open('$TJ'))['targets'])") || exit 1
  say "fresh_lists.py --targets \"$T\" -> $OUT_LISTS"
  $PYT code/corpus1b/fresh_lists.py --lists data/c1b/prompt_lists --done 'logs/c1b/gen/wave[AB]/gpu*/done.txt' --out $OUT_LISTS \
    --targets "$T" 2>&1 | tee -a $L/lists.log
  [ ${PIPESTATUS[0]} = 0 ] || { say "fresh_lists.py failed (see $L/lists.log)"; exit 1; }
  pyrun "$TJ" "$OUT_LISTS" <<'PY' || exit 1    # how many prompts did each domain get? IF is the scarce one: > 20% short is an error
tg = json.load(open(sys.argv[1]))["domains"]; out = sys.argv[2]
def count(d):
    for ext in (".jsonl", ".jsonl.gz"):
        p = os.path.join(out, d + ext)
        if os.path.exists(p):
            import gzip
            op = gzip.open if ext.endswith("gz") else open
            with op(p, "rt") as fh: return sum(1 for l in fh if l.strip())
    return None
def manifest_short(node, d, want):    # a shortfall the manifest reports for domain d: a fraction (float in [0, 1]) or a count
    if isinstance(node, dict):
        if isinstance(node.get(d), dict):
            for k, v in node[d].items():
                if "short" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                    if "pct" in k.lower() or "percent" in k.lower(): return v / 100
                    return v if (isinstance(v, float) and v <= 1) else v / want
        for v in node.values():
            r = manifest_short(v, d, want)
            if r is not None: return r
    return None
try: man = json.load(open(os.path.join(out, "manifest.json")))
except Exception as e: man = None; print(f"WARNING: no readable manifest.json in {out}: {e}")
bad = []
print(f"{'domain':7s} {'wanted':>8s} {'in list':>8s} {'manifest shortfall':>18s} {'shortfall':>9s}")
for d, v in tg.items():
    want, got = v["prompts"], count(d)
    ms = manifest_short(man, d, want) if man is not None else None
    fs = max(0.0, 1 - got / want) if got is not None else None           # the list file and the manifest: the worse one counts
    sf = max([x for x in (fs, ms) if x is not None], default=None)
    print(f"{d:7s} {want:8d} {'?' if got is None else got:>8} {'-' if ms is None else f'{ms:.1%}':>18s} {'?' if sf is None else f'{sf:.1%}':>9s}")
    if sf is None and d == "if": bad.append("IF shortfall cannot be read (no if list, no shortfall in the manifest)")
    elif sf is not None and sf > 0.20 and d == "if": bad.append(f"IF shortfall {sf:.1%} > 20%")
    elif sf is not None and sf > 0.20: print(f"WARNING: {d} is {sf:.1%} short of its target")
if bad:
    print("ERROR: " + "; ".join(bad) + f" -- see {out}/manifest.json; IF prompts are the scarce ones: lower the IF share in MIX (or TOKENS), or build more IF prompts"); sys.exit(1)
PY
}

stage_gen(){
  [ -f $OUT_LISTS/manifest.json ] || { say "no $OUT_LISTS/manifest.json: run the lists stage first"; exit 1; }
  mkdir -p $WC
  say "gen: wave C on $NGPU GPUs (SGLang), domains $FRESH_DOMS, lists $OUT_LISTS"
  per_gpu gen $PYS code/corpus1b/gen_sglang.py --wave C --prompt-dir $OUT_LISTS --extra-done 'logs/c1b/gen/wave[AB]/gpu*/done.txt' \
    --domains $FRESH_DOMS $GEN_FLAGS
  gen_counts
}
gen_counts(){   # kept answers per domain from the stats.jsonl of every GPU. A line carries the CUMULATIVE per-domain counts of one
                # process; a restart (elapsed_h or a count going down) starts a new base, so deltas are summed.
  pyrun "$WC" "$TJ" <<'PY'
tg = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else None
tot = {}
files = sorted(glob.glob(os.path.join(sys.argv[1], "gpu*", "stats.jsonl")))
for f in files:
    prev, pel = {}, -1.0
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l); cur = r.get("by_domain", {}); el = r.get("elapsed_h", 0.0)
        reset = el <= pel or any(cur.get(d, {}).get(k, 0) < prev.get(d, {}).get(k, 0) for d in prev for k in ("prompts", "kept"))
        for d, c in cur.items():
            for k in ("prompts", "kept", "gen_tokens"):
                tot.setdefault(d, {}).setdefault(k, 0)
                tot[d][k] += c.get(k, 0) - (0 if reset else prev.get(d, {}).get(k, 0))
        prev, pel = cur, el
print(f"gen stats from {len(files)} stats.jsonl files in {sys.argv[1]}")
print(f"{'domain':7s} {'decoded':>8s} {'kept':>8s} {'keep rate':>9s} {'gen Mtok':>9s} {'est. corpus Mtok':>16s} {'wanted Mtok':>11s}")
for d in sorted(tot):
    v = tot[d]; est = v["kept"] * tg["tok_per_kept"][d] / 1e6 if tg and d in tg["tok_per_kept"] else float("nan")
    want = tg["domains"][d]["fresh_tokens"] / 1e6 if tg and d in tg["domains"] else float("nan")
    print(f"{d:7s} {v['prompts']:8d} {v['kept']:8d} {v['kept'] / max(1, v['prompts']):9.3f} {v['gen_tokens'] / 1e6:9.2f} {est:16.2f} {want:11.2f}")
if not tot: print("no stats yet"); sys.exit(1)
PY
}

stage_drafts(){
  [ -d $MERGED ] || { say "no drafter at $MERGED"; exit 1; }
  ls $WC/gpu*/records.*.jsonl.gz >/dev/null 2>&1 || { say "no wave-C records under $WC: run the gen stage first"; exit 1; }
  say "drafts: wave C with $MERGED on $NGPU GPUs (SGLang)"
  per_gpu draft $PYS code/corpus1b/draft_sglang.py --wave C --drafter-dir $MERGED $DRAFT_FLAGS
  say "drafts: all $NGPU processes finished; the pack stage fails if a kept record has no drafts"
}

stage_pack(){   # kept answers + drafts -> training items; every kept record needs its drafts
  $PYT code/corpus1b/to_trainset.py --wave C 2>&1 | tee -a $L/pack.log
  [ ${PIPESTATUS[0]} = 0 ] || { say "to_trainset.py failed (see $L/pack.log)"; exit 1; }
  pyrun $TC/build_summary.json <<'PY' || exit 1
c = json.load(open(sys.argv[1]))["counts"]; print("pack:", c)
if c.get("no_draft", 0) > 0: print(f"ERROR: {c['no_draft']} kept records have no drafts: rerun the drafts stage"); sys.exit(1)
if c.get("written", 0) < 0.95 * c.get("kept_records", 0): print("ERROR: fewer than 95% of the kept records became items"); sys.exit(1)
PY
}

stage_index(){   # wave C (all rows, fresh) + science repeated from A/B up to REPEAT_SCIENCE corpus tokens, minus the holdouts
  local f rs; rs=$(repeat_science) || exit 1
  for f in $HOLD $HOLD_A $TC/index.jsonl logs/c1b/trainset/waveA/index.jsonl logs/c1b/trainset/waveB/index.jsonl; do
    [ -f $f ] || { say "missing $f"; exit 1; }; done
  say "index: fresh wave C + science=$rs repeated from waves A/B, excluding $HOLD and $HOLD_A"
  $PYT code/judge_round/build_fresh_index.py --fresh $TC/index.jsonl \
    --repeat-index logs/c1b/trainset/waveA/index.jsonl --repeat-index logs/c1b/trainset/waveB/index.jsonl \
    --repeat "science=$rs" --exclude-pids $HOLD --exclude-pids $HOLD_A --out $FI/index.jsonl --strict 2>&1 | tee -a $L/index.log
  [ ${PIPESTATUS[0]} = 0 ] || { say "build_fresh_index.py failed or the science budget was not met (see $L/index.log)"; exit 1; }
}

stage_report(){
  [ -f $FI/build_summary.json ] && $PYT - $FI/build_summary.json <<'PY'
import json, sys
sys.path.insert(0, "code/judge_round")
import build_fresh_index as B
print(B.table(json.load(open(sys.argv[1]))))
PY
  pyrun <<'PY'
import datetime
print("\nstage timing (markers in " + os.environ["FRESH_L"] + "):")
tot = 0.0; el = {}
for s in os.environ["FRESH_STAGES"].split():
    p = os.path.join(os.environ["FRESH_L"], ".done_" + s)
    if not os.path.exists(p): print(f"  {s:8s} not done"); continue
    m = dict(l.strip().split("=", 1) for l in open(p) if "=" in l)
    t0, t1 = float(m["start"]), float(m["end"]); el[s] = t1 - t0; tot += t1 - t0
    print(f"  {s:8s} {datetime.datetime.fromtimestamp(t0):%m-%d %H:%M} -> {datetime.datetime.fromtimestamp(t1):%H:%M}  {(t1 - t0) / 3600:7.2f} h")
print(f"  {'total':8s} {tot / 3600:35.2f} h")
tj = os.environ["FRESH_TJ"]
if os.path.exists(tj) and "gen" in el:
    t = json.load(open(tj)); print(f"  gen: projected {t['hours_on_ngpu']:.2f} h on {t['ngpu']} GPUs (x{t['margin']} with margin {t['hours_on_ngpu'] * t['margin']:.2f} h), measured {el['gen'] / 3600:.2f} h")
PY
}

# ---------------------------------------------------------------------------
run_stage(){   # run_stage NAME: skip a finished stage (FORCE=1 reruns), write the marker with start / end epoch seconds
  local s=$1 t0
  if [ -f $L/.done_$s ] && [ "$FORCE" != 1 ]; then say "$s: done ($(grep '^at=' $L/.done_$s | cut -d= -f2)); FORCE=1 reruns"; return 0; fi
  rm -f $L/.done_$s; t0=$(date +%s); say "$s: start"
  stage_$s || { say "$s: FAILED"; exit 1; }
  printf 'start=%s\nend=%s\nat=%s\n' $t0 $(date +%s) "$(date '+%Y-%m-%d %H:%M:%S')" > $L/.done_$s
  say "$s: done in $(( $(date +%s) - t0 )) s"
}
case ${1:-help} in
targets|lists|gen|drafts|pack|index) run_stage $1 ;;
report) stage_report ;;
all)    for s in $STAGES; do run_stage $s; done; stage_report ;;
*) echo "usage: $0 targets|lists|gen|drafts|pack|index|report|all   (see the header for the env knobs)"; exit 2 ;;
esac
