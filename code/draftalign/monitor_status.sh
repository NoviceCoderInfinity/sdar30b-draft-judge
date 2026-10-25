#!/bin/bash
# Read-only status snapshot of the draft-aligned round-2b pipeline.
# Prints a compact block for every phase of code/draftalign/orchestrate_r2b_v4.sh.
# Touches nothing: no kills, no launches, no writes outside stdout.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=logs/draftalign/datagen
T2=logs/draftalign/train/t2b_union_flag
DR=logs/draftalign/train_drafter/d1

echo "TIME $(date -u '+%F %T')Z / $(TZ=Asia/Kolkata date '+%F %T') IST"

echo "-- PROCS"
# snapshot ps ONCE, then match against the snapshot: this way neither grep itself nor a Claude
# tool shell whose command line contains the pattern can ever be counted (the hazard the project notes warn about)
PSOUT="$(ps -eo pid,args --no-headers | grep -v shell-snapshots)"
cnt(){ printf '%s\n' "$PSOUT" | grep -c -- "$1"; }
pids(){ printf '%s\n' "$PSOUT" | grep -- "$1" | awk '{print $1}' | tr '\n' ','; }
echo "orchestrator_v4_pids=$(pids 'orchestrate_r2b_v[45].sh')  queue_runner=$(cnt 'gpu_queue.sh')  datagen_workers=$(cnt 'run_da_decode.py --model')  trainers=$(cnt 'train.py --index')  drafter_trainer=$(cnt 'train_drafter.py')"
echo "queue.txt=$([ -f logs/draftalign/queue.txt ] && wc -l < logs/draftalign/queue.txt || echo ABSENT) queue.paused=$([ -f logs/draftalign/queue.paused ] && wc -l < logs/draftalign/queue.paused || echo ABSENT)"

echo "-- DATAGEN (need 8 / 8 / 2 .done)"
for d in gsm8k_pool_3000_7273_dv-s8_blind_q50 math_pool_0_7500_3_dv-s8_blind_q50 mbpp_pool_dv-s8_blind_q50 gsm8k_pool_0_3000_dv-s8_blind_q50; do
  n=$(ls $D/$d/*.done 2>/dev/null | wc -l)
  echo "[$d] done=$n"
  for f in $D/$d/shard*of*.log; do
    [ -e "$f" ] || continue
    grep -q "SHARD DONE" "$f" && continue
    tail -qn1 "$f" | sed "s|^|   $(basename $f): |"
  done
done

echo "-- CRASHES (datagen shard logs; queue-cell/trainer logs written in the last 40 min)"
found=0
for f in $D/*/shard*of*.log; do
  [ -e "$f" ] || continue
  if grep -qE "Traceback|CUDA out of memory|Killed" "$f"; then echo "   $f: $(grep -nE 'Traceback|CUDA out of memory|Killed' "$f" | tail -1)"; found=1; fi
done
# queue cells and trainer logs: only files written in the last 40 min, so historical tracebacks do not re-alert forever
for f in $(find logs/draftalign/da_decode logs/draftalign/train logs/draftalign/train_drafter -name '*.log' ! -name '*.crash*' -mmin -40 2>/dev/null); do
  if grep -qE "Traceback|CUDA out of memory|Killed|AssertionError" "$f"; then echo "   $f: $(grep -E 'Traceback|CUDA out of memory|Killed|AssertionError' "$f" | tail -1 | cut -c1-200)"; found=1; fi
done
[ $found -eq 0 ] && echo "   none"

echo "-- ORCHESTRATOR LOG (last 4)"
tail -4 logs/draftalign/orchestrate_round1.log 2>/dev/null | sed 's/^/   /'

echo "-- TRAINING"
if [ -d "$T2" ]; then
  echo "T2b ckpts: $(ls -d $T2/step* 2>/dev/null | wc -l) latest=$(ls -d $T2/step* 2>/dev/null | sort | tail -1)"
  tail -3 logs/draftalign/train/t2b_union_flag.log 2>/dev/null | sed 's/^/   /'
else echo "T2b: not started"; fi
if [ -d "$DR" ] || [ -f logs/draftalign/train_drafter/d1.log ]; then
  echo "drafter ckpts: $(ls -d $DR/step* 2>/dev/null | wc -l) latest=$(ls -d $DR/step* 2>/dev/null | sort | tail -1)"
  tail -3 logs/draftalign/train_drafter/d1.log 2>/dev/null | sed 's/^/   /'
else echo "drafter: not started"; fi
for j in results/metrics/draftalign/drafter_agreement_d1.json results/metrics/draftalign/drafter_accuracy_stock.json results/metrics/draftalign/drafter_accuracy_d1.json; do
  [ -f "$j" ] && echo "   RESULT $j: $(tr -d '\n' < $j | cut -c1-300)"
done

echo "-- T2b TREND (held-out every 100 steps; train loss in 200-step windows; stop rule in the cron prompt)"
[ -f "$T2/train.rank0.jsonl" ] && /venv/main/bin/python - "$T2/train.rank0.jsonl" <<'PY' 2>/dev/null
import json, sys, statistics
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
_d={}
for r in rows: _d[(r.get('event'), r['step'])]=r   # a resume re-logs steps: keep the last row per (event, step)
rows=sorted(_d.values(), key=lambda r: (r['step'], r.get('event') or ''))
tr=[r for r in rows if r.get('event')!='eval' and r['step']>1060]     # skip the x0.3 partial row right after the resume
ev=[r for r in rows if r.get('event')=='eval' and r['step']>1057]
def m(rs,k): return statistics.mean(r[k] for r in rs)
print("   held-out (64 states): " + "  ".join(f"s{r['step']}: acc {r['draft_pass1_acc']:.3f}/prec {r['draft_pass1_conf_precision']:.3f}" for r in ev[-6:]) + "   [T2a end s1000: acc 0.579/prec 0.961; copy-draft 0.556]")
if len(ev)>=6:
    a=m(ev[-3:],'draft_pass1_acc'); b=m(ev[-6:-3],'draft_pass1_acc'); print(f"   3-eval mean acc: last {a:.4f} vs previous {b:.4f} (delta {a-b:+.4f}); min precision last 3: {min(r['draft_pass1_conf_precision'] for r in ev[-3:]):.3f}")
wins=[tr[i:i+20] for i in range(0,len(tr),20) if len(tr[i:i+20])==20]
if wins:
    print("   train loss 200-step windows: " + "  ".join(f"s{w[0]['step']}-{w[-1]['step']}: {m(w,'loss'):.3f}" for w in wins[-4:]) + f"   [T2a last 200: 1.311]")
    if len(wins)>=2: print(f"   last window vs previous: {m(wins[-1],'loss')-m(wins[-2],'loss'):+.3f} loss, {m(wins[-1],'ce')-m(wins[-2],'ce'):+.3f} ce")
_n=max(tr, key=lambda r: r.get('utc',''))   # newest by wall clock (after a resume the max step can be the pre-crash row)
print(f"   last row: step {_n['step']}/3057 loss {_n['loss']:.3f} ce {_n['ce']:.3f} acc {_n['acc']:.3f} lr {_n['lr']:.2e} utc {_n['utc']}")
PY
echo "-- T2b EVALS"
ls logs/draftalign/train/eval_t2b_*.log >/dev/null 2>&1 && for f in logs/draftalign/train/eval_t2b_*.log; do echo "   $(basename $f): $(tail -qn1 $f | cut -c1-140)"; done || echo "   not started"

echo "-- GPU / DISK"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' | '; echo
df -h /workspace | tail -1
