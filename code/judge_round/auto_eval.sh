#!/usr/bin/env bash
# Phase 5 auto-pipeline after epoch-2 training (re-armed 03:05 IST, 24 Oct). Training self-stops via --stop-at 04:05 IST
# and writes rec_main_ep2/final. This watcher: (1) waits for train_c1b_rec to vanish OR the 04:12 IST hard deadline, at
# which point it kills any survivor BY PID; (2) picks the NEWEST COMPLETE checkpoint (adapter.safetensors+trainer_state.pt),
# preferring rec_main_ep2/final, then rec_main_ep2/tok_*, then rec_main/final, then rec_main/tok_*; (3) launches the rd1
# eval (8 GPUs, plain harness) with that adapter, then the vanilla multiplier check. Eval must START by 04:15 IST.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
HARDKILL=1792795320   # 04:12 IST (2026-10-23 22:42 UTC): stop training by PID if --stop-at (04:05) somehow didn't
log(){ echo "[auto-eval $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
ntrain(){ ps -eo pid,args --no-headers | grep -E 'train_c1b_rec' | grep -v grep | wc -l; }

# 1. wait for training to stop, but no later than the 04:12 hard deadline
while [ "$(ntrain)" -gt 0 ] && [ "$(date -u +%s)" -lt "$HARDKILL" ]; do sleep 15; done
if [ "$(ntrain)" -gt 0 ]; then
  log "training still running at the 04:12 hard deadline -> stopping by PID"
  for p in $(ps -eo pid,args --no-headers | grep -E 'train_c1b_rec|torchrun' | grep -v grep | awk '{print $1}'); do kill -TERM "$p" 2>/dev/null; done
  sleep 20
  for p in $(ps -eo pid,args --no-headers | grep -E 'train_c1b_rec|torchrun' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
fi
tmux kill-session -t train 2>/dev/null
sleep 8   # let the final save + GPU memory settle

# 2. newest COMPLETE checkpoint (both files present), by trainer_state.pt mtime
ADAPTER=$(/venv/p2/bin/python - <<'PY'
import glob, os
ROOT="/workspace/diffusion-moe-general-expert"
cands = (glob.glob(ROOT+"/logs/phase5/judge/rec_main_ep2/final")
       + glob.glob(ROOT+"/logs/phase5/judge/rec_main_ep2/tok_*")
       + glob.glob(ROOT+"/logs/phase5/judge/rec_main/final")
       + glob.glob(ROOT+"/logs/phase5/judge/rec_main/tok_*"))
ok = [d for d in cands if os.path.isfile(d+"/adapter.safetensors") and os.path.isfile(d+"/trainer_state.pt")]
ok.sort(key=lambda d: os.path.getmtime(d+"/trainer_state.pt"))
print(ok[-1] if ok else "")
PY
)
export ADAPTER
TOK=$([ -f "$ADAPTER/trainer_state.json" ] && /venv/p2/bin/python -c "import json;print('%.2fM'%(json.load(open('$ADAPTER/trainer_state.json'))['tokens']/1e6))" 2>/dev/null || echo '?')
log "newest complete checkpoint = $ADAPTER ($TOK tokens)"
[ -n "$ADAPTER" ] || { log "NO COMPLETE CHECKPOINT FOUND — aborting eval"; exit 1; }

# safety: free any lingering train procs by PID before eval grabs the GPUs
for p in $(ps -eo pid,args --no-headers | grep -E 'train_c1b_rec|torchrun' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 3

# 3. eval
log "launching rd1 eval (7 sets, 8 GPUs) with ADAPTER=$ADAPTER"
tmux new -d -s eval "ADAPTER='$ADAPTER' bash code/judge_round/eval_rd1.sh 2>&1 | tee -a logs/phase5/eval/eval.log"
while tmux has-session -t eval 2>/dev/null; do sleep 30; done
log "eval finished"

# 4. vanilla multiplier check
log "launching vanilla multiplier check"
tmux new -d -s vanilla 'bash code/judge_round/vanilla_check.sh 2>&1 | tee -a logs/phase5/vanilla/vanilla.log'
while tmux has-session -t vanilla 2>/dev/null; do sleep 20; done
log "VANILLA DONE — pipeline complete; ready for final backups + STATUS"
