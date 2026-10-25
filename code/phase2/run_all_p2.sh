#!/usr/bin/env bash
# Phase 2 MVP for one rental, in one command (run it in tmux): F2 (corpus) + S1 (size) on all seven sets, zero-shot,
# fully logged. F1 is the family fallback: it runs in F2's place only if F2 fails its fit check.
#
#   bash code/phase2/run_all_p2.sh 2>&1 | tee -a logs/phase2/run_all.log
#
#  1. setup_p2.sh CPU/network steps (venv, models, drafters F2 S1 F1, references, IFEval-541 and its stock run from
#     Phase 1) -- safe while other jobs hold GPUs
#  2. the HF backup loop (backup_p2.py --loop 30) in the background
#  3. waits until EVERY GPU is free (< 1000 MiB for 3 checks in a row, 60 s apart): it never touches other jobs
#  4. setup_p2.sh go/no-go GPU checks (tests, adapter hash, stock 16/16, re-draft smoke test); any failure stops here
#  5. the plan, in this order (nothing is skipped; each step resumes where it stopped):
#       F2 re-draft            step 3 for F2 first: its "does it fit" answer in ~20 min, not after hours
#       S0 on IFEval-541       the S0 judge + DES48 reference round 3 does not have
#       F2                     probe -> decode (7 sets) -> report -> gate.  If F2's re-draft was stopped by a rule
#                              (does not fit), F1 runs here instead: re-draft -> probe -> decode -> report -> gate
#       S1                     re-draft -> probe -> decode (7 sets) -> report -> gate
#     The stock speedup reference is round 3's records (six sets) and Phase 1's stock run (IFEval-541): no stock decode.
#     Every decode logs, per item: every forward's experts per layer (bitmask) and every row's top-8 expert ids and
#     router probabilities (P2_EXPERT_LOG=full), the judge's probabilities at every pass, the drafter's confidences,
#     and the running metrics (experts per stage, accuracy, delta vs S0, speedup) -- see docs/PHASE2_DRAFTER_SWAP.md.
#  6. STATUS.md with every gate and report, final plots (plot_p2.py; also every 30 min in the background), and a
#     final backup.
# P2_DEADLINE_IST ("YYYY-MM-DD HH:MM", optional): at that time logs/phase2/STOP is created (running decodes stop after
#   their current item, records kept) and a final backup runs. Nothing is skipped in advance; if the plan's estimate
#   exceeds the time left, it is only WARNED in the log.
# Env passed through: GPUS, NSH, P2_HF_REPO, P2_MODELS_FROM, P2_C1B_EVAL_FROM, HF_HOME, P2_EXPERT_LOG.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=/venv/p2/bin/python
mkdir -p logs/phase2
say(){ echo "[run_all $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
if [ -n "${P2_DEADLINE_IST:-}" ]; then
  P2_DEADLINE=$(TZ=Asia/Kolkata date -d "$P2_DEADLINE_IST" +%s) || { say "bad P2_DEADLINE_IST"; exit 1; }
fi
export P2_DEADLINE=${P2_DEADLINE:-0} P2_NO_RETRAIN=1 P2_EXPERT_LOG=${P2_EXPERT_LOG:-full} P2_DRAFTERS=${P2_DRAFTERS:-"F2 S1 F1"}
[ "$P2_DEADLINE" != 0 ] && say "deadline $(TZ=Asia/Kolkata date -d @$P2_DEADLINE '+%Y-%m-%d %H:%M IST')"
[ -e logs/phase2/STOP ] && { say "logs/phase2/STOP exists (a previous deadline?): remove it to run"; exit 3; }

# the plan: label | hours (estimate) | command.  Estimates: round-3 seconds per item x items / 8 GPUs, x1.30 (F2) /
# x1.20 (S1) / x1.15 (F1) for a weaker or uncached drafter, +3 min per set for loading; +5% for the full expert log.
# The family step picks its arm when it starts: F2, or F1 if F2's re-draft was stopped by a rule (does not fit).
FAMILY='if [ -f logs/phase2/F2/.stopped_redraft ]; then echo "F2 does not fit (logs/phase2/redraft/F2/STOPPED.json): F1 runs in its place"; bash code/phase2/run_phase2.sh F1; else bash code/phase2/run_phase2.sh F2; fi'
PLAN=(
  "F2 re-draft (does it fit?)|0.35|bash code/phase2/run_phase2.sh F2 redraft"
  "S0 on IFEval-541|0.70|P2_SETS=ifeval541 bash code/phase2/run_phase2.sh S0 decode report"
  "family: F2, 7 sets (F1 if F2 does not fit)|3.65|$FAMILY"
  "size: S1, 7 sets|3.47|bash code/phase2/run_phase2.sh S1"
)
hsum(){ python3 -c "import sys; print(round(sum(float(x) for x in sys.argv[1:]), 2))" "$@"; }
total=$(hsum $(for p in "${PLAN[@]}"; do echo "$p" | cut -d'|' -f2; done))
say "plan: ${#PLAN[@]} steps, ~${total} h after the GPU checks (+/- 20%)"

# 1. CPU / network setup
SKIP_CHECKS=1 bash code/phase2/setup_p2.sh || { say "setup (CPU steps) FAILED: logs/phase2/setup/"; exit 1; }

# 2. backup loop
if ! pgrep -f "backup_p2[.]py --loop" > /dev/null; then
  nohup $PY code/phase2/backup_p2.py --loop 30 >> logs/phase2/hf_backup.log 2>&1 &
  say "HF backup loop started (logs/phase2/hf_backup.log)"
fi
if ! pgrep -f "backup_githu[b].sh --loop" > /dev/null; then   # text logs + plots to the phase2-logs branch
  nohup bash code/phase2/backup_github.sh --loop 15 >> logs/phase2/gh_backup.log 2>&1 &
  say "GitHub log backup loop started (logs/phase2/gh_backup.log)"
fi
if ! pgrep -f "plot_p2[.]py --loop" > /dev/null; then          # CPU only: plots of every log so far, PNG + W&B
  CUDA_VISIBLE_DEVICES= nohup $PY code/phase2/plot_p2.py --loop 30 >> logs/phase2/plots.log 2>&1 &
  say "plot loop started (logs/phase2/plots.log, logs/phase2/plots/)"
fi

# deadline watcher: STOP + final backup
if [ "$P2_DEADLINE" != 0 ]; then
  ( while [ "$(date +%s)" -lt "$P2_DEADLINE" ]; do sleep 30; done
    touch logs/phase2/STOP
    echo "[run_all] DEADLINE: logs/phase2/STOP created; final backup" >> logs/phase2/run_all.log
    sleep 120; $PY code/phase2/backup_p2.py >> logs/phase2/hf_backup.log 2>&1 ) &
fi

# 3. wait for every GPU to be free
IFS=, read -ra G <<< "${GPUS:-0,1,2,3,4,5,6,7}"
ok=0; waited=0
while [ $ok -lt 3 ]; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g=" ${G[*]} " 'index(g, " "$1" ") && $2 >= 1000' | wc -l)
  if [ "$busy" = 0 ]; then ok=$((ok + 1)); else
    [ $((waited % 10)) = 0 ] && say "waiting for the GPUs: $busy busy (other jobs are never touched)"; ok=0; fi
  [ $ok -lt 3 ] && { sleep 60; waited=$((waited + 1)); }
  [ -e logs/phase2/STOP ] && { say "STOP while waiting"; exit 3; }
done
say "all GPUs free"

# 4. GPU checks
bash code/phase2/setup_p2.sh || { say "setup GO/NO-GO CHECKS FAILED: logs/phase2/setup/ -- nothing else runs"; exit 2; }

# 5. the plan
declare -a RC
left_h(){ python3 -c "import time; print(round(($P2_DEADLINE - time.time()) / 3600, 2))"; }
for i in "${!PLAN[@]}"; do
  IFS='|' read -r label est cmd <<< "${PLAN[$i]}"
  [ -e logs/phase2/STOP ] && { RC[$i]="not started (STOP)"; say "not started: $label (STOP)"; continue; }
  if [ "$P2_DEADLINE" != 0 ]; then
    rest=$(hsum $(for j in $(seq $i $((${#PLAN[@]} - 1))); do echo "${PLAN[$j]}" | cut -d'|' -f2; done))
    lh=$(left_h)
    python3 -c "import sys; sys.exit(0 if $rest > $lh else 1)" \
      && say "WARNING: the remaining plan needs ~$rest h, $lh h left before the deadline (running anyway; the deadline stops it cleanly)"
  fi
  say "step $((i + 1))/${#PLAN[@]}: $label (estimate ~$est h)"
  bash -c "$cmd"; rc=$?
  RC[$i]=$rc
  say "step $((i + 1)): $label -> exit $rc"
  # F2 did not fit (stop rule at its re-draft): that is a result; the family step then runs F1 instead
done

# 6. summary + backup
{
  echo "# Phase 2 status ($(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST'))"; echo
  echo "| step | exit |"; echo "|---|---|"
  for i in "${!PLAN[@]}"; do echo "| $(echo "${PLAN[$i]}" | cut -d'|' -f1) | ${RC[$i]:-?} |"; done
  echo; echo "exit 0 = done, 3 = stopped by a rule (see logs/phase2/<arm>/pipeline.log and STOPPED.json), 1 = crashed"
  [ -f logs/phase2/F2/.stopped_redraft ] && { echo; echo "F2 did not fit (logs/phase2/redraft/F2/STOPPED.json): F1 ran in its place"; }
  for a in F2 F1 S1; do g=logs/phase2/gate/${a}_zeroshot/gate.md; [ -f $g ] && { echo; cat $g; }; done
  for a in F2 F1 S1 S0; do r=logs/phase2/report/$a/experts.md; [ -f $r ] && { echo; cat $r; }; done
} > logs/phase2/STATUS.md
say "summary: logs/phase2/STATUS.md"
CUDA_VISIBLE_DEVICES= $PY code/phase2/plot_p2.py --once >> logs/phase2/plots.log 2>&1 && say "final plots done" || say "final plots FAILED (logs/phase2/plots.log)"
$PY code/phase2/backup_p2.py >> logs/phase2/hf_backup.log 2>&1 && say "final backup done" || say "final backup FAILED (logs/phase2/hf_backup.log)"
