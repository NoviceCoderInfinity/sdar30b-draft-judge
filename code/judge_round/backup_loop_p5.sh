#!/usr/bin/env bash
# Phase 5 box session 1: every INTERVAL minutes (default 15), niced, (a) push code + STATUS.md + results/ to the branch
# phase5-box-s1 (explicit `git add` paths, staged diff secret-scanned first), (b) upload the phase 5 outputs to the public
# HF dataset Arushhh/phase5-loop-guard (backup_p5.py, allowlist + secret scan).  Every pass writes start/end lines to
# logs/phase5/events.tsv.  A secret-scan hit (exit 2) STOPS the loop and leaves logs/phase5/BACKUP_STOPPED.
#   tmux new -d -s s1_backup 'bash code/judge_round/backup_loop_p5.sh 2>&1 | tee -a logs/phase5/backup_loop.log'
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}; cd "$ROOT" || exit 1
INTERVAL=${INTERVAL:-15}; BRANCH=${BRANCH:-phase5-box-s1}
set -a; . /workspace/.env; set +a
PY=${PY:-/venv/p2/bin/python}; "$PY" -c "import huggingface_hub" 2>/dev/null || PY=/workspace/dl/venv/bin/python
EV=logs/phase5/events.tsv; mkdir -p logs/phase5
ev(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> $EV; }
say(){ echo "[backup-loop $(date -u +%FT%TZ)] $*"; }
AUTH="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"
[ "$BRANCH" = phase5-loop-guard ] && { say "refusing to push to phase5-loop-guard"; exit 1; }
while true; do
  # (a) git
  ev git_push start
  git add -- code STATUS.md results 2>/dev/null
  if nice -n 19 ionice -c3 $PY code/judge_round/backup_p5.py --scan-staged; then
    if ! git diff --cached --quiet; then
      git -c user.name=Arush777 -c user.email=23je0154@iitism.ac.in commit -q -m "phase 5 box s1: status/results $(date -u +%FT%TZ)" && say "committed"
    fi
    git -c http.extraHeader="$AUTH" push -q origin HEAD:refs/heads/$BRANCH 2>&1 | grep -v -i -E 'token|authorization'
    rc=${PIPESTATUS[0]}; ev git_push "end rc=$rc"; say "push rc=$rc"   # PIPESTATUS[0]=git (the `if … then :` form reset it to 0, masking rejects)
  else
    git reset -q; ev git_push "STOP secret scan hit"; say "SECRET SCAN HIT on staged diff: loop stopped"; touch logs/phase5/BACKUP_STOPPED; exit 2
  fi
  # (b) HF
  ev hf_upload start
  nice -n 19 ionice -c3 $PY code/judge_round/backup_p5.py --tag loop; rc=$?
  ev hf_upload "end rc=$rc"
  if [ $rc = 2 ]; then say "SECRET SCAN HIT in the upload stage: loop stopped"; touch logs/phase5/BACKUP_STOPPED; exit 2; fi
  # (c) the logs branch phase5-logs (operator 23 Oct 15:15 IST): small text logs only, own worktree, same secret scan
  nice -n 19 ionice -c3 bash code/judge_round/push_logs_p5.sh; rc=$?
  if [ $rc = 2 ]; then say "SECRET SCAN HIT in the logs-branch stage: loop stopped"; touch logs/phase5/BACKUP_STOPPED; exit 2; fi
  say "sleeping $INTERVAL min"; sleep $((INTERVAL * 60))
done
