#!/usr/bin/env bash
# Phase 2 log backup to GitHub: the text logs (records, running metrics, stop events, summaries, reports, job logs)
# and the plots go to the orphan branch `phase2-logs` of this repository, as ONE commit that is replaced on every
# sync (force-push), so the branch never accumulates history. The large npz arrays (experts/, judge/, traces/) and the
# downloaded references (logs/phase2/ref/) are not pushed: they go to Hugging Face (backup_p2.py).
#   bash code/phase2/backup_github.sh            # one sync
#   bash code/phase2/backup_github.sh --loop 15  # every 15 min, until logs/phase2/STOP exists
# Before every push the staged files are scanned for the .env values of HF_TOKEN / WANDB_API_KEY / GITHUB_TOKEN and
# for hf_ token patterns; a hit aborts the sync.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WT=${P2_GH_WORKTREE:-$ROOT/../dmge_p2_logbackup}
BR=phase2-logs
say(){ echo "[gh-backup $(date -u +%H:%M:%S) UTC] $*"; }

sync_once(){
  if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
    git -C "$ROOT" worktree add --detach "$WT" > /dev/null 2>&1 || { say "worktree add failed"; return 1; }
    ( cd "$WT" && git checkout -q --orphan $BR && git rm -rfq . ) || { say "orphan branch setup failed"; return 1; }
  fi
  mkdir -p "$WT/logs/phase2"
  rsync -a --delete --prune-empty-dirs \
    --exclude 'ref/' --exclude 'experts/' --exclude 'judge/' --exclude 'traces/' --exclude 'items/' \
    --include '*/' --include '*.jsonl' --include '*.json' --include '*.md' --include '*.log' --include '*.png' \
    --include '*.txt' --include '*.sh' --exclude '*' \
    "$ROOT/logs/phase2/" "$WT/logs/phase2/"
  local pats=()
  if [ -f "$ROOT/.env" ]; then
    while IFS='=' read -r k v; do
      v=${v%\"}; v=${v#\"}
      case $k in HF_TOKEN|WANDB_API_KEY|GITHUB_TOKEN) [ ${#v} -ge 12 ] && pats+=("$v") ;; esac
    done < <(grep -E '^(HF_TOKEN|WANDB_API_KEY|GITHUB_TOKEN)=' "$ROOT/.env")
  fi
  for p in "${pats[@]}"; do
    grep -rqF -- "$p" "$WT/logs" && { say "SECRET FOUND in the stage: sync aborted"; return 2; }
  done
  grep -rqE 'hf_[A-Za-z0-9]{30,}' "$WT/logs" && { say "SECRET FOUND (hf_ token pattern): sync aborted"; return 2; }
  cd "$WT" || return 1
  git add -A
  local msg="phase 2 logs, $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')"
  if git rev-parse -q --verify HEAD > /dev/null; then git commit -q --amend -m "$msg"; else git commit -q -m "$msg"; fi
  git push -q -f origin HEAD:$BR && say "pushed $(git ls-files | wc -l) files to $BR" || { say "push failed"; return 1; }
}

if [ "${1:-}" = "--loop" ]; then
  every=${2:-15}
  while [ ! -e "$ROOT/logs/phase2/STOP" ]; do sync_once; sleep $((every * 60)); done
  sync_once
else
  sync_once
fi
