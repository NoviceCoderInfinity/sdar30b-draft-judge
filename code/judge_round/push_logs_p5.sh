#!/usr/bin/env bash
# Phase 5 logs branch (operator, 23 Oct 2026 15:15 IST), as Phase 4's phase4-logs: an ORPHAN branch phase5-logs in its own
# worktree ($WT), holding ONLY small text logs and stats (allowlist below, files <= 5 MB). Never traces, records.*.jsonl.gz,
# prompt lists, data/, weights or .env: those live on Hugging Face only. Same secret scan as backup_p5.py before every commit
# (a hit stops this step with exit 2). Commit as Arush777, push with http.extraHeader. Called by backup_loop_p5.sh (niced).
#   bash code/judge_round/push_logs_p5.sh          # one pass
set -u
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}; WT=${WT:-/workspace/p5_logs_wt}; BR=phase5-logs; MAXMB=5
cd "$ROOT" || exit 1
set -a; . /workspace/.env; set +a
PY=${PY:-/venv/p2/bin/python}; "$PY" -c "import huggingface_hub" 2>/dev/null || PY=/workspace/dl/venv/bin/python
EV=$ROOT/logs/phase5/events.tsv
ev(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> $EV; }
say(){ echo "[push-logs $(date -u +%FT%TZ)] $*"; }
AUTH="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"
ev git_push_logs start
# the worktree: an orphan branch, created once (no files from phase5-box-s1)
if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
  git worktree add --orphan -b $BR "$WT" >/dev/null 2>&1 || git worktree add --orphan $BR "$WT" >/dev/null 2>&1 || { say "worktree add failed"; ev git_push_logs "end rc=1 worktree"; exit 1; }
  git -C "$WT" config user.name Arush777; git -C "$WT" config user.email 23je0154@iitism.ac.in
fi
# the allowlist: text logs and small stats only, copied with their relative paths; /workspace/dl -> logs/dl
copy(){ # copy SRC_ROOT DEST_SUBDIR find-args...  (files <= MAXMB only)
  local src=$1 dst=$2; shift 2
  ( cd "$src" && find . -type f -size -$((MAXMB * 1024 + 1))k "$@" -print0 ) | ( cd "$src" && rsync -a --from0 --files-from=- . "$WT/$dst/" ); }
mkdir -p "$WT/logs/dl"
copy "$ROOT" "" \( -path './logs/phase5/*' -o -path './logs/c1b/fresh/*' -o -path './logs/c1b/fresh_targets.json' \
     -o -regex './logs/c1b/gen/waveC/gpu[0-9]*/\(stats\.jsonl\|done\.txt\)' \) \
     -not -name '*.gz' -not -name '*.npz' -not -name '*.pt' -not -name '*.safetensors' -not -path '*/trace/*' -not -path '*/smoke_trace/*' \
     -not -path '*/judge_side/*' -not -path '*/drafter_side/*' -not -path '*/judge_trace/*' -not -path '*/drafter_trace/*' -not -path '*/judge_sgl/*' \
     -not -name '.env' -not -name '*.env'
copy /workspace/dl logs/dl \( -name 'events.tsv' -o -name '*.log' \)
cd "$WT" || exit 1
git add -f -A -- logs 2>/dev/null   # -f: the shared .git/info/exclude lists logs/c1b/
if ! nice -n 19 "$PY" "$ROOT/code/judge_round/backup_p5.py" --scan-staged --git-dir "$WT"; then
  git reset -q; say "SECRET SCAN HIT in the logs branch stage: step stopped"; ev git_push_logs "STOP secret scan hit"; touch "$ROOT/logs/phase5/BACKUP_STOPPED"; exit 2; fi
if ! git diff --cached --quiet || ! git rev-parse --verify -q HEAD >/dev/null; then
  git -c user.name=Arush777 -c user.email=23je0154@iitism.ac.in commit -q -m "phase 5 logs $(date -u +%FT%TZ)" || { say "commit failed"; ev git_push_logs "end rc=1 commit"; exit 1; }
fi
git -c http.extraHeader="$AUTH" push -q origin HEAD:refs/heads/$BR 2>&1 | grep -v -i -E 'token|authorization'; rc=${PIPESTATUS[0]}
n=$(git ls-files | wc -l); sz=$(git ls-files -z | xargs -0 du -cb 2>/dev/null | tail -1 | cut -f1)
say "branch $BR commit $(git rev-parse --short HEAD) files $n size $(numfmt --to=iec ${sz:-0}) push rc=$rc"
ev git_push_logs "end rc=$rc commit=$(git rev-parse --short HEAD) files=$n bytes=${sz:-0}"
exit $rc
