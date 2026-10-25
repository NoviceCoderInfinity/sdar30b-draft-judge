#!/usr/bin/env bash
# Backups of the 1B-corpus artefacts to the PUBLIC Hugging Face repos when .env sets BACKUP_TO_HF=1,
# HF_PUSH_REPO=<user>/<dataset repo> (the corpus) and HF_CKPT_REPO=<user>/<model repo> (the checkpoints).
# Uploads generation records, drafts, training items, prompt lists, eval sets, probes, logs and eval summaries
# (code/corpus1b/hf_sync.py data) and every complete checkpoint, verified file by file on the Hub, after which
# local checkpoints older than the 3 newest of their run are deleted (hf_sync.py ckpt). Never .env, never
# hf_cache or the models, never GPQA records. docs/PHASE1_HF_ARTIFACTS.md (the links) is pushed to GitHub.
#   bash code/corpus1b/backup_hf.sh              # once: data + checkpoints + links
#   bash code/corpus1b/backup_hf.sh data         # the corpus only (run_wave_a.sh: before training)
#   bash code/corpus1b/backup_hf.sh ckpt         # checkpoints + prune + links
#   nohup bash code/corpus1b/backup_hf.sh loop > logs/c1b/backup_hf.log 2>&1 &     # every 30 min
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home}
[ "${BACKUP_TO_HF:-0}" = "1" ] && [ -n "${HF_PUSH_REPO:-}" ] && [ -n "${HF_CKPT_REPO:-}" ] \
  || { echo "backup off (set BACKUP_TO_HF=1, HF_PUSH_REPO and HF_CKPT_REPO)"; exit 0; }
PY=/venv/train/bin/python
say(){ echo "[backup $(date -u +%H:%M:%S) UTC] $*"; }

push_links(){  # commit + push only the links doc (and the README pointer) to the branch, token never in a URL
  git diff --quiet -- docs/PHASE1_HF_ARTIFACTS.md 2>/dev/null && git ls-files --error-unmatch docs/PHASE1_HF_ARTIFACTS.md >/dev/null 2>&1 && return 0
  [ -n "${GITHUB_TOKEN:-}" ] || { say "no GITHUB_TOKEN: links doc not pushed"; return 0; }
  local br=${GIT_PUSH_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}         # the branch this box runs (never a hardcoded one)
  [ "$br" = HEAD ] && { say "detached HEAD: links doc not pushed"; return 0; }
  git add docs/PHASE1_HF_ARTIFACTS.md && git commit -q -m "phase 1: HF artefact links (auto)" -- docs/PHASE1_HF_ARTIFACTS.md \
    && git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
         push -q origin "HEAD:$br" && say "links doc pushed to GitHub ($br)" || say "links doc push failed (retry next round)"
}
data(){ timeout ${DATA_TIMEOUT:-1500} $PY code/corpus1b/hf_sync.py data \
          || { say "data upload failed or hit ${DATA_TIMEOUT:-1500} s (resumes next round)"; return 1; }; }
ckpt(){ $PY code/corpus1b/hf_sync.py ckpt || say "checkpoint sync reported a problem"; push_links; }
once(){ ckpt; data; ckpt; }

case "${1:-once}" in
  data) data ;;
  ckpt) ckpt ;;
  once) once ;;
  loop) while true; do say "round"; once; say "round done"; sleep ${BACKUP_EVERY:-1800}; done ;;
  ckptloop) while true; do ckpt; sleep ${CKPT_EVERY:-300}; done ;;     # checkpoints only, every 5 min
  *) echo "usage: $0 [once|data|ckpt|loop|ckptloop]"; exit 1 ;;
esac
