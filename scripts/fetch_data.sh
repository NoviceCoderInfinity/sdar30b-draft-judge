#!/usr/bin/env bash
# Fetch a phase's data (corpus, training inputs, checkpoints, eval records, logs) from the data repo
# Anupam-Rawat-IITB/sdar30b-draft-judge-data into data_hf/<phase>/, keeping each phase's original paths.
#
#   bash scripts/fetch_data.sh phase4                     # everything of Phase 4 (8.3 GiB)
#   bash scripts/fetch_data.sh phase5 --into-repo         # and copy it into the repo root (paths the code expects)
#   bash scripts/fetch_data.sh phase1 --include "logs/c1b/trainset/*"   # only part of it
#   bash scripts/fetch_data.sh samples                    # the readable corpus sample (small)
#
# --into-repo copies data_hf/<phase>/* into the repo root without overwriting any existing file, so the training and
# data-generation scripts of that phase find their inputs where they were (logs/c1b/..., logs/phase5/..., data/c1b/...).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PHASE=${1:?usage: fetch_data.sh <phase1..phase5|samples|all> [--include PATTERN] [--into-repo]}; shift
INC="*"; INTO=0
while [ $# -gt 0 ]; do
  case $1 in --include) INC=$2; shift 2;; --into-repo) INTO=1; shift;; *) echo "unknown option $1"; exit 2;; esac
done
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
[ "$PHASE" = all ] && PHASES="phase1 phase2 phase3 phase4 phase5 samples" || PHASES=$PHASE
for P in $PHASES; do
  echo "[fetch] $P ($INC) -> data_hf/$P"
  HF_HUB_OFFLINE=0 $PY -c "import sys; from huggingface_hub import snapshot_download as s; \
print(s('Anupam-Rawat-IITB/sdar30b-draft-judge-data', repo_type='dataset', local_dir='data_hf', allow_patterns=[sys.argv[1] + '/' + sys.argv[2]]))" "$P" "$INC"
  if [ $INTO = 1 ] && [ "$P" != samples ]; then
    rsync -a --ignore-existing --exclude README_source.md "data_hf/$P/" ./
    # Phase 5's packed training items ship as one tar (see docs/PHASE5_RESUME.md)
    [ -f logs/c1b/trainset/waveC/items.tar ] && [ ! -d logs/c1b/trainset/waveC/items ] && \
      tar -xf logs/c1b/trainset/waveC/items.tar -C logs/c1b/trainset/waveC/
    echo "[fetch] $P copied into the repo root"
  fi
done
