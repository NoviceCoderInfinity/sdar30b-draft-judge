#!/usr/bin/env bash
# One command -> one directory with EVERYTHING from Phase 1: code + docs (GitHub), the generated corpus, logs, eval
# records and results (HF dataset), and the judge checkpoints (HF model repo), laid out exactly as on the training box.
#   bash retrieve_all.sh [DEST]            # DEST defaults to ./phase1_all
#   WEIGHTS=final bash retrieve_all.sh     # only the final checkpoints (main/final, main_128m/final); default: all
#   WEIGHTS=none  bash retrieve_all.sh     # no weights
#   ITEMS=0 bash retrieve_all.sh           # don't unpack the training-item tars
set -euo pipefail
DEST=${1:-phase1_all}; WEIGHTS=${WEIGHTS:-all}; ITEMS=${ITEMS:-1}
git clone -b phase1-datacorpus https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git "$DEST"
cd "$DEST"
python3 -m pip install -q "huggingface_hub>=0.24"
WEIGHTS=$WEIGHTS python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
# the dataset mirrors the repo root: data/..., logs/c1b/{gen,draft,trainset,train,eval,...}, docs/, HANDOVER.md
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir=".")
w = os.environ["WEIGHTS"]
if w != "none":   # the model repo holds <run>/<checkpoint>/...; on the box they live under logs/c1b/train/<run>/<checkpoint>/
    pats = None if w == "all" else ["main/final/*", "main_128m/final/*", "main_128m/tok_000128000000/*", "*.md"]
    snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", local_dir="logs/c1b/train", allow_patterns=pats)
PY
if [ "$ITEMS" = 1 ]; then   # each items_tar/<h2>.tar holds items/<h2>/*.npz (the paths index.jsonl refers to)
  for w in logs/c1b/trainset/wave*; do for t in "$w"/items_tar/*.tar; do [ -f "$t" ] && tar -xf "$t" -C "$w"; done; done
fi
echo "done: $(pwd). Start with RETRIEVE.md, docs/metrics_28.5M.md and docs/results/phase1/."
