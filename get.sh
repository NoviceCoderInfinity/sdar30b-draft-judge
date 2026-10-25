#!/usr/bin/env bash
# One command from nothing to a working setup:
#   curl -sL https://raw.githubusercontent.com/NoviceCoderInfinity/sdar30b-draft-judge/main/get.sh | bash -s phase4-rd1
# Clones the repo into ./sdar30b-draft-judge (or $DEST), runs setup.sh for the setup, and, when GPUs are present,
# decodes one GSM8K item per GPU as a check. Setups: vanilla phase1 phase2-coda phase2-qwen3 phase3-rd1 phase3-rd2 phase4-rd1
# phase4-rd2 phase5-rd1 (or "all").
set -euo pipefail
SETUP=${1:-phase4-rd1}
DEST=${DEST:-sdar30b-draft-judge}
REPO=${REPO:-https://github.com/NoviceCoderInfinity/sdar30b-draft-judge.git}
if [ -d "$DEST/.git" ]; then git -C "$DEST" pull -q --ff-only; else git clone -q "$REPO" "$DEST"; fi
cd "$DEST"
bash setup.sh "$SETUP"
if command -v nvidia-smi > /dev/null && [ "$SETUP" != all ]; then
  ./run.sh "$SETUP" gsm8k500 --limit 1
fi
echo
echo "Ready in $(pwd). Evaluate with:  . env.sh && ./run.sh $SETUP all"
