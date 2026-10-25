#!/usr/bin/env bash
# One-time setup for one decode setup (or all of them), from this clone of the repo.
#
#   bash setup.sh phase4-rd1        # environment + what phase4-rd1 needs (models, weights, benchmarks)
#   bash setup.sh all               # everything every setup needs
#   bash setup.sh --list            # the setups
#
# Steps (each is skipped once done; re-run to resume):
#   1. .venv: Python 3.12, torch 2.11.0 (CUDA 12.8 wheels), the exact package pins of the last run (env/requirements.txt)
#   2. the pinned base models into hf_cache/hub: SDAR-30B-A3B-Chat-b32 @ c351bbc (~61 GB), SDAR-1.7B-Chat-b32 @ 2c83c05
#      (~3.4 GB) when the setup uses it, and the Phase 2 released drafters when needed
#   3. the trained weights into weights/: judge (Anupam-Rawat-IITB/SDAR-30B-judge-LoRA, root = the 128M checkpoint),
#      drafter (Anupam-Rawat-IITB/SDAR-1.7B-drafter-LoRA, root = merged weights), Phase 5 judge only for phase5-rd1
#   4. the seven benchmarks into hf_home (read offline by every decode) + the frozen IFEval-541 file
# Afterwards: source env.sh, then ./run.sh <setup> <benchmark>.
# Needs: Linux, NVIDIA GPUs with CUDA 12.8+ drivers, python3.12, ~80 GB free disk. No token: all repos are public.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
HFU=Anupam-Rawat-IITB
JUDGE_REPO=$HFU/SDAR-30B-judge-LoRA
DRAFTER_REPO=$HFU/SDAR-1.7B-drafter-LoRA
DATA_REPO=$HFU/sdar30b-draft-judge-data
SETUPS=$(ls setups | sed 's/\.sh$//' | tr '\n' ' ')
say(){ echo "[setup $(date '+%H:%M:%S')] $*"; }
mark(){ mkdir -p .setup; touch ".setup/$1"; }
done_(){ [ -f ".setup/$1" ]; }

[ "${1:-}" = "--list" ] && { for s in $SETUPS; do ( ROOT=$ROOT; . setups/$s.sh; printf '  %-14s %s\n' "$s" "$DESC" ); done; exit 0; }
WANT=${1:?usage: bash setup.sh <setup|all|--list>}
[ "$WANT" = all ] && WANT=$SETUPS
NEEDS=""
for s in $WANT; do
  [ -f setups/$s.sh ] || { echo "unknown setup '$s' (bash setup.sh --list)"; exit 2; }
  NEEDS="$NEEDS $(ROOT=$ROOT; . setups/$s.sh; echo $NEEDS)"
done
need(){ [[ " $NEEDS " == *" $1 "* ]]; }

cat > env.sh <<EOF
# source this before running anything: . env.sh
. "$ROOT/.venv/bin/activate"
export HF_HOME="$ROOT/hf_home" NLTK_DATA="$ROOT/hf_home/nltk_data" TOKENIZERS_PARALLELISM=false
EOF
PY=$ROOT/.venv/bin/python

# ---- 1. environment -------------------------------------------------------------------------------------------------
if ! done_ env; then
  PYBIN=${PYBIN:-$(command -v python3.12 || command -v python3)}
  say "venv .venv with $($PYBIN --version)"
  $PYBIN -m venv .venv && $PY -m pip install -q -U pip wheel
  $PY -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
  $PY -m pip install -q -r env/requirements.txt
  $PY -m pip install -q pytest
  mkdir -p hf_home/nltk_data && $PY -m nltk.downloader -q -d hf_home/nltk_data punkt punkt_tab
  $PY -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
  mark env
fi
. ./env.sh

# ---- 2. pinned base models and drafters -----------------------------------------------------------------------------
models=b32; need d32 && models=b32,d32
if ! done_ "models_$models"; then
  say "base models ($models) -> hf_cache/hub"
  HF_HUB_OFFLINE=0 $PY code/drafter_lora/hf_progress.py --models $models
  mark "models_$models"
fi
for k in F2 S1; do
  if need $k && ! done_ "drafter_$k"; then
    say "Phase 2 drafter $k"
    HF_HUB_OFFLINE=0 $PY -c "import sys; sys.path[:0]=['code/compat','code']; import research_protocol; \
import os; os.environ['HF_HUB_OFFLINE']='0'; from phase2 import drafters as D; print('$k', D.download('$k'))"
    mark "drafter_$k"
  fi
done

# ---- 3. trained weights ---------------------------------------------------------------------------------------------
hfdl(){ HF_HUB_OFFLINE=0 $PY -c "import sys; from huggingface_hub import snapshot_download as s; \
s(sys.argv[1], repo_type=sys.argv[2], local_dir=sys.argv[3], allow_patterns=sys.argv[4].split(','), \
  ignore_patterns=(sys.argv[5].split(',') if len(sys.argv) > 5 else None))" "$@" > /dev/null; }
whash(){ $PY -c "import sys; sys.path[:0]=['code/compat','code']; from draftalign.run_da_decode import adapter_weights_hash as h; print(h('$1')[:8])"; }
if need judge && ! done_ w_judge; then
  say "judge (128M checkpoint) -> weights/judge"
  hfdl $JUDGE_REPO model weights/judge "adapter.safetensors,adapter_config.json"
  [ "$(whash weights/judge)" = 35805974 ] || { say "judge weights hash mismatch (want 35805974)"; exit 3; }
  mark w_judge
fi
if need drafter && ! done_ w_drafter; then
  say "drafter (Phase 4, merged) -> weights/drafter"
  hfdl $DRAFTER_REPO model weights/drafter "*" "training/*,README.md,.gitattributes"
  mark w_drafter
fi
if need judge_phase5 && ! done_ w_judge_phase5; then
  say "Phase 5 judge (155.66M, failed) -> weights/judge_phase5"
  hfdl $DATA_REPO dataset .setup/dl "phase5/logs/phase5/judge/rec_main_ep2/final/adapter*"
  mkdir -p weights/judge_phase5 && cp .setup/dl/phase5/logs/phase5/judge/rec_main_ep2/final/adapter* weights/judge_phase5/
  [ "$(whash weights/judge_phase5)" = f08e0a0d ] || { say "Phase 5 judge hash mismatch (want f08e0a0d)"; exit 3; }
  mark w_judge_phase5
fi

# ---- 4. benchmarks --------------------------------------------------------------------------------------------------
if ! done_ benchmarks; then
  say "benchmarks -> hf_home (+ frozen IFEval-541 -> data/c1b/eval)"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $PY - <<'EOF'
from datasets import load_dataset
for args, kw in [(("openai/gsm8k", "main"), {"split": "test"}), (("google-research-datasets/mbpp", "sanitized"), {"split": "test"}),
                 (("fingertap/GPQA-Diamond",), {"split": "test"}), (("openai_humaneval",), {"split": "test"}),
                 (("AI-MO/aimo-validation-aime",), {"split": "train"}), (("HuggingFaceH4/MATH-500",), {"split": "test"}),
                 (("google/IFEval",), {"split": "train"})]:
    print(" ", args[0], len(load_dataset(*args, **kw)), flush=True)
EOF
  hfdl $DATA_REPO dataset .setup/dl "phase1/data/c1b/eval/ifeval541.jsonl*"
  mkdir -p data/c1b/eval && cp .setup/dl/phase1/data/c1b/eval/ifeval541.jsonl* data/c1b/eval/
  n=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; import research_protocol as P; \
print(' '.join(f'{s}={len(P.items_for_split(s))}' for s in ('gsm8k500','math500','humaneval','mbpp','gpqa','aime','ifeval541')))")
  say "benchmarks load offline: $n"
  mark benchmarks
fi
say "done. Next:  . env.sh && ./run.sh ${1} gsm8k500 --limit 8"
