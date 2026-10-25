#!/usr/bin/env bash
# Phase 2 drafter swap, NO expert pool, on a fresh 8-GPU box: everything in one command (run it in tmux).
#   git clone -b phase2-drafter-swap git@github.com:NoviceCoderInfinity/diffusion-moe-general-expert.git dmge_p2
#   cd dmge_p2 && cp ~/.env .env          # HF_TOKEN (+ WANDB_API_KEY, DA_WANDB_PROJECT, ENABLE_WANDB=1)
#   bash code/phase2/replicate_nopool.sh 2>&1 | tee -a logs/phase2/replicate.log
# Steps (each resumes):
#  1. setup_p2.sh CPU steps: /venv/p2, the 30B model, the drafters (CoDA F2, Qwen3-0.6B S1) at pinned revisions,
#     round-3 stock records, the benchmark datasets, IFEval-541 and Phase 1's stock IFEval run
#  2. Phase 1's 128M judge (Arushhh/c1b-judge-lora-sdar30b-b32 main_128m/tok_000128000000, weights hash 35805974) and
#     Phase 1's no-pool SDAR-1.7B + 128M judge records on all 7 sets (the reference A)
#  3. backup loops: Hugging Face (backup_p2.py), GitHub logs branch (backup_github.sh), plots to W&B (plot_p2.py)
#  4. decode, no pool, 8 shards per set: S1_nopool then F2_nopool on humaneval mbpp gsm8k500 ifeval541 math500 gpqa aime
#  5. tables (compare_table_p2.py), the analysis (analyze_nopool_p2.py), final plots and backups
# Env: ARMS (default "S1 F2"), GPUS, NSH.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=/venv/p2/bin/python; mkdir -p logs/phase2
say(){ echo "[replicate $(date -u +%H:%M:%S) UTC] $*"; }
SKIP_CHECKS=1 P2_DRAFTERS="F2 S1" bash code/phase2/setup_p2.sh || { say "setup FAILED"; exit 1; }
if [ ! -f logs/phase2/ref/.done_nopool_refs ]; then
  HF_HUB_OFFLINE=0 $PY - <<'PYEOF' || { echo "reference download FAILED"; exit 1; }
import glob, os, shutil
from huggingface_hub import snapshot_download
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", allow_patterns=["main_128m/tok_000128000000/adapter*"],
                  local_dir="logs/phase2/ref/_hf_c1b_ckpt")
dst = "logs/phase2/ref/c1b_128M/tok_000128000000"; shutil.rmtree(dst, ignore_errors=True)
shutil.copytree("logs/phase2/ref/_hf_c1b_ckpt/main_128m/tok_000128000000", dst)
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir="logs/phase2/ref/_hf_c1b_eval",
                  allow_patterns=["logs/c1b/eval/probe128M/*_nopool/*.jsonl", "logs/c1b/eval/checkpoint128M_nopool/gpqa_judge/*.jsonl"])
src = {s: f"logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/probe128M/{s}_nopool" for s in ["aime", "gsm8k500", "humaneval", "ifeval541", "math500", "mbpp"]}
src["gpqa"] = "logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/checkpoint128M_nopool/gpqa_judge"
for s, d in src.items():
    out = f"logs/phase2/ref/S0_128M_nopool_by_set/{s}"; os.makedirs(out, exist_ok=True)
    for f in glob.glob(d + "/*.jsonl"): shutil.copy(f, out)
PYEOF
  h=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; from draftalign.run_da_decode import adapter_weights_hash; print(adapter_weights_hash('logs/phase2/ref/c1b_128M/tok_000128000000'))")
  [ "${h:0:8}" = 35805974 ] || { say "128M judge hash $h != 35805974"; exit 1; }
  touch logs/phase2/ref/.done_nopool_refs
fi
pgrep -f "backup_p2[.]py --loop" > /dev/null || nohup $PY code/phase2/backup_p2.py --loop 15 >> logs/phase2/hf_backup.log 2>&1 &
pgrep -f "backup_githu[b].sh --loop" > /dev/null || nohup bash code/phase2/backup_github.sh --loop 15 >> logs/phase2/gh_backup.log 2>&1 &
pgrep -f "plot_p2[.]py --loop" > /dev/null || CUDA_VISIBLE_DEVICES= nohup $PY code/phase2/plot_p2.py --loop 10 >> logs/phase2/plots.log 2>&1 &
export P2_ADAPTER=logs/phase2/ref/c1b_128M/tok_000128000000 P2_S0_REF_ROOT=logs/phase2/ref/S0_128M_nopool_by_set \
       P2_POOL=none P2_TAG=_nopool P2_EXPERT_LOG=full P2_NO_RETRAIN=1
for A in ${ARMS:-S1 F2}; do
  say "arm ${A}_nopool"; bash code/phase2/run_phase2.sh $A decode report; say "arm ${A}_nopool -> exit $?"
done
for A in ${ARMS:-S1 F2}; do $PY code/phase2/compare_table_p2.py ${A}_nopool humaneval mbpp gsm8k500 ifeval541 math500 gpqa aime; done \
  > logs/phase2/analysis/tables/TABLES.md 2>&1
mkdir -p logs/phase2/analysis/nopool
CUDA_VISIBLE_DEVICES= $PY code/phase2/analyze_nopool_p2.py > logs/phase2/analysis/nopool/run.log 2>&1
CUDA_VISIBLE_DEVICES= $PY code/phase2/plot_p2.py --once >> logs/phase2/plots.log 2>&1
$PY code/phase2/backup_p2.py >> logs/phase2/hf_backup.log 2>&1; bash code/phase2/backup_github.sh >> logs/phase2/gh_backup.log 2>&1
$PY code/phase2/publish_bundle_p2.py >> logs/phase2/hf_backup.log 2>&1
say "done: logs/phase2/analysis/tables/TABLES.md, logs/phase2/analysis/nopool/REPORT.md"
