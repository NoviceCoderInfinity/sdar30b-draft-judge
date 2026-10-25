#!/usr/bin/env bash
# fast smoke builds (300 rows per source) of the code and IF prompt lists into /dev/shm sandboxes
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TOKENIZERS_PARALLELISM=false C1B_DECONTAM_WORKERS=16
mkdir -p /dev/shm/c1b_sandbox/pl_smoke_code /dev/shm/c1b_sandbox/pl_smoke_if
cp data/c1b/prompt_lists/chat.jsonl /dev/shm/c1b_sandbox/pl_smoke_if/
/venv/train/bin/python code/corpus1b/build_prompt_lists.py --domains code --limit-per-source 300 --workers 8 \
  --out /dev/shm/c1b_sandbox/pl_smoke_code > logs/c1b/stage1/smoke_pl_code.log 2>&1 &
/venv/train/bin/python code/corpus1b/build_prompt_lists.py --domains if --limit-per-source 300 --workers 8 \
  --out /dev/shm/c1b_sandbox/pl_smoke_if > logs/c1b/stage1/smoke_pl_if.log 2>&1 &
wait; echo "smoke builds finished $(date -u +%H:%M:%S)"
