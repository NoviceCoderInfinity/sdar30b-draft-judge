#!/usr/bin/env bash
# One backup pass (user, 2026-09-15): everything to the PUBLIC Anupam-Rawat-IITB HF dataset, relevant code/docs to GitHub.
#   (1) data + pipeline code/strategy/env  -> Nemotron-data-subset-training-diffusion-moe/   (backup_r3_hf.py)
#   (2) train logs, eval records + summary, plateau probe, report -> Nemotron-split-moe-train/r3_A/  (checkpoints: ckpt_sync_r3.sh)
#   (3) code + docs (never logs/, .env, caches) -> GitHub branch round3-2026-09-14  (git_push_r3.sh)
# Serialised with flock so the 30-min loop and the deadline crons never run two passes at once.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
exec 9>/dev/shm/dlm/backup_r3.lock; flock 9
set -a; . ./.env; set +a; export HF_HOME=/root/.cache/huggingface
ts() { TZ=Asia/Kolkata date '+%F %H:%M:%S IST'; }
echo "$(ts) backup pass start"
/venv/r3/bin/python code/draftalign/eval_summary_r3.py > /dev/null 2>&1 || echo "$(ts) eval_summary FAILED"
/venv/r3/bin/python docs/report/build_r3_report.py --no-eval > /dev/null 2>&1 || echo "$(ts) report build FAILED"
/venv/r3/bin/python code/draftalign/backup_r3_hf.py 2>&1 | grep -E "^\[backup\]|Traceback|Error" | tail -2 | cut -c1-200
/venv/r3/bin/python code/draftalign/backup_split_moe_train.py 2>&1 | grep -E "split_backup|Traceback|Error" | tail -2 | cut -c1-200
bash code/draftalign/git_push_r3.sh 2>&1 | tail -1
echo "$(ts) backup pass done"
