#!/usr/bin/env bash
# Every 30 min: backup_once_r3.sh (HF Anupam-Rawat-IITB + GitHub).  nohup bash code/draftalign/backup_loop_r3.sh >> logs/draftalign/backup_loop_r3.log 2>&1 &
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
while :; do bash code/draftalign/backup_once_r3.sh; sleep 1800; done
