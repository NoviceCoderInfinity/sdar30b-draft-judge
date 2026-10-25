#!/usr/bin/env bash
# resume the paused downloads (IF, code, codeforces) once the maths + science sources are in
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
busy(){ ps -eo args --no-headers | grep -E "download.py --only (om[r]|omi2gs[m]|osr[2]) " | grep -vc grep; }
while [ "$(ps -eo args --no-headers | grep -cE 'download.py --only (om[r]|omi2gs[m],omi2math|osr[2]) ')" -gt 0 ]; do sleep 20; done
echo "[resume_dl $(date -u +%H:%M:%S)] maths/science sources in; resuming ifcomp, cc+oci, cf"
/venv/train/bin/python code/corpus1b/download.py --only ifcomp --include-optional --out data/c1b/dl_parts/manifest_5.json >> logs/c1b/dl/dl_5.log 2>&1 &
/venv/train/bin/python code/corpus1b/download.py --only cc,oci --include-optional --out data/c1b/dl_parts/manifest_6.json >> logs/c1b/dl/dl_6.log 2>&1 &
/venv/train/bin/python code/corpus1b/download.py --only cf --include-optional --out data/c1b/dl_parts/manifest_1.json >> logs/c1b/dl/dl_1.log 2>&1 &
wait; echo "[resume_dl $(date -u +%H:%M:%S)] all resumed downloads finished"
