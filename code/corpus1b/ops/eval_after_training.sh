#!/usr/bin/env bash
# user 2026-09-29 ~14:15 IST: after training (stops 14:45 IST) evaluate the judge on the six standard sets + IFEval-541,
# plus the released model and blind accept on IFEval-541 (eval_ckpt.sh checkpointB). Waits until 14:44:30 IST AND
# until the trainer (torchrun) has exited (its final save + probe done), then runs the queue on all 8 GPUs, no deadline cut.
cd /workspace/dmge
alive(){ pgrep -f "^/venv/train/bin/python /venv/train/bin/torchrun .*train_c1b.py" > /dev/null; }
while [ "$(date +%s)" -lt 1790673270 ] || alive; do sleep 10; done
echo "[eval-after-training $(TZ=Asia/Kolkata date +%H:%M:%S) IST] trainer exited: $(grep -E '^\[(stop-at|done)\]' logs/c1b/train/main.log | tail -1)"
DEADLINE_FILE=/dev/null bash code/corpus1b/eval_ckpt.sh checkpointB
echo "[eval-after-training $(TZ=Asia/Kolkata date +%H:%M:%S) IST] eval queue finished"
