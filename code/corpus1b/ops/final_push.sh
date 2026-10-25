#!/usr/bin/env bash
# user 2026-09-29 ~14:35 IST: everything done by 18:35 IST, then the instance is shut down (billing).
# Waits until the checkpointB eval queue finishes or 18:10 IST. At the cutoff it stops launching and kills the
# still-running eval shards (they could never reach HF in time), writes the summary of the finished shards, then
# pushes checkpoints + data + eval results to HF (data capped at 15 min -> done by ~18:30), then writes READY_TO_KILL.
cd /workspace/dmge
OUT=logs/c1b/eval/checkpointB; CUT=1790685600      # 18:10 IST
say(){ echo "[final-push $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*"; }
fin(){ grep -q 'eval queue finished' logs/c1b/eval_checkpointB.log 2>/dev/null; }
while [ "$(date +%s)" -lt $CUT ] && ! fin; do sleep 30; done
if fin; then say "evals finished"
else
  say "18:10 cutoff: evals still running -> no more launches, killing the running shards, partial summary"
  cp $OUT/queue_checkpointB.txt $OUT/not_launched_checkpointB.txt 2>/dev/null; rm -f $OUT/queue_checkpointB.txt
  pkill -f "[r]un_da_decode.py.*$OUT/"; sleep 20; pkill -9 -f "[r]un_da_decode.py.*$OUT/"
  /venv/train/bin/python code/corpus1b/eval_summary.py --dir $OUT && say "partial summary -> $OUT/summary.md"
fi
DATA_TIMEOUT=900 bash code/corpus1b/backup_hf.sh once
say "HF push round done"; date +%s > logs/c1b/READY_TO_KILL
