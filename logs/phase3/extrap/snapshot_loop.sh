#!/usr/bin/env bash
# Safety snapshots until the run is DONE: every 15 min, every 5 min from 07 Oct 01:45 IST (credit ends ~02:40 IST).
cd /workspace/dmge
while true; do
  /venv/p2/bin/python logs/phase3/extrap/snapshot.py 2>&1 | grep "^\[snapshot" | sed -E 's/(hf|ghp|github_pat)_[A-Za-z0-9_]{8,}/***/g' >> logs/phase3/snapshot.log
  grep -q "DONE:" logs/phase3/run.log && { echo "[snapshot] run DONE, loop ends" >> logs/phase3/snapshot.log; break; }
  now=$(date +%s); late=$(TZ=Asia/Kolkata date -d "2026-10-07 01:45" +%s)
  if [ "$now" -ge "$late" ]; then sleep 300; else sleep 900; fi
done
