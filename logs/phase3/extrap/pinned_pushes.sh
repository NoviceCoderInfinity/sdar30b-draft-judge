#!/usr/bin/env bash
# Extra safety snapshots pinned to 02:20 and 02:30 IST on 07 Oct (credit expected to end ~02:40 IST).
cd /workspace/dmge
for t in "2026-10-07 02:20" "2026-10-07 02:30"; do
  target=$(TZ=Asia/Kolkata date -d "$t" +%s)
  while [ "$(date +%s)" -lt "$target" ]; do sleep 20; done
  /venv/p2/bin/python logs/phase3/extrap/snapshot.py 2>&1 | grep "^\[snapshot" | sed -E 's/(hf|ghp|github_pat)_[A-Za-z0-9_]{8,}/***/g' | sed "s/^\[snapshot/[pinned $t IST snapshot/" >> logs/phase3/snapshot.log
done
