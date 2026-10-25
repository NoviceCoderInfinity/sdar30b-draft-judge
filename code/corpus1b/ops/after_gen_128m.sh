#!/usr/bin/env bash
# Waits for wave_b_128m.sh ("WAVE B 128M DONE"), verifies each domain (index >= 95% of kept, no_draft == 0),
# builds the merged index, launches training in tmux train128 and watches it: relaunch on crash (max 3, --resume).
cd /workspace/dmge
say(){ echo "[after-gen $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*"; }
until grep -q 'WAVE B 128M DONE' logs/c1b/waveB_128m.log 2>/dev/null; do sleep 30; done
say "generation finished"
/venv/train/bin/python - <<'PY' || { say "VERIFY FAILED - not launching training"; exit 1; }
import glob, json, sys
sys.path.insert(0, "code/corpus1b"); import common as C
kept = {}; idx = {}
for f in glob.glob("logs/c1b/gen/waveB/gpu*/records.*.jsonl.gz"):
    for r in C.read_jsonl(f):
        if r.get("kept") in (True, "True"): kept[r["domain"]] = kept.get(r["domain"], 0) + 1
for e in C.read_jsonl("logs/c1b/trainset/waveB/index.jsonl"): idx[e["domain"]] = idx.get(e["domain"], 0) + 1
ok = all(idx.get(d, 0) >= 0.95 * k for d, k in kept.items())
print("[after-gen] kept", kept, "index", idx, "ok", ok); sys.exit(0 if ok else 1)
PY
nice /venv/train/bin/python logs/c1b/dl/build_index_128m.py 2>&1 | grep -vE 'Warning|warn' | tail -7
mkdir -p logs/c1b/train/main_128m
tmux new -d -s train128 'bash logs/c1b/dl/train_128m.sh'; say "training launched (tmux train128)"
alive(){ pgrep -f "^/venv/train/bin/python /venv/train/bin/torchrun .*index_128m" > /dev/null; }
sleep 120; n=0
while true; do
  if alive; then sleep 30; continue; fi
  grep -qE '^\[(stop-at|done)\]' logs/c1b/train/main_128m.log && { say "training finished: $(grep -E '^\[(stop-at|done)\]' logs/c1b/train/main_128m.log | tail -1)"; break; }
  n=$((n+1)); [ $n -gt 3 ] && { say "too many relaunches, giving up"; exit 1; }
  say "training exited unexpectedly: relaunch $n with --resume"; tmux kill-session -t train128 2>/dev/null
  tmux new -d -s train128 'bash logs/c1b/dl/train_128m.sh resume'; sleep 150
done
say "TRAIN 128M DONE"
