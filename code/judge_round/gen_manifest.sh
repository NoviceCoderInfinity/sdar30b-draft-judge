#!/usr/bin/env bash
# Phase 5 reproducibility manifest: sha256 of every input training needs + every checkpoint. Regenerated at training
# start (states final) and after each checkpoint. Big pinned models (30B/1.7B) are recorded by path+revision, not hashed.
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
OUT=logs/phase5/MANIFEST.sha256; mkdir -p logs/phase5
tmp=$OUT.tmp; : > "$tmp"
{
  echo "# Phase 5 MANIFEST — generated $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST'), git $(git rev-parse HEAD)"
  echo "# pinned models (not hashed): b32=$(/venv/p2/bin/python -c "import sys;sys.path.insert(0,'code');import research_protocol as P;print(P.MODELS['b32'].get('repo',P.MODELS['b32']),P.MODELS['b32'].get('revision',''))" 2>/dev/null)"
  echo "# merged drafter dir: logs/phase4/train/main/merged"
} >> "$tmp"
# sha256 the tractable inputs (quietly skip missing ones)
LIST=$(ls -1 \
  logs/c1b/trainset/fresh/train_index.jsonl \
  logs/c1b/trainset/fresh/index.jsonl \
  logs/phase5/record_pids*.json \
  logs/phase5/judge_states/rec_main/*.npy logs/phase5/judge_states/rec_main/meta.json logs/phase5/judge_states/rec_main/summary.json \
  logs/phase5/judge/start_128m/* \
  2>/dev/null)
# states npz across all four record roots + the merged drafter weight files + every rec_main checkpoint file
LIST="$LIST $(find logs/phase5/judge_sglang/record*/ -name 'states.*.npz' 2>/dev/null)"
LIST="$LIST $(find logs/phase4/train/main/merged -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.json' \) 2>/dev/null)"
LIST="$LIST $(find logs/phase5/judge/rec_main -type f 2>/dev/null)"
LIST="$LIST $(find logs/phase5/judge/rec_main_ep2 -type f 2>/dev/null)"
# shellcheck disable=SC2086
nice -n 19 ionice -c3 sha256sum $LIST >> "$tmp" 2>/dev/null
mv "$tmp" "$OUT"
echo "[manifest $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] wrote $OUT: $(grep -c '^[0-9a-f]' "$OUT") files hashed"
