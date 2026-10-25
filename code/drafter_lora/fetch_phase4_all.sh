#!/usr/bin/env bash
# Phase 4 (drafter LoRA): fetch EVERYTHING from the public backups with one command, into this clone of branch
# phase4-drafter-lora (which itself holds the code, results/phase4_drafter/ and STATUS.md).
#
#   bash code/drafter_lora/fetch_phase4_all.sh              # everything (the HF dataset is ~8.3 GiB, mostly weights)
#   bash code/drafter_lora/fetch_phase4_all.sh --no-weights  # everything except adapter checkpoints / final adapter / merged drafter
#   bash code/drafter_lora/fetch_phase4_all.sh --dry-run     # show what would be fetched, download nothing
#
# Fetched into ./phase4_backup/ (kept out of git via .git/info/exclude):
#   phase4_backup/logs/  GitHub branch phase4-logs (git worktree): every log of the run, no weights, no ref/
#   phase4_backup/hf/    HF dataset Arushhh/phase4-drafter-lora (public): everything under logs/phase4 (adapter
#                        checkpoints, final adapter, merged bf16 drafter, all decode records incl. the vanilla re-decode,
#                        gate outputs, session scratchpad), plus results/phase4_drafter/ and STATUS.md
# Needs: git, python3 with huggingface_hub (pip install huggingface_hub).  No token: both backups are public.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
HF_REPO=${P4_HF_REPO:-Arushhh/phase4-drafter-lora}
PY=${PY:-python3}
dry=0; weights=1
for a in "$@"; do
  case $a in --no-weights) weights=0;; --dry-run) dry=1;; *) echo "unknown option: $a"; exit 2;; esac
done
export HF_HUB_OFFLINE=0
if [ $dry = 0 ]; then
  grep -qx "phase4_backup/" .git/info/exclude 2>/dev/null || echo "phase4_backup/" >> .git/info/exclude
fi

echo "== GitHub branch phase4-logs (all logs)"
git fetch -q origin phase4-logs
git log -1 --format="   commit %h, %cd: %s" --date=iso FETCH_HEAD
if [ $dry = 0 ]; then
  if [ -d phase4_backup/logs ]; then
    git -C phase4_backup/logs checkout -q --detach FETCH_HEAD
  else
    mkdir -p phase4_backup && git worktree add -q --detach phase4_backup/logs FETCH_HEAD
  fi
  echo "   -> $ROOT/phase4_backup/logs"
fi

echo "== Hugging Face dataset $HF_REPO (public)"
"$PY" - "$HF_REPO" "$dry" "$weights" <<'EOF'
import sys
repo, dry, weights = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    sys.exit("   huggingface_hub is missing: pip install huggingface_hub")
WEIGHTS = ["*/ckpt/*", "*/final_adapter/*", "*/merged/*"]
heavy = lambda f: any(x in f for x in ("/ckpt/", "/final_adapter/", "/merged/"))
info = HfApi().dataset_info(repo, files_metadata=True)
files = [(s.rfilename, s.size or 0) for s in info.siblings]
sel = [f for f in files if weights or not heavy(f[0])]
print(f"   {len(sel)} of {len(files)} files, {sum(s for _, s in sel) / 2**30:.2f} GiB; last updated {info.last_modified}")
if not dry:
    p = snapshot_download(repo, repo_type="dataset", local_dir="phase4_backup/hf",
                          ignore_patterns=None if weights else WEIGHTS)
    print("   ->", p)
EOF
echo "== done: code + results in this branch; logs in phase4_backup/logs; the full HF backup in phase4_backup/hf"
