#!/usr/bin/env bash
# Push every round-3 checkpoint to the public HF dataset, then prune local disk to
#   {latest 2} ∪ {best by probes (probes_r3/<run>/best.json)}.
# A step dir is deleted only after its adapter.safetensors sha256 matches the uploaded LFS sha.
#   nohup bash code/draftalign/ckpt_sync_r3.sh <run_dir> > logs/draftalign/ckpt_sync_r3.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN="$1"; NAME=$(basename "$RUN"); PR=logs/draftalign/probes_r3/$NAME
set -a; . ./.env; set +a; export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_PROGRESS_BARS=1
unset HF_TOKEN   # .env holds a different account; push as the logged-in HF account (Anupam-Rawat-IITB)
REPO=Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts; DEST=Nemotron-split-moe-train/$NAME/checkpoints
while :; do
  for sd in $(ls -d "$RUN"/step* 2>/dev/null | sort); do
    st=$(basename "$sd"); [ -f "$sd/adapter.safetensors" ] && [ -f "$sd/trainer_state.pt" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$sd/trainer_state.pt") )) -lt 120 ] && continue     # still being written
    [ -f "$sd/.pushed" ] && continue
    /venv/r3/bin/python - "$sd" "$DEST/$st" <<'PY' && touch "$sd/.pushed"
import hashlib, os, sys
from huggingface_hub import HfApi
sd, dest = sys.argv[1], sys.argv[2]; api = HfApi(); repo = os.environ.get("REPO_OVERRIDE", "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts")
assert api.whoami()["name"] == "Anupam-Rawat-IITB", "not logged in as Anupam-Rawat-IITB"
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=sd, path_in_repo=dest, ignore_patterns=[".pushed"],
                  commit_message=f"r3 checkpoint {dest}")
local = hashlib.sha256(open(os.path.join(sd, "adapter.safetensors"), "rb").read()).hexdigest()
info = api.get_paths_info(repo, [dest + "/adapter.safetensors"], repo_type="dataset")[0]
assert info.lfs and info.lfs.sha256 == local, f"sha mismatch after upload: {info.lfs and info.lfs.sha256} != {local}"
print(f"[ckpt_sync] pushed {dest} sha {local[:16]}", flush=True)
PY
  done
  # prune: keep latest 2 (by step) + best
  best=""; [ -f "$PR/best.json" ] && best="step$(printf '%06d' "$(python3 -c "import json;print(json.load(open('$PR/best.json'))['step'])")")"
  all=($(ls -d "$RUN"/step* 2>/dev/null | xargs -n1 basename | sort)); n=${#all[@]}
  for ((i=0; i<n-2; i++)); do
    st=${all[$i]}; [ "$st" = "$best" ] && continue; [ -f "$RUN/$st/.pushed" ] || continue
    [ -d "$PR/.claim_$st" ] && [ ! -f "$PR/$st/holdout.log" ] && continue            # probe in progress on it
    rm -rf "$RUN/$st" && echo "$(date -u +%FT%TZ) pruned $st (kept latest 2 + best=$best)"
  done
  [ -f "$RUN/done" ] && [ -z "$(ls -d "$RUN"/step* 2>/dev/null | while read d; do [ -f "$d/.pushed" ] || echo x; done)" ] && { echo "[ckpt_sync] run finished, all pushed"; break; }
  sleep 300
done
