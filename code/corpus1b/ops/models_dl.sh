#!/usr/bin/env bash
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home HF_HUB_OFFLINE=0
/venv/train/bin/python - <<'PY'
import sys, os; sys.path.insert(0, "code")
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download
import research_protocol as P
for k in ("b32", "d32"):
    m = P.MODELS[k]
    print(k, snapshot_download(m["repo"], revision=m["revision"], cache_dir=m["cache_dir"], max_workers=16), flush=True)
PY
rc=$?; echo "models rc=$rc $(date -u)"; [ $rc -eq 0 ] && touch logs/c1b/stage1/.done_models
