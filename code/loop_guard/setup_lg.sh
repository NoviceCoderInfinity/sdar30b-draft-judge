#!/usr/bin/env bash
# Phase 5 setup on a FRESH 8x RTX PRO 6000 box: Phase 3/4's setup (venv /venv/p2, the 30B and 1.7B, the 128M judge,
# references, benchmark sets, 16/16 decode checks), then the Phase 4 inputs this experiment needs, from the public
# Phase 4 backup: the merged drafter and the IFEval records of rd1, rd2 and the same-box vanilla decode.
#   bash code/loop_guard/setup_lg.sh          # from the repo root; needs HF_TOKEN in the environment (read access is enough)
set -eu
ROOT=${ROOT:-/workspace/diffusion-moe-general-expert}
PY=${PY:-/venv/p2/bin/python}
cd $ROOT
L=logs/phase5; mkdir -p $L
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/setup_lg.log; }

# The three .done_* markers of the Phase 3 box used to be tracked in git and made a fresh box skip the venv, the model
# downloads and the tests (Phase 4 STATUS.md, 21:15 IST). They are deleted on this branch; this is a belt-and-braces check.
for m in p2_setup judge_refs tests; do [ -f logs/phase3_8gpu/setup/.done_$m ] && { say "stale marker .done_$m present: moving it aside"; mkdir -p $L/stale_markers; mv logs/phase3_8gpu/setup/.done_$m $L/stale_markers/; }; done

say "1/3 Phase 3/4 setup (venv, models, judge adapter, references, benchmark sets, 16/16 checks; Phase 4 measured ~27 min)"
bash code/phase3_8gpu/setup_p3_8gpu.sh 2>&1 | tee -a $L/setup_p3.log; [ ${PIPESTATUS[0]} = 0 ] || { say "setup_p3_8gpu.sh FAILED (see $L/setup_p3.log)"; exit 1; }

say "2/3 Phase 4 inputs from the public backup Arushhh/phase4-drafter-lora"
HF_HUB_OFFLINE=0 $PY - <<'PYEOF' || { echo "Phase 4 download FAILED"; exit 1; }
import os
from huggingface_hub import snapshot_download
sets = ["ifeval541", "math500", "gpqa", "aime", "humaneval", "mbpp", "gsm8k500"]
pats = ["logs/phase4/train/main/merged/*", "logs/phase4/train/main/holdout_pids.json", "logs/phase4/gate1/drafts/*"]
pats += [f"logs/phase4/decode/{arm}/{s}/*.jsonl" for arm in ("rd1", "rd2", "rd1_gate3", "vanilla_here") for s in sets]
snapshot_download("Arushhh/phase4-drafter-lora", repo_type="dataset", local_dir=".", allow_patterns=pats,
                  token=os.environ.get("HF_TOKEN"), max_workers=16)
# the 128M judge checkpoint WITH its optimizer state, for the judge round's --continue-from
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", local_dir="logs/c1b/train", allow_patterns=["main_128m/tok_000128000000/*"],
                  token=os.environ.get("HF_TOKEN"), max_workers=8)
PYEOF
for d in logs/phase4/decode/rd1/ifeval541 logs/phase4/decode/rd2/ifeval541 logs/phase4/decode/rd1_gate3/ifeval541 logs/phase4/decode/vanilla_here/ifeval541 logs/phase4/decode/vanilla_here/gsm8k500; do
  n=$(ls $d/*.jsonl 2>/dev/null | wc -l); say "$d: $n record files (expected 8)"; [ "$n" -ge 8 ] || exit 1
done
[ -f logs/c1b/train/main_128m/tok_000128000000/trainer_state.pt ] || { say "128M judge checkpoint (with optimizer) missing"; exit 1; }
[ "$(ls logs/phase4/gate1/drafts/drafts.shard*.npz | wc -l)" -ge 8 ] || { say "Phase 4 gate1 drafts missing"; exit 1; }
[ -f logs/phase4/train/main/merged/config.json ] || { say "merged drafter missing"; exit 1; }
$PY code/drafter_lora/check_merged.py logs/phase4/train/main/merged 2>&1 | tee -a $L/setup_lg.log || say "check_merged.py reported a problem: read the output above before continuing"

say "3/3 CPU tests (decode loop, re-draft, ban, loop guard)"
$PY -m pytest -q code/tests/test_da_decode_loopguard.py code/tests/test_da_decode_redraft.py code/tests/test_da_decode_full.py code/tests/test_drafter_lora.py 2>&1 | tail -3 | tee -a $L/setup_lg.log
[ ${PIPESTATUS[0]} = 0 ] || { say "tests FAILED: stop and report"; exit 1; }
say "setup done"
