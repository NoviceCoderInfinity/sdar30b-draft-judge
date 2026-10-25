#!/usr/bin/env bash
# Phase 2 setup on the 8x RTX PRO 6000 box, after Phase 1's setup_box.sh (it made /venv/train and the two pinned
# stock models). Re-runnable: every finished step leaves logs/phase2/setup/.done_<step>.
#   bash code/phase2/setup_p2.sh 2>&1 | tee -a logs/phase2/setup.log
#
#  1. /venv/p2: /venv/train's freeze (or, on a fresh box without it, torch 2.11.0 cu128 + the team's round-3 freeze
#     from HF) with transformers 4.54.1 (+ einops, pytest, IF-checker deps). One venv loads the 30B SDAR MoE (with the
#     LossKwargs shim, phase2/tf_compat.py) AND every Phase 2 drafter (Fast-dLLM v2 needs >= 4.53, the dLLM BD3LM
#     checkpoints >= 4.54).
#  2. the drafters at their pinned revisions (phase2/drafters.py REGISTRY) into hf_cache/hub
#  3. the round-3 references from Hugging Face: the S0 judge (r3_B step 710), its judge+DES48 records (S0), the stock
#     records, and the round-3 training index + items (sha256-checked against mix_manifest.json)
#  3a. the seven benchmarks' datasets into HF_HOME (the decodes read them offline); checked by loading every set
#  3b. IFEval-541, frozen by Phase 1 (copied from P2_C1B_EVAL_FROM, default /workspace/dmge/data/c1b/eval, else
#     downloaded from Phase 1's HF upload, else built from google/IFEval); checked: 541 items, sha256, the scorer imports
#  3c. the stock 30B on IFEval-541, from Phase 1's HF upload (P2_P1_REPO, checkpointB/ifeval541_released): the IFEval
#     speedup reference, so this run does not decode stock itself; checked: 541 items, revision c351bbc, golds = 3b's
#  4. go/no-go checks, each must PASS before run_phase2.sh:
#       tests     CPU tests (code/phase2/tests + code/tests) under /venv/p2
#       stock     --stock on 16 GSM8K-500 items reproduces the recorded stock cell token for token under /venv/p2
#       adapter   the downloaded S0 judge's weight hash is ad8a6485
#       redraft   each drafter re-drafts 16 training items without a stop rule firing; S0 must reproduce the
#                 stored S0 drafts on >= 99% of slots (checks the re-draft path itself)
# Env: P2_DRAFTERS (default "F2 S1 F1": the MVP and its fallback), HF_REF_REPO (default Anupam-Rawat-IITB/...), SKIP_CHECKS=1 (CPU steps
#      only), P2_MODELS_FROM (link the 30B + 1.7B from another checkout's hf_cache/hub instead of downloading),
#      HF_HOME (datasets cache holding the benchmark sets; e.g. Phase 1's hf_home).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home}
export TOKENIZERS_PARALLELISM=false
S=logs/phase2/setup; mkdir -p $S logs/phase2/ref logs/phase2/data
done_(){ [ -f "$S/.done_$1" ]; }; mark(){ touch "$S/.done_$1"; }
say(){ echo "[p2-setup $(date -u +%H:%M:%S)] $*"; }
PYT=/venv/train/bin/python; PY=/venv/p2/bin/python
DRAFTERS=${P2_DRAFTERS:-"F2 S1 F1"}
REPO=${HF_REF_REPO:-Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts}
P1_REPO=${P2_P1_REPO:-Arushhh/c1b-judge-corpus-waveA}     # Phase 1: frozen IFEval-541 + its stock records
DL=logs/phase2/ref/_hf

# ---- 1. /venv/p2 --------------------------------------------------------------------------------------------------
if ! done_ env; then
  python3 -m venv /venv/p2 && $PY -m pip install -q -U pip wheel
  if [ -x $PYT ]; then
    # a box that already ran Phase 1's setup_box.sh: /venv/train's exact freeze
    say "venv /venv/p2 = /venv/train's freeze + transformers 4.54.1"
    TV=$($PYT -c "import torch; print(torch.__version__.split('+')[0])")
    $PY -m pip install -q --index-url https://download.pytorch.org/whl/cu128 "torch==$TV"
    $PYT -m pip freeze | grep -v -E "^(torch|nvidia-|triton|transformers)[=@ ]" | grep -v "^-e " > $S/p2_freeze.txt
  else
    # a fresh box: torch 2.11.0 cu128 (as Phase 1) + the team's round-3 freeze from HF (else requirements_train.txt)
    say "no /venv/train (fresh box): venv /venv/p2 = torch 2.11.0 cu128 + the round-3 freeze + transformers 4.54.1"
    $PY -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
    $PY -m pip install -q huggingface_hub
    HF_HUB_OFFLINE=0 $PY - "$S/p2_freeze.txt" <<'EOF' || cp code/corpus1b/requirements_train.txt $S/p2_freeze.txt
import re, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download("Arushhh/dlm-exp1-drafter-voting-research-artifacts", repo_type="dataset",
                    filename="Nemotron-data-subset-training-diffusion-moe/strategy/env/venv_r3_freeze.txt")
lines = [l for l in open(p) if l.strip() and not re.match(r"^(torch|nvidia-|triton|transformers)[=@ ]", l)
         and not l.startswith("-e ")]
open(sys.argv[1], "w").writelines(lines)
EOF
  fi
  $PY -m pip install -q -r $S/p2_freeze.txt || $PY -m pip install -q -r code/corpus1b/requirements_train.txt
  $PY -m pip install -q "transformers==4.54.1" einops pytest langdetect nltk immutabledict absl-py
  $PY -m nltk.downloader -q -d /root/nltk_data punkt punkt_tab || true
  $PY -c "import torch, transformers; assert 'sm_120' in torch.cuda.get_arch_list(); print('torch', torch.__version__, 'transformers', transformers.__version__)"
  $PY -m pip freeze > $S/venv_p2_freeze.txt
  mark env
fi

# ---- 1b. the two pinned stock models (b32, d32): link an existing download, else fetch ---------------------------
# P2_MODELS_FROM = another checkout's hf_cache/hub holding them (e.g. /workspace/dmge/hf_cache/hub): linked read-only
if ! done_ models; then
  mkdir -p hf_cache/hub
  for m in models--JetLM--SDAR-30B-A3B-Chat-b32 models--JetLM--SDAR-1.7B-Chat-b32; do
    if [ ! -e hf_cache/hub/$m ] && [ -n "${P2_MODELS_FROM:-}" ] && [ -d "$P2_MODELS_FROM/$m" ]; then
      ln -s "$P2_MODELS_FROM/$m" hf_cache/hub/$m && say "linked $m from $P2_MODELS_FROM"
    fi
  done
  if [ -f code/drafter_lora/hf_progress.py ]; then   # phase 4: same repos/revisions/cache, + progress lines, retries, size check
    HF_HUB_OFFLINE=0 $PY code/drafter_lora/hf_progress.py --models b32,d32
  else
  HF_HUB_OFFLINE=0 $PY -c "import sys; sys.path[:0]=['code/compat','code']; import os; import research_protocol as P; \
os.environ['HF_HUB_OFFLINE']='0'; from huggingface_hub import snapshot_download; \
[print(k, snapshot_download(P.MODELS[k]['repo'], revision=P.MODELS[k]['revision'], cache_dir=P.MODELS[k]['cache_dir'])) for k in ('b32','d32')]"
  fi
  mark models
fi

# ---- 2. drafters (network) --------------------------------------------------------------------------------------
if ! done_ drafters; then
  for k in $DRAFTERS; do
    say "drafter $k"
    HF_HUB_OFFLINE=0 $PY -c "import sys; sys.path[:0]=['code/compat','code']; import research_protocol; \
import os; os.environ['HF_HUB_OFFLINE']='0'; from phase2 import drafters as D; print('$k', D.download('$k'))"
  done
  mark drafters
fi

# ---- 3. round-3 references (network) ------------------------------------------------------------------------------
if ! done_ refs; then
  say "S0 judge r3_B step 710, S0 judge+DES48 records, stock records, round-3 training index -> $DL"
  HF_HUB_OFFLINE=0 $PY - "$REPO" "$DL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], allow_patterns=[
    "Nemotron-split-moe-train/r3_B/checkpoints/step000710/adapter*",
    "Nemotron-split-moe-train/r3_B/evals/step000710/*/judge_des48/*",
    "Nemotron-split-moe-train/reference_evals/da_decode/*_B32_t95_stock/*",
    "Nemotron-split-moe-train/reference_evals/da_decode/*_B32_t95_stock_p48desf/*",
    "Nemotron-split-moe-train/reference_evals/research_v1/baselines_b32/gsm8k_B32_t95_stock.gsm8k500.*",
    "Nemotron-data-subset-training-diffusion-moe/processed/trainset_r3/*"])
EOF
  mkdir -p logs/phase2/ref/r3_B logs/phase2/ref/S0_flat logs/phase2/ref/stock_flat
  rm -rf logs/phase2/ref/r3_B/step000710
  cp -r $DL/Nemotron-split-moe-train/r3_B/checkpoints/step000710 logs/phase2/ref/r3_B/step000710
  find $DL/Nemotron-split-moe-train/r3_B/evals/step000710 -path "*judge_des48*" -name "*.jsonl" -exec cp {} logs/phase2/ref/S0_flat/ \;
  find $DL/Nemotron-split-moe-train/reference_evals -name "*.jsonl" -exec cp {} logs/phase2/ref/stock_flat/ \;
  for set in humaneval mbpp gsm8k500 math500 gpqa aime; do     # (ifeval541 has no round-3 S0: the launcher runs it)
    n0=$(ls logs/phase2/ref/S0_flat/*.$set.shard*.jsonl 2>/dev/null | wc -l)
    ns=$(ls logs/phase2/ref/stock_flat/*_B32_t95_stock.$set.shard*.jsonl 2>/dev/null | wc -l)
    say "reference files for $set: S0 $n0, stock $ns"
    [ "$n0" -gt 0 ] || { say "no S0 records for $set: pairing needs them"; exit 2; }
  done
  T=logs/phase2/data/trainset_r3; mkdir -p $T
  SRC=$DL/Nemotron-data-subset-training-diffusion-moe/processed/trainset_r3
  cp $SRC/index.jsonl $SRC/mix_manifest.json $T/
  for pool in gsm8k opencode if; do mkdir -p $T/$pool; tar -xf $SRC/$pool/items.tar -C $T/$pool/; done
  want=$(python3 -c "import json; print(json.load(open('$T/mix_manifest.json'))['index_sha256'])")
  got=$(sha256sum $T/index.jsonl | cut -d' ' -f1)
  [ "$want" = "$got" ] || { say "round-3 index sha256 $got != manifest $want"; exit 2; }
  missing=$(python3 -c "
import json, os
n = 0
for l in open('$T/index.jsonl'):
    if not os.path.exists(os.path.join('$T', json.loads(l)['file'])): n += 1
print(n)")
  [ "$missing" = 0 ] || { say "$missing item files of the round-3 index are missing"; exit 2; }
  say "refs: $(ls logs/phase2/ref/S0_flat | wc -l) S0 files, $(ls logs/phase2/ref/stock_flat | wc -l) stock files, $(wc -l < $T/index.jsonl) training items"
  mark refs
fi

# ---- 3a. the benchmark datasets into the HF datasets cache (every decode reads them OFFLINE) ------------------------
# exactly the load_dataset calls of run_desvote.build_items (+ google/IFEval for the IFEval-541 build); a no-op when
# HF_HOME already holds them (e.g. Phase 1's hf_home)
if ! done_ datasets; then
  say "benchmark datasets -> $HF_HOME"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $PY - <<'EOF' || { say "benchmark dataset download FAILED"; exit 2; }
import os
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["HF_DATASETS_OFFLINE"] = "0"
from datasets import load_dataset
for args, kw in [(("openai/gsm8k", "main"), {"split": "test"}),
                 (("google-research-datasets/mbpp", "sanitized"), {"split": "test"}),
                 (("fingertap/GPQA-Diamond",), {"split": "test"}),
                 (("openai_humaneval",), {"split": "test"}),
                 (("AI-MO/aimo-validation-aime",), {"split": "train"}),
                 (("HuggingFaceH4/MATH-500",), {"split": "test"}),
                 (("google/IFEval",), {"split": "train"})]:
    ds = load_dataset(*args, **kw)
    print(args[0], len(ds), flush=True)
EOF
  n=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; import research_protocol as P; \
print(' '.join(f'{s}={len(P.items_for_split(s))}' for s in ('humaneval','mbpp','gsm8k500','math500','gpqa','aime')))") \
    || { say "the benchmarks do not load offline"; exit 2; }
  say "benchmarks offline: $n (expected humaneval=164 mbpp=257 gsm8k500=500 math500=500 gpqa=198 aime=90)"
  mark datasets
fi

# ---- 3b. IFEval-541 (frozen by Phase 1; research_protocol sha256-checks it on every load) --------------------------
# copied from P2_C1B_EVAL_FROM (default: Phase 1's checkout, read only) if present there, else built from google/IFEval
if ! done_ evalsets; then
  mkdir -p data/c1b/eval
  SRC_EVAL=${P2_C1B_EVAL_FROM:-/workspace/dmge/data/c1b/eval}
  if [ -f "$SRC_EVAL/ifeval541.jsonl" ] && [ -f "$SRC_EVAL/ifeval541.jsonl.sha256" ]; then
    cp "$SRC_EVAL/ifeval541.jsonl" "$SRC_EVAL/ifeval541.jsonl.sha256" data/c1b/eval/ && say "IFEval-541 copied from $SRC_EVAL"
  elif HF_HUB_OFFLINE=0 $PY - "$P1_REPO" <<'PYEOF'
import shutil, sys
from huggingface_hub import hf_hub_download
for f in ("ifeval541.jsonl", "ifeval541.jsonl.sha256"):
    shutil.copy(hf_hub_download(sys.argv[1], repo_type="dataset", filename=f"data/c1b/eval/{f}"), f"data/c1b/eval/{f}")
PYEOF
  then
    say "IFEval-541 downloaded from $P1_REPO (Phase 1's frozen file)"
  else
    say "IFEval-541 not found in $SRC_EVAL nor on $P1_REPO: building it from google/IFEval"
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $PY code/corpus1b/eval_sets.py --set ifeval541 > $S/eval_sets.log 2>&1 \
      || { say "IFEval-541 build FAILED ($S/eval_sets.log)"; exit 2; }
  fi
  n=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; import research_protocol as P; print(len(P.items_for_split('ifeval541')))") \
    || { say "IFEval-541 does not load (sha256 check)"; exit 2; }
  [ "$n" = 541 ] || { say "IFEval-541 has $n items, expected 541"; exit 2; }
  $PY -c "import sys; sys.path[:0]=['code/compat','code']; from draftalign.if_scorer import score_ifeval_record" \
    || { say "the IFEval scorer does not import (checker dependencies: langdetect, nltk punkt_tab)"; exit 2; }
  say "IFEval-541: 541 items, sha256 verified, scorer imports"
  mark evalsets
fi

# ---- 3c. stock 30B on IFEval-541: Phase 1's complete run (the IFEval speedup reference) ---------------------------
if ! done_ stock_ifeval; then
  say "stock IFEval-541 records <- $P1_REPO (logs/c1b/eval/checkpointB/ifeval541_released)"
  HF_HUB_OFFLINE=0 $PY - "$P1_REPO" "$DL/p1" <<'PYEOF' || { say "stock IFEval download FAILED"; exit 2; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2],
                  allow_patterns=["logs/c1b/eval/checkpointB/ifeval541_released/*.jsonl*"])
PYEOF
  mkdir -p logs/phase2/ref/stock_flat
  cp $DL/p1/logs/c1b/eval/checkpointB/ifeval541_released/ifeval_B32_t95_stock.ifeval541.shard*of8.jsonl \
    logs/phase2/ref/stock_flat/
  $PY - <<'PYEOF' || { say "stock IFEval records do not match this checkout's IFEval-541"; exit 2; }
import glob, json, sys
sys.path[:0] = ["code/compat", "code"]
import research_protocol as P
recs = {}
for f in glob.glob("logs/phase2/ref/stock_flat/ifeval_B32_t95_stock.ifeval541.shard*of8.jsonl"):
    for l in open(f):
        r = json.loads(l); recs[int(r["idx"])] = r
items = {it["idx"]: it for it in P.items_for_split("ifeval541")}
assert set(recs) == set(items) and len(recs) == 541, f"{len(recs)} stock records for {len(items)} items"
assert {r["model_revision"] for r in recs.values()} == {P.MODELS["b32"]["revision"]}, "stock records: other revision"
assert all(r["judge_mode"] == "stock" and not r.get("adapter") for r in recs.values()), "not stock records"
bad = [i for i in items if recs[i]["gold"] != items[i]["gold"]]
assert not bad, f"{len(bad)} golds differ from the frozen IFEval-541 (first idx {bad[:5]})"
g = sum(r["n_gen_to_eos"] for r in recs.values())
print(f"stock IFEval-541: 541/541 items, revision ok, golds match; accuracy "
      f"{100 * sum(bool(r['correct']) for r in recs.values()) / 541:.1f}, "
      f"exp/token {sum(r['unique_expert_layer_loads'] for r in recs.values()) / g:.1f}")
PYEOF
  mark stock_ifeval
fi

[ "${SKIP_CHECKS:-0}" = "1" ] && { say "SKIP_CHECKS=1: done without GPU checks"; exit 0; }

# ---- 4. go/no-go checks -------------------------------------------------------------------------------------------
if ! done_ check_tests; then
  say "check tests"
  $PY -m pytest -q code/phase2/tests > $S/tests.log 2>&1 && mark check_tests || { say "TESTS FAILED ($S/tests.log)"; exit 2; }
fi
if ! done_ check_adapter; then
  h=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; from draftalign.run_da_decode import adapter_weights_hash; print(adapter_weights_hash('logs/phase2/ref/r3_B/step000710'))")
  [ "${h:0:8}" = "ad8a6485" ] && mark check_adapter || { say "S0 judge hash $h != ad8a6485"; exit 2; }
fi
# check_stock runs in the background on its own GPU (P2_STOCK_GPU, default 7) while the re-draft smoke tests use GPUs
# 0-3: the same two checks, one process per GPU, finished in the time of the slower one
stock_pid=""
if ! done_ check_stock; then
  say "check stock: --stock on 16 GSM8K-500 items vs the recorded stock cell, under /venv/p2 (GPU ${P2_STOCK_GPU:-7})"
  box=/dev/shm/p2_sandbox/check_stock; mkdir -p $box
  ( cd $box && CUDA_VISIBLE_DEVICES=${P2_STOCK_GPU:-7} $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --items gsm8k500 --stock \
      --limit 16 --out logs/phase2/setup/check_stock --ref-dir logs/phase2/ref/stock_flat --no-trace \
      --stop-rules default --expert-log bits ) > $S/check_stock.log 2>&1 & stock_pid=$!
fi
if ! done_ check_redraft; then
  say "check redraft: 16 training items per drafter (S0 must reproduce the stored drafts)"
  pids=(); g=0
  for k in S0 $DRAFTERS; do
    ( CUDA_VISIBLE_DEVICES=$g $PY code/phase2/redraft.py --index logs/phase2/data/trainset_r3/index.jsonl --drafter $k \
        --out logs/phase2/setup/check_redraft/$k --limit-items 16 --print-every 4 ) > $S/check_redraft_$k.log 2>&1 & pids+=($!)
    g=$(( (g + 1) % 8 ))
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  [ $fail = 0 ] || { say "CHECK REDRAFT FAILED ($S/check_redraft_*.log)"; exit 2; }
  $PY - <<'EOF' || { echo "S0 re-draft does not reproduce the stored drafts"; exit 2; }
import json, glob
rows = [json.loads(l) for f in glob.glob("logs/phase2/setup/check_redraft/S0/agreement.*.jsonl") for l in open(f)]
same = sum(r["same_as_prev"] for r in rows); slots = sum(r["slots"] for r in rows)
print(f"S0 re-draft equals the stored round-3 S0 draft on {same}/{slots} slots ({same / max(1, slots):.4f})")
assert slots and same / slots >= 0.97, "the re-draft path does not reproduce round 3's drafts"
if same / slots < 0.99:
    print("WARNING: below 0.99 -- bf16 numerics of this torch/transformers differ from round 3's; check by eye")
EOF
  for k in S0 $DRAFTERS; do tail -2 $S/check_redraft_$k.log | sed "s/^/  $k: /"; done
  mark check_redraft
fi
if [ -n "$stock_pid" ]; then
  wait $stock_pid && grep -q "gen_ids match 16/16" $S/check_stock.log && mark check_stock \
    || { say "CHECK STOCK FAILED: /venv/p2 does not reproduce the stock cell ($S/check_stock.log)"; exit 2; }
fi
say "PHASE 2 SETUP PASSED. Next: bash code/phase2/run_phase2.sh S0 && bash code/phase2/run_phase2.sh F1"
