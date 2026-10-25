#!/usr/bin/env bash
# Stage 1 on the 8x RTX PRO 6000 box (~3 h): environments, the two PINNED stock models, datasets, prompt lists,
# eval sets, SGLang plugins, then the four go/no-go checks (gen, draft, train, speed).
# Re-runnable: every finished step leaves logs/c1b/stage1/.done_<step> and is skipped next time.
#   bash code/corpus1b/setup_box.sh 2>&1 | tee -a logs/c1b/setup.log
# Env knobs: SGL_FLAGS (extra gen/draft flags, e.g. "--attention-backend triton --disable-cuda-graph"),
#            SKIP_CHECKS=1 (build everything, run no GPU check), WORKERS (CPU workers, default nproc-4).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home}                  # datasets cache: on the big disk, inside the repo
export TOKENIZERS_PARALLELISM=false
mkdir -p logs/c1b/stage1 hf_cache/hub "$HF_HOME"
S=logs/c1b/stage1; done_(){ [ -f "$S/.done_$1" ]; }; mark(){ touch "$S/.done_$1"; }
say(){ echo "[setup $(date -u +%H:%M:%S)] $*"; }
PYT=/venv/train/bin/python; PYS=/venv/sgl/bin/python
WORKERS=${WORKERS:-$(( $(nproc) - 4 ))}
SGL_FLAGS=${SGL_FLAGS:-}

# ---- 0. the box -------------------------------------------------------------------------------------------
say "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee $S/gpus.txt
[ "$(nvidia-smi -L | wc -l)" -ge 8 ] || { say "need 8 GPUs"; exit 1; }
avail=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9); say "free disk ${avail}G"
[ "$avail" -ge 450 ] || { say "need >= 450 GB free under $ROOT"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || say "WARNING: no HF_TOKEN in .env (gated eval sets like Idavidrein/gpqa will be skipped)"

# ---- 1. environments --------------------------------------------------------------------------------------
if ! done_ env; then
  say "venv /venv/train (torch 2.11.0 cu128 + the round-3 freeze)"
  python3 -m venv /venv/train && $PYT -m pip install -q -U pip wheel
  $PYT -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
  if HF_HUB_OFFLINE=0 $PYT - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("Arushhh/dlm-exp1-drafter-voting-research-artifacts", repo_type="dataset",
                    filename="Nemotron-data-subset-training-diffusion-moe/strategy/env/venv_r3_freeze.txt")
import re, shutil
lines = [l for l in open(p) if l.strip() and not re.match(r"^(torch|nvidia-|triton)", l)]
open("/tmp/c1b_freeze.txt", "w").writelines(lines)
EOF
  then $PYT -m pip install -q -r /tmp/c1b_freeze.txt || $PYT -m pip install -q -r code/corpus1b/requirements_train.txt
  else $PYT -m pip install -q -r code/corpus1b/requirements_train.txt; fi
  $PYT -m pip install -q datasketch pytest sentence-transformers || true
  $PYT -c "import torch; a=torch.cuda.get_arch_list(); assert 'sm_120' in a, a; print('torch', torch.__version__, a)"
  say "venv /venv/sgl (sglang 0.5.10.post1)"
  python3 -m venv /venv/sgl && $PYS -m pip install -q -U pip wheel
  $PYS -m pip install -q "sglang[all]==0.5.10.post1"
  $PYS -m pip install -q -r code/corpus1b/requirements_sgl_extra.txt
  $PYS code/corpus1b/sglang_plugin/install.py
  for py in $PYT $PYS; do $py -m nltk.downloader -q -d /root/nltk_data punkt punkt_tab || true; done
  mark env
fi
$PYS code/corpus1b/sglang_plugin/install.py --check

# ---- 2. the two pinned stock models (block 32) --------------------------------------------------------------
if ! done_ models; then
  say "models: JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc, JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05"
  HF_HUB_OFFLINE=0 $PYT - <<'EOF'
import sys; sys.path.insert(0, "code")
import os; os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download
import research_protocol as P
for k in ("b32", "d32"):
    m = P.MODELS[k]
    d = snapshot_download(m["repo"], revision=m["revision"], cache_dir=m["cache_dir"])
    print(k, d)
EOF
  mark models
fi

# ---- 3. datasets (network) ------------------------------------------------------------------------------------
if ! done_ data; then
  say "datasets -> $HF_HOME"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $PYT code/corpus1b/download.py --include-optional
  mark data
fi

# ---- 4. prompt lists + eval sets (CPU, background) ------------------------------------------------------------
# Four domain groups build in parallel processes (chat must precede if: IF's composer excludes the chat prompts,
# so they share one process). Each group leaves .done_prompts_<group> and is skipped on a rerun; the manifest
# is merged under a lock. The spot-check runs once every group is done.
if ! done_ prompts; then
  GW=$(( WORKERS / 4 > 4 ? WORKERS / 4 : 4 ))
  say "prompt lists (background, 4 parallel groups x $GW workers) -> data/c1b/prompt_lists"
  (
    gp=()
    for g in chat,if maths science code; do
      t=${g//,/_}
      [ -f $S/.done_prompts_$t ] && continue
      ( $PYT code/corpus1b/build_prompt_lists.py --domains $g --workers "$GW" > $S/prompt_lists_$t.log 2>&1 \
          && touch $S/.done_prompts_$t ) & gp+=($!)
    done
    for p in "${gp[@]}"; do wait $p || true; done
    for t in chat_if maths science code; do [ -f $S/.done_prompts_$t ] || { echo "prompt group $t failed"; exit 1; }; done
    $PYT code/corpus1b/spotcheck.py > $S/prompt_lists.log 2>&1 && touch $S/.done_prompts
  ) &
  PROMPTS_PID=$!
else
  PROMPTS_PID=""
fi
if ! done_ evalsets; then
  say "eval sets -> data/c1b/eval"
  $PYT code/corpus1b/eval_sets.py --set all > $S/eval_sets.log 2>&1 && mark evalsets
fi

[ "${SKIP_CHECKS:-0}" = "1" ] && { say "SKIP_CHECKS=1: not running the GPU checks"; [ -n "$PROMPTS_PID" ] && wait $PROMPTS_PID; exit 0; }

# ---- 5. check GEN: SGLang vs the reference stock decoder (run_da_decode.py --stock) ---------------------------
if ! done_ check_gen; then
  say "check gen: reference stock decode (GPUs 0-3 gsm8k_pool:0:200, 4-7 opencode_pool:0:100)"
  pids=()
  for k in 0 1 2 3; do
    box=/dev/shm/c1b_sandbox/ref_gsm_$k; mkdir -p $box
    ( cd $box && CUDA_VISIBLE_DEVICES=$k $PYT $ROOT/code/draftalign/run_da_decode.py --model b32 --items gsm8k_pool:0:200 \
        --stock --shard $k --nshards 4 --gpu $k --out logs/c1b/parity/ref/gsm8k_pool --no-trace ) > $S/ref_gsm_$k.log 2>&1 & pids+=($!)
    box=/dev/shm/c1b_sandbox/ref_code_$k; mkdir -p $box
    ( cd $box && CUDA_VISIBLE_DEVICES=$((k+4)) $PYT $ROOT/code/draftalign/run_da_decode.py --model b32 --items opencode_pool:0:100 \
        --stock --shard $k --nshards 4 --gpu $((k+4)) --out logs/c1b/parity/ref/opencode_pool --no-trace ) > $S/ref_code_$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  say "check gen: SGLang on the same items"
  pids=()
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k $PYS code/corpus1b/gen_sglang.py --items gsm8k_pool:0:200 --gpu $k --num-gpus 4 \
      --out logs/c1b/parity/sglang $SGL_FLAGS > $S/sgl_gsm_$k.log 2>&1 & pids+=($!)
    CUDA_VISIBLE_DEVICES=$((k+4)) $PYS code/corpus1b/gen_sglang.py --items opencode_pool:0:100 --gpu $k --num-gpus 4 \
      --out logs/c1b/parity/sglang_code $SGL_FLAGS > $S/sgl_code_$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  # one tree holding both SGLang parity outputs (gpu0..7)
  rm -rf logs/c1b/parity/sglang_all; mkdir -p logs/c1b/parity/sglang_all
  i=0; for d in logs/c1b/parity/sglang/gpu* logs/c1b/parity/sglang_code/gpu*; do
    [ -d "$d" ] && cp -r "$d" logs/c1b/parity/sglang_all/gpu$i && i=$((i+1)); done
  $PYT code/corpus1b/parity_check.py gen --ref-dir logs/c1b/parity/ref --sgl-dir logs/c1b/parity/sglang_all \
    && mark check_gen || { say "CHECK GEN FAILED (see $S/report.jsonl)"; exit 2; }
fi

# ---- 6. check DRAFT: SGLang drafts vs drafter.py::draft_block ----------------------------------------------------
if ! done_ check_draft; then
  say "check draft"
  CUDA_VISIBLE_DEVICES=0 $PYS code/corpus1b/draft_sglang.py --wave A --gpu 0 --num-gpus 1 \
    --gen logs/c1b/parity/sglang_all --out logs/c1b/parity/draft $SGL_FLAGS > $S/draft_parity.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PYT code/corpus1b/parity_check.py draft --parity-gen logs/c1b/parity/sglang_all \
    --parity-draft logs/c1b/parity/draft && mark check_draft || { say "CHECK DRAFT FAILED"; exit 2; }
fi

# ---- 7. check TRAIN: two-stream trainer vs the round-3 path on the real 30B ----------------------------------------
if ! done_ check_train; then
  say "check train"
  $PYT code/corpus1b/to_trainset.py --wave A --gen logs/c1b/parity/sglang_all --draft logs/c1b/parity/draft \
    --out logs/c1b/parity/trainset > $S/parity_trainset.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PYT code/corpus1b/parity_check.py train --index logs/c1b/parity/trainset/index.jsonl \
    && mark check_train || { say "CHECK TRAIN FAILED"; exit 2; }
fi

# ---- 8. check SPEED on the real wave-A mix (needs the prompt lists) ------------------------------------------------
[ -n "$PROMPTS_PID" ] && { say "waiting for the prompt lists"; wait $PROMPTS_PID || { say "PROMPT LISTS FAILED ($S/prompt_lists*.log)"; exit 2; }; }
done_ prompts || { say "prompt lists missing ($S/prompt_lists.log)"; exit 2; }
if ! done_ check_speed; then
  say "check speed: 1,500 wave-A prompts per GPU in their natural mix"
  pids=()
  for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k $PYS code/corpus1b/gen_sglang.py --wave A --gpu $k --num-gpus 8 --limit-total 1500 \
      --out logs/c1b/speed --no-check $SGL_FLAGS > $S/speed_$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  $PYT code/corpus1b/parity_check.py speed --gen-dir logs/c1b/speed && mark check_speed \
    || { say "CHECK SPEED FAILED: generation would exceed the plan's budget ($S/report.jsonl)"; exit 2; }
fi
say "STAGE 1 PASSED: all four checks. Next: bash code/corpus1b/run_wave_a.sh"
