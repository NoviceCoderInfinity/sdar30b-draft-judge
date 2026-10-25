#!/usr/bin/env bash
# 128M -> 228M continuation, step 1 of 2: set up a FRESH 8x RTX PRO 6000 box (~1-1.5 h). Re-runnable: every finished
# step leaves logs/c1b/c228_state/.done_<step> and is skipped next time. Run from the repo root (/workspace/dmge):
#   bash code/corpus1b/ops/c228_setup.sh 2>&1 | tee -a logs/c1b/c228_setup.log
# Follows docs/DATA_GENERATION_CONTINUATION.md section 4 (new box): CUDA 12.9 toolkit beside the image's CUDA (FlashInfer
# JIT on sm_120), the two venvs + pinned models via setup_box.sh with the prompt-list/datasets steps skipped (the lists
# are final and come from HF), then only what this run needs from HF (~8 GB): prompt lists (+ late-decontam flags),
# eval sets, wave-B generation/draft state, the training indexes + item tars, the 128M checkpoint. Then a smoke test:
# SGLang GSM8K-500 on 8 GPUs must give accuracy 90.2 +- 3 and passes/block 10.06 +- 5% (the Phase 1 numbers).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
[ -f .env ] || { echo "no .env in $ROOT (write it first; see docs/C228_CONTINUE.md)"; exit 1; }
git check-ignore -q .env || { echo ".env is not git-ignored: refusing"; exit 1; }
set -a; . ./.env; set +a
export HF_HOME=$ROOT/hf_home TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
S=logs/c1b/c228_state; mkdir -p $S logs/c1b/stage1 hf_cache/hub "$HF_HOME"
say(){ echo "[c228-setup $(date -u +%H:%M:%S) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST] $*"; }
done_(){ [ -f "$S/.done_$1" ]; }; mark(){ touch "$S/.done_$1"; }

# ---- 0. the box ------------------------------------------------------------------------------------------------
say "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee $S/gpus.txt
[ "$(nvidia-smi -L | wc -l)" -ge 8 ] || { say "need 8 GPUs"; exit 1; }
say "CPU: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//') x $(nproc) threads; RAM $(free -g | awk '/Mem:/{print $2}') GB"
avail=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9); say "free disk ${avail}G"
[ "$avail" -ge 450 ] || { say "need >= 450 GB free under $ROOT (setup_box.sh checks the same)"; exit 1; }

# ---- 1. CUDA 12.9 toolkit (nvcc + cudart only, never a driver) -----------------------------------------------------
if [ ! -x /usr/local/cuda-12.9/bin/nvcc ]; then
  say "installing cuda-nvcc-12-9 + cuda-cudart-dev-12-9"
  export DEBIAN_FRONTEND=noninteractive
  if ! apt-cache policy cuda-nvcc-12-9 2>/dev/null | grep -q Candidate; then
    . /etc/os-release; UB="ubuntu${VERSION_ID//./}"
    curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${UB}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq
  fi
  apt-get install -y -qq cuda-nvcc-12-9 cuda-cudart-dev-12-9
fi
/usr/local/cuda-12.9/bin/nvcc --version | tail -1

# ---- 2. venvs + pinned models via setup_box.sh (prompt lists, source datasets and eval-set builds skipped) ---------
touch logs/c1b/stage1/.done_prompts logs/c1b/stage1/.done_prompts_chat_if logs/c1b/stage1/.done_prompts_maths \
      logs/c1b/stage1/.done_prompts_science logs/c1b/stage1/.done_prompts_code logs/c1b/stage1/.done_data \
      logs/c1b/stage1/.done_evalsets
SKIP_CHECKS=1 bash code/corpus1b/setup_box.sh
/venv/train/bin/python -c "import torch; a=torch.cuda.get_arch_list(); assert 'sm_120' in a, a; print('torch', torch.__version__, 'sm_120 ok')"

# ---- 3. this run's data from Hugging Face (~8 GB) ----------------------------------------------------------------
if ! done_ hfdata; then
  say "HF: prompt lists, eval sets, wave-B state, trainset indexes + item tars, main_128m logs, 128M checkpoint"
  /venv/train/bin/python - <<'EOF'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir=".", token=tok, max_workers=16,
                  allow_patterns=["data/c1b/prompt_lists/*", "data/c1b/eval/**",
                                  "logs/c1b/gen/waveB/**", "logs/c1b/draft/waveB/**",
                                  "logs/c1b/trainset/waveA/index.jsonl", "logs/c1b/trainset/waveA/build_summary.json",
                                  "logs/c1b/trainset/waveA/items_tar/*", "logs/c1b/trainset/waveB/index.jsonl",
                                  "logs/c1b/trainset/waveB/build_summary.json", "logs/c1b/trainset/waveB/items_tar/*",
                                  "logs/c1b/trainset/index_128m.jsonl", "logs/c1b/trainset/holdout_pids_waveA.txt",
                                  "logs/c1b/train/main_128m/*", "logs/c1b/train/main/config.json"])
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", local_dir="logs/c1b/train", token=tok,
                  allow_patterns=["main_128m/tok_000128000000/*", "main_128m/config.json"])
EOF
  mark hfdata
fi
if ! done_ items; then
  say "unpacking item tars"
  for w in waveA waveB; do for t in logs/c1b/trainset/$w/items_tar/*.tar; do tar -xf "$t" -C logs/c1b/trainset/$w; done; done
  for w in waveA waveB; do say "$w items: $(find logs/c1b/trainset/$w/items -name '*.npz' | wc -l) npz, index $(wc -l < logs/c1b/trainset/$w/index.jsonl) rows"; done
  mark items
fi
# the lists must be the exact Phase 1 lists (docs/DATA_GENERATION_CONTINUATION.md 4.2)
sha256sum -c - <<'EOF'
dc4a63ffbad0251e5214be466697315804e1d72a05b1172b430afc545e1fec95  data/c1b/prompt_lists/maths.jsonl
b1eee9e86cbef0167672c85b2676b77652edd0e556004fb15d3a453c955ed715  data/c1b/prompt_lists/science.jsonl
211842669f44988f9127e204f5184fe1e66d12fa0341a4d61a1411e28de68270  data/c1b/prompt_lists/code.jsonl
18a293f4b6598dbca1205dde08297c8e8a40fad4058fc368a873e295bb9d6167  data/c1b/prompt_lists/if.jsonl
c39cbc6c8fa5e1a519d59df317f73b47cf2610f58513b578fd5d3b4e20295190  data/c1b/prompt_lists/chat.jsonl
EOF
ls data/c1b/prompt_lists/postdecontam_*.json >/dev/null || { say "postdecontam_*.json missing (to_trainset needs them)"; exit 1; }
test -f logs/c1b/train/main_128m/tok_000128000000/trainer_state.pt || { say "128M checkpoint missing"; exit 1; }
say "128M checkpoint: $(cat logs/c1b/train/main_128m/tok_000128000000/trainer_state.json | tr -d '\n' | cut -c1-120)"

# ---- 4. smoke: SGLang (FlashInfer + CUDA graphs) reproduces the Phase 1 stock decode on GSM8K-500 ------------------
if ! done_ smoke; then
  say "smoke: SGLang GSM8K-500 on 8 GPUs (accuracy 90.2 +- 3, passes/block 10.06 +- 5%)"
  rm -rf logs/c1b/parity/c228_smoke; mkdir -p logs/c1b/parity/c228_smoke
  t0=$(date +%s); pids=()
  for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k env CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH \
      SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648 /venv/sgl/bin/python code/corpus1b/gen_sglang.py --items gsm8k500 \
      --gpu $k --num-gpus 8 --out logs/c1b/parity/c228_smoke > logs/c1b/parity/c228_smoke_gpu$k.log 2>&1 & pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  say "smoke decode rc=$rc in $(( $(date +%s) - t0 )) s"
  /venv/train/bin/python - <<'PY'
import glob, json, sys
sys.path[:0] = ["code/compat", "code", "code/corpus1b"]
import common as C, parity_check as PC, research_protocol as P
rows = [r for f in sorted(glob.glob("logs/c1b/parity/c228_smoke/gpu*/records.*.jsonl.gz")) for r in C.read_jsonl(f)]
ok = [r for r in rows if r.get("kept")]
sc = PC.score(PC.family(P.split_base("gsm8k500")), ok)
acc = 100.0 * sum(sc.values()) / max(1, len(sc))
ppb = PC.ppb_weighted([r["ppb"] for r in ok])
passed = len(ok) == 500 and abs(acc - 90.2) <= 3.0 and abs(ppb - 10.06) / 10.06 <= 0.05
rep = {"n": len(rows), "kept": len(ok), "acc": round(acc, 2), "ppb": round(ppb, 3), "PASS": passed}
print("[smoke] " + json.dumps(rep), flush=True)
json.dump(rep, open("logs/c1b/c228_state/smoke.json", "w"))
sys.exit(0 if passed else 1)
PY
  mark smoke
fi
say "SETUP DONE. Next: write /workspace/c228_deadline.env (END=<unix>, MODE=full|box), then run c228_run.sh in tmux"
