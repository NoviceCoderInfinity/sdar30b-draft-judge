#!/usr/bin/env bash
# User 2026-09-30 ~20:00 IST: ~8.5 h more generation toward the 1B plan (same settings as wave B last night):
# all remaining wave-B code, then a RANDOM 115K of the remaining wave-B chat prompts, then one drafting pass
# (the wave-A flags 256 / 0.70), then conversion per domain into logs/c1b/trainset/waveB (appends).
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=$PWD/hf_home TOKENIZERS_PARALLELISM=false
C129="CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
GEN_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"
DRAFT_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
PYS=/venv/sgl/bin/python; PYT=/venv/train/bin/python
say(){ echo "[waveB-rest $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*"; }
per_gpu(){
  local name=$1; shift; local pids=()
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1 \
      || { sleep 30; CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1; } ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}
say "gen code start (all remaining wave-B code)"
per_gpu gen_waveB_code "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave B --domains code --gpu {K} --num-gpus 8" && say "gen code done" || say "gen code: a GPU failed twice"
say "gen chat start (random 115K subset)"
per_gpu gen_waveB_chat2 "SUBSET_PROMPT_DIR=/workspace/dmge/logs/c1b/gen/chat_subset_prompts $GEN_ENV $PYS logs/c1b/dl/gen_subset.py --wave B --domains chat --gpu {K} --num-gpus 8" && say "gen chat done" || say "gen chat: a GPU failed twice"
say "draft start"
per_gpu draft_waveB2 "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave B --gpu {K} --num-gpus 8 --max-running 256 --mem-fraction 0.70" && say "draft done" || say "draft: a GPU failed twice"
for d in code chat; do say "to_trainset $d"; $PYT code/corpus1b/to_trainset.py --wave B --domains $d > logs/c1b/trainset_waveB2_$d.log 2>&1 && say "to_trainset $d ok" || say "to_trainset $d FAILED"; done
say "WAVE B REST DONE"
