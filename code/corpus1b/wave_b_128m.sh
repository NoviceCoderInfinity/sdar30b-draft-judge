#!/usr/bin/env bash
# User 2026-09-29 ~20:00 IST (33 h top-up): extend the corpus from 74.76M to ~128M tokens at the plan's mix, then train.
# Wave-B generation per domain with the wave-A settings (docs/DATA_GENERATION_CONTINUATION.md 4.3). Science and IF: all
# remaining wave-B prompts. Maths and chat: time-boxed with --stop-at (rows are processed in hash order, so a time box
# is an unbiased sample of every source; --limit is NOT used because it takes the first rows in list order = one source).
# Then one drafting pass over every kept record, then training items per domain (logs/c1b/trainset/waveB).
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=$PWD/hf_home TOKENIZERS_PARALLELISM=false
C129="CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
GEN_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"
DRAFT_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
PYS=/venv/sgl/bin/python; PYT=/venv/train/bin/python
mkdir -p logs/c1b/gen/waveB logs/c1b/draft/waveB
say(){ echo "[waveB-128m $(TZ=Asia/Kolkata date +%H:%M:%S) IST] $*"; }
per_gpu(){
  local name=$1; shift; local pids=()
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1 \
      || { sleep 30; CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1; } ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}
gen(){ local d=$1 box=$2 extra=""; [ -n "$box" ] && extra="--stop-at $(( $(date +%s) + box*60 ))"
  say "gen $d start ${box:+(time box ${box} min)}"
  per_gpu gen_waveB_$d "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave B --domains $d --gpu {K} --num-gpus 8 $extra" \
    && say "gen $d done" || say "gen $d: a GPU failed twice (see logs/c1b/gen_waveB_$d.gpu*.log)"; }
gen science ""
gen if ""
gen maths 68
gen chat 36
say "draft start"
per_gpu draft_waveB "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave B --gpu {K} --num-gpus 8 --max-running 256 --mem-fraction 0.70" \
  && say "draft done" || say "draft: a GPU failed twice (see logs/c1b/draft_waveB.gpu*.log)"
for d in science if maths chat; do
  say "to_trainset $d"; $PYT code/corpus1b/to_trainset.py --wave B --domains $d > logs/c1b/trainset_waveB_$d.log 2>&1 && say "to_trainset $d ok" || say "to_trainset $d FAILED"
done
say "WAVE B 128M DONE"
