#!/usr/bin/env bash
# Post-training eval for r3_B (user, 2026-09-15): wait for r3_B training to exit, push its checkpoints to HF right away,
# then run ONE priority queue on all 8 GPUs with a hard cutoff (default 17:30 IST; the server goes off at 18:00 IST):
#   1. r3_B final checkpoint, judge + DES48, full sets: HumanEval-164, MBPP-257, MATH-500, GPQA-Diamond-198, GSM8K-500, AIME-90
#   2. leftover time: the missing stock + DES baselines, shortest first -- AIME DES48, GPQA DES48, AIME DES32, GPQA DES32
# Decode + scoring are identical to eval_wave_r3.sh: code sets decode without --score from a sandbox cwd and are scored
# by run_score_r3.sh as the unprivileged scorer; GSM8K/MATH/GPQA/AIME score inline (no code execution).
#   nohup bash code/draftalign/eval_after_r3B.sh [17:30] > logs/draftalign/eval_after_r3B.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
CUT=$(TZ=Asia/Kolkata date -d "today ${1:-17:30}" +%s)
PY=/venv/r3/bin/python; RUN=$ROOT/logs/draftalign/train/r3_B; CK=$ROOT/logs/draftalign/eval_ckpts
PRB=$ROOT/logs/draftalign/probes_r3/r3_B; PRA=$ROOT/logs/draftalign/probes_r3/r3_A
Q=$ROOT/logs/draftalign/queue_eval_r3B.txt; LOG=$ROOT/logs/draftalign/eval_r3B.log
ist() { TZ=Asia/Kolkata date '+%H:%M:%S IST'; }
echo "$(ist) waiting for r3_B training to exit (cutoff $(TZ=Asia/Kolkata date -d @$CUT '+%H:%M IST'))"

# ---- 1. wait for training to exit
while pgrep -f "draftalign/train[.]py.*r3_B" >/dev/null; do sleep 30; done
ST=$(ls -d "$RUN"/step* 2>/dev/null | xargs -n1 basename | sort | tail -1)
[ -n "$ST" ] && [ -f "$RUN/$ST/adapter.safetensors" ] || { echo "$(ist) r3_B has no checkpoint -- abort"; exit 1; }
echo "$(ist) r3_B training exited; last logged: $(grep '^\[step' "$RUN/train.log" | tail -1 | cut -c1-60); evaluating $ST"
touch "$RUN/done"      # ckpt_sync_r3.sh exits once every step dir is pushed

# ---- 2. push every checkpoint that is not on HF yet, immediately (network only, runs next to the evals)
(
  unset HF_TOKEN; export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_PROGRESS_BARS=1
  for sd in $(ls -d "$RUN"/step* | sort); do
    [ -f "$sd/.pushed" ] && continue
    $PY - "$sd" "Nemotron-split-moe-train/r3_B/checkpoints/$(basename "$sd")" <<'PYEOF' && touch "$sd/.pushed"
import hashlib, os, sys
from huggingface_hub import HfApi
sd, dest = sys.argv[1], sys.argv[2]; api = HfApi(); repo = "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"
assert api.whoami()["name"] == "Anupam-Rawat-IITB", "not logged in as Anupam-Rawat-IITB"
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=sd, path_in_repo=dest, ignore_patterns=[".pushed"],
                  commit_message=f"r3_B checkpoint {dest} (pushed at training end)")
local = hashlib.sha256(open(os.path.join(sd, "adapter.safetensors"), "rb").read()).hexdigest()
info = api.get_paths_info(repo, [dest + "/adapter.safetensors"], repo_type="dataset")[0]
assert info.lfs and info.lfs.sha256 == local, f"sha mismatch after upload: {info.lfs and info.lfs.sha256} != {local}"
print(f"[final_push] pushed {dest} sha {local[:16]}", flush=True)
PYEOF
  done
  echo "$(TZ=Asia/Kolkata date '+%H:%M:%S IST') final push pass done"
) >> "$ROOT/logs/draftalign/ckpt_push_r3B_final.log" 2>&1 &

# ---- 3. the queue (priority order; shard sizes keep every job to ~20 min so the cutoff loses little)
mkdir -p "$CK/r3B_$ST" && cp "$RUN/$ST/adapter.safetensors" "$RUN/$ST/adapter_config.json" "$CK/r3B_$ST/"
: > "$Q"; : > "$Q.lo"; rm -f "$LOG.stop"
line() {  # out_dir items nshards extra_args
  local d="$1" items="$2" n="$3" extra="$4" score="--score"; case "$items" in humaneval*|mbpp*) score="";; esac
  mkdir -p "$d"
  for k in $(seq 0 $((n-1))); do
    local sb="/dev/shm/dlm/sandbox/evalB_$(echo "${d#$ROOT/logs/draftalign/probes_r3/}" | tr / _)_s$k"
    echo "mkdir -p $sb && cd $sb && set -a && . $ROOT/.env && set +a && export HF_HOME=/root/.cache/huggingface && CUDA_VISIBLE_DEVICES={GPU} $PY $ROOT/code/draftalign/run_da_decode.py --model b32 --drafter d32 --items $items $extra --shard $k --nshards $n --gpu {GPU} --no-trace $score --out $d > $d/shard${k}of$n.log 2>&1" >> "$Q"
  done
}
J="--adapter $CK/r3B_$ST --judge full --draft s8 --pool 48 --pool-rule des --pool-flush"
line "$PRB/$ST/humaneval/judge_des48" humaneval 1 "$J"
line "$PRB/$ST/mbpp/judge_des48" mbpp 2 "$J"
line "$PRB/$ST/math500/judge_des48" math500 12 "$J"
line "$PRB/$ST/gpqa/judge_des48" gpqa 9 "$J"
line "$PRB/$ST/gsm8k500/judge_des48" gsm8k500 5 "$J"
line "$PRB/$ST/aime/judge_des48" aime 4 "$J"
S48="--stock --pool 48 --pool-rule des --pool-flush"; S32="--stock --pool 32 --pool-rule des --pool-flush"
line "$PRA/stock/aime/stock_des48" aime 5 "$S48"
line "$PRA/stock/gpqa/stock_des48" gpqa 10 "$S48"
line "$PRA/stock/aime/stock_des32" aime 4 "$S32"
line "$PRA/stock/gpqa/stock_des32" gpqa 8 "$S32"
echo "$(ist) queued $(grep -c . "$Q") decode jobs for r3_B $ST + stock DES baselines"
bash code/draftalign/sched_r3.sh "$Q" "$Q.lo" "$LOG" &
SCHED=$!

score_pass() {
  for d in "$PRB/$ST"/humaneval/* "$PRB/$ST"/mbpp/*; do
    [ -d "$d" ] || continue
    n1=$(ls "$d"/*.jsonl.done 2>/dev/null | wc -l); n2=$(ls "$d"/scored/*.jsonl.done 2>/dev/null | wc -l)
    [ "$n1" -gt "$n2" ] && bash code/draftalign/run_score_r3.sh "$d" once "$d/scored" >> "$d/score.log" 2>&1
  done
}
summ() { $PY code/draftalign/eval_summary_r3B.py --step "$ST" >> "$ROOT/logs/draftalign/eval_summary_r3B.log" 2>&1; }
while kill -0 $SCHED 2>/dev/null; do
  if [ "$(date +%s)" -ge "$CUT" ]; then
    echo "$(ist) cutoff: stopping the scheduler and every decode still running"
    touch "$LOG.stop"; pkill -f "run_da_decode[.]py"; sleep 10; break
  fi
  score_pass; summ; sleep 60
done
wait $SCHED 2>/dev/null
score_pass; summ
echo "$(ist) evals over: $(ls "$PRB/$ST"/*/*/*.jsonl.done 2>/dev/null | wc -l)/33 r3_B shards, $(ls "$PRA"/stock/{aime,gpqa}/stock_des*/*.jsonl.done 2>/dev/null | wc -l)/27 baseline shards, $(ls "$PRB/$ST"/*/*/scored/*.jsonl.done 2>/dev/null | wc -l)/3 code shards scored"
bash code/draftalign/backup_once_r3.sh
echo "$(ist) final backup pass done"
