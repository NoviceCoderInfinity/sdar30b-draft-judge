#!/usr/bin/env bash
# Phase 5 CP6 (Plan A): continue the 128M judge on the RECORDED states, ~32M more tokens (-> 160M), 8 GPUs, step-0 check.
# Builds: (1) noop_rids.txt from the record logs (items a scattered-block no-op fired on -> excluded); (2) a filtered
# training index = fresh-index rows that were recorded, minus the decontam exclusions and the no-op items; then A0
# index_states over BOTH record roots, then torchrun train_c1b_rec. Training recipe/numerics unchanged (continue-from).
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$ROOT/code
. logs/phase5/sgl_setup/sgl_env.sh 2>/dev/null
JS=logs/phase5/judge_states; mkdir -p "$JS"
say(){ echo "[cp6 $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }

# 1. no-op'd rids (these items have an incomplete block -> exclude from training), across all record roots/logs
grep -ho 'NOOP_SCATTERED rid=[^ ]*' logs/phase5/judge_sglang/record*/record_gpu*.log 2>/dev/null \
  | sed 's/NOOP_SCATTERED rid=//' | sort -u > "$JS/noop_rids.txt"
say "no-op items excluded: $(grep -c . "$JS/noop_rids.txt" 2>/dev/null || echo 0)"

# 2. filtered training index: recorded pids (done.txt of both roots) minus exclusions minus no-op items
/venv/p2/bin/python - <<'PY'
import json, glob, os
ROOT="/workspace/diffusion-moe-general-expert"; JS=ROOT+"/logs/phase5/judge_states"
done=set()
for f in glob.glob(ROOT+"/logs/phase5/judge_sglang/record*/gpu*/done.txt"):
    done|={l.strip() for l in open(f) if l.strip()}
excl=set(l.strip() for l in open(ROOT+"/logs/phase5/cp6_prep/exclude_pids.txt") if l.strip())
noop=set(l.strip() for l in open(JS+"/noop_rids.txt") if l.strip()) if os.path.exists(JS+"/noop_rids.txt") else set()
keep=done-excl-noop
n=0
# write the filtered index INTO the source index's own dir so each row's relative "file" path (../waveC/items/...) still resolves
with open(ROOT+"/logs/c1b/trainset/fresh/train_index.jsonl","w") as o:
    for line in open(ROOT+"/logs/c1b/trainset/fresh/index.jsonl"):
        if not line.strip(): continue
        if json.loads(line)["pid"] in keep: o.write(line); n+=1
print(f"recorded={len(done)} excl_hit={len(done&excl)} noop={len(noop)} -> train_index rows={n}")
PY
say "train_index built"

# 3. A0 index_states over both record roots, joined to the filtered index
/venv/p2/bin/python code/judge_round/index_states.py \
  logs/phase5/judge_sglang/record logs/phase5/judge_sglang/record_run1 \
  logs/phase5/judge_sglang/record2 logs/phase5/judge_sglang/record3 \
  --index logs/c1b/trainset/fresh/train_index.jsonl --out "$JS/rec_main" 2>&1 | tail -20
[ -f "$JS/rec_main/meta.json" ] || { say "index_states FAILED"; exit 1; }
say "A0 states index built: $(/venv/p2/bin/python -c "import json;s=json.load(open('$JS/rec_main/summary.json'));print(s.get('items'),'items',s.get('blocks'),'blocks',s.get('passes'),'passes')" 2>/dev/null)"
# final reproducibility manifest (states + train_index now final) and per-domain counts for STATUS/RECORDINGS
bash code/judge_round/gen_manifest.sh
/venv/p2/bin/python -c "import json;s=json.load(open('$JS/rec_main/summary.json'));print('[cp6] per-domain training items:', s.get('items_by_domain'))" 2>/dev/null | tee -a logs/phase5/judge/cp6_launch.log

# 4. train: continue 128M -> 160M (+32M), save at +8M (136M) and +32M (160M), step-0 check on 256 states
/venv/p2/bin/torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_rec.py \
  --states "$JS/rec_main" --index logs/c1b/trainset/fresh/train_index.jsonl \
  --out logs/phase5/judge/rec_main --continue-from logs/c1b/train/main_128m/tok_000128000000 \
  --max-tokens 160e6 --save-at-tokens 136e6,144e6,152e6,160e6 --step0-states 256 --step0-min-agree 0.85 2>&1 | tee -a logs/phase5/judge/rec_main_train.log
say "train_c1b_rec exited rc=${PIPESTATUS[0]}"
