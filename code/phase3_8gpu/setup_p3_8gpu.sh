#!/usr/bin/env bash
# 8-GPU re-run copy of code/phase3/setup_p3.sh: identical except that its logs and done-markers go to logs/phase3_8gpu/setup
# (the original writes logs/phase3/setup, where the 4-GPU run's committed .done_* markers would skip every step).
# Phase 3 setup on a fresh 8x RTX PRO 6000 box. Re-runnable: every finished step leaves logs/phase3/setup/.done_<step>.
#   bash code/phase3/setup_p3.sh            (run_p3.sh calls it first)
#  1. Phase 2's CPU setup, unchanged (code/phase2/setup_p2.sh, SKIP_CHECKS=1): /venv/p2 = torch 2.11.0 cu128 + the
#     team's round-3 freeze + transformers 4.54.1; the 30B and SDAR-1.7B at their pinned revisions; round-3 stock
#     records; the 7 benchmark datasets; IFEval-541 (Phase 1's frozen file) and Phase 1's stock IFEval-541 records.
#  2. Phase 1's 128M judge (Arushhh/c1b-judge-lora-sdar30b-b32 main_128m/tok_000128000000, weights hash 35805974) and
#     Phase 1's no-pool SDAR-1.7B + 128M judge records (the reproduction reference), as code/phase2/replicate_nopool.sh.
#  3. CPU tests of the decode loop (seconds).
#  4. Two GPU checks in parallel, 16 items each (a few minutes):
#       stock  --stock on 16 GSM8K-500 items must reproduce the recorded stock outputs token for token (16/16)
#       judge  SDAR-1.7B + 128M judge on 16 HumanEval items vs Phase 1's records (>= 12/16 identical; warns below 16)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_HOME=${HF_HOME:-$ROOT/hf_home} TOKENIZERS_PARALLELISM=false
S=logs/phase3_8gpu/setup; mkdir -p $S
done_(){ [ -f "$S/.done_$1" ]; }; mark(){ touch "$S/.done_$1"; }
say(){ echo "[p3-setup $(date -u +%H:%M:%S)] $*"; }
PY=/venv/p2/bin/python
AD=logs/phase2/ref/c1b_128M/tok_000128000000
REFJ=logs/phase2/ref/S0_128M_nopool_by_set

# ---- 1. Phase 2's CPU setup (environment, pinned models, references, datasets) ------------------------------------
if ! done_ p2_setup; then
  say "Phase 2 CPU setup: /venv/p2, models, references, benchmark datasets (about 30-60 min on a fresh box)"
  SKIP_CHECKS=1 P2_DRAFTERS=S1 bash code/phase2/setup_p2.sh
  mark p2_setup
fi

# ---- 2. the 128M judge + Phase 1's no-pool records ------------------------------------------------------------------
if ! done_ judge_refs; then
  say "128M judge + Phase 1 no-pool SDAR-1.7B + judge records"
  HF_HUB_OFFLINE=0 $PY - <<'PYEOF'
import glob, os, shutil
from huggingface_hub import snapshot_download
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", allow_patterns=["main_128m/tok_000128000000/adapter*"],
                  local_dir="logs/phase2/ref/_hf_c1b_ckpt")
dst = "logs/phase2/ref/c1b_128M/tok_000128000000"; shutil.rmtree(dst, ignore_errors=True)
shutil.copytree("logs/phase2/ref/_hf_c1b_ckpt/main_128m/tok_000128000000", dst)
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir="logs/phase2/ref/_hf_c1b_eval",
                  allow_patterns=["logs/c1b/eval/probe128M/*_nopool/*.jsonl", "logs/c1b/eval/checkpoint128M_nopool/gpqa_judge/*.jsonl"])
src = {s: f"logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/probe128M/{s}_nopool" for s in ["aime", "gsm8k500", "humaneval", "ifeval541", "math500", "mbpp"]}
src["gpqa"] = "logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/checkpoint128M_nopool/gpqa_judge"
for s, d in src.items():
    out = f"logs/phase2/ref/S0_128M_nopool_by_set/{s}"; os.makedirs(out, exist_ok=True)
    for f in glob.glob(d + "/*.jsonl"): shutil.copy(f, out)
PYEOF
  h=$($PY -c "import sys; sys.path[:0]=['code/compat','code']; from draftalign.run_da_decode import adapter_weights_hash; print(adapter_weights_hash('$AD'))")
  [ "${h:0:8}" = 35805974 ] || { say "128M judge hash $h != 35805974"; exit 2; }
  mark judge_refs
fi

# ---- 3. CPU tests ---------------------------------------------------------------------------------------------------
if ! done_ tests; then
  say "CPU tests (decode loop, full judge, re-draft)"
  # the two deselected tests are stale since Phase 1/2: their fake args lack `drafter` (drafter_info needs it), so they
  # fail on the phase2 branch too; they test run-name helpers only, not the decode
  $PY -m pytest -q code/tests/test_da_decode_redraft.py code/tests/test_da_decode_full.py code/tests/test_da_decode.py \
    --deselect code/tests/test_da_decode_full.py::test_runner_adapter_and_pool_helpers \
    --deselect code/tests/test_da_decode.py::test_runner_helpers_and_analyzer \
    > $S/tests.log 2>&1 || { say "TESTS FAILED ($S/tests.log)"; tail -30 $S/tests.log; exit 2; }
  tail -1 $S/tests.log
  mark tests
fi

# ---- 4. two reproduction checks, in parallel (GPU 0: stock, GPU 1: judge) -------------------------------------------
if ! done_ checks; then
  say "checks: stock (16 GSM8K-500 items) and judge (16 HumanEval items), in parallel"
  rm -rf $S/check_stock $S/check_judge
  mkdir -p /dev/shm/p3_sandbox/check_stock /dev/shm/p3_sandbox/check_judge
  ( cd /dev/shm/p3_sandbox/check_stock && CUDA_VISIBLE_DEVICES=0 $PY $ROOT/code/draftalign/run_da_decode.py --model b32 \
      --items gsm8k500 --stock --limit 16 --ref-dir logs/phase2/ref/stock_flat --no-trace --score \
      --out $S/check_stock ) > $S/check_stock.log 2>&1 & p1=$!
  ( cd /dev/shm/p3_sandbox/check_judge && CUDA_VISIBLE_DEVICES=1 $PY $ROOT/code/draftalign/run_da_decode.py --model b32 \
      --items humaneval --drafter d32 --draft s8 --judge full --adapter $AD --limit 16 \
      --ref-dir $REFJ/humaneval --p2-integrity --no-trace --score --out $S/check_judge ) > $S/check_judge.log 2>&1 & p2=$!
  wait $p1 || { say "stock check crashed ($S/check_stock.log)"; tail -20 $S/check_stock.log; exit 2; }
  wait $p2 || { say "judge check crashed ($S/check_judge.log)"; tail -20 $S/check_judge.log; exit 2; }
  $PY - <<'PYEOF' || exit 2
import glob, json
S = "logs/phase3_8gpu/setup"
def recs(d):
    out = []
    for f in glob.glob(f"{d}/*.jsonl"):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out
st, jd = recs(f"{S}/check_stock"), recs(f"{S}/check_judge")
st_ok = sum(bool(r.get("ref_match", {}).get("gen_ids")) for r in st)
j_ref = [r for r in jd if "ref_match" in r]
j_ok = sum(bool(r["ref_match"].get("gen_ids")) for r in j_ref)
res = {"stock_items": len(st), "stock_identical": st_ok, "judge_items": len(jd), "judge_with_reference": len(j_ref),
       "judge_identical": j_ok, "gpu": (st or jd or [{}])[0].get("gpu")}
json.dump(res, open(f"{S}/checks.json", "w"), indent=1)
print("[p3-setup] checks:", res)
assert len(st) == 16 and st_ok == 16, f"stock check: {st_ok}/{len(st)} identical (must be 16/16): the environment differs"
if not j_ref:
    print("[p3-setup] WARNING: no Phase 1 judge record matched by cell name; judge reproduction not verified")
else:
    assert j_ok >= 12, f"judge check: {j_ok}/{len(j_ref)} identical to Phase 1 (need >= 12)"
    if j_ok < len(j_ref):
        print(f"[p3-setup] note: judge {j_ok}/{len(j_ref)} identical (bf16 near-tie flips between environments)")
PYEOF
  mark checks
fi
say "PHASE 3 SETUP PASSED"
