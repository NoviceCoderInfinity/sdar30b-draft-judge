#!/usr/bin/env bash
# the one smoke test: SGLang (gen_sglang.py --items) on GSM8K-500, 8 GPUs, vs the stock reference (RESULTS.md s1)
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=/workspace/dmge/hf_home HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
until [ -f logs/c1b/stage1/.done_models ]; do sleep 10; done
echo "[smoke $(date -u +%H:%M:%S)] model ready; SGLang GSM8K-500 on 8 GPUs ${SGL_FLAGS:-}"
rm -rf logs/c1b/parity/smoke_fi; mkdir -p logs/c1b/parity/smoke_fi; DEVS=(0 1 2 3 4 5 6 7)
pids=()
for k in 0 1 2 3 4 5 6 7; do dev=${DEVS[$k]}
  CUDA_VISIBLE_DEVICES=$dev /venv/sgl/bin/python code/corpus1b/gen_sglang.py --items gsm8k500 --gpu $k --num-gpus 8 \
    --out logs/c1b/parity/smoke_fi ${SGL_FLAGS:-} > logs/c1b/parity/smoke_fi_gpu$k.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "[smoke $(date -u +%H:%M:%S)] decode rc=$rc"
/venv/train/bin/python - <<'PY'
import glob, json, sys
sys.path[:0] = ["code/compat", "code", "code/corpus1b"]
import common as C, parity_check as PC, research_protocol as P
rows = [r for f in sorted(glob.glob("logs/c1b/parity/smoke_fi/gpu*/records.*.jsonl.gz")) for r in C.read_jsonl(f)]
ok = [r for r in rows if r.get("kept")]
sc = PC.score(PC.family(P.split_base("gsm8k500")), ok)
acc = 100.0 * sum(sc.values()) / max(1, len(sc))
ppb = PC.ppb_weighted([r["ppb"] for r in ok])
eos = sum(bool(r.get("eos")) for r in rows) / max(1, len(rows))
passed = len(ok) == 500 and abs(acc - 90.2) <= 3.0 and abs(ppb - 10.06) / 10.06 <= 0.05
rep = {"n": len(rows), "kept": len(ok), "acc": round(acc, 2), "ref_acc": 90.2, "ppb": round(ppb, 3), "ref_ppb": 10.06,
       "eos_rate": round(eos, 3), "PASS": passed}
print("[smoke] " + json.dumps(rep), flush=True)
open("logs/c1b/stage1/report.jsonl", "a").write(json.dumps({"check": "smoke_gen_gsm8k500_flashinfer_cudagraph", **rep}) + "\n")
open("logs/c1b/stage1/.done_smoke_fi" if passed else "logs/c1b/stage1/.failed_smoke_fi", "w").close()
PY
