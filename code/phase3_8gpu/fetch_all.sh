#!/usr/bin/env bash
# Rebuild everything the 8-GPU Phase 3 analysis reads that is not in git, from public Hugging Face repos, into the same
# paths the run used. Download only: no venv, no GPU, no checks beyond counts and the judge's hash.
#
#   git clone -b phase3-redraft-8gpu https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git dmge
#   cd dmge && bash code/phase3_8gpu/fetch_all.sh                 # references + judge, about 1 GB
#   FETCH_MODELS=1 bash code/phase3_8gpu/fetch_all.sh             # optional: + the 30B and 1.7B at pinned revisions (~63 GB)
#
# Already in git (nothing to fetch): code/, results/, docs/, paper/, logs/phase3/ (first 4-GPU run) and logs/phase3_8gpu/
# (every rd1 / rd2 / base decode record, setup checks, events, host snapshots). The same logs/phase3_8gpu + results are
# mirrored on https://huggingface.co/datasets/Arushhh/phase3-redraft.
# Fetched here:
#   logs/phase2/ref/c1b_128M/tok_000128000000      the 128M judge LoRA (Arushhh/c1b-judge-lora-sdar30b-b32, sha256 35805974...)
#   logs/phase2/ref/S0_128M_nopool_by_set/<bench>  Phase 1 base records (Arushhh/c1b-judge-corpus-waveA)
#   logs/phase2/ref/stock_flat                     stock 30B records: round 3 (Anupam-Rawat-IITB/dlm-exp1-...) + IFEval (Phase 1)
#   data/c1b/eval/ifeval541.jsonl(.sha256)         Phase 1's frozen IFEval-541
#   hf_cache/hub (FETCH_MODELS=1)                  JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc, JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05
# To decode again (GPU box): bash code/phase3_8gpu/setup_p3_8gpu.sh builds /venv/p2 and runs the reproduction checks.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PY=${PY:-python3}
$PY -c "import huggingface_hub" 2>/dev/null || $PY -m pip install -q huggingface_hub
export HF_HUB_OFFLINE=0 FETCH_MODELS=${FETCH_MODELS:-0}
$PY - <<'EOF'
import glob, hashlib, json, os, shutil
from huggingface_hub import snapshot_download, hf_hub_download

def say(*a): print("[fetch_all]", *a, flush=True)
DL = "logs/phase2/ref/_hf"
SETS = ["ifeval541", "math500", "gpqa", "aime", "humaneval", "mbpp", "gsm8k500"]

# 1. the 128M judge
say("128M judge LoRA <- Arushhh/c1b-judge-lora-sdar30b-b32")
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", allow_patterns=["main_128m/tok_000128000000/adapter*"],
                  local_dir="logs/phase2/ref/_hf_c1b_ckpt")
dst = "logs/phase2/ref/c1b_128M/tok_000128000000"; shutil.rmtree(dst, ignore_errors=True)
shutil.copytree("logs/phase2/ref/_hf_c1b_ckpt/main_128m/tok_000128000000", dst)
h = hashlib.sha256(open(f"{dst}/adapter.safetensors", "rb").read()).hexdigest()
assert h.startswith("35805974"), f"judge adapter sha256 {h[:16]} != 35805974..."
say("judge adapter sha256", h[:16])

# 2. Phase 1 base records (SDAR-1.7B + 128M judge, no pool) and Phase 1's frozen IFEval-541 + its stock records
say("Phase 1 base records, IFEval-541, stock IFEval records <- Arushhh/c1b-judge-corpus-waveA")
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir="logs/phase2/ref/_hf_c1b_eval",
                  allow_patterns=["logs/c1b/eval/probe128M/*_nopool/*.jsonl", "logs/c1b/eval/checkpoint128M_nopool/gpqa_judge/*.jsonl",
                                  "logs/c1b/eval/checkpointB/ifeval541_released/*.jsonl*"])
src = {s: f"logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/probe128M/{s}_nopool" for s in SETS if s != "gpqa"}
src["gpqa"] = "logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/checkpoint128M_nopool/gpqa_judge"
for s, d in src.items():
    out = f"logs/phase2/ref/S0_128M_nopool_by_set/{s}"; os.makedirs(out, exist_ok=True)
    for f in glob.glob(d + "/*.jsonl"):
        shutil.copy(f, out)
os.makedirs("data/c1b/eval", exist_ok=True)
for f in ("ifeval541.jsonl", "ifeval541.jsonl.sha256"):
    shutil.copy(hf_hub_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", filename=f"data/c1b/eval/{f}"), f"data/c1b/eval/{f}")
want = open("data/c1b/eval/ifeval541.jsonl.sha256").read().split()[0]
assert hashlib.sha256(open("data/c1b/eval/ifeval541.jsonl", "rb").read()).hexdigest() == want, "IFEval-541 sha256 mismatch"

# 3. stock 30B records: round 3 (6 benchmarks) + Phase 1's IFEval-541 run
say("stock records <- Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts")
snapshot_download("Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts", repo_type="dataset", local_dir=DL,
                  allow_patterns=["Nemotron-split-moe-train/reference_evals/da_decode/*_B32_t95_stock/*",
                                  "Nemotron-split-moe-train/reference_evals/da_decode/*_B32_t95_stock_p48desf/*",
                                  "Nemotron-split-moe-train/reference_evals/research_v1/baselines_b32/gsm8k_B32_t95_stock.gsm8k500.*"])
os.makedirs("logs/phase2/ref/stock_flat", exist_ok=True)
for f in glob.glob(f"{DL}/Nemotron-split-moe-train/reference_evals/**/*.jsonl", recursive=True):
    shutil.copy(f, "logs/phase2/ref/stock_flat/")
for f in glob.glob("logs/phase2/ref/_hf_c1b_eval/logs/c1b/eval/checkpointB/ifeval541_released/ifeval_B32_t95_stock.ifeval541.shard*of8.jsonl"):
    shutil.copy(f, "logs/phase2/ref/stock_flat/")

# counts: what the report and page builders expect
def n(pattern):
    ids = set()
    for f in glob.glob(pattern):
        for line in open(f):
            if line.strip():
                ids.add(json.loads(line)["idx"])
    return len(ids)
pre = {"ifeval541": "ifeval", "gsm8k500": "gsm8k"}
for s in SETS:
    b = n(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
    st = n(f"logs/phase2/ref/stock_flat/{pre.get(s, s)}_B32_t95_stock.{s}.shard*.jsonl")
    say(f"{s:10s} Phase 1 base records {b:4d}, stock records {st:4d}")

# 4. optional: the two models at the pinned revisions (code/research_protocol.py::MODELS)
if os.environ["FETCH_MODELS"] == "1":
    for repo, rev in (("JetLM/SDAR-30B-A3B-Chat-b32", "c351bbc37d240aa6871f167e8f92d694281b0c22"),
                      ("JetLM/SDAR-1.7B-Chat-b32", "2c83c05d84be90a01c72443b2123ebf0376f6fd4")):
        say("model", repo, "@", rev[:7], "->", snapshot_download(repo, revision=rev, cache_dir="hf_cache/hub"))
say("done")
EOF
