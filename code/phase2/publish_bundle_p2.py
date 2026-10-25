"""Make the Hugging Face backup repo a self-contained bundle (rerun any time; idempotent).

    /venv/p2/bin/python code/phase2/publish_bundle_p2.py

Adds, next to what backup_p2.py syncs (phase2/ = every log, record, expert / judge / trace array, analysis and plot):
  README.md                      the layout and the one-command download
  code/dmge_phase2-drafter-swap_<commit>.tar.gz   `git archive` of the checked-out commit (all code)
  assets/datasets/               IFEval-541 (Phase 1's frozen file + sha256) and the benchmark dataset manifest
  assets/results/                TABLES.md (all comparison tables) and metrics.json
Secrets are never staged: .env is not in git, and the README / tables are generated text.
"""
import json
import os
import subprocess
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("P2_HF_REPO", "Arushhh/phase2-drafter-swap")
SETS = ["humaneval", "mbpp", "gsm8k500", "ifeval541", "math500", "gpqa", "aime"]

README = """# Phase 2 drafter swap (no expert pool): code, results, data, weights in one place

Frozen SDAR-30B-A3B-Chat-b32 + the Phase 1 LoRA judge (128M-token corpus, weights hash 35805974), **no expert pool**,
nothing trained. Only the drafter changes: SDAR-1.7B (reference, Phase 1) vs **CoDA-1.7B** (`F2_nopool`) vs
**Qwen3-0.6B diffusion** (`S1_nopool`), on HumanEval 164, MBPP 257, GSM8K-500, IFEval-541, MATH-500, GPQA-D 198, AIME 90.

## Get everything with one command

Tested 2026-10-05 from a clean environment without a token: 2,217 files, ~21 GB, about 7 min.

```bash
pip install -U huggingface_hub          # provides the `hf` command (huggingface-cli was removed in huggingface_hub 1.x)
hf download {repo} --repo-type dataset --local-dir phase2_bundle    # all ~21 GB; no token needed (public)
```

## Layout

| path | what |
|---|---|
| `assets/results/TABLES.md`, `metrics.json` | every comparison table: accuracy, Δ vs SDAR-1.7B with 95% CI, experts/token, MoE-only / full-pipeline / drafter-only speedups, passes/block, draft tokens kept |
| `phase2/analysis/nopool/` | the analysis: `REPORT.md` (dip? why? scope), `IMPROVEMENT_PLAN.md`, `analysis.json`, figures, scripts |
| `phase2/decode/F2_nopool/<set>/`, `phase2/decode/S1_nopool/<set>/` | per-item records (`*.jsonl`), running metrics (`*.running.jsonl`), and `experts.tar` / `judge.tar` / `traces.tar` (per-forward expert bitmasks + router top-8, every judge read's probabilities, drafter confidences) |
| `phase2/plots/` | all plots (also on W&B `arushsharma536-spe-iit/phase2-drafter-swap`) |
| `phase2/jobs/`, `phase2/*.log`, `phase2/setup/` | job logs, launcher logs, environment freezes, setup checks |
| `assets/judge_128M_tok_000128000000/` | the judge LoRA weights used (from `Arushhh/c1b-judge-lora-sdar30b-b32`) |
| `assets/ref_S0_SDAR1.7B_judge128M_nopool/<set>/` | reference A: SDAR-1.7B + judge, no pool (Phase 1 records) |
| `assets/ref_stock_30B/` | reference B: stock 30B decode records |
| `assets/datasets/` | IFEval-541 (frozen, + sha256) and the manifest of the other benchmark datasets (public HF datasets) |
| `code/*.tar.gz` | the code at the commit that produced this (`git archive`); also on GitHub `NoviceCoderInfinity/diffusion-moe-general-expert` branch `phase2-drafter-swap` |

Not part of the results: `phase2/decode/F2/`, `phase2/jobs/decode_F2*` (non-nopool) and `phase2/analysis/logprob/` are an
earlier run with the DES48 expert pool that was stopped at the user's request; ignore them.

## Replicate on a new 8-GPU box

Unpack the code tarball (or clone the branch), put `HF_TOKEN` in `.env`, then
`bash code/phase2/replicate_nopool.sh` (see `docs/PHASE2_NOPOOL_REPLICATE.md`).

Models and drafters are public at pinned revisions: JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc, JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05,
Salesforce/CoDA-v0-Instruct @ f2d40ce, dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 @ 60f77bb.
"""

DATASETS = {
    "humaneval": "openai/openai_humaneval (test, 164)", "mbpp": "google-research-datasets/mbpp sanitized (test, 257)",
    "gsm8k500": "openai/gsm8k main test, fixed 500-item subset (code/research_protocol.py frozen split)",
    "math500": "HuggingFaceH4/MATH-500 (500)", "gpqa": "Idavidrein/gpqa gpqa_diamond (198)",
    "aime": "AIME 90 items (code/run_desvote.py build_items)", "ifeval541": "assets/datasets/ifeval541.jsonl (Phase 1 frozen, 541)",
}


def main():
    api = HfApi()
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "README.md"), "w").write(README.format(repo=REPO))
        tar = os.path.join(td, f"dmge_phase2-drafter-swap_{commit}.tar.gz")
        subprocess.check_call(["git", "-C", ROOT, "archive", "--format=tar.gz", "-o", tar, "HEAD"])
        ds = os.path.join(td, "datasets"); os.makedirs(ds)
        for f in ("ifeval541.jsonl", "ifeval541.jsonl.sha256"):
            src = os.path.join(ROOT, "data", "c1b", "eval", f)
            if os.path.exists(src):
                open(os.path.join(ds, f), "wb").write(open(src, "rb").read())
        json.dump(DATASETS, open(os.path.join(ds, "MANIFEST.json"), "w"), indent=1)
        res = os.path.join(td, "results"); os.makedirs(res)
        for src, name in ((os.path.join(ROOT, "logs/phase2/analysis/tables/TABLES.md"), "TABLES.md"),
                          (os.path.join(ROOT, "logs/phase2/analysis/tables/metrics.json"), "metrics.json")):
            if os.path.exists(src):
                open(os.path.join(res, name), "wb").write(open(src, "rb").read())
        api.upload_file(path_or_fileobj=os.path.join(td, "README.md"), path_in_repo="README.md", repo_id=REPO,
                        repo_type="dataset", commit_message="bundle README")
        api.upload_file(path_or_fileobj=tar, path_in_repo=f"code/{os.path.basename(tar)}", repo_id=REPO,
                        repo_type="dataset", commit_message=f"code snapshot {commit}")
        api.upload_folder(folder_path=ds, path_in_repo="assets/datasets", repo_id=REPO, repo_type="dataset",
                          commit_message="datasets")
        api.upload_folder(folder_path=res, path_in_repo="assets/results", repo_id=REPO, repo_type="dataset",
                          commit_message="result tables")
    print(f"bundle published to https://huggingface.co/datasets/{REPO} (code {commit})")


if __name__ == "__main__":
    main()
