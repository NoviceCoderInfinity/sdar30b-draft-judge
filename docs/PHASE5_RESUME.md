# Phase 5 — reproduce & resume on a fresh box

Everything needed to resume the judge-round training from ANY checkpoint, or re-run the eval, on a clean box. Backed up
on GitHub (branch `phase5-box-s1`) + the public HF dataset `Arushhh/phase5-loop-guard`. Pinned code commit: **`2faa797`**
(record the exact commit from `logs/phase5/env/versions.txt`).

## 0. What is where
- Code + docs: GitHub `NoviceCoderInfinity/diffusion-moe-general-expert` branch `phase5-box-s1`.
- Data/states/checkpoints: HF dataset `Arushhh/phase5-loop-guard` (layout in HUGGINGFACE.md §10).
- Integrity: `logs/phase5/MANIFEST.sha256` (sha256 of every training input + checkpoint; pinned models recorded by revision).
- Env: `logs/phase5/env/` (pip freeze of /venv/p2 and /venv/sgl, versions, driver, git commit).

## 1. Clone the code at the pinned commit
```
git clone https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git
cd diffusion-moe-general-expert && git checkout 2faa797     # or the commit in logs/phase5/env/versions.txt
```

## 2. Download the data/states/checkpoints from HF
```
export HF_HUB_OFFLINE=0
hf download Arushhh/phase5-loop-guard --repo-type dataset --local-dir .
# the ~150k packed training items are shipped as one tar:
tar -xf logs/c1b/trainset/waveC/items.tar -C logs/c1b/trainset/waveC/     # -> items/*
```
You also need the pinned base models (not in this dataset): `JetLM/SDAR-30B-A3B-Chat-b32` @ rev `c351bbc37d240aa6871f167e8f92d694281b0c22`
and the Phase 4 merged drafter at `logs/phase4/train/main/merged` (from the Phase 2/4 backups). `research_protocol` loads the
30B from `hf_cache/hub` (set `HF_HOME=$PWD/hf_home` for the offline eval datasets).

## 3. Verify integrity against the manifest
```
sha256sum -c logs/phase5/MANIFEST.sha256    # every training input + checkpoint must say OK
```

## 4. Environment
Use the Vast base image venvs (`/venv/p2` = HF/torch trainer, torch 2.11.0+cu128; `/venv/sgl` = SGLang 0.5.10.post1,
torch 2.9.1+cu128; driver 615.71.09, CUDA 12.8/12.9). To rebuild: `uv pip install -r logs/phase5/env/p2_freeze.txt`
(resp. `sgl_freeze.txt`). The SGLang judge plugin must be installed into the sgl venv:
`C1B_CODE_DIR=$PWD/code /venv/sgl/bin/python code/corpus1b/sglang_plugin/install.py` (its c1b_judge_loop.py sha256 must
match `code/corpus1b/sglang_plugin/c1b_judge_loop.py`).

## 5. Resume training from ANY checkpoint
Checkpoints (each a dir with `adapter.safetensors`, `adapter_config.json`, `trainer_state.json`, `trainer_state.pt`):
- `logs/phase5/judge/start_128m`  — the 128M starting point (tokens=128,003,432)
- `logs/phase5/judge/rec_main/tok_000136000000` … `tok_000160000000` — the retrained checkpoints (+8M each)

`--continue-from` loads the adapter + the AdamW optimizer state + the token/step count, so training and the per-block
sampler continue deterministically (train_c1b's fixed seed + token accounting; recipe is unchanged = train_c1b defaults:
override-weight 2 / code 3, beta 0.5 @ temp 0.5, calib 1.0, lr 8e-5, clip 1.0).
```
export HF_HOME=$PWD/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$PWD/code
# easiest: the whole handover launcher rebuilds the index + step-0 check + trains:
bash code/judge_round/train_cp6.sh
# or resume directly from a specific checkpoint CKPT:
torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_rec.py \
  --states logs/phase5/judge_states/rec_main --index logs/c1b/trainset/fresh/train_index.jsonl \
  --out logs/phase5/judge/rec_main --continue-from logs/phase5/judge/rec_main/tok_000136000000 \
  --max-tokens 160e6 --save-at-tokens 136e6,144e6,152e6,160e6 --step0-states 256 --step0-min-agree 0.90
```
(If `logs/phase5/judge_states/rec_main` is absent, rebuild it: `code/judge_round/index_states.py` over the four record
roots `logs/phase5/judge_sglang/{record,record_run1,record2,record3}` with `--index logs/c1b/trainset/fresh/train_index.jsonl`
— exactly what `train_cp6.sh` does. The filtered `train_index.jsonl` MUST sit in `logs/c1b/trainset/fresh/` so each item's
relative `file` path resolves.)

## 6. Evaluate (plain HF harness — the only source of every reported number)
rd1, the 7 sets, with the final judge adapter; passes-per-block is compared to Phase 4 (7.68 rd1):
```
cd /dev/shm && mkdir -p p5eval && cd p5eval      # run code-pool decodes from a sandbox cwd (AGENTS.md §9)
for S in ifeval541 math500 gpqa aime humaneval mbpp gsm8k500; do
  CUDA_VISIBLE_DEVICES=0 HF_HOME=<repo>/hf_home /venv/p2/bin/python <repo>/code/draftalign/run_da_decode.py \
    --model b32 --items $S --drafter <repo>/logs/phase4/train/main/merged --draft s8 \
    --judge full --adapter <repo>/logs/phase5/judge/rec_main/tok_000160000000 \
    --full-redraft-every 1 --redraft-ban-rejected --loop-guard 12 --score --out <repo>/logs/phase5/eval/<S>
done
```
Vanilla multiplier check (small sample per set, paired to the Phase 4 vanilla records; speedup is DERIVED = our
seconds / Phase 4 vanilla seconds × the measured this-box/Phase-4-box multiplier): see `code/judge_round/speed_check_b.sh`
and `run_vanilla_here.sh`. Every speed/accuracy/passes number is a plain-harness run — never an SGLang number.
```
