# Phase 1: building the 1B-token judge corpus and training on it (branch `phase1-datacorpus`)

Plan and numbers: `docs/CORPUS_1B_PLAN_2026-09-28.md` (Notion: "Phase 1: 1B-token training corpus for the judge").
Box: one 8x RTX PRO 6000 Blackwell (96 GB), CUDA >= 12.8, >= 64 cores, >= 512 GB RAM, >= 500 GB disk.
Models (stock, block 32, pinned in `code/research_protocol.py::MODELS`): the judge is a LoRA on
`JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc`; the drafter is `JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05`.

## Three commands
```bash
bash code/corpus1b/setup_box.sh      # stage 1, ~3 h: envs, models, datasets, prompt lists, eval sets, 4 checks
bash code/corpus1b/run_wave_a.sh     # wave A, ~14 h: gen -> drafts -> items -> LR test -> train to 128M -> 32M test -> evals -> gate_A.md
bash code/corpus1b/run_wave_b.sh     # wave B, ~74 h, only if gate_A.md says CONTINUE
```
All three resume where they stopped. Code sets are always executed in `/dev/shm` sandboxes (AGENTS.md 9(p)).

## File map (`code/corpus1b/`)
| file | what |
|---|---|
| `common.py` | schemas (prompt row, key types, generation record), pids, wave split (12.8% A), loop rule |
| `sources.py` | every dataset: HF repo, config, split, fields, licence, quota, template, key type, extractor |
| `download.py` | downloads sources + eval sets into `$HF_HOME`, records resolved revisions |
| `extract_maths.py`, `extract_science.py`, `extract_chat.py`, `extract_code.py`, `extract_if.py`, `compose_if.py` | source rows -> prompt rows |
| `decontam.py`, `dedupe.py` | exact + 13-gram decontamination vs every eval set (+ embedding check for GPQA-like sources); MinHash dedupe |
| `build_prompt_lists.py`, `spotcheck.py` | the prompt lists `data/c1b/prompt_lists/<domain>.jsonl` + manifest; 200-key spot-check files |
| `checkers.py` | entry checks per key type (math, asserts, humaneval, stdio, mc, free, ifeval, chat = ends + no loop + English) |
| `eval_sets.py` | frozen eval sets `ifeval541`, `finqa500`, `chartqa300`, `chat800` (registered in research_protocol + both runners) |
| `sglang_plugin/` | `LowConfidenceTraced` (30B stock rule + commit-pass trace), `StaticDraftTraced` (1.7B s8 drafts + confidence), `install.py`, YAML configs |
| `gen_sglang.py` | per-GPU generation from the pinned 30B, answers rebuilt from the trace, checked, resumable |
| `draft_sglang.py` | per-GPU teacher-forced drafts of every kept block |
| `to_trainset.py` | records + drafts -> `logs/c1b/trainset/wave<W>/` in draftalign/build_train_set format (+ draft_conf, domain) |
| `train_c1b.py` | the trainer (round-3 recipe on the two-stream forward `draftalign/train_fast.py`), chat beta 0 / weight 1, WSD by tokens, milestones, per-domain probe |
| `parity_check.py` | stage-1 checks: gen, draft, train, speed |
| `eval_ckpt.sh`, `eval_summary.py`, `gate_report.py` | evaluation queues on the existing harness; summaries; the checkpoint-A gate |
| `setup_box.sh`, `run_wave_a.sh`, `run_wave_b.sh`, `backup_hf.sh` | orchestration |
| `tests/` | CPU tests: fast trainer == round-3 loss and gradients; gen trace -> items -> sampler; trainer resume/continue; builders; checkers |

Changed round-3 files: `draftalign/dataset.py` (draft_conf, item grouping, per-domain holdout), `losses.py` (optional
cert_mask), `adapter.py` + `da_decode.py` (optional zero-init confidence embedding), `train_fast.py` (new). With the new
options off, every round-3 path is unchanged (the existing tests pass; two runner tests fail identically on round 3).

## Datasets (all public; gated ones need HF_TOKEN read access)
Training sources, with splits, fields and quotas: `code/corpus1b/sources.py` (`SOURCES`). Maths: openai/gsm8k,
EleutherAI/hendrycks_math, nvidia/OpenMathInstruct-2, AI-MO/NuminaMath-1.5, zwhe99/DeepMath-103K, nvidia/OpenMathReasoning.
Code: nvidia/OpenCodeInstruct, google-research-datasets/mbpp (train/validation/prompt), likaixin/TACO-verified,
codeparrot/apps, deepmind/code_contests, open-r1/codeforces. Science: TIGER-Lab/WebInstruct-verified,
nvidia/Nemotron-Science-v1, nvidia/OpenScienceReasoning-2, allenai/ai2_arc. IF: nvidia/Nemotron-RL-instruction_following,
allenai/IF_multi_constraints_upto5, allenai/RLVR-IFeval, allenai/WildChat-4.8M (composed). Chat: allenai/WildChat-1M,
HuggingFaceH4/ultrachat_200k, OpenAssistant/oasst2. Held out: dreamerdeo/finqa, nmayorga7/chartqa-tables + ahmed-masry/ChartQA.
Decontamination sets: `sources.EVAL_SETS`. Download: `python code/corpus1b/download.py` (setup_box.sh does it).

## Evaluation
Checkpoint A (128M judge + DES48): the paper's five benchmarks (GSM8K-500, MATH-500, AIME, HumanEval, MBPP) + GPQA-Diamond
+ IFEval + FinQA + ChartQA; the released model on the three new sets; the 32M test (main vs confidence embedding, tau
0.90/0.95/0.98 on MBPP + HumanEval). Final (1B judge, with and without DES48): all of those + chat800 (speed, grader
quality, loop and no-stop rates). Released-model references for the paper's sets: RESULTS.md section 1.

## Troubleshooting
| symptom | fix |
|---|---|
| SGLang errors on sm_120 / attention | `SGL_FLAGS="--attention-backend triton --disable-cuda-graph"` (then `torch_native`); rerun the failed step |
| SGLang OOM | lower `--max-running` (default 128) or `--mem-fraction` (0.85) via SGL_FLAGS |
| check gen fails (acc > 3 pts or passes/blk > 5% off) | the engine is not decoding the reference rule: confirm gen_b32.yaml loaded (block 32, threshold 0.95); do not generate on a mismatched engine |
| check speed fails | generation would exceed the plan (73 h both waves); raise --max-running, enable CUDA graphs, or cut IF share (sources.QUOTAS) |
| trainer OOM | `--max-states-micro 32` (default 48); keep layer checkpointing on |
| a gen/draft process dies | rerun the wave script: it resumes from done.txt |
| disk < 50 GB | `bash code/corpus1b/backup_hf.sh` then delete `logs/c1b/gen/*/gpu*/trace` |
