# 1B-token training corpus for the judge

Plan, 28 Sep 2026. Nothing has been run.

The judge learns from the 30B model's own answers:

1. Take prompts from public datasets.
2. Let SDAR-30B-A3B answer them.
3. Keep only the answers a program verifies.
4. Cut the kept answers into 32-token blocks.

"1B tokens" counts prompt + kept answer. That is the same unit as round 3's 1.84M, so this is 543× round 3.

## Domains

| domain | share | tokens | why |
|---|---:|---:|---|
| Maths | 25% | 250M | 3 of our 6 benchmarks. Round 3 had GSM8K only, and MATH-500 sits at −3.2. |
| Code | 25% | 250M | Where the judge breaks: one wrong token kills a program. Round 2 had 68 code items (HumanEval −15.9); round 3 had 2,344 (−3.0). |
| Science | 15% | 150M | GPQA shows the biggest loss (−8.6 with DES48, +0.5 without). No earlier mix had science. |
| Instruction following | 15% | 150M | Checkable by program. FLARE's mix holds IF at 20%; we keep it close. Adds IFEval as a benchmark. |
| General chat | 20% | 200M | What users actually send. There is no checker, so we keep the model's own complete answers. |

**Charts are out.** SDAR is text-only and chart datasets need the image. Their text form, tables, becomes a held-out test domain we never train on.

**Why volume per domain matters more than share.** Dong et al. 2023 find that "data amounts directly influence each ability, while the data ratio is insignificant". Our own rounds agree (the code row above). Every domain here gets at least 250K kept items.

## Sources

All public and ungated, with no non-commercial licences. Every prompt is decontaminated before use (see *From prompt to blocks*).

### Maths (250M)

| dataset | subset / split | prompt → answer | size | licence | notes |
|---|---|---|---:|---|---|
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `main` / `train` | `question` → number after `####` in `answer` | 7,473 | MIT | drop the 200 frozen calib/valid items |
| [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | all 7 subjects / `train` | `problem` → last `\boxed{}` in `solution` | 7,500 | MIT | `level` 1–5; never the test split (MATH-500 comes from it) |
| [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | `train` | `problem` → `expected_answer` | 14M pairs, ~600K unique problems | CC-BY-4.0 | GSM/MATH-level volume; dedupe on `problem`; augmented problems have majority-vote answers |
| [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | `train` | `problem` → `answer` | 896,215 | Apache-2.0 | MATH L3–5 and olympiad; drop `answer == "proof"` (12%) |
| [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) | `train` | `question` → `final_answer` | 103,022 | MIT | hard, `difficulty` 0–10; the most careful decontamination of any candidate |
| [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | `cot` | `problem` → `expected_answer` | 306K unique problems | CC-BY-4.0 | olympiad (AoPS); keep `problem_type == "has_answer_extracted"` |

Split within maths: GSM-level 20%, MATH L1–3 30%, MATH L4–5 35%, olympiad 15%. Olympiad is capped because the model rarely solves it (AIME 6.7%) and long solutions hit the 1,024-token cap.

### Code (250M)

| dataset | subset / split | prompt → check | size | licence | notes |
|---|---|---|---:|---|---|
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `train`, `average_test_score == 1.0` | `input` → run `unit_tests` | 5M (1.64M pass) | CC-BY-4.0 | 80% of code; MBPP-style and HumanEval-style prompts, 60/40 as in round 3 |
| [likaixin/TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified) | `train` | `question` → stdin/stdout `input_output` | 12,898 | MIT | competitive programming (this row and the next three: 20% of code) |
| [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | `train`, introductory + interview | `question` → `input_output` | 5,000 | MIT | uses a loading script; read the repo's `train.jsonl` directly |
| [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) | `train` | `description` → `public_tests` / `private_tests` / `generated_tests` | 13,328 | CC-BY-4.0 | Codeforces ids make decontamination exact |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | `verifiable` / `train` | `description` → `official_tests` | 8,338 | CC-BY-4.0 | |
| [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | `full` / `train` + `validation` + `prompt` | `text` → `test_list` | 474 | CC-BY-4.0 | never `test` (our eval) |

A code answer counts only if it passes executable tests, so every source must ship tests. Competitive problems get a 1,024-token answer cap; the evals stay at 512.

### Science (150M)

| dataset | subset / split | prompt → answer | size | licence | notes |
|---|---|---|---:|---|---|
| [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified) | `train`, `category` in Physics, Chemistry, Biology | `question` → `answer` (`answer_type` says how to compare) | 228,736 (all categories) | Apache-2.0 | free-form, so it can't be guessed |
| [nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | `MCQ`, `RQA` | user turn → letter or `\boxed{}` in the assistant turn | 174,155 + 52,179 | CC-BY-4.0 | 10 options, so a guess is right 10% of the time |
| [nvidia/OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) | `train` | `input` → `expected_answer` | 1.6M | CC-BY-4.0 | |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | `ARC-Challenge` + `ARC-Easy` / `train` | `question` + `choices` → `answerKey` | 3,370 | CC-BY-SA-4.0 | easy anchor |

Half free-form, half multiple choice. Nemotron-Science-v1 and OpenScienceReasoning-2 were built to resemble GPQA. They get an extra embedding-based near-duplicate check against GPQA, and one small run without them tests whether any GPQA gain survives.

### Instruction following (150M)

| dataset | split | prompt → check | size | licence |
|---|---|---|---:|---|
| [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | `train` | `prompt` → `instruction_id_list` + `kwargs` | ~46K | ODC-BY |
| [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | `train` | `messages` → `ground_truth` | 95,373 | ODC-BY |
| [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | `train` | `messages` → `ground_truth` | 14,973 | ODC-BY |

These supply about 15% of the slice; round 3's strict filter kept only ~6K prompts from them. The other 85% we compose the way NVIDIA built its set:
- Take an English first-turn prompt from [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (ODC-BY), disjoint from the chat prompts.
- Add 1–3 constraints from the 30 IFEval checks round 3 allowed, worded exactly as the checker states them.
- Score strictly with the vendored IFEval checker: every constraint must pass.

### General chat (200M)

| dataset | split | prompt | size | licence | share |
|---|---|---|---:|---|---:|
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | `train` | first user turn, `language == "English"`, `toxic == false` | 837,989 conversations | ODC-BY | ~55% |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | `prompt` | 207,865 | MIT | ~43% |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | `train` | root messages (`role == "prompter"`, no `parent_id`), `lang == "en"` | 13,854 trees | Apache-2.0 | ~2% |

An answer is kept if it ends within 1,024 tokens, has no repetition loop, and is in English.

### Held-out test domain (never trained on)

| dataset | split | used | licence |
|---|---|---:|---|
| [dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa) (mirror of `ibm-research/finqa`) | `test` | 500 of 1,147 | CC-BY-4.0 (upstream) |
| [Multilingual-Multimodal-NLP/TableBench](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) | `TQA_test`, fact-checking + numerical reasoning | ~300 of 886 | Apache-2.0 |

New benchmarks next to these: [google/IFEval](https://huggingface.co/datasets/google/IFEval) (541, Apache-2.0), and chat judged pairwise against the released model.

### Not used, and why

- **Contaminated:**
  - `agentica-org/DeepScaleR-Preview-Dataset` contains AIME 2022–23, which is part of our AIME eval.
  - `qwedsacf/competition_math` includes the MATH test set in its train split.
  - `m-a-p/SuperGPQA` lists our AIME eval among its sources.
- **Non-commercial licences:** KodCode, SciQ, MegaScience, `facebook/natural_reasoning`, camel-ai.
- **Licence problems:** `lmsys/lmsys-chat-1m` can't be redistributed, OpenHermes-2.5 has no licence, and Magpie comes with Llama naming terms.
- **Chart QA** needs images.
- **The FLARE snapshot** (`Arushhh/Llama-HybridDiffusion-processed-data-run1`): answers are 3K–27K-token `<think>` traces, its maths answers are unverified, and its IF rows lost their constraint data. We keep only its domain shape.

## From prompt to blocks

1. **Decontaminate.** Check every prompt against all eval sets: GSM8K test, the whole MATH test set, AIME 2022–25, GPQA, HumanEval, MBPP, IFEval, IFBench and the held-out tables.
   - Exact match plus 13-gram overlap; positive controls must catch 100%.
   - An embedding check for the GPQA-style sources.
   - Near-duplicates removed within each domain.
2. **Answer.** SDAR-30B answers each prompt: block 32, threshold 0.95, cap 512 tokens for code and 1,024 otherwise.
3. **Check.** Exact answer for maths and science, unit tests for code, the IFEval checker for IF.
4. **Draft.** SDAR-1.7B drafts each kept block from the correct prefix, exactly as at inference.
5. **Cut.** `build_train_set.py` cuts each kept answer into 32-token blocks.

**Why the 30B's own answers.** The judge learns whether the 30B would commit a token, so training on another model's answers would teach it to agree with that model instead. Target-generated data beats fixed data for EAGLE, and dParallel and VAT (NeurIPS 2026) train on the target's own outputs too.

## Experiments this corpus feeds

- **Scaling:** checkpoints at 1.84M, 8M, 32M, 128M, 512M and 1B tokens, all from one run.
- **Diversity:** this mix vs round 3's mix, at the same 1.84M tokens.
- **Generalisation:** five 32M runs that each leave out one domain, plus the never-trained tables set.
- **Gates:**
  - First reproduce round 3's judge with the new pipeline (plain decoding instead of blind accept) to within 2 points.
  - Generate the last 872M tokens only if 128M still beats 32M.

## Timeline (8× RTX PRO 6000 Blackwell, 96 GB)

| step | where | GPU-hours | time |
|---|---|---:|---:|
| Build a batched decoding engine and a faster trainer | engineers | — | ~2 weeks |
| Build prompt pools (download, filter, decontaminate) | CPU box | — | ~3 h |
| Benchmark engine and trainer | 1 GPU | 2 | 2 h |
| Generate answers: 3.3M prompts through SDAR-30B, plus 1.7B drafts | 8 GPUs, or separate single GPUs with ≥ 80 GB | 465 | 2.4 days (1.8–3.0) |
| Check answers, cut into 12.8M blocks | CPU box, alongside generation | — | ~2 h |
| Train the judge, 1 epoch over 1B tokens | 8 GPUs | 86 | 11 h (8–14) |
| Evaluate: 9 benchmarks, with and without DES48 | 8 GPUs | 183 | 23 h |
| **Total** | | **~740** | **~4 days of GPU time after the engineering** |

- **Where the ranges come from:** the engine and trainer are assumed to reach 12–20% of the GPU's 503.8 BF16 TFLOPS. The 2-hour benchmark measures this before we rent 8 GPUs.
- **Generation doesn't need the 8-GPU box.** Each prompt is independent and a job resumes after interruption, so it runs on cheaper single GPUs (the 30B uses 61 GB). Only training needs the 8 GPUs together.
- **IF is the expensive domain.** It is ~40% of generation time for 15% of tokens: only 35% of answers pass, and the model takes 22 passes per block. Moving 5 points of share from IF to code cuts generation by ~12%.
- **Today's code can't do this.** Round 3 spent 46.8 GPU-hours to make 2.06M tokens; at that rate 1B tokens takes 4–6 months on 8 GPUs. That is why the engine and trainer come first.

Calculator: [`corpus_1b_budget.py`](corpus_1b_budget.py).
