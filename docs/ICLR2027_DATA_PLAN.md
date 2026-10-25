# Data plan — three phases

What to draw, from where, how much. Nothing else.
Unit = **block** (32 slots). Today's set = **65,521 blocks**: GSM8K 62.7%, MATH 37.0%, MBPP 0.2%.
The problem this mix fixes: code is 0.2% of training, and code is where the judge breaks.

> **2026-09-14 — round 3 draws per `docs/TRAINING_RECIPE_2026-09-09.md` → *Amendments 2026-09-14
> (approved)*, not per the phases below:** ~32,000 blocks = 35% code (`nvidia/OpenCodeInstruct`,
> frozen 12,000-item pool) · 35% maths (`openai/gsm8k` only, re-decoded) · 30% program-verifiable
> IF (`allenai/IF_multi_constraints_upto5` + `allenai/RLVR-IFeval`). Its code prompts follow recipe
> §3.5 (target 40% HumanEval-shape, landed at 34.1%), not Arm A's 60% docstring-completion split
> below. `nvidia/OpenCodeInstruct` is now ✅ VERIFIED.

## Mix at a glance

```
                GSM8K        MATH              CODE            SCIENCE
today   65,521  ██████████▏  ██████▏           ▏               ·
        %       62.7         37.0              0.2             0

P1-A   114,187  ██▏          ███████████▏      ████            ·
        %       12.3         64.8              23.0            0

P2     160,187  █▍           ████████▏         ██████▏         █▊
        %        8.7         46.2              35.1            10.0

P3     236,187  █            ██████▍           ███████▏        ██▉
        %        5.9         36.4              40.7            16.9
```
Code 0.2% → 23% → 35% → 41%. GSM8K 62.7% → 5.9%. MATH holds absolute mass, falls in share.

## Phase 1 — MVP gate (three arms, matched mass)

Shared base **88,187 blocks**, then each arm adds 26,000 so only the *kind* of data differs.

| # | source | split / selection | domain | new prompts | blocks | verified |
|---|---|---|---|---:|---:|---|
| 1 | `openai/gsm8k` | `main`, train pool — subsample of the 41,093 already on disk | maths short | 0 | 14,000 | ✅ local |
| 2 | `EleutherAI/hendrycks_math` | train, decoded slice `0:7500:3` + 23 collision-fix items | maths long | 0 | 24,598 | ✅ local |
| 3 | `google-research-datasets/mbpp` | `sanitized` train (120) + 1 collision-fix item | code | 0 | 163 | ✅ local |
| 4 | `EleutherAI/hendrycks_math` | train, **undecoded**, Level 4 (1,160) + Level 5 (1,516) | maths hard | 2,676 | 24,150 | ✅ local |
| 5 | `EleutherAI/hendrycks_math` | train, **undecoded**, Levels 1–3 (2,323) | maths easy/med | 2,323 | 25,208 | ✅ local |
| 6 | `google-research-datasets/mbpp` | `sanitized` `prompt` (7) + `validation` (43) | code | 50 | 68 | ✅ local |
| | **shared base** | | | **5,049** | **88,187** | |

| arm | adds | blocks | total | download? |
|---|---|---:|---:|---|
| **R** | nothing — today's index, collision-fixed | 0 | 65,854 | no |
| **B** | 26,000 more *existing* GSM8K blocks | 26,000 | **114,187** | no |
| **A** | 26,000 **new code** blocks | 26,000 | **114,187** | **yes** |

**Arm A code source:** `nvidia/OpenCodeInstruct` `train` ✅ **VERIFIED 2026-09-14** (700/700 sampled rows carry tests; 12,000-item pool built, `code/configs/opencode_pool.json`) — 60% of the draw transformed to docstring-completion shape (signature + docstring → body), 40% left in native instruction shape so the fix is not format-specific. Requires in-row executable tests; without them this slice cannot exist.

## Phase 2 — submission

| # | source | split / selection | domain | new prompts | blocks | verified |
|---|---|---|---|---:|---:|---|
| 1–5 | carried from Phase 1 | GSM8K 14,000 · MATH 73,956 · MBPP 231 | maths + code | 0 | 88,187 | ✅ local |
| 6 | code corpus (`nvidia/OpenCodeInstruct`) | Phase-1 draw 26,000 **+ 30,000 new**, disjoint by row id | code | ~15,000 | **56,000** | ✅ VERIFIED 2026-09-14 |
| 7 | science corpus — `TIGER-Lab/MMLU-Pro` (preferred), else `nvidia/Llama-Nemotron-Post-Training-Dataset` `science` or `nvidia/Nemotron-Post-Training-Dataset-v1` `stem` | short machine-checkable gold required | science | ~4,000 | **16,000** | ⚠️ UNVERIFIED |
| | **total** | | | **~19,000** | **160,187** | |

## Phase 3 — rebuttal

| # | source | selection | domain | blocks | verified |
|---|---|---|---|---:|---|
| 1 | everything from Phase 2 | carried | — | 160,187 | mixed |
| 2 | code corpus | +40,000, disjoint by row id from Phases 1–2 | code | **+40,000** | ✅ VERIFIED 2026-09-14 (`nvidia/OpenCodeInstruct`) |
| 3 | science corpus | +24,000 (→ 40,000 total) | science | **+24,000** | ⚠️ UNVERIFIED |
| 4 | `EleutherAI/hendrycks_math` | L4+L5 rejection sampling, 3 extra attempts on the 2,676 hard items | maths hard | **+12,000** | ✅ local |
| | **total** | | | **236,187** | |

## Never drawn from — held out

`GSM8K-500` · the whole `openai/gsm8k` test split · the 200 frozen calib/valid indices · **the entire `EleutherAI/hendrycks_math` test split** (5,000 — MATH-500 is drawn from it) · `mbpp/sanitized/test` (257) · HumanEval · GPQA-Diamond · AIME.
`mbpp` `train`/`prompt`/`validation` **are** training data — existing protocol, unchanged.
Contamination check run: **0 of 500** MATH-500 items appear in the hendrycks_math train split.

## Verify before buying anything

| id | what to check | why it blocks |
|---|---|---|
| `nvidia/OpenCodeInstruct` ✅ **VERIFIED 2026-09-14** | resolves; licence; config/split names; **fraction of rows with a non-empty in-row `unit_tests`** — resolved: cc-by-4.0, ungated, split `train`; 700/700 sampled rows carry tests (median 10 asserts); 12,000-item pool built (`code/configs/opencode_pool.json`) | the whole code programme rests on it (26k / 56k / 96k blocks). No executable tests → no correctness filter → no code slice |
| `TIGER-Lab/MMLU-Pro`, `…/Llama-Nemotron-Post-Training-Dataset` `science`, `…/Nemotron-Post-Training-Dataset-v1` `stem` | split names are recollections — a wrong one **loads empty rather than erroring** | Phase 2 science slice |
| `google-research-datasets/mbpp` config `full` | repo present, only `sanitized` cached | ~500 extra never-decoded code prompts |
| `nvidia/OpenCodeGeneticInstruct`, `nvidia/OpenScienceReasoning-2` | **the id strings may be confabulated** — search, don't fetch | optional second code/science format |

Pull **one parquet shard** to check, never the repo.
