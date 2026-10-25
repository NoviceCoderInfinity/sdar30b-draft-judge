# DATA

Every dataset in this project, its exact Hugging Face identity, its item count,
and how the training pool was built. All splits are defined in one place —
`code/run_desvote.py::build_items` — which `code/research_protocol.py` delegates
to, so evaluation and training-data generation cannot drift apart.

Frozen selections live in `code/configs/`, are sha256-checked on load, and are
committed to this repo:

| file | what |
|---|---|
| `code/configs/prompts.json` | frozen train pools + their provenance + the decontamination audit. sha256 `16ef849b10194a3f391f1d59786878565822a4e7f4e409bd97d9300857a82aad` (also in `prompts.json.sha256`) |
| `code/configs/gsm8k_test_500.json` | the 500 GSM8K test indices |
| `code/configs/research_protocol_v1.json` | the frozen experiment protocol; its sha goes into every launch record |
| `code/configs/opencode_pool.json` | round-3 code pool: 12,000 `nvidia/OpenCodeInstruct` rows + build provenance and decontamination audit (§7.2). sha256 `d636ff2eeca76c0051073278ec805bf1fd13d2baa7d231dd0c856b04ce24eadb` (also in `opencode_pool.json.sha256`) |
| `code/configs/if_pool.json` | round-3 verifiable-IF pool + filter funnel and audit (§7.3). sha256 `2d3761e9ce8fe56705f40183d1d987a3a947ef5af289ba9e100d96c80446367d` as built 2026-09-14T14:49Z — **TBD (IF pool build in progress)**: check `if_pool.json.sha256` before quoting |

---

## 1. The six evaluation datasets

All decoded greedy, block 32, threshold 0.95 (strict `>`), chat template applied
by `research_protocol.encode_prompt`.

| split name | HF dataset | revision / config | HF split | **n** | gen cap | scorer |
|---|---|---|---|---:|---:|---|
| `gsm8k500` | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866`, config `main` | `test`, filtered | **500** | 1024 | `math_postprocess_v2` + `MATHEvaluator(v2).is_equiv` against `gsm8k_dataset_postprocess(gold)` |
| `math500` | `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `test` | **500** | 1024 | `math_postprocess_v2` + `MATHEvaluator(v2).is_equiv` against the dataset's own `answer` |
| `mbpp` | `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, config `sanitized` | `test` | **257** | **512** | `MBPPEvaluator` — **executes the asserts**; per-item `is_correct` from its `details` |
| `aime` | `AI-MO/aimo-validation-aime` | — | **`train`** (the only split) | **90** | 1024 | integer match on the last `\boxed{}`; golds are 0-padded to 3 digits and canonicalised with `int()` |
| `gpqa` | `fingertap/GPQA-Diamond` (ungated Diamond mirror) | — | `test` | **198** | 1024 | multiple-choice letter extraction, compared to `answer.strip().upper()` |
| `humaneval` | `openai_humaneval` | — | `test` | **164** | **512** | executes the reference `test` against the generated `entry_point` |

Notes that matter:

* **`aime` reads the `train` split.** `AI-MO/aimo-validation-aime` has no test
  split; the 90 items are the whole validation set. This is not a bug.
* **`gpqa` is a mirror, not the gated original.** `fingertap/GPQA-Diamond` is an
  ungated copy of GPQA-Diamond's 198 multiple-choice items. Numbers are not
  directly comparable to published GPQA-Diamond scores that use the gated
  release and a different prompt.
* **MBPP and HumanEval get half the generation budget** (512 vs 1024) —
  `research_protocol.GEN_LEN`. A missing key there raises, which is the one place
  adding a dataset fails loudly.
* **MATH-500 accuracy is not comparable to a published SDAR number.** No SDAR
  config for MATH-500 was gated in the original P1 protocol; it is valid as a
  relative comparison across arms only (`analyze_s1b.py` docstring).
* **`gsm8k500` is a frozen random subset**, not the full test set:
  `sorted(random.Random(0).sample(range(1319), 500))`, frozen 2026-09-03, with
  `sha256_of_questions = 47fd837e44fdb380d7113a913cc352e4eca477e67907e1c90ec3402ad722f239`.
  `idx` is the index into the **full** 1,319-item test split, so items pair
  exactly with any other run on that split.

Scoring code is vendored, not pip-installed: `code/vendor/oc_datasets/{gsm8k,
math,mbpp}.py` plus a minimal `code/vendor/opencompass/` shim, copied verbatim
from `JetAstra/SDAR@main` so accuracy is byte-identical to the harness that
produced the published SDAR numbers. Provenance and hashes are in
`code/vendor/PROVENANCE.md` and `code/vendor/VENDORED.sha256`.
`analyze_s1b.py::per_item_correct` is the single dispatch point; it adds the
`aime`, `gpqa` and `humaneval` branches on top of the vendored three.

## 2. The held-out development split

`calibvalid` — **100 frozen GSM8K *train* prompts** (50 calibration + 50
validation), the first 100 of the 200 frozen train items in `prompts.json`.
Thresholds were tuned here. These 100 are **excluded from every training set**
(`build_train_set.py --exclude-frozen`), so they are held out from the judge, but
they are *train*-distribution, not test. Every accuracy on `calibvalid` should be
read as a development number.

`research_protocol` also exposes `calib` (the first 50) and `valid` (the next 50)
separately, and a slice syntax `name:lo:hi[:step]` with Python semantics, e.g.
`--items math_pool:0:7500:3`.

## 3. The training pools

Declared in `prompts.json`, decontaminated against the test sets before anything
was decoded:

| pool | HF dataset | revision / configs | split | pool size | frozen items removed |
|---|---|---|---|---:|---:|
| `gsm8k_pool` | `openai/gsm8k` | `7403…9866`, `main` | `train` | **7,473** | the 200 frozen train ranks |
| `math_pool` | `EleutherAI/hendrycks_math` | `21a5…f4a3`, 7 subject configs concatenated in the frozen order | `train` | **7,500** | 200 frozen items (hash-checked) |
| `mbpp_pool` | `google-research-datasets/mbpp` | `4bb6…030c`, `sanitized` | `train` | **120** | 100 frozen items |

MATH subject concat order (this defines `global_index`): `algebra,
counting_and_probability, geometry, intermediate_algebra, number_theory,
prealgebra, precalculus`.

**Decontamination audit** (`prompts.json::decontamination_validation`): method is
whitespace-normalised exact text match. Overlaps found: GSM8K train↔test **0**,
MATH train↔MATH-500 **0**, MBPP train↔test **0**. Positive controls both pass at
100% (`math500_items_found_in_math_test_split: 500/500`,
`gsm8k_test_self_match: 1319/1319`), which is what makes the three zeros
meaningful rather than a broken matcher.

Round 3 adds two pools declared outside `prompts.json` — `opencode_pool`
(`code/configs/opencode_pool.json`) and `if_pool` (`code/configs/if_pool.json`), each carrying
its own decontamination audit — and re-decodes a `gsm8k_pool` slice (§7).

## 4. How the judge's training data was made

This section is **round 2** — `trainset_union_r2`, the published judge. Round 3 runs the same
three stages on new pools (§7).

The training set is **the model's own correct answers, re-decoded under the
blind rule, then corrupted**. Three stages:

### Stage 1 — decode the pools with the training-free blind rule

```bash
run_da_decode.py --model b32 --drafter d32 --items gsm8k_pool:0:3000 \
  --draft s8 --judge blind --tau-acc-q .50 --shard k --nshards 8 --gpu {GPU} --score \
  --out logs/draftalign/datagen/gsm8k_pool_0_3000_dv-s8_blind_q50
```

Four datagen runs (`math_pool:0:7500:3` takes every third MATH problem;
`mbpp_pool` is all 120). Each writes per-item records **and per-block NPZ
traces** — the traces are what make the corruption reproducible, because they
record which slots the real trajectory had committed before each pass, and what
the drafter proposed.

### Stage 2 — build the index

```bash
bash code/draftalign/build_union_index.sh
# -> build_train_set.py --only-correct --max-seq 1200 --exclude-frozen  (x4)
# -> logs/draftalign/trainset_union_r2/index.jsonl
```

Filters: **`--only-correct`** (the model must have got the answer right — the
target trajectory has to be a good one), `--max-seq 1200`, `--exclude-frozen`
(never a calibvalid or frozen-rank prompt), and the test splits are excluded by
construction.

Result, from the four `build_summary.json` files:

| source | kept items | dropped: incorrect | dropped: frozen | usable blocks |
|---|---:|---:|---:|---:|
| `gsm8k_pool:0:3000` | 2,843 | 157 | 0 | 16,905 |
| `gsm8k_pool:3000:7273` | 4,043 | 230 | 0 | 24,188 |
| `math_pool:0:7500:3` | 1,684 | 779 | 37 | 24,267 |
| `mbpp_pool` | 68 | 50 | 2 | 161 |
| **total** | **8,638** | **1,216** | **39** | **65,521** |

**8,638 of 9,893 prompts kept** (8,638 + 1,216 + 39 = 9,893), **65,521 blocks**.
`index.jsonl` is exactly 8,638 lines. Every block has a draft
(`draft_blocks == usable_blocks` in all four summaries).

Mix by **block** (the unit of a training example): GSM8K 62.7 %, MATH 37.0 %,
MBPP 0.2 %. By **item**: GSM8K 79.7 %, MATH 19.5 %, MBPP 0.8 %. RESULTS.md §5
uses the block share.

MBPP is effectively absent — 161 blocks out of 65,521 — because only 68 of the
120 sanitized-train problems were solved correctly, and each is short. This is
the single best predictor of where the judge damages accuracy (MBPP: −14.0).

### Stage 3 — corrupt on the fly

`dataset.py::StateSampler` draws a fresh state each time an (item, block) example
is visited; nothing is pre-materialised. See `ARCHITECTURE.md` §6 for the exact
distribution. The run in use: `p_pass1 = 0.4`, `p_draftfull = 0.3`,
`p_allmask = 0.0`, `max_seq = 1200`, 150 held-out items, seed 1.

## 5. Where the raw decodes are

`logs/draftalign/datagen/` (four directories, ~32k files) and
`logs/draftalign/da_decode/` — neither is in this repo. See `EXCLUDED.md`.

Round 3's decodes are under `logs/draftalign/datagen/*_r3/` and its indices under
`logs/draftalign/trainset_r3/<pool>/`. On the 2026-09-14 container `logs` is RAM-backed
(`/dev/shm`); both are backed up to the public HF folder
`Nemotron-data-subset-training-diffusion-moe/` every 30 min (`HUGGINGFACE.md` §9).

## 6. Adding a dataset — the checklist

1. `code/run_desvote.py::build_items` — the item list, prompt template and gold.
2. `code/research_protocol.py::GEN_LEN` — the generation cap (**a missing key
   raises**; this is the only step that fails loudly).
3. `code/research_protocol.py::test_items` — add the name to its `assert`.
4. `code/analyze_s1b.py::per_item_correct` — the scorer branch.
5. `code/draftalign/run_da_decode.py` lines ~333–339 **and**
   `code/run_research_baselines.py` lines ~196–202 — the dataset-family map.
   **Both.** The fall-through is `gsm8k`, silently.
6. `code/draftalign/eval_table.py::splits` and
   `code/draftalign/compare_table.py::SPLITS` — otherwise the cells exist on
   disk and appear in no report (which is the current state for AIME, GPQA and
   HumanEval).

A **training pool** takes step 1 as a loader in `research_protocol.POOLS` instead of
`build_items`; `items_for_split` resolves `POOLS` before it ever reaches `test_items`.
The round-3 pools against this checklist:

| step | `opencode_pool` | `if_pool` |
|---|---|---|
| 1. items | `research_protocol.opencode_pool_items` in `POOLS` — **done** | TBD (IF pool build in progress) — `docs/IF_POOL_WIRING_PATCH.md` |
| 2. `GEN_LEN` | `"opencode_pool": 512` — **done** | TBD (IF pool build in progress); the pool was sized for a 1,024-token generation (`if_pool.json::length_fit.gen_len`) |
| 3. `test_items` | not needed (a pool) | not needed (a pool) |
| 4. scorer | reuses `per_item_correct("mbpp")` — no new branch | `code/draftalign/if_scorer.py` exists; its branch TBD (IF pool build in progress) |
| 5. family map, **both** runners | `"mbpp" if base.startswith("opencode")` — **done in both** | the `if*` line — TBD (IF pool build in progress) |
| 6. report dicts | not needed (a training pool, not an eval set) | not needed |

Until step 2 lands, `run_da_decode.py` fails loudly on an `if_pool` split at `GEN_LEN[base]`.
A patch that adds steps 1–2 but not step 5 would score every IF item with the GSM8K
extractor — silently.

## 7. Round-3 pools (2026-09-14)

Target, by block of 32 slots: **35% code / 35% maths / 30% IF = 11,200 / 11,200 / 9,600
blocks**, ~32,000 in total (`docs/TRAINING_RECIPE_2026-09-09.md` §2 and its *Amendments
2026-09-14*). Every pool goes through §4's three stages unchanged — Stage 1 with the stock
30B + stock 1.7B drafter under the blind rule, Stage 2 with `--only-correct --max-seq 1200
--exclude-frozen` — one pool at a time. Stage 2 is the preprocessing gate before training.

| pool | Stage 1 out (`logs/draftalign/datagen/…`) | Stage 2 (`logs/draftalign/trainset_r3/…`) | a decode is correct when |
|---|---|---|---|
| maths | `gsm8k_pool_0_2400_dv-s8_blind_q50_r3` | `build_r3_index.sh gsm8k` → `gsm8k/` | §1's GSM8K scorer accepts the answer |
| code | `opencode_pool_0_6000_dv-s8_blind_q50_r3` | `build_r3_index.sh opencode 6000` → `opencode/` | the program passes the row's unit tests (MBPP sandbox) |
| IF | TBD (IF pool build in progress) | `build_r3_index.sh if <N>` → `if/` | the response passes every constraint (`if_scorer.score_ifeval`) |

### 7.1 Maths — GSM8K only, re-decoded

| | |
|---|---|
| pool | `gsm8k_pool` — `openai/gsm8k` @ `7403…9866`, `main`, `train`; 7,273 items once the 200 frozen ranks are removed (§3) |
| slice | `gsm8k_pool:0:2400` — 2,400 prompts, 8 shards (`code/draftalign/launch_r3_datagen_gsm8k.sh`, `/venv/r3`) |
| reused from round 2 | **nothing** — the same prompts sit inside round 2's `gsm8k_pool:0:3000` decode (§4), which is not used |
| measured so far (run in progress) | ~92–94% correct · 6.30 blocks/item · 16.1 s/item on one RTX PRO 6000 Blackwell |
| frozen guard | items carry split `train_pool`, so `--exclude-frozen` applies; round 2's GSM8K slices dropped 0 there (§4) |
| MATH | not in the round-3 mix — a deviation from recipe §2 (*Amendment* A1) |
| arm-C filler | `gsm8k_pool:2400:4400` — 2,000 prompts, 32 shards, queued low-priority → `gsm8k_pool_2400_4400_dv-s8_blind_q50_r3` |

### 7.2 Code — `nvidia/OpenCodeInstruct`, a frozen 12,000-item pool

| | |
|---|---|
| source | `nvidia/OpenCodeInstruct` @ `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`, split `train`, 50 parquet shards (5,000,000 rows; 1,641,305 at `average_test_score == 1.0`), each shard's sha256 checked against the download metadata before it is read; CC-BY-4.0 |
| builder | `code/draftalign/build_opencode_pool.py` (CPU only) → `code/configs/opencode_pool.json` + `.sha256` |
| loader | `research_protocol.opencode_pool_items()` — sha256-checked on load; `idx` = rank in the frozen hash order, 0–11,999; split `opencode_pool`, so `build_train_set`'s frozen-GSM8K guard cannot fire |
| scan order | every row sorted by sha256(id), walked from the smallest hash across all 50 shards; stops at the 12,000th row that passes every filter |
| gold | the row's `unit_tests`, each entry stripped, joined by newlines; scored by `analyze_s1b.per_item_correct("mbpp")`, which **executes** the asserts |
| `GEN_LEN` | 512, as MBPP |
| cell prefix | `mbpp_` (the scorer family): `mbpp_B32_t95_dv-s8_blind_q50.opencode_pool:0:6000.shard{k}of32.jsonl` |

Filter funnel (`opencode_pool.json::counts`), in recipe §3.3's order:

| step | kept | dropped |
|---|---:|---:|
| rows scanned | 39,708 | — |
| `average_test_score == 1.0` | 12,967 | 26,741 |
| contamination: unit-test assert lines vs MBPP-257, function bodies vs HumanEval-164 | 12,935 | 32 — MBPP 10, HumanEval 22, both 0 |
| length fit: templated prompt + reference output ≤ 1,200 tokens (b32 tokenizer) | 12,044 | 891 |
| dedup on normalised `input` | **12,000** | 44 |

**Decontamination.** Normalised exact match (all whitespace removed, lowercased) of every
unit-test `assert` line against MBPP-257's 776 assert lines (`sanitized` test @ `4bb6404f…`),
and of every function body in the reference output, with and without its docstring, against
HumanEval-164's canonical solutions (@ `7dce6050…`). Positive controls: 257 / 257 MBPP and
164 / 164 HumanEval items flagged; matching whole outputs instead flags 0 / 164.

**Prompt shapes — a small deviation from recipe §3.5.**

| | MBPP shape | HumanEval shape |
|---|---:|---:|
| §3.5 target | 60% | 40% |
| assigned by `sha256("template:" + id) % 100 < 40` (the 12,935 rows past contamination) | 7,733 | 5,202 |
| **selected** | **7,906** (65.9%) | **4,094** (34.1%) |
| first 6,000 (the queued slice) | 3,973 | 2,027 |

A HumanEval-assigned row whose stub (imports + the signature of the one function the tests
call + the input as docstring) cannot be derived cleanly gets the MBPP shape. 1,090 rows fell
back:

| reason | rows |
|---|---:|
| tests call a class | 663 |
| input unfit for a docstring | 155 |
| tests use other definitions | 132 |
| tests call several functions | 57 |
| tests call no defined function | 47 |
| decorated or one-line def | 28 |
| stub does not round-trip | 8 |
| **total** | **1,090** |

| lengths (b32 tokenizer, templated prompt via `encode_prompt`) | |
|---|---:|
| median prompt | **405** tokens |
| median reference answer | **172** tokens |
| median total | 590 tokens |
| reference answers over `GEN_LEN` 512 | 114 |

**Scorer gate.** The reference outputs of 200 rows pass **193 / 200** through the MBPP scorer.
The 7 misses are row defects: flask not installed, missing fixtures, nondeterministic tests,
one contradictory assert.

**Decode.** `opencode_pool:0:6000` in 32 stride shards (`items[k::32]`, 187–188 items each),
queued high-priority in `logs/draftalign/queue_r3_hi.txt`, each shard in its own sandbox cwd
(`AGENTS.md` §9 (p)). Any finished subset of shards is a uniform sample of the slice, so it is
trimmed to the 11,200-block target once yield is measured (`AGENTS.md` §9 (n)).

### 7.3 IF — program-verifiable instruction following

Approved 2026-09-14 in place of recipe §2's `Nemotron-IF-Domain-Cascade`; the rationale is
*Amendment* A2. The numbers below are from `code/configs/if_pool.json` as built at
2026-09-14T14:49Z (sha256 `2d3761e9…`). The IF recipe and audit (`docs/IF_RECIPE_2026-09-14.md`)
and the runner wiring (`docs/IF_POOL_WIRING_PATCH.md`) are **TBD (IF pool build in progress)**.

| | |
|---|---|
| sources | `allenai/IF_multi_constraints_upto5` @ `2e3a77407b7fce69f95b248d64a884e3ae1c2423` (95,373 rows) · `allenai/RLVR-IFeval` @ `47c03c73621c4aab2b824b7818681117d662770e` (14,973 rows) · both split `train`, ODC-BY |
| checker | open-instruct IFEvalG @ `172e379eada34137885bd1c36543c4f59cb113e4`, vendored byte-identical in `code/vendor/ifeval/`; 54 instruction ids, 30 admitted, 24 excluded with a reason each (`if_pool.json::id_policy`) |
| scorer | `code/draftalign/if_scorer.py::score_ifeval(pred, gold)` — IFEval strict mode; correct **iff every constraint passes**; only the text after the last `</think>` is scored; an unclosed `<think>`, a checker that raises, or a generation without a clean EOS scores 0 |
| gold | `{"instruction_id_list": [...], "kwargs": [...]}` — google/IFEval's input format |
| prompt | the dataset's single user message, verbatim; `encode_prompt` applies the chat template |
| builder | `code/draftalign/build_if_pool.py --n 6000` |
| selected | **1,803** of 6,000 requested — every eligible row: RLVR-IFeval 534, IF_multi 1,269; 1 / 2 / 3 constraints = 1,173 / 410 / 220 |
| prompt tokens | median 71, p99 539, max 594 (limit 600) |
| decontamination | normalised exact match or containment against google/IFEval (541), allenai/IFBench_test (300) and the six eval sets; 2 hits dropped; positive controls 541 / 541, 541 / 541, 50 / 50 |
| `POOLS`, `GEN_LEN`, dispatch, decode slice | TBD (IF pool build in progress) |
| yield | unmeasured. 9,600 blocks from 1,803 items needs ≥ 5.32 kept blocks per item even if every item passes |

Filter funnel (`if_pool.json::filters`), rows kept after each step:

| filter | kept |
|---|---:|
| loaded | 110,346 |
| single turn | 110,346 |
| gold parses and maps (RLVR functions or relations with no IFEval equivalent dropped) | 106,632 |
| every id implemented | 106,632 |
| id policy (the 24 excluded ids) | 38,979 |
| 1–3 constraints | 36,429 |
| deterministic (identical checker args under two RNG seeds) | 36,135 |
| feasible in a 1,024-token generation | 32,944 |
| constraint text verbatim in the prompt | 32,944 |
| origin (maths, code, tables, science, FLAN, safety probes, persona prompts dropped) | 13,209 |
| task screen (code-, maths- or QA-looking base prompts dropped) | 7,980 |
| English base prompt | 3,941 |
| dedup | 1,908 |
| prompt ≤ 600 tokens | 1,805 |
| decontamination | **1,803** |

The Nemotron route stays closed unless constraint recovery on those rows reaches ≥ 99%
hand-audited precision.
