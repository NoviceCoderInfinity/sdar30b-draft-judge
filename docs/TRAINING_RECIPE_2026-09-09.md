# Training recipe — round 3 (2026-09-09)

## Amendments 2026-09-14 (approved)

Approved by the user on 2026-09-14. Where an amendment and the 2026-09-09 text below
disagree, the amendment wins; the original text is kept unchanged as the record of what
was planned. §6's last open item ("IF has no automatic verifier") is resolved by A2.

| # | amendment | overrides | state on 2026-09-14 |
|---|---|---|---|
| A1 | **Maths = GSM8K only, re-decoded fresh with the stock models** — no MATH, no round-2 decode reused | §2 maths row ("MATH + GSM8K, already decoded on disk — no, free") | `gsm8k_pool:0:2400` decoding |
| A2 | **IF = program-verifiable IF** (`allenai/IF_multi_constraints_upto5` + `allenai/RLVR-IFeval`), kept only when every constraint passes a strict checker | §2 IF row (`Nemotron-IF-Domain-Cascade`), §2 "Why the IF slice comes from the processed mixture", §6 last item | pool built; wiring and decode TBD (IF pool build in progress) |
| A3 | **Code prompt shapes land at 7,906 MBPP / 4,094 HumanEval**, not 60 / 40 — a small deviation | §3.5 | pool frozen and wired; `opencode_pool:0:6000` queued |
| A4 | **6 GPUs train + 2 GPUs evaluate**; batch 6 × 5 = 30 or 6 × 6 = 36 states/step, **TBD** | §4 `world × accum` (8 × 4 = 32), §4.3 wall-clock on 8 GPUs | not started — preprocessing gate first |

**Unchanged:** the shares and targets — 35% code / 35% maths / 30% IF = 11,200 / 11,200 /
9,600 blocks of 32 slots, ~32,000 in total, with IF held at ~30% at the user's explicit
request; the §3.1 pipeline and its Stage-2 filters (`--only-correct --max-seq 1200
--exclude-frozen`); `--pool-Ms 128`; the LoRA configuration; and §4.1's fresh start from the
stock base — `JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc37d240aa6871f167e8f92d694281b0c22` with the
stock drafter `JetLM/SDAR-1.7B-Chat-b32` @ `2c83c05d84be90a01c72443b2123ebf0376f6fd4`, block 32
everywhere. The team checkpoints (`t2b_union_flag/step003057` and every other step) are
documentation only, never init or resume — inventory in
`docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md`.

### A1 — Maths: GSM8K only, decoded again

| | |
|---|---|
| slice | `gsm8k_pool:0:2400` — the first 2,400 of the 7,273-item pool |
| decode | §3.1 Stage 1 unchanged, stock 30B + stock 1.7B drafter, 8 shards (`code/draftalign/launch_r3_datagen_gsm8k.sh`) |
| out | `logs/draftalign/datagen/gsm8k_pool_0_2400_dv-s8_blind_q50_r3` |
| reused | **nothing** — these prompts were also decoded in round 2's `gsm8k_pool:0:3000`; those records are not used |
| measured so far | ~92–94% correct · 6.30 blocks/item · 16.1 s/item on one RTX PRO 6000 Blackwell |
| sizing (launcher note) | 2,400 prompts × ~5.95 blocks × ~0.947 survival ≈ 13.5k blocks against the 11,200 target |
| MATH | not in the round-3 mix |
| arm C (§5) | its extra GSM8K comes from `gsm8k_pool:2400:4400`, queued as low-priority filler |

§2's "**no — free**" no longer holds for maths: it is decoded like the other two domains.

### A2 — IF: program-verifiable prompts, gated on every constraint (a deviation)

**Why.** The IF set on the user's Hugging Face account —
`Arushhh/Llama-HybridDiffusion-processed-data-run1` / `Nemotron-IF-Domain-Cascade` — is raw
Nemotron chat IF. Its README says it is **not** NVIDIA's constraint-prefiltered ~362K selection
used by the FLARE paper (https://tokflare.github.io/, arXiv 2606.01774), and its rows carry no
machine-checkable constraint arguments, so `--only-correct` has nothing to gate on (§6's last item).

| | |
|---|---|
| sources | `allenai/IF_multi_constraints_upto5` @ `2e3a77407b7fce69f95b248d64a884e3ae1c2423` + `allenai/RLVR-IFeval` @ `47c03c73621c4aab2b824b7818681117d662770e` |
| checker | open-instruct's IFEvalG (IFEval's 25 verifiers + IFBench-train's 29), vendored byte-identical in `code/vendor/ifeval/` — provenance in its `PROVENANCE.md` |
| scorer | `code/draftalign/if_scorer.py::score_ifeval` — strict IFEval mode; an item is correct **iff every constraint passes**; fail-closed on an unclosed `<think>`, a checker that raises, or a generation that does not end cleanly at EOS |
| gate | Stage 2 `--only-correct`: only the 30B's decodes that pass every constraint become blocks |
| frozen pool | `code/configs/if_pool.json`, built by `code/draftalign/build_if_pool.py`; numbers in `DATA.md` §7.3 |
| recipe + audit | `docs/IF_RECIPE_2026-09-14.md` — **TBD (IF pool build in progress)** |
| runner wiring | `docs/IF_POOL_WIRING_PATCH.md` — **TBD (IF pool build in progress)** |
| Nemotron route | may be added later **only** if constraint recovery on those rows reaches ≥ 99% hand-audited precision |

### A3 — Code: the frozen OpenCodeInstruct pool

`code/configs/opencode_pool.json` (12,000 items, sha256 `d636ff2e…`), built by
`code/draftalign/build_opencode_pool.py`. §3.3's filters ran in §3.3's order over all 50 shards,
walked in sha256(id) order: 39,708 rows scanned → 12,967 at `average_test_score == 1.0` →
contamination drop (MBPP 10 / HumanEval 22) → length fit ≤ 1,200 (891 dropped) → dedup (44
dropped) → 12,000. Full funnel, audit and lengths: `DATA.md` §7.2.

**Template mix — the deviation from §3.5.** The id hash assigned ~60 / 40 (7,733 / 5,202 of the
12,935 rows past the contamination drop), but 1,090 HumanEval-assigned rows had no cleanly
derivable stub and fell back to the MBPP shape — 663 of them because their tests call a class:

| | MBPP shape | HumanEval shape |
|---|---:|---:|
| §3.5 target | 60% | 40% |
| selected (12,000) | **7,906** (65.9%) | **4,094** (34.1%) |
| first 6,000 (the queued slice) | 3,973 | 2,027 |

**§3.2's four fixes are in:** `opencode*` dispatches to the `mbpp` scorer in both
`run_da_decode.py` and `run_research_baselines.py`; gold is the row's unit-test asserts, scored
by `per_item_correct("mbpp")`; the split is `opencode_pool`; `GEN_LEN["opencode_pool"] = 512`.
`research_protocol.opencode_pool_items` loads the pool sha256-checked. Cells are named `mbpp_…`.

| check | result |
|---|---|
| reference outputs through the MBPP scorer | **193 / 200** pass; the 7 misses are row defects — flask not installed, missing fixtures, nondeterministic tests, one contradictory assert |
| median prompt / reference answer (templated prompt, b32 tokenizer) | 405 / 172 tokens — §2's 184 / 187 were a 700-row sample |
| decode | `opencode_pool:0:6000` queued in 32 stride shards; any finished subset is a uniform sample, so the slice is trimmed to the 11,200-block target once yield is measured |

**Two §3.4 numbers differ from the frozen build.** §3.4 reports 0 HumanEval collisions and 738
MBPP-257 assert lines (12-shard sample). The build counts 776 MBPP-257 assert lines and found
22 HumanEval hits in 12,967 rows, matching each function body of the reference output with and
without its docstring; its positive controls flag 257 / 257 MBPP and 164 / 164 HumanEval items,
and whole-output matching would flag 0 / 164 (`opencode_pool.json::decontamination`).

### A4 — Next phase: 6 GPUs train, 2 evaluate (not started)

| setting | §4 as written | amended |
|---|---|---|
| GPUs | 8 training ranks | **6 train + 2 evaluate** |
| world × accum | 8 × 4 = 32 states/step | 6 × 5 = 30 or 6 × 6 = 36 — **TBD, chosen in the training phase** |
| checkpoints | `--save-every 100`, keep all | every 100 steps, **each backed up to the public HF repo** |
| evaluation | `--eval-every 100`, §4.2 | every 100 steps, per §4.2 |
| tracking | not specified | W&B |
| wall-clock | §4.3: ≈ 16.1 h on 8 GPUs | §4.3's figure assumes 8 training GPUs |

Order: the round-3 preprocessing gate (Stage 2 per pool, `code/draftalign/build_r3_index.sh`)
comes first; training starts after it.

Where the rest lives: commands and live status — `WORKTREE.md` §2.7 and §3; the `/venv/r3`
environment — `WORKTREE.md`, *Known environment hazard*; the HF backup — `HUGGINGFACE.md` §9;
footguns — `AGENTS.md` §9 (l)–(s).

---

*The 2026-09-09 recipe follows, unchanged except for the resolution note on §6's last item.*

The recipe for the next judge-adapter run. It supersedes the mix that produced
checkpoint `t2b_union_flag/step003057`, which every number in `RESULTS.md` and the
deck currently comes from.

Three things change, and nothing else:

1. the **data mix** — code goes from 0.25% of blocks to 35%,
2. the **training length** — a small set for ~3 epochs instead of a large set for ~1,
3. the **starting point** — fresh from the stock base, not from a checkpoint.

The expert pool stays at `--pool-Ms 128`, the LoRA configuration stays byte-identical,
and the decoder used to generate data stays the training-free blind rule. One
experiment at a time.

---

## 1. Why: the evidence that forced each change

### 1.1 Training past ~step 2200 bought no compute; its accuracy effect is unresolved

Nine checkpoints from the two completed runs were each evaluated on the same 100
`calibvalid` prompts at the same operating point (`dv-s8_full_c90_o99`). Scored from
the records under `logs/draftalign/da_decode/gsm8k_B32_t95_dv-s8_full_c90_o99_ad*`:

| checkpoint | accuracy | Δ vs stock | fwd/blk | compute vs stock |
|---|---:|---:|---:|---:|
| stock | 95.0 | — | 10.47 | — |
| `t2a` step 200 | 85.0 | −10.0 | 13.50 | 0.775× |
| `t2a` step 400 | 93.0 | −2.0 | 9.87 | 1.061× |
| `t2a` step 600 | 93.0 | −2.0 | 9.40 | 1.114× |
| `t2a` step 800 | **94.0** | **−1.0** | 9.07 | 1.154× |
| `t2a` step 1057 | **94.0** | **−1.0** | 9.09 | 1.151× |
| `t2b` step 2200 | 93.0 | −2.0 | **8.27** | **1.266×** |
| `t2b` step 2400 | 93.0 | −2.0 | 8.46 | 1.238× |
| `t2b` step 2600 | 91.0 | −4.0 | 8.40 | 1.246× |
| `t2b` step 3057 | 89.0 | −6.0 | 8.43 | 1.242× |

**The speed gain saturates by step 2200 while accuracy keeps falling** — at this
operating point. Step 3057 is 1.242× at −6.0; step 2200 is 1.266× at −2.0.

**But this does not reproduce at the headline operating point, and there the two
checkpoints cannot be told apart.** Steps 2200 and 3057 were re-run at `c95`
(`--judge full --tau-commit 0.95`, the deck's settings 3/4/5) on the same 100 prompts and
paired item-by-item:

| arm | acc 3057 | acc 2200 | Δ [95% CI] | discordant (late/early) | McNemar exact p |
|---|---:|---:|---|---:|---:|
| judge alone | 94.0 | 93.0 | +1.0 [−5.0, +7.0] | 5 / 4 | 1.000 |
| judge + DES 32 | 88.0 | 88.0 | +0.0 [−8.0, +8.0] | 8 / 8 | 1.000 |
| judge + DES 48 | 94.0 | 90.0 | +4.0 [+0.0, +9.0] | **5 / 1** | 0.219 |

The DES-48 row is the only one that moves, and it rests on **six discordant items** — the
bootstrap interval's lower bound sits on zero with 6.2% of resamples at or below it.
Flipping two items erases it. That is a direction, not a demonstrated effect.

So the resolved position is narrower than either "3057 is over-trained" or "3057 is
better":

* **What is solid: the extra 857 steps bought essentially no compute gain** — 787.2 →
  780.8 exp/tok at `c95`, 8.27 → 8.43 fwd/blk at `c90_o99`. Whatever those steps did, it
  was not to make decoding cheaper.
* **What is unresolved: their effect on accuracy.** At `c90_o99` a monotone decline across
  nine checkpoints; at `c95`, n=100 cannot distinguish the two. One plausible reading is a
  calibration shift — longer training makes the judge more decisive, which is harmless at
  a strict 0.95 commit threshold and harmful at the looser `c90`/override-0.99 rule whose
  operating point was tuned on an earlier checkpoint. **That reading is consistent with
  the data but not established by it.**

Caveats stated plainly: n=100, and `calibvalid` is the set these thresholds were tuned on.
Both checkpoints get the same home-field advantage so the *comparison* is fair, but the
absolute numbers are optimistic — step 3057's judge+DES-48 scores 94.0 here and 87.8 on
held-out GSM8K-500. **No `t2a` checkpoint has been evaluated at `c95` at all**, so the
early-vs-late question at the headline operating point is open (§6).

This unresolved question is precisely why §4.2 checkpoints every 100 steps and probes at
the headline operating point: round 3 should settle by construction what round 2 left
ambiguous.

### 1.2 Small set × more epochs beat large set × one pass

```
t2a: 1057 steps x (8 ranks x 4 accum = 32 states) = 33,824 draws / 16,334 blocks = 2.07 epochs
t2b: 2000 steps x (7 ranks x 5 accum = 35 states) = 70,000 draws / 64,428 blocks = 1.09 epochs
```

`t2a` — the smaller set at 2.07 epochs — is the best checkpoint the project has
produced **at the `c90_o99` operating point** (94.0, tied with its own step 800). `t2b`
at 1.09 epochs over 3.9× the data is the worst there (89.0). **This comparison has not
been made at `c95`**, where `t2b` step 3057 scores 94.0 — so treat "small set × more
epochs wins" as a hypothesis this round is designed to test, not as settled.

**Confound to record:** `t2b` changed six things at once (index, `world_size` 8→7,
`grad_accum` 4→5, `seed` 0→1, `warmup` 100→50, `holdout_items` 100→150), so "more
data, fewer epochs" is not cleanly separated from the schedule change. Round 3 changes
the mix only and restores `t2a`'s 8 × 4 = 32 batch.

### 1.3 The damage is code-shaped, not training-share-shaped

| eval set | share of `t2b` training | Δ accuracy (judge) |
|---|---:|---:|
| GSM8K-500 | 62.7% | −1.4 |
| MATH-500 | 37.0% | −4.4 |
| MBPP | 0.2% | −14.0 |
| HumanEval | **never seen** | −13.4 |
| GPQA-Diamond | **never seen** | −4.5 |
| AIME | **never seen** | −1.1 |

GPQA and AIME are also 0% of training and lose far less than HumanEval. Training share
does not predict the damage; **whether the output is program source text does.** Both
code sets lose 13–14 points; nothing else loses more than 4.5.

A second observation cuts against the intuitive "the drafter is bad at code" story:
drafter/final-token agreement is *higher* on code — HumanEval **0.846**, MBPP 0.626 —
than on GSM8K (0.541). The drafter is more agreed-with on code and more damaging there.
The working hypothesis is brittleness: a wrong accepted token in program source usually
fails the test, whereas in natural-language reasoning it is often recoverable.

### 1.4 What the mix has to supply

The adapter is 26,740,736 parameters (0.089% of the model) on attention `q/k/v/o` across
48 layers plus a 2,048-dim draft-flag vector. Experts, router, embeddings and head are
frozen. It is not learning Python — the frozen 30B already knows Python. It is learning
**a decision rule: when to trust the drafter.**

A decision rule needs both classes:

* **negatives — "do not defer":** code, where deferring is catastrophic. `t2b` had 161
  blocks of it (0.246%).
* **positives — "deferring is safe":** maths and general English. `t2b` had 99.75%.

Training on code + maths alone risks the mirror failure — the judge flips to "never
defer", accuracy returns and the speedup disappears. Instruction-following/chat data is
cheap, is safe to defer on, and is what holds the prior in place while the code data
teaches the exception.

---

## 2. The mix

Target **32,000 usable blocks**, ~3 epochs.

| domain | share | blocks | source | new decode? |
|---|---:|---:|---|---|
| code | 35% | 11,200 | `nvidia/OpenCodeInstruct` | yes |
| maths | 35% | 11,200 | MATH + GSM8K, already decoded on disk | **no — free** |
| IF / general | 30% | 9,600 | `Nemotron-IF-Domain-Cascade` (from the processed Nemotron mixture) | yes |
| science | 0% | — | deferred | — |

For comparison, `t2b`'s mix by block was GSM8K 62.7% · MATH 37.0% · MBPP 0.246%.

### Why `nvidia/OpenCodeInstruct` for code

**It is the only candidate that carries executable unit tests.** `--only-correct` is
what keeps wrong programs out of training, and for code "correct" can only be
established by running asserts.

| candidate | verdict |
|---|---|
| **`nvidia/OpenCodeInstruct`** | ✅ `unit_tests`, `tests_execution_status`, `average_test_score`; **700/700 sampled rows carry tests, median 10 asserts** |
| the processed Nemotron mixture | ❌ processing kept 5 columns and dropped the tests |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` | ❌ `messages, used_in, uuid, license` — no tests |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` `SFT/code` | ❌ no tests; outputs carry long `<think>` traces |

Measured on the downloaded snapshot (50 parquet shards, full scan):

```
total rows                : 5,000,000
average_test_score == 1.0 : 1,641,305  (32.83%)
domain                    : generic 3,614,351 / algorithmic 1,385,649
generation_algorithm      : self-instruct 2,653,523 / evol-instruct 2,346,477
```

Length fit (b32 tokenizer, 700 rows sampled across 7 offsets): prompt median **184**
tokens, reference answer median **187**, combined median 373, **99.0% fit
`--max-seq 1200`**. This is why the code slice is *not* drawn from the processed
Nemotron mixture, whose `Long-SFT-3K` pool is ≥3,000 tokens by construction — every row
there would be discarded by our own filter after costing 9–14× more to decode.

Licence cc-by-4.0, ungated.

### Why the IF slice comes from the processed mixture

Its `Nemotron-IF-Domain-Cascade` pool is a separate directory (3.73 GB of a 271.73 GB
repo), so it is not subject to the `Long-SFT-3K` length floor. Science is **not** taken
from that mixture: science rows exist only inside `Long-SFT-3K`, so they fail the same
length test that rules out its code.

---

## 3. Data preprocessing

### 3.1 The pipeline (unchanged)

Stage 1 — decode each prompt once with the **training-free blind rule**, stock 30B +
1.7B drafter, no trained judge anywhere in the loop:

```
run_da_decode.py --model b32 --drafter d32 --items <split> \
                 --draft s8 --judge blind --tau-acc-q .50 --score --out <dir>
```

Stage 2 — chop traces into 32-token blocks:

```
build_train_set.py --records <dir> --traces <dir>/traces --layout da \
                   --split-name <split> --out <out> \
                   --only-correct --max-seq 1200 --exclude-frozen
```

One prompt → one item → `n_blocks` training examples. Measured yields: MBPP **2.37**
blocks/row, GSM8K **5.95**, MATH **14.41**. A block is re-randomised every time it is
drawn (rewind to a random pass, refill open slots with draft or `[MASK]`), and the seed
includes the epoch — so repeated epochs are not repeated examples. That multiplies the
blocks you have; it cannot manufacture new ones.

### 3.2 Four fixes required to add a code pool

A literal copy of the MBPP path would fail. All four edits are additive.

1. **Scorer dispatch fall-through.** `run_da_decode.py` resolves the scorer by split
   prefix and **anything unrecognised falls through to the GSM8K numeric-answer
   extractor**. A code split scored that way returns wrong/null `correct` for every
   record and `--only-correct` discards ~100% of the decode. `opencode` must be routed
   to the `mbpp` branch. The same dispatch exists in `run_research_baselines.py`.
2. **Reuse the MBPP sandbox, write no new correctness code.**
   `analyze_s1b.per_item_correct(ds="mbpp", rows)` calls
   `MBPPEvaluator().score(preds, golds)` taking the tests **from the row itself**. So
   setting `gold = "\n".join(unit_tests)` scores OpenCodeInstruct through the exact,
   already-validated execution path that produced the existing MBPP blocks.
3. **Frozen-index guard.** `build_train_set.frozen_train_idxs()` returns the 100 frozen
   *GSM8K* calib/valid indices and is applied to any split named `train`/`train_pool` —
   which silently dropped 2 MBPP items. The new pool's per-item split is
   `opencode_pool`, so the guard cannot fire.
4. **`GEN_LEN` registration** (`opencode_pool: 512`, matching MBPP). A missing key
   raises at startup, which is the one place adding a dataset fails loudly.

### 3.3 Row filters

Applied in order: `average_test_score == 1.0` → contamination drop → length fit →
dedup on normalised `input`.

### 3.4 Contamination audit

Two of the six eval sets are code, and OpenCodeInstruct is self-instruct/evol-instruct
generated, so this is a real risk. Normalised exact match (`\s+` stripped, lowercased)
of 738 MBPP-257 assert lines and 164 HumanEval-164 function bodies against 1,200,000
rows (12 of 50 shards), restricted to the 488,870 candidates at score 1.0:

| check | collisions | rate |
|---|---:|---:|
| MBPP-257 assert lines | 107 | 0.0219% |
| HumanEval-164 function bodies | **0** | 0.0000% |

The MBPP hits are short generic asserts (`assert count_set_bits(2) == 1`) that look
coincidental rather than copied. They are dropped regardless.

### 3.5 Prompt shape — 60 / 40

MBPP and HumanEval use different prompt shapes and we evaluate on both. Generating code
blocks in one shape only risks a format-specific fix that does not transfer.

* **60%** `MBPP_TMPL` — *"You are an expert Python programmer, and here is your task:
  …Your code should pass these tests: …"* (matches MBPP-257)
* **40%** `HUMANEVAL_TMPL` — signature + docstring completion (matches HumanEval-164)

Assignment is deterministic by hash of the row id. Templates are imported from
`run_desvote`, not retyped.

---

## 4. Training configuration

Fresh from the stock base — `JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc37d240aa6871f167e8f92d694281b0c22`.

| setting | value | why |
|---|---|---|
| **init** | **stock base, not a checkpoint** | see §4.1 |
| `--pool-Ms` | **128** | keeps the run comparable to every existing number; training under a pruned pool is a separate experiment |
| `--lora-r` / `alpha` / dropout | 32 / 64 / 0 | byte-identical to `t2a`/`t2b` |
| targets | `q_proj,k_proj,v_proj,o_proj` × 48 layers = 192 modules, + `da_draft_embed` [2048] | 26,740,736 params = 0.089% |
| `--train-gates` | false | router stays frozen |
| `--draft-flag` | true | |
| world × accum | **8 × 4 = 32 states/step** | matches `t2a`'s peak config, not `t2b`'s 7 × 5 |
| epochs | ~3 over 32,000 blocks ≈ 3,000 steps | §1.2 |
| `--save-every` | **100** | ~30 checkpoints; an adapter checkpoint is ~107 MB |
| `--keep-checkpoints` | **0** (keep all) | the dose-response curve must exist by construction |
| `--eval-every` | 100, **holdout stratified by domain** | see §4.2 |
| `--max-seq` | 1200 | unchanged |
| `p_pass1` / `p_draftfull` / `p_allmask` | 0.4 / 0.3 / 0.0 | unchanged from `t2a`/`t2b` |

### 4.1 Why fresh from stock, not from checkpoint 1057 or 3057

The defect is a **learned prior, not missing knowledge**. Checkpoint 1057 is
`t2a_gsm8k_flag` trained on 100% GSM8K — the purest form of the "defer to the drafter"
prior we believe causes the code collapse. Starting the code experiment from there means
arguing that bias back down rather than not installing it.

Further: 3057 is a *two-stage curriculum* (`t2a` GSM8K-only to 1057, then
`[resume] from step 1057` into the union), so any arm branching off it reports a
curriculum result, not a mix result; and the ablation arms need a common initialisation
or they are not ablations.

Warm-up cost is negligible — the curve in §1.1 shows step 200 at −10.0 and step 400
already at −2.0, i.e. roughly 400 steps to become useful, which is **0.4 epochs** on a
32,000-block set.

### 4.2 Detecting catastrophic forgetting

The `t2b` holdout was a random 150 items, which gave it **2 MBPP blocks** —
structurally incapable of detecting code forgetting. Round 3 stratifies the holdout by
domain (~50 items each) so the `eval_every 100` curve reports per-domain token accuracy.
This costs no extra GPU time and is the early-warning signal.

An end-task probe every 400 steps on 100 GSM8K-calibvalid + 100 MBPP + 100 HumanEval
gives the accuracy curve alongside it.

### 4.3 Compute

`4.841 GPU-s per training state` (measured). 32,000 blocks × 3 epochs = 96,000 states
= **129.1 GPU-h ≈ 16.1 h wall on 8 GPUs**. Data generation and probes bring the round to
roughly 229 GPU-h.

---

## 5. Planned arms

All three from the same fresh stock init, same schedule, `pool_Ms 128`:

| arm | change | question |
|---|---|---|
| **A** | the mix in §2 | does code data fix code? |
| **C** | matched mass: the 11,200 code blocks swapped for 11,200 more GSM8K blocks | is any gain in A the *code*, or just more data? |
| **B** (later) | arm A's data, `--pool-Ms 48` | does training under the pruned pool resolve the judge/DES conflict? |

Arm B is deliberately deferred: changing the mix and the pool together would make an
MBPP improvement unattributable.

---

## 6. Open items

* **Blocks-per-prompt and survival for the new pools are not yet measured.** The MBPP
  2.37 / GSM8K 5.95 / MATH 14.41 spread is 6×, so the code yield is being measured by a
  300-prompt calibration decode before the full draw is sized. The ~6.4 blocks/prompt
  figure used for planning is a proxy from reference-answer length, not a measurement.
* **IF pool row lengths are unmeasured** — the dataset viewer is disabled on that repo,
  so the length distribution must be measured locally before the IF slice is sized.
* **Checkpoints 1057 and 2200 have never been evaluated on MBPP or HumanEval.** Every
  code number we hold is from step 3057 alone. If 1057 is already ≈ −14 on MBPP the
  deferral prior is confirmed as the cause; if it is ≈ −5, the code collapse is largely
  a training-length artifact and §1.3 needs rewriting. (Step 2200 has now been measured
  on `calibvalid` at `c95` — see §1.1 — but not on either code set.)
* **No `t2a` checkpoint has been evaluated at `c95`.** Until that exists, the
  early-vs-late and small-set-vs-large-set comparisons rest on the `c90_o99` cells,
  where the conclusion is the opposite of `c95`'s. Round 3's per-checkpoint probes must
  run at the headline operating point.
* **The two GPQA `stock + DES` cells are partial** (65 and 80 of 198), so the question of
  whether GPQA's double-digit loss under `judge + DES` comes from the pruning rather than
  the judge is not yet settled on paired items.
* **IF has no automatic verifier**, so `--only-correct` cannot gate it. Accepting it
  unfiltered, or gating it with a format/length heuristic, is a recipe change that should
  be pre-registered rather than defaulted.
  **Resolved 2026-09-14 by Amendment A2 (top of this file):** round 3 takes
  program-verifiable IF prompts and keeps only the decodes that pass every constraint under a
  strict IFEval-style checker (`code/draftalign/if_scorer.py`), approved by the user on 2026-09-14.
