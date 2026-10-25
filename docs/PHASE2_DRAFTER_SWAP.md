# Phase 2: drafter swap ablations (branch `phase2-drafter-swap`)

Plan: Notion "Phase 2: Drafter swap ablations (conversion corpus, size, compute)" under Conscious Engines.
The 30B MoE (`JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc`, block 32) stays frozen. Drafters are released checkpoints used
as downloaded. **The only thing ever trained is the LoRA judge adapter.**

Branched from `phase1-datacorpus` (7e4358e) to reuse Phase 1's box setup (`/venv/train`, the pinned models, eval
sets). Phase 1 files are untouched except two default-off hooks in `code/draftalign/run_da_decode.py` and a no-op
shim call in `code/research_protocol.py::load_sdar`.

## This run: the MVP (decided 2026-10-05): F2 + S1, all seven benchmarks, zero-shot, fully logged

| | |
|---|---|
| Family (corpus) | **F2** CoDA 1.7B: same Qwen3-1.7B base, size and tokenizer as S0; code-heavy corpus. **F1** Fast-dLLM v2 1.5B is the fallback: it runs in F2's place only if F2's re-draft is stopped by the fit rule |
| Size | **S1** Qwen3-0.6B bd3lm: the only drafter below 1.7B that covers all seven sets; best-case charged speedup 1.39x vs S0's 1.24x (HumanEval) |
| Not in this run | S6 (4B dense, over the 3B-active budget, best case ~1.01x), S2, and every unwired candidate |
| Benchmarks | HumanEval (164), MBPP (257), GSM8K-500, IFEval-541, MATH-500, GPQA-Diamond (198), AIME (90): nothing skipped |
| Steps | 2–5 (plug-in checks, agreement, probe, zero-shot decode). No training of any kind |
| Rules | `rules.json` v2: IFEval in-domain, AIME near-domain; no threshold changed |

Why these two: S0 → F2 changes only the corpus (and the generation style: CoDA is a full-sequence model, so read its
results as a pattern across domains relative to S0, not as absolute levels). S0 → S1 answers "is a smaller drafter
worth it": it saves ~89 expert-equivalents per token of drafter cost on HumanEval, so it wins only if its weaker drafts
raise the MoE's experts per token by less than ~13.6% with accuracy inside S0's interval. S1 changes size and corpus
together, so its result is "this 0.6B drafter", not size alone.

## One command for a rental

```bash
export P2_MODELS_FROM=/workspace/dmge/hf_cache/hub     # only on a box where Phase 1 ran; omit on a fresh box
bash code/phase2/run_all_p2.sh 2>&1 | tee -a logs/phase2/run_all.log
```

A fresh box (no Phase 1) also works. Setup builds `/venv/p2` from torch 2.11.0 cu128 and the team's round-3 freeze,
then downloads the models, drafters, references and benchmark datasets. Allow ~45–60 min more.

`run_all_p2.sh` does the following:
1. Runs the CPU-only setup: venv, models, the drafters F2, S1 and F1 (the fallback), round 3's references, the
   benchmark datasets, IFEval-541 (Phase 1's frozen file) and Phase 1's stock run on IFEval-541 (step 3c, checked:
   541/541 items, revision `c351bbc`, golds equal to the frozen file).
2. Starts the Hugging Face backup loop (every 30 min).
3. Waits until every GPU is free; it never touches other jobs.
4. Runs the go/no-go checks.
5. Runs the plan below, in order. Nothing is skipped.
6. Writes `STATUS.md` and runs a final backup.

`P2_DEADLINE_IST` is optional. If it is set, the launcher only warns when time looks short. At the deadline it
creates `logs/phase2/STOP`: running decodes stop after their current item and keep their records. Then a final
backup runs.

**The plan (8x RTX PRO 6000; hours, ±20%):**

| # | Step | Estimate |
|---|---|---:|
| 0 | Go/no-go checks (+ final gate and backup at the end) | 0.53 |
| 1 | **F2 re-draft first**, so its "does it fit" answer comes in ~20 min | 0.35 |
| 2 | **S0 on IFEval-541** (round 3 has no S0 judge run on IFEval) | 0.70 |
| 3 | **F2**: probe, 7 sets, report, gate (**F1** in full instead, ~3.43 h, if F2 does not fit) | 3.65 |
| 4 | **S1**: re-draft, probe, 7 sets, report, gate | 3.47 |
| | **Total** | **~8.7 h** (+ ~1 h setup on a fresh box) |

Per set (minutes; re-draft and probe once per drafter):

| Run | Re-draft | Probe | HumanEval | MBPP | GSM8K-500 | IFEval | MATH-500 | GPQA-D | AIME | Total (min) | Total (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F2 | 21.0 | 10.5 | 7.1 | 9.0 | 23.6 | 54.9 | 54.1 | 38.5 | 21.4 | 240.1 | 4.00 |
| S1 | 5.3 | 10.5 | 6.5 | 8.3 | 21.8 | 50.6 | 49.9 | 35.5 | 19.7 | 208.1 | 3.47 |
| S0 on IFEval | – | – | – | – | – | 42.0 | – | – | – | 42.0 | 0.70 |
| Total | 26.3 | 21.0 | 13.6 | 17.3 | 45.4 | 147.5 | 104.0 | 74.0 | 41.1 | 490.2 | 8.17 |

Per item: round-3 s/item × 1.30 for F2, × 1.20 for S1 (× 1.15 for F1). Per set: +3 min for loading. Per arm: +5% for
the full expert log. Within an arm the sets run in priority order: HumanEval, MBPP, GSM8K, IFEval, MATH, GPQA, AIME.

The prompt for Claude Code on the box is in `docs/PHASE2_INSTANCE_PROMPT.md`.

**What each arm is paired against:**
- **Six sets:** round 3's **S0 records** (r3_B step 710 + DES48, downloaded by setup). This is valid because the
  setup's stock check proves the new stack reproduces round 3 token for token.
- **IFEval:** the **local S0 run** from step 2. A local run always wins over round 3 when both exist; set
  `P2_S0_REF=S0` to force local.
- **Speedup reference:** stock records from round 3 (six sets) and Phase 1's stock run on IFEval-541
  (`Arushhh/c1b-judge-corpus-waveA`, `logs/c1b/eval/checkpointB/ifeval541_released`). No stock decode in this run.

Step 6 runs only with `P2_NO_RETRAIN=0`. Single stages:

```bash
bash code/phase2/setup_p2.sh                  # venv, models, drafters, references, 4 go/no-go checks
bash code/phase2/run_phase2.sh S1             # one arm (redraft -> probe -> decode -> gate)
bash code/phase2/run_phase2.sh S1 decode      # one stage
/venv/p2/bin/python code/phase2/backup_p2.py  # one backup sync
```

Everything resumes where it stopped (`logs/phase2/<ARM>/.done_<stage>`). `touch logs/phase2/STOP` stops every
Phase 2 process at its next check.

## Drafters (`code/phase2/drafters.py`)

| key | drafter | vocab / mask | attention, logit shift | charged per forward |
|---|---|---|---|---:|
| S0 | JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05 (baseline, `--drafter d32`) | Qwen3 / 151669 | block-causal cache, no shift | 364.64 |
| F1 | Efficient-Large-Model/Fast_dLLM_v2_1.5B @ 25093b6 | Qwen2.5 / 151665 (remapped) | its own block cache, shifted | 327.16 |
| F2 | Salesforce/CoDA-v0-Instruct @ f2d40ce | Qwen3 / 151669 | full bidirectional, recompute, shifted | 364.64 |
| S1 | dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 @ 60f77bb | Qwen3 / 151669 | block-causal cache, no shift | 126.32 |
| S2 | dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1 @ 4bcf3a0 (code sets only) | Qwen2.5 / 151665 (remapped) | block-causal cache, no shift | 104.70 |
| S6 | JetLM/SDAR-4B-Chat-b32 @ d4dbd34 (over the 3B limit: ceiling only) | Qwen3 / 151669 | SDAR SeqCache | 852.47 |

Charged cost = (body + lm_head parameters, no input embedding) / 4,718,592. F3, S3, S4, S5 and S7 are not wired;
`drafters.NOT_WIRED` says why (tokenizer bridge, block 4, or not wired yet).

The Qwen2.5 drafters lack Qwen3's `<tool_response>`, `</tool_response>`, `<think>` and `</think>` (ids 151665–151668).
On input these become `<|fim_pad|>`, counted in `n_unmapped`. On output the drafter can never propose them.

**transformers.** No single unpatched version loads both SDAR and these drafters, so Phase 2 runs in `/venv/p2`:
`/venv/train`'s freeze plus transformers 4.54.1 and einops. Two shims make that work:

- `phase2/tf_compat.py` restores `LossKwargs`, a TypedDict used only in an annotation in the SDAR code.
- `code/compat/dllm/` is an empty package that satisfies the dLLM checkpoints' import scan.

`setup_p2.sh` then checks on the real 30B model that `--stock` still reproduces the recorded stock cell token for
token. If it does not, stop and report: the fallback is to run the drafter in its own process.

## The pipeline per arm (`code/phase2/run_phase2.sh`)

| stage | what | output |
|---|---|---|
| redraft | the drafter re-drafts every usable block of the round-3 index (4,444 items, 23,558 blocks) against the teacher-forced prefix, s8, exactly as at inference. This is also step 3, drafter agreement: the new drafter and S0 are scored on the same blocks | `logs/phase2/redraft/<A>/{index.jsonl, summary.json, items/}` |
| probe | the S0 judge (r3_B step 710) on the new drafts, on round 3's 150 held-out items. Reports keep rate, fix rate, precision of commits above 0.95, ECE and draft accuracy, per domain | `logs/phase2/probe/<A>__r3_B/probe.json` |
| decode | zero-shot: the arm plus the S0 judge plus DES48 on HumanEval, MBPP, GSM8K-500, MATH-500 and GPQA-Diamond, 8 shards each | `logs/phase2/decode/<A>/<set>/` |
| gate | the verdict against S0 under `rules.json` pass_fail | `logs/phase2/gate/<A>_zeroshot/gate.{md,json}` |
| retrain | **only if** the gate says `retrain_needed`: trains the LoRA judge adapter from stock on the arm's re-drafted index with round 3's calibrated recipe for 2 epochs. The control **S0ctl** is the same trainer and recipe on S0's own drafts, run once | `logs/phase2/train/<A>/`, `logs/phase2/train/S0ctl/` |
| probe_rt, decode_rt, gate_rt | the same three stages with the retrained adapter, paired against S0ctl | `logs/phase2/gate/<A>_retrained/` |

The S0 arm reruns round 3's S0 cell on this box with the expert log on. `--p2-integrity` compares it with round 3's
records and warns if they differ. Every arm is paired against this rerun, not against the old records.

## Stop rules, checked at every step (`code/phase2/rules.json`, fixed 2026-09-29)

Each Phase 2 output records the file's sha256. When a rule fires it is logged to `<run>/stop_events.jsonl`. A `stop`
rule also writes `<run>/STOPPED.json` and exits with code 3; the orchestrator reads 3 as "stopped by rule", not as a
crash. In a decode, a stop in one shard also stops the other shards of that cell.

| stage | checked after | stop | warn |
|---|---|---|---|
| redraft | every item | NaN logits; agreement < 0.05 (a remap bug); raw [MASK] argmax on > 50% of open slots; agreement < 0.5 × S0's on the same blocks after 512 blocks (the drafter does not fit) | agreement < 0.85 × S0's |
| probe | once | fewer than 100 rows | precision of commits above 0.95 < 0.99 |
| decode | every item, across the whole cell (every shard's records) | a `--stock` run that no longer matches the stock cell; score errors > 20%; > 50% of items never stop (after 30); futility: the paired Δ vs S0 has its 99% upper bound below −10 points (after 60 pairs) | agreement < 0.5 × S0's; judge fallback on > 50% of blocks; 5× slower than S0; the S0 rerun differs from round 3 |
| train | every optimizer step and every probe | non-finite loss or gradient; gradient norm 50× its median; loss EMA 1.5× its best for 20 steps; held-out commit precision < 0.99 at two probes in a row (after 2M tokens); held-out CE rising at 3 probes in a row | throughput under 30% of median; GPU memory > 97% |

## What is logged (everything that cannot be recomputed later)

**Decode.** Records stay in the round-3 format and gain Phase 2 fields. None of the new fields enters the config
hash.

- **Per item, experts:** `exp_*`, the expert-activation summary.
  - Per forward kind (prefill, judge, denoise, kvflush):
    - forwards per item;
    - experts per layer **in one forward** (mean and max), and summed over the 48 layers;
    - the union per layer over the item;
    - load entropy and max-load share;
    - the router's top-8 probability mass.
  - Per block forward: experts per layer, and the 48-layer profile.
  - The 10 most-loaded (layer, expert) pairs.
- **Per item, judge probabilities:** `judge_*`, the judge's mean top-1 probability and the share of slots above 0.95,
  at pass 1, 2, 3, … of each block.
- **Per item, cost:** `drafter_key`, `drafter_eq_per_forward` and `charged_exp_per_token`.
- **Expert npz:** `experts/<set>/<idx>.npz` holds, for every forward and every layer:
  - a bitmask of the experts that ran (the SINK's dispatched set, so under DES this is the restricted set);
  - per-kind token counts per (layer, expert).

  With `--expert-log full` (**the default for this run**) it also keeps every row's top-8 expert ids and router
  probabilities: about 5–10 MB per item, ~35 GB for the whole run.
- **Judge npz:** `judge/<set>/<idx>.npz` holds every judge read:
  - the forward index and the pass within its block;
  - per slot: the top-1 probability, the argmax, the probability of the draft token, the draft token, and the open
    flag.
- **Traces:** `traces/<set>/<idx>.npz`, as in round 3, hold the drafter's confidence per slot and every stock pass's
  confidences.
- **Probe:** the probe also records which experts the judge routes to on the held-out states (`probe/…/experts.npz`,
  `experts/*` in `probe.json`).
- **Report:** `report/<ARM>/experts.{md,json}` (`expert_report.py`) collects, per set:
  - experts by forward kind;
  - the per-layer profile and the top 20 (layer, expert) pairs;
  - judge probabilities by pass;
  - the drafter's confidence against its actual correctness (10-bin reliability, ECE).
- **Running metrics:** `<shard>.running.jsonl` holds the running metrics over the whole cell, one row per item:
  - accuracy;
  - experts per token, drafter forwards per token and charged experts per token;
  - speedup vs stock;
  - passes per block;
  - agreement;
  - judge commits, tentatives and remasks per block;
  - fallback and truncation rates;
  - seconds per item;
  - the paired Δ vs S0 with 95% and 99% intervals;
  - experts per layer per forward and the union per item, for each forward kind;
  - the router's top-8 mass per kind;
  - the judge's mean top-1 probability at passes 1–3, and its pass-1 commit share.

  The same metrics also go to W&B under `run/*`, and the per-block traces are kept as in round 3.

**Train.** Rank-0 metrics are all-reduced across ranks.

- `steps.jsonl`, one row per optimizer step: loss, loss EMA, grad norm, lr, states, seconds, GPU memory, and whether
  the update was skipped.
- `train.rank0.jsonl` every 10 steps: per-domain loss, CE, certainty, calibration, accuracy and top-1.
- The expert window every 10 steps, in `experts/step_<n>.npz` and W&B `experts/*`: per layer, the experts that the
  prefix pass and the trained block pass routed to.
- `probe.jsonl` every 100 steps: the extended probe.
- `best/` is the checkpoint with the lowest held-out CE whose commit precision is ≥ 0.99. `step_<n>/` is saved every
  100 steps.

## Files (`code/phase2/`)

| file | what |
|---|---|
| `rules.json`, `stoprules.py` | the frozen rules; `StopMonitor`, `paired_delta` |
| `drafters.py` | registry, costs, `VocabMap`, the three wrappers (`CachedBlockCausal`, `FastDLLMv2Seq`, `FullRecomputeShifted`) and the SDAR path |
| `explog.py` | `ExpertLogMoEMeter` (decode) and `TrainExpertCapture` (training) |
| `decode_hooks.py` | the per-item hooks for `run_da_decode.py` (`--expert-log`, `--stop-rules`, `--p2-*`) |
| `redraft.py`, `p2probe.py`, `probe_p2.py`, `train_p2.py`, `gate_p2.py` | the stages |
| `expert_report.py` | per-arm report: experts by stage, router mass, judge probabilities by pass, drafter reliability |
| `backup_p2.py`, `run_all_p2.sh` | Hugging Face backup; the one-command plan for a rental |
| `p2_jobs.py`, `run_phase2.sh`, `setup_p2.sh` | orchestration (every GPU job in its own `/dev/shm/p2_sandbox/<job>` cwd, AGENTS.md 9(p)) |
| `tf_compat.py` | the `LossKwargs` shim |
| `tests/` | CPU tests (20 + 2 + 2): rules, the vocabulary map, cache and shift equivalence of each wrapper against a full forward, the expert meter against the SINK's own count, the decode path on a fake MoE with a Phase 2 drafter, cell-wide futility, the trainer end to end (per-step rows, the probe schedule, resume, a rule stopping the run) and the job runner |

## Open decisions (not settled here)

- **GPQA-Diamond** is the only truly out-of-domain set in the plan, but GPQA was removed from the paper on 21 Sep. It
  is kept in the default sets; `P2_SETS` drops it.
- **The block-size ablation** (16 / 8) from the original Phase 2 list is not in the plan.
- **CoDA (F2)** is a full-sequence model. It sees `[prefix + current block]` with no trailing masks
  (`P2_CODA_TRAILING=N` adds them). Its wall clock is inflated by recomputation; its charged cost is not.
- **Retrain recipe:** 1,400 steps × 32 states from stock, lr 2e-5 cosine to 0.5, round 3's calibrated loss. This
  matches r3_B's total (r3_A 700 + r3_B 710 steps) in one run, with the S0ctl control to isolate the drafter.
