# Phase 3: re-draft between judge passes

**Question.** Today the slots the judge rejects are re-masked and the 30B fills them itself, a few per pass, often one
forced token per pass. Does handing those slots back to the drafter cut judge passes without losing accuracy?

## Method, per 32-token block

1. SDAR-1.7B drafts every slot (8 steps), exactly as today.
2. Judge pass (30B + the 128M LoRA): commit where p > 0.95 (C); keep the draft where the judge agrees but is unsure (t);
   re-mask where it disagrees (M). If nothing clears 0.95, the most confident slot is committed (forced), as today.
3. After every K judge passes, the drafter re-fills **only the M slots**. It sees only verified tokens: the prompt plus
   the committed slots. The t slots are hidden as [MASK] in its input and keep their token in the block.
   Steps = ceil(#M / 4), at most 8 (the first draft's rate). Logged as drafter kind `redraft`.
4. Repeat 2–3 until every slot is committed. Only the judge commits, so every token's last step is a judge pass.

## Setups (both on all 7 benchmarks)

| Setup | Rule | Flag |
|---|---|---|
| rd1 | re-draft after every judge pass | `--full-redraft-every 1` |
| rd2 | re-draft after every 2 judge passes | `--full-redraft-every 2` |

Benchmarks: HumanEval 164, MBPP 257, GSM8K-500, IFEval-541, MATH-500, GPQA-Diamond 198, AIME 90.

**Identical to every earlier run:** `run_da_decode.py` (PyTorch, no SGLang); JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc and
SDAR-1.7B-Chat-b32 @ 2c83c05; block 32; threshold 0.95; tau-commit 0.95; s8 first draft; `--judge full`; Phase 1's 128M
judge (Arushhh/c1b-judge-lora-sdar30b-b32, main_128m/tok_000128000000, weights hash 35805974); **no expert pool (no
DES)**; greedy; the same benchmark files and length limits (`research_protocol.GEN_LEN`); 8 shards, one per GPU;
`--score --no-trace`; the Phase 2 environment (`/venv/p2`: torch 2.11.0 cu128, transformers 4.54.1); 8× RTX PRO 6000.

## Run

```bash
bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log     # inside tmux; resumes if re-run
```

It runs setup (`setup_p3.sh`: Phase 2's environment and downloads, the 128M judge, Phase 1's reference records), the
checks, rd1 and rd2 on each benchmark (alternating which goes first), the analysis and a backup after every benchmark.

**Checks (kept minimal):** CPU tests; stock on 16 GSM8K-500 items must match the recorded outputs token for token
(16/16); SDAR-1.7B + judge on 16 HumanEval items vs Phase 1's records (at least 12/16 identical).

## Outputs

- `logs/phase3/decode/<rd1|rd2>/<benchmark>/`: per-item records; `jobs/`: shard logs; `setup/`: setup, checks
- `results/phase3/TABLE.md` and `metrics.json` (also committed to this branch)
- Hugging Face dataset `Arushhh/phase3-redraft` (public): logs, results and the code that ran, synced after every benchmark

## Reading the results

- **Compare with today's judge and stock** (Phase 1 / round-3 records): accuracy, passes per block, expert loads per
  token, drafter forwards per token. These don't depend on the machine.
- **Tokens per second** compares rd1 with rd2 only (same box). Earlier timings come from other machines.
- **Speed needs fewer judge passes per block than today** (5.9 on HumanEval up to 15.6 on IFEval). One drafter step
  costs about 9.5 ms and one judge pass about 250 ms (Phase 1 box), so each re-draft pays for itself if it saves about
  0.04 judge passes per drafter step.
- Diagnostics per setup: re-draft rounds and slots per block; how often a re-drafted token equals the judge's top
  choice from the pass before; how often it repeats the rejected draft; next-slot copy rate at draft and at mask slots.

## Time (US box, $11.75/hr)

Setup about 1 h; each setup about 2.8 h on the 7 benchmarks; about 6.5 h in all, up to 8.5 h if the box's CPU slows
the decode loop (about $80–100).

## Code changes (all off by default: every earlier setting runs exactly as before)

- `code/draftalign/da_decode.py`: `judge_full_block(redraft_every=..., diag=...)`, `copy_diag`, new record fields
  (written only when the options are on)
- `code/draftalign/drafter.py`: `draft_block(static_steps=..., kind=...)`
- `code/draftalign/run_da_decode.py`: `--full-redraft-every`, `--full-redraft-rate`, `--judge-diag`; cell suffix `_rd<K>`
- `code/tests/test_da_decode_redraft.py`
