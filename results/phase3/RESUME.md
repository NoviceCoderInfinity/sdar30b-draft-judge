# Phase 3 run state: FINISHED (stopped by the project lead after GSM8K-500)

Final as of 06 Oct 2026 ~23:40 IST. Full results, formulas, drift analysis and how to reproduce: `results/phase3/REPORT.md`
(numbers in `results/phase3/final_numbers.json`); results page: https://claude.ai/artifact/MDo497vHUmVX3zCDAbV9ny

## Done (every item, on a shared 4x RTX PRO 6000 Server vast.ai box, 4 shards)

| Benchmark | base (SDAR-1.7B + trained 128M judge) | rd1 | rd2 |
|---|---|---|---|
| HumanEval (164) | done, 164/164 token-identical to Phase 1 | done | done |
| MBPP (257) | done, 257/257 token-identical to Phase 1 | done | done |
| GSM8K-500 (500) | done, 500/500 token-identical to Phase 1 | done | done |

Base pipeline also ran on the first 12 items of MATH-500, AIME, GPQA-Diamond, IFEval-541 (12/12 token-identical to Phase 1 each).

## Not run (lead's decision, 06 Oct ~23:17 IST)

rd1 / rd2 on MATH-500, AIME, GPQA-Diamond, IFEval-541. Reason: the box was shared (gpu_frac 0.5) and host speed drifted
between runs by up to ~11%, as large as the speedups being measured; see REPORT.md section 3. Timing runs should be redone on
a whole machine with base and rd decoded at the same time (see GENERAL_RULES.md on main).

## How to continue on any box

1. Restore `logs/phase3/` from the Hugging Face dataset `Arushhh/phase3-redraft` into a checkout of branch `phase3-redraft`.
2. Remove `logs/phase3/STOP` (it was set to stop the run).
3. `tmux new -d -s p3 'NSH=4 P3_SETS="$(cat logs/phase3/extrap/order.txt)" bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log'`
   (finished benchmarks are skipped via `.complete` markers; keep NSH=4 to reuse partial shards).
