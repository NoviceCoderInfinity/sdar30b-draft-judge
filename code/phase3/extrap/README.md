# Phase 3: extrapolation to the Phase 1 box, progress, safety snapshots

Working copies live in `logs/phase3/extrap/` on the run box (the scripts read and write there); this folder is the
same code, kept in git. Run from the repository root with `/venv/p2/bin/python -I`.

- `extrap.py ROOT p2_metrics.json OUT`: per run, accuracy, passes/block, expert loads/token, and tokens/s on the
  Phase 1 box (8x RTX PRO 6000 Server). Headline = same-box measurement:
  - HumanEval, MBPP: today's judge (run_p3.sh `judge` arm = rd flags minus `--full-redraft-every`) decoded on every
    item on this box under the same 4-GPU load; old-box tok/s = today's judge measured there x rd's same-box speedup.
  - Other five: today's judge on the first 12 items here (`judge_sample`); old-box tok/s = rd tok/s here (all items) x
    this box's slowdown on those 12 items. Checked against the full baselines: -2.4% / +0.4%; bootstrap 90% range
    with 12 items about +-4-8%, unbiased.
  - Reference columns: the Phase 2 page's cost model, and a single 16-item calibration ratio.
- `calib.py`: the 16-item setup-check calibration (same items, token-identical outputs on both boxes).
- `status.py`: progress table (setup | benchmark | done/total | ETA IST | old-box tok/s).
- `build_artifact.py` + `page_template.html`: the results page (Table 1 in the Phase 2 page's layout).
- `snapshot.py`, `snapshot_loop.sh`: partial analysis + RESUME.md + HF/GitHub push every 15 min (5 min near the
  expected end of credit); same secret scan as `backup_p3.py`.
- `baseline_then_switch.sh`, `samples_then_rd.sh`, `switch_order.sh`, `order.txt`: the baseline runs and the
  short-first benchmark order (humaneval mbpp aime gsm8k500 gpqa math500 ifeval541), applied between benchmarks.
- `p2_metrics.json`: the Phase 2 page's per-benchmark metrics (stock, today's judge, CoDA, Qwen).
