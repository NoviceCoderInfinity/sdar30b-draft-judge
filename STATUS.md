# Phase 4 status (drafter LoRA) — 8x RTX PRO 6000 Blackwell box

Times are IST (UTC+5:30). Updated: 2026-10-13 10:00 IST. Branch `phase4-drafter-lora`, started from commit `a6d7e27`.

| | |
|---|---|
| Box up since | 2026-10-12 20:45:41 IST (container start) |
| Box hours used | 13.2 of 25 |
| Running | nothing: all 8 GPUs idle since 09:56 IST (vanilla 30B same-box decode finished: 2,250/2,250, token-identical to the vanilla records) |
| Last gate | Gate 4 done (all 2,250 items, rd1 + rd2): pooled speed vs vanilla 1.33x / 1.34x (chained); accuracy vs Phase 3 +0.1 / -0.7 pts; drafter forwards -7.6% / -5.3%; judge passes +0.6% / -2.1%; IFEval loops 104 / 107 (target <= 60, not met) |
| Result | measured on this box, Phase 4 vs vanilla per answer: rd1 1.25x [1.23, 1.27], rd2 1.26x [1.24, 1.28] (GPU time 1.22x / 1.24x); the chained estimate 1.33x / 1.34x was 6.5% / 6.2% high; REPORT.md section 9c, vanilla_same_box.md |
| Next step | operator: stop the box (GPUs idle)? restart the HF uploader with the live-W&B fix (e885d03)? save ANALYSIS.md? |
| Dashboard | https://claude.ai/artifact/863c7J9dPKnaVRuJpVs9VP (private; refreshed at every checkpoint and result) |
| HF dataset | https://huggingface.co/datasets/Arushhh/phase4-drafter-lora (PUBLIC; everything under logs/phase4 incl. checkpoints + merged; uploader restarted by the operator 00:33 IST, every 20 min, PID 27305; first pass 1,388 files / 8.00 GiB, scan clean) |
| GitHub | `phase4-drafter-lora`: code + STATUS.md + results every 20 min (no logs). Logs: branch `phase4-logs` (orphan, root `logs/` only, like `phase2-logs`; no `ref/`), separate clone `/workspace/diffusion-moe-general-expert-logs`, pushed 23:17 IST (`00b5865`), synced every 20 min (stop: `touch logs/phase4/LOGS_LOOP_STOP`) |

## Operator decisions (2026-10-12)

1. **No smoke test** (21:28 IST). `p4.sh smoke` is skipped; `p4.sh train` started right after setup. The plan artifact
   and the box prompt both list the smoke test; the operator overrode it. Still in force: the trainer asserts 34,865,152
   trainable parameters, logs the KL at step 1, and stops itself if its step-20 speed projection is over 6 h.
2. **Setup checks: only if they take under 5 minutes**, else skip (21:28 IST). The two decode checks started 21:30:00
   IST and were killed at the 5-minute mark (21:35:01 IST). By then both had decoded all 16 items: stock 16/16 and
   judge 16/16 token-identical to Phase 1 (`logs/phase4/setup_checks_from_records.json`). The kill happened in process
   teardown, so `setup_p3_8gpu.sh` logged "stock check crashed" and `p4.sh setup` exited 1. The rest of `p4.sh setup`
   (fetch, tests, selfcheck, sm_120) was run by hand with the same commands, selfcheck capped at 5 minutes.
3. **OOM in training: stop and ask** (no `--micro-tokens` change without the operator).
4. **Any discrepancy: tell the operator first and proceed only once confirmed.**
5. **Selfcheck 3/6 FAIL: proceed to training** (21:50 IST), recorded as known deviations (below).
6. **GPU time and CPU time must be measured** (22:00 IST). `nvidia-ml-py==13.615.71` (the Phase 3 box's version) was
   missing from `/venv/p2`, so `gpu_meter.py` would have written `gpm: unavailable` and no GPU/CPU time. Installed;
   the GPM counters work in this container (GPUs 4-5: GPM 64.7% vs nvidia-smi 65-67%). No code changed.
7. **No same-box baselines** (22:03 IST). Phase 3 records come from a box 1.78x slower than this one on identical
   work (judge check: 233.9 s vs 131.7 s), and the vanilla records mostly from a Workstation-Edition box 1.86x
   slower (16 stock GSM8K items). The operator chose not to re-decode the baselines here. So time-based comparisons
   (time per answer, tok/s, GPU/CPU/MoE/drafter time) against Phase 3 and vanilla are **cross-machine raw ratios**
   and are labeled that way. Counts (accuracy, judge passes per block, experts per token) do not depend on the
   machine and are compared directly.
8. **W&B stays on for the decodes** (22:08 IST), unlike Phase 3's decodes (ENABLE_WANDB=0): one W&B run per GPU
   shard. Per-item logging happens after the item's timer stops.

9. **HF: public, everything, every 20 min** (22:55 IST). `backup_p4.py` allowlist extended to all of
   `logs/phase4/**` (incl. optimizer states); dataset switched to public. Restarting the uploader was blocked by Claude
   Code's auto-mode classifier (data exfiltration to a public repo); the operator decides (run it with `!`, or allow).
10. **Logs only on a separate GitHub branch** (22:55 IST): `logs-phase4` in a second clone; `phase4-drafter-lora` no
    longer gets logs, and the 6 log files committed there earlier were untracked (commit `c61c914`; files kept on disk).
11. **Artifact + two sub-agents** (22:50 IST): one builds/refreshes the dashboard, one owns the logs branch.
12. **Logs branch** (23:10 IST): named `phase4-logs` (matches `phase2-logs`), downloaded reference records (`ref/`) left out.
13. **Same-box baselines deferred** (23:20 IST): no re-runs for now; continue the plan; the operator decides later.
14. **Stage chain** (23:50 IST): `scratchpad/p4_chain.sh` (PID 23482) starts gate1 -> gate2 -> gate3 -> gate4, each in its own
    tmux window, each only after the previous stage exited 0 (PASS / done); stops at the first non-zero exit or the STOP
    file. It also pauses before gate3 if Gate 1's merge check shows the merged drafter > 0.5 points below the adapter
    (Gates 3-4 decode with the merged drafter); that pause is a question for the operator, not a gate. Log: `logs/phase4/chain.log`.
15. **Gates are temporary checkpoints** (00:40 IST 13 Oct), operator quoted: "Ask me for the output, but if I don't
    respond within 5 minutes, I want you to proceed with the gate. Treat the gate as a temporary checkpoint. If I
    respond, follow my decision. If I don't respond, continue with the next step because we can't waste GPU hours
    waiting on a gate." Implemented in chain v2 (`scratchpad/p4_chain2.sh`, PID 29357, tested on 7 scenarios with a
    stub): a non-zero stage exit (3 = gate stop rule, else an error) or a bad merge fidelity opens a 300 s DECISION
    WINDOW; `touch logs/phase4/STOP` = stop, `touch logs/phase4/DECISION_CONTINUE` = continue now, no answer = continue.
    A stage whose input is missing is skipped (gate2 needs Gate 1's drafts; gates 3-4 need the merged drafter).

## Selfcheck (`logs/phase4/selfcheck.json`, 34 s, GPU 0) and the diagnostic (`logs/phase4/diag_selfcheck.json`)

| check | result | why / does it matter |
|---|---|---|
| count | PASS 34,865,152 | |
| kl0 | PASS KL sum 0, loss = CE 3.2613 | |
| step | PASS 3.2613 -> 2.8028 | |
| forward | FAIL argmax 0.631, max rel diff 0.348 (bar 0.99 / 3e-2, bf16) | In fp32 (no autocast), two-stream == the model's own forward at every perturbation (argmax 1.0000, max rel diff <= 7e-5): the training forward is exact. In bf16, the stock drafter itself reaches only 0.977 (no perturbation), so the bf16 bar is unreachable; selfcheck's perturbation (0.05) makes the adapter 84% of the weights' norm. |
| merge | FAIL argmax 0.6875 over 96 positions (bar 0.97) | At a realistic adapter (dW/W = 2%) merged vs un-merged: 0.9688 (93/96). bf16 rounding of merged weights. Real measurement: Gate 1 merge-fidelity on 300 held-out items. |
| parity | FAIL 0.9106 of 2,596 slots, 8 items (bar 0.95) | HF replay vs stored SGLang drafts; kernels round differently and s8 commits by confidence, so an early flip changes the block. Replay agrees with the 30B on 0.5348 vs stored 0.5242. Affects inputs of first-draft states only (40%), never targets. The artifact's 95% bar is a smoke-test stop condition. |

## Measured on this box

- Downloads: 30B 61.08 GB in 10m41s (95.2 MB/s), 1.7B 4.08 GB in 0m42s; corpus 2.31 GB (516 files) in 4m12s
  (9.1 MB/s: many 4 MB files, per-file overhead).
- Setup decode checks (same 16 items as Phase 3's setup): judge 131.7 s here vs 233.9 s on the Phase 3 box
  (0.56x the time) vs 107.6 s on the Phase 1 box; stock 285.9 s for 16 GSM8K-500 items.
- Corpus: waveA 165,438 + waveB 166,380 index rows, 0 missing npz. Phase 3 rd1 / rd2 and vanilla records: all 2,250
  items of the 7 sets each.
- Tests: Phase 4 CPU tests 29 passed; decode-loop tests 60 passed (2 deselected, as in Phase 3).
- `/venv/p2`: Python 3.12.14, torch 2.11.0+cu128 (sm_120), transformers 4.54.1, wandb 0.29.0.
- Trainer plan: 331,818 items, 3,000 held out (chat 676, code 602, if 622, maths 610, science 490); one pass over
  328,818 items = 133.2M corpus tokens in 1,285 steps; warmup 26; saves at 129, 257, 386, 514, 643, 771, 900, 1028,
  1157, 1285; 25% check at step 322.

## Log (IST)

- 21:15. Keys in `/workspace/.env` (mode 600). W&B project `Conscious_Engines` (operator), entity
  `arushsharma536-spe-iit`.
- 21:15. Three stale markers, `logs/phase3_8gpu/setup/.done_{p2_setup,judge_refs,tests}`, are tracked in git
  (committed by the Phase 3 box's backup, `f32cf7d`). With them present, `setup_p3_8gpu.sh` would skip building
  `/venv/p2`, the model downloads and the judge adapter. I moved them, not deleted, to
  `logs/phase4/stale_phase3_box_markers/` and did not commit the move. No code changed.
- 21:15. `p4.sh setup` started.
- 21:30–21:35. Setup decode checks, 16/16 + 16/16, killed at the 5-minute cap (decision 2).
- 21:35–21:42. Rest of setup by hand: fetch, tests, selfcheck (3/6 FAIL).
- 21:45. Diagnostic of the selfcheck failures (read-only).
- 21:50. `mkdir logs/phase4/train`: `p4.sh train` tees to `logs/phase4/train/main.log` but no stage creates that
  directory, so the log would not be written. No code changed.
- 21:51. `p4.sh train` started. HF backup loop started (`logs/phase4/backup.log`).
- 22:42. 25% check PASS: +1.84 points (40.78% vs 38.94%), 521 s replay of 3,000 held-out items.
- 22:55. Operator: HF public + everything every 20 min; logs to a separate `logs-phase4` branch/clone; artifact and
  two sub-agents. GPU telemetry logging every 60 s (`logs/phase4/gpu_telemetry.csv`).
- 22:58. HF dataset made public; uploader restart blocked by the safety classifier (no HF upload running).
- 23:00. Dashboard published: https://claude.ai/artifact/863c7J9dPKnaVRuJpVs9VP
- 23:17. `phase4-logs` pushed (`380f369` root, `00b5865` tip; 53 files, 1.12 MiB); 20-min sync loop running.
- 23:20. Operator: baselines deferred; continue the current plan.
- 23:50. Stage chain started (waits for training to end, then gates in order).
- 00:29. Training done (1,285 steps, 9,137 s); merged check PASS; the chain started Gate 1 at 00:29:22.
- 00:33. Operator started the HF uploader (`backup_p4.py --loop 20`, public): first pass 8.00 GiB, scan clean.
- 00:37. Chain v2 replaced the chain (5-minute decision windows; no answer = continue); it waits for the running Gate 1.
- 00:54-00:57. Gate 1 FAIL (+2.45 pts, retention 89.1%); window 00:57:20-01:02:21, no answer: continued.
- 01:03. Gate 2 PASS (precision 99.21% -> 98.73%, -0.48 pts; commit share 30.74% -> 31.21%).
- 01:45. Gate 3 FAIL (0.33% fewer judge passes/block; accuracy and loops ok); window 01:45:15-01:50:15, no answer:
  Gate 4 started 01:50:15.
- 02:45-03:35. Operator asked why training did not cut judge passes: three read-only diagnosis agents (REPORT.md
  section 8b): decode = Phase 3 code + drafter + ban; no decode or loss bug; the judge's 0.95 rule caps the pass savings.
- 03:20. Fix 93fd175: vanilla comparisons read only each set's own cell (stray gsm8k...aime cell).
- 03:31. Gate 4 IFEval done in both modes (rd1 02:39, rd2 03:31).
- 06:41. Gate 4 decodes done (GSM8K rd2 last); the report step crashed on a p4.sh quoting bug (fixed 108c126) and was re-run at 06:42: gate4_rd1.json, gate4_rd2.json.
- 06:54. Operator: run vanilla 30B stock on this box for all 7 benchmarks (same-machine baseline, no extrapolation). Started 06:54:11 (events.tsv).
- 07:03. Measurement loop started (every 20 min). First 64 vanilla questions: tokens, passes per block and forward counts identical to the earlier vanilla records.
- 07:08. vanilla_same_box.py (d414c20, c802de5); pytest code/tests/test_drafter_lora.py 21 passed.
- 07:12. HF upload failed at 07:04 (3rd time in 19 passes): the Hub rejects the whole commit when a live W&B `.wandb` file
  (here the vanilla decode's) grows between hashing and upload, so the final Gate 4 files are not on HF yet (last good
  upload 06:37). Fix in backup_p4.py: skip `.wandb` files written in the last 15 min (uploaded after their run ends);
  pytest 21 passed. The running uploader (PID 27305, started by the operator) needs a restart to use it: asked the operator.
- 07:52-09:56. Vanilla here: IFEval 07:52, MATH-500 08:38, GPQA 09:10, AIME 09:26, HumanEval 09:30, MBPP 09:36, GSM8K 09:56;
  2,250/2,250 token-identical. HF uploads succeeded 07:56, 08:16, 08:36, 08:56, 09:16; failed 07:04, 07:31, 09:36 (live W&B
  file); the operator has not restarted the uploader (fix e885d03 waits for a restart).
- 09:57. Final: vanilla_same_box.py (exact decomposition added), judge_lora_cost.py, baseline_tables.py (measured column);
  REPORT.md section 9c; pytest 21 passed.
- 10:11. HF upload succeeded (4,480 files; REPORT.md and FINAL_TABLE.md on HF identical to local). Logs branch synced 09:59
  (caa8c41). Artifact v46: every Phase 4 speed vs vanilla is the measured one. Waiting on the operator: stop the box /
  re-decode Phase 3 here / save ANALYSIS.md.
- 10:39. Operator: every speed on this machine, vanilla = base, no chained figures. phase3_same_box.py (a81a04f): Phase 3's recorded
  work priced at this machine's measured per-forward costs (checks: P4 within 1.0%, vanilla -0.7..+2.5%): P3 vs vanilla
  1.20x / 1.26x, P4 vs P3 1.04x / 1.00x. baseline_tables, FINAL_TABLE, REPORT (9, 9c, new 9d) updated; artifact rebuilt.
