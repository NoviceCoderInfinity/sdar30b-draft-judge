# Phase 5 — recorded judge states (CP5 recording runs)

> **24 Oct 05:55 IST — FINAL.** Server 57349095 being removed. Recording totals: **191 states.npz** (`record` 64 · `record_run1` 8 · `record2` 105 · `record3` 14); filtered training index **47,011 items**. The judge trained on these states FAILED (see below); the states are correct and kept for the record. Final report: `results/phase5/REPORT.md`.
>
> **24 Oct 04:50 IST — PHASE 5 = FAIL.** The judge retrained on these SGLang-recorded states degenerates in live HF decode (non-terminating repetition loops; ifeval541 acc 56%→~8%). Root cause = train/decode engine mismatch: states recorded in SGLang, training+eval in HF (SGLang↔HF judge-argmax agreement only 0.8978, CP3 vanilla token agreement 0.596). The recorded-state training construction itself is correct (target=vanilla T, loss rows=open non-committed, flag on draft-filled). Full analysis: **logs/phase5/FAILURE_ANALYSIS.md**. 128M (Phase 4) retained as the production judge. The recording details below are kept for the record.

**As of 23:41 IST, 23 Oct 2026. Recording in progress (run-2, run-3). This file is regenerated at each recording/training event.**

All runs: `judge_sglang.py --mode record`, rd1 + `--redraft-ban` + `--loop-guard 12`, 128M judge adapter
(`logs/c1b/train/main_128m/tok_000128000000`) + Phase 4 merged drafter, block 32, `--mem-fraction 0.70 --drafter-mem-fraction 0.10`.
Batched `--max-running 64`; a shard that takes the batched-scheduler fault is relaunched at `--max-running 1` (resumes via `done.txt`).
States are written at `logs/phase5/judge_sglang/<folder>/gpu<k>/states.NNNNN.npz` (per 256-item chunk).

## Recording runs

| folder | pid list (count) | how drawn | GPUs | start → end IST | items done | states npz / MB | status | in training index |
|---|---|---|---|---|---|---|---|---|
| `record_run1` | (from `record_pids.json` full / then `record_pids_sel.json`) → 2,048 | the first 256-item chunk each GPU completed during the initial full-index / selection runs (moved aside when the 16k run started) | 0–7 | 20:38 → 21:09 | 2,048 | 8 / 83.1 MB | done (superseded, kept) | yes (what it holds) |
| `record` | `record_pids_16k.json` (15,999) | fixed-seed (20261023) random, **stratified** by the selection's per-domain item share (maths 3987 / code 3614 / chat 4606 / IF 3792); exclusions already applied (drawn from the 101,880 post-exclusion selection) | 0–7 | 21:10 → 22:43 | 15,999 | 64 / 438.0 MB | done | yes |
| `record2` | `record_pids_2nd.json` (30,001) | fixed-seed (20261023002) random, stratified, **disjoint** from `record` | 0–4, 6, 7 | 22:34 (relaunched 22:52) → ~00:35 (proj) | 14,336 (growing) | 56 / 336.6 MB | **running** | yes (what finishes) |
| `record3` | `record_pids_3rd.json` (3,501) | fixed-seed (20261023003) random, stratified, disjoint from `record` ∪ `record2` | 5 | 22:50 → (stopped at run-2 end, keep finished) | 1,792 (growing, slow ~20/min) | 7 / 39.8 MB | **running**, will be stopped by PID when run-2 finishes | yes (what finishes) |

Supervisor: `code/judge_round/record_sup.sh` (runs 0–4,6,7) + `code/judge_round/record3_gpu5.sh` (GPU 5). Handover to training: `code/judge_round/auto_handover.sh` (starts training when run-2 finishes on all 7 GPUs, or 01:10 IST latest; stops run-3 by PID, keeps its finished items).

## Planned / dropped / unused pid lists (NOT recorded as separate runs)

| file | count | note |
|---|---|---|
| `record_pids.json` | 150,866 | full fresh index (every recorded pid of the index). Superseded by the subset plan (Plan A); not run as a whole. |
| `record_pids_sel.json` | 101,880 | the 44M-plan domain-weighted selection (post-exclusion). Superseded by the random-subset plan; not run as a whole. |
| `record_pids_4th.json` | 540 | 4th GPU-5 slice, fixed-seed (20261023004), disjoint. **DROPPED** (operator 23:30 IST: no 4th slice; start training when run-2 ends). |

## Items training will see (filtered index = recorded pids − 12,770 exclusions − no-op items)

As of 23:41 (grows until recording stops): **33,387 items**, 0 that hit the exclusion list, 0 no-op-excluded.
Per-domain: **chat 14,432 · code 10,327 · IF 4,641 · maths 3,987**. (chat leads because its shorter answers record fastest; maths/code/IF climb as run-2/run-3's stratified slices complete.) Finalized per-domain counts are written by `train_cp6.sh` (the `index_states` summary) at training start. Exclusions: `logs/phase5/cp6_prep/exclude_pids.txt` (IFEval 13-gram decontam + base_pid-flagged + 2 dups). No-op items: `logs/phase5/judge_states/noop_rids.txt` (currently 0).

## Backup / restore

States are backed up to the public HF dataset **Arushhh/phase5-loop-guard** at `logs/phase5/judge_sglang/<folder>/gpu*/states.*.npz` (the backup allowlist includes `logs/phase5`). The transient `judge_side/`, `judge_trace/`, `drafter_side/`, `drafter_trace/` dirs are **NOT uploaded** (created and deleted mid-run). See HUGGINGFACE.md §10 for the full dataset layout and the `hf download` restore.

**24 Oct 00:15 IST** — handover latest-start moved 00:55 → **01:00 IST** (run-2 ends ~00:58, its last items are maths/scarcest; trigger still = run-2 complete). Training still stops 04:15 IST. Watcher relaunched once (killed by PID, exactly one instance).
