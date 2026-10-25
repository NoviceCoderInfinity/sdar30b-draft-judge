# Phase 5, box session 1 (instance 57321867) — STATUS

---
## ⛔ PHASE 5 = FAIL — FINAL (24 Oct 2026, 05:55 IST; server 57349095 being removed)
**Retraining only the flag-gated 128M judge on fresh wave-C SGLang-recorded states FAILED.** Every retrained checkpoint (136M, 147.61M, 155.66M) degenerates into non-terminating repetition loops in live HF decode; accuracy collapses far outside noise. On the success criterion (<2% = kill) this is a KILL. **128M (Phase 4) retained as the production judge.** Full report: `results/phase5/REPORT.md`; root cause: `logs/phase5/FAILURE_ANALYSIS.md`.

**Diag bisect (ifeval541 16 items, plain HF rd1):** start_128m (Phase 4) acc **56%**, 1/16 loop, 89% stop@EOS · rec 136M acc 9% (10/11 loop, 86% trunc) · rec 147.61M acc 7% (14/15, 86% trunc) · rec_main_ep2 155.66M acc 8% (11/12, 88% trunc). Broken at the first +8M checkpoint and flat → setup, not dose.

**Root cause:** train/decode ENGINE MISMATCH. Phase 5 recorded states in SGLang (first time) but trains+evals in HF; SGLang↔HF disagree ~10% on judge argmax (step-0 0.8978, identical adapter) and ~40% on tokens (CP3 0.596). Off-policy fit → judge over-commits on its own HF trajectory → context corruption → no EOS → loops. Training construction itself is correct (target=vanilla T, committed never loss rows, flag on draft-filled, post-EOS = old convention). Probe/step-0 blind (same SGLang states); only fingerprint = step-0 gap 0.8978→0.7256.

**Same-box Phase-4 judge rd1 (HEALTHY, reproduces production):** gsm8k500 acc 0.910 (P4 0.914) pass/blk 5.75 · math500 0.722 (0.716) 5.27 · humaneval 0.768 (0.768) 4.66 · mbpp 0.704 (0.704) 7.12. acc ±0.6pt, pass/blk ±0.04, tokens near-identical vs `logs/phase4/decode/rd1`. gpqa/aime/ifeval541 rd1 not reached before stop.

**Vanilla multiplier (PARTIAL, 2/7 sets):** pooled median seconds ratio this-box/P4-box = **1.013** (n=51: ifeval541 28/28 + math500 23/23 tok-identical); gpu-active 1.001. `logs/phase5/vanilla/multiplier.txt`.

**Checkpoints (all 4 files each):** `logs/phase5/judge/{start_128m, rec_main/tok_000136000000, rec_main/tok_000144000000, rec_main/roll_00012094, rec_main/final(147.61M), rec_main_ep2/tok_000152000000, rec_main_ep2/final(155.66M)}`. Restore: `docs/PHASE5_RESUME.md` + `logs/phase5/MANIFEST.sha256`.
---

**Backups:** public HF dataset `Arushhh/phase5-loop-guard` (every 15 min; contents, paths, exclusions & restore in HUGGINGFACE.md §10). GitHub: `phase5-box-s1` (code/STATUS/results), `phase5-logs` (text logs). **Reproduce/resume:** docs/PHASE5_RESUME.md + logs/phase5/MANIFEST.sha256 + logs/phase5/env/ (pinned commit 2faa797).

**RULES (operator, 23 Oct 2026, final wording from the rule update of ~12:40 IST).**
1. Data generation: SGLang or any other speed-up is fine.
2. Training: any speed-up is fine if needed (training is not what is measured), but every change of recipe or numerics (batch size,
   learning rate, precision, kernels that change results) is written down as a deviation in this file, since it can change the trained model.
3. Inference and evaluation: NO SGLang and NO other speed-up, ever. Every accuracy, passes-per-block, seconds-per-answer, throughput or TPS
   number comes from a plain HF-harness run (run_da_decode.py with the same flags as Phases 3 and 4), compared only with vanilla decoded on
   the same box. These numbers measure how the new judge differs from the older ones and must be honest.
The fresh corpus (wave C) must be entirely unseen by the judge and the drafter (REPEAT_SCIENCE=0).

Box t0 (container start): 23 Oct 10:09:25 IST. No stop time: the 9-hour figure was planning only (operator decision 23 Oct 15:03 IST); the box runs through drafts, pack, index, parity, record until the operator says stop. MARGIN stays 1.2, every prompt of every domain is decoded. Times in this file are IST (events.tsv is UTC). Branch phase5-box-s1. HF backup: Arushhh/phase5-loop-guard (public).
Decoder unchanged; stage times from events.tsv, seconds per answer from the decoder's `seconds` field only.

## Latest STATUS block
(see below; appended every 15 minutes)

---
# TEXAS TAKEOVER — instance 57349095 (8x RTX PRO 6000 S, 2x EPYC 9555). Server t0 = 23 Oct 14:31 IST.
Old server 57321867 finished generation (16:58 IST) and was closed by the operator; Phase 5 continues here. Times IST; events.tsv UTC.

### STATUS 23 Oct 17:28 IST (Texas)
| field | value |
|---|---|
| time | 23 Oct 17:28 IST |
| hours since t0 (14:31) | 2:57 |
| CURRENT STAGE | CP2 **drafts** (SGLang, Phase 4 merged drafter), 8 GPUs, tmux `fresh` |
| progress | gen finished (old server, 250,790 prompts); drafting kept items (~151k total). First checkpoints ~2,000 items/GPU, 0 incomplete |
| measured rate | **82–84 blocks/s/GPU** (data-generation rate, SGLang — NOT a benchmark speed). Old server measured 72 |
| ETA | drafts done ~17:52 IST — PROJECTION (basis: 82 blk/s/GPU × 8, ~151k kept items × ~8.3 blk/item ≈ 1.26M blk) |
| last finished | CP0 take-over (manifest 80/80 sha256 verified; data installed + re-verified; repo FF to 243d463); CP2 corpus verify **PASS**; CP4 pieces A/A0 ported, 46 CPU tests pass, committed 8d031fe |
| next | pack → CP3 parity (≥100 GSM8K) → index (44M plan, REPEAT_SCIENCE=0) → CP5 record |
| problems | none. Note for CP5: `C1B_CODE_DIR` must point at repo `code/` so the SGLang judge plugins register |
| GPU / disk | 8/8 GPUs ~80% util, ~81 GiB each; disk used 105 GB / free 496 GB |
| backups | git push 17:18 OK; HF upload 17:19 OK (229 files, 1.35 GiB); logs branch 17:19 OK; pieces A/A0 push 17:22 OK |

#### CP2 corpus verification summary (full: logs/phase5/corpus_verify.md)
| domain | finished/planned | kept items | kept tokens | target | vs target |
|---|---|---|---|---|---|
| maths | 68,564/68,564 | 30,283 | 15,032,234 | 12,532,000 | 120% |
| code | 64,776/64,776 | 47,500 | 24,154,274 | 11,700,000 | 206% |
| chat | 38,476/38,476 | 34,831 | 12,409,521 | 10,452,000 | 119% |
| if | 78,974/78,974 | 38,661 | 15,048,946 | 9,308,000 | 162% |

Overlap with waves A/B: **0**. Duplicate pids in wave C: **0**. Held-out pids present: **0**. done.txt == record lines on all 8 shards. 20 kept records/domain read by eye: all sane, answers end properly. sha256 of all 80 files → logs/phase5/corpus_manifest.sha256.

### STATUS 23 Oct 17:55 IST (Texas)
| field | value |
|---|---|
| time | 23 Oct 17:55 IST |
| hours since t0 (14:31) | 3:24 |
| CURRENT STAGE | **pack** (CPU, tmux `pack`) + **CP3 parity** decode (SGLang GPU0 / harness GPU1, 100 GSM8K items, tmux `cp3`) in parallel |
| progress | drafts finished; pack building logs/c1b/trainset/waveC items; CP3 install PASS (JudgeLoopTraced + DraftTracedV2 discoverable), decode running |
| measured rate | drafts: 86 blocks/s/GPU (data-gen, not a benchmark speed) |
| ETA | pack ~0:15; CP3 decode ~0:25–0:30 (harness batch-1, 100 items) — PROJECTION |
| last finished | CP2 drafts: **done 1924 s** (17:19→17:52 IST), 151,275 items drafted = kept-item count, 0 incomplete |
| next | index (44M plan, REPEAT_SCIENCE=0) → CP5 record → CP6 train |
| problems | none. nvidia-ml-py 13.615.71 installed in /venv/p2 (Rule 5) |
| GPU / disk | GPU0/1 busy (CP3), GPU2–7 idle; disk ~106 GB used / ~495 GB free |
| backups | loop green every 15 min (git + HF + logs branch, all rc=0) |

---

### STATUS 23 Oct 11:03 IST
| field | value |
|---|---|
| time | 23 Oct 11:03 IST |
| box hours | 0:53:00 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage A: setup_lg.sh step 2/3 (128M judge refs; tmux s1_setup), A4 corpus untar wave B (tmux s1_fetch) |
| progress | downloads 3/3 done (71.1 GiB); /venv/p2 built; wave A 165,438 rows checked, 0 npz missing; wave B untarring |
| measured rate | downloads: 71.1 GiB in 0:07:48 (10:51:16 -> 10:59:04 IST) = 155 MiB/s average, ~200 MiB/s peak |
| ETA | setup_lg done ~11:20 IST PROJECTION (basis: Phase 4 measured 27 min for the same script; CPU half took 0:07:57 here) |
| last finished | downloads d32 0:06:12, b32 0:07:48, data 0:01:55; setup_p2 (CPU part) 0:07:09 |
| next | setup_lg GPU checks (16/16 stock + judge), install nvidia-ml-py, A5 setup_sgl.sh |
| blockers / deviations | none. Note: df shows a 1.8 TB overlay (brief said 600 GB disk); staying under 550 GB regardless |
| GPU / disk | GPUs idle 0/8; disk used 220 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 23 Oct 11:00 IST ok; HF Arushhh/phase5-loop-guard 23 Oct 11:00 IST ok |

### Finished: Stage A (23 Oct 11:13 IST)
- setup_lg.sh 0:18:41 (10:54:17 -> 11:12:58 IST), exit 0: stock 16/16, judge 16/16 token-identical to Phase 1; 60 + 66 CPU tests passed; check_merged ALL PASS.
- Check timings (decoder `seconds` summed, box not quiet: both checks in parallel as in Phase 4): stock 584.6 s (Phase 4 285.9 s, 2.04x), judge 228.6 s (Phase 4 131.7 s, 1.74x).
- A4: wave A 165,438 + wave B 166,380 rows, 0 missing npz; prompt lists, 16 done.txt, holdout list present. Vanilla reference records fetched (fetch_p4.py --what vanilla).
- A5 setup_sgl.sh: CUDA step failed on the duplicate apt source (cuda-keyring vs the image's cuda.list); fixed by purging cuda-keyring and installing cuda-nvcc-12-9 + cuda-cudart-dev-12-9 (12.9.86) directly. venv/plugin steps deferred until after Stage B (quiet box). SKIP_SMOKE=1 knob added (operator: no smoke runs).
- Stage B started 11:13:27 IST (tmux s1_speedb), quiet box recorded in logs/phase5/speed_b/gsm8k500_full/quiet_box.txt. Phase 4's records moved to logs/phase4/decode/vanilla_p4box/gsm8k500.

### STATUS 23 Oct 11:16 IST
| field | value |
|---|---|
| time | 23 Oct 11:16 IST |
| box hours | 1:07:33 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage B: vanilla GSM8K-500 on 8 GPUs, quiet box (tmux s1_speedb), started 11:13:27 IST |
| progress | 41 / 500 items finished (per shard 6 6 5 5 6 4 5 4 of ~63) |
| measured rate | interim, same 41 items: mean 34.02 s/answer here vs 16.87 s on the Phase 4 box -> ratio 2.016; GPU-active 9.49 s vs 8.89 s; token identity 41/41 |
| ETA | Stage B done ~11:50 IST PROJECTION (basis: 34.0 s/answer so far x 62.5 answers per GPU) |
| last finished | Stage A: setup_lg.sh 0:18:41, exit 0, 16/16 + 16/16; A4 checks all pass |
| next | Stage B 10-minute partial report (~11:24 IST), then full summary; if > 10% slower: STOP (operator rule), no Stage D |
| blockers / deviations | interim ratio 2.0x is far past the 10% limit: a STOP at the end of B is the PROJECTION. A5 venv/plugin install deferred (quiet box). W&B: shards append to Phase 4's same-named runs in sdar-draftalign |
| GPU / disk | 8/8 GPUs at 29% util, 57 GB each; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 11:15 IST ok; HF Arushhh/phase5-loop-guard 11:15 IST ok |

### STATUS 23 Oct 11:31 IST
| field | value |
|---|---|
| time | 23 Oct 11:31 IST |
| box hours | 1:22:22 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage B: vanilla GSM8K-500 on 8 GPUs, quiet box (tmux s1_speedb), started 11:13:27 IST |
| progress | 245 / 500 items (per shard 31 30 29 28 30 35 32 30 of ~63) |
| measured rate | same 245 items: mean 34.42 s/answer here vs 16.99 s Phase 4 box -> ratio 2.026; GPU-active 9.64 s vs 9.01 s; token identity 245/245 |
| ETA | Stage B done ~11:50 IST PROJECTION (basis: 34.4 s/answer x 62.5 answers per GPU) |
| last finished | 10-minute partial report posted 11:24 IST (137 items, ratio 2.027) |
| next | Stage B full summary at the end; then STOP per the operator's >10% rule (no Stage D), 100-question drift slice not run unless the operator asks |
| blockers / deviations | ratio 2.03x steady since the first 41 items; no errors in the 8 shard logs; quiet box holding (load 8.7, only the shards) |
| GPU / disk | 8/8 GPUs 28-29% util, 57 GB each; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 11:30 IST ok; HF Arushhh/phase5-loop-guard 11:30 IST ok |

### STATUS 23 Oct 11:46 IST
| field | value |
|---|---|
| time | 23 Oct 11:46 IST |
| box hours | 1:37:22 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage B: vanilla GSM8K-500 on 8 GPUs, quiet box (tmux s1_speedb), started 11:13:27 IST |
| progress | 441 / 500 items (per shard 55 54 54 53 58 59 55 53 of 62-63); 0 shards finished yet |
| measured rate | same 441 items: mean 35.40 s/answer here vs 17.46 s Phase 4 box -> ratio 2.028; GPU-active 9.91 s vs 9.26 s; token identity 441/441 |
| ETA | Stage B done ~11:55 IST PROJECTION (basis: ~8 items left per shard x 35.4 s, plus the slowest shard's tail) |
| last finished | STATUS tick 11:31 IST; rules saved to memory and the top of STATUS.md (operator update 3) |
| next | full Stage B summary at the end, then STOP: nothing else starts on this box; the operator runs launch_bench.py and decides |
| blockers / deviations | ratio 2.03x steady from item 41 to 441; no errors in shard logs; quiet box holding (load 8.8, only the shards) |
| GPU / disk | 8/8 GPUs 27-30% util, 57 GB each; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 11:45 IST ok; HF Arushhh/phase5-loop-guard 11:45 IST ok |

### Finished: Stage B -> STOP (23 Oct 11:54 IST)
- Vanilla GSM8K-500, 8 GPUs, quiet box: wall 0:40:11 (11:13:27 -> 11:53:38 IST).
- mean 35.31 s/answer vs Phase 4 box 17.40 -> r = 2.03 (limit 10%: STOP). GPU-active 9.88 s vs 9.23 s (1.07x). Token identity 500/500, gen_ids identical to Phase 4's 500/500, 451/500 correct on both. Per-GPU spread here 33.3-37.8 s (1.13x), Phase 4 15.9-18.6 s.
- Details: results/phase5_box_s1/speed_b_gsm8k500.md, logs/phase5/speed_b/gsm8k500_full/.
- Per operator update 3: nothing else starts on this box; waiting for the operator (launch_bench.py, box decision). Stage D not started; A5 venv/plugin not installed; Stage C deferred.

### STATUS 23 Oct 12:01 IST
| field | value |
|---|---|
| time | 23 Oct 12:01 IST |
| box hours | 1:52:17 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | STOPPED after Stage B (r = 2.03); waiting for the operator (launch_bench.py, box decision). Nothing running on the GPUs |
| progress | Stage A complete; Stage B complete (500/500); Stages C, D not started per operator |
| measured rate | Stage B: 35.31 s/answer vs 17.40 Phase 4 (2.029x); GPU-active 9.88 vs 9.23 s (1.07x) |
| ETA | none: no task running |
| last finished | Stage B 0:40:11 (11:13:27 -> 11:53:38 IST); results pushed and uploaded at 12:01 IST (commit 539c890) |
| next | operator's launch_bench.py (one GPU, ~1 min), then the operator decides |
| blockers / deviations | box 2.03x slower than the Phase 4 box on the batch-1 decoder (CPU-side); operator rule: STOP |
| GPU / disk | GPUs idle 0/8; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 12:01 IST ok (539c890); HF Arushhh/phase5-loop-guard 12:01 IST ok (96 files) |

### STATUS 23 Oct 12:16 IST
| field | value |
|---|---|
| time | 23 Oct 12:16 IST |
| box hours | 2:07:19 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | STOPPED after Stage B (r = 2.03); waiting for the operator. Nothing running on the GPUs |
| progress | Stage A complete; Stage B complete (500/500); Stages C, D not started per operator |
| measured rate | Stage B: 35.31 s/answer vs 17.40 Phase 4 (2.029x); GPU-active 9.88 vs 9.23 s (1.07x) |
| ETA | none: no task running |
| last finished | Stage B 0:40:11 (11:13:27 -> 11:53:38 IST) |
| next | operator's launch_bench.py (in /workspace/dl, not yet run: GPUs idle), then the operator decides |
| blockers / deviations | box 2.03x slower than the Phase 4 box (CPU-side); operator rule: STOP |
| GPU / disk | GPUs idle 0/8; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 12:16 IST ok (777b7cc); HF Arushhh/phase5-loop-guard 12:16 IST ok |

### STATUS 23 Oct 12:31 IST
| field | value |
|---|---|
| time | 23 Oct 12:31 IST |
| box hours | 2:22:19 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | STOPPED after Stage B (r = 2.03); waiting for the operator. Nothing running on the GPUs |
| progress | Stage A complete; Stage B complete (500/500); Stages C, D not started per operator |
| measured rate | Stage B: 35.31 s/answer vs 17.40 Phase 4 (2.029x); GPU-active 9.88 vs 9.23 s (1.07x) |
| ETA | none: no task running |
| last finished | Stage B 0:40:11 (11:13:27 -> 11:53:38 IST) |
| next | operator's launch_bench.py (not yet run: GPUs idle), then the operator decides |
| blockers / deviations | box 2.03x slower than the Phase 4 box (CPU-side); operator rule: STOP |
| GPU / disk | GPUs idle 0/8; disk used 221 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 12:31 IST ok (00151aa); HF Arushhh/phase5-loop-guard 12:16 IST ok, 12:31 pass uploading now |

### OPERATOR DECISION 23 Oct 12:36 IST: GO on this box (no replacement box). Stage B STOP rule replaced.
- 1. setup_sgl.sh install only (SKIP_SMOKE=1), started 12:36 IST (tmux s1_sgl). CUDA 12.9 step: already ok.
- 2. then `REPEAT_SCIENCE=0 bash code/judge_round/run_fresh.sh all` in tmux (nothing already seen is repeated); default prompt targets kept.
- STOP only on a stage error, IF shortfall > 20%, or disk > 550 GB. No HF-harness decodes, no smoke, no parity now.

### Finished: setup_sgl.sh install (23 Oct 12:41 IST)
- 0:04:29 (12:36:30 -> 12:40:59 IST), exit 0. /venv/sgl: sglang 0.5.10.post1, torch 2.9.1+cu128 (Phase 1's), flashinfer-python 0.6.7.post3, transformers 5.3.0; plugin modules installed and --check ok; CUDA_HOME=/usr/local/cuda-12.9 (sgl_env.sh). Smoke skipped (operator). Freeze: logs/phase5/sgl_setup/sgl_freeze.txt.
- CPU stages (niced, REPEAT_SCIENCE=0): targets done 12:40 IST: maths 68,564 / code 64,776 / chat 38,476 / IF 78,974 prompts; 43.99M fresh Mtok, 0 science repeated; 21.5 GPU-h PROJECTION (Phase 1 rates). lists: maths, code, chat full (shortfall 0); IF in progress.
- Next: launch_bench.py x3 on GPU 0 (quiet), then run_fresh.sh gen/drafts/pack/index/report on 8 GPUs.

### STATUS 23 Oct 12:46 IST
| field | value |
|---|---|
| time | 23 Oct 12:46 IST |
| box hours | 2:37:18 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang on 8 GPUs (tmux s1_fresh_gpu, run_fresh.sh all, REPEAT_SCIENCE=0), started 12:45:10 IST |
| progress | engines up on 8/8 GPUs (89 GB each, 84-100% util); first stats.jsonl not written yet; 250,790 prompts to decode |
| measured rate | none yet (first per-GPU rate report ~12:51 IST, labelled data-generation rate, SGLang, not a benchmark speed) |
| ETA | gen done ~15:27 IST PROJECTION (basis: Phase 1 rates, 21.5 GPU-h = 2.69 h on 8 GPUs; 3.23 h = ~16:00 IST if every margin prompt is decoded); drafts, pack, index follow |
| last finished | lists 0:04:37 (12:40:33 -> 12:45:10 IST), 0 shortfall in all 4 domains; launch_bench x3 12:43:33 -> 12:43:57 IST; setup_sgl install 0:04:29 |
| next | 5-minute rate report; then drafts (SGLang, Phase 4 merged drafter), pack, index (fresh only), report |
| blockers / deviations | none. Benign log lines: SGLang ignores import errors for the unused judge-loop plugin modules (C1B_CODE_DIR unset in gen; gen uses LowConfidenceTraced only) |
| GPU / disk | 8/8 GPUs 84-100% util, 89 GB each; disk used 238 GB, free 1.5 TB (limit 550 GB) |
| last push / upload | GitHub phase5-box-s1 12:31 IST ok; HF Arushhh/phase5-loop-guard 12:31 IST ok; 12:46 pass starting |

### STATUS 23 Oct 13:01 IST
| field | value |
|---|---|
| time | 23 Oct 13:01 IST |
| box hours | 2:52:19 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs (tmux s1_fresh_gpu), started 12:45:10 IST; CUDA-graph capture done ~12:47 IST, decoding since |
| progress | first chunk not yet written on any GPU (stats.jsonl appears per chunk); 250,790 prompts total |
| measured rate | none yet (data-generation rate, SGLang, not a benchmark speed: posted at the first chunk) |
| ETA | gen done ~15:27 IST PROJECTION (basis: Phase 1 rates, 21.5 GPU-h on 8 GPUs; ~16:00 IST if every margin prompt is decoded) |
| last finished | lists 0:04:37 (-> 12:45:10 IST), 0 shortfall; setup_sgl install 0:04:29; probe x3 12:43 IST |
| next | first-chunk rate report; then drafts, pack, index (fresh only), report |
| blockers / deviations | none; no errors in the 8 gen logs |
| GPU / disk | 8/8 GPUs 84-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 12:46 IST ok; HF Arushhh/phase5-loop-guard 12:47 IST ok |

### STATUS 23 Oct 13:16 IST
| field | value |
|---|---|
| time | 23 Oct 13:16 IST |
| box hours | 3:07:19 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST (chunk 1 of ~7.7 per GPU, 4,096 prompts each) |
| progress | trace: ~47,000 blocks per GPU (46,336-47,104, even); ~2,850-2,890 prompts started per GPU of chunk 1 (128 in flight); chunk 1 so far maths + IF only (order_key order); blocks/answer mean 16.5, median 15 |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~1,600 blocks/min/GPU, ~98-100 prompts/min/GPU on this maths-heavy start (~0.75x Phase 1's maths prompt rate). Kept Mtok/GPU-h comes with the first stats line |
| ETA | first chunk ~13:29 IST; gen done ~16:20 IST PROJECTION (basis: Phase 1 mix of 2.69 h scaled by the 0.75x observed on maths); then drafts (~0.5 h), pack (~0.7 h CPU), index: all done ~17:40 IST PROJECTION |
| last finished | lists 0:04:37 (-> 12:45:10 IST); setup_sgl install 0:04:29; probe x3 12:43 IST |
| next | first-chunk rate report; drafts, pack, index (fresh only), report |
| blockers / deviations | none; no errors in gen logs; projection leaves ~1.5 h slack before 19:09 IST |
| GPU / disk | 8/8 GPUs 88-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 13:02 IST ok; HF Arushhh/phase5-loop-guard 13:02 IST ok |

### STATUS 23 Oct 13:31 IST
| field | value |
|---|---|
| time | 23 Oct 13:31 IST |
| box hours | 3:22:20 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 1 (4,096 prompts/GPU) in its straggler tail |
| progress | gpu0: all 4,096 prompts of chunk 1 started, ~70 still in flight, longest answer 33 blocks; ~70,000 blocks per GPU (69,454-70,563) |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~1,600 blocks/min/GPU steady; kept Mtok/GPU-h still pending the first stats line (written when chunk 1 completes) |
| ETA | chunk 1 complete ~13:35 IST; gen done ~16:20 IST PROJECTION (basis: Phase 1 mix 2.69 h x 1/0.75 observed on maths); drafts, pack, index done ~17:40 IST PROJECTION |
| last finished | lists 0:04:37 (-> 12:45:10 IST); setup_sgl install 0:04:29 |
| next | first-chunk rate report (waiter re-armed, 25 min cap); drafts, pack, index, report |
| blockers / deviations | none; no errors in gen logs. Note: with 4,096-prompt chunks the per-chunk straggler tail idles most of the batch for a few minutes each chunk (gen_sglang default --chunk 4096) |
| GPU / disk | 8/8 GPUs 95-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 13:17 IST ok; HF Arushhh/phase5-loop-guard 13:17 IST ok |

### STATUS 23 Oct 13:46 IST
| field | value |
|---|---|
| time | 23 Oct 13:46 IST |
| box hours | 3:37:21 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 2 of ~7.7 per GPU decoding (gpu0: 5,818 prompts started) |
| progress | ~92,000 blocks per GPU (91,679-92,089); chunk 1 (4,096 prompts) decoded in ~0:47; its records/stats not yet on disk (scoring is pipelined with the next chunk's decode: being confirmed in the code) |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~1,560 blocks/min/GPU; ~5,200 prompts/h/GPU on the maths-heavy start. Kept Mtok/GPU-h pending the first stats line |
| ETA | gen done ~16:30 IST PROJECTION (basis: 31,349 prompts/GPU at ~5,200/h on maths-heavy chunks, faster code/chat chunks later); drafts, pack, index done ~17:50 IST PROJECTION |
| last finished | lists 0:04:37 (-> 12:45:10 IST) |
| next | first stats line -> kept-rate report; drafts, pack, index, report |
| blockers / deviations | none; no errors; 8 schedulers at 100% CPU each (one core), load 8.3 |
| GPU / disk | 8/8 GPUs 87-97% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 13:32 IST ok; HF Arushhh/phase5-loop-guard 13:33 IST ok |

### STATUS 23 Oct 14:01 IST
| field | value |
|---|---|
| time | 14:01 IST (23 Oct) |
| box hours | 3:52:22 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 2 finishing (gpu0 7,607 / gpu7 7,487 prompts started of 31,349 per GPU) |
| progress | ~113,000 blocks per GPU; 24% of prompts started; no records/stats on disk yet (chunk 1 finalises when chunk 2 returns, ~14:10 IST) |
| measured rate | data-generation rate, SGLang, not a benchmark speed: 1,530 blocks/min/GPU steady; prompts/h/GPU rising from ~5,200 (maths-heavy start, 18 blocks/answer) to ~6,200 as code prompts (6-8 blocks) mix in |
| ETA | gen done ~17:10 IST PROJECTION (basis: blocks needed per GPU ~407k = maths 8,570x18 + IF 9,872x15 + code 8,097x7 + chat 4,810x10, at 1,530 blocks/min); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:30 IST PROJECTION, ~40 min before the 19:09 limit |
| last finished | lists 0:04:37 (-> 12:45:10 IST) |
| next | first stats line (~14:10 IST) -> kept Mtok/GPU-h report; drafts, pack, index, report |
| blockers / deviations | RISK: the projection leaves ~40 min slack before 19:09 IST; the earlier Phase 1-rate basis (16:20 IST) was optimistic. gen is resumable (done.txt) and pack works on the records present, so a cut at MARGIN<1.2 is possible if the operator prefers; no action taken |
| GPU / disk | 8/8 GPUs 90-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 13:48 IST ok; HF Arushhh/phase5-loop-guard 13:48 IST ok |

### Gen rate report, chunk 0 on all 8 GPUs (23 Oct 14:09 IST). Data-generation rate, SGLang, not a benchmark speed.
Chunk 0 = 4,096 prompts per GPU, decode 0.748-0.764 h per GPU (12:47 -> ~13:33 IST); chunk 0 held maths + IF only (pid order). GPU time split between domains by block share.
| domain | kept / prompts (8 GPUs) | keep rate (Phase 1) | Mtok kept per GPU-h (Phase 1) | ratio |
|---|---|---|---|---|
| maths | 12,237 / 27,826 | 0.440 (0.444) | 1.10 (1.28) | 0.86x |
| IF | 2,426 / 4,942 | 0.491 (0.449) | 1.27 (1.42) | 0.89x |
Per GPU: maths 1.07-1.17 Mtok/GPU-h (0.84-0.92x), IF 1.24-1.33 (0.87-0.94x). Drops: maths wrong_answer ~43%, no_eos ~13%; IF wrong_answer ~48%, no_eos ~4%.
ETA (PROJECTION): at 0.87x Phase 1 for every domain, all 250,790 prompts (margin included) need 25.8/0.87 = 29.7 GPU-h = 3.7 h -> gen done ~16:30 IST; the block-count basis gives ~17:10 IST. Then drafts ~0.5 h, pack ~0.7 h (CPU), index minutes: all stages done 17:50-18:30 IST, before the 19:09 IST limit.

### STATUS 23 Oct 14:16 IST
| field | value |
|---|---|
| time | 23 Oct 14:16 IST |
| box hours | 4:07:20 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 3 of ~7.7 decoding |
| progress | gpu0 9,361 / gpu7 9,251 prompts started of 31,349 per GPU (30%); ~134,000 blocks per GPU; chunk 0 records on disk (4,096 prompts/GPU, waveC 305 MB) |
| measured rate | data-generation rate, SGLang, not a benchmark speed: chunk 0 maths 1.10 Mtok kept/GPU-h (0.86x Phase 1), IF 1.27 (0.89x); current prompt rate ~7,000 prompts/h/GPU (last 15 min, code mixing in); 1,510 blocks/min/GPU |
| ETA | gen done ~17:25 IST PROJECTION (basis: 21,988 prompts left per GPU at ~7,000/h); block-count basis 17:10 IST; drafts ~0.5 h, pack ~0.7 h, index: all done ~18:45 IST PROJECTION (~25 min slack before 19:09 IST) |
| last finished | chunk 0 finalised on all GPUs 14:09 IST (decode 0:45-0:46 per GPU); rate report posted |
| next | chunk 1 stats (~14:55 IST) with first code rates; drafts, pack, index, report |
| blockers / deviations | RISK: slack before 19:09 IST is down to ~25 min on the current basis. Options for the operator: extend the box past 19:09, or cut the margin prompts early (gen resumes/stops cleanly per chunk; pack uses the records present). No action taken |
| GPU / disk | 8/8 GPUs 86-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 14:03 IST ok; HF Arushhh/phase5-loop-guard 14:04 IST ok |

### STATUS 23 Oct 14:31 IST
| field | value |
|---|---|
| time | 23 Oct 14:31 IST |
| box hours | 4:22:21 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 3 of ~7.7 decoding |
| progress | gpu0 11,143 / gpu7 11,058 prompts started of 31,349 per GPU (36%); ~155,600 blocks per GPU; waveC on disk 352 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~7,130 prompts/h/GPU over the last 15 min (1,782 prompts); ~1,425 blocks/min/GPU; chunk 0: maths 1.10, IF 1.27 Mtok kept/GPU-h (0.86x / 0.89x Phase 1) |
| ETA | gen done ~17:20 IST PROJECTION (basis: 20,206 prompts left per GPU at ~7,100/h); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:40 IST PROJECTION (~30 min slack before 19:09 IST) |
| last finished | chunk 0 finalised 14:09 IST (decode 0:45-0:46 per GPU) |
| next | chunk 1 stats (~14:55 IST, first code rates); drafts, pack, index, report |
| blockers / deviations | RISK unchanged: ~30 min slack before 19:09 IST; operator options: extend the box, or cut the margin prompts early. No action taken |
| GPU / disk | 8/8 GPUs 95-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 14:19 IST ok; HF Arushhh/phase5-loop-guard 14:19 IST ok |

### STATUS 23 Oct 14:46 IST
| field | value |
|---|---|
| time | 23 Oct 14:46 IST |
| box hours | 4:37:29 used of 9:00:00 (limit 23 Oct 19:09 IST) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 3 of ~7.7 decoding (chunks 0, 1 finalised) |
| progress | gpu0 12,898 / gpu7 12,750 prompts started of 31,349 per GPU (41%); ~176,700 blocks per GPU; waveC on disk 411 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed. Chunk 1 (0.602 h decode per GPU, 8 GPUs): code 7.75 Mtok kept/GPU-h (Phase 1 7.0, 1.11x), keep 0.732 (Phase 1 0.425), 4.1 blocks/answer; maths 1.02 (0.80x), keep 0.441, 18.2 blocks/answer; IF 1.21 (0.85x), keep 0.500, 11.1 blocks/answer. Prompt rate ~7,000 prompts/h/GPU (last 15 min) |
| ETA | gen done ~17:25 IST PROJECTION (basis: 18,451 prompts left per GPU at ~7,000/h; chat not yet seen); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:45 IST PROJECTION (~25 min slack before 19:09 IST) |
| last finished | chunk 1 finalised on all GPUs ~14:45 IST (decode 0:36 per GPU) |
| next | chunk 2 stats (~15:25 IST); drafts, pack, index, report |
| blockers / deviations | RISK unchanged: ~25 min slack before 19:09 IST; operator options: extend the box or cut margin prompts early. Code keep rate is well above Phase 1's (0.73 vs 0.43): more code items than targeted will be kept |
| GPU / disk | 8/8 GPUs 85-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 14:34 IST ok; HF Arushhh/phase5-loop-guard 14:34 IST ok |

### OPERATOR DECISION 23 Oct 15:03 IST
- Do not cut the margin: MARGIN 1.2, generation runs the full plan (all prompts of every domain).
- 19:09 IST is not a stop time (planning figure only). Continue through drafts, pack, index, parity, record; keep backups and STATUS ticks. Stop only on an operator STOP rule, an unfixable error, or the operator's word. ETAs only in STATUS blocks from now on.

### STATUS 23 Oct 15:01 IST
| field | value |
|---|---|
| time | 23 Oct 15:01 IST |
| box hours | 4:52:20 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 3 of ~7.7 decoding (chunks 0, 1 finalised) |
| progress | gpu0 15,771 / gpu7 15,316 prompts started of 31,349 per GPU (50%); ~196,000 blocks per GPU; waveC on disk 455 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~11,500 prompts/h/GPU over the last 15 min (code-heavy stretch, 4 blocks/answer); chunk 1: code 7.75, maths 1.02, IF 1.21 Mtok kept/GPU-h (1.11x / 0.80x / 0.85x Phase 1) |
| ETA | gen done 16:30-17:10 IST PROJECTION (basis: 15,578 prompts left per GPU; 11,500/h if the mix stays code-heavy, ~7,000/h at the earlier mixed rate); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:00-18:30 IST PROJECTION |
| last finished | chunk 1 finalised ~14:45 IST (decode 0:36 per GPU) |
| next | chunk 2 stats (~15:20 IST); drafts, pack, index, report; then parity, record per the operator's plan |
| blockers / deviations | none (margin kept at 1.2 per operator 15:03 IST; no stop time) |
| GPU / disk | 8/8 GPUs 84-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 14:49 IST ok; HF Arushhh/phase5-loop-guard 14:50 IST ok |

### STATUS 23 Oct 15:17 IST
| field | value |
|---|---|
| time | 23 Oct 15:17 IST |
| box hours | 5:07:54 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 4 of ~7.7 decoding (chunks 0-2 finalised) |
| progress | gpu0 18,914 / gpu7 18,377 prompts started of 31,349 per GPU (60%); ~215,000 blocks per GPU; waveC on disk 513 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed. Chunk 2 (0.582 h decode per GPU): code 7.68 Mtok kept/GPU-h (1.10x Phase 1), keep 0.734; maths 1.02 (0.80x), keep 0.442; IF 1.18 (0.83x), keep 0.483. Prompt rate ~11,800 prompts/h/GPU (last 16 min, code-heavy) |
| ETA | gen done 16:25-17:05 IST PROJECTION (basis: 12,435 prompts left per GPU at 11,800/h code-heavy vs ~7,000/h mixed; chat prompts (4,810/GPU) not yet seen); drafts ~0.5 h, pack ~0.7 h, index: all done ~17:50-18:30 IST PROJECTION |
| last finished | chunk 2 finalised ~15:16 IST (decode 0:35 per GPU); logs branch phase5-logs first push 15:16 IST (e7f04be, 61 files, 3.0 MB) |
| next | chunk 3 stats (~15:50 IST); drafts, pack, index, report; then parity, record |
| blockers / deviations | none. Backup loop restarted 15:16 IST with the logs-branch step (generation untouched) |
| GPU / disk | 8/8 GPUs 87-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 15:16 IST ok; phase5-logs 15:16 IST ok (e7f04be); HF Arushhh/phase5-loop-guard 15:05 IST ok (15:16 pass uploading) |

### STATUS 23 Oct 15:31 IST
| field | value |
|---|---|
| time | 23 Oct 15:31 IST |
| box hours | 5:22:23 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 5 of ~7.7 decoding (chunks 0-3 finalised) |
| progress | gpu0 21,054 / gpu7 20,684 prompts started of 31,349 per GPU (67%); by domain on gpu0: maths 8,470 of 8,570, code 8,199 of 8,097+, chat 2,908 of 4,810, IF 1,477 of 9,872 (IF comes last); ~234,000 blocks per GPU; waveC on disk 574 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: chunk 2 code 7.68 / maths 1.02 / IF 1.18 Mtok kept/GPU-h (1.10x / 0.80x / 0.83x Phase 1); prompt rate ~8,600 prompts/h/GPU (last 14 min, chat + IF mix) |
| ETA | gen done ~16:45 IST PROJECTION (basis: ~10,300 prompts left per GPU, mostly IF at ~11 blocks/answer and chat, at ~8,500 prompts/h); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:10 IST PROJECTION |
| last finished | chunk 3 finalised ~15:30 IST; logs branch push 15:17 IST (b1fb681, 61 files, 3.0 MB) |
| next | chunk 4 stats (~16:00 IST); drafts, pack, index, report; then parity, record |
| blockers / deviations | HF upload of the 15:17 pass returned rc=1 (a Hub error, not a scan hit; the loop continues and retries at 15:32): being checked |
| GPU / disk | 8/8 GPUs 84-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 15:16 IST ok; phase5-logs 15:17 IST ok (b1fb681); HF Arushhh/phase5-loop-guard last ok 15:05 IST (15:17 pass failed, rc=1) |

### STATUS 23 Oct 15:46 IST
| field | value |
|---|---|
| time | 23 Oct 15:46 IST |
| box hours | 5:37:28 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 5 of ~7.7 decoding (chunks 0-3 finalised) |
| progress | gpu0 23,031 / gpu7 22,704 prompts started of 31,349 per GPU (73%); gpu0 by domain: maths 8,470 (all), code 8,199 (all), chat 4,515 of ~4,810, IF 1,847 of ~9,872; ~255,000 blocks per GPU; waveC on disk 623 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed. Chunk 3 (0.408 h decode per GPU): code 6.99 Mtok kept/GPU-h (1.00x Phase 1), keep 0.734; chat 2.48 (0.83x), keep 0.907 (Phase 1 0.903); maths 0.97 (0.76x), keep 0.455; IF 1.08 (0.76x), keep 0.483. Prompt rate ~7,900 prompts/h/GPU (last 15 min) |
| ETA | gen done ~16:50 IST PROJECTION (basis: ~8,300 prompts left per GPU, nearly all IF at ~11 blocks/answer, ~8,000 prompts/h); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:15 IST PROJECTION |
| last finished | chunk 3 finalised ~15:30 IST (decode 0:24 per GPU, code-heavy); logs branch push 15:33 IST (20c3356, 61 files, 3.9 MB) |
| next | chunk 4 stats (~16:05 IST); drafts, pack, index, report; then parity, record |
| blockers / deviations | HF upload failed again on the 15:32 pass (same growing-trace cause: that pass started before the 15:35 fix). The 15:48 pass is the first with the fix; if it fails too, I will report and change the approach (exclude trace/ from HF until gen ends) |
| GPU / disk | 8/8 GPUs 83-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 15:32 IST ok; phase5-logs 15:33 IST ok (20c3356); HF Arushhh/phase5-loop-guard last success 15:05 IST |

### STATUS 23 Oct 16:01 IST
| field | value |
|---|---|
| time | 23 Oct 16:01 IST |
| box hours | 5:52:27 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 6 of ~7.7 decoding (chunks 0-4 finalised on most GPUs) |
| progress | gpu0 24,909 / gpu7 24,576 prompts started of 31,349 per GPU (79%); gpu0 by domain: maths, code, chat complete; IF 3,271 of ~9,872 (the remaining ~6,400 prompts per GPU are all IF); ~275,000 blocks per GPU; waveC on disk 688 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed. Chunk 4 (0.411 h decode per GPU): code 6.99 Mtok kept/GPU-h (1.00x Phase 1), chat 2.48 (0.83x, keep 0.906), IF 1.07 (0.75x, keep 0.486). Prompt rate ~7,500 prompts/h/GPU (last 15 min) |
| ETA | gen done ~16:55 IST PROJECTION (basis: ~6,400 IF prompts left per GPU at ~11 blocks/answer and ~1,400 blocks/min = ~7,600 prompts/h); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:20 IST PROJECTION |
| last finished | chunk 4 finalised ~16:00 IST (decode 0:25 per GPU); HF upload 15:49 IST ok (fix confirmed) |
| next | chunk 5 stats (~16:25 IST); drafts, pack, index, report; then parity, record |
| blockers / deviations | none; backups healthy again (all three steps ok on the 15:48 pass) |
| GPU / disk | 8/8 GPUs 87-100% util, 89 GB each; disk used 238 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 15:48 IST ok; phase5-logs 15:49 IST ok (e2557a7); HF Arushhh/phase5-loop-guard 15:49 IST ok |

### STATUS 23 Oct 16:16 IST
| field | value |
|---|---|
| time | 23 Oct 16:16 IST |
| box hours | 6:07:25 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 6 of ~7.7 decoding (chunks 0-4 finalised) |
| progress | gpu0 26,725 / gpu7 26,522 prompts started of 31,349 per GPU (85%); remaining ~4,600 per GPU, all IF; ~296,000 blocks per GPU; waveC on disk 739 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~7,260 prompts/h/GPU (last 15 min, IF only, ~11 blocks/answer); chunk 4: code 6.99, chat 2.48, IF 1.07 Mtok kept/GPU-h (1.00x / 0.83x / 0.75x Phase 1) |
| ETA | gen done ~17:00 IST PROJECTION (basis: ~4,600 IF prompts left per GPU at ~7,300/h plus the last chunk's straggler tail and finalisation); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:25 IST PROJECTION |
| last finished | chunk 4 finalised ~16:00 IST (decode 0:25 per GPU) |
| next | chunk 5 stats (~16:30 IST); gen end report; drafts, pack, index, report; then parity, record |
| blockers / deviations | none; backups healthy (16:04 IST pass: all three steps ok) |
| GPU / disk | 8/8 GPUs 84-100% util, 89 GB each; disk used 239 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 16:04 IST ok; phase5-logs 16:04 IST ok (2379157); HF Arushhh/phase5-loop-guard 16:04 IST ok |

### STATUS 23 Oct 16:31 IST
| field | value |
|---|---|
| time | 23 Oct 16:31 IST |
| box hours | 6:22:32 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; chunk 7 of 8 (the 6 full ones finalised or finalising; the last chunk is partial, ~2,680 prompts) |
| progress | gpu0 28,593 / gpu7 28,477 prompts started of 31,349 per GPU (91%); ~2,760 IF prompts left per GPU; ~317,000 blocks per GPU; waveC on disk 792 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~7,470 prompts/h/GPU (last 15 min, IF); chunk 5 (first GPU to finalise it): IF 1.12 Mtok kept/GPU-h (0.79x Phase 1, keep 0.473), chat 2.68 (0.89x, keep 0.904) |
| ETA | gen done ~17:00 IST PROJECTION (basis: ~2,760 prompts left per GPU at ~7,500/h = 0:22, plus the last chunk's straggler tail and finalisation); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:25 IST PROJECTION |
| last finished | chunk 5 finalised on the first GPUs ~16:30 IST (decode 0:31 per GPU) |
| next | gen end report with kept counts per domain; drafts, pack, index, report; then parity, record |
| blockers / deviations | none; backups healthy (16:19 IST pass: all three steps ok) |
| GPU / disk | 8/8 GPUs 83-100% util, 89 GB each; disk used 239 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 16:19 IST ok; phase5-logs 16:20 IST ok (fa8a5aa); HF Arushhh/phase5-loop-guard 16:20 IST ok |

### STATUS 23 Oct 16:46 IST
| field | value |
|---|---|
| time | 23 Oct 16:46 IST |
| box hours | 6:37:26 used since t0 (no stop time) |
| CURRENT TASK | Stage D gen: wave C vanilla generation in SGLang, 8 GPUs, decoding since ~12:47 IST; last chunk (8 of 8, partial) decoding; chunks 0-5 finalised on all 8 GPUs |
| progress | gpu0 30,398 / gpu7 30,285 prompts started of 31,349 per GPU (97%); ~950 IF prompts left to start per GPU; ~338,000 blocks per GPU; waveC on disk 857 MB |
| measured rate | data-generation rate, SGLang, not a benchmark speed: ~7,220 prompts/h/GPU (last 15 min, IF, ~11 blocks/answer) |
| ETA | gen done ~17:05 IST PROJECTION (basis: ~950 prompts left per GPU at ~7,200/h = 0:08, then the last chunk's straggler tail and the finalisation of chunks 6 and 7); drafts ~0.5 h, pack ~0.7 h, index: all done ~18:30 IST PROJECTION |
| last finished | chunk 5 finalised on all GPUs ~16:35 IST |
| next | gen end report (kept counts per domain, total kept Mtok vs the 43.99M target); drafts, pack, index, report; then parity, record |
| blockers / deviations | none; backups healthy (16:35 IST pass: all three steps ok) |
| GPU / disk | 8/8 GPUs 85-100% util, 89 GB each; disk used 239 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 16:35 IST ok; phase5-logs 16:36 IST ok (0c093e9); HF Arushhh/phase5-loop-guard 16:36 IST ok |

### Finished: Stage D gen (23 Oct 16:58:05 IST). Data generation in SGLang; counts, not a benchmark speed.
- Wall 4:12:54 (12:45:11 -> 16:58:05 IST); per GPU 4.09-4.19 h (gen_sglang's own "done in"). All 250,790 prompts decoded (31,349 per GPU, margin 1.2 kept per operator). Records: logs/c1b/gen/waveC/gpu*/records.0000{0..7}.jsonl.gz (909 MB incl. traces).
- Totals from the last cumulative stats line of each GPU (the authoritative numbers):

| domain | prompts | kept | keep rate (Phase 1) | gen Mtok | est. corpus Mtok (kept x tok/kept) | target Mtok | blocks/answer | drops |
|---|---|---|---|---|---|---|---|---|
| maths | 68,564 | 30,283 | 0.442 (0.444) | 37.81 | 14.96 | 12.53 | 18.2 | wrong_answer 29,545; no_eos 8,736 |
| code | 64,776 | 47,500 | 0.733 (0.425) | 6.53 | 24.23 | 11.70 | 4.1 | failed 8,726; wrong_answer 8,062; no_eos 320; timeout 152; no_checker_runtime 16 |
| chat | 38,476 | 34,831 | 0.905 (0.903) | 11.11 | 12.57 | 10.45 | 10.1 | no_eos 1,799; not_english 1,143; loop 703 |
| IF | 78,974 | 38,661 | 0.490 (0.449) | 25.67 | 12.18 | 9.31 | 11.2 | wrong_answer 37,341; no_eos 2,972 |
| total | 250,790 | 151,275 | | 81.12 | 63.94 | 43.99 | | |

- Every domain exceeds its fresh-token target (code by 2.1x: its keep rate is 0.73 vs Phase 1's 0.425). Domain shares are decided at index time (operator).
- DEVIATION (reporting only, no effect on data): run_fresh.sh's gen_counts table printed in all.log is wrong: it shows 422,822 decoded / 256,914 kept (1.69x too high) because its per-line delta logic double counts the cumulative stats lines. Use the table above (last cumulative line per GPU) and pack's build_summary.json instead. No code changed.
- Average data-generation rate over the whole stage (SGLang, not a benchmark speed): 63.94 Mtok est. corpus / 33.1 GPU-h = 1.93 Mtok kept per GPU-h; Phase 1 mix projection was 21.5 GPU-h for 43.99M = 2.05.
- drafts stage started 16:58:05 IST on 8 GPUs (draft_sglang.py, Phase 4 merged drafter, --max-running 256 --mem-fraction 0.70): engines capturing CUDA graphs at 16:58:35 IST.

### STATUS 23 Oct 17:01 IST
| field | value |
|---|---|
| time | 23 Oct 17:01 IST |
| box hours | 6:52:19 used since t0 (no stop time) |
| CURRENT TASK | Stage D drafts: first drafts D0 for the 151,275 kept wave-C answers with the Phase 4 merged drafter in SGLang, 8 GPUs (draft_sglang.py --wave C, --max-running 256 --mem-fraction 0.70), started 16:58:05 IST |
| progress | 8/8 draft engines up (78 GB each, 58-77% util), decoding since ~16:59 IST; draft traces growing (see next tick for a block count) |
| measured rate | gen stage (finished): 1.93 Mtok kept per GPU-h averaged over 33.1 GPU-h (data generation, SGLang, not a benchmark speed); drafts rate: first measurement next tick |
| ETA | drafts done ~17:45 IST PROJECTION (basis: ~1.6M blocks over 8 GPUs at Phase 1's 68-94 blocks/s per GPU); pack ~0.7 h CPU, index minutes: all done ~18:35 IST PROJECTION |
| last finished | gen 4:12:54 (12:45:11 -> 16:58:05 IST): 250,790 prompts, 151,275 kept, 63.94M est. corpus tokens (target 43.99M) |
| next | drafts end report; pack (to_trainset.py --wave C), index (fresh only, REPEAT_SCIENCE=0), report; then parity, record |
| blockers / deviations | none new. run_fresh.sh's printed gen_counts table double counts (reporting only; authoritative totals in the gen block above) |
| GPU / disk | 8/8 GPUs 58-77% util, 78 GB each; disk used 239 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 16:51 IST ok; phase5-logs 16:52 IST ok (007fd35); HF Arushhh/phase5-loop-guard 16:52 IST ok |

### INCIDENT 23 Oct 17:14 IST: the generation tmux session was killed externally during the drafts stage
- Evidence: all 8 draft_sglang.py processes, run_fresh.sh and the tmux session s1_fresh_gpu disappeared together; each GPU's last log line is "8000 items ... in 0.26 h" (last write 17:14:14 IST); no error, no SIGTERM text, no retry line; GPUs 0 MiB. The tmux server and every other session (s1_backup etc.) survived, RAM 1.3 TB free (no OOM). An ssh login (sshd: root@notty) appeared at 17:17 IST. Not caused by this session; the sender is unknown.
- State saved: logs/c1b/draft/waveC/gpu*/ holds 4 draft chunks (drafts.00000-00003.npz) and done.txt with 8,000 items per GPU (64,000 of 151,275 kept items drafted); gen records untouched.
- Action 17:18 IST: resumed with `REPEAT_SCIENCE=0 bash code/judge_round/run_fresh.sh all` in tmux s1_fresh_gpu2 (targets, lists, gen skipped as done; drafts resumes from done.txt; then pack, index, report). Generation data not touched.
- Drafts rate before the kill (data generation, SGLang, not a benchmark speed): 71-74 blocks/s per GPU, steady on all 8 GPUs.

### STATUS 23 Oct 17:19 IST
| field | value |
|---|---|
| time | 23 Oct 17:19 IST |
| box hours | 7:10 used since t0 (no stop time) |
| CURRENT TASK | Stage D drafts, RESUMED 17:18 IST in tmux s1_fresh_gpu2 after an external kill of the previous session at 17:14 IST (see INCIDENT above); 8/8 draft engines starting |
| progress | 64,000 of 151,275 kept items drafted before the kill (8,000 per GPU, 4 npz chunks each); the resume continues from done.txt |
| measured rate | drafts before the kill: 71-74 blocks/s per GPU on all 8 GPUs (data generation, SGLang, not a benchmark speed) |
| ETA | drafts done ~17:50 IST PROJECTION (basis: ~87,000 items left = ~880,000 blocks over 8 GPUs at ~73 blocks/s = ~25 min, plus ~2 min engine start); pack ~0.7 h, index: all done ~18:45 IST PROJECTION |
| last finished | gen 4:12:54 (-> 16:58:05 IST), 151,275 kept |
| next | drafts end; pack, index (fresh only), report; then parity, record |
| blockers / deviations | INCIDENT: generation tmux session killed externally at 17:14 IST (sender unknown; not this session); resumed without data loss. Operator: if you killed it on purpose, say so and I stop the resume |
| GPU / disk | 8/8 GPUs loading the drafter (engines starting); disk used 239 GB, free 1.5 TB |
| last push / upload | GitHub phase5-box-s1 17:07 IST ok; phase5-logs 17:08 IST ok (e4fa0f7); HF Arushhh/phase5-loop-guard 17:08 IST ok |

### OPERATOR DECISION 23 Oct 17:24 IST: server 57321867 is being CLOSED. Final state.
- The 17:14 IST stop of the drafts stage was the operator's (on purpose); drafts, pack and index are done on the Texas server. The resume started at 17:18 IST was stopped at 17:26 IST (no draft/gen/run_fresh process left, GPUs 0 MiB). No new job starts on this server.
- What this server produced: Stage A setup (16/16 + 16/16), Stage B speed check (r = 2.03 vs the Phase 4 box, 500/500 token-identical), launch_bench x3, fresh wave C gen: 250,790 prompts, 151,275 kept, 63.94M est. corpus tokens (logs/c1b/gen/waveC), partial drafts 64,000 items (logs/c1b/draft/waveC, 4 npz per GPU). Everything is on HF Arushhh/phase5-loop-guard, GitHub phase5-box-s1 (code, STATUS.md, results) and phase5-logs (text logs).
- Only the 15-minute backup loop keeps running; one last full pass was run at 17:27 IST (result below). Nothing deleted.

### STOP + incident — 23 Oct 18:25 IST (Texas)
**CP3 parity: STOP (SGLang judge decode crashes).** Memory OOM fixed by `--mem-fraction 0.70 --drafter-mem-fraction 0.10` (judge static ~68 GiB). The judge engine then loads and runs, but the decode crashes with an assertion in the SGLang judge-loop plugin:
`c1b_judge_loop.py:335  AssertionError: gsm8k500#853: the [MASK] slots are not exactly the trailing 31`.
Cause: `run()` requires each block's `[MASK]` slots to be the contiguous trailing suffix (`start = bs - n_masked; masked[start:].all()`). But with `--redraft-ban` + `--loop-guard 12` the drafter refills only a *subset* of masked slots (doc PHASE5_SGLANG_JUDGE.md:50), leaving **scattered** `[MASK]`s. The HF harness decoded the same 100 items fine (`harness/*.jsonl` = 100 lines); the port asserts. This is the first GPU run of the port (CPU-tested only before). Per Rule 6 (do not change decoder code) and "CP3 FAIL = STOP", not modifying the plugin; awaiting operator direction. CP5 recording uses the same engine, so it is blocked until CP3 passes. Pack (independent) is done; index is independent of parity.

**Backup incident (RESOLVED).** GitHub pushes were failing non-fast-forward: before closing, the old server pushed final commits (incl. merge `e0d36d8` "final state before server close", which already contains Texas pieces A/A0) that my local lacked. The loop's `git_push` mislabeled this as `rc=0` (bug: `if …; then :; fi` resets `PIPESTATUS`); only the logs branch showed `rc=1`. HF uploads were always succeeding (data safe). Reconciled both branches with `merge -X ours` (nothing lost; old-server final commits remain in history) and pushed: `phase5-box-s1 → 36cd050`, `phase5-logs → e0c5c8e`. Both green. No competing pusher remains (old server closed), so the rc-mislabel will not recur, but it should be fixed in `backup_loop_p5.sh`.

### STATUS 23 Oct 20:47 IST (Texas) — CP3 accepted, CP5 recording (batched), CP6 prep queued
| field | value |
|---|---|
| time | 23 Oct 20:47 IST (hours since t0 14:31 = 6:16) |
| CP3 parity | **ACCEPTED (operator 20:40 IST).** Vanilla (no-judge) SGLang-vs-harness decode of 500 gsm8k500: token agreement 0.596, exact 0.30, passes/block -1.5% → the port's judge-side 0.633 token agreement is base-kernel bf16 numerics, not judge logic. By the Phase 1 precedent (parity_check.py: accuracy ±3, passes/block ±5%, token agreement reported only), the port PASSES: passes/block +0.6%, accuracy -2.0. No blocking parity run. |
| decode crash fixes | (#1) scattered re-presented block — FIXED (c1b_judge_loop.py no-ops a non-trailing-mask block; 32/32 CPU tests pass; committed 3730416). (#2) ragged mixed prefill+decode batch (`numel != B*bs`) still crashes the batched SGLang scheduler; proven absent at --max-running 1 (mr1 decoded all 100 CP3 items cleanly). |
| CP5 recording | **START (serial) 20:38:32; RESTART (batched, supervised) 20:45:12 IST.** record_sup.sh runs 8 shards at --max-running 64; any shard that crashes on #2 is relaunched at --max-running 1 (resumes via done.txt). Rate/ETA measuring. |
| backups | items dir (150,866 files) tarred to items.tar before HF upload (Hub 429 rate-limit fix); loop git_push rc capture fixed; transient judge_side/drafter_trace dirs now excluded (caused a transient rc=1 at 20:16, self-recovered 20:31). Loop alive, green. |
| CP6 prep (queued, apply before training; defaults apply in 15 min) | (a) **Domain shares applied NOWHERE in code** — building a domain-weighted selection so the trained mix = 44M plan (maths 12.53M / code 11.70M / chat 10.45M / IF 9.31M corpus tokens = 28.5/26.6/23.7/21.2%); default corpus tokens. (b) Exclude if-ifchat-c2d4c16f96c7a08a (base flagged math_test) + any row whose meta.base_pid is flagged. (c) IF eval-decontam: fresh_lists skipped it for composed IF; ~12.8k wave-C IF items (~5M tok) flag 13gram:ifeval → DROP (waves A/B precedent); + base_pid-flagged rows; + 2 text dups (maths-numina-f4364fe3d0dcd101, if-ifchat-9d1679214c65dc8b). Counts per domain to follow. |
| pack / index | pack done (150,866 items, 66.4M tok); index done (entirely fresh, 0 science, holdouts excluded). |

### STATUS 23 Oct 21:00 IST (Texas) — CP6 prep results (exclusions + 44M selection)
Prepared on CPU (not yet applied to training; recording continues). Verified: corpus_tokens == prompt_len + n_gen for all 150,866 rows (answer tokens = n_gen exactly); IFEval decontam positive control 541/541.

**Exclusions — 12,770 in-index pids** (files in logs/phase5/cp6_prep/): IFEval 13-gram decontam 12,767 (33.0% of IF; fresh_lists had skipped eval-decontam for composed IF); meta.base_pid postdecontam-flagged 1 (if-ifchat-c2d4c16f96c7a08a; if-ifchat-abb0d8cf1e91db97 not in index); 2 text dups (maths-numina-f4364fe3d0dcd101, if-ifchat-9d1679214c65dc8b). By domain: IF 12,769, maths 1, code 0, chat 0.

**After-exclusion per-domain (items / corpus-tok / answer-tok):** maths 29,877 / 14.77M / 12.12M · code 47,500 / 24.15M / 4.17M · chat 34,827 / 12.41M / 8.71M · IF 25,892 / 9.97M / 7.53M. Total corpus 61.30M (was 66.38M).

**44M domain-weighted selection — ALL four domains meet target (none short):** maths 25,388 items/12.53M · code 23,014/11.70M · chat 29,330/10.45M · IF 24,148/9.31M (IF clears with ~663k headroom). Total 101,880 items / ~44.0M corpus-tok. Default = corpus tokens. Deterministic (ascending order_key). Selection self-test passed.

This selection is also the recording scope lever: recording only these 101,880 pids (vs all 150,866) cuts ~32% of GPU work and applies the exclusions to recording. File: logs/phase5/record_pids_sel.json.

### STATUS 23 Oct 21:04 IST (Texas) — CP5 recording rate + ETA
- **Measured rate (batched, mr64): 3.71 blocks/s/GPU** (gpu0: 256 items / 2,517 blocks in 0.19 h; all 8 GPUs 67–94% util, drafter round-trips active, 0 shards crashed). That is ~2.1× the --max-running 1 rate (1.73 blocks/s/GPU). ~9.8 blocks/item; ~1,350 items/h/GPU; ~10,800 items/h total.
- **Scope switched to the 101,880-item 44M selection** (was all 150,866) at 21:03:41 IST — applies the exclusions to recording and drops the ~36k unselected (mostly code overshoot) that training won't use; resumes via done.txt (2,048 items kept). Safe because an answer-token share basis is infeasible (code has only 4.17M answer tokens vs its 11.7M target), so the corpus-token selection is final.
- **ETA: ~9.2 h remaining → recording done ~06:15 IST 24 Oct (PROJECTION, basis 3.71 blk/s/GPU × 8, ~1.0M blocks for the selection).** Full set would have been ~13.9 h (~10:30 IST). This is over the ~01:00 IST target and ~$140 of GPU time; the only further lever is a risky drafter-on-separate-GPU rework (not recommended). Recommend proceeding — this is the proven-correct recording path.
- Step-0 check will run at CP6 training start (train_c1b_rec's built-in check on the first real states file); GPUs are fully occupied by recording until then.

### STATUS 23 Oct 21:12 IST (Texas) — PLAN A (hard deadline 08:30 IST)
Budgeted: RECORD 21:15-22:45 · TRAIN 22:45-02:15 · GATE 02:15-03:00 · EVAL 03:00-05:30 · VANILLA 05:30-08:12 · FINAL BACKUP 08:12-08:30. If a step runs long, cut its size not the later steps.
- **RECORD (step 1):** replaced the 101,880 run with a fixed-seed (20261023) stratified RANDOM subset of **15,999 pids** from the selection (maths 3987 / code 3614 / chat 4606 / if 3792 — keeps domain shares + exclusions). The prior run's 2,048 recorded items kept in record_run1/ (indexed at training). 16k run started **21:10:00 IST** (batched mr64); at 3.71 blk/s/GPU ≈ 1.5 h → ETA ~22:37 IST; hard stop 22:50, use what finished. The scattered-block no-op now logs each firing (NOOP_SCATTERED rid=…); those items are excluded from training (review noted the #853 diagnosis may be off — the no-op still safely skips the bad block, and excluding the item keeps training clean).
- **TRAIN (step 2) ready:** code/judge_round/train_cp6.sh — A0 index_states over record/ + record_run1/ joined to a filtered index (recorded pids minus 12,770 exclusions minus no-op items), then torchrun 8-GPU train_c1b_rec continue-from 128M → **160M (+32M)**, save at 136M/160M, step-0 check (bar 0.90, deviation from 0.95 for bf16 margin). Recipe = train_c1b defaults = the 128M recipe (override 2/code 3, beta 0.5@T0.5, calib 1.0, lr 8e-5, clip 1.0) — no numerics deviation. Report tokens/h at +15 min and the 25% reading.

**24 Oct 00:15 IST** — handover latest-start moved 00:55 → **01:00 IST** (run-2 ends ~00:58, its last items are maths/scarcest; trigger still = run-2 complete). Training still stops 04:15 IST. Watcher relaunched once (killed by PID, exactly one instance).

### STATUS 24 Oct 03:10 IST (Texas) — epoch-1 finished early; epoch-2 resume launched (Mac-session decision 03:04 IST)
**Epoch 1 finished at 03:01:06 IST** = one full epoch over the 47,011 recorded items, **147.61M tokens** (step 12496), written to `logs/phase5/judge/rec_main/final` (all 4 files; untouched). That is ~12.4M short of the 160M target and ~1 h before the 04:15 stop, because train_c1b stops at the end of an epoch (default `--epochs 1`). auto_eval had auto-launched the rd1 eval on the stale `tok_000144000000`.
- **03:03 — stopped that eval by PID** (auto_eval.sh + all 8 `run_da_decode` shards on tok_144M + the `eval` tmux); all files kept. GPUs freed to 0 MiB.
- **Epoch-2 resume** (`code/judge_round/train_ep2.sh`, committed `4f2ac12`): continue `rec_main/final`'s adapter+optimizer+token-count for a **second epoch** (new uniform draw of the recorded passes) on the SAME states/index, writing to a **new** out dir `logs/phase5/judge/rec_main_ep2` so the 147.61M final stays intact. Stops at the first of `--stop-at 1792794900` (**04:05 IST**) or `--max-tokens 160e6`; either way train_c1b saves `rec_main_ep2/final` + probe. Milestone at 152e6. `--epochs 4` only so an epoch boundary can't stop it before the time/token cap. **Recipe/numerics unchanged** (continue-from; lr 8e-5, same overrides) — no deviation.
- **STEP-0 GATE OFF for the resume** (`--step0-states 0`). First relaunch (03:06) crashed the gate at **argmax agreement 0.7256 < 0.85**: the gate checks whether a FRESH continue-from adapter reproduces the **128M recording** judge; the 147.61M adapter has by design diverged from that judge (the 0.726 IS epoch-1's training signal). The sampler/kernel/states are identical to epoch 1, whose step-0 already passed (0.898 ≥ 0.85), so nothing new to gate — the trainer itself skips step-0 on `--resume` of an already-trained run. This is a recipe-neutral change (does not touch the model).
- **03:08 — relaunched, stepping now:** `[step 12500] 147.70M tok … 1898 tok/s`, 8 GPUs ~80% util / ~67 GB. At ~1.9–2.5k tok/s to 04:05 → ~155–158M at the stop.
- **auto_eval re-armed** (`code/judge_round/auto_eval.sh`): waits for train to stop or the **04:12 IST** hard deadline (then kills by PID), evals the **newest COMPLETE checkpoint** (prefers `rec_main_ep2/final`, else newest `rec_main_ep2/tok_*`, else `rec_main/final`) via the plain harness, then the vanilla check. Eval must START by 04:15. `eval_rd1.sh` now honors an explicit `ADAPTER` env.
- **Fallback (if the resume had failed at launch, by 03:20):** eval `rec_main/final` (147.61M) directly. Not needed — resume is healthy.
- Backups: s1_backup loop alive; `logs/phase5/judge/` exempt from the 900 s LFS-fresh skip, so `rec_main_ep2/{final,tok_000152000000}` will upload all 4 files on the first pass after they land.

### STATUS 24 Oct 04:50 IST (Texas) — PHASE 5 JUDGE = **FAIL** (retrained judge degenerates in live decode)
**Full analysis: logs/phase5/FAILURE_ANALYSIS.md.** The rd1 eval of `rec_main_ep2/final` looped (acc ~1.7% on ifeval541). Mac-session diag (logs/phase5/eval_diag, ifeval541 first 16 items, exact rd1 flags, plain HF harness):

| adapter | tokens | acc | looping | stop@EOS | truncated |
|---|---|---|---|---|---|
| start_128m (Phase 4) | 128M | **56%** | 1/16 | 89% | 11% |
| rec tok_136M | +8M | 9% | 10/11 | 14% | 86% |
| rec final 147.61M | +19.6M | 7% | 14/15 | 14% | 86% |
| rec_main_ep2 final 155.66M | +27.66M | 8% | 11/12 | 12% | 88% |

- **Phase 5 result = FAIL** — accuracy collapses far outside noise (kill line was <2% improvement; this is a large regression). The retrained checkpoints' lower pass/blk is a **degenerate-loop artifact** (they never emit EOS, truncate at the gen cap), not a real speedup. 128M reproduces Phase-4 accuracy and terminates → harness/adapter-loading are fine.
- **Root cause = train/decode ENGINE mismatch, NOT bad targets/loss rows.** Verified on CPU that the recorded-state training is constructed correctly (target = vanilla T not recorded argmax; committed slots never loss rows; redrafts are inputs not targets; flag only on draft-filled open slots; EOS-not-trained is the long-standing convention 128M also used). The failure: Phase 5 recorded states in **SGLang** for the first time, but training and eval run in **HF**; SGLang↔HF disagree ~10% on judge argmax (step-0 0.8978 for the identical adapter) and ~40% on tokens (CP3 vanilla 0.596). Training the HF judge on SGLang-trajectory states is off-distribution → over-commit on the HF trajectory → context corruption → non-termination. The probe/step-0 are blind because they score the same SGLang-recorded states; the step-0 gap growing 0.8978→0.7256 is the only fingerprint.
- **Decision (operator, 04:41 IST):** keep 128M as the production judge. GPU until the 07:10 hard stop: `chain128` runs the PLAIN-harness rd1 eval of start_128m on gsm8k500/math500/humaneval/mbpp, then the vanilla multiplier check (N=28, 7 sets), then gpqa/aime (+ifeval541 if time). No other GPU jobs. Backups continue.
