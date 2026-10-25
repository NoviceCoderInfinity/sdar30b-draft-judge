# Phase 4: a LoRA on the SDAR-1.7B drafter (rd1 / rd2): report

Every number here was measured on this box unless it is labelled **projection**. Times are IST (UTC+5:30); the logs
themselves are in UTC. Status of this file: **complete** (all stages done 13 Oct 06:41 IST; same-machine vanilla baseline 06:54-09:56 IST;
this version 09:58 IST).

## 1. Summary so far

| Stage | Result |
|---|---|
| Setup | env + models + data OK; decode checks 16/16 stock + 16/16 judge token-identical to Phase 1 |
| Selfcheck | 3/6 FAIL (forward, merge, parity); a diagnostic traced forward and merge to bf16 numerics (fp32: exact); operator: proceed |
| Training | one pass, 1,285 steps, 328,818 items, 133.2M tokens, 9,137 s (2.54 h); 6.65 s/step median |
| 25% check | **PASS**: held-out first-draft agreement 38.94% -> 40.78% (+1.84 pts, bar +1.00) |
| Merged drafter | `check_merged.py` 3/3 PASS; merge fidelity on 300 held-out items: adapter 42.33%, merged 42.22% (-0.11 pts) |
| Gate 1 | **FAIL**: agreement 38.94% -> 41.40% (+2.45 pts, bar +3.0, target +7); retention 89.11% (bar 97%, fail < 95%) |
| Gate 2 | **PASS**: judge commit precision 99.21% -> 98.73% (-0.48 pts, bar 0.5); commit share 30.74% -> 31.21% (rises) |
| Gate 3 | **FAIL**: judge passes/block 0.33% fewer (bar 5%, fail < 2%); pooled dAcc +1.04 [-1.21, +3.11] (ok); IFEval loops 22 vs 25 (ok) |
| Gate 4 | done, all 2,250 items per mode: success criteria NOT met (IFEval loops 104 / 107 vs <= 60; IFEval CI vs vanilla below 0; judge passes -0.6% / +2.1% fewer vs >= 5%); pooled accuracy CI vs vanilla covers 0 in both modes; per-answer speed vs Phase 3 only cross-machine |
| Speed on this machine (vanilla 30B = 1.00×) | vanilla decoded here for all 2,250 questions (token-identical to the vanilla records). Per answer: Phase 4 **1.25×** [1.23, 1.27] (rd1) and **1.26×** [1.24, 1.28] (rd2), measured; Phase 3 1.20× [1.18, 1.22] / 1.26× [1.24, 1.28] (its recorded work priced at this machine's costs); Phase 4 vs Phase 3 **1.04×** / **1.00×** (sections 9c, 9d) |

Gates are temporary checkpoints (operator, 13 Oct 00:40 IST): a failed gate opens a 5-minute decision window, then the
run continues unless the operator says stop. Gate 1 failed; no answer came in the window; the run continued.

## 2. Machine and software

* 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, driver 615.71.09, 600 W, PCIe Gen5 x16),
  2x AMD EPYC 9555 (256 threads), 755 GB RAM, vast.ai container 55572817. Full detail: `machine.txt`.
* Python 3.12.14, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900, sm_120), transformers 4.54.1, numpy 2.5.2,
  huggingface_hub 0.36.2, safetensors 0.8.0, wandb 0.29.0, nvidia-ml-py 13.615.71. Full freeze: `pip_freeze.txt`.
* Models: JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc3 (judge base, frozen), JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05d (drafter),
  128M judge LoRA `logs/phase2/ref/c1b_128M/tok_000128000000` (weights hash 35805974, frozen, never merged).
* Speed of this box vs the earlier boxes on identical work (measured): the setup judge check (16 HumanEval items)
  took 131.7 s here vs 233.9 s on the Phase 3 box (0.563x) vs 107.6 s on the Phase 1 box; 16 stock GSM8K items took
  0.538x the time of the vanilla-record box (RTX PRO 6000 Blackwell Workstation Edition).

## 3. Timeline (IST, 12-13 Oct)

| Time | Event |
|---|---|
| 20:45:41 | box up |
| 21:15:39 | `p4.sh setup` started |
| 21:29 | 30B (61.08 GB, 10m41s) and 1.7B (4.08 GB, 42 s) downloaded |
| 21:30:00-21:35:01 | decode checks, 16/16 + 16/16 (killed at the operator's 5-minute cap while exiting) |
| 21:35-21:42 | corpus (2.31 GB, 4m12s), Phase 3 + vanilla records, tests (29 + 60 passed), selfcheck |
| 21:51:02 | `p4.sh train` started (no smoke test: operator) |
| 21:57 | step-0 held-out agreement 38.94% (326 s) |
| 22:42 | 25% check PASS (+1.84 pts; 521 s replay) |
| 00:29:18 | training done; merged; `check_merged.py` PASS |
| 00:29:22-00:57:15 | Gate 1 (FAIL); 00:57:20-01:02:21 decision window, no answer: continued |
| 01:02:21-01:03:39 | Gate 2 (PASS) |
| 01:03:42 | Gate 3 started |
| 01:45:15 | Gate 3 FAIL (0.33% fewer judge passes/block); 01:45:15-01:50:15 decision window, no answer: continued |
| 01:50:15-06:41:03 | Gate 4: rd1 and rd2, all 2,250 items each; report step re-run 06:42 after fix 108c126 |
| 06:54:11-09:55:54 | vanilla 30B stock decode on this box, all 7 benchmarks, 2,250 questions (`run_vanilla_here.sh`; operator): section 9c |

## 4. Operator decisions and deviations from the plan

1. No smoke test (the artifact and the box prompt list one).
2. Setup checks only if under 5 minutes; they completed 16/16 + 16/16 at the cap; the rest of setup ran by hand
   with the same commands.
3. OOM in training: stop and ask (no OOM happened).
4. Every discrepancy is confirmed with the operator before acting.
5. Selfcheck 3/6 FAIL accepted after the diagnostic (section 5).
6. `nvidia-ml-py` installed so the decodes record GPU-active and CPU time (it was missing from `/venv/p2`).
7. Same-box baselines (Phase 3 rd1 / rd2 and vanilla re-decoded here) deferred by the operator; time-based
   comparisons against the Phase 3 and vanilla records are therefore **cross-machine ratios, not measured speedups**.
8. W&B on for the decodes too (Phase 3 ran its decodes with W&B off).
9. HF dataset `Arushhh/phase4-drafter-lora` made public; everything under `logs/phase4` uploaded every 20 min
   (`backup_p4.py` allowlist extended, `*.pt` no longer denied).
10. Logs only on the GitHub branch `phase4-logs` (separate clone); `phase4-drafter-lora` holds code, STATUS.md,
    results.
11. Dashboard artifact + two sub-agents (dashboard, logs branch).
12. Logs branch named `phase4-logs`, without the downloaded reference records.
13. Baselines deferred (see 7).
14-15. Stage chain: gates start in order automatically; a failed gate opens a 5-minute decision window, then continues.
16. Stale `.done_*` markers of the Phase 3 box (tracked in git) were moved aside before setup; `logs/phase4/train`
    created so `p4.sh train`'s tee could write `main.log`. No pipeline code changed. Added code:
    `code/drafter_lora/report_p4.py` (read-only measurement report); `backup_p4.py` allowlist change (item 9).

## 5. Setup checks

* Decode checks (same 16 items as Phase 1 / Phase 3): stock 16/16 and judge 16/16 token-identical; stock 285.9 s,
  judge 131.7 s.
* Corpus: wave A 165,438 + wave B 166,380 index rows, 0 missing npz. Phase 3 rd1 / rd2 and vanilla records: all
  2,250 items of the 7 sets each.
* Selfcheck (`logs/phase4/selfcheck.json`, 34 s): count PASS (34,865,152), KL 0 PASS, one step lowers the loss PASS
  (3.2613 -> 2.8028); forward FAIL (bf16 argmax 0.631), merge FAIL (0.6875), parity FAIL (0.9106 on 8 items).
* Diagnostic (`logs/phase4/diag_selfcheck.json`): in fp32 the two-stream forward equals the model's own forward at
  every perturbation (argmax 1.0000, max rel diff <= 7e-5); in bf16 the stock model itself reaches 0.977, so the
  bf16 bar is unreachable; selfcheck's perturbation makes the adapter 84% of the weights' norm; at a realistic adapter
  (2%) merged vs un-merged agree on 93/96 positions.

## 6. Training

Config (`logs/phase4/train/main/config.json`): LoRA r 32 / alpha 64 / dropout 0 on q, k, v, o, gate, up, down of all
28 layers (34,865,152 trainable: attention 12,845,056 + MLP 22,020,096); CE + 0.1 KL (KL only where the original's
argmax = the 30B's token); AdamW 0.9 / 0.95, eps 1e-8, wd 0, lr 8e-5, warmup 26 steps (2%), cosine to 10%, clip
1.0; 256 items per step, `--micro-tokens 16384`; 40% first-draft / 60% re-draft states; 3,000 held out (chat 676,
code 602, if 622, maths 610, science 490; numpy default_rng(0)); one pass over 328,818 items.

Speed: 6.65 s/step median (p10 6.37, p90 7.12); 38.5 items/s; 15,554 corpus tokens/s; projection at step 20:
6.75 s/step x 1,285 = 2.41 h; actual 9,137 s from step 1 to the end (includes the 25% check, 521 s).

Training curves (means over step windows, training batches; acc = trained drafter's argmax == the 30B's token on the
hidden slots, orig = the original drafter on the same slots):

| Steps | CE | KL | acc | orig | gap (pts) | first-draft gap | re-draft gap | retention | grad norm |
|---|---|---|---|---|---|---|---|---|---|
| 1-20 | 4.696 | 0.017 | 0.3612 | 0.3649 | -0.37 | -0.14 | -0.53 | 0.958 | 4.96 |
| 21-100 | 3.147 | 0.091 | 0.3582 | 0.3667 | -0.85 | +0.15 | -1.53 | 0.874 | 1.33 |
| 101-322 | 2.979 | 0.075 | 0.3720 | 0.3648 | +0.72 | +1.30 | +0.33 | 0.902 | 1.03 |
| 323-640 | 2.924 | 0.076 | 0.3785 | 0.3666 | +1.19 | +1.62 | +0.90 | 0.905 | 0.93 |
| 641-960 | 2.908 | 0.077 | 0.3799 | 0.3652 | +1.47 | +1.80 | +1.24 | 0.907 | 0.97 |
| 961-1285 | 2.903 | 0.077 | 0.3807 | 0.3642 | +1.65 | +1.93 | +1.46 | 0.908 | 0.99 |

KL at step 1: exactly 0. No NaN / inf. Memory: 26-58 GB per GPU (nvidia-smi sample at 22:35 IST).

## 7. Gates 1 and 2

Gate 1 (`gate1.json`; 3,000 held-out items, 664,946 valid slots; s8 first-draft replay from the clean 30B prefix):

| | untrained | trained | change |
|---|---|---|---|
| first-draft agreement with the 30B | 38.94% | 41.40% | +2.45 pts (FAIL: bar +3.0) |
| retention | | 89.11% | FAIL (bar 97%, fail < 95%) |
| chat (676) | 24.84% | 26.87% | +2.02 (retention 83.26%) |
| code (602) | 64.69% | 69.55% | +4.85 (92.83%) |
| if (622) | 20.94% | 23.32% | +2.39 (82.84%) |
| maths (610) | 52.86% | 54.73% | +1.87 (91.39%) |
| science (490) | 40.51% | 44.50% | +3.99 (89.20%) |
| re-draft probe agreement, no ban / ban | 21.85% / 30.54% | 21.22% / 31.35% | |
| re-draft repeats the rejected token, no ban / ban | 26.43% / 0% | 37.31% / 0% | |
| parity of the untrained HF replay with the stored SGLang drafts | 81.07% | | |

Decomposition: the trained drafter keeps 34.70% of slots right (89.11% of 38.94%), gains 6.70 pts of new right slots
and loses 4.24 pts: net +2.45.

Gate 2 (`gate2.json`; frozen judge, 23,855 draft-filled pass-1 states, 256 per domain per rank layout):

| | stored drafts | trained drafts |
|---|---|---|
| commit share (top-1 > 0.95) | 30.74% | 31.21% (rises) |
| commit precision | 99.21% | 98.73% (-0.48 pts; PASS, bar 0.5) |
| commits on rows where the draft is wrong: share / precision | 4.85% / 92.70% | 4.91% / 87.84% |
| maths: commit precision; on wrong-draft rows | 99.16%; 86.62% | 98.02%; 70.93% |

## 8. Gate 3: FAIL on judge passes per block; accuracy and loops ok

rd1 + ban (`--full-redraft-every 1 --redraft-ban-rejected`, merged drafter `..._dr3d174340`, judge 128M) on
items[k::32], k = 0..7, of the 7 sets (579 items), 01:03:42-01:45:11 IST, paired with Phase 3 rd1 on the same items.
Verdict `gate3.json`; all measured quantities `measured_rd1_gate3.json`.

| per set (paired n) | accuracy new vs P3 | d acc [95% CI] | judge passes/block fewer | loops new vs P3 | length cap new vs P3 |
|---|---|---|---|---|---|
| IFEval (136) | 50.7 vs 50.0 | +0.7 [-5.9, +6.6] | 3.3% | 22 vs 25 | 24 vs 26 |
| MATH-500 (128) | 71.9 vs 72.7 | -0.8 [-4.7, +3.1] | 1.1% | 4 vs 4 | 10 vs 13 |
| GPQA (54) | 38.9 vs 35.2 | +3.7 [-3.7, +11.1] | -3.2% | 1 vs 2 | 2 vs 6 |
| AIME (24) | 20.8 vs 20.8 | 0.0 | -1.1% | 0 vs 0 | 8 vs 8 |
| HumanEval (44) | 70.5 vs 65.9 | +4.5 [-4.5, +13.6] | 1.5% | 1 vs 1 | 1 vs 2 |
| MBPP (65) | 67.7 vs 64.6 | +3.1 [-4.6, +12.3] | 2.0% | 0 vs 0 | 0 vs 0 |
| GSM8K-500 (128) | 91.4 vs 91.4 | 0.0 [-2.3, +2.3] | 1.9% | 0 vs 0 | 0 vs 0 |
| **pooled (579)** | **65.46 vs 64.42** | **+1.04 [-1.21, +3.11]** | **0.33%** | **28 vs 32** | **45 vs 55** |

Counts on the same 579 items (machine-independent): judge passes 50,740 vs 52,714 (3.7% fewer per answer); blocks
6,711 vs 6,949 and tokens 197,544 vs 205,924 (answers 3.4% / 4.1% shorter); drafter forwards (draft + re-draft)
158,648 vs 177,737 (10.7% fewer); 30B expert loads 145.4M vs 151.6M (4.1% fewer; 736 per token in both); re-drafts
repeating the rejected token 0% vs 26.5%. The pooled passes-per-block reduction (0.33%) is smaller than most per-set
reductions because the new answers are shortest on the cheap sets (MATH, HumanEval), which shifts the block mix.
Against vanilla on the same items: accuracy 65.5 vs 66.1, dAcc -0.7 [-3.8, +2.4]; judge passes/block 7.56 vs 12.72;
experts/token 736 vs 1,161.
Times (this box; the Phase 3 records come from a 1.78x slower box, so these are cross-machine, not speedups):
27.1 s per answer here vs 59.3 s there; GPU active 2.36 h (54% of wall) vs 2.59 h; MoE 3.86 h vs 8.37 h;
drafter 0.46 h vs 1.09 h.
Decision window 01:45:15-01:50:15 IST, no operator answer: Gate 4 started 01:50:15 IST.


## 8b. Diagnosis (13 Oct 02:45-03:35 IST; three read-only agents; evidence in the session scratchpad diag/A, B, C)

Asked by the operator after Gate 3: why does a drafter trained toward the 30B not cut judge passes?

* **Decode code (agent A).** Phase 4's rd1/rd2 is Phase 3's loop plus exactly two changes, (a) the trained merged
  drafter and (b) `--redraft-ban-rejected`. With the ban off, HEAD is code-identical to Phase 3: a CPU differential
  run of 76 rd1/rd2 scenarios gave byte-identical output for both Phase 3 trees and HEAD. Rebuilding Phase 3's
  config_hash from Phase 4's config (changing only drafter, cell, adapter path, ban) reproduces it 8/8, so all 29 other
  settings match. The merged drafter differs from stock only in the 196 LoRA-target matrices. The ban comes from the
  plan artifact ("Decoder change", section 3b) and Phase 3's Table 15 proposal ("Not run"). Phase 4 is deterministic
  (136/136 IFEval items identical between Gate 3 and Gate 4). No run isolates the ban from the drafter.
* **Decode records (agent C).** No decode bug (all 1,301 records pass every accounting invariant). On blocks with
  identical context and final tokens in both phases, the trained drafter gets +5.0 pts more slots right (Gate 3),
  but the extra right slots become tentative (judge agrees, confidence <= 0.95), not commits: pass-1 commits +0.2
  pts, and 70-82% of these blocks take the same number of judge passes. Blocks with > 10 passes hold 58% of all
  passes (87% on IFEval) and stay mostly wrong. The ban removes repeats (26.5% -> 0%) and makes re-drafts accepted
  more (rejected 41% vs 58% on all-re-draft passes), giving 13% fewer re-draft events and 8% fewer drafter forwards,
  but not fewer judge passes. Accuracy flips are below the Phase 3 noise floor.
* **Training vs inference (agent B).** No bug in the loss; the gain transfers (+1.5 to +8.8 pts first-draft agreement
  on non-final blocks per set). Over 907 identical blocks, wrong first-draft slots 13.62 -> 12.78 per block while
  passes 6.546 -> 6.537 (-0.13%, CI [-0.95, +0.64]); even a perfect first draft gets 26.5/32 slots committed at
  pass 1 and needs ~1.94 passes. Weak points of the recipe: the KL term was ~0.3% of the loss (retention ~0.90), and
  post-EOS slots were never loss rows, so the trained drafter no longer drafts the padding after <|im_end|>
  (85.3% -> 1.4% on held-out post-EOS slots; no pass cost).
* **Conclusion.** The judge's strict 0.95 commit rule, with the slowest slot setting the pass count, caps what a
  better drafter can save; the drafter improvement shows up as fewer drafter forwards (8-13%), not fewer judge passes.
  Suggested tests (need GPUs): Gate 2's judge probe with oracle drafts (D = T); a decode with an eased commit rule.
* **Reporting fix (93fd175).** The flat vanilla bundle holds a stray `gsm8k_B32_t95_stock.aime` cell that the glob
  `*_B32_t95_stock.{set}` mixed into AIME (idx 0, 1, 4, 5, 8, 12). `compare_gates.py` / `report_p4.py` now read only
  each set's own cell; tests 29 passed; measured files recomputed.

## 9. Gate 4: done (01:50:15-06:41:03 IST, all 2,250 items per mode, rd1 and rd2 + ban)

Final table (P3 = Phase 3 untrained drafter, no ban; P4 = Phase 4 trained drafter + ban; delta = P4 vs P3). Speed vs
vanilla is time per answer on this machine, vanilla 30B = 1.00× (section 9c). For **P4** it is Σ seconds vanilla / Σ
seconds P4 over the same questions, both decoded here [95% paired bootstrap]. For **P3** (decoded on the Phase 3
machine) it is Σ seconds vanilla / Σ P3 seconds on this machine, where P3's recorded per-forward work is priced at this
machine's measured per-forward costs (`phase3_same_box.json`; the pricing reproduces the measured P4 times within ±1.0%
and the vanilla times within -0.7% to +2.5% per benchmark):

| Benchmark (questions) | Mode | Drafter fwd/block: P3 | P4 | Δ | Judge passes/block: P3 | P4 | Δ | Accuracy %: vanilla | P3 | P4 | P4 − P3 (95% range) | Speed vs vanilla on this machine: P3 [95%] | P4 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IFEval (541) | rd1 | 36.7 | 32.2 | **-12.1%** | 11.31 | 11.31 | **+0.0%** | 52.7 | 46.6 | 46.2 | **-0.4** (-3.3 to +2.8) | 1.06× [1.02, 1.10] | **1.10×** [1.06, 1.14] |
| IFEval (541) | rd2 | 23.7 | 21.2 | **-10.2%** | 12.23 | 11.53 | **-5.7%** | 52.7 | 48.1 | 46.0 | **-2.0** (-5.5 to +1.3) | 1.13× [1.09, 1.17] | **1.09×** [1.04, 1.13] |
| MATH-500 (500) | rd1 | 18.7 | 17.9 | **-4.3%** | 5.24 | 5.23 | **-0.1%** | 72.4 | 70.8 | 71.6 | **+0.8** (-1.8 to +3.2) | 1.21× [1.18, 1.23] | **1.28×** [1.25, 1.30] |
| MATH-500 (500) | rd2 | 13.4 | 12.9 | **-4.0%** | 5.46 | 5.28 | **-3.2%** | 72.4 | 70.2 | 70.6 | **+0.4** (-2.8 to +3.4) | 1.26× [1.24, 1.29] | **1.31×** [1.28, 1.33] |
| GPQA (198) | rd1 | 31.3 | 29.4 | **-6.1%** | 9.21 | 9.49 | **+3.0%** | 37.4 | 37.4 | 36.9 | **-0.5** (-6.6 to +5.1) | 1.37× [1.30, 1.43] | **1.39×** [1.33, 1.44] |
| GPQA (198) | rd2 | 20.1 | 19.1 | **-5.1%** | 9.56 | 9.61 | **+0.6%** | 37.4 | 37.9 | 36.9 | **-1.0** (-7.1 to +5.1) | 1.43× [1.37, 1.49] | **1.44×** [1.38, 1.50] |
| AIME (90) | rd1 | 22.1 | 20.5 | **-7.2%** | 6.38 | 6.20 | **-2.8%** | 6.7 | 5.6 | 7.8 | **+2.2** (+0.0 to +5.6) | 1.37× [1.31, 1.43] | **1.43×** [1.36, 1.51] |
| AIME (90) | rd2 | 15.1 | 14.6 | **-3.7%** | 6.53 | 6.43 | **-1.6%** | 6.7 | 5.6 | 7.8 | **+2.2** (-3.3 to +7.8) | 1.42× [1.35, 1.50] | **1.45×** [1.38, 1.52] |
| HumanEval (164) | rd1 | 17.0 | 16.0 | **-6.2%** | 4.58 | 4.67 | **+1.8%** | 78.0 | 75.6 | 76.8 | **+1.2** (-3.7 to +6.1) | 1.08× [1.00, 1.16] | **1.26×** [1.15, 1.37] |
| HumanEval (164) | rd2 | 11.4 | 11.7 | **+2.7%** | 4.65 | 4.63 | **-0.5%** | 78.0 | 79.9 | 75.0 | **-4.9** (-9.8 to +0.0) | 1.32× [1.21, 1.43] | **1.26×** [1.17, 1.36] |
| MBPP (257) | rd1 | 25.6 | 24.4 | **-4.5%** | 7.17 | 7.12 | **-0.7%** | 65.4 | 71.2 | 70.4 | **-0.8** (-5.4 to +3.9) | 1.21× [1.17, 1.26] | **1.22×** [1.18, 1.27] |
| MBPP (257) | rd2 | 15.1 | 15.6 | **+3.4%** | 7.30 | 7.31 | **+0.1%** | 65.4 | 69.3 | 70.4 | **+1.2** (-3.5 to +6.2) | 1.29× [1.25, 1.34] | **1.29×** [1.24, 1.34] |
| GSM8K (500) | rd1 | 19.7 | 18.7 | **-5.0%** | 5.93 | 5.77 | **-2.7%** | 90.2 | 91.6 | 91.4 | **-0.2** (-1.8 to +1.4) | 1.28× [1.25, 1.30] | **1.31×** [1.28, 1.34] |
| GSM8K (500) | rd2 | 13.6 | 13.2 | **-3.3%** | 6.03 | 5.90 | **-2.2%** | 90.2 | 92.0 | 91.6 | **-0.4** (-1.6 to +0.8) | 1.30× [1.27, 1.32] | **1.34×** [1.32, 1.37] |
| **All 7 pooled (2,250)** | **rd1** | 25.8 | 23.9 | **-7.6%** | 7.64 | 7.68 | **+0.6%** | 65.5 | 64.4 | 64.5 | **+0.1** (-1.2 to +1.4) | 1.20× [1.18, 1.22] | **1.25×** [1.23, 1.27] |
| **All 7 pooled (2,250)** | **rd2** | 17.2 | 16.3 | **-5.3%** | 7.99 | 7.83 | **-2.1%** | 65.5 | 64.9 | 64.2 | **-0.7** (-2.1 to +0.7) | 1.26× [1.24, 1.28] | **1.26×** [1.24, 1.28] |

Gate 4 success criteria (`gate4_rd1.json`, `gate4_rd2.json`, compare_gates.py):

| criterion | rd1 | rd2 |
|---|---|---|
| IFEval loops <= 60 | 104: NOT met | 107: NOT met |
| IFEval accuracy CI vs vanilla covers 0 or better | -6.5 [-10.4, -2.4]: NOT met | -6.7 [-10.5, -2.6]: NOT met |
| pooled accuracy CI vs vanilla covers 0 or better | -1.0 [-2.6, +0.5]: met | -1.3 [-2.9, +0.2]: met |
| >= 5% fewer judge passes per block than Phase 3 | -0.6% (more passes): NOT met | 2.1%: NOT met |
| faster per answer than Phase 3, tok/s not lower | met: 1.04× [1.03, 1.05] per answer on this machine (section 9d) | not met: 1.00× [0.99, 1.02] |

Tables against both baselines (Phase 3 same mode | vanilla) and everything against vanilla: `baseline_tables.md`.
Pooled, on this machine (vanilla = 1.00×): rd1 P3 1.20×, P4 1.25×, P4 vs P3 1.04×; rd2 P3 1.26×, P4 1.26×, P4 vs P3
1.00× (sections 9c, 9d).
Report fix (108c126): `p4.sh`'s `${VAN_TPL:-...{set}...}` ended at the '}' of {set}, so Gate 4's compare_gates got
`{set.shard*.jsonl}` and crashed after all decodes had finished; the default now sits in its own variable and only
the report step was re-run (06:42 IST; every decode cell was already complete).

## 9c. Speed vs the vanilla 30B, measured on this box (13 Oct 06:54-09:56 IST)

The operator asked for a measured same-machine baseline for every speed figure.
`code/drafter_lora/run_vanilla_here.sh` decoded the vanilla 30B on this box: all 7 benchmarks, every question, 8 shards
(one per GPU), with W&B on as in the Phase 4 decodes. The decode is `run_da_decode.py --stock`: no drafter, no judge
adapter, block 32, threshold 0.95.

**All 2,250 answers are token-identical to the earlier vanilla records.** `ref_match` checked tokens, passes per block,
commits per pass and forward counts. So vanilla's accuracy, passes and expert counts are unchanged, and only its timing
is new. Phase 4's rd1 / rd2 records are the Gate 4 decodes, made on this box. Phase 3 on this machine: section 9d.

**Definitions** (`code/drafter_lora/vanilla_same_box.py` -> `vanilla_same_box.json/.md`). Over a set I of questions
(one benchmark, or all 2,250 pooled), with V = vanilla and P = Phase 4:

* wall-clock speed-up: S_w(I) = Σ_I seconds_V / Σ_I seconds_P. `seconds` is synchronize + perf_counter around the whole decode.
* GPU-time speed-up: S_g(I) = Σ_I gpu_active_s_V / Σ_I gpu_active_s_P (NVML GPM).
* tokens/s speed-up: S_t(I) = (Σ tokens_P / Σ seconds_P) / (Σ tokens_V / Σ seconds_V).
* 95% ranges: paired bootstrap over the questions of I (2,000 resamples, numpy default_rng(0)).
* exact decomposition: S_w = (N_V/N_P) x [(F/N)_V/(F/N)_P] x [(M/F)_V/(M/F)_P] x [(M/W)_P/(M/W)_V].
  N = generated tokens, F = 30B forwards, M = 30B forward seconds, W = wall-clock seconds, each summed over I.

| Benchmark (questions) | Mode | Speed-up, wall-clock [95%] | GPU time [95%] | Tokens/s [95%] | Accuracy %: vanilla → P4 (Δ [95%]) | Tokens/answer: vanilla → P4 |
|---|---|---|---|---|---|---|
| IFEval (541) | rd1 | 1.10× [1.06, 1.14] | 1.07× [1.03, 1.11] | 1.51× [1.41, 1.61] | 52.7 → 46.2 (-6.5 [-10.4, -2.4]) | 298 → 407 |
| IFEval (541) | rd2 | 1.09× [1.04, 1.13] | 1.07× [1.03, 1.12] | 1.52× [1.44, 1.62] | 52.7 → 46.0 (-6.7 [-10.5, -2.6]) | 298 → 417 |
| MATH-500 (500) | rd1 | 1.28× [1.25, 1.30] | 1.24× [1.22, 1.27] | 1.28× [1.25, 1.30] | 72.4 → 71.6 (-0.8 [-3.8, +2.2]) | 518 → 519 |
| MATH-500 (500) | rd2 | 1.31× [1.28, 1.33] | 1.27× [1.25, 1.30] | 1.33× [1.30, 1.35] | 72.4 → 70.6 (-1.8 [-5.0, +1.4]) | 518 → 525 |
| GPQA (198) | rd1 | 1.39× [1.33, 1.44] | 1.36× [1.30, 1.41] | 1.34× [1.29, 1.39] | 37.4 → 36.9 (-0.5 [-7.6, +6.6]) | 556 → 537 |
| GPQA (198) | rd2 | 1.44× [1.38, 1.50] | 1.42× [1.37, 1.48] | 1.39× [1.33, 1.44] | 37.4 → 36.9 (-0.5 [-8.1, +6.6]) | 556 → 534 |
| AIME (90) | rd1 | 1.43× [1.36, 1.51] | 1.38× [1.31, 1.46] | 1.40× [1.33, 1.47] | 6.7 → 7.8 (+1.1 [-2.2, +4.4]) | 820 → 802 |
| AIME (90) | rd2 | 1.45× [1.38, 1.52] | 1.42× [1.35, 1.49] | 1.38× [1.32, 1.45] | 6.7 → 7.8 (+1.1 [-3.3, +6.7]) | 820 → 783 |
| HumanEval (164) | rd1 | 1.26× [1.15, 1.37] | 1.23× [1.13, 1.35] | 1.35× [1.27, 1.45] | 78.0 → 76.8 (-1.2 [-7.9, +5.5]) | 99 → 107 |
| HumanEval (164) | rd2 | 1.26× [1.17, 1.36] | 1.25× [1.17, 1.36] | 1.38× [1.29, 1.48] | 78.0 → 75.0 (-3.0 [-9.1, +3.0]) | 99 → 109 |
| MBPP (257) | rd1 | 1.22× [1.18, 1.27] | 1.22× [1.18, 1.27] | 1.30× [1.25, 1.36] | 65.4 → 70.4 (+5.1 [-0.8, +11.3]) | 51 → 54 |
| MBPP (257) | rd2 | 1.29× [1.24, 1.34] | 1.29× [1.24, 1.34] | 1.35× [1.30, 1.41] | 65.4 → 70.4 (+5.1 [-0.8, +10.9]) | 51 → 53 |
| GSM8K (500) | rd1 | 1.31× [1.28, 1.34] | 1.30× [1.27, 1.32] | 1.34× [1.32, 1.36] | 90.2 → 91.4 (+1.2 [-0.6, +3.0]) | 168 → 171 |
| GSM8K (500) | rd2 | 1.34× [1.32, 1.37] | 1.33× [1.31, 1.36] | 1.36× [1.34, 1.38] | 90.2 → 91.6 (+1.4 [-0.4, +3.2]) | 168 → 170 |
| **All 7 pooled (2,250)** | rd1 | **1.25× [1.23, 1.27]** | 1.22× [1.20, 1.24] | 1.35× [1.32, 1.37] | 65.5 → 64.5 (-1.0 [-2.6, +0.5]) | 319 → 345 |
| **All 7 pooled (2,250)** | rd2 | **1.26× [1.24, 1.28]** | 1.24× [1.22, 1.26] | 1.37× [1.35, 1.40] | 65.5 → 64.2 (-1.3 [-2.9, +0.2]) | 319 → 347 |

The decomposition below is exact: for every row the product equals S_w.

| Benchmark (questions) | Mode | Answer length N_V/N_P | 30B forwards per token | ms per 30B forward: vanilla → P4 (ratio) | Time outside 30B forwards | Product = S_w |
|---|---|---|---|---|---|---|
| IFEval (541) | rd1 | 0.731 | 1.794 | 200.6 → 211.0 (0.951) | 0.883 | **1.102** |
| IFEval (541) | rd2 | 0.715 | 1.769 | 200.6 → 214.6 (0.935) | 0.919 | **1.087** |
| MATH-500 (500) | rd1 | 0.998 | 1.525 | 244.9 → 261.0 (0.938) | 0.893 | **1.275** |
| MATH-500 (500) | rd2 | 0.987 | 1.514 | 244.9 → 257.1 (0.953) | 0.919 | **1.308** |
| GPQA (198) | rd1 | 1.035 | 1.620 | 223.1 → 239.8 (0.930) | 0.888 | **1.385** |
| GPQA (198) | rd2 | 1.041 | 1.603 | 223.1 → 238.4 (0.936) | 0.923 | **1.442** |
| AIME (90) | rd1 | 1.022 | 1.629 | 239.3 → 248.1 (0.965) | 0.890 | **1.430** |
| AIME (90) | rd2 | 1.048 | 1.576 | 239.3 → 251.6 (0.951) | 0.920 | **1.446** |
| HumanEval (164) | rd1 | 0.929 | 1.652 | 232.3 → 253.3 (0.917) | 0.893 | **1.256** |
| HumanEval (164) | rd2 | 0.910 | 1.666 | 232.3 → 257.1 (0.904) | 0.916 | **1.255** |
| MBPP (257) | rd1 | 0.937 | 1.612 | 222.8 → 243.8 (0.914) | 0.885 | **1.222** |
| MBPP (257) | rd2 | 0.955 | 1.584 | 222.8 → 240.8 (0.925) | 0.922 | **1.290** |
| GSM8K (500) | rd1 | 0.981 | 1.623 | 248.9 → 272.2 (0.915) | 0.901 | **1.313** |
| GSM8K (500) | rd2 | 0.987 | 1.593 | 248.9 → 270.4 (0.921) | 0.928 | **1.343** |
| **All 7 pooled (2,250)** | rd1 | 0.925 | 1.601 | 224.6 → 237.4 (0.946) | 0.889 | **1.246** |
| **All 7 pooled (2,250)** | rd2 | 0.919 | 1.577 | 224.6 → 237.6 (0.946) | 0.921 | **1.262** |

Per answer, pooled over all 2,250 questions, vanilla → Phase 4 (per benchmark: `vanilla_same_box.md`):

| Mode | Wall s | GPU-active s | CPU s (decode thread) | 30B forward s | Drafter s | Experts/token | Judge passes/block | Looping answers |
|---|---|---|---|---|---|---|---|---|
| rd1 | 34.2 → 27.5 | 18.1 → 14.9 (52.8% → 54.1% of wall) | 34.2 → 27.5 | 34.1 → 24.3 | 2.9 (10.5% of wall) | 1167 → 741 | 12.79 → 7.68 | 58 → 123 |
| rd2 | 34.2 → 27.1 | 18.1 → 14.5 (52.8% → 53.6% of wall) | 34.2 → 27.1 | 34.1 → 24.9 | 2.0 (7.5% of wall) | 1167 → 754 | 12.79 → 7.83 | 58 → 127 |

**Judge LoRA cost per 30B forward** (`code/drafter_lora/judge_lora_cost.py` -> `judge_lora_cost.json/.md`).

* **What is compared.** The 128M judge LoRA is flag-gated: on a judge pass with no drafted slot, every q/k/v/o
  projection returns the frozen base output (`adapter.py`). That happens on 3.8-5.5% of rd2's judge passes, and never in
  rd1. Within rd2 the two kinds of pass are compared at matched experts per layer (bins 1 wide, at least 30 forwards of
  each kind per bin).
* **Result.** A judge pass that runs the LoRA costs +8.4 to +9.2 ms more. Per benchmark:
  IFEval +8.9; MATH-500 +9.0; GPQA +9.2; AIME +8.6; GSM8K +8.4. On IFEval the gap is +6.6 to +11.2 ms in every one of its 27 bins.
* **Fit check.** The least-squares fit gives +8.6 to +10.3 ms on all 7 benchmarks.
* **Without the LoRA vs vanilla.** Matched the same way, a judge pass without the LoRA costs -2.3 to
  +5.3 ms more than a vanilla forward, depending on the benchmark. These are different runs, and offsets
  of a few ms exist between runs, so no further cost is resolved.

**Findings.**

1. **Pooled over all 2,250 questions, Phase 4 is 1.25× (rd1) and 1.26× (rd2) faster per
   answer than the vanilla 30B, measured on the same machine.**
   * GPU time: 1.22× / 1.24×.
   * Tokens/s: 1.35× / 1.37×.
   * Per benchmark: 1.09× (IFEval rd2) to 1.45× (AIME rd2).
2. **Where the speed comes from** (pooled rd1, factors of the decomposition):
   * **Gain:** Phase 4 needs 38% fewer 30B forwards per token (0.297 vs 0.476; factor 1.601). Three losses offset part of it:
   * **Longer answers** (factor 0.925): 345 vs 319 tokens per answer, mostly IFEval,
     where Phase 4 loops on 104 answers against vanilla's 44.
   * **Dearer 30B forwards** (factor 0.946): 237.4 vs 224.6 ms; the judge LoRA adds about 9 ms to each judge forward.
   * **Drafter and host time** (factor 0.889): the drafter alone is 10.5% of Phase 4's wall-clock.
3. **The speed-up tracks answer length.**
   * IFEval: 1.10× / 1.09×, where Phase 4's answers are 37-40% longer.
   * MATH-500, GPQA, AIME and GSM8K: 1.28-1.45×, with answers of the same length or shorter.
   * On the short code answers (51-109 tokens) the fixed costs per forward weigh more: HumanEval 1.26×, MBPP 1.22× / 1.29×.
4. **Every speed in this report is now on this machine.** The earlier figure that chained three ratios measured on
   different machines (1.33× / 1.34× pooled) was 0.7% (MATH-500 rd2) to 12.7% (MBPP rd1) too high per benchmark.
5. **The decode is limited by the CPU and kernel launches, not by GPU compute.**
   * The GPU is active 52.8% (vanilla) and 54.1% / 53.6% (Phase 4) of wall-clock.
   * Decode-thread CPU time equals wall-clock in both arms. It includes CUDA spin-waits.
6. **Accuracy vs vanilla is unchanged from Gate 4,** since the tokens are identical.
   * Pooled: -1.0 [-2.6, +0.5] / -1.3 [-2.9, +0.2] points.
   * Only IFEval is clearly below vanilla: −6.5 / −6.7 points.

## 9d. Phase 3 on this machine, and Phase 4 vs Phase 3

Phase 3 was decoded on the Phase 3 machine. Its work is machine-independent and recorded per forward (decoding is
deterministic: the vanilla re-decode here matched the earlier records on 2,250/2,250 questions). `code/drafter_lora/
phase3_same_box.py` prices that recorded work at this machine's measured per-forward costs, fitted per benchmark on
every 30B forward of Phase 4 rd1 + rd2 here: judge forward ms = a + d x [judge LoRA runs] + b x context + c x experts
per layer; KV-flush ms = a' + b' x context + c' x experts; prefill = Phase 4's measured prefill of the same question;
drafter = Phase 4's measured ms per drafter forward of each kind x Phase 3's drafter forward counts; host time =
Phase 4's measured host time per 30B forward. Check: the same pricing reproduces the measured Phase 4 times within
±1.0% and the measured vanilla times within -0.7% to +2.5% on every benchmark (last column). Ranges: paired bootstrap
over questions (2,000, default_rng(0)).

| Benchmark (questions) | Mode | P3 vs vanilla | P4 vs vanilla (measured) | P4 vs P3 | Seconds/answer: vanilla / P3 / P4 | Pricing check: P4 / vanilla |
|---|---|---|---|---|---|---|
| IFEval (541) | rd1 | 1.06× [1.02, 1.10] | 1.10× [1.06, 1.14] | **1.04× [1.01, 1.07]** | 44.4 / 41.8 / 40.3 | +0.7% / +1.0% |
| IFEval (541) | rd2 | 1.13× [1.09, 1.17] | 1.09× [1.04, 1.13] | **0.97× [0.94, 0.99]** | 44.4 / 39.4 / 40.8 | -0.7% / +1.0% |
| MATH-500 (500) | rd1 | 1.21× [1.18, 1.23] | 1.28× [1.25, 1.30] | **1.06× [1.03, 1.08]** | 40.4 / 33.4 / 31.7 | -0.7% / -0.7% |
| MATH-500 (500) | rd2 | 1.26× [1.24, 1.29] | 1.31× [1.28, 1.33] | **1.03× [1.01, 1.06]** | 40.4 / 31.9 / 30.9 | +0.7% / -0.7% |
| GPQA (198) | rd1 | 1.37× [1.30, 1.43] | 1.39× [1.33, 1.44] | **1.01× [0.98, 1.05]** | 70.2 / 51.5 / 50.7 | -0.2% / +0.6% |
| GPQA (198) | rd2 | 1.43× [1.37, 1.49] | 1.44× [1.38, 1.50] | **1.01× [0.97, 1.05]** | 70.2 / 49.2 / 48.7 | +0.2% / +0.6% |
| AIME (90) | rd1 | 1.37× [1.31, 1.43] | 1.43× [1.36, 1.51] | **1.05× [1.00, 1.09]** | 75.0 / 54.8 / 52.4 | +0.3% / -0.7% |
| AIME (90) | rd2 | 1.42× [1.35, 1.50] | 1.45× [1.38, 1.52] | **1.02× [0.97, 1.07]** | 75.0 / 52.8 / 51.8 | -0.3% / -0.7% |
| HumanEval (164) | rd1 | 1.08× [1.00, 1.16] | 1.26× [1.15, 1.37] | **1.17× [1.07, 1.27]** | 9.0 / 8.4 / 7.2 | +0.7% / +2.5% |
| HumanEval (164) | rd2 | 1.32× [1.21, 1.43] | 1.26× [1.17, 1.36] | **0.95× [0.88, 1.04]** | 9.0 / 6.8 / 7.2 | -0.7% / +2.5% |
| MBPP (257) | rd1 | 1.21× [1.17, 1.26] | 1.22× [1.18, 1.27] | **1.01× [0.98, 1.03]** | 7.8 / 6.4 / 6.4 | -1.0% / +2.2% |
| MBPP (257) | rd2 | 1.29× [1.25, 1.34] | 1.29× [1.24, 1.34] | **1.00× [0.97, 1.03]** | 7.8 / 6.0 / 6.0 | +1.0% / +2.2% |
| GSM8K (500) | rd1 | 1.28× [1.25, 1.30] | 1.31× [1.28, 1.34] | **1.03× [1.01, 1.05]** | 17.4 / 13.6 / 13.3 | -0.5% / +2.0% |
| GSM8K (500) | rd2 | 1.30× [1.27, 1.32] | 1.34× [1.32, 1.37] | **1.03× [1.02, 1.05]** | 17.4 / 13.4 / 13.0 | +0.5% / +2.0% |
| **All 7 pooled (2,250)** | rd1 | 1.20× [1.18, 1.22] | 1.25× [1.23, 1.27] | **1.04× [1.03, 1.05]** | 34.2 / 28.6 / 27.5 | -0.0% / +0.5% |
| **All 7 pooled (2,250)** | rd2 | 1.26× [1.24, 1.28] | 1.26× [1.24, 1.28] | **1.00× [0.99, 1.02]** | 34.2 / 27.2 / 27.1 | +0.0% / +0.5% |

Pooled, on this machine with the vanilla 30B as the base: Phase 3 1.20× / 1.26×, Phase 4 1.25× / 1.26×, so the trained
drafter + ban made rd1 1.04× faster than Phase 3 and rd2 1.00× (no change). Phase 3's earlier headline (1.37× / 1.35×)
was tokens per second, which counts its longer, looping answers as speed.

## 10. How each quantity is measured (per item, in the decode records)

* accuracy: `correct`, scored inline by the vendored OpenCompass checkers / the IFEval scorer.
* time per answer: `seconds` = torch.cuda.synchronize + perf_counter around the whole decode of the item.
* MoE time: `moe_seconds` = every 30B forward timed with synchronize + perf_counter; per kind in `ms_by_kind`.
* drafter time: `drafter_seconds`, per kind in `drafter_ms_by_kind`, same timer.
* GPU time: `gpu_active_s` = NVML GPM graphics-active share x the item's interval (hardware counters).
* CPU time: `cpu_thread_s` (decoding thread), `cpu_process_s` (process), OS clocks; includes CUDA spin-waits.
* experts activated: `unique_expert_layer_loads` = distinct (layer, expert) pairs dispatched per forward, summed.
* judge passes / block = sum(n_forward - n_prefill - n_flush) / sum(n_blocks); tok/s = sum(tokens) / sum(seconds).
* Paired delta accuracy: bootstrap over items, 2,000 resamples, numpy default_rng(0).

## 11. Commands

```bash
bash code/drafter_lora/p4.sh setup     # 21:15 IST; checks killed at the 5-min cap, rest by hand (same commands)
bash code/drafter_lora/p4.sh train     # 21:51 IST
bash code/drafter_lora/p4.sh gate1     # by the stage chain, 00:29 IST
bash code/drafter_lora/p4.sh gate2     # 01:02 IST
bash code/drafter_lora/p4.sh gate3     # 01:03 IST
/venv/p2/bin/python code/drafter_lora/report_p4.py --arm rd1_gate3 --new logs/phase4/decode/rd1_gate3 \
  --phase3 logs/phase4/ref/phase3/logs/phase3_8gpu/decode/rd1 \
  --vanilla 'logs/phase4/ref/phase2/assets/ref_stock_30B/*_B32_t95_stock.{set}.shard*.jsonl' \
  --out results/phase4_drafter/measured_rd1_gate3.json
bash code/drafter_lora/run_vanilla_here.sh   # 06:54-09:56 IST: vanilla 30B on this box, all 7 benchmarks (section 9c)
/venv/p2/bin/python code/drafter_lora/vanilla_same_box.py --out results/phase4_drafter/vanilla_same_box.json
/venv/p2/bin/python code/drafter_lora/judge_lora_cost.py --out results/phase4_drafter/judge_lora_cost.json
/venv/p2/bin/python code/drafter_lora/phase3_same_box.py --out results/phase4_drafter/phase3_same_box.json
/venv/p2/bin/python code/drafter_lora/baseline_tables.py --out results/phase4_drafter/baseline_tables.json
```

## 12. Where everything is

* Everything in one command: `bash code/drafter_lora/fetch_phase4_all.sh` (see `PHASE4_BACKUPS.md` at the repo root)
* Dashboard: https://claude.ai/artifact/863c7J9dPKnaVRuJpVs9VP
* HF (public): https://huggingface.co/datasets/Arushhh/phase4-drafter-lora (checkpoints, merged drafter, all logs)
* GitHub: `phase4-drafter-lora` (code, STATUS.md, results), `phase4-logs` (all logs)
* W&B: https://wandb.ai/arushsharma536-spe-iit/Conscious_Engines/runs/phase4_main (+ one run per decode shard)
