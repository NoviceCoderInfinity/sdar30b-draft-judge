# HANDOFF — Phase 1 wave A, 30B judge LoRA (written 2026-09-29 14:25 IST by session workspace-82)

Everything below is RUNNING DETACHED (nohup/setsid/tmux). Taking over needs NO restart and NO kill: just watch the logs.
Do not touch the GPU processes. Least invasive = read-only monitoring + the checks listed at the end.

## 0. User rules (verbatim intent; they override everything)
- Do not change the decode rule, the loss, the data mix, the models. Judge base `JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc`,
  drafter `JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05`, block 32. Unchanged: loss, LR 8e-5, WSD schedule, states-per-step.
- Write only under /workspace/dmge and /dev/shm sandboxes. Never run generated code from the repo root.
- Never print, log or commit a token (`/workspace/dmge/.env` holds GITHUB_TOKEN / HF token; load with `set -a; . ./.env; set +a`).
- Git: commit as **Arush777 <23je0154@iitism.ac.in>** (already the repo config), **no Claude trailers / attribution**, branch
  `phase1-datacorpus`, push token-safe:
  `git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' push -q origin HEAD:phase1-datacorpus`
  (verify with the same helper + `ls-remote`). HEAD now: 32b9b57.
- All times/ETAs in **IST**, measured (not guessed); answer the user first, briefly. Never leave GPUs idle. No evals except
  the approved set (section 4). Do NOT start wave B. HF = public, account **Arushhh**.
- The user is very sensitive to: idle GPUs, wrong ETAs, slow answers, anything shady. Be exact and honest.

## 1. Timeline (IST, 2026-09-29)
| when | what (automatic) |
|---|---|
| now → 14:38 | `relaunch_1345.sh` watcher alive: relaunches training if it crashes (max 3), then exits at 14:38 |
| 14:45 | training stops at the next step boundary (`logs/c1b/train/main/STOP_AT` = 1790673300), saves `main/final` (~28.3M tokens), runs the final probe (`probe.jsonl`, tag `final`), torchrun exits |
| ≤ ~14:50 | `ckptloop` pushes + verifies `main/final` on HF (marker `logs/c1b/hf_sync/ckpt/main__final.json`, its `tokens` must equal `main/final/trainer_state.json`) |
| ≥ 14:44:30 and trainer exited | `eval_after_training.sh` runs `eval_ckpt.sh checkpointB` (54 shards, 8 GPUs, no deadline cut) |
| ~18:01 | evals done (measured estimate 3 h 13 m; independent sim 3 h 09 m), `eval_summary.py` → `logs/c1b/eval/checkpointB/summary.md` |
Box: current end **15:00 IST**; the user is topping up — recommended to **19:00 IST** (absolute minimum ~18:15). If the
box is NOT extended, evals die at 15:00 (the trained weights are on HF by ~14:50, so nothing trained is lost).
`/workspace/deadline.env`: END=1790674200 (15:00), DEADLINE=1790673300 (14:45 = train stop). Update END once the top-up is known.

## 2. Running processes (do NOT kill; all independent of any Claude session)
| pid | what | log |
|---|---|---|
| tmux `train4` → torchrun 450580 (+8 ranks) | training, `logs/c1b/dl/train_resume3.sh` | `logs/c1b/train/main.log`, `logs/c1b/train/main/train.rank0.jsonl` (every 10 steps: tokens, tokens_per_s, dom_tokens, utc) |
| 443778 | `logs/c1b/dl/relaunch_1345.sh` (exits 14:38) | `logs/c1b/dl/relaunch_1345.log` |
| 443316 | `backup_hf.sh ckptloop`: every 5 min `hf_sync.py ckpt` (push every new checkpoint, verify size+sha256, prune local beyond the 5 newest only after verify) + links doc to GitHub | `logs/c1b/backup_ckptloop.log` |
| 314528 | `backup_hf.sh loop`: every 30 min ckpt; data (corpus/logs/probes/eval summaries → dataset repo); ckpt | `logs/c1b/backup_hf.log` |
| 454550 | `logs/c1b/dl/eval_after_training.sh` (waits, then evals) | `logs/c1b/eval_checkpointB.log`; jobs: `logs/c1b/eval/checkpointB/logs/*.log`, `.../gpu_queue.log` |
| 2441… | tensorboard (Vast system service) | ignore |
`hf_sync.py` takes an exclusive flock (`logs/c1b/hf_sync/hub.lock`), so the two backup loops serialize safely.
No Claude subagents are running (the old 15-min monitor subagent finished; the 5-min ckptloop replaced it).

## 3. Training state
- Run `logs/c1b/train/main`. 11.79M = checkpoint A (evaluated last night). 11.79M → 14.47M: original plan (old order).
  From 14.47M (11:53 IST): domain weights **maths 25 / code 25 / science 15 / IF 15 / chat 20** % of tokens
  (`--domain-weights`, science+IF may repeat, max 4 passes; untrained items keep their order). Expert loop = stock path
  (the grouped-GEMM path `fast_moe.py` checked exact — loss rel diff 0.0, grad cosine 1.0 — but 0.85× speed, so not used).
- 14:23 IST: step 7110, **26.51M tokens**; rate ~5.2–5.4M tokens/h (~1,400–1,530 tok/s, ~29.5 states/s); expected ~28.3M at 14:45.
- Checkpoints every 1.8M tokens (~20 min): local now `final`(23.19M, 13:45) + `roll_00005876/6185/6492/6875`; all verified on HF:
  https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32 (links doc: `docs/PHASE1_HF_ARTIFACTS.md`, README links it).
- Stop control: `echo <unix time> > logs/c1b/train/main/STOP_AT` moves the stop (re-read every step, no restart);
  `touch logs/c1b/train/main/STOP` stops cleanly now (final + probe). Resume = `bash logs/c1b/dl/train_resume3.sh` in tmux
  (picks the checkpoint with the most tokens; the weighted plan resumes at its own step; plan build takes ~2 s).
- Tokens/h: before the switch (old plan, 11:23–11:51) 11.79M→14.47M in ~28 min ≈ 5.7M/h; after (weighted) 5.42M/h
  (12:09–12:37), 5.23M/h over the 30 min containing the 13:45 restart. Restarts cost ~1 min (relaunch → first step 41 s).
- Weighted-phase per-domain tokens = `dom_tokens` in the last jsonl row (shares have held 25.0/24.9/15.0/15.0/20.0).
  STILL OWED: per-domain tokens for 11.79M→14.47M (old plan) — replay `plan_epoch` for all 8 ranks over steps 4889..5255
  on CPU (see `domain_weighted_units` in `code/corpus1b/train_c1b.py` for how plans are rebuilt) and sum `tokens_share` by domain.

## 4. Evals (user-approved 14:15 IST: "the standard eval + IFEval-541; judge, stock, blind all present")
- Queue `eval_ckpt.sh checkpointB` (54 shards, longest first): judge (+DES48 pool, same as last night) on GSM8K-500,
  MATH-500, MBPP, HumanEval, GPQA-Diamond, AIME, IFEval-541 (44); IFEval-541 blind accept 8 (`--judge blind --tau-acc-q 0.50
  --draft s8` = RESULTS.md §2 config); IFEval-541 stock 2 (shards 6,7; shards 0–5 copied from `logs/c1b/eval/checkpointA/`).
- It freezes the newest checkpoint (most tokens → `main/final` after the 14:45 stop) into `logs/c1b/train/main/eval_checkpointB`
  and writes sha256 + tokens to `logs/c1b/eval/checkpointB/checkpoint.txt`. (A dry run left a copy of roll_00006875 there;
  it is replaced at launch — check checkpoint.txt says ~28.3M tokens / main/final after 14:45.)
- Stock and blind for the 6 standard sets = RESULTS.md §1 and §2 (same model, harness, settings). Caveat for the report:
  their ms/token came from an earlier machine → compare speed by passes/blk and exp/token, not ms/token.
- Deliverable: per dataset judge vs stock vs blind (acc, passes/blk, exp/token, ms/token w/ caveat, loop/no-stop); IFEval =
  prompt-level strict. Push eval summaries to HF; update the scorecard artifact
  https://claude.ai/artifact/EnUMLBXsDhQbXv4MAMeecb (sources `/dev/shm/c1b_artifact/{build.py,extract.py}`).
- **REMIND THE USER**: ChartQA-300 + FinQA-500 were deferred (judge + blind missing; stock FinQA 2 shards missing).

## 5. Final report still owed (the user's NEW TASK list)
tokens trained (total), tokens/h before/after the switch, trained tokens per domain since 11.79M (section 3), probe per
domain (`logs/c1b/train/main/probe.jsonl`, rows tagged `final`), confirmation that the final checkpoint + logs + probe are
on HF (verify via the marker files / `hf_sync.py status`).

## 6. Pitfalls learned the hard way
- `pkill -f`/`grep` patterns can match your own shell → use `pgrep -P`, bracketed patterns, or exact-args `awk`.
- Foreground `sleep` is blocked in this harness; waiters must be background scripts.
- ptrace/py-spy is not permitted in this container.
- nvidia-smi utilization samples bounce 0–100 %: the trainer is CPU/launch-bound (~55–75 % average); 0 % on one GPU in a
  single sample is normal. Real stall = no new `[step` line in main.log for > 2 min.
- The earlier stall (11:53–12:08) was an O(N²) loop in the domain-weight planner (fixed in 0e742f5).
- `/dev/shm` is noexec. Tests that touch the trainer must run with `CUDA_VISIBLE_DEVICES=""` (never take a training GPU).
- Memory notes: `/root/.claude/projects/-workspace/memory/` (MEMORY.md index) — deadlines, HF policy, git identity, reminder.
