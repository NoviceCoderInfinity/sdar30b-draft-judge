# HANDOVER to "Phase1: pt 2" (written 2026-09-29 14:26 IST by session workspace-82, CPU only)

Nothing was stopped, killed, restarted or edited for this handover. Longer notes (rules, numbers, pitfalls):
`logs/c1b/HANDOFF.md`. Every job below runs DETACHED (tmux `train4`, or setsid/nohup and reparented to PID 1), so ending
the workspace-82 session kills NOTHING.

## 1. Live processes I started
| PID | what | log | ends |
|---|---|---|---|
| tmux `train4`: pane 450577 `bash logs/c1b/dl/train_resume3.sh` → torchrun 450580 + 8 ranks | training `logs/c1b/train/main` (domain weights maths25/code25/sci15/if15/chat20, LR 8e-5 WSD, 128 states/step, ckpt every 1.8M tokens) | `logs/c1b/train/main.log`, `logs/c1b/train/main/train.rank0.jsonl` (every 10 steps: tokens, tokens_per_s, dom_tokens, utc) | first step boundary after **14:45:00 IST** (`logs/c1b/train/main/STOP_AT` = 1790673300, re-read every step) → saves `main/final` (~28.3M tokens), final probe (`probe.jsonl`), exits; script then appends `[train_resume3] exit …` to main.log |
| 443778 `bash logs/c1b/dl/relaunch_1345.sh` (wrapper parent 443773, ppid 1) | if the torchrun is gone, relaunches `train_resume3.sh` in tmux `train4` (kills `train3`/`train4` first), max 3 times | `logs/c1b/dl/relaunch_1345.log` | exits by itself at **14:38 IST** (1790672880) — never relaunches after that |
| 443316 `bash code/corpus1b/backup_hf.sh ckptloop` (ppid 1) | every 5 min: `hf_sync.py ckpt` = push every complete checkpoint (newest first) to https://huggingface.co/Arushhh/c1b-judge-lora-sdar30b-b32, verify size+sha256, write marker `logs/c1b/hf_sync/ckpt/main__<ckpt>.json`, prune local ckpts beyond the 5 newest ONLY after a fresh verify; then commits+pushes `docs/PHASE1_HF_ARTIFACTS.md` to GitHub (as Arush777) | `logs/c1b/backup_ckptloop.log` | never (loop); keep it until the final checkpoint's marker shows ~28.3M tokens |
| 314528 `bash code/corpus1b/backup_hf.sh loop` (ppid 1) | every 30 min: ckpt; `hf_sync.py data` (corpus, trainer logs/probes/configs, eval summaries → dataset repo, 25-min cap); ckpt | `logs/c1b/backup_hf.log` | never (loop) |
| 454550 `bash logs/c1b/dl/eval_after_training.sh` (wrapper parent 454546, ppid 1) | the eval launcher (section 3) | `logs/c1b/eval_checkpointB.log` | after the eval queue drains + summary (~18:01 IST est.) |
The two `/bin/bash -c source …/shell-snapshots/…` wrappers (443773, 454546) are leftover Bash-tool shells waiting on
their child; they exit with it. Leave them alone (killing them is unnecessary).
Not mine: tensorboard (PID 2441 tree, Vast service). tmux: `0` = workspace-82 (claude), `25` = you, `train4` = training —
never `tmux kill-server` (it would kill training).
`hf_sync.py` takes an exclusive flock (`logs/c1b/hf_sync/hub.lock`), so both backup loops serialize safely.

## 2. In-session tasks I own
None alive: no subagents (the old 15-min monitor subagent finished long ago), no Monitor, no cron (CronList empty), no
run_in_background Bash task (the one at 12:09 was rejected). All safe to drop = this whole session can be dropped.

## 3. Eval launcher (user-approved 14:15 IST)
Trigger: loops every 10 s until `date +%s >= 1790673270` (14:44:30 IST) **and** no process matches
`^/venv/train/bin/python /venv/train/bin/torchrun .*train_c1b.py`; then runs
`DEADLINE_FILE=/dev/null bash code/corpus1b/eval_ckpt.sh checkpointB` (no deadline cut), which:
- copies the checkpoint with the most tokens among `main/final` and `main/roll_*` to `logs/c1b/train/main/eval_checkpointB`
  and writes path + adapter sha256 + tokens to `logs/c1b/eval/checkpointB/checkpoint.txt` (a dry run at 14:20 left a copy
  of roll_00006875 and its checkpoint.txt there; the real launch replaces both — expect main/final, ~28.3M tokens);
- builds 54 shards longest first: judge (+DES48) on ifeval541 (8), blind accept on ifeval541 (8, `--judge blind
  --tau-acc-q 0.50 --draft s8` = RESULTS.md §2), stock ifeval541 shards 6,7 (0–5 already copied from checkpointA), judge on
  math500 (8), gpqa (8), aime (4), gsm8k500 (8), mbpp (4), humaneval (4); runs `code/draftalign/gpu_queue.sh` on GPUs 0–7;
- output dir `logs/c1b/eval/checkpointB/` (cells `<set>_judge_des48`, `ifeval541_blind`, `ifeval541_released`; job logs
  `logs/`, `gpu_queue.log`, sandboxes `/dev/shm/c1b_sandbox/eval/checkpointB/`), then `eval_summary.py` → `summary.md`.
Measured estimate 3 h 13 m (25.5 GPU-h) → done ~18:01 IST if started ~14:48. Box currently ends 15:00 IST; the user was
told to top up to 19:00 IST (absolute minimum ~18:15). If the box is not extended the evals die at 15:00 (trained
weights are on HF by ~14:50). `/workspace/deadline.env` END=1790674200 (15:00) still — update once the top-up is known.
Edge: if training dies between 14:38 and 14:44:30 nobody relaunches it and the evals start on the newest checkpoint.

## 4. Promises to the user still open
1. **Remind the user about ChartQA-300 + FinQA-500** (deferred at 14:15 IST) when these evals finish: judge + blind
   missing on both, stock FinQA shards 6,7 missing (memory note `deferred-evals-chartqa-finqa.md`).
2. Eval deliverable: per dataset judge vs stock vs blind (acc, passes/blk, exp/token, loop/no-stop; IFEval = prompt-level
   strict). Stock + blind for GSM8K/MATH/MBPP/HumanEval/GPQA/AIME come from RESULTS.md §1/§2 (same model/harness/settings);
   their ms/token was measured on an earlier machine → not comparable with ms/token here. Push the eval results to HF;
   update the scorecard artifact https://claude.ai/artifact/EnUMLBXsDhQbXv4MAMeecb (`/dev/shm/c1b_artifact/build.py`, `extract.py`).
3. Final training report: total tokens trained; tokens/h before the switch (~5.7M/h, 11:23–11:51) and after (5.42M/h
   12:09–12:37; 5.23M/h over the half hour containing the 13:45 restart); trained tokens per domain since 11.79M — the
   weighted part (from 14.47M) is `dom_tokens` in the last jsonl row, the 11.79M→14.47M part (old plan, steps 4889..5255)
   is NOT computed yet (replay `plan_epoch` for 8 ranks on CPU, sum `tokens_share` by domain); probe per domain (rows
   tagged `final` in `logs/c1b/train/main/probe.jsonl`); confirm final checkpoint + logs + probe on HF.
4. The standing request (NEW TASK): one status line every 30 min while training (tokens, tokens/h, per-domain) — I did not
   keep a strict cadence. Training ends 14:45.

## 5. Half-done / uncommitted / gotchas
- Git clean at 32b9b57 (Arush777, pushed, verified with ls-remote); only `data/`, `hf_home/` untracked (never commit them).
  `logs/c1b/HANDOFF.md` and this file are not committed. Commits: Arush777, NO Claude trailers; push via the
  credential-helper command in HANDOFF.md §0 (never put the token in a URL/config; never print it).
- The ckptloop commits/pushes the links doc every round → a concurrent `git commit` may hit `.git/index.lock`; retry.
- `pkill -f`/grep patterns can match your own shell (bit me twice) → use exact-args `awk` or `pgrep -P`.
- Foreground `sleep` is blocked; ptrace/py-spy not permitted; `/dev/shm` is noexec; CPU tests touching the trainer must use
  `CUDA_VISIBLE_DEVICES=""`.
- nvidia-smi samples bounce 0–100 % (trainer is CPU/launch-bound, ~55–75 % avg). Real stall = no new `[step` line in
  main.log for > 2 min.
- Stop/move training without restart: `echo <unix> > logs/c1b/train/main/STOP_AT`; clean stop now: `touch logs/c1b/train/main/STOP`.
