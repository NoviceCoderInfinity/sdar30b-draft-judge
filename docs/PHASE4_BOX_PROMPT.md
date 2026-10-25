# Box prompt: phase 4, the drafter LoRA (paste into Claude Code on the rented box)

Implements the "SDAR Drafter LoRA Runbook" (12 Oct 2026). Assumes the operator has already cloned the repository under
`/workspace` (GitHub over SSH) and will paste a temporary Hugging Face token and the W&B key into the chat. Paste
everything below the line into Claude Code, started inside tmux on the box. Rotate both keys after the run.

---

You are Claude Code on a rented 8x RTX PRO 6000 (96 GB, Blackwell) box, working for Arush. Your job: train a LoRA adapter
on the SDAR-1.7B **drafter** with the code already written on branch `phase4-drafter-lora`, then run its gates and the
rd1 / rd2 evaluation. The 30B judge and its 128M LoRA stay frozen. Read this whole prompt before running anything.

## 0. What you have access to
- **The repository**, already cloned under `/workspace` (GitHub `NoviceCoderInfinity/diffusion-moe-general-expert`).
  Find it (`ls /workspace`; usually `/workspace/diffusion-moe-general-expert`) and work from its root. Every path below
  is relative to that root.
- **GitHub over SSH**: `git fetch`, `git push` work with no token. Check with `git remote -v` and `ssh -T git@github.com`.
- **Hugging Face**: a temporary token, given to you in this chat. It must read the `Arushhh/*` repos and write the
  dataset `Arushhh/phase4-drafter-lora`.
- **Weights & Biases**: an API key, given to you in this chat, for the training curves.
- **8 GPUs, 96 GB each**, and `/dev/shm` for decode sandboxes.

## 1. Keys (do this first, without echoing any key)
Write the keys to `/workspace/.env`, mode 600. Never print, log, commit or upload them:
```
HF_TOKEN=<the Hugging Face token from the chat>
WANDB_API_KEY=<the W&B key from the chat>
WANDB_ENTITY=arushsharma536-spe-iit
WANDB_PROJECT=phase4-drafter-lora
ENABLE_WANDB=1
```
Use the W&B entity and project the operator gives, if they give other ones. Load it in every shell with
`set -a; . /workspace/.env; set +a`. `code/drafter_lora/p4.sh` loads it by itself.

## 2. Repository and branch
```bash
cd /workspace/diffusion-moe-general-expert          # or wherever the clone is
git fetch origin && git checkout phase4-drafter-lora && git pull --ff-only
git config user.name Arush777 && git config user.email 23je0154@iitism.ac.in
git log --oneline -1                                 # report this commit
```

## 3. Files: read these first, in this order
1. `docs/PHASE4_DRAFTER_LORA.md`: the design, the run order, §2a "How training proceeds" (what healthy logs look
   like), §3 the loss, §4 every choice and deviation from the runbook.
2. `AGENTS.md`: the Phase 4 paragraph, §8 metric definitions, §9 footguns (read all of it).
3. `code/drafter_lora/p4.sh`: **the entry point.** One stage per runbook step. You run this; you do not call the
   Python scripts directly unless a stage fails and you need to debug it.

What each Phase 4 file does (all in `code/drafter_lora/`):

| file | role |
|---|---|
| `p4.sh` | stages `setup`, `smoke`, `train`, `gate1`, `gate2`, `gate3`, `gate4`, `backup` |
| `train_drafter_lora.py` | the trainer (torchrun, 8 GPUs): one pass, checkpoints every 10%, 25% check, speed projection, merge |
| `lora.py` | LoRA r 32 / alpha 64 on q,k,v,o,gate,up,down; save, load, merge |
| `data.py` | corpus items, the 3,000-item held-out split, the 40% / 60% training states, two-stream batches |
| `forward.py` | the two-stream forward and the loss (CE + 0.1 KL on slots the original gets right) |
| `replay.py` | s8 first-draft replay and re-draft probe (used by the 25% check and Gate 1) |
| `eval_agree.py` | Gate 1 and the smoke parity check |
| `judge_probe.py` | Gate 2 |
| `compare_gates.py` | Gates 3 and 4 against Phase 3 and vanilla records |
| `selfcheck.py`, `smoke_check.py`, `check_merged.py` | setup self-check, smoke verdict, merged-drafter check |
| `fetch_p4.py`, `hf_progress.py` | downloads with progress lines (corpus, Phase 3 and vanilla records, models) |
| `backup_p4.py` | secret scan + upload to `Arushhh/phase4-drafter-lora`; `--scan-staged` before every push |

The existing code it builds on (read, do not change unless a stage fails because of it):
`code/draftalign/da_decode.py` (rd1 / rd2 decoding; `--redraft-ban-rejected`), `code/draftalign/run_da_decode.py`
(the decode driver: `--drafter`, `--full-redraft-every`), `code/draftalign/drafter.py` (`draft_block`, the ban),
`code/draftalign/train_drafter.py` (checkpoint format), `code/draftalign/train_fast.py` (attention pieces),
`code/draftalign/eval_drafter_agreement.py` (s8 replay, `load_drafter`), `code/corpus1b/train_c1b.py` (judge probe),
`code/phase3_8gpu/setup_p3_8gpu.sh` (environment, models, judge adapter, 16/16 checks), `code/phase3_8gpu/run_p3_8gpu.sh`
and `report_8gpu.py` (how Phase 3 ran and reported). Tests: `code/tests/test_drafter_lora.py`,
`code/tests/test_da_decode_redraft.py`.

## 4. Rules. If one cannot be met, STOP and ask Arush (through the operator)
1. Run the stages of `p4.sh` in order: `setup`, `smoke`, `train`, `gate1`, `gate2`, `gate3`, `gate4`. Never start a
   stage until the previous one printed PASS or done. A stage that exits 3 is a stop rule: report its JSON and ask.
2. Never change the trainer's defaults (LoRA r 32 / alpha 64 / dropout 0 on the 7 projections, 34,865,152 trainable;
   CE + 0.1 KL; AdamW 0.9 / 0.95, lr 8e-5, 2% warmup, cosine to 10%, clip 1.0; 256 items per step; one pass over
   328,818 items; 3,000 held out). The only knob you may change after the smoke test, with the measured numbers
   reported, is `--micro-tokens` (memory). Never cut the data, tune, change a threshold or edit the loss to make a gate
   pass.
3. If a stage fails on a bug (not a gate), you may fix the code in `code/drafter_lora/`: smallest fix, run
   `/venv/p2/bin/python -m pytest -q code/tests/test_drafter_lora.py code/tests/test_da_decode_redraft.py`, commit,
   report what you changed. Anything outside `code/drafter_lora/`: ask first.
4. The judge stays frozen: the 128M adapter at `logs/phase2/ref/c1b_128M/tok_000128000000`, never merged.
5. Never state a speed or a time you have not measured on this box. Label projections as projections.
6. Box time: at most 25 hours. Judge retraining (runbook section 8) is NOT part of this job.
7. Gate 4 runs BOTH rd1 and rd2 on all 7 sets. Neither is optional.

## 5. Monitoring, for the whole run
Run each stage in its own tmux window and watch from another:
```bash
tmux new -d -s p4 'bash code/drafter_lora/p4.sh <stage> 2>&1 | tee -a logs/phase4/<stage>.out'
tail -f logs/phase4/p4.log           # one line per stage event (start, PASS, STOP)
tail -f logs/phase4/setup.log        # downloads: "[dl 30B judge base] 23.41/61.07 GB (38.3%) files 7/25 112.4 MB/s ETA 5m35s"
tail -f logs/phase4/train/main.log   # training: one line per step (loss, ce, kl, acc vs original, it/s, tok/s)
watch -n 30 nvidia-smi
cat logs/phase4/events.tsv           # timestamps of every stage and decode cell
```
In monitoring shells use bracketed patterns (`pgrep -f "train_drafter_lor[a]"`, `run_da_decod[e].py`), never the bare
name (AGENTS.md footgun d). Keep `STATUS.md` at the repo root: what is running, the last gate, the next step, box
hours used.

## 6. The stages
1. **setup** (about 1 h, mostly downloads). Runs `setup_p3_8gpu.sh` (builds `/venv/p2`: torch 2.11.0 cu128,
   transformers 4.54.1; downloads the 30B (61 GB), the 1.7B, the 128M judge adapter and the benchmark sets), then the
   corpus (2.3 GB, 331,818 items) and the Phase 3 and vanilla records, all with progress lines in
   `logs/phase4/setup.log`. Checks: vanilla and base decoding 16/16 token-identical to Phase 1; 165,438 + 166,380
   index rows with every npz present; the CPU tests; torch has sm_120; `selfcheck.py` on the real drafter. Anything
   else: stop. Then make sure W&B is importable: `/venv/p2/bin/python -c "import wandb"`, else
   `/venv/p2/bin/pip install wandb` and record the version.
2. **smoke** (about 1 h). 200 steps on 2,000 items; the untrained drafter's replay vs the stored drafts on 200
   held-out items; a 30-step speed probe of the full configuration. `logs/phase4/smoke/smoke.json` must show PASS for:
   34,865,152 parameters, KL exactly 0 at step 1, loss falling with no NaN, parity >= 95%, projected pass <= 6 h.
   Report the measured s/step and the projection. OOM: halve `--micro-tokens` and say so. Over 6 h: stop and ask.
3. **train** (the smoke projection; about 1,285 steps). Checkpoints every 10% in `logs/phase4/train/main/ckpt/`. At
   25% it replays the held-out items and stops itself (exit 3, `STOP.json`) unless first-draft agreement rose >= +1.0
   point. If it dies, re-run the same command: it resumes from `ckpt/latest`. At the end: `final_adapter/`,
   `merged/`, and `check_merged.py` must pass.
4. **gate1**: first-draft agreement on the 3,000 held-out items >= +3.0 points (target +7); retention >= 97% (fail
   below 95%); re-draft probe with and without the ban; merge fidelity. `results/phase4_drafter/gate1.json`.
5. **gate2**: the frozen judge's commit precision on the trained drafts within 0.5 points of the stored drafts'; the
   commit share should rise. `results/phase4_drafter/gate2.json`.
6. **gate3** (about 1.4 h): rd1 + ban on items[k::32], k = 0..7, of all 7 sets (ifeval541 math500 gpqa aime humaneval
   mbpp gsm8k500), paired with Phase 3 rd1: >= 5% fewer judge passes per block (fail below 2%), pooled accuracy CI
   covers 0 or better, IFEval loops no higher. `results/phase4_drafter/gate3.json`.
7. **gate4** (about 10 h): rd1 and rd2 + ban, all 2,250 items. `results/phase4_drafter/gate4_{rd1,rd2}.json`.
   Success: IFEval loops <= 60 in both modes; IFEval and pooled accuracy CIs vs vanilla cover 0 or better; faster per
   answer than Phase 3 with tok/s not lower; >= 5% fewer judge passes.

Stop between runs: `touch logs/phase4/STOP`. Priorities if time runs short: setup, smoke, train, Gate 1; then Gates
2-3; then Gate 4 (both modes). If both modes of Gate 4 cannot fit, stop and ask.

## 7. Backups, commits, reporting
- Start the HF backup right after setup:
  `nohup /venv/p2/bin/python code/drafter_lora/backup_p4.py --loop 30 > logs/phase4/backup.log 2>&1 &`
  (private dataset `Arushhh/phase4-drafter-lora`; a secret-scan hit stops it, check `logs/phase4/backup.log`). Also
  run it once at every gate.
- Every 30 minutes and at every gate: commit code, `STATUS.md` and small logs to `phase4-drafter-lora` and push over
  SSH. Before each push: `git add <files> && /venv/p2/bin/python code/drafter_lora/backup_p4.py --scan-staged`
  (exit 2 = do NOT push). No Co-Authored-By or tool attribution in commit messages. Never `git add -A` (`.env`,
  `logs/` and records stay out).
- Tell the operator at: setup done; smoke done (measured s/step, projected pass time); the 25% check; each gate;
  anything that stops the run. Write `results/phase4_drafter/REPORT.md` with every number, command, package version
  (`/venv/p2/bin/pip freeze`), GPU and timing, measured on this box.

Start now: section 1, then section 2, then read the files in section 3, then `p4.sh setup`.
