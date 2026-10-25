# Phase 1 operations scripts (copies of logs/c1b/dl/*, which is git-ignored)

These are the exact scripts that ran the Phase 1 jobs on the 8x RTX PRO 6000 box (2026-09-28 to 09-30). They are run
from the repo root and refer to their original location `logs/c1b/dl/`; copy them back there (or adjust paths) to reuse.

| script | what it did |
|---|---|
| `train_resume*.sh` | resumes of the first run `logs/c1b/train/main` (0 → 28.5M tokens); `train_resume3.sh` is the last one |
| `wave_b_128m.sh` → `code/corpus1b/wave_b_128m.sh` | wave-B generation to ~128M corpus tokens (tracked copy) |
| `after_gen_128m.sh` | verifies wave B, builds the merged index (`code/corpus1b/build_index_128m.py`), launches `train_128m.sh`, relaunches on crash |
| `train_128m.sh` | continuation 28.5M → 128M (`--continue-from logs/c1b/train/main/final`, `--holdout-pids`) |
| `eval_after_training.sh`, `eval_after_128m.sh` | eval queues for the 28.5M (checkpointB) and 128M checkpoints |
| `run_aime_long.py` | AIME with a longer generation limit (`docs/AIME_unconstrained_token.md`) |
| `replay_seen.py`, `seen_report.py`, `seen_render.py` | exact CPU replay of what the trainer saw, and the per-domain report |
| `verify_before_kill.py`, `final_push.sh` | pre-shutdown checks (checkpoint + results on HF, git pushed) and final HF push |
| `relaunch_1345.sh`, `status.sh`, `supervise_wa*.sh`, `swap_at_handoff.sh` | wave-A / first-run supervision helpers |
| `run_parallel_dl.sh`, `resume_dl.sh`, `models_dl.sh`, `smoke*.sh`, `gpu6_code.sh`, `prompt_watcher.py` | download and smoke-test helpers |

Result documents from the same runs are in `docs/results/phase1/`.

## 128M -> 228M continuation (branch `phase1-continue-228m`, docs/C228_CONTINUE.md)

| script | what it does |
|---|---|
| `c228_setup.sh` | fresh-box setup: CUDA 12.9 toolkit, venvs + models (setup_box.sh), this run's data from HF, SGLang smoke test |
| `c228_run.sh` | plan -> convert the pending wave-B code -> generate maths/chat/code (time boxes) -> drafts -> items -> `build_index_228m.py` -> train from `main_128m/tok_000128000000` -> eval -> push -> verify |
| `c228_plan.py` | time plan from `/workspace/c228_deadline.env` (END, MODE=full\|box) with the Phase 1 measured rates |
| `c228_count.py` | generated / converted tokens per domain since the run started |
| `c228_eval.sh`, `c228_compare.py` | base-decode eval (no pool) of the newest checkpoint; per-item comparison with the 128M judge and stock |
| `c228_verify.py` | pre-shutdown check (checkpoint, index, eval on HF; branch pushed) -> `logs/c1b/READY_TO_KILL` |
