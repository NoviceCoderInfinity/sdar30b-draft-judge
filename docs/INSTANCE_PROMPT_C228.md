# Box prompt: Phase 1 continuation 128M → 228M (paste into Claude Code on the 8× RTX PRO 6000 box)

**Fill these two lines before pasting:**
```
HOURS={{HOURS}}      # how long this rental runs from now, e.g. 22 (full plan) or 11
MODE={{MODE}}        # full = 100M new tokens converted + train to 228M + full eval (needs ~22 h); box = whatever fits in HOURS
```
If either is still a placeholder, ask the user once, then proceed.

You operate this rented box for the user (Arush). The job is "more of the same SFT": continue the Phase 1 data pipeline
exactly where it stopped, convert new data, continue training the judge from 128M tokens with the identical recipe,
evaluate, and push everything to Hugging Face and GitHub as you go. The code is ready; do not rewrite it.

## 1. Secrets (store them, never print them)

Create `/root/c228.env` with exactly these lines, `chmod 600` it, and later copy it to `/workspace/dmge/.env`:
```
GITHUB_TOKEN={{GITHUB_TOKEN}}
HF_TOKEN={{HF_TOKEN}}
BACKUP_TO_HF=1
HF_PUSH_REPO=Arushhh/c1b-judge-corpus-waveA
HF_CKPT_REPO=Arushhh/c1b-judge-lora-sdar30b-b32
GIT_PUSH_BRANCH=phase1-continue-228m
```
Never echo these values, never put a token in a URL, a git config, a commit, a log or this chat. Load them with
`set -a; . /root/c228.env; set +a`. Git authenticates only through the credential helper shown below.

## 2. Non-negotiables (if one cannot be met: STOP and ask the user)

1. **Same pipeline and recipe as the 128M run, unchanged.**
   - Stock `JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc` answers via SGLang (`LowConfidenceTraced`, threshold 0.95, greedy).
   - Kept answers get `JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05` s8 teacher-forced drafts, then `to_trainset.py`.
   - `train_c1b.py` continues from `main_128m/tok_000128000000` with: LoRA r32/α64 on q,k,v,o + draft flag; LR 8e-5,
     WSD horizon 1e9; loss β 0.5 (chat 0), calibration 1.0, override weights 2 / code 3 / chat 1; domain weights
     maths 25 / code 25 / science 15 / IF 15 / chat 20, science+IF repeats ≤ 4.
   - The scripts encode all of this; do not change flags.
2. **Continue exactly where Phase 1 stopped.**
   - Wave-B generation resumes from the downloaded `done.txt` files, in the same hash order.
   - Never regenerate or re-convert an existing item.
   - Never retrain the maths/code/chat items main_128m already trained on (`build_index_228m.py` replays that run and
     must print `replay ... OK`).
3. **Nothing may be lost.**
   - The backup loops push data every 20 min and checkpoints every 5 min (every 2M tokens), verified file by file.
   - The scripts also push after every stage.
   - Never delete anything that is not verified on HF. Never `tmux kill-server`.
4. **MODE=full is a hard target: 100M new tokens converted, training to 228M tokens, and the eval.** If the plan or the
   pre-training check says it will not fit, the scripts exit with code 3: stop and ask the user. Do not cut the target
   yourself.
5. **Git identity.** Commit only as `Arush777 <23je0154@iitism.ac.in>`, with no Claude / co-author trailer. Push only
   branch `phase1-continue-228m`.

**Priorities when time is short:** (1) never lose data; (2) training tokens; (3) the eval; (4) conversion beyond what
training needs.

## 3. State at handoff (verified 8 Oct 2026)

- **Judge:** 128,003,432 tokens, step 11,675, on HF. That run trained maths 25.2M, code 23.8M, chat 20.2M, IF 15.2M,
  science 15.1M.
- **Converted corpus:** 134.4M tokens.
  - Wave A: 74.76M (165,438 items). Wave B: 59.67M (166,380 items; science and IF complete, maths/chat partial).
  - The 128M index holds 106.2M of it, with ~6.7M never trained.
- **Generated but never converted:** wave-B code from 30 Sep (65,536 decoded, 48,459 kept ≈ 24.3M tokens). It is
  converted first.
- **Never decoded:** maths 616,768 prompts, code ~553K, chat 257,243.
  - Science and IF pools are exhausted; their 30% of the mix is further passes over existing items.
  - Prompt lists must match the Phase 1 sha256 (the setup checks).
- **Data:** HF dataset `Arushhh/c1b-judge-corpus-waveA` (same paths as on the box). Weights: `Arushhh/c1b-judge-lora-sdar30b-b32`.
- **Code:** `NoviceCoderInfinity/diffusion-moe-general-expert`, branch `phase1-continue-228m`. Read
  `docs/C228_CONTINUE.md` and `docs/DATA_GENERATION_CONTINUATION.md` §4–5.

## 4. Measured rates (Phase 1 box) and the plan

- **Rates:** generation (kept corpus tokens per GPU-hour) maths 1.28M, chat 2.73M, code 6.32M; drafts 68–94
  blocks/s/GPU; training 9.5M tokens/h on 8 GPUs; eval (base decode, 7 benchmarks) ~2.5 h.
- **These come from another host.** This host's CPU may be slower (the trainer and decoder are CPU-launch-bound), so:
  - the scripts use time boxes;
  - report the real rates in your first status.
- **MODE=full:** ≈ 22 h including setup.
  - Setup 1.2, pending code 0.2, generation 5.0 (maths 3.5, chat 1.3, code 0.2), drafts 0.9, overheads 1.2,
    training 10.5, eval 2.5.
  - About $255 at $11.7/h.
- **MODE=box:** for an 11 h rental, ~51M converted, ~54M trained, quick eval (GSM8K-500, HumanEval, MBPP). For 10 h,
  ~48M / ~47M.

## 5. Steps

```bash
# 0. FIRST, the deadline (the rental clock is already running): replace <HOURS> and <MODE> with the two values above.
#    END = now + HOURS h - 15 min margin. Setup time counts against it.
echo "END=$(( $(date +%s) + <HOURS>*3600 - 900 ))" > /workspace/c228_deadline.env && echo "MODE=<MODE>" >> /workspace/c228_deadline.env
cat /workspace/c228_deadline.env
command -v git curl python3 tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq git curl python3 python3-venv tmux)
# 1. secrets (section 1), then the code
set -a; . /root/c228.env; set +a
cd /workspace && git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
  clone -b phase1-continue-228m https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git dmge
cd /workspace/dmge && cp /root/c228.env .env && chmod 600 .env && git check-ignore .env   # must print .env
git config user.name Arush777 && git config user.email 23je0154@iitism.ac.in
git config credential.helper '!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'   # reads the env var at push time; no token stored
mkdir -p logs/c1b
# 2. setup (~1-1.5 h; resumable)
tmux new -d -s c228setup 'bash code/corpus1b/ops/c228_setup.sh 2>&1 | tee -a logs/c1b/c228_setup.log'
#    wait for "SETUP DONE" in logs/c1b/c228_setup.log (it must also print the smoke line with "PASS": true)
# 3. the run (resumable: rerun the same command after any failure). It prints the plan first (logs/c1b/c228_state/plan.md).
tmux new -d -s c228 'bash code/corpus1b/ops/c228_run.sh 2>&1 | tee -a logs/c1b/c228.log'
```

The scripts load `.env` themselves. Run `git` only from `/workspace/dmge`. Shell variables do not survive between your
commands, so always source `/root/c228.env` again before a git command that needs the token. MODE=full needs about
22 h of work, so rent at least 23 h for it. With less time, use MODE=box.

## 6. What to watch and report (every 30 min, one short block, IST times)

- **Stage:** `tail -3 logs/c1b/c228.log`. Counts per domain: `/venv/train/bin/python code/corpus1b/ops/c228_count.py`.
- **Generation:** the `left` and `elapsed_h` fields in `tail -1 logs/c1b/gen/waveB/gpu*/stats.jsonl`.
- **Training:** tokens and tokens/s in the last row of `logs/c1b/train/main_228m/train.rank0.jsonl`.
- **Backups:** the last good round, from `tail -2 logs/c1b/backup_loop.log logs/c1b/backup_ckptloop.log`.
- **Every status:** projected end vs `logs/c1b/c228_state/plan.md`.
- **First status:** the plan, plus the measured rates vs section 4.
- **At the end:** paste `logs/c1b/eval/checkpoint228M_nopool/compare.md` (accuracy and passes/block vs the 128M judge
  and stock), the final tokens trained, and the HF links.
- **When to tell the user the box can be stopped:** only after `logs/c1b/READY_TO_KILL` exists (written when
  `c228_verify.py` prints ALL PASS).

## 7. Known pitfalls (Phase 1)

- **Killing processes:** `pkill -f` / `grep` can match your own shell. Use bracketed patterns (`[g]en_sglang.py`).
- **Waiting:** foreground `sleep` may be blocked in your harness. Poll with background scripts.
- **Where code runs:** `/dev/shm` is noexec, and generated code runs in `/dev/shm/c1b_sandbox` subprocesses (the default).
- **CPU-only checks** must use `CUDA_VISIBLE_DEVICES=""` and must never touch a GPU in use.
- **GPU utilisation** swings 0–100% because the trainer is CPU-bound. A real stall is no new `[step` line in
  `logs/c1b/train/main_228m.log` for more than 2 min.
- **SGLang on sm_120** needs the CUDA 12.9 toolkit env (the scripts set it). Draft OOM was fixed with 256 running and
  0.70 memory (in the scripts).
- **HF 429 / "LFS pointer" errors:** the loops retry. Never weaken `hf_sync.py`'s secret scan.
- **A failed GPU process:** relaunched once automatically. Rerunning a stage resumes from `done.txt` and the indexes.
- **The stopped Vast instance 54614176** has a PyTorch 2.5.1 / CUDA 12.4 image. That is fine: the setup installs the
  CUDA 12.9 toolkit and its own cu128 venvs. The driver must support CUDA ≥ 12.8.
