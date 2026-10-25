# Prompt for Claude Code on the GPU instance (paste as the first message)

```text
You are running Phase 1 (the 1B-token judge corpus), WAVE A ONLY, on this 8x RTX PRO 6000 Blackwell box.
Work autonomously. Ask me only if a step fails its check twice after your fixes. Never print, log or commit a token.

1. Secrets: ~/.env holds HF_TOKEN, GITHUB_TOKEN (classic PAT, repo scope) and optionally WANDB_API_KEY,
   HF_PUSH_REPO, BACKUP_TO_HF=1. Load with: set -a; . ~/.env; set +a
2. GitHub access must print 200, else stop and tell me:
   curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: token $GITHUB_TOKEN" \
     https://api.github.com/repos/NoviceCoderInfinity/diffusion-moe-general-expert
3. Clone without putting the token in the URL or git config:
   git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
     clone -b phase1-datacorpus https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git /workspace/dmge
   cd /workspace/dmge && cp ~/.env .env && git check-ignore .env
4. Read AGENTS.md (sections 1 and 9), docs/CORPUS_1B_BUILD.md and docs/CORPUS_1B_PLAN_2026-09-28.md if present.
5. The code was written without a GPU. First make it run:
   - create /venv/train as setup_box.sh does, then run: /venv/train/bin/python -m pytest -q code/corpus1b/tests code/tests
     Fix real failures in code/corpus1b (two runner tests, test_runner_helpers_and_analyzer and
     test_runner_adapter_and_pool_helpers, also fail on round 3: ignore those).
   - if build_prompt_lists.py, spotcheck.py or any extractor/checker is missing or incomplete, finish it from the
     contracts in code/corpus1b/common.py and sources.py (every source's extractor, filters, quotas).
   Commit fixes to phase1-datacorpus with clear messages (code only, never data/logs).
6. Stage 1 (~3 h): tmux new -s s1 'bash code/corpus1b/setup_box.sh 2>&1 | tee -a logs/c1b/setup.log'
   All four checks (gen, draft, train, speed) must PASS (logs/c1b/stage1/report.jsonl) before wave A.
   On failure use the troubleshooting table in docs/CORPUS_1B_BUILD.md, then rerun (steps resume).
7. Wave A (~14 h): tmux new -s wa 'bash code/corpus1b/run_wave_a.sh 2>&1 | tee -a logs/c1b/waveA.log'
   Every 15 minutes check: prompts/GPU-hour per domain vs stage 1, pass rates, EOS rate, GPU memory, free disk
   (keep > 50 GB). Use bracketed pgrep patterns (AGENTS.md 9(d)). If BACKUP_TO_HF=1: nohup bash code/corpus1b/backup_hf.sh loop &
8. When logs/c1b/gate_A.md exists, show me its tables and verdict and the hours each stage took. Do NOT start wave B
   until I say so.
Hard rules: do not change the decode rule, the loss, the data mix or the models (stock JetLM/SDAR-30B-A3B-Chat-b32 @
c351bbc as the judge base, stock JetLM/SDAR-1.7B-Chat-b32 @ 2c83c05 as drafter, block 32). Never write outside
/workspace/dmge except /dev/shm sandboxes. Never run generated code from the repo root.
```
