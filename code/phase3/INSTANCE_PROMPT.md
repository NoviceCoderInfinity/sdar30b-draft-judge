You are running Phase 3 of the diffusion-moe-general-expert project, unattended, on this rented 8x RTX PRO 6000 box.
Goal: run two decode setups on 7 benchmarks and save everything. Change nothing about the setting.

Tokens (never print, echo, log or commit them):
HF_TOKEN=<paste>
GITHUB_TOKEN=<paste>

1. Check the box: `nvidia-smi` shows 8 RTX PRO 6000 GPUs and a driver for CUDA 12.8 or newer; at least 200 GB free
   disk. Install git, tmux and python3-venv with apt-get if missing.
2. Clone and configure:
   ```
   export HF_TOKEN=... GITHUB_TOKEN=...
   cd /workspace 2>/dev/null || cd ~
   git clone -b phase3-redraft https://x-access-token:$GITHUB_TOKEN@github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git dmge
   cd dmge && git remote set-url origin https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git
   printf 'HF_TOKEN=%s\nGITHUB_TOKEN=%s\n' "$HF_TOKEN" "$GITHUB_TOKEN" > .env      # .env is gitignored
   ```
3. Read `code/phase3/README.md`. Then start:
   `mkdir -p logs/phase3 && tmux new -d -s p3 'bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log'`
4. Check every 20 minutes: `tail -5 logs/phase3/run.log` and `nvidia-smi`. The script does the setup, the checks,
   both setups on all 7 benchmarks, the analysis and a backup after every benchmark by itself.
5. You MAY fix: missing system packages, pip / network / download errors (retry), a crashed run (start the same tmux
   command again; it resumes where it stopped).
   You may NOT: change any decode flag, model, revision, adapter, benchmark, length limit, block size, threshold or
   shard count; add an expert pool (DES); use SGLang; edit `code/draftalign/*`. If a fix would need any of these,
   stop and write down what happened.
6. Stop and report, without improvising, if a setup check fails, a benchmark fails twice, or the secret scan blocks a
   backup.
7. When `run.log` says DONE: confirm `logs/phase3/backup.log` shows the Hugging Face upload and the GitHub push
   succeeded, print `results/phase3/TABLE.md`, and write a 10-line summary: accuracy and passes per block of rd1 and rd2
   against today's judge, tokens per second rd1 vs rd2, and anything unusual.
