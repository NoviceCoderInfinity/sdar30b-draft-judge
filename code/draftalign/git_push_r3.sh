#!/usr/bin/env bash
# Commit the round-3 code/docs/configs to branch round3-2026-09-14 and push (never logs/, hf_cache/, .env).
# Also refreshes the checkpoint + probe tables in docs/ROUND3_STATUS_2026-09-15.md from the live state.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
python3 - <<'PY'
import glob, json, os, re, time, subprocess
p = "docs/ROUND3_STATUS_2026-09-15.md"; s = open(p).read()
base = "https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_A/checkpoints/"
rows = []
log = open("logs/draftalign/ckpt_sync_r3_r3_A.log").read() if os.path.exists("logs/draftalign/ckpt_sync_r3_r3_A.log") else ""
for m in re.finditer(r"pushed (?:Nemotron-data-subset-training-diffusion-moe/checkpoints/r3_A|Nemotron-split-moe-train/r3_A/checkpoints)/(step\d+) sha (\w+)", log):
    rows.append(f"| {int(m.group(1)[4:])} | [{m.group(1)}]({base}{m.group(1)}) · sha256 `{m.group(2)}…` | yes |")
tbl = "| step | HF link | pushed |\n|---|---|---|\n" + ("\n".join(rows) if rows else "| — | (none yet) | |")
s = re.sub(r"<!-- CKPT_TABLE_START -->.*?<!-- CKPT_TABLE_END -->", "<!-- CKPT_TABLE_START -->\n" + tbl + "\n<!-- CKPT_TABLE_END -->", s, flags=re.S)
st = "logs/draftalign/probes_r3/r3_A/summary.txt"
if os.path.exists(st) and open(st).read().strip():
    s = re.sub(r"<!-- PROBES_START -->.*?<!-- PROBES_END -->", "<!-- PROBES_START -->\n```\n" + open(st).read().strip() + "\n```\n<!-- PROBES_END -->", s, flags=re.S)
tl = "logs/draftalign/train/r3_A/train.log"
if os.path.exists(tl):
    last = [l for l in open(tl) if l.startswith("[step")]
    if last: s = re.sub(r"\| measured \|.*?\|\n", f"| measured | 12.9 s/step; stopped 1560 @08:16 IST, resumed from step001500; latest: `{last[-1].strip()[:120]}` |\n", s)
s = re.sub(r"_Last update: .*_", time.strftime("_Last update: %Y-%m-%d %H:%M UTC_", time.gmtime()), s)
open(p, "w").write(s)
PY
git checkout -q -B round3-2026-09-14 2>/dev/null || git checkout -q round3-2026-09-14
git add -A code docs README.md AGENTS.md ARCHITECTURE.md DATA.md EXCLUDED.md HUGGINGFACE.md WORKTREE.md RESULTS.md 2>/dev/null
git reset -q -- logs hf_cache .env ':(glob)**/__pycache__/**' ':(glob)**/*.pyc' 2>/dev/null
if ! git diff --cached --quiet; then
  git -c user.name="NoviceCoderInfinity" -c user.email="anupam.rawat.iitb@gmail.com" commit -q -m "round 3: $(date -u +%FT%TZ) — data, wiring, launchers, live status

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Reb9ndoeHLWGAzVHP54DdV" && git push -q -u origin round3-2026-09-14 && echo "$(date -u +%FT%TZ) pushed $(git rev-parse --short HEAD)"
else echo "nothing to commit"; fi
