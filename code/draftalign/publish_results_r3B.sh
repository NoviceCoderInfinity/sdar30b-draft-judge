#!/usr/bin/env bash
# Push r3_B eval results to GitHub AS THEY COME IN (user, 2026-09-15 15:15 IST). logs/ is never committed, so each pass
# copies the small result files into docs/results/r3_B/ and commits only that folder:
#   eval_summary.{md,json} (r3_B step000710 judge+DES48 vs stock / stock+DES / blind / r3_A epochs, full sets),
#   calibration JSON + PNGs from training (calib/), the run config, and the r3_A eval summary for reference.
# Shares the backup flock so it never races backup_once_r3.sh / git_push_r3.sh. Loops every 10 min until 17:52 IST.
#   nohup bash code/draftalign/publish_results_r3B.sh [--once] > logs/draftalign/publish_results_r3B.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
DST=docs/results/r3_B; PRB=logs/draftalign/probes_r3/r3_B; RUN=logs/draftalign/train/r3_B
END=$(TZ=Asia/Kolkata date -d "today 21:30" +%s)
ist() { TZ=Asia/Kolkata date '+%H:%M:%S IST'; }

publish() {
  exec 9>/dev/shm/dlm/backup_r3.lock; flock 9
  /venv/r3/bin/python code/draftalign/eval_summary_r3B.py --step step000710 > /dev/null 2>&1 || echo "$(ist) summary FAILED"
  mkdir -p "$DST/calib"
  cp "$PRB/eval_summary.md" "$PRB/eval_summary.json" "$DST/" 2>/dev/null
  cp "$RUN"/calib/*.json "$RUN"/calib/*.png "$DST/calib/" 2>/dev/null
  cp "$RUN/config.json" "$DST/train_config.json" 2>/dev/null
  /venv/r3/bin/python docs/report/build_results_page.py > /dev/null 2>&1 && cp docs/report/r3_results.html "$DST/r3_results.html"
  cp logs/draftalign/probes_r3/r3_A/eval_summary.md "$DST/r3_A_eval_summary.md" 2>/dev/null
  python3 - "$DST" <<'PY'
import json, sys, time
dst = sys.argv[1]; r = json.load(open(f"{dst}/eval_summary.json"))
done = {k: d["rows"]["r3B_judge_des48"]["status"] for k, d in r["datasets"].items()}
done_j = {k: d["rows"]["r3B_judge"]["status"] for k, d in r["datasets"].items()}
L = ["# r3_B results (live)", "",
     f"Updated {r['generated_ist']}. Checkpoint: `r3_B/step000710` = epoch 2 of the calibrated recipe (from r3_A step 700, "
     "β 0.5 + calibration BCE + override weights 2/3). Weights: "
     "https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train/r3_B/checkpoints", "",
     "**Results-only page (rows = models, + ICLR 2027 abstract): https://claude.ai/artifact/BD1EFygV8vfBcnLKs7Zd4V** · full report (method, curves, calibration): https://claude.ai/artifact/43SqDUC2VTkrsfddjN9jXg", "",
     "- `ABSTRACT_ICLR2027.md` — the draft ICLR 2027 abstract (storyline + numbers, regenerated from the eval files)",
     "- `eval_summary.md` — full-set results, every row paired against stock and stock + DES48 (95% bootstrap CI)",
     "- `calib/` — training-time calibration evals (ECE/NLL vs mask ratio, reliability grids, joint calibration, trends)",
     "- `r3_A_eval_summary.md` — the previous recipe's epoch-1/2 evals (subsets for GSM8K/MATH)", "",
     "Status of r3_B judge + DES48 per set: " + ", ".join(f"{k} {v}" for k, v in done.items()),
     "Status of r3_B judge (no DES, queued last; may be cut at the 17:30 IST cutoff): " + ", ".join(f"{k} {v}" for k, v in done_j.items()), "",
     "| set | r3_B judge+DES48 acc | Δ vs stock [CI] | Δ vs stock+DES48 [CI] | speedup vs stock (exp/token) | r3_B judge (no DES) acc · Δ vs stock · speedup | r3_A epoch 2 judge+DES48 Δ vs stock |",
     "|---|---:|---|---|---:|---|---|"]
def d(x):
    return "—" if not x or x.get("d_acc") is None else f"{x['d_acc']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}] (n={x['n_common_scored']})"
for k, ds in r["datasets"].items():
    c = ds["rows"]["r3B_judge_des48"]; a = ds["rows"]["r3A_e2_judge_des48"]
    sp = c["vs_stock"]["speedup_exp"] if c.get("vs_stock") and c["vs_stock"].get("speedup_exp") else None
    acc = "—" if c["acc"] is None else "%.1f" % c["acc"]
    j = ds["rows"]["r3B_judge"]; jv = j.get("vs_stock") or {}
    jcell = "—" if not j["n_records"] else ("%s (%s, n=%d) · %s · %s" % ("—" if j["acc"] is None else "%.1f" % j["acc"], j["status"], j["n_scored"],
             d(j.get("vs_stock")), "%.2f×" % jv["speedup_exp"] if jv.get("speedup_exp") else "—"))
    L.append(f"| {ds['label']} | {acc} ({c['status']}) | {d(c.get('vs_stock'))} | "
             f"{d(c.get('vs_stock_des48'))} | {'—' if sp is None else f'{sp:.2f}×'} | {jcell} | {d(a.get('vs_stock'))} |")
open(f"{dst}/README.md", "w").write("\n".join(L) + "\n")
PY
  git checkout -q round3-2026-09-14 2>/dev/null
  git add "$DST"
  if ! git diff --cached --quiet -- "$DST"; then
    git -c user.name="NoviceCoderInfinity" -c user.email="anupam.rawat.iitb@gmail.com" commit -q -m "r3_B eval results $(TZ=Asia/Kolkata date '+%H:%M IST') — $(python3 -c "import json;r=json.load(open('$DST/eval_summary.json'));print(', '.join(k+' '+d['rows']['r3B_judge_des48']['status'] for k,d in r['datasets'].items()))")

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TLpsaHBEsH253QgFvijZQ9" -- "$DST" \
      && git push -q -u origin round3-2026-09-14 && echo "$(ist) pushed $(git rev-parse --short HEAD)"
  else
    echo "$(ist) no new results"
  fi
  flock -u 9
}

publish
[ "${1:-}" = "--once" ] && exit 0
while [ "$(date +%s)" -lt "$END" ]; do sleep 600; publish; done
echo "$(ist) publish loop finished"
