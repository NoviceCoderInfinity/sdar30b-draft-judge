"""Safety snapshot while the run goes on (does not touch the decodes): partial analysis + extrapolation + RESUME.md,
then HF upload (own stage dir, same secret scan as backup_p3.py) and a GitHub push of results/phase3.
Run from the repo root with /venv/p2/bin/python. Skips the git part while the run's own backup is running."""
import json, os, re, shutil, subprocess, sys, time, glob, datetime as dt
sys.path.insert(0, "code/phase3")
import backup_p3 as B
L = "logs/phase3/extrap"; PY = "/venv/p2/bin/python"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
def sh(cmd, out=None):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if out: open(out, "w").write(r.stdout)
    return r
def busy(): return subprocess.run(["pgrep", "-f", "bin/python code/phase3/backup_p3[.]py"], capture_output=True).returncode == 0
now = dt.datetime.now(IST).strftime("%d %b %H:%M IST")
# 1. refresh tables (CPU only)
sh([PY, "code/phase3/analyze_p3.py"], "logs/phase3/analysis_snapshot.log")
sh([PY, "-I", f"{L}/extrap.py", os.getcwd(), f"{L}/p2_metrics.json", f"{L}/latest"])
st = sh([PY, "-I", f"{L}/status.py", os.getcwd(), f"{L}/latest.json"]).stdout
st = "\n".join(l for l in st.splitlines() if "cached" not in l and "Found the latest" not in l)
open(f"{L}/status.md", "w").write(st + "\n")
if "DONE:" in open("logs/phase3/run.log").read():   # run finished: the final page builder (same as the published page)
    sh([PY, "-I", f"{L}/build_final.py", "results/phase3/final_numbers.json", "results/phase3/REPORT.md", f"{L}/phase3_redraft.html"])
else:
    sh([PY, "-I", f"{L}/build_artifact.py", f"{L}/latest.json", f"{L}/status.md", f"{L}/phase3_redraft.html"])
os.makedirs("results/phase3", exist_ok=True)
shutil.copy(f"{L}/latest.md", "results/phase3/EXTRAPOLATION_PHASE1_BOX.md")
shutil.copy(f"{L}/latest.json", "results/phase3/extrapolation_phase1_box.json")
# 2. RESUME.md: what is done, what is left, how to continue
rows = [l for l in st.splitlines() if l.startswith("| rd")]
left = [r for r in rows if "| done |" not in r]
FINAL_STATE = "DONE:" in open("logs/phase3/run.log").read()
if not FINAL_STATE: open("results/phase3/RESUME.md", "w").write(f"""# Phase 3 run state ({now})

Box: 4x RTX PRO 6000 Blackwell Server (vast.ai), NSH=4 shards, branch phase3-redraft (code, results and all logs). Credit was expected to run out
around 07 Oct 02:40 IST; this file is refreshed every few minutes so the last copy shows where the run stopped.

## Progress (ETAs for this 4-GPU box; tok/s extrapolated to the Phase 1 8x box)

{st}

## Left to do ({len(left)} of 14 runs not finished)

{chr(10).join(left) if left else "nothing: all 14 runs finished"}

## How to continue on any box

Finished items are never redone: each shard skips items already in its records.
1. Restore `logs/phase3/` from the Hugging Face dataset `Arushhh/phase3-redraft` into the checkout (same paths).
2. Same branch, same `.env`; then `tmux new -d -s p3 'NSH=4 P3_SETS="$(cat logs/phase3/extrap/order.txt)" bash code/phase3/run_p3.sh 2>&1 | tee -a logs/phase3/run.log'`.
   Setup steps rerun on a fresh box (downloads, checks); finished benchmarks are skipped via `.complete` markers.
   NOTE: records are per shard (`*.shard{{k}}of{{N}}.jsonl`); resuming with a different shard count re-splits the items,
   so keep NSH=4 to reuse partial shards exactly (complete benchmarks are unaffected).
3. Extrapolation to the Phase 1 box: `logs/phase3/extrap/` (extrap.py, status.py, build_artifact.py; method in
   EXTRAPOLATION_PHASE1_BOX.md and the results page).
""")
# 3. HF upload (own stage)
B.load_env() if hasattr(B, "load_env") else None
from huggingface_hub import HfApi
stage = "/dev/shm/p3_stage_snap"; shutil.rmtree(stage, ignore_errors=True)
shutil.copytree("logs/phase3", f"{stage}/logs/phase3", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree("results/phase3", f"{stage}/results/phase3")
shutil.copytree("code/phase3", f"{stage}/code/phase3", ignore=shutil.ignore_patterns("__pycache__"))
hits = B.scan(stage)
if hits:
    print(f"[snapshot {now}] SECRET SCAN HIT, nothing uploaded: {hits}"); sys.exit(3)
ok_hf = False
for a in range(3):
    try:
        HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path=stage, repo_id="Arushhh/phase3-redraft", repo_type="dataset",
            commit_message=f"phase 3 snapshot {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
        ok_hf = True; break
    except Exception as e:
        print(f"[snapshot {now}] HF attempt {a+1} failed: {type(e).__name__}: {str(e)[:200]}"); time.sleep(20)
print(f"[snapshot {now}] Hugging Face: {'uploaded' if ok_hf else 'FAILED'}")
# 4. GitHub (results/phase3 only, like backup_p3), skipped while the run's own backup is running
if busy():
    print(f"[snapshot {now}] GitHub: run's backup in progress, skipped this round"); sys.exit(0)
if B.scan("results/phase3"):
    print(f"[snapshot {now}] SECRET SCAN HIT in results/phase3, GitHub skipped"); sys.exit(3)
tok = os.environ["GITHUB_TOKEN"]; url = f"https://x-access-token:{tok}@github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git"
git = ["git", "-c", "user.name=Arush777", "-c", "user.email=23je0154@iitism.ac.in"]
for f in ["extrap.py", "calib.py", "status.py", "build_artifact.py", "page_template.html", "snapshot.py", "snapshot_loop.sh",
          "baseline_then_switch.sh", "samples_then_rd.sh", "switch_order.sh", "order.txt", "p2_metrics.json"]:
    if os.path.exists(f"{L}/{f}"): shutil.copy(f"{L}/{f}", f"code/phase3/extrap/{f}")
if B.scan("code/phase3/extrap") or B.scan("logs/phase3"):
    print(f"[snapshot {now}] SECRET SCAN HIT in code/phase3/extrap, GitHub skipped"); sys.exit(3)
sh(git + ["add", "-f", "results/phase3", "code/phase3/extrap", "logs/phase3"])
if sh(["git", "diff", "--cached", "--quiet"]).returncode:
    sh(git + ["commit", "-q", "-m", f"phase 3: snapshot {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} (partial results, extrapolation, RESUME.md)"])
BR = sh(["git", "branch", "--show-current"]).stdout.strip() or "phase3-redraft"
p = sh(["git", "push", url, f"HEAD:{BR}"])
if p.returncode:
    sh(["git", "fetch", "-q", url, BR])
    changed = sh(["git", "diff", "--name-only", "HEAD...FETCH_HEAD"]).stdout.split()
    risky = [f for f in changed if f.startswith("code/draftalign/") or f == "code/phase3/run_p3.sh"]
    if risky:
        print(f"[snapshot {now}] GitHub: remote moved and touches {risky}; not merged, push skipped"); sys.exit(0)
    r = sh(git + ["rebase", "-q", "FETCH_HEAD"])
    if r.returncode:
        sh(["git", "rebase", "--abort"]); print(f"[snapshot {now}] GitHub: rebase failed, push skipped"); sys.exit(0)
    BR = sh(["git", "branch", "--show-current"]).stdout.strip() or "phase3-redraft"
p = sh(["git", "push", url, f"HEAD:{BR}"])
print(f"[snapshot {now}] GitHub: {'pushed' if p.returncode == 0 else 'push FAILED: ' + (p.stdout + p.stderr).replace(tok, '***')[-300:]}")
