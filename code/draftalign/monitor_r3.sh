#!/usr/bin/env bash
# Compact round-3 status for the lightweight monitor (IST times, hours left to the 18:00 IST server shutdown).
#   RUN=r3_B (default) — r3_B = fresh recipe from r3_A step000700 (epoch 1): step N ≈ epoch 1 + N/710.4
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN=${RUN:-r3_B} /venv/r3/bin/python - <<'PY'
import json, datetime as dt, subprocess, os, glob
IST = dt.timezone(dt.timedelta(hours=5, minutes=30)); now = dt.datetime.now(IST)
dead = now.replace(hour=18, minute=0, second=0, microsecond=0)
run = os.environ["RUN"]; d = f"logs/draftalign/train/{run}"
cfg = json.load(open(f"{d}/config.json"))["config"]; total = int(cfg["steps"])
ep0 = 1.0 if (cfg.get("init_adapter") or run == "r3_B") else 0.0   # r3_B starts from the epoch-1 weights (resumes drop the flag)
rows, evs = {}, []
p0 = f"{d}/train.rank0.jsonl"
for l in (open(p0) if os.path.exists(p0) else []):
    if l.strip():
        r = json.loads(l)
        if r.get("event") == "eval": evs.append(r)
        elif "step" in r: rows[r["step"]] = r
log = open(f"{d}/train.log").read().split("[resume] from step")[-1]     # tracebacks since the last (re)start
ranks = int(subprocess.run(f"pgrep -fc 'train[.]py.*{run}'", shell=True, capture_output=True, text=True).stdout.strip() or 0)
if not rows:
    print(f"{now:%H:%M} IST | {run}: no step logged yet | ranks {ranks} | tracebacks {log.count('Traceback')}"); raise SystemExit
st = sorted(rows); last = rows[st[-1]]
tail = [rows[s] for s in st[-20:] if "sec_per_step" in rows[s]]
sps = sum(r["sec_per_step"] for r in tail) / max(len(tail), 1)
def eta(step): return (now + dt.timedelta(seconds=max(0, step - last["step"]) * sps)).strftime("%H:%M")
nxt = (last["step"] // 100 + 1) * 100
push = [l for l in open(f"logs/draftalign/ckpt_sync_r3_{run}.log") if "pushed" in l] if os.path.exists(f"logs/draftalign/ckpt_sync_r3_{run}.log") else []
bl = open("logs/draftalign/backup_loop_r3.log").read().splitlines() if os.path.exists("logs/draftalign/backup_loop_r3.log") else []
print(f"{now:%H:%M} IST ({(dead-now).total_seconds()/3600:.1f} h to 18:00 shutdown) | {run} step {last['step']}/{total} (epoch {ep0 + last['step']*32/22734:.2f}) "
      f"loss {last['loss']:.2f} ce {last['ce']:.2f} cal {last.get('calib', 0):.3f} acc {last['acc']:.3f} top1 {last['top1']:.3f} | {sps:.1f} s/step | ckpt {nxt} ~{eta(nxt)}, final {total} ~{eta(total)} IST")
e = evs[-1] if evs else {}
print(f"ranks {ranks}/9 | tracebacks {log.count('Traceback')} | last holdout: " + (f"step {e['step']} draft_acc {e.get('draft_pass1_acc', float('nan')):.4f} conf_prec {e.get('draft_pass1_conf_precision', float('nan')):.4f} override: acc {e.get('draft_pass1_ovr_acc', float('nan')):.4f} conf_prec {e.get('draft_pass1_ovr_conf_precision', float('nan')):.4f}" if e else "-"))
print(f"last ckpt push: {push[-1].strip()[-60:] if push else '-'} | ckpt_sync alive {int(subprocess.run('pgrep -fc \"ckpt_sync_r3[.]sh\"', shell=True, capture_output=True, text=True).stdout.strip() or 0)}")
q = "logs/draftalign/queue_eval_r3B.txt"
if os.path.exists(q + ".launched"):
    nb = len(glob.glob("logs/draftalign/probes_r3/r3_B/step*/*/judge_des48/*.jsonl.done")); nj = len(glob.glob("logs/draftalign/probes_r3/r3_B/step*/*/judge/*.jsonl.done")); ns = len(glob.glob("logs/draftalign/probes_r3/r3_A/stock/[ag][ip][mq]*/stock_des*/*.jsonl.done"))
    ql = sum(1 for l in open(q) if l.strip()) if os.path.exists(q) else 0
    print(f"post-train eval: r3_B judge+DES48 shards done {nb}/33 | stock DES baseline shards done {ns}/27 | r3_B judge (no DES) shards done {nj}/37 | queued not started {ql} | decodes running {int(subprocess.run('pgrep -fc \"run_da_decode[.]py\"', shell=True, capture_output=True, text=True).stdout.strip() or 0)}")
print(f"backup loop alive {int(subprocess.run('pgrep -fc \"backup_loop_r3[.]sh\"', shell=True, capture_output=True, text=True).stdout.strip() or 0)} | last: {[x for x in bl if 'pass done' in x or 'FAILED' in x or 'Traceback' in x][-1:] }")
PY
