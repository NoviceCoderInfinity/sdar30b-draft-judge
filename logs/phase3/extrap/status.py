"""Progress table: ablation | benchmark | items done/total | ETA (IST). Run from the repo root with /venv/p2/bin/python -I.
ETA model: finished items use measured seconds; remaining items = old-box S0 seconds of the same item x the arm's
rd/S0 factor (from extrap json) x box ratio; 4 shards items[k::4] in parallel; +model load +2 min backup per benchmark."""
import glob, json, os, re, sys, datetime as dt
ROOT = sys.argv[1]; EX = sys.argv[2]
sys.path[:0] = [f"{ROOT}/code/compat", f"{ROOT}/code"]; os.chdir(ROOT)
import research_protocol as P
ex = json.load(open(EX)); RATIO = ex["box_ratio_total"]; K = ex["k_arm_vs_S0_oldbox"]
k_of = lambda a: K.get(a) or K.get("rd1") or 1.1
SETS = ["humaneval", "mbpp", "gsm8k500", "ifeval541", "math500", "gpqa", "aime"]; ARMS = ["rd1", "rd2"]; NSH = 4
LOAD = 75.0; GAP = 120.0
IST = dt.timezone(dt.timedelta(hours=5, minutes=30)); now = dt.datetime.now(dt.timezone.utc)
def load(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if l.strip():
                try: r = json.loads(l); out[int(r["idx"])] = r
                except ValueError: pass
    return out
ORD = open("logs/phase3/extrap/order.txt").read().split() if os.path.exists("logs/phase3/extrap/order.txt") else SETS
done_first = [x for x in SETS if x in ("humaneval", "mbpp")]
order = []
for i, s in enumerate(ORD):
    for j in range(2): order.append((ARMS[(i + j) % 2], s))
starts, dones = {}, {}
for l in open("logs/phase3/run.log"):
    m = re.match(r"\[p3 (\d\d:\d\d:\d\d) UTC\] (start|done) (rd\d) (\w+)", l)
    if m:
        t = dt.datetime.combine(now.date(), dt.time.fromisoformat(m.group(1)), dt.timezone.utc)
        if t > now + dt.timedelta(minutes=5): t -= dt.timedelta(days=1)
        (starts if m.group(2) == "start" else dones)[(m.group(3), m.group(4))] = t
# baseline pause: rd paused for today's judge on this box; queued rd cells start when the baseline ends
paused = restarted = None
for l in open("logs/phase3/run.log"):
    m = re.match(r"\[p3 (\d\d:\d\d:\d\d) UTC\] (paused rd run|rd paused|restart with)", l)
    if m:
        t = dt.datetime.combine(now.date(), dt.time.fromisoformat(m.group(1)), dt.timezone.utc)
        if t > now + dt.timedelta(minutes=5): t -= dt.timedelta(days=1)
        if "paused" in m.group(2): paused = t
        else: restarted = t
BASE_SETS = ["gsm8k500"]          # lead 06 Oct 22:22 IST: full base-pipeline run on GSM8K-500 after rd2 GSM8K
def base_rem(bs):
    d = f"logs/phase3/decode/judge/{bs}"
    if glob.glob(d + "/.complete"): return 0.0
    Jb = load(d + "/*.jsonl"); S0b = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{bs}/*.jsonl")
    r = (ex["cells"].get(f"rd1/{bs}") or {}).get("box_ratio_judge_12") or RATIO
    its = [int(it["idx"]) for it in P.items_for_split(bs)]
    return max(sum(S0b[i]["seconds"] * r for i in its[k::4] if i in S0b and i not in Jb) for k in range(4)) + (0 if Jb else LOAD) + 60
clock = now
if paused and (not restarted or restarted < paused):
    rem = 0.0
    for bs in ["aime", "gsm8k500", "gpqa", "math500", "ifeval541"]:
        J = load(f"logs/phase3/decode/judge_sample/{bs}/*.jsonl")
        if len(glob.glob(f"logs/phase3/decode/judge_sample/{bs}/*.jsonl.done")) >= 4: continue
        S0b = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{bs}/*.jsonl")
        first = [int(it["idx"]) for it in P.items_for_split(bs)][:12]
        rem += max(sum(S0b[i]["seconds"] * RATIO for i in first[k::4] if i in S0b and i not in J) for k in range(4)) + (0 if J else LOAD)
    clock = now + dt.timedelta(seconds=rem + 60 + sum(base_rem(b) for b in BASE_SETS))
    for k in list(starts):
        if starts[k] < paused and k not in dones: del starts[k]
rows = []
for a, s in order:
    items = [int(it["idx"]) for it in P.items_for_split(s)]; n = len(items)
    R = load(f"logs/phase3/decode/{a}/{s}/*.jsonl"); S0 = load(f"logs/phase2/ref/S0_128M_nopool_by_set/{s}/*.jsonl")
    pred = lambda i: S0[i]["seconds"] * k_of(a) * RATIO if i in S0 else 15.0
    if (a, s) in dones:
        end = dones[(a, s)]; state = "done"
    elif (a, s) in starts and starts[(a, s)] == max(starts.values()) and (a, s) not in dones:
        rem = max(sum(pred(i) for i in items[k::NSH] if i not in R) for k in range(NSH))
        end = now + dt.timedelta(seconds=rem); state = "running"
    else:
        wall = max(sum(pred(i) for i in items[k::NSH]) for k in range(NSH)) + LOAD
        end = clock + dt.timedelta(seconds=wall); state = "queued"
    if s != order[-1][1] or a != order[-1][0]:
        nxt_gap = GAP if (a, s) == [(x, y) for x, y in order if y == s][-1] else 0
    clock = end + dt.timedelta(seconds=nxt_gap) if state != "done" else max(clock, end)
    if s in BASE_SETS and (a, s) == [(x, y) for x, y in order if y == s][-1]:
        Jb = load(f"logs/phase3/decode/judge/{s}/*.jsonl")
        bdone = bool(glob.glob(f"logs/phase3/decode/judge/{s}/.complete"))
        brun = (not bdone) and paused is not None and (not restarted or restarted < paused)
        if bdone: bend = now; bstate = "done"
        elif brun: bend = now + dt.timedelta(seconds=base_rem(s)); bstate = "running"
        else:
            bend = max(clock, now) + dt.timedelta(seconds=base_rem(s)); bstate = "queued"
            clock = bend
        base_row = ("base", s, f"{len(Jb)}/{n}", bend.astimezone(IST).strftime("%d %b %H:%M") if not bdone else "done", "measured here (no extrapolation)", bstate)
    else:
        base_row = None
    c = ex["cells"].get(f"{a}/{s}")
    tps = (f"{c['tps_headline']:.1f} ({c['tps_headline']/c['tps_stock_table1']:.2f}x stock, {c['tps_headline']/c['tps_S0_old']:.2f}x S0)" + ("" if c.get("tps_old_samebox") else " est.") + ("" if c["complete"] else " partial")) if c else "-"
    rows.append((a, s, f"{len(R)}/{n}", end.astimezone(IST).strftime("%d %b %H:%M"), tps, state))
    if base_row: rows.append(base_row)
print(f"Progress at {now.astimezone(IST).strftime('%d %b %H:%M')} IST")
print("| Ablation | Benchmark | Items done/total | ETA (IST) | Extrapolated tok/s, old 8x box | State |\n|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(r) + " |")
print(f"\nRun finished (incl. final analysis + backup): ~{(clock + dt.timedelta(minutes=5)).astimezone(IST).strftime('%d %b %H:%M')} IST")
