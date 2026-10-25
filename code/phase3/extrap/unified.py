"""One table for the whole run: two rows per benchmark (rd1, rd2), status, MEASURED numbers on this 4-GPU box and the
EXTRAPOLATED numbers for the Phase 1 box (8x RTX PRO 6000 Server). Run from the repo root:
  /venv/p2/bin/python -I logs/phase3/extrap/unified.py logs/phase3/extrap/latest.json logs/phase3/extrap/status.md
Measured (this box): accuracy, passes/block, expert loads/token, tok/s = sum tokens / sum seconds, speedup vs today's
judge on identical items decoded here (full set for HumanEval/MBPP; first 12 items elsewhere, marked +), wall time =
run.log start -> done. Extrapolated (Phase 1 box): tok/s = Phase 1 box measured judge tok/s x same-box speedup (12-item
box ratio where no full baseline, marked +); x stock vs the Phase 2 page's stock; 8-GPU wall time from per-item
times replayed over 8 shards (items[k::8])."""
import glob, json, os, re, sys, datetime as dt
EX, STATUS = sys.argv[1], sys.argv[2]
ex = json.load(open(EX)); cells = ex["cells"]; proj = ex.get("projection_min", {})
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SETS = [("humaneval", "HumanEval", 164), ("mbpp", "MBPP", 257), ("gsm8k500", "GSM8K-500", 500), ("math500", "MATH-500", 500),
        ("aime", "AIME", 90), ("gpqa", "GPQA-D", 198), ("ifeval541", "IFEval-541", 541)]
def load(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if l.strip():
                try: r = json.loads(l); out[int(r["idx"])] = r
                except ValueError: pass
    return out
prog = {}
for l in open(STATUS):
    if l.startswith("| rd") or l.startswith("| base"):
        p = [x.strip() for x in l.strip().strip("|").split("|")]; prog[(p[0], p[1])] = p
upd = [l for l in open(STATUS) if l.startswith("Progress at")][0].replace("Progress at", "").strip()
# measured wall here: run.log start -> done (last attempt-1 start before the done)
walls, today = {}, dt.datetime.now(dt.timezone.utc).date()
starts = {}; dtimes = {}
for l in open("logs/phase3/run.log"):
    m = re.match(r"\[p3 (\d\d:\d\d:\d\d) UTC\] (start|done) (rd\d|judge) (\w+)", l)
    if not m: continue
    t = dt.datetime.combine(today, dt.time.fromisoformat(m.group(1)))
    if m.group(2) == "start": starts[(m.group(3), m.group(4))] = t
    elif (m.group(3), m.group(4)) in starts:
        dtimes[(m.group(3), m.group(4))] = t
        walls[(m.group(3), m.group(4))] = (t - starts[(m.group(3), m.group(4))]).total_seconds() / 60 % 1440   # run.log has time of day only: a run spanning midnight UTC wraps
rows = []
for s, name, n in SETS:
    full = os.path.exists(f"logs/phase3/decode/judge/{s}/.complete")      # only a COMPLETE base run is a full baseline
    J = load(f"logs/phase3/decode/judge/{s}/*.jsonl") if full else load(f"logs/phase3/decode/judge_sample/{s}/*.jsonl")
    for a in ("rd1", "rd2"):
        c = cells.get(f"{a}/{s}"); p = prog.get((a, s)); state = p[5] if p else "queued"; done = p[2] if p else f"0/{n}"
        eta = p[3].replace("06 Oct ", "").replace("07 Oct ", "07/") if p else ""
        if not c:
            rows.append([name, a, state, done, "", "", "", "", "", "", "", "", "", "", "", "", ("ETA " + eta) if eta else ""]); continue
        R = load(f"logs/phase3/decode/{a}/{s}/*.jsonl"); com = [i for i in R if i in J]
        sp = ""; jt = ""; dl = ""
        if com:
            tr = sum(R[i]["n_gen_to_eos"] for i in com) / sum(R[i]["seconds"] for i in com)
            tj = sum(J[i]["n_gen_to_eos"] for i in com) / sum(J[i]["seconds"] for i in com)
            sp = f"{tr / tj:.2f}x" + ("" if full else "+") + f" ({len(com)})"
            jt = f"{tj:.2f}" + ("" if full else f" ({len(com)} items; rd on them {tr:.2f})")
            dl = f"{tr - tj:+.2f} ({100 * (tr / tj - 1):+.0f}%)" + ("" if full else "+")
        d = c.get("delta_vs_S0")
        acc = f"{c['acc']:.1f}" + (f" ({d[0]:+.1f})" if d else "")
        wall = f"{walls[(a, s)]:.0f} min" if (a, s) in walls and c["complete"] else ""
        x = c.get("tps_headline"); xs = (x / c["tps_stock_table1"]) if x else None
        flag = "" if c.get("samebox_method") == "full baseline" else "+"
        oj = c.get("tps_S0_old"); odl = f"{x - oj:+.1f} ({100 * (x / oj - 1):+.0f}%)" + flag if (x and oj) else ""
        w8 = proj.get(f"{a}/{s}", {}).get("old8")
        rows.append([name, a, state, done, acc, f"{c['passes_blk']:.2f}", f"{c['exp_tok']:.0f}", jt, f"{c['tps_box']:.2f}", dl, sp, wall,
                     f"{x:.1f}{flag}" if x else "", odl, f"{xs:.2f}x" if xs else "", f"{w8:.0f} min" if (w8 and c["complete"]) else "",
                     ("finished " if state == "done" else "ETA ") + eta])
    # third row: the BASE PIPELINE (SDAR-1.7B + trained judge = today's judge) where it has a full run on this box
    pb = prog.get(("base", s))
    if full or pb:
        Jf = load(f"logs/phase3/decode/judge/{s}/*.jsonl"); bdone = bool(glob.glob(f"logs/phase3/decode/judge/{s}/.complete"))
        bstate = "done" if bdone else (pb[5] if pb else "queued")
        if Jf:
            g = sum(r["n_gen_to_eos"] for r in Jf.values()); sec = sum(r["seconds"] for r in Jf.values())
            bacc = f"{100 * sum(bool(r['correct']) for r in Jf.values()) / len(Jf):.1f}"
            bpass = sum(r["n_forward"] - r["n_prefill"] - r["n_flush"] for r in Jf.values()) / sum(r["n_blocks"] for r in Jf.values())
            bload = sum(r["unique_expert_layer_loads"] for r in Jf.values()) / g
            ref = json.load(open("logs/phase3/extrap/p2_metrics.json"))[s]["S0_nopool"]
            stx = next((cells[f"{a2}/{s}"]["tps_stock_table1"] for a2 in ("rd1", "rd2") if cells.get(f"{a2}/{s}")), None)
            oldt = 1000 / ref["ms_tok"]
            bwall = f"{walls[('judge', s)]:.0f} min" if ("judge", s) in walls and bdone else ""
            rows.append([name, "base (your pipeline)", bstate, f"{len(Jf)}/{n}", bacc, f"{bpass:.2f}", f"{bload:.0f}", f"{g / sec:.2f}",
                         f"{g / sec:.2f}", "reference", "1.00x (ref)", bwall, f"{oldt:.1f} (measured on old box)", "reference",
                         f"{oldt / stx:.2f}x" if stx else "", "", (("finished " + (dtimes[("judge", s)].replace(tzinfo=dt.timezone.utc).astimezone(IST).strftime("%H:%M") if ("judge", s) in dtimes else "")) if bdone else ("ETA " + pb[3].replace("06 Oct ", "").replace("07 Oct ", "07/") if pb else ""))])
        else:
            rows.append([name, "base (your pipeline)", bstate, f"0/{n}", "", "", "", "", "", "", "", "", "", "", "", "",
                         ("ETA " + pb[3].replace("06 Oct ", "").replace("07 Oct ", "07/")) if pb else ""])
H = ["Benchmark", "Setup", "Status", "Items", "Accuracy % (Δ vs your pipeline)", "Passes/block", "Expert loads/token",
     "MEASURED here: your pipeline tok/s (same items)", "MEASURED here: tok/s (all items)", "MEASURED here: Δ tok/s vs your pipeline",
     "MEASURED here: speedup vs your pipeline (items)", "MEASURED here: wall (4 GPUs)",
     "EXTRAPOLATED Phase 1 box: tok/s", "EXTRAPOLATED: Δ tok/s vs your pipeline", "EXTRAPOLATED: × stock", "EXTRAPOLATED: wall (8 GPUs)", "Time (IST)"]
print(f"Status at {upd}" + ("" if upd.endswith("IST") else " IST"))
print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
# (base rows are built in the loop above)
ICON = {"done": "✅ done", "running": "🟡 running", "queued": "🔴 to do"}
for r in rows:
    r[2] = ICON.get(r[2], r[2]); print("| " + " | ".join(r) + " |")
print("\n+ = speedup vs your pipeline from its first-12-item baseline only (not fully measured; can be off by up to ~13%).")
