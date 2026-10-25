#!/usr/bin/env python
"""Render docs/report/r3_report.html: round 3b (r3_B calibrated retrain, via build_r3B_section.py) + round 3a (r3_A) -- idea, data, method, curves, evals.

    set -a; . ./.env; set +a
    /venv/r3/bin/python docs/report/build_r3_report.py              # W&B from cache (pulled on first run), fresh eval summary
    /venv/r3/bin/python docs/report/build_r3_report.py --refresh    # re-pull W&B history into r3_wandb_cache.json
    /venv/r3/bin/python docs/report/build_r3_report.py --no-eval    # reuse the existing eval_summary.json as-is

Read-only over logs/; writes only docs/report/r3_wandb_cache.json, docs/report/r3_report.html and (via
code/draftalign/eval_summary_r3.py) probes_r3/r3_A/eval_summary.{json,md}. No network at view time: every chart is
inline SVG generated here, with its data embedded. All times IST.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.join(ROOT, "docs", "report")
CACHE = os.path.join(HERE, "r3_wandb_cache.json")
OUT = os.path.join(HERE, "r3_report.html")
TRAIN = os.path.join(ROOT, "logs", "draftalign", "train", "r3_A")
PR = os.path.join(ROOT, "logs", "draftalign", "probes_r3", "r3_A")
EVAL_SCRIPT = os.path.join(ROOT, "code", "draftalign", "eval_summary_r3.py")
WANDB_PATH = "arushsharma536-spe-iit/sdar-draftalign/r3_A"
WANDB_URL = "https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_A"
HF_REPO = "https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"
HF = HF_REPO + "/tree/main"
GH_BRANCH = "https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert/tree/round3-2026-09-14"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_r3B_section as RB
STOP_TIME = dt.datetime(2026, 9, 15, 8, 16, tzinfo=IST)   # `done` marker mtime of the original run (02:46 UTC)

TRAIN_EX, HOLD_EX, STATES = 22734, 824, 32
EPOCH = TRAIN_EX / STATES                       # 710.44 steps
PLANNED, STOP = 2175, 1560
XMAX = 1600          # widened at render time when the resumed run goes past it
RESUME = None        # step the resumed run started from, when one exists
TRAIN_KEYS = ["loss", "ce", "cert", "acc", "top1", "uniq_experts", "lr", "tokens_per_s", "grad_norm", "sec_per_step"]
EVAL_KEYS = ["acc", "top1", "conf_frac", "conf_precision"]
esc = html.escape


# ================================================================================================================
# data
# ================================================================================================================
def to_ist(utc_str):
    if not utc_str:
        return None
    t = dt.datetime.strptime(utc_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return t.astimezone(IST)


def load_env():
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        for line in open(os.path.join(ROOT, ".env")):
            m = re.match(r"\s*(?:export\s+)?(WANDB_API_KEY)\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                os.environ[m.group(1)] = m.group(2)
    except OSError:
        pass


def pull_wandb():
    load_env()
    import wandb
    api = wandb.Api(timeout=60)
    run = api.run(WANDB_PATH)
    rows = []
    for r in run.scan_history(page_size=1000):
        keep = {k: v for k, v in r.items() if (k in ("step", "_step", "_timestamp") or k.startswith(("train/", "eval/")))
                and isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))}
        if len(keep) > 2:
            rows.append(keep)
    now = dt.datetime.now(dt.timezone.utc)
    return {"source": "wandb", "path": WANDB_PATH, "url": WANDB_URL, "state": run.state,
            "created_at": str(run.created_at), "heartbeat_at": str(getattr(run, "heartbeat_at", "") or ""),
            "pulled_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": rows}


def local_rows():
    rows = []
    for line in open(os.path.join(TRAIN, "train.rank0.jsonl")):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "eval":
            rows.append({"step": r["step"], **{f"eval/{k}": v for k, v in r.items() if k not in ("step", "event")}})
        elif "loss" in r:
            rows.append({"step": r["step"], "utc": r.get("utc"), **{f"train/{k}": v for k, v in r.items()
                                                                    if k not in ("step", "utc") and isinstance(v, (int, float))}})
    return rows


def get_wandb(refresh):
    if not refresh and os.path.exists(CACHE):
        return json.load(open(CACHE))
    try:
        data = pull_wandb()
        tmp = CACHE + ".tmp"; json.dump(data, open(tmp, "w")); os.replace(tmp, CACHE)
        print(f"[wandb] pulled {len(data['rows'])} rows -> {CACHE}")
        return data
    except Exception as e:                                         # offline / auth: fall back to the rank-0 log
        print(f"[wandb] pull failed ({type(e).__name__}: {str(e)[:120]}); using train.rank0.jsonl")
        if os.path.exists(CACHE):
            return json.load(open(CACHE))
        return {"source": "train.rank0.jsonl (W&B pull failed)", "state": "unknown", "rows": local_rows(), "url": WANDB_URL}


def series(rows, key):
    pts = {}
    for r in rows:
        if key in r and ("step" in r or "_step" in r):
            pts[int(r.get("step", r.get("_step")))] = float(r[key])
    return sorted(pts.items())


def ema(pts, w=0.8):
    out, s = [], 0.0
    for t, (x, y) in enumerate(pts, 1):
        s = w * s + (1 - w) * y
        out.append((x, s / (1 - w ** t)))
    return out


def train_facts():
    cfg = json.load(open(os.path.join(TRAIN, "config.json")))["config"]
    log = open(os.path.join(TRAIN, "train.log"), errors="replace").read()
    first = re.search(r"W(\d{4}) (\d\d:\d\d:\d\d)", log)
    launch = None
    if first:
        launch = dt.datetime.strptime(f"2026{first.group(1)} {first.group(2)}", "%Y%m%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).astimezone(IST)
    data = re.search(r"\[data\] (.*)", log); ident = re.search(r"\[draft-flag\] step-0 identity check (.*)", log)
    trainable = re.search(r"\[model\] wrapped .*?trainable (\{.*?\})", log)
    loc = local_rows(); tr = [r for r in loc if "train/loss" in r]
    last = max(tr, key=lambda r: r["step"]) if tr else {}
    pre, resumed_first = [], None
    for r in tr:
        if pre and r["step"] <= pre[-1]["step"]:
            resumed_first = r; break
        pre.append(r)
    done = os.path.join(TRAIN, "done")
    stop_t = (dt.datetime.fromtimestamp(os.path.getmtime(done), IST)
              if os.path.exists(done) and resumed_first is None and os.path.getmtime(done) < STOP_TIME.timestamp() + 3600 else STOP_TIME)
    saves = re.findall(r"\[save\] .*/(step\d+)", log)
    return {"cfg": cfg, "launch": launch, "data_line": data.group(1) if data else None,
            "identity": ident.group(1) if ident else None, "trainable": trainable.group(1) if trainable else None,
            "last_step": last.get("step"), "last_utc": last.get("utc"), "stop_time": stop_t,
            "pre": pre, "resumed_first": resumed_first,
            "sigterm": "got signal: 15" in log, "saves": saves, "local": loc,
            "sec_per_step_med": sorted(r["train/sec_per_step"] for r in tr)[len(tr) // 2] if tr else None}


def run_eval_summary(skip):
    path = os.path.join(PR, "eval_summary.json")
    if not skip:
        p = subprocess.run([sys.executable, EVAL_SCRIPT], capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[-400:])
    return json.load(open(path)) if os.path.exists(path) else None


# ================================================================================================================
# formatting
# ================================================================================================================
MINUS = "\u2212"


def num(x, nd=1, sign=False, suffix=""):
    if x is None:
        return "—"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s.replace("-", MINUS) + suffix


def ist_s(t, fmt="%H:%M IST"):
    return t.strftime(fmt) if t else "—"


# ================================================================================================================
# SVG charts
# ================================================================================================================
W, H = 480, 236
ML, MR, MT, MB = 50, 18, 30, 30
PW, PH = W - ML - MR, H - MT - MB


def nice_ticks(lo, hi, n=4):
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    raw = (hi - lo) / n; mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-15)
    t, ticks = math.floor(lo / step + 1e-9) * step, []
    while t <= hi + step * 0.999:
        ticks.append(round(t, 12)); t += step
        if t > math.ceil(hi / step - 1e-9) * step + 1e-12:
            break
    if ticks[-1] < hi - 1e-12:
        ticks.append(round(ticks[-1] + step, 12))
    return ticks, step


def tick_label(v, step):
    d = 0 if step >= 1 else (1 if step >= 0.1 else (2 if step >= 0.01 else 3))
    return f"{v:,.{d}f}".replace("-", MINUS)


def events_svg(xs):
    out = []
    ev = [(EPOCH, "epoch 1", "middle", "ev", MT - 17), (2 * EPOCH, "epoch 2", "middle", "ev", MT - 17),
          (STOP, "stop", "start", "ev stop", MT - 5)]
    if 3 * EPOCH <= XMAX:
        ev.append((3 * EPOCH, "epoch 3", "middle", "ev", MT - 17))
    if RESUME is not None:
        ev.append((RESUME, "resumed", "end", "ev res", MT - 5))
    for x, lab, anchor, cls, y in ev:
        px = xs(x)
        out.append(f'<line class="{cls}" x1="{px:.1f}" x2="{px:.1f}" y1="{MT}" y2="{MT + PH}"/>')
        dx = {"middle": 0, "end": -3, "start": 3}[anchor]
        out.append(f'<text class="evlab" x="{px + dx:.1f}" y="{y}" text-anchor="{anchor}">{lab}</text>')
    return "".join(out)


def line_chart(cid, label, sers, *, ylo=None, yhi=None, scale=1.0, dec=3, suffix="", cap100=False):
    """sers: [{name, pts, cls (raw|smooth|eval|base), color, tip}]; pts in data units (scaled by `scale`)."""
    ys = [y * scale for s in sers for _, y in s["pts"]]
    if not ys:
        return f'<div class="chart-empty">no data for {esc(label)}</div>', None
    lo = min(ys) if ylo is None else ylo; hi = max(ys) if yhi is None else yhi
    pad = (hi - lo) * 0.08 or abs(hi) * 0.05 or 1
    ticks, step = nice_ticks(lo - (pad if ylo is None else 0), hi + (pad if yhi is None else 0))
    y0, y1 = ticks[0], ticks[-1]
    xs = lambda x: ML + PW * x / XMAX
    yv = lambda y: MT + PH * (1 - (y - y0) / (y1 - y0))
    g = [f'<svg class="chart" id="{cid}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}">',
         f'<defs><clipPath id="{cid}-c"><rect x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></clipPath></defs>']
    for t in ticks:
        if cap100 and t > 100 + 1e-9:
            continue
        g.append(f'<line class="grid" x1="{ML}" x2="{ML + PW}" y1="{yv(t):.1f}" y2="{yv(t):.1f}"/>')
        g.append(f'<text class="tick" x="{ML - 7}" y="{yv(t) + 4:.1f}" text-anchor="end">{tick_label(t, step)}</text>')
    for x in range(0, XMAX + 1, 400 if XMAX <= 2000 else 500):
        g.append(f'<text class="tick" x="{xs(x):.1f}" y="{H - 10}" text-anchor="middle">{x:,}</text>')
    g.append(f'<line class="axis" x1="{ML}" x2="{ML + PW}" y1="{MT + PH}" y2="{MT + PH}"/>')
    g.append(events_svg(xs))
    g.append(f'<g clip-path="url(#{cid}-c)">')
    for s in sers:
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs(x):.1f},{yv(y * scale):.1f}" for i, (x, y) in enumerate(s["pts"]))
        g.append(f'<path class="ln {s["cls"]}" style="stroke:{s["color"]}" d="{d}"/>')
    g.append("</g>")
    for s in sers:
        if s["cls"] == "eval":
            for x, y in s["pts"]:
                g.append(f'<circle class="dot" cx="{xs(x):.1f}" cy="{yv(y * scale):.1f}" r="4" style="fill:{s["color"]}"/>')
    g.append(f'<line class="xhair" x1="0" x2="0" y1="{MT}" y2="{MT + PH}" visibility="hidden"/><g class="hdots"></g>')
    g.append(f'<rect class="hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></svg>')
    spec = {"id": cid, "xd": [0, XMAX], "yd": [y0, y1], "box": [ML, MT, PW, PH], "epoch": EPOCH,
            "series": [{"name": s["name"], "color": s["color"], "dec": dec, "suffix": suffix,
                        "pts": [[x, round(y * scale, 4)] for x, y in s["pts"]]} for s in sers if s.get("tip", True) and s["cls"] != "raw"]}   # raw stays drawn; tooltip reads the EMA (page-size budget)
    return "".join(g), spec


SHAPE = {None: "circle", 32: "square", 48: "diamond"}


def mark(shape, x, y, color, hollow, tip, r=5.5):
    fill = "var(--panel)" if hollow else color
    stroke = color if hollow else "var(--panel)"
    attrs = f'class="pt" tabindex="0" data-tip="{esc(tip)}" style="fill:{fill};stroke:{stroke}"'
    if shape == "circle":
        return f'<circle {attrs} cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if shape == "square":
        return f'<rect {attrs} x="{x - r + .5:.1f}" y="{y - r + .5:.1f}" width="{2 * r - 1:.1f}" height="{2 * r - 1:.1f}" rx="1.5"/>'
    q = r * 1.3
    return f'<path {attrs} d="M{x:.1f},{y - q:.1f} L{x + q:.1f},{y:.1f} L{x:.1f},{y + q:.1f} L{x - q:.1f},{y:.1f} Z"/>'


def scatter_chart(cid, label, pts):
    """pts: [{x (exp/token), y (acc %), color, pool, hollow, tip}]"""
    if not pts:
        return None
    xs_, ys_ = [p["x"] for p in pts], [p["y"] for p in pts]
    xt, xstep = nice_ticks(min(xs_) * 0.92, max(xs_) * 1.06, 4)
    yt, ystep = nice_ticks(min(ys_) - 3, min(max(ys_) + 3, 100), 4)
    sw, sh, ml, mb, mt, mr = 480, 260, 50, 42, 16, 18
    pw, ph = sw - ml - mr, sh - mt - mb
    X = lambda v: ml + pw * (v - xt[0]) / (xt[-1] - xt[0]); Y = lambda v: mt + ph * (1 - (v - yt[0]) / (yt[-1] - yt[0]))
    g = [f'<svg class="chart" id="{cid}" viewBox="0 0 {sw} {sh}" role="img" aria-label="{esc(label)}">']
    for t in yt:
        if t > 100 + 1e-9:
            continue
        g.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{ml - 7}" y="{Y(t) + 4:.1f}" text-anchor="end">{tick_label(t, ystep)}</text>')
    for t in xt:
        g.append(f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 17}" text-anchor="middle">{tick_label(t, xstep)}</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    g.append(f'<text class="axlab" x="{ml + pw}" y="{sh - 6}" text-anchor="end">expert loads per token (lower = cheaper)</text>')
    g.append(f'<text class="axlab" x="{ml}" y="{sh - 6}" text-anchor="start">accuracy %</text>')
    for p in sorted(pts, key=lambda p: not p["hollow"]):
        g.append(mark(SHAPE[p["pool"]], X(p["x"]), Y(p["y"]), p["color"], p["hollow"], p["tip"]))
    g.append("</svg>")
    return "".join(g)


def direction_chart(cid, label, rows):
    """rows: [{label, e1, e2, latest}] each a (d_acc, ci, n) tuple or None -- Δacc vs stock, pts."""
    vals = [v for r in rows for k in ("e1", "e2", "final") if r[k] for v in (r[k][0],)]
    if not vals:
        return None
    m = max(4.0, max(abs(v) for v in vals) * 1.15)
    ticks, step = nice_ticks(-m, m, 4)
    lo, hi = ticks[0], ticks[-1]; lim = max(abs(lo), abs(hi)); lo, hi = -lim, lim
    sw, ml, mr, mt, rh = 480, 112, 18, 14, 38
    sh = mt + rh * len(rows) + 30; pw = sw - ml - mr
    X = lambda v: ml + pw * (v - lo) / (hi - lo)
    g = [f'<svg class="chart" id="{cid}" viewBox="0 0 {sw} {sh}" role="img" aria-label="{esc(label)}">']
    for t in nice_ticks(lo, hi, 4)[0]:
        if t < lo - 1e-9 or t > hi + 1e-9:
            continue
        g.append(f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{mt + rh * len(rows)}"/>'
                 f'<text class="tick" x="{X(t):.1f}" y="{sh - 12}" text-anchor="middle">{tick_label(t, step) if abs(t) > 1e-9 else "0"}</text>')
    for i, r in enumerate(rows):
        cy = mt + rh * i + rh / 2
        g.append(f'<text class="rowlab" x="{ml - 10}" y="{cy + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        if not any(r[k] for k in ("e1", "e2", "final")):
            g.append(f'<text class="pend" x="{ml + pw / 2:.1f}" y="{cy + 4:.1f}" text-anchor="middle">pending</text>'); continue
        if r["e1"] and r["e2"]:
            g.append(f'<line class="conn" x1="{X(r["e1"][0]):.1f}" x2="{X(r["e2"][0]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        for k, shape, hollow, name in (("e1", "circle", True, "epoch 1 (step 700)"), ("e2", "circle", False, "epoch 2 (step 1,400)"),
                                       ("final", "square", False, "final (step 1,900)")):
            if r[k]:
                d, ci, n = r[k]
                tip = f"{r['label']} · {name}: Δacc {num(d, 1, True)} pts vs stock" + (f" [{num(ci[0], 1, True)}, {num(ci[1], 1, True)}]" if ci else "") + f" · n={n}"
                col = "var(--s-judge)" if k != "final" else "var(--s-judge-2)"
                g.append(mark(shape, X(d), cy, col, hollow, tip, r=5))
    g.append("</svg>")
    return "".join(g)


# ================================================================================================================
# HTML pieces
# ================================================================================================================
STATUS_TXT = {"complete": "complete", "partial": "partial", "unscored": "decoded, scoring pending", "pending": "pending",
              "dropped": "dropped", "scheduled": "scheduled", "n/a": "not planned"}
STATUS_ICON = {"complete": "✓", "partial": "◐", "unscored": "…", "pending": "○", "dropped": "×", "scheduled": "◷", "n/a": "–"}
NEXT_REFRESH = "11:57 IST"
RESUMED_AT = "09:53 IST"
FINAL_STOP = "≈11:21 IST"


def pill(c):
    st = c["status"]
    extra = ""
    if st == "partial":
        extra = f" {c['n_scored']}/{c['n_planned']}"
    elif st == "unscored":
        extra = f" {c['n_records']}/{c['n_planned']}" + (f" · {c['score_errors']} scorer errors ignored" if c.get("score_errors") else "")
    elif st == "complete":
        extra = f" {c['n_scored']}"
    if st == "dropped" and c["n_records"]:
        extra = f" · {c['n_scored']}/{c['n_planned']} ran"
    return f'<span class="pill st-{st.replace("/", "")}"><span aria-hidden="true">{STATUS_ICON[st]}</span> {STATUS_TXT[st]}{extra}</span>'


def dcell(cmp, key="d_acc"):
    if not cmp or cmp.get(key) is None:
        return '<td class="num muted">—</td>'
    d, ci = cmp["d_acc"], cmp["ci"]
    sig = ci and (ci[0] > 0 or ci[1] < 0)
    cls = ("pos" if d > 0 else "neg") if sig else ""
    icon = ("▲ " if d > 0 else "▼ ") if sig else ""
    return (f'<td class="num {cls}">{icon}{num(d, 1, True)}<span class="ci"> [{num(ci[0], 1, True)}, {num(ci[1], 1, True)}]</span>'
            f'<span class="nn">n={cmp["n_common_scored"]}</span></td>')


def speed_cell(cmp):
    if not cmp or not cmp.get("speedup_exp"):
        return '<td class="num muted">—</td>'
    return f'<td class="num">{cmp["speedup_exp"]:.2f}×<span class="nn">n={cmp["n_common"]}</span></td>'


def arm_row(label, c, is_ref=False):
    v = c.get("vs_stock"); vp = c.get("vs_pool_stock"); vprev = c.get("vs_prev")
    ncell = f'{c["n_scored"]}<span class="nn">of {c["n_planned"]}' + (f' · {c["n_records"]} decoded' if c["n_records"] != c["n_scored"] else "") + "</span>"
    return ("<tr>"
            f'<th scope="row">{esc(label)}</th><td>{pill(c)}</td><td class="num">{ncell}</td>'
            f'<td class="num strong">{num(c["acc"])}</td>'
            + ('<td class="num muted">reference</td>' if is_ref else dcell(v))
            + (dcell(vp) if not is_ref else '<td class="num muted">—</td>')
            + (dcell(vprev) if vprev is not None or not is_ref else '<td class="num muted">—</td>')
            + f'<td class="num">{num(c["passes_blk"], 2)}</td><td class="num">{num(c["exp_tok"], 0)}</td>'
            + ('<td class="num">1.00×</td>' if is_ref and c["n_records"] else speed_cell(v))
            + f'<td class="num">{num(c["sec_item"], 1)}</td></tr>')


def step_table(ev, skey, dsets=None):
    head = ("<thead><tr><th scope=\"col\">arm</th><th scope=\"col\">status</th><th scope=\"col\" class=\"num\">n scored</th>"
            "<th scope=\"col\" class=\"num\">acc %</th><th scope=\"col\" class=\"num\">Δacc vs stock</th>"
            "<th scope=\"col\" class=\"num\">vs stock, same pool</th><th scope=\"col\" class=\"num\">vs previous round, same arm</th>"
            "<th scope=\"col\" class=\"num\">passes/blk</th><th scope=\"col\" class=\"num\">exp/token</th>"
            "<th scope=\"col\" class=\"num\">speedup (exp)</th><th scope=\"col\" class=\"num\">s/item</th></tr></thead>")
    body = []
    for d in ev["datasets"]:
        if dsets and d["key"] not in dsets:
            continue
        k = d["key"]; u = ev["universe"][k]
        body.append(f'<tr class="grp"><th colspan="11" scope="rowgroup">{esc(d["label"])} <span class="muted">· {u["n"]} items · {esc(u["how"])}</span></th></tr>')
        for nm in ("stock", "stock_des32", "stock_des48"):
            body.append(arm_row(ev["arm_labels"][nm], ev["stock"][k][nm], is_ref=(nm == "stock")))
        for nm in ("judge", "judge_des32", "judge_des48"):
            body.append(arm_row(ev["arm_labels"][nm], ev["cells"][k][skey][nm]))
    return f'<div class="tscroll"><table class="data">{head}<tbody>{"".join(body)}</tbody></table></div>'


def coverage_table(ev):
    ds = ev["datasets"]
    h = "".join(f'<th scope="col">{esc(d["label"])}</th>' for d in ds)
    rows = []
    for s in ev["steps"]:
        for nm in ("judge", "judge_des32", "judge_des48"):
            cells = "".join(f"<td>{pill(ev['cells'][d['key']][s['key']][nm])}</td>" for d in ds)
            rows.append(f'<tr><th scope="row"><span class="mono">{s["key"]}</span> {esc(s["label"])} · {esc(ev["arm_labels"][nm])}</th>{cells}</tr>')
    for nm in ("stock", "stock_des32", "stock_des48"):
        cells = "".join(f"<td>{pill(ev['stock'][d['key']][nm])}<span class=\"src\">{esc(src_short(ev['stock'][d['key']][nm]))}</span></td>" for d in ds)
        rows.append(f'<tr class="stockrow"><th scope="row">{esc(ev["arm_labels"][nm])}</th>{cells}</tr>')
    return f'<div class="tscroll"><table class="data cov"><thead><tr><th scope="col">cell</th>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def src_short(c):
    if c["source"] == "mirror":
        return "team record"
    if c["status"] == "pending":
        return "new decode"
    return "new decode"


def hcell(c, ref=False, span=1):
    sp = f' colspan="{span}"' if span > 1 else ""
    if c is None:
        return f'<td{sp} class="muted">—</td>'
    if c["status"] in ("complete", "partial") and c["acc"] is not None:
        v = c.get("vs_stock"); d = '<span class="muted">reference</span>' if ref else ""
        if not ref and v and v.get("d_acc") is not None:
            sig = v["ci"][0] > 0 or v["ci"][1] < 0
            d = (f'<span class="hd {("pos" if v["d_acc"] > 0 else "neg") if sig else ""}">{num(v["d_acc"], 1, True)} vs stock</span>'
                 f'<span class="ci"> [{num(v["ci"][0], 1, True)}, {num(v["ci"][1], 1, True)}]</span>')
        part = "" if c["status"] == "complete" else f" · partial {c['n_scored']}/{c['n_planned']}"
        return (f'<td{sp}><span class="hv">{c["acc"]:.1f}%</span> {d}'
                f'<span class="nn">{num(c["exp_tok"], 0)} exp/token · {num(c["passes_blk"], 2)} passes/blk · n={c["n_scored"]}{part}</span></td>')
    return f"<td{sp}>{pill(c)}</td>"


def headline_table(ev):
    if not ev:
        return ""
    cols = [("step000700", "epoch 1 · step 700"), ("step001400", "epoch 2 · step 1,400"), ("step001500", "step 1,500 · dropped, partial")]
    rows = []
    for dk in ("humaneval", "mbpp"):
        d = next(x for x in ev["datasets"] if x["key"] == dk)
        rows.append(f'<tr class="grp"><th colspan="4" scope="rowgroup">{esc(d["label"])} <span class="muted">· {ev["universe"][dk]["n"]} items</span></th></tr>')
        for nm in ("stock", "stock_des48"):
            rows.append(f'<tr><th scope="row">{esc(ev["arm_labels"][nm])}</th>{hcell(ev["stock"][dk][nm], ref=(nm == "stock"), span=3)}</tr>')
        for nm in ("judge", "judge_des48"):
            rows.append(f'<tr><th scope="row">{esc(ev["arm_labels"][nm])}</th>'
                        + "".join(hcell(ev["cells"][dk].get(sk, {}).get(nm)) for sk, _ in cols) + "</tr>")
    head = "".join(f'<th scope="col">{esc(l)}</th>' for _, l in cols)
    return ('<p class="kick" style="margin:1rem 0 .2rem">Headline: code accuracy, epoch 1 → epoch 2, with Δ vs stock (paired, 95% CI), expert loads per token and passes per block. Step 1,500 is a partial extra.</p>'
            f'<div class="tscroll"><table class="data"><thead><tr><th scope="col">arm</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def status_box(ev, tf, latest_step):
    now = dt.datetime.now(IST); last = to_ist(tf.get("last_utc"))
    live = bool(last and (now - last).total_seconds() < 900)
    li = [f"<li><strong>Training (r3_A):</strong> finished. It stopped at step 1,560 at 08:16 IST, resumed from <code>step001500</code> at {RESUMED_AT} "
          f"with the same recipe, and was stopped for good at step 2,000 (2.82 epochs) at 11:44 IST; last logged row step {latest_step:,} at {ist_s(last)}. "
          f"No further training on this recipe: the calibrated retrain r3_B (<a href=\"#r3b\">Round 3b above</a>) starts from its step-700 weights.</li>"]
    if ev:
        for s in ev["steps"]:
            mode = s.get("mode", "normal")
            cells = [(d, nm, ev["cells"][d["key"]][s["key"]][nm]) for d in ev["datasets"] for nm in ("judge", "judge_des32", "judge_des48")]
            if mode == "normal":
                done = [x for x in cells if x[2]["status"] == "complete"]
                per = ", ".join(f'{d["label"]} ({ev["universe"][d["key"]]["n"]})' for d in ev["datasets"])
                rest = [f'{d["label"]} {ev["arm_labels"][nm]}: {STATUS_TXT[c["status"]]}' for d, nm, c in cells if c["status"] != "complete"]
                li.append(f"<li><strong>Complete:</strong> <code>{s['key']}</code> ({esc(s['label'])}): {len(done)} of {len(cells)} cells, judge / DES32 / DES48 on {esc(per)}."
                          + (f" Not yet complete: {esc('; '.join(rest))}." if rest else "") + "</li>")
            elif mode == "dropped":
                ran = [f'{d["label"]} {ev["arm_labels"][nm]} {c["n_scored"]}/{c["n_planned"]}' for d, nm, c in cells if c["n_records"]]
                li.append(f"<li><strong>Dropped:</strong> <code>{s['key']}</code>, {esc(s.get('note', ''))}. "
                          + (f"Only {esc('; '.join(ran))} ran; those numbers are kept as a reference row." if ran else "Nothing ran.") + "</li>")
            elif mode == "scheduled":
                fc = [x for x in cells if x[0]["key"] in ("humaneval", "mbpp")]
                landed = [x for x in fc if x[2]["n_records"]]
                if not landed:
                    li.append(f"<li><strong>Scheduled:</strong> <code>{s['key']}</code> (final, ≈{s['step'] / EPOCH:.2f} epochs): judge, DES48, DES32 on HumanEval and MBPP, 11:25–11:55 IST.</li>")
                else:
                    done = [x for x in fc if x[2]["status"] == "complete"]
                    li.append(f"<li><strong>Running:</strong> <code>{s['key']}</code> (final): {len(done)} of {len(fc)} HumanEval/MBPP cells complete, "
                              f"{len(landed)} with records so far (window 11:25–11:55 IST).</li>")
        sc = [(d, nm, ev["stock"][d["key"]][nm]) for d in ev["datasets"] for nm in ("stock", "stock_des32", "stock_des48")]
        sdone = [x for x in sc if x[2]["status"] == "complete"]
        miss = [f'{d["label"]} {ev["arm_labels"][nm]}' for d, nm, c in sc if c["status"] != "complete"]
        li.append(f"<li><strong>Stock baselines:</strong> {len(sdone)} of {len(sc)} cells complete (stock / DES32 / DES48 on all four datasets)"
                  + (f"; missing: {esc('; '.join(miss))}" if miss else "") + ".</li>")
    li.append(f"<li><strong>This page:</strong> generated {now:%H:%M IST}.</li>")
    return '<div class="statusbox"><h3>What is in this report</h3><ul>' + "".join(li) + "</ul>" + headline_table(ev) + "</div>"


# ================================================================================================================
# page
# ================================================================================================================
CSS = r"""
:root{
  --paper:#F4F6F7; --panel:#FFFFFF; --ink:#14181C; --ink-2:#3C444C; --ink-3:#66707A;
  --rule:#DCE1E4; --rule-2:#E9EDEF; --code-bg:#ECF0F2; --link:#005F94;
  --grid:#E3E7EA; --axis:#C3CAD0; --ev:#8A949C;
  --s-judge:#2a78d6; --s-judge-2:#184f95; --s-prev:#eb6834; --s-stock:#7A838B;
  --p-code:#2a78d6; --p-math:#eb6834; --p-if:#1baf7a; --k-ok:#2a78d6; --k-bad:#eb6834; --k-mask:#1baf7a;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; --good-text:#006300; --crit-text:#B02A2A;
  --pill-good:#E4F3E4; --pill-warn:#FDF1D6; --pill-bad:#F6DADA; --pill-pend:#ECEFF1; --pill-uns:#E3EDF9;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0E1216; --panel:#161B21; --ink:#E7EBEE; --ink-2:#B4BDC4; --ink-3:#8A949C;
    --rule:#2A323A; --rule-2:#1E252B; --code-bg:#1B2229; --link:#6CB8E6;
    --grid:#232A31; --axis:#3A434B; --ev:#6F7A83;
    --s-judge:#3987e5; --s-judge-2:#86b6ef; --s-prev:#d95926; --s-stock:#8A949C;
    --p-code:#3987e5; --p-math:#d95926; --p-if:#199e70; --k-ok:#3987e5; --k-bad:#d95926; --k-mask:#199e70;
    --good-text:#3DC23D; --crit-text:#E66767;
    --pill-good:#15301A; --pill-warn:#35290C; --pill-bad:#3A1C1C; --pill-pend:#20272E; --pill-uns:#162A40;
  }
}
:root[data-theme="dark"]{
  --paper:#0E1216; --panel:#161B21; --ink:#E7EBEE; --ink-2:#B4BDC4; --ink-3:#8A949C;
  --rule:#2A323A; --rule-2:#1E252B; --code-bg:#1B2229; --link:#6CB8E6;
  --grid:#232A31; --axis:#3A434B; --ev:#6F7A83;
  --s-judge:#3987e5; --s-judge-2:#86b6ef; --s-prev:#d95926; --s-stock:#8A949C;
  --p-code:#3987e5; --p-math:#d95926; --p-if:#199e70; --k-ok:#3987e5; --k-bad:#d95926; --k-mask:#199e70;
  --good-text:#3DC23D; --crit-text:#E66767;
  --pill-good:#15301A; --pill-warn:#35290C; --pill-bad:#3A1C1C; --pill-pend:#20272E; --pill-uns:#162A40;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:74rem;margin:0 auto;padding-inline:clamp(16px,3vw,28px);padding-block:0 5rem}
a{color:var(--link)} a:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--link);outline-offset:2px}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86em}
code{background:var(--code-bg);padding:.1em .32em;border-radius:3px}
h1,h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;text-wrap:balance;letter-spacing:-.01em}
h1{font-size:clamp(2rem,4.4vw,3rem);line-height:1.08;margin:0 0 .7rem;max-width:20ch}
h2{font-size:1.6rem;line-height:1.2;margin:0 0 .5rem}
h3{font-size:1rem;font-weight:600;margin:2rem 0 .6rem}
p,ul,ol{max-width:40rem;margin:0 0 1rem} ul,ol{padding-left:1.15rem} li{margin-bottom:.35rem}
.muted{color:var(--ink-3)} .strong{font-weight:600}
.kick{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
header.mast{padding-block:3rem 1.8rem;border-bottom:1px solid var(--rule)}
.deck{font-size:1.08rem;color:var(--ink-2);max-width:44rem}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,12.5rem),1fr));gap:.9rem 1.6rem;margin:1.6rem 0 0;padding:0}
.facts div{border-top:1px solid var(--rule);padding-top:.45rem}
.facts dt{font-family:"IBM Plex Mono",monospace;font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
.facts dd{margin:.1rem 0 0;font-size:.95rem;font-variant-numeric:tabular-nums}
section.sec{display:grid;grid-template-columns:8.5rem minmax(0,1fr);gap:0 2rem;padding-block:2.6rem;border-top:1px solid var(--rule-2)}
section.sec>.lab{padding-top:.45rem}
.body{min-width:0}
@media (max-width:760px){section.sec{grid-template-columns:minmax(0,1fr)} section.sec>.lab{padding:0 0 .3rem}}
.lede{color:var(--ink-2);font-size:1.02rem}
.callout{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:1rem 1.2rem;margin:1.2rem 0;max-width:46rem}
.callout p:last-child,.callout ul:last-child{margin-bottom:0}
.eq{font-family:"IBM Plex Mono",monospace;font-size:.84rem;line-height:1.7;background:var(--panel);border:1px solid var(--rule);
  border-radius:6px;padding:.85rem 1.05rem;margin:1rem 0;overflow-x:auto;max-width:46rem;white-space:pre}
.tscroll{overflow-x:auto;margin:.8rem 0 1.4rem;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table.data{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
table.data th,table.data td{padding:.5rem .7rem;border-bottom:1px solid var(--rule-2);text-align:left;vertical-align:top;white-space:nowrap}
table.data thead th{font-weight:600;font-size:.74rem;color:var(--ink-3);background:var(--panel);border-bottom:1px solid var(--rule)}
table.data tbody th{font-weight:500}
table.data .num{text-align:right}
table.data tr.grp th{background:var(--code-bg);font-weight:600;font-size:.82rem;white-space:normal}
table.data tr:last-child td,table.data tr:last-child th{border-bottom:none}
table.data .nn{display:block;font-size:.72rem;color:var(--ink-3)}
table.data .ci{color:var(--ink-3);font-size:.78rem}
table.data td.pos{color:var(--good-text)} table.data td.neg{color:var(--crit-text)}
table.cov td{vertical-align:middle} table.cov .src{display:block;font-size:.7rem;color:var(--ink-3);margin-top:.15rem}
table.cov tr.stockrow th{color:var(--ink-2)}
.pill{display:inline-flex;gap:.3rem;align-items:center;font-size:.74rem;padding:.12rem .5rem;border-radius:999px;white-space:nowrap;color:var(--ink)}
.st-complete{background:var(--pill-good)} .st-partial{background:var(--pill-warn)} .st-unscored{background:var(--pill-uns)}
.st-pending{background:var(--pill-pend);color:var(--ink-3)}
.split{display:flex;gap:2px;height:28px;border-radius:5px;overflow:hidden;margin:.6rem 0 .5rem;max-width:46rem}
.split span{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:.45rem 1.2rem;font-size:.82rem;color:var(--ink-2);margin:.3rem 0 .6rem;padding:0;list-style:none;max-width:none}
.legend li{display:flex;align-items:center;gap:.4rem;margin:0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px}
.lk{display:inline-block;width:18px;height:0;border-top:2px solid}
.lk.faint{opacity:.35}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,21rem),1fr));gap:1.2rem 1.6rem;margin:1rem 0}
figure.fig{margin:0;position:relative;min-width:0}
figure.fig figcaption{font-size:.8rem;color:var(--ink-3);margin-top:.2rem;max-width:40rem}
figure.fig .ft{font-size:.9rem;font-weight:600;margin:0 0 .15rem}
figure.fig .fv{font-size:.8rem;color:var(--ink-3);font-variant-numeric:tabular-nums;margin-bottom:.2rem}
svg.chart{display:block;width:100%;height:auto;overflow:visible;font-family:"IBM Plex Sans",system-ui,sans-serif}
svg .grid{stroke:var(--grid);stroke-width:1} svg .axis{stroke:var(--axis);stroke-width:1} svg .zero{stroke:var(--axis);stroke-width:1.5}
svg .tick{fill:var(--ink-3);font-size:12px;font-variant-numeric:tabular-nums}
svg .axlab{fill:var(--ink-3);font-size:11.5px}
svg .rowlab{fill:var(--ink-2);font-size:12.5px} svg .pend{fill:var(--ink-3);font-size:12px;font-style:italic}
svg .ev{stroke:var(--ev);stroke-width:1;opacity:.6} svg .ev.stop{stroke:var(--crit);opacity:.85}
svg .evlab{fill:var(--ink-3);font-size:11px}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .ln.raw{stroke-width:1.1;opacity:.35} svg .ln.base{stroke-width:2}
svg .dot{stroke:var(--panel);stroke-width:2}
svg .pt{stroke-width:2;cursor:default} svg .pt:focus{outline:none;stroke:var(--link)}
svg .conn{stroke:var(--s-judge);stroke-width:2;opacity:.45}
svg .xhair{stroke:var(--ink-3);stroke-width:1}
svg .hit{fill:transparent}
.tip{position:absolute;pointer-events:none;background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:6px;
  padding:.4rem .55rem;font-size:.76rem;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.12);white-space:nowrap;z-index:5;font-variant-numeric:tabular-nums}
.tip b{font-weight:600}
.chart-empty,.placeholder{border:1px dashed var(--rule);border-radius:6px;padding:1.2rem;color:var(--ink-3);font-size:.88rem;background:var(--panel)}
.matrix{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,15rem),1fr));gap:.8rem;margin:1rem 0}
.mcell{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:.75rem .9rem}
.mcell h4{margin:0 0 .45rem;font-size:.85rem;font-weight:600}
.mcell dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:.15rem .8rem;font-size:.82rem;font-variant-numeric:tabular-nums}
.mcell dt{color:var(--ink-2)} .mcell dd{margin:0;text-align:right}
.mcell dd.pos{color:var(--good-text)} .mcell dd.neg{color:var(--crit-text)}
details.cmd{margin:.6rem 0}
footer{border-top:1px solid var(--rule);padding-top:1.4rem;color:var(--ink-3);font-size:.82rem}
.statusbox{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:1rem 1.2rem;margin:0 0 1.6rem;max-width:60rem}
.statusbox h3{margin:0 0 .5rem;font-size:1.05rem} .statusbox ul{max-width:none}
.st-dropped{background:var(--pill-bad)} .st-scheduled{background:var(--pill-uns)} .st-na{background:var(--pill-pend);color:var(--ink-3)}
.hv{font-weight:600} .hd.pos{color:var(--good-text)} .hd.neg{color:var(--crit-text)}
.chip{display:inline-block;font-size:.7rem;padding:.05rem .45rem;border-radius:999px;background:var(--pill-pend);color:var(--ink-2);white-space:nowrap;font-weight:500;vertical-align:1px}
.chip.c-good{background:var(--pill-good);color:var(--ink)} .chip.c-bad{background:var(--pill-bad);color:var(--ink)} .chip.c-warn{background:var(--pill-warn);color:var(--ink)} .chip.c-info{background:var(--pill-uns);color:var(--ink)}
.hyp{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:.95rem 1.15rem;margin:.9rem 0;max-width:48rem}
.hyp h4{margin:0 0 .45rem;font-size:.98rem;font-weight:600;line-height:1.4;text-wrap:balance}
.hyp p{margin:0 0 .55rem} .hyp p:last-child{margin-bottom:0}
svg .ev.res{stroke:var(--s-judge);opacity:.8}
details.cmd summary{cursor:pointer;font-size:.88rem;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""
(function(){
  var specs = JSON.parse(document.getElementById('chart-data').textContent);
  function tipFor(fig){ var t = fig.querySelector('.tip'); if(!t){ t=document.createElement('div'); t.className='tip'; t.hidden=true; fig.appendChild(t);} return t; }
  function place(fig, tip, cx, cy){
    var r = fig.getBoundingClientRect(); var x = cx - r.left + 14, y = cy - r.top + 12;
    tip.hidden = false; var w = tip.offsetWidth;
    if (x + w > r.width) x = Math.max(0, cx - r.left - w - 14);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function nearest(pts, x){ var lo=0, hi=pts.length-1; while(hi-lo>1){ var m=(lo+hi)>>1; if(pts[m][0]<x) lo=m; else hi=m; }
    return Math.abs(pts[lo][0]-x) <= Math.abs(pts[hi][0]-x) ? pts[lo] : pts[hi]; }
  specs.forEach(function(s){
    var svg = document.getElementById(s.id); if(!svg) return; var fig = svg.closest('figure'); var tip = tipFor(fig);
    var xh = svg.querySelector('.xhair'), hd = svg.querySelector('.hdots'), NS='http://www.w3.org/2000/svg';
    var bx=s.box, X=function(v){return bx[0]+bx[2]*(v-s.xd[0])/(s.xd[1]-s.xd[0]);}, Y=function(v){return bx[1]+bx[3]*(1-(v-s.yd[0])/(s.yd[1]-s.yd[0]));};
    function move(ev){
      var p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; var q = p.matrixTransform(svg.getScreenCTM().inverse());
      var xv = s.xd[0] + (q.x - bx[0]) / bx[2] * (s.xd[1]-s.xd[0]);
      var lines = [], sx = null; hd.innerHTML = '';
      s.series.forEach(function(se){ if(!se.pts.length) return; var n = nearest(se.pts, xv); if(sx===null) sx = n[0];
        var c = document.createElementNS(NS,'circle'); c.setAttribute('cx',X(n[0])); c.setAttribute('cy',Math.max(bx[1],Math.min(bx[1]+bx[3],Y(n[1]))));
        c.setAttribute('r',4); c.setAttribute('class','dot'); c.style.fill = se.color; hd.appendChild(c);
        lines.push('<span class="lk" style="border-color:'+se.color+'"></span> '+se.name+': <b>'+n[1].toFixed(se.dec)+se.suffix+'</b>'); });
      if(sx===null) return;
      xh.setAttribute('x1',X(sx)); xh.setAttribute('x2',X(sx)); xh.setAttribute('visibility','visible');
      tip.innerHTML = (s.xunit==='epoch' ? '<b>epoch '+sx.toFixed(2)+'</b>' : '<b>step '+sx.toLocaleString()+'</b> · epoch '+(sx/s.epoch).toFixed(2))+'<br>'+lines.join('<br>');
      place(fig, tip, ev.clientX, ev.clientY);
    }
    svg.addEventListener('pointermove', move);
    svg.addEventListener('pointerleave', function(){ tip.hidden = true; xh.setAttribute('visibility','hidden'); hd.innerHTML=''; });
  });
  document.querySelectorAll('svg .pt').forEach(function(el){
    var fig = el.closest('figure'); var tip = tipFor(fig);
    function show(ev){ tip.textContent = el.getAttribute('data-tip'); var r = el.getBoundingClientRect();
      place(fig, tip, ev && ev.clientX ? ev.clientX : r.right, ev && ev.clientY ? ev.clientY : r.top); }
    el.addEventListener('pointerenter', show); el.addEventListener('focus', function(){ show(null); });
    el.addEventListener('pointerleave', function(){ tip.hidden = true; }); el.addEventListener('blur', function(){ tip.hidden = true; });
  });
})();
"""


def sec(label, title, inner, sid):
    return (f'<section class="sec" id="{sid}"><div class="lab"><span class="kick">{esc(label)}</span></div>'
            f'<div class="body"><h2>{esc(title)}</h2>{inner}</div></section>')


def fig(title, value_line, svg, caption, legend=""):
    return (f'<figure class="fig"><div class="ft">{esc(title)}</div>'
            + (f'<div class="fv">{value_line}</div>' if value_line else "") + legend + svg
            + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>")


def render(wb, tf, ev):
    global XMAX, RESUME
    specs = []
    rows = list(wb["rows"]); resf = tf.get("resumed_first")
    stepof = lambda r: int(r.get("step", r.get("_step", -1)))
    if resf:                                  # the resumed run supersedes W&B's copy of every step after the resume point
        RESUME = resf["step"] - 10
        rows = [r for r in rows if stepof(r) <= RESUME or not any(k.startswith(("train/", "eval/")) for k in r)]
    wsteps = {stepof(r) for r in rows if "train/loss" in r}
    wev = {stepof(r) for r in rows if "eval/draft_pass1_acc" in r}
    extra = [r for r in tf["local"] if ("train/loss" in r and r["step"] not in wsteps) or ("eval/draft_pass1_acc" in r and r["step"] not in wev)]
    rows += extra                             # file order: a resumed duplicate comes later and wins in series()
    xs_extra = sorted({r["step"] for r in extra if "train/loss" in r})
    local_only = "" if not xs_extra else (f"{xs_extra[0]:,}" if len(xs_extra) == 1 else f"{xs_extra[0]:,}–{xs_extra[-1]:,}")
    cfg = tf["cfg"]
    latest_step = tf["last_step"] or STOP
    last_step = STOP if resf else latest_step
    XMAX = max(1600, int(math.ceil(latest_step / 100.0)) * 100)
    # ---------------- training curves
    TCH = [("loss", "Total loss", "CE + β·certainty", 3, 1.0, "", 100),
           ("ce", "Cross-entropy on open slots", "", 3, 1.0, "", 100),
           ("cert", "Certainty term (entropy where already right)", "", 3, 1.0, "", 0),
           ("acc", "Train accuracy on open slots", "argmax = target", 1, 100.0, "%", 0),
           ("top1", "Mean top-1 probability (train)", "confidence, not accuracy", 1, 100.0, "%", 0),
           ("uniq_experts", "Unique experts per layer-forward", "of 128", 1, 1.0, "", 0),
           ("lr", "Learning rate", "×10⁻⁵", 2, 1e5, "", 0),
           ("tokens_per_s", "Throughput", "tokens / s", 0, 1.0, "", 0),
           ("grad_norm", "Gradient norm (pre-clip)", "clip 1.0", 2, 1.0, "", 100)]
    tfigs = []
    for key, title, sub, dec, scale, suffix, skip in TCH:
        pts = series(rows, f"train/{key}")
        if not pts:
            continue
        sm = ema(pts)
        kept = [y * scale for x, y in pts if x >= skip]
        ylo = yhi = None; off = ""
        if skip and kept:
            ylo, yhi = min(kept), max(kept)
            above = [(x, y) for x, y in pts if y * scale > yhi]
            if above:
                off = f" Steps {above[0][0]}–{above[-1][0]} are above the axis (peak {max(y for _, y in above) * scale:,.{dec}f}); the axis is fitted to steps ≥ {skip}."
        svg, spec = line_chart(f"t-{key}", title, [
            {"name": "raw", "pts": pts, "cls": "raw", "color": "var(--s-judge)"},
            {"name": "EMA 0.8", "pts": sm, "cls": "smooth", "color": "var(--s-judge)"}],
            ylo=ylo, yhi=yhi, scale=scale, dec=dec, suffix=suffix)
        specs.append(spec)
        vl = f"step {pts[-1][0]:,}: raw {pts[-1][1] * scale:,.{dec}f}{suffix} · EMA {sm[-1][1] * scale:,.{dec}f}{suffix}" + (f" · {esc(sub)}" if sub else "")
        tfigs.append(fig(title, vl, svg, off.strip()))
    t_legend = ('<ul class="legend"><li><span class="lk faint" style="border-color:var(--s-judge)"></span>raw, every 10 steps</li>'
                '<li><span class="lk" style="border-color:var(--s-judge)"></span>EMA smoothed (weight 0.8, debiased)</li>'
                '<li><span class="lk" style="border-color:var(--ev)"></span>epoch boundary (710.4 · 1,420.9)</li>'
                '<li><span class="lk" style="border-color:var(--crit)"></span>stopped at step 1,560</li></ul>')

    # ---------------- holdout curves
    efigs, hold = [], {}
    for key, title, cap in [("acc", "Accuracy on open slots", "argmax equals the target token"),
                            ("top1", "Mean top-1 probability", "average probability of the argmax token: a confidence, not an accuracy"),
                            ("conf_frac", "Confident fraction", "share of slots with top-1 probability > 0.95"),
                            ("conf_precision", "Confident precision", "accuracy on the confident slots")]:
        dp = series(rows, f"eval/draft_pass1_{key}"); am = series(rows, f"eval/allmask_pass1_{key}")
        hold[key] = (dp, am)
        if not dp:
            continue
        svg, spec = line_chart(f"h-{key}", title, [
            {"name": "judge on drafter-filled pass 1", "pts": dp, "cls": "eval", "color": "var(--s-judge)"},
            {"name": "stock (all-mask pass 1)", "pts": am, "cls": "base", "color": "var(--s-stock)"}],
            scale=100.0, dec=1, suffix="%", cap100=True)
        specs.append(spec)
        vl = f"step {dp[-1][0]:,}: judge {dp[-1][1] * 100:.1f}% · stock {am[-1][1] * 100:.1f}%" if am else ""
        efigs.append(fig(title, vl, svg, esc(cap)))
    h_legend = ('<ul class="legend"><li><span class="lk" style="border-color:var(--s-judge)"></span>judge on a drafter-filled pass-1 block (<code>draft_pass1</code>)</li>'
                '<li><span class="lk" style="border-color:var(--s-stock)"></span>stock model, all-mask pass 1 (<code>allmask_pass1</code>), flat by construction</li></ul>')

    # ---------------- derived headline numbers
    def hv(key, i):
        dp, am = hold.get(key, ([], []))
        return (dp[i][0], dp[i][1] * 100) if dp and len(dp) > abs(i) else (None, None)
    acc_first, acc_last = hv("acc", 0), hv("acc", -1)
    base_acc = hold["acc"][1][-1][1] * 100 if hold.get("acc") and hold["acc"][1] else None
    base_cf = hold["conf_frac"][1][-1][1] * 100 if hold.get("conf_frac") and hold["conf_frac"][1] else None
    cf_last = hv("conf_frac", -1); cp_last = hv("conf_precision", -1)
    dpacc = dict(hold["acc"][0]) if hold.get("acc") else {}
    acc800 = dpacc.get(800); acc1400 = dpacc.get(1400)
    ep_stop = last_step / EPOCH

    # W&B vs local cross-check
    loc = {r["step"]: r for r in tf["pre"]}
    diffs = [abs(y - loc[x]["train/loss"]) for x, y in series(wb["rows"], "train/loss") if x in loc and (RESUME is None or x <= RESUME)]
    xcheck = (f"W&B <code>train/loss</code> matches <code>train.rank0.jsonl</code> at {len(diffs)} logged steps "
              f"(max |Δ| {max(diffs):.2e}).") if diffs and wb.get("source") == "wandb" else ""
    if local_only and wb.get("source") == "wandb":
        xcheck += (f" Steps {local_only} come from <code>train.rank0.jsonl</code>: the resumed run from step {RESUME:,} (duplicate steps keep the resumed row), plus anything W&amp;B lost at SIGTERM."
                   if resf else f" Step {local_only} exists only in <code>train.rank0.jsonl</code> (W&amp;B lost its last row at SIGTERM) and is filled in from there.")

    # ---------------- eval pieces
    ev_html = []
    if ev is None:
        ev_html.append('<div class="placeholder">No <code>eval_summary.json</code> yet. Run <code>code/draftalign/eval_summary_r3.py</code>.</div>')
    else:
        cov = ev["coverage"]
        ev_html.append(f'<p class="lede">Snapshot {esc(ev["generated_ist"])}. Cells: {cov["complete"]} complete, {cov["partial"]} partial, {cov["unscored"]} decoded but not scored, '
                       f'{cov.get("scheduled", 0)} scheduled, {cov.get("dropped", 0)} dropped (step 1,500, by decision) and {cov["pending"]} pending. '
                       f'Re-running the builder picks up new records.</p>')
        ev_html.append('<h3>Coverage of the eval wave</h3>')
        ev_html.append('<p>Checkpoints <code>step000700</code> (epoch 1) and <code>step001400</code> (epoch 2), <code>step001500</code> was dropped at 09:40 IST so training could resume; only its HumanEval judge (complete) and part of judge + DES32 ran. No later checkpoint eval is scheduled before the shutdown. '
                       'Code sets are decoded without scoring; their accuracy counts only once a <code>scored/</code> shard is finished. '
                       'Stock rows come from the team’s block-32 records where they exist, otherwise from this wave’s new stock decodes.</p>')
        ev_html.append(coverage_table(ev))

        # direction view
        ev_html.append('<h3>Epoch 1 → epoch 2: which way is it moving?</h3>')
        ev_html.append('<p>Δaccuracy of each judge arm against stock, paired on the items both cover. A hollow dot is epoch 1 (step 700), '
                       'a filled dot epoch 2 (step 1,400). Code sets come first because code is what round 3 was built to repair.</p>')
        legend = ('<ul class="legend"><li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" style="fill:var(--panel);stroke:var(--s-judge);stroke-width:2"/></svg>epoch 1 · step 700</li>'
                  '<li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" style="fill:var(--s-judge)"/></svg>epoch 2 · step 1,400</li>'
                  '</ul>')
        dfigs = []
        for d in ev["datasets"]:
            k = d["key"]; rws = []
            for nm in ("judge", "judge_des32", "judge_des48"):
                r = {"label": ev["arm_labels"][nm]}
                for tag, s in (("e1", "step000700"), ("e2", "step001400"), ("final", "step001900")):
                    v = ev["cells"][k].get(s, {}).get(nm, {}).get("vs_stock")
                    r[tag] = (v["d_acc"], v["ci"], v["n_common_scored"]) if v and v.get("d_acc") is not None else None
                rws.append(r)
            svg = direction_chart(f"d-{k}", f"{d['label']} Δacc vs stock by checkpoint", rws)
            notes = []
            for nm in ("judge", "judge_des32", "judge_des48"):
                e = ev["direction"][k][nm]["e2_vs_e1"]
                if e and e.get("d_acc") is not None:
                    notes.append(f"{esc(ev['arm_labels'][nm])}: epoch 2 − epoch 1 = {num(e['d_acc'], 1, True)} pts "
                                 f"[{num(e['ci'][0], 1, True)}, {num(e['ci'][1], 1, True)}], {e['a_only']} items fixed / {e['b_only']} broken, "
                                 f"exp/token {num(e['ref_exp_tok'], 0)} → {num(e['exp_tok'], 0)} (n={e['n_common_scored']})")
                fz = ev["direction"][k][nm].get("final_vs_e2")
                if fz and fz.get("d_acc") is not None:
                    notes.append(f"{esc(ev['arm_labels'][nm])}: final − epoch 2 = {num(fz['d_acc'], 1, True)} pts "
                                 f"[{num(fz['ci'][0], 1, True)}, {num(fz['ci'][1], 1, True)}] (n={fz['n_common_scored']})")
            cap = "<br>".join(notes) if notes else "Direct epoch-2 vs epoch-1 pairing appears here once both checkpoints have scored items."
            body = svg if svg else f'<div class="placeholder">{esc(d["label"])}: no scored judge items yet.</div>'
            dfigs.append(fig(d["label"], "", body, cap))
        ev_html.append(legend + f'<div class="grid2">{"".join(dfigs)}</div>')

        # tradeoff scatter
        ev_html.append('<h3>Accuracy against expert work</h3>')
        ev_html.append('<p>Only complete cells are plotted, so every point in a panel sits on the same items. Colour marks the model: stock, the round-3 judge, or the '
                       'previous round’s judge. Shape marks the expert pool, and fill marks the epoch.</p>')
        sleg = ('<ul class="legend"><li><span class="sw" style="background:var(--s-stock)"></span>stock</li>'
                '<li><span class="sw" style="background:var(--s-judge)"></span>round-3 judge (r3_A)</li>'
                '<li><span class="sw" style="background:var(--s-prev)"></span>previous round (T2b step 3057)</li>'
                '<li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" style="fill:var(--ink-3)"/></svg>full MoE</li>'
                '<li><svg width="14" height="14" aria-hidden="true"><rect x="2" y="2" width="10" height="10" rx="1.5" style="fill:var(--ink-3)"/></svg>DES32</li>'
                '<li><svg width="16" height="16" aria-hidden="true"><path d="M8,1 L15,8 L8,15 L1,8 Z" style="fill:var(--ink-3)"/></svg>DES48</li>'
                '<li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" style="fill:var(--panel);stroke:var(--s-judge);stroke-width:2"/></svg>hollow = epoch 1</li></ul>')
        sfigs = []
        pool_of = {"stock": None, "stock_des32": 32, "stock_des48": 48, "judge": None, "judge_des32": 32, "judge_des48": 48}
        for d in ev["datasets"]:
            k = d["key"]; pts = []; missing = []
            def add(c, color, nm, hollow, who):
                if c["status"] == "complete" and c["acc"] is not None and c["exp_tok"]:
                    pts.append({"x": c["exp_tok"], "y": c["acc"], "color": color, "pool": pool_of[nm], "hollow": hollow,
                                "tip": f"{who} · {ev['arm_labels'][nm]}: {c['acc']:.1f}% at {c['exp_tok']:,.0f} exp/token · n={c['n_scored']}"})
                elif c["status"] != "pending" or who.startswith("r3"):
                    missing.append(f"{who} {ev['arm_labels'][nm]} ({STATUS_TXT[c['status']]})")
            for nm in ("stock", "stock_des32", "stock_des48"):
                add(ev["stock"][k][nm], "var(--s-stock)", nm, False, "stock")
            for nm in ("judge", "judge_des32", "judge_des48"):
                if ev["prev_round"][k][nm]["status"] != "pending":
                    add(ev["prev_round"][k][nm], "var(--s-prev)", nm, False, "previous round")
                add(ev["cells"][k]["step000700"][nm], "var(--s-judge)", nm, True, "r3 epoch 1")
                add(ev["cells"][k]["step001400"][nm], "var(--s-judge)", nm, False, "r3 epoch 2")
            svg = scatter_chart(f"s-{k}", f"{d['label']} accuracy vs expert loads", pts)
            miss = [m for m in missing if not m.startswith("stock")]
            cap = ("Not plotted yet: " + esc("; ".join(miss)) + ".") if miss else ""
            sfigs.append(fig(d["label"], f"{len(pts)} complete cells", svg or '<div class="placeholder">no complete cells yet</div>', cap))
        ev_html.append(sleg + f'<div class="grid2">{"".join(sfigs)}</div>')

        # headline tables
        ev_html.append('<h3>Headline tables: six arms per checkpoint</h3>')
        ev_html.append('<p>Δ columns are paired: both sides are computed on exactly the items they share, and <span class="mono">n</span> under each Δ is that shared count. '
                       'Brackets hold a bootstrap 95% CI (2,000 resamples). <span class="pos-t">▲</span>/<span class="neg-t">▼</span> mark CIs that exclude zero. '
                       'Speedup is stock exp/token ÷ arm exp/token on the same items. Stock rows are the same in every table.</p>')
        for s in ev["steps"]:
            mode = s.get("mode", "normal")
            ev_html.append(f'<h3 class="kick" style="margin-top:1.6rem">{esc(s["key"])} · {esc(s["label"])}</h3>')
            if mode == "dropped":
                ran = [(d, nm, ev["cells"][d["key"]][s["key"]][nm]) for d in ev["datasets"] for nm in ("judge", "judge_des32", "judge_des48")
                       if ev["cells"][d["key"]][s["key"]][nm]["n_records"]]
                rr = "".join(f'<tr><th scope="row">{esc(d["label"])} · {esc(ev["arm_labels"][nm])}</th><td>{pill(c)}</td><td class="num">{c["n_scored"]}<span class="nn">of {c["n_planned"]}</span></td>'
                             f'<td class="num strong">{num(c["acc"])}</td>{dcell(c.get("vs_stock"))}<td class="num">{num(c["exp_tok"], 0)}</td>{speed_cell(c.get("vs_stock"))}</tr>' for d, nm, c in ran)
                ev_html.append(f'<p>Dropped: {esc(s.get("note", ""))}. Everything not listed below never ran.</p>'
                               + ('<div class="tscroll"><table class="data"><thead><tr><th scope="col">cell</th><th scope="col">status</th><th scope="col" class="num">n scored</th>'
                                  '<th scope="col" class="num">acc %</th><th scope="col" class="num">Δacc vs stock</th><th scope="col" class="num">exp/token</th>'
                                  f'<th scope="col" class="num">speedup (exp)</th></tr></thead><tbody>{rr}</tbody></table></div>' if ran else ""))
            elif mode == "scheduled" and not any(ev["cells"][d]["step001900" if s["key"] == "step001900" else s["key"]][nm]["n_records"]
                                                 for d in ("humaneval", "mbpp") for nm in ("judge", "judge_des32", "judge_des48")):
                ev_html.append(f'<div class="placeholder">{esc(s.get("note", "scheduled"))}: judge, DES48 and DES32. The table appears here as records land.</div>')
            else:
                ev_html.append(step_table(ev, s["key"], ("humaneval", "mbpp") if mode == "scheduled" else None))

        # previous round
        ev_html.append('<h3>Previous round, for documentation only</h3>')
        ev_html.append(f'<p>The team’s published round-2 judge, T2b step 3057 (cell tag <code>{esc(ev["prev_round_tag"])}</code>, weights sha256 <code>734af65681ab6bb8</code>, per '
                       '<code>docs/CHECKPOINTS_AND_BASELINES_2026-09-14.md</code> §2.3.1). It was trained on a mix that was 0.2% code, and it is shown on the same item universe. '
                       'It is not an initialisation for r3_A; r3_A starts fresh from stock.</p>')
        prow = []
        for d in ev["datasets"]:
            for nm in ("judge", "judge_des32", "judge_des48"):
                c = ev["prev_round"][d["key"]][nm]
                if c["status"] == "pending":
                    prow.append(f'<tr><th scope="row">{esc(d["label"])}</th><td>{esc(ev["arm_labels"][nm])}</td><td class="num muted" colspan="5">no record in the team mirror</td></tr>'); continue
                v = c["vs_stock"]
                prow.append(f'<tr><th scope="row">{esc(d["label"])}</th><td>{esc(ev["arm_labels"][nm])}</td><td class="num">{c["n_scored"]}</td>'
                            f'<td class="num strong">{num(c["acc"])}</td>{dcell(v)}<td class="num">{num(c["exp_tok"], 0)}</td>{speed_cell(v)}</tr>')
        ev_html.append('<div class="tscroll"><table class="data"><thead><tr><th scope="col">dataset</th><th scope="col">arm</th><th scope="col" class="num">n</th>'
                       '<th scope="col" class="num">acc %</th><th scope="col" class="num">Δacc vs stock</th><th scope="col" class="num">exp/token</th>'
                       f'<th scope="col" class="num">speedup (exp)</th></tr></thead><tbody>{"".join(prow)}</tbody></table></div>')

    # ---------------- at a glance
    glance = []
    if ev:
        for s in ("step001400", "step000700"):
            have = [(d, nm) for d in ev["datasets"] for nm in ("judge", "judge_des32", "judge_des48")
                    if ev["cells"][d["key"]][s][nm]["vs_stock"] and ev["cells"][d["key"]][s][nm]["vs_stock"].get("d_acc") is not None]
            if have:
                break
        mcells = []
        for d in ev["datasets"]:
            dl = []
            for nm in ("judge", "judge_des32", "judge_des48"):
                c = ev["cells"][d["key"]]["step001400"][nm]; v = c["vs_stock"]
                if v and v.get("d_acc") is not None:
                    sig = v["ci"][0] > 0 or v["ci"][1] < 0
                    cls = ("pos" if v["d_acc"] > 0 else "neg") if sig else ""
                    dl.append(f'<dt>{esc(ev["arm_labels"][nm])}</dt><dd class="{cls}">{num(v["d_acc"], 1, True)} pts · n={v["n_common_scored"]}'
                              + (f' · {v["speedup_exp"]:.2f}×' if v.get("speedup_exp") else "") + "</dd>")
                else:
                    sp = f' · {v["speedup_exp"]:.2f}× exp on {v["n_common"]}' if v and v.get("speedup_exp") else ""
                    dl.append(f'<dt>{esc(ev["arm_labels"][nm])}</dt><dd class="muted">{STATUS_TXT[c["status"]]}' + (f' ({c["n_records"]} decoded{sp})' if c["n_records"] else "") + "</dd>")
            mcells.append(f'<div class="mcell"><h4>{esc(d["label"])}</h4><dl>{"".join(dl)}</dl></div>')
        glance.append('<p class="kick" style="margin:1.4rem 0 .2rem">Epoch 2 (step 1,400): Δacc vs stock, paired · speedup in expert loads per token</p><div class="matrix">' + "".join(mcells) + "</div>")

    bullets = [
        f"Training ran {last_step:,} of {PLANNED:,} planned steps ({ep_stop:.2f} epochs at {EPOCH:.1f} steps/epoch). "
        f"The user stopped it at {ist_s(tf['stop_time'])} once the holdout gains had flattened.",
        (f"Holdout accuracy of the judge on drafter-filled pass-1 blocks: {acc_first[1]:.1f}% at step {acc_first[0]:,} → {acc_last[1]:.1f}% at step {acc_last[0]:,}. "
         f"Stock sits at {base_acc:.1f}%." if acc_last[0] else "Holdout curves unavailable."),
        (f"Holdout accuracy rose {100 * (acc800 - dpacc[200]):+.1f} pts from step 200 to 800, then only {100 * (acc1400 - acc800):+.1f} pts from 800 to 1,400, "
         "a gain close to the ±~1 pt noise of 64 states. That is the flattening behind the stop."
         if acc800 is not None and acc1400 is not None and 200 in dpacc else ""),
        (f"Confident slots (p > 0.95): {cf_last[1]:.1f}% of slots at {cp_last[1]:.1f}% precision, against stock’s {base_cf:.1f}% at 100%. "
         "So the judge commits more than twice as many slots in one pass." if cf_last[0] and base_cf else ""),
    ]
    if resf:
        bullets.insert(1, f"Training resumed from step {RESUME:,} at {RESUMED_AT}; latest logged step {latest_step:,} "
                          f"({latest_step / EPOCH:.2f} epochs, last row {ist_s(to_ist(tf['last_utc']))}). Same recipe; it was stopped for good at step 2,000 (11:44 IST) when the calibrated r3_B retrain replaced it; every checkpoint is on HF.")
    import plateau_r3 as PL
    try:
        plateau_html, plateau_li = PL.render(sys.modules[__name__], tf, ev, specs)
    except Exception as e:                    # never lose the whole page to the diagnosis section
        import traceback; traceback.print_exc()
        plateau_html, plateau_li = f'<div class="placeholder">diagnosis failed to render: {esc(type(e).__name__)}: {esc(str(e)[:200])}</div>', ""
    glance_html = status_box(ev, tf, latest_step) + "<ul>" + "".join(f"<li>{esc(b)}</li>" for b in bullets if b) + plateau_li + "</ul>" + "".join(glance)

    # ---------------- static-ish sections
    mix = json.load(open(os.path.join(ROOT, "logs", "draftalign", "trainset_r3", "mix_manifest.json")))
    tot = mix["total_blocks"]
    pools = [("code", "nvidia/OpenCodeInstruct", mix["opencode"], "var(--p-code)", "program passes the row’s unit tests (74% of decodes)"),
             ("maths", "openai/gsm8k train", mix["gsm8k"], "var(--p-math)", "final answer correct (94%)"),
             ("verifiable IF", "if_pool (see below)", mix["if"], "var(--p-if)", "strict pass on every constraint (35.5% of decodes)")]
    split_bar = '<div class="split" role="img" aria-label="training blocks by pool">' + "".join(
        f'<span style="width:{100 * p[2]["blocks"] / tot:.2f}%;background:{p[3]}"></span>' for p in pools) + "</div>"
    pool_rows = "".join(
        f'<tr><th scope="row"><span class="sw" style="background:{p[3]};vertical-align:-1px"></span> {esc(p[0])}</th><td>{esc(p[1])}</td>'
        f'<td class="num">{p[2]["blocks"]:,}</td><td class="num">{100 * p[2]["share"]:.1f}%</td><td class="num">{p[2]["items"]:,}</td>'
        f'<td class="num">{p[2]["cap"]:,}</td><td>{esc(p[4])}</td></tr>' for p in pools)
    hold_bar = (f'<div class="split" role="img" aria-label="train vs holdout examples" style="height:18px">'
                f'<span style="width:{100 * TRAIN_EX / (TRAIN_EX + HOLD_EX):.2f}%;background:var(--s-stock)"></span>'
                f'<span style="width:{100 * HOLD_EX / (TRAIN_EX + HOLD_EX):.2f}%;background:var(--s-judge)"></span></div>')

    idea = f"""
<p class="lede">SDAR-30B-A3B writes a block of 32 slots at a time. Each pass it commits only the slots whose top token clears 0.95, so a block takes 8–12 full MoE forwards
(stock: 8.26 passes/blk on HumanEval, 11.93 on MBPP, 10.06 on GSM8K-500; <code>RESULTS.md</code> §1). The work is repeated passes over mostly-right guesses.</p>
<p>A 1.7B drafter with the same tokenizer can fill the whole block in 8 cheap steps. The big model should then only have to <em>judge</em> each drafted token: keep it,
or replace it. The stock model cannot do that. At a filled slot it echoes the next token (93.6% of slots), because SDAR is trained only on [MASK] slots. The in-distribution
workaround is sycophantic: it confirms 76% of the drafter’s wrong tokens at &gt; 0.95 and decodes at 67.0% against 95.0% (<code>docs/EXPLAINER_DRAFT_ALIGNED_TRAINING.md</code> §3).
Judging a draft is a skill the frozen model lacks. No threshold or vote can add it; training has to.</p>
<div class="callout"><p><strong>The judge.</strong> A LoRA on the attention projections, plus a learned 2,048-dim <em>draft flag</em> added to the input embedding of every tentative draft slot.
The LoRA delta applies only at flagged slots, and the router gates stay frozen. A forward with no flagged slot is therefore bit-identical to the stock model, and the step-0 identity check in
<code>train.log</code> confirms it: <code>{esc(tf["identity"] or "n/a")}</code>. At inference (<code>--judge full --draft s8</code>) the judge reads the draft-filled block and commits
the slots it is confident about. DES32/DES48 further restrict each forward to a 32- or 48-expert pool voted from the block, which cuts expert loads.</p></div>"""

    need = """
<p class="lede">The published round-2 judge (T2b step 3057) was fast, but its accuracy collapsed on program source.</p>
<div class="tscroll" style="max-width:34rem"><table class="data"><thead><tr><th scope="col">eval set</th><th scope="col" class="num">share of T2b training</th><th scope="col" class="num">Δacc (judge)</th></tr></thead><tbody>
<tr><th scope="row">GSM8K-500</th><td class="num">62.7%</td><td class="num">−1.4</td></tr>
<tr><th scope="row">MATH-500</th><td class="num">37.0%</td><td class="num">−4.4</td></tr>
<tr><th scope="row">MBPP</th><td class="num">0.2%</td><td class="num neg">−14.0</td></tr>
<tr><th scope="row">HumanEval</th><td class="num">never seen</td><td class="num neg">−13.4</td></tr>
<tr><th scope="row">GPQA-Diamond</th><td class="num">never seen</td><td class="num">−4.5</td></tr>
<tr><th scope="row">AIME</th><td class="num">never seen</td><td class="num">−1.1</td></tr></tbody></table></div>
<p>Training share does not predict the damage; <strong>being code does</strong> (<code>docs/TRAINING_RECIPE_2026-09-09.md</code> §1.3). The adapter is not learning Python.
It is learning a decision rule, <em>when to trust the drafter</em>, and that rule needs negatives where deferring is catastrophic (code), positives where it is safe
(maths, instruction following), and enough epochs on a small set (§1.2: T2a at 2.07 epochs beat T2b at 1.09 epochs over 3.9× the data at <code>c90_o99</code>).</p>
<ul><li><strong>Fresh from stock</strong>, not from 1057 or 3057, so the “defer” prior learnt on GSM8K-only data is never installed (§4.1).</li>
<li><strong>Code-heavy mix</strong> 35 / 35 / 30 (code / GSM8K / verifiable IF), every row re-decoded by the stock pipeline.</li>
<li><strong>Recipe-exact batch</strong> 8 GPUs × grad-accum 4 = 32 states per step (T2b had changed six things at once).</li>
<li><strong>Checkpoint every 100 steps</strong>, so the early-vs-late question round 2 left open is settled by construction.</li></ul>"""

    data = f"""
<p class="lede">{tot:,} training blocks (32-token blocks of correct decodes) from {mix["items"]:,} items. The decodes were made by the training-free blind rule
(<code>--draft s8 --judge blind --tau-acc-q .50</code>, stock 30B + stock 1.7B), and only items whose output was verified correct are kept.</p>
{split_bar}
<div class="tscroll"><table class="data"><thead><tr><th scope="col">pool</th><th scope="col">source</th><th scope="col" class="num">blocks</th><th scope="col" class="num">share</th>
<th scope="col" class="num">items</th><th scope="col" class="num">cap</th><th scope="col">a decode is admitted when</th></tr></thead><tbody>{pool_rows}
<tr class="grp"><th scope="row" colspan="2">total</th><td class="num">{tot:,}</td><td class="num">100%</td><td class="num">{mix["items"]:,}</td><td class="num">24,000</td><td>index sha256 <code>{mix["index_sha256"][:8]}…</code></td></tr></tbody></table></div>
<h3>Train / holdout split</h3>
<p>The trainer holds out whole items, not blocks: <strong>150 items → {HOLD_EX:,} examples</strong> for the holdout, <strong>{TRAIN_EX:,} examples</strong> for training
(<code>train.log</code>: <code>{esc(tf["data_line"] or "")}</code>). At {STATES} states per step, <strong>one epoch is {EPOCH:.1f} steps</strong>.</p>
{hold_bar}
<ul class="legend"><li><span class="sw" style="background:var(--s-stock)"></span>train {TRAIN_EX:,} (96.5%)</li><li><span class="sw" style="background:var(--s-judge)"></span>holdout {HOLD_EX:,} (3.5%)</li></ul>
<h3>What to know about the pools</h3>
<ul>
<li><strong>Maths is GSM8K only.</strong> The recipe had MATH + GSM8K; dropping MATH is a flagged deviation (recipe Amendment A1). MATH-500 in the eval is therefore out-of-domain.</li>
<li><strong>Verifiable IF</strong> (<code>code/configs/if_pool.json</code>, 6,000 items): 5,015 from <code>nvidia/Nemotron-RL-instruction_following</code> (WildChat prompts, IFEvalG ids), 731 from
<code>allenai/IF_multi_constraints_upto5</code> and 254 from <code>allenai/RLVR-IFeval</code>. Each has 1–3 constraints, checked strictly by the vendored open-instruct IFEvalG checker @172e379. The Nemotron-RL top-up
was added after the user approved “allenai verifiable IF”, and it contains some edgy WildChat prompts (a content audit was requested).</li>
<li><strong>Code</strong> comes from the frozen 12,000-row OpenCodeInstruct pool, decoded without scoring and scored post hoc from a sandbox cwd, because generated programs can write files.</li>
<li>The caps were 8,400 / 8,400 / 7,200. Code and IF ran out of admitted blocks first, which is why the shares land at 35.2 / 35.7 / 29.1.</li></ul>"""

    trn = tf["trainable"]
    method = f"""
<div class="tscroll" style="max-width:46rem"><table class="data"><tbody>
<tr><th scope="row">base</th><td><code>JetLM/SDAR-30B-A3B-Chat-b32</code> @ <code>{cfg["model_revision"][:8]}</code> (128 experts × 48 layers, 8 active), block 32</td></tr>
<tr><th scope="row">drafter</th><td><code>JetLM/SDAR-1.7B-Chat-b32</code>, stock, frozen</td></tr>
<tr><th scope="row">adapter</th><td>LoRA r {cfg["lora_r"]} / α {cfg["alpha"]:.0f} / dropout {cfg["lora_dropout"]} on q, k, v, o × 48 layers + draft flag; gates frozen (<code>train_gates={cfg["train_gates"]}</code>); trainable <code>{esc(trn or "")}</code></td></tr>
<tr><th scope="row">optimiser</th><td>AdamW, lr {cfg["lr_lora"]:.0e}, weight decay {cfg["weight_decay"]}, warmup {cfg["warmup"]}, cosine to {cfg["min_lr_frac"]}× ({cfg["lr_lora"] * cfg["min_lr_frac"]:.0e}), grad clip {cfg["clip"]}</td></tr>
<tr><th scope="row">batch</th><td>{cfg["world_size"]} GPUs × grad-accum {cfg["grad_accum"]} = {cfg["world_size"] * cfg["grad_accum"]} states/step, one full bf16 model per rank, manual all-reduce</td></tr>
<tr><th scope="row">schedule</th><td>{cfg["steps"]:,} steps planned (3.06 epochs); stopped at {last_step:,} ({ep_stop:.2f} epochs); checkpoint every {cfg["save_every"]}, holdout eval every {cfg["eval_every"]} on {cfg["eval_states"]} states</td></tr>
<tr><th scope="row">states</th><td>p_pass1 {cfg["p_pass1"]}, p_draftfull {cfg["p_draftfull"]}, p_allmask {cfg["p_allmask"]}; max seq {cfg["max_seq"]}; expert pool during training {cfg["pool_Ms"]} (none)</td></tr>
<tr><th scope="row">wall clock</th><td>launched {ist_s(tf["launch"], "%d %b %H:%M IST")}; median {tf["sec_per_step_med"] or 0:.1f} s/step; last logged step {(tf["pre"][-1]["step"] if tf["pre"] else last_step):,} at {ist_s(to_ist(tf["pre"][-1].get("utc") if tf["pre"] else tf["last_utc"]))}; stopped (SIGTERM) {ist_s(tf["stop_time"])}{f"; resumed from step {RESUME:,} at {RESUMED_AT}, latest step {latest_step:,}" if resf else ""}</td></tr>
<tr><th scope="row">config hash</th><td><code>3b7ca2dc900a1b53</code> · <code>logs/draftalign/train/r3_A/config.json</code> · launcher <code>code/draftalign/launch_r3_train.sh</code></td></tr>
</tbody></table></div>
<h3>One training example</h3>
<p>An example is one (item, block) pair. Its <em>state</em> is re-drawn every time the example is visited (<code>code/draftalign/dataset.py</code>):</p>
<ul><li>with probability {cfg["p_pass1"]}, nothing is committed: the pass-1 state the judge meets at inference;</li>
<li>otherwise a recorded denoising pass is drawn and everything the stock trajectory had committed before it is fixed;</li>
<li>open slots hold the drafter’s token with probability 1 − m and [MASK] with probability m, where m ~ U(0, 1); with probability {cfg["p_draftfull"]}, every open slot holds the draft (the exact inference regime); all-mask states are off ({cfg["p_allmask"]});</li>
<li>the prefix is clean: the prompt plus the final tokens of earlier blocks, exactly what the KV cache holds at inference.</li></ul>
<h3>The loss</h3>
<div class="eq">L = (1/|O|) Σ_(i∈O) −log p_i(T_i)                       cross-entropy on every open slot
  + β · (1/|C|) Σ_(i∈C) H(softmax(z_i / τ))                certainty forcing, only where already right
C = {{ i ∈ O : argmax p_i = T_i }},  β = {cfg["beta"]},  τ = {cfg["temp"]},  [MASK] column removed from H</div>
<p><strong>Targets T are the final tokens of the stock pipeline’s own correct decodes, not ground-truth text.</strong> The judge learns to reproduce what stock would
have written, in fewer passes. The certainty term (dParallel) pushes already-right slots past the 0.95 commit threshold.</p>"""

    wb_state = wb.get("state", "unknown")
    curves = f"""
<p class="lede">W&amp;B run <a href="{WANDB_URL}"><code>{WANDB_PATH}</code></a>, pulled {esc(ist_s(to_ist(wb.get("pulled_utc")), "%d %b %H:%M IST")) if wb.get("pulled_utc") else "from the local log"}
(source: {esc(wb.get("source", ""))}; run state as read: <code>{esc(wb_state)}</code>, it read crashed between the 08:16 IST SIGTERM and the 09:53 IST resume, which logs into the same run). {xcheck}</p>
{t_legend}<div class="grid2">{"".join(tfigs)}</div>"""

    holdout = f"""
<p class="lede">Every 200 steps the trainer scores {cfg["eval_states"]} holdout states in two regimes at a 0.95 threshold. With 64 states, differences of about one point are noise.</p>
<ul><li><code>draft_pass1</code>: the judge on a pass-1 block filled by the drafter. This is the quantity training moves.</li>
<li><code>allmask_pass1</code>: the same model on an all-[MASK] block. No slot is flagged, so the adapter is inert and this line <strong>is</strong> the stock model, bit for bit. It is flat by design, a baseline and not a failure to learn.</li>
<li><strong>Confident fraction</strong> is the share of slots whose top-1 probability exceeds 0.95; <strong>confident precision</strong> is accuracy on those slots. Accuracy is measured against the stock decode’s final tokens.</li></ul>
{h_legend}<div class="grid2">{"".join(efigs)}</div>"""

    read = """
<h3>How to read the numbers</h3>
<div class="eq">passes/blk = Σ(n_forward − n_prefill − n_flush) / Σ n_blocks
exp/token  = Σ unique_expert_layer_loads / Σ n_gen_to_eos
acc        = 100 · Σ correct / n              (vendored OpenCompass evaluators, per item)
Δacc       = paired on shared idx; bootstrap 95% CI, 2,000 resamples
speedup    = stock exp/token ÷ arm exp/token, on the same idx</div>
<p>All are block- or token-weighted sums over items (<code>AGENTS.md</code> §8), not means of per-item means. exp/token counts distinct (layer, expert) pairs that actually ran, so under a pool it counts the restricted set.
It does not charge the drafter’s own forwards: at 364.64 expert-equivalents each, those erase most of the blind rule’s saving (<code>RESULTS.md</code> §2.1).</p>
<h3>Caveats</h3>
<ul><li><strong>Small subsets.</strong> GSM8K is the first 100 of 500 and MATH the first 50. On 50–164 items a single flipped item moves accuracy by 0.6–2 pts; read the CIs, not the point estimates.</li>
<li><strong>Uneven schedule.</strong> Training stopped at step 1,560 (2.2 epochs) at 08:16 IST and resumed from <code>step001500</code> at 09:53 IST with the same recipe and the same 2,175-step cosine. It keeps running on that schedule until the server shuts down at 12:20 IST, with a checkpoint every 100 steps pushed to HF (step 2,000 ≈11:45 IST, step 2,100 ≈12:07 IST). Checkpoints after step 1,400 have no downstream eval. Steps 1,510–1,560 were trained twice; the charts keep the resumed run’s rows.</li>
<li><strong>Targets are stock decodes.</strong> The best a perfect judge can do is match stock accuracy at fewer passes, so a positive Δacc is noise or luck, not a skill the objective teaches.</li>
<li><strong>Two stock sources.</strong> Stock rows mix the team’s block-32 records with this wave’s new stock decodes (HumanEval DES32/48 and GSM8K plain stock). Both run the same code and model revision, and every comparison is paired on idx.</li>
<li><strong>Previous round is documentation only.</strong> T2b step 3057 had a different mix, batch and initialisation; it is not an ablation of r3_A.</li>
<li><strong>Scoring verified.</strong> An earlier socket-path bug in the code-set scorer is fixed. The fixed sandboxed scorer re-scored four older team cells (HumanEval and MBPP; stock and the previous round’s judge + DES48) with 0 item mismatches on 842 items. Decode config is identical to the team’s older runs: model revision, s8 drafter, full judge, τ 0.95, DES pool rule and splits. A scored record with <code>score_error</code> would still be treated as invalid, not incorrect.</li></ul>
<h3>Next steps</h3>
<ul><li>No further downstream eval is scheduled before the 12:20 IST shutdown. Steps 2,000 and 2,100 (and any later save) go to the public HF backup, so their HumanEval/MBPP evals can run elsewhere with the identical decode config.</li>
<li>If epoch 2 beats epoch 1 on code, report <code>step001400</code> with DES48 as the headline arm; if they tie, prefer the earlier and cheaper checkpoint.</li>
<li>Before the server dies at 12:20 IST: the probe records, <code>eval_summary.json</code> and this page go to the public HF backup (<code>Nemotron-split-moe-train/r3_A/evals/</code>) and the GitHub branch.</li></ul>"""

    prov = f"""
<ul><li>W&amp;B: <a href="{WANDB_URL}">{WANDB_URL}</a></li>
<li>Public HF backup: <a href="{HF_REPO}">Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts</a></li>
<li>Checkpoints, train logs and evals: <a href="{HF}/Nemotron-split-moe-train">…/Nemotron-split-moe-train</a>. Checkpoints are at <code>r3_A/checkpoints/stepNNNNNN</code> (step000100–step001500), logs at <code>r3_A/train_logs/</code>, evals at <code>r3_A/evals/</code></li>
<li>Data (unprocessed, processed, strategy; no checkpoints here): <a href="{HF}/Nemotron-data-subset-training-diffusion-moe">…/Nemotron-data-subset-training-diffusion-moe</a></li>
<li>Code and docs: GitHub branch <a href="{GH_BRANCH}"><code>round3-2026-09-14</code></a></li>
<li>Eval records: <code>logs/draftalign/probes_r3/r3_A/&lt;step&gt;/&lt;ds&gt;/&lt;arm&gt;/</code>; team stock mirror <code>/dev/shm/dlm/artifacts_mirror/logs/draftalign/da_decode/</code></li>
<li>Tables as markdown: <code>logs/draftalign/probes_r3/r3_A/eval_summary.md</code></li></ul>
<div class="eq">set -a; . ./.env; set +a
/venv/r3/bin/python code/draftalign/eval_summary_r3.py          # tables only
/venv/r3/bin/python docs/report/build_r3_report.py              # tables + this page (W&amp;B from cache)
/venv/r3/bin/python docs/report/build_r3_report.py --refresh    # also re-pull W&amp;B history</div>"""

    now = dt.datetime.now(IST)
    evB = json.load(open(RB.EVAL_JSON)) if os.path.exists(RB.EVAL_JSON) else None
    try:
        r3b_html, r3b_assets = RB.render(sys.modules[__name__], specs, evB)
    except Exception as e:                    # never lose the whole page to the new part
        import traceback; traceback.print_exc()
        r3b_html, r3b_assets = f'<div class="placeholder">r3_B part failed to render: {esc(type(e).__name__)}: {esc(str(e)[:200])}</div>', []
    rb_last = max(RB.load_rows(RB.TRAIN_B)[0] or {0: 0})
    rb_head = "—"
    if evB:
        hv_ = evB["datasets"]["humaneval"]["rows"]["r3B_judge_des48"].get("vs_stock")
        gv_ = evB["datasets"]["gsm8k500"]["rows"]["r3B_judge_des48"].get("vs_stock")
        rb_head = " · ".join(f"{lab} {num(v['d_acc'], 1, True)} vs stock at {v['speedup_exp']:.2f}×" for lab, v in (("HumanEval", hv_), ("GSM8K", gv_)) if v and v.get("d_acc") is not None)
    cov = ev["coverage"] if ev else None
    last_ist = to_ist(tf["last_utc"]); live = bool(last_ist and (now - last_ist).total_seconds() < 900)
    facts = [("runs", f'<a href="{RB.WB_B}">r3_B</a> (calibrated, step {rb_last:,} = epoch 2, done 14:31 IST) · <a href="{WANDB_URL}">r3_A</a> (step {latest_step:,}, stopped 11:44 IST)'),
             ("headline arm", f'r3_B/step000710 judge + DES48 · {rb_head}'),
             ("stopped at", f"step {last_step:,} of {PLANNED:,} · {ep_stop:.2f} epochs" + (f" · resumed, stopped for good at step {latest_step:,}" if resf else "")),
             ("stop decisions", f"{ist_s(tf['stop_time'], '%d %b %H:%M IST')} holdout flat · 15 Sep 11:44 IST recipe change (r3_B)"),
             ("training data", f"{tot:,} blocks · 35 / 35 / 29 code/GSM8K/IF"),
             ("holdout acc, judge", f"{acc_last[1]:.1f}% vs stock {base_acc:.1f}%" if acc_last[0] and base_acc else "—"),
             ("eval cells", f"{cov['complete']} complete · {cov['unscored'] + cov['partial']} in progress · {cov.get('scheduled', 0)} scheduled · {cov.get('dropped', 0)} dropped" if cov else "not run"),
             ("page generated", now.strftime("%d %b %Y %H:%M IST"))]
    facts_html = '<dl class="facts">' + "".join(f"<div><dt>{esc(k)}</dt><dd>{v}</dd></div>" for k, v in facts) + "</dl>"

    head = ('<title>SDAR Round-3 Judge</title>\n'
            '<meta name="description" content="Round-3 draft-aligned judge LoRA for SDAR-30B-A3B at block 32: idea, data, training, holdout and downstream evals.">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">\n'
            f"<style>{CSS}{RB.CSS_EXTRA}</style>\n")
    page = [head, '<div class="wrap">',
            '<header class="mast"><p class="kick">ICLR 2027 · draft-aligned decoding · round 3 · SDAR-30B-A3B-Chat-b32 + SDAR-1.7B drafter</p>',
            '<h1>A draft-aligned judge, trained on code</h1>',
            '<p class="deck">Round 3 retrains the judge that decides when SDAR-30B-A3B should keep a small drafter’s tokens. Round 3a (r3_A) starts fresh from stock on a code-heavy mix '
            'to repair the 13–14 point code collapse of the round-2 judge; round 3b (r3_B) retrains one epoch from r3_A’s epoch-1 weights with a calibrated objective. '
            'Both parts regenerate from the training logs and the eval records on disk.</p>',
            facts_html, "</header>",
            r3b_html,
            sec("summary", "What the r3_A report showed", glance_html, "summary"),
            sec("idea", "Why a judge, and what it is", idea, "idea"),
            sec("need", "Why round 3 exists", need, "need"),
            sec("data", "The data and the split", data, "data"),
            sec("method", "How it was trained", method, "method"),
            sec("training", "Training curves", curves, "training"),
            sec("holdout", "Holdout during training", holdout, "holdout"),
            sec("diagnosis", "Why the curves flattened", plateau_html, "plateau"),
            sec("evals", "Downstream evals", "".join(ev_html), "evals"),
            sec("reading", "How to read, caveats, next steps", read, "reading"),
            sec("provenance", "Where everything is", prov, "provenance"),
            f'<footer>Generated {now.strftime("%Y-%m-%d %H:%M IST")} by <code>docs/report/build_r3_report.py</code>. All times IST (UTC+5:30).</footer>',
            "</div>",
            f'<script type="application/json" id="chart-data">{json.dumps([s for s in specs if s]).replace("</", "<\\/")}</script>',
            f"<script>{JS}</script>"]
    return "\n".join(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-pull W&B history into the cache")
    ap.add_argument("--no-eval", action="store_true", help="do not re-run eval_summary_r3.py; read the existing json")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    wb = get_wandb(args.refresh)
    tf = train_facts()
    ev = run_eval_summary(args.no_eval)
    if not args.no_eval:
        p = subprocess.run([sys.executable, RB.EVAL_SCRIPT, "--step", RB.EVAL_STEP], capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[-400:])
    page = render(wb, tf, ev)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(page)
    os.replace(tmp, args.out)
    print(f"[report] {len(page) / 1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
