"""Round 3b (r3_B, calibrated retrain) part of docs/report/r3_report.html. Imported by build_r3_report.py.

Reads only: logs/draftalign/train/{r3_A,r3_B}/train.rank0.jsonl, logs/draftalign/train/r3_B/calib/*.json|png,
logs/draftalign/plateau_probe_r3/*.json, logs/draftalign/probes_r3/r3_B/eval_summary.json.
Writes only: docs/report/r3_assets/*.png (copies of the latest calibration PNGs).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "r3_assets")
TRAIN_A = os.path.join(ROOT, "logs", "draftalign", "train", "r3_A")
TRAIN_B = os.path.join(ROOT, "logs", "draftalign", "train", "r3_B")
CALIB = os.path.join(TRAIN_B, "calib")
PROBE = os.path.join(ROOT, "logs", "draftalign", "plateau_probe_r3")
EVAL_JSON = os.path.join(ROOT, "logs", "draftalign", "probes_r3", "r3_B", "eval_summary.json")
EVAL_SCRIPT = os.path.join(ROOT, "code", "draftalign", "eval_summary_r3B.py")
EVAL_STEP = "step000710"
STEPS_PER_EPOCH = 22734 / 32          # 710.44
B_OFFSET = 1.0                        # r3_B starts from r3_A step000700 = epoch 1

HF = "https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train"
GH = "https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert/tree/round3-2026-09-14"
WB_B = "https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_B"
WB_A = "https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign/runs/r3_A"

FOCUS = ["stock_des48", "r3B_judge_des48", "prev_judge_des48", "r3A_e1_judge_des48", "r3A_e2_judge_des48", "r3A_s2000_judge_des48"]
COV_ROWS = ["stock", "stock_des48", "r3B_judge_des48", "prev_judge_des48", "r3A_e1_judge_des48", "r3A_e2_judge_des48",
            "r3A_s2000_judge_des48", "stock_des32", "blind", "r3B_judge"]
ARM_COLOR = {"stock_des48": "var(--s-stock)", "r3B_judge_des48": "var(--r3b)", "prev_judge_des48": "var(--s-prev)",
             "r3A_e1_judge_des48": "var(--r3a-light)", "r3A_e2_judge_des48": "var(--r3a)", "r3A_s2000_judge_des48": "var(--r3a-dark)"}
MINUS = "−"


# ------------------------------------------------------------------------------------------------ data
def load_rows(d):
    tr, ev = {}, {}
    for line in open(os.path.join(d, "train.rank0.jsonl")):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "eval":
            ev[int(r["step"])] = r
        elif "loss" in r:
            tr[int(r["step"])] = r          # later duplicate (after a restart) wins, like the trainer's own resume
    return dict(sorted(tr.items())), dict(sorted(ev.items()))


def ser(rows, key, offset, fn=None):
    out = []
    for s, r in rows.items():
        v = fn(r) if fn else r.get(key)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out.append((offset + s / STEPS_PER_EPOCH, float(v)))
    return out


def ema(pts, w=0.8):
    out, s = [], 0.0
    for t, (x, y) in enumerate(pts, 1):
        s = w * s + (1 - w) * y
        out.append((x, s / (1 - w ** t)))
    return out


def train_meta():
    cfg = json.load(open(os.path.join(TRAIN_B, "config.json")))["config"]
    log = open(os.path.join(TRAIN_B, "train.log"), errors="replace").read()
    init = re.search(r"\[init-adapter\] (\S+): (\d+) tensors", log)
    resume = re.search(r"\[resume\] from step (\d+)", log)
    done = os.path.join(TRAIN_B, "done")
    return {"cfg": cfg, "init": init.group(1) if init else None, "init_n": init.group(2) if init else None,
            "resume": int(resume.group(1)) if resume else None, "done": os.path.exists(done),
            "n_saves": len(re.findall(r"\[save\] ", log))}


def latest_calib():
    steps = sorted({os.path.basename(p)[:10] for p in glob.glob(os.path.join(CALIB, "step*.json"))})
    full = [s for s in steps if all(os.path.exists(os.path.join(CALIB, f"{s}_{k}.png"))
                                    for k in ("ece_nll_vs_t", "reliability_grid", "joint_calibration", "train_confidence"))]
    return steps, (full[-1] if full else (steps[-1] if steps else None))


def copy_assets(step):
    """Copy the latest step's four PNGs + trend into docs/report/r3_assets/; drop stale step PNGs. Returns [(name, published path)]."""
    os.makedirs(ASSETS, exist_ok=True)
    want = [f"{step}_{k}.png" for k in ("ece_nll_vs_t", "reliability_grid", "joint_calibration", "train_confidence")] + ["trend_calibration.png"]
    out = []
    for n in want:
        src = os.path.join(CALIB, n)
        if os.path.exists(src):
            dst = os.path.join(ASSETS, n)
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst)
            out.append(n)
    for p in glob.glob(os.path.join(ASSETS, "step*.png")):
        if os.path.basename(p) not in want:
            os.remove(p)
    return out


def plateau_precision():
    """Confident-override precision from the r3_A plateau probe: holdout_pass1|{all,opencode}|draft_bad."""
    out = {}
    for p in sorted(glob.glob(os.path.join(PROBE, "step*.json"))):
        g = json.load(open(p)).get("groups", {})
        a = g.get("holdout_pass1|all|draft_bad", {}); c = g.get("holdout_pass1|opencode|draft_bad", {}); ok = g.get("holdout_pass1|all|draft_ok", {})
        out[os.path.basename(p)[:-5]] = (a.get("precision"), c.get("precision"), ok.get("precision"), a.get("commit_share"))
    return out


# ------------------------------------------------------------------------------------------------ charts
W, H = 480, 236
ML, MR, MT, MB = 50, 18, 30, 30
PW, PH = W - ML - MR, H - MT - MB


def fmt(v, dec):
    return f"{v:,.{dec}f}".replace("-", MINUS)


def epoch_chart(Hm, cid, label, sers, *, xlo, xhi, ylo=None, yhi=None, scale=1.0, dec=3, suffix="", cap100=False, marks=()):
    """Same anatomy as build_r3_report.line_chart but x is epochs. sers: [{name, pts, cls, color, dash?}]."""
    esc = Hm.esc
    ys = [y * scale for s in sers for _, y in s["pts"]]
    if not ys:
        return f'<div class="chart-empty">no data for {esc(label)}</div>', None
    lo = min(ys) if ylo is None else ylo; hi = max(ys) if yhi is None else yhi
    pad = (hi - lo) * 0.08 or abs(hi) * 0.05 or 1
    ticks, step = Hm.nice_ticks(lo - (pad if ylo is None else 0), hi + (pad if yhi is None else 0))
    y0, y1 = ticks[0], ticks[-1]
    xs = lambda x: ML + PW * (x - xlo) / (xhi - xlo)
    yv = lambda y: MT + PH * (1 - (y - y0) / (y1 - y0))
    g = [f'<svg class="chart" id="{cid}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}">',
         f'<defs><clipPath id="{cid}-c"><rect x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></clipPath></defs>']
    for t in ticks:
        if cap100 and t > 100 + 1e-9:
            continue
        g.append(f'<line class="grid" x1="{ML}" x2="{ML + PW}" y1="{yv(t):.1f}" y2="{yv(t):.1f}"/>')
        g.append(f'<text class="tick" x="{ML - 7}" y="{yv(t) + 4:.1f}" text-anchor="end">{Hm.tick_label(t, step)}</text>')
    xt = 0.5 if xhi - xlo > 1.2 else 0.25
    t = math.ceil(xlo / xt - 1e-9) * xt
    while t <= xhi + 1e-9:
        g.append(f'<text class="tick" x="{xs(t):.1f}" y="{H - 10}" text-anchor="middle">{t:g}</text>'); t += xt
    g.append(f'<text class="axlab" x="{ML + PW}" y="{H - 10}" text-anchor="start" dx="4">ep</text>')
    g.append(f'<line class="axis" x1="{ML}" x2="{ML + PW}" y1="{MT + PH}" y2="{MT + PH}"/>')
    for e in (1, 2):
        if xlo < e < xhi:
            g.append(f'<line class="ev" x1="{xs(e):.1f}" x2="{xs(e):.1f}" y1="{MT}" y2="{MT + PH}"/>'
                     f'<text class="evlab" x="{xs(e):.1f}" y="{MT - 17}" text-anchor="middle">epoch {e}</text>')
    for x, lab, cls in marks:
        if xlo < x < xhi:
            g.append(f'<line class="ev {cls}" x1="{xs(x):.1f}" x2="{xs(x):.1f}" y1="{MT}" y2="{MT + PH}"/>'
                     f'<text class="evlab" x="{xs(x) + 3:.1f}" y="{MT - 5}" text-anchor="start">{esc(lab)}</text>')
    g.append(f'<g clip-path="url(#{cid}-c)">')
    for s in sers:
        pts = [(x, y) for x, y in s["pts"] if xlo - 0.05 <= x <= xhi + 0.05]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs(x):.1f},{yv(y * scale):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        g.append(f'<path class="ln {s["cls"]}" style="stroke:{s["color"]}"{dash} d="{d}"/>')
    g.append("</g>")
    for s in sers:
        if s["cls"] == "eval":
            for x, y in s["pts"]:
                if xlo <= x <= xhi:
                    g.append(f'<circle class="dot" cx="{xs(x):.1f}" cy="{yv(y * scale):.1f}" r="4" style="fill:{s["color"]}"/>')
    g.append(f'<line class="xhair" x1="0" x2="0" y1="{MT}" y2="{MT + PH}" visibility="hidden"/><g class="hdots"></g>')
    g.append(f'<rect class="hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></svg>')
    spec = {"id": cid, "xd": [xlo, xhi], "yd": [y0, y1], "box": [ML, MT, PW, PH], "epoch": 1, "xunit": "epoch",
            "series": [{"name": s["name"], "color": s["color"], "dec": dec, "suffix": suffix,
                        "pts": [[round(x, 3), round(y * scale, 4)] for x, y in s["pts"] if xlo - 0.05 <= x <= xhi + 0.05]}
                       for s in sers if s.get("tip", True)]}
    return "".join(g), spec


def tradeoff_chart(Hm, cid, label, pts, xr, yr):
    """pts: [{x speedup, y d_acc, lo, hi, color, tip, partial}] with CI whiskers; shared axes across datasets."""
    esc = Hm.esc
    sw, sh, ml, mb, mt, mr = 480, 250, 46, 40, 14, 18
    pw, ph = sw - ml - mr, sh - mt - mb
    xt, xstep = Hm.nice_ticks(xr[0], xr[1], 4); yt, ystep = Hm.nice_ticks(yr[0], yr[1], 4)
    X = lambda v: ml + pw * (v - xt[0]) / (xt[-1] - xt[0]); Y = lambda v: mt + ph * (1 - (v - yt[0]) / (yt[-1] - yt[0]))
    g = [f'<svg class="chart" id="{cid}" viewBox="0 0 {sw} {sh}" role="img" aria-label="{esc(label)}">']
    for t in yt:
        g.append(f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{ml - 7}" y="{Y(t) + 4:.1f}" text-anchor="end">{Hm.tick_label(t, ystep)}</text>')
    for t in xt:
        g.append(f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 17}" text-anchor="middle">{Hm.tick_label(t, xstep)}×</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    g.append(f'<text class="axlab" x="{ml + pw}" y="{sh - 6}" text-anchor="end">speedup vs stock (expert loads per token)</text>')
    g.append(f'<text class="axlab" x="{ml}" y="{sh - 6}" text-anchor="start">Δacc vs stock, pts</text>')
    for p in pts:
        x = X(p["x"]); y = Y(p["y"])
        if p["lo"] is not None:
            g.append(f'<line class="whisk" style="stroke:{p["color"]}" x1="{x:.1f}" x2="{x:.1f}" y1="{Y(p["hi"]):.1f}" y2="{Y(p["lo"]):.1f}"/>')
        g.append(Hm.mark("circle", x, y, p["color"], p.get("partial", False), p["tip"], r=5.5))
    g.append("</svg>")
    return "".join(g)


# ------------------------------------------------------------------------------------------------ tables
def sig(ci):
    return bool(ci) and (ci[0] > 0 or ci[1] < 0)


def dcell(Hm, v, key="vs_stock"):
    if not v or v.get("d_acc") is None:
        return '<td class="num muted">—</td>'
    d, ci = v["d_acc"], v["ci"]; s = sig(ci)
    cls = ("pos" if d > 0 else "neg") if s else ""
    icon = ("▲ " if d > 0 else "▼ ") if s else ""
    return (f'<td class="num {cls}">{icon}{Hm.num(d, 1, True)}<span class="ci"> [{Hm.num(ci[0], 1, True)}, {Hm.num(ci[1], 1, True)}]</span>'
            f'<span class="nn">n={v["n_common_scored"]}</span></td>')


def cov_cell(c, n):
    st = c["status"]; ns = c["n_scored"]
    if st == "complete":
        return f'<td class="cv cv-ok">{ns}/{n}</td>'
    if st == "partial":
        return f'<td class="cv cv-part">{ns}/{n}<span class="nn">partial</span></td>'
    if c["n_records"]:
        return f'<td class="cv cv-part">{c["n_records"]} decoded<span class="nn">unscored</span></td>'
    return f'<td class="cv cv-pend">0/{n}<span class="nn">pending</span></td>'


def coverage_matrix(Hm, ev):
    esc = Hm.esc; ds = ev["datasets"]
    head = "".join(f'<th scope="col">{esc(d["label"])}<span class="nn">n={d["n"]}</span></th>' for d in ds.values())
    rows = []
    for i, a in enumerate(COV_ROWS):
        if a == "stock_des32":
            rows.append(f'<tr class="grp"><th colspan="{len(ds) + 1}" scope="rowgroup">secondary arms</th></tr>')
        cells = "".join(cov_cell(d["rows"][a], d["n"]) for d in ds.values())
        rows.append(f'<tr><th scope="row">{esc(ev["labels"][a])}</th>{cells}</tr>')
    return f'<div class="tscroll"><table class="data cov2"><thead><tr><th scope="col">arm</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def headline_tables(Hm, ev):
    esc, num = Hm.esc, Hm.num
    body = []
    for k, d in ev["datasets"].items():
        body.append(f'<tr class="grp"><th colspan="7" scope="rowgroup">{esc(d["label"])} <span class="muted">· {d["n"]} items</span></th></tr>')
        st = d["rows"]["stock"]
        body.append(f'<tr><th scope="row">{esc(ev["labels"]["stock"])}</th><td class="num strong">{num(st["acc"])}</td><td class="num muted">reference</td>'
                    f'<td class="num muted">—</td><td class="num">1.00×</td><td class="num">{num(st["passes_blk"], 2)}</td><td class="num">{st["n_scored"]}</td></tr>')
        for a in FOCUS:
            c = d["rows"][a]
            if c["status"] == "pending" and not c["n_records"]:
                if a == "r3A_s2000_judge_des48":
                    continue
                body.append(f'<tr><th scope="row">{esc(ev["labels"][a])}</th><td class="num muted" colspan="6">pending</td></tr>'); continue
            v = c.get("vs_stock"); v48 = c.get("vs_stock_des48")
            part = f'<span class="chip c-warn">partial {c["n_scored"]}/{c["n_planned"]}</span> ' if c["status"] == "partial" else ""
            sp = f'{v["speedup_exp"]:.2f}×<span class="nn">n={v["n_common"]}</span>' if v and v.get("speedup_exp") else "—"
            body.append(f'<tr><th scope="row">{part}{esc(ev["labels"][a])}</th><td class="num strong">{num(c["acc"])}</td>{dcell(Hm, v)}'
                        + ('<td class="num muted">reference</td>' if a == "stock_des48" else dcell(Hm, v48))
                        + f'<td class="num">{sp}</td><td class="num">{num(c["passes_blk"], 2)}</td><td class="num">{c["n_scored"]}<span class="nn">of {c["n_planned"]}</span></td></tr>')
    head = ('<thead><tr><th scope="col">arm</th><th scope="col" class="num">acc %</th><th scope="col" class="num">Δ vs stock [95% CI]</th>'
            '<th scope="col" class="num">Δ vs stock + DES48 [CI]</th><th scope="col" class="num">speedup (exp/token)</th>'
            '<th scope="col" class="num">passes/blk</th><th scope="col" class="num">n scored</th></tr></thead>')
    return f'<div class="tscroll"><table class="data">{head}<tbody>{"".join(body)}</tbody></table></div>'


def calib_tables(Hm, c0, c1, s0, s1):
    esc, num = Hm.esc, Hm.num
    def f(x, nd=3):
        return "—" if x is None else f"{x:+.{nd}f}".replace("-", MINUS)
    def ci(v):
        return f'<span class="ci"> [{v[0]:.3f}, {v[1]:.3f}]</span>' if v else ""
    rows = [f'<tr class="grp"><th colspan="4" scope="rowgroup">masked slots: ECE by denoising time t (higher t = more of the block still open)</th></tr>']
    for b0, b1 in zip(c0["masked_by_t"], c1["masked_by_t"]):
        rows.append(f'<tr><th scope="row">t {b1["t_lo"]:.1f}–{b1["t_hi"]:.1f} <span class="nn">n={b1["n"]:,}</span></th>'
                    f'<td class="num">{b0["ece"]:.3f}{ci(b0["ece_ci"])}</td><td class="num">{b1["ece"]:.3f}{ci(b1["ece_ci"])}</td>'
                    f'<td class="num">{f(b1["gap"])}</td></tr>')
    rows.append('<tr class="grp"><th colspan="4" scope="rowgroup">conf − acc by slot kind (positive = overconfident)</th></tr>')
    for k, lab in (("mask", "mask: slot was [MASK]"), ("confirm", "confirm: draft present and right"), ("override", "override: draft present and wrong")):
        a, b = c0["by_kind"][k], c1["by_kind"][k]
        rows.append(f'<tr><th scope="row">{esc(lab)} <span class="nn">n={b["n"]:,} · acc {100 * b["acc"]:.1f}%</span></th>'
                    f'<td class="num">{f(a["gap"])}<span class="ci"> ECE {a["ece"]:.3f}</span></td><td class="num">{f(b["gap"])}<span class="ci"> ECE {b["ece"]:.3f}</span></td>'
                    f'<td class="num">{f(b["gap"] - a["gap"])}</td></tr>')
    rows.append(f'<tr class="grp"><th colspan="4" scope="rowgroup">joint calibration under the decoder rule (commit p &gt; {c1["tau"]}): product of confidences − empirical P(all correct), by tokens committed in one pass</th></tr>')
    for a, b in zip(c0["joint"]["tau_rule"], c1["joint"]["tau_rule"]):
        kl = f'k {b["k_lo"]}' if b["k_lo"] == b["k_hi"] else f'k {b["k_lo"]}–{b["k_hi"]}'
        rows.append(f'<tr><th scope="row">{kl} <span class="nn">n={b["n_states"]} states · empirical {b["p_all_correct"]:.3f} [{b["p_all_correct_ci"][0]:.3f}, {b["p_all_correct_ci"][1]:.3f}]</span></th>'
                    f'<td class="num">{f(a["gap"])}</td><td class="num">{f(b["gap"])}</td><td class="num">{f(b["gap"] - a["gap"])}</td></tr>')
    head = (f'<thead><tr><th scope="col">quantity</th><th scope="col" class="num">{esc(s0)} <span class="nn">first probe</span></th>'
            f'<th scope="col" class="num">{esc(s1)} <span class="nn">latest</span></th><th scope="col" class="num">change (latest − first)</th></tr></thead>')
    return f'<div class="tscroll"><table class="data">{head}<tbody>{"".join(rows)}</tbody></table></div>'


# ------------------------------------------------------------------------------------------------ section
def render(Hm, specs, ev):
    esc, num, fig = Hm.esc, Hm.num, Hm.fig
    trA, evA = load_rows(TRAIN_A); trB, evB = load_rows(TRAIN_B)
    meta = train_meta(); cfg = meta["cfg"]
    lastB = max(trB) if trB else 0; lastA = max(trA) if trA else 0
    lastB_utc = trB[lastB].get("utc") if trB else None
    xhi_all = round(max(lastA / STEPS_PER_EPOCH, B_OFFSET + lastB / STEPS_PER_EPOCH) + 0.05, 2)
    restart = (B_OFFSET + meta["resume"] / STEPS_PER_EPOCH) if meta["resume"] else None
    marks = [(restart, "restart", "res")] if restart else []
    A = {"name": "r3_A", "color": "var(--r3a)"}; B = {"name": "r3_B", "color": "var(--r3b)"}

    # ---- training curves, epoch-aligned
    tfigs = []
    def ab_chart(cid, title, key, fn=None, scale=1.0, dec=3, suffix="", sub="", ylo=None, yhi=None):
        pa, pb = ser(trA, key, 0.0, fn), ser(trB, key, B_OFFSET, fn)
        if not pa and not pb:
            return
        sers = [{"name": "r3_A raw", "pts": pa, "cls": "raw", "color": A["color"], "tip": False},
                {"name": "r3_A EMA", "pts": ema(pa), "cls": "smooth", "color": A["color"]},
                {"name": "r3_B raw", "pts": pb, "cls": "raw", "color": B["color"], "tip": False},
                {"name": "r3_B EMA", "pts": ema(pb), "cls": "smooth", "color": B["color"]}]
        svg, spec = epoch_chart(Hm, cid, title, sers, xlo=0, xhi=xhi_all, scale=scale, dec=dec, suffix=suffix, ylo=ylo, yhi=yhi, marks=marks)
        specs.append(spec)
        va = f"r3_A step {lastA:,}: {pa[-1][1] * scale:,.{dec}f}{suffix}" if pa else ""
        vb = f"r3_B step {lastB:,}: {pb[-1][1] * scale:,.{dec}f}{suffix}" if pb else ""
        tfigs.append(fig(title, " · ".join(x for x in (vb, va) if x), svg, esc(sub)))
    ab_chart("b-acc", "Train accuracy on open slots", "acc", scale=100, dec=1, suffix="%", sub="argmax = target. r3_B is plotted from epoch 1, where it starts (r3_A step 700 weights).")
    ab_chart("b-gap", "Overconfidence gap: mean top-1 − accuracy", None, fn=lambda r: r["top1"] - r["acc"], scale=100, dec=1, suffix=" pts",
             sub="Positive = the model claims more than it gets right. r3_A's certainty term (β=2) kept widening this; r3_B (β=0.5 + calibration BCE) holds it near zero.")
    # r3_B loss terms
    terms = [("ce", "CE (weighted)", "var(--c1)", False), ("cert", "certainty H (raw)", "var(--c2)", False), ("calib", "calibration BCE (raw)", "var(--c3)", False),
             ("loss_cert_term", "β·cert (in loss)", "var(--c2)", True), ("loss_calib_term", "λ·calib (in loss)", "var(--c3)", True), ("cum_loss", "cumulative mean loss", "var(--c4)", True)]
    sers = []
    for k, nm, col, dash in terms:
        p = ser(trB, k, B_OFFSET)
        if p:
            sers.append({"name": nm, "pts": ema(p) if not dash else p, "cls": "smooth", "color": col, "dash": dash})
    if sers:
        svg, spec = epoch_chart(Hm, "b-terms", "r3_B loss terms", sers, xlo=B_OFFSET - 0.02, xhi=xhi_all, dec=3, marks=marks)
        specs.append(spec)
        last = trB[lastB]
        vl = f"step {lastB:,}: ce {last.get('ce', float('nan')):.3f} · cert {last.get('cert', float('nan')):.3f} · calib {last.get('calib', float('nan')):.3f} · loss {last.get('loss', float('nan')):.3f}"
        present = [k for k, *_ in terms if any(k in r for r in trB.values())]
        first_term = min((s for s, r in trB.items() if "loss_cert_term" in r), default=None)
        tfigs.append(fig("r3_B loss terms (EMA; dashed = as they enter the loss, raw)", vl, svg,
                         esc(f"Series present in the log: {', '.join(present)}. loss_cert_term, loss_calib_term and cum_loss are logged from step {first_term:,} (the restart that added calibration logging).") if first_term else ""))
    # ECE train and per-kind CE
    p = ser(trB, "ece_train", B_OFFSET)
    if p:
        svg, spec = epoch_chart(Hm, "b-ece", "r3_B training ECE", [{"name": "ece_train raw", "pts": p, "cls": "raw", "color": B["color"], "tip": False},
                                                                  {"name": "ece_train EMA", "pts": ema(p), "cls": "smooth", "color": B["color"]}],
                                xlo=B_OFFSET - 0.02, xhi=xhi_all, dec=4, marks=marks)
        specs.append(spec)
        tfigs.append(fig("r3_B training ECE (top-1 of the MASK-suppressed softmax vs correctness)", f"step {lastB:,}: {p[-1][1]:.4f} · logged from step {min(s for s, r in trB.items() if 'ece_train' in r):,}", svg,
                         "Expected calibration error on the training batch, 20 confidence bins. Lower is better; this is the quantity the BCE term targets."))
    kinds = [("ce_mask", "mask", "var(--k-mask)"), ("ce_confirm", "confirm", "var(--k-ok)"), ("ce_override", "override", "var(--k-bad)")]
    sers = [{"name": nm, "pts": ema(ser(trB, k, B_OFFSET)), "cls": "smooth", "color": col} for k, nm, col in kinds if ser(trB, k, B_OFFSET)]
    if sers:
        svg, spec = epoch_chart(Hm, "b-kind", "r3_B cross-entropy by slot kind", sers, xlo=B_OFFSET - 0.02, xhi=xhi_all, dec=3, marks=marks)
        specs.append(spec)
        last = trB[lastB]
        tfigs.append(fig("r3_B cross-entropy by slot kind (EMA)", f"step {lastB:,}: mask {last.get('ce_mask', float('nan')):.3f} · confirm {last.get('ce_confirm', float('nan')):.3f} · override {last.get('ce_override', float('nan')):.3f}", svg,
                         "Unweighted per-kind CE. Override slots (draft present and wrong) stay the hard case by an order of magnitude over confirms."))
    t_legend = ('<ul class="legend"><li><span class="lk" style="border-color:var(--r3b)"></span>r3_B (calibrated recipe)</li>'
                '<li><span class="lk" style="border-color:var(--r3a)"></span>r3_A (original recipe)</li>'
                '<li><span class="lk faint" style="border-color:var(--ink-3)"></span>raw every 10 steps; solid = EMA 0.8</li>'
                '<li><span class="lk" style="border-color:var(--ev)"></span>epoch boundary</li>'
                '<li><span class="lk" style="border-color:var(--s-judge)"></span>r3_B restart at step ' + (f'{meta["resume"]:,}' if meta["resume"] else "—") + '</li></ul>')

    # ---- holdout, both runs, epochs 1–2
    hfigs = []
    hold = {}
    for key, title, cap in [("draft_pass1_acc", "Holdout accuracy, judge on drafter-filled pass 1", "argmax equals the stock trajectory token; 64 states, ±~1 pt noise"),
                            ("draft_pass1_conf_frac", "Holdout confident fraction (p > 0.95)", "share of open slots the judge would commit in one pass"),
                            ("draft_pass1_conf_precision", "Holdout confident precision", "accuracy on the committed slots")]:
        pa = ser(evA, key, 0.0); pb = ser(evB, key, B_OFFSET); hold[key] = (pa, pb)
        if not pa and not pb:
            continue
        svg, spec = epoch_chart(Hm, f"bh-{key}", title, [{"name": "r3_B", "pts": pb, "cls": "eval", "color": B["color"]},
                                                          {"name": "r3_A", "pts": pa, "cls": "eval", "color": A["color"]}],
                                xlo=0.8, xhi=2.1, scale=100, dec=1, suffix="%", cap100=True)
        specs.append(spec)
        vl = (f"r3_B step {lastB - lastB % 100 if evB else 0:,}: {pb[-1][1] * 100:.1f}%" if pb else "") + (f" · r3_A at epoch 2 (step 1,400): {dict(evA).get(1400, {}).get(key, float('nan')) * 100:.1f}%" if 1400 in evA else "")
        hfigs.append(fig(title, vl, svg, esc(cap)))
    for key, title, cap, dec in [("draft_pass1_ovr_conf_frac", "r3_B: confident fraction on override slots", "share of override slots (draft present and wrong) committed at p > 0.95", 2),
                                  ("draft_pass1_ovr_conf_precision", "r3_B: precision of confident overrides", "accuracy of those commits; r3_A did not log this split", 1)]:
        pb = ser(evB, key, B_OFFSET)
        if not pb:
            continue
        svg, spec = epoch_chart(Hm, f"bh-{key}", title, [{"name": "r3_B", "pts": pb, "cls": "eval", "color": B["color"]}], xlo=0.8, xhi=2.1, scale=100, dec=dec, suffix="%", cap100=True)
        specs.append(spec)
        hfigs.append(fig(title, f"latest: {pb[-1][1] * 100:.{dec}f}%", svg, esc(cap)))
    h_legend = ('<ul class="legend"><li><span class="lk" style="border-color:var(--r3b)"></span>r3_B, every 100 steps</li>'
                '<li><span class="lk" style="border-color:var(--r3a)"></span>r3_A, every 200 steps</li></ul>')

    # ---- calibration
    steps, latest = latest_calib()
    assets = copy_assets(latest) if latest else []
    c0 = json.load(open(os.path.join(CALIB, f"{steps[0]}.json"))) if steps else None
    c1 = json.load(open(os.path.join(CALIB, f"{latest}.json"))) if latest else None
    calib_html = []
    if c0 and c1:
        ov0, ov1 = c0["by_kind"]["override"], c1["by_kind"]["override"]
        m1 = c1["by_kind"]["mask"]; cf1 = c1["by_kind"]["confirm"]
        worst_t = max(c1["masked_by_t"], key=lambda b: abs(b["gap"]))
        under_all = all(b["gap"] <= 0.005 for b in c1["masked_by_t"])
        j = c1["joint"]["tau_rule"]; jpos = [b for b in j if b["gap"] > 0]; jneg = [b for b in j if b["gap"] <= 0]
        calib_html.append(f'<p class="lede">The trainer probed calibration on {c1["n_states"]} teacher-forced holdout states ({c1["n_slots"]:,} open slots) at steps {esc(", ".join(s[4:].lstrip("0") for s in steps))}. '
                          f'Numbers below compare the first probe (<code>{esc(steps[0])}</code>) with the latest (<code>{esc(latest)}</code>); the PNGs are the trainer’s own plots for the latest step.</p>')
        calib_html.append(calib_tables(Hm, c0, c1, steps[0], latest))
        # interpretation, phrased from the numbers
        li = []
        li.append(f"<strong>Masked slots</strong> ({m1['n']:,}) are {'slightly underconfident at every t' if under_all else 'close to calibrated'}: conf − acc {MINUS if m1['gap'] < 0 else '+'}{abs(m1['gap']):.3f} overall, "
                  f"worst in t {worst_t['t_lo']:.1f}–{worst_t['t_hi']:.1f} ({'−' if worst_t['gap'] < 0 else '+'}{abs(worst_t['gap']):.3f}, ECE {worst_t['ece']:.3f}), which is also where most masked slots live (n={worst_t['n']:,}). "
                  "Underconfidence costs passes, not correctness.")
        li.append(f"<strong>Confirm slots</strong> ({cf1['n']:,}, {100 * cf1['acc']:.1f}% right) are well calibrated: conf − acc {'−' if cf1['gap'] < 0 else '+'}{abs(cf1['gap']):.3f}, ECE {cf1['ece']:.3f}.")
        grew = ov1["gap"] > ov0["gap"]
        li.append(f"<strong>Override slots</strong> ({ov1['n']:,}, only {100 * ov1['acc']:.1f}% right) are overconfident: conf − acc +{ov1['gap']:.3f} (ECE {ov1['ece']:.3f} [{ov1['ece_ci'][0]:.3f}, {ov1['ece_ci'][1]:.3f}]), "
                  + (f"and the gap <strong>grew</strong> from +{ov0['gap']:.3f} at {steps[0][4:].lstrip('0')} to +{ov1['gap']:.3f}. This is the watch item: the 2×/3× override weights push CE on these slots, and the BCE term has not fully countered the extra confidence."
                     if grew else f"and the gap moved from +{ov0['gap']:.3f} to +{ov1['gap']:.3f}."))
        li.append(f"<strong>Joint calibration under the decoder rule</strong> (commit every slot with p &gt; {c1['tau']}): the product of the committed confidences sits at or below the empirical all-correct rate in "
                  f"{len(jneg)} of {len(j)} k buckets, so the independence assumption is conservative there; the exception is "
                  + (", ".join(f"k {b['k_lo']}{'' if b['k_lo'] == b['k_hi'] else '–' + str(b['k_hi'])} (+{b['gap']:.3f}, n={b['n_states']})" for b in jpos) if jpos else "none") + ".")
        calib_html.append("<ul>" + "".join(f"<li>{x}</li>" for x in li) + "</ul>")
        calib_html.append('<p class="muted" style="font-size:.9rem">Caveat: the joint calibration is a single-forward simulation on teacher-forced holdout states; “correct” means equal to the stock training-trajectory token at that slot, not to ground truth, and the decoder rule is applied once per state rather than iterated to the end of the block.</p>')
        imgs = [("trend_calibration.png", "Trend over probes: masked ECE by t, conf − acc by slot kind, joint gap by k"),
                (f"{latest}_reliability_grid.png", f"{latest}: reliability diagrams by t and slot kind"),
                (f"{latest}_ece_nll_vs_t.png", f"{latest}: ECE and top-1 NLL against t"),
                (f"{latest}_joint_calibration.png", f"{latest}: product of confidences vs empirical P(all correct), decoder rule and top-k"),
                (f"{latest}_train_confidence.png", f"{latest}: training-batch confidence histogram and reliability")]
        calib_html.append('<div class="pngs">' + "".join(
            f'<figure class="fig png"><a href="r3_assets/{n}"><img src="r3_assets/{n}" alt="{esc(cap)}" loading="lazy"></a><figcaption>{esc(cap)} <span class="mono">r3_assets/{esc(n)}</span></figcaption></figure>'
            for n, cap in imgs if n in assets) + "</div>")
    else:
        calib_html.append('<div class="placeholder">No calibration probes found in <code>logs/draftalign/train/r3_B/calib/</code>.</div>')

    # ---- why: plateau numbers
    pp = plateau_precision()
    late = [k for k in ("step000700", "step001000", "step001400", "step001500") if k in pp and pp[k][0] is not None]
    prec_all = [100 * pp[k][0] for k in late]; prec_code = [100 * pp[k][1] for k in late if pp[k][1] is not None]
    prec_ok = [100 * pp[k][2] for k in late if pp[k][2] is not None]
    why = f"""
<p class="lede">r3_A's holdout accuracy stopped moving after epoch 1 while its certainty term kept sharpening the distribution (the r3_A section below,
<a href="#plateau">“Why the curves flattened”</a>, has the full diagnosis). Two things in the plateau probe made that a recipe problem rather than a data problem.</p>
<ul>
<li><strong>Confident overrides were not precise enough to commit.</strong> Where the draft was present and wrong (<code>holdout_pass1|all|draft_bad</code>), the slots r3_A committed at p &gt; 0.95 were right only
{num(min(prec_all), 1)}–{num(max(prec_all), 1)}% of the time across steps {esc(", ".join(k[4:].lstrip("0") for k in late))}"""
    if prec_code:
        why += f" (code pool: {num(min(prec_code), 1)}–{num(max(prec_code), 1)}%)"
    why += ". A single wrong committed token breaks a program.</li>"
    if prec_ok:
        why += (f"<li><strong>Confident confirms degraded</strong> as training went on: precision on committed <code>draft_ok</code> slots went {num(prec_ok[0], 1)}% → {num(prec_ok[-1], 1)}% over the same checkpoints, "
                "while the confident fraction kept rising. The β=2 certainty term was buying confidence, not correctness.</li>")
    why += f"""<li><strong>So r3_B changes the objective, not the data:</strong> the same {22734:,}-example split, the same LoRA, initialised from r3_A's epoch-1 weights, with a weaker certainty term, an explicit calibration
loss on the decoder's commit confidence, and extra weight on the override slots that matter for code.</li></ul>"""

    # ---- loss
    b, lam, ow, cw, tau = cfg["beta"], cfg["calib"], cfg["override_weight"], cfg["code_override_weight"], cfg["temp"]
    loss = f"""
<p class="kick" style="margin:0 0 .3rem">r3_A (original)</p>
<div class="eq">L = (1/n) Σ_i CE(z_i, T_i)  +  β · (1/|C|) Σ_(i∈C) H(softmax(z̃_i / τ))
C = {{ i : argmax z_i = T_i }},   β = 2,   τ = {tau},   z̃ = logits with the [MASK] column removed</div>
<p class="kick" style="margin:1rem 0 .3rem">r3_B (calibrated)</p>
<div class="eq">L = Σ_i w_i · CE_i / Σ_i w_i                              weighted cross-entropy
  + β · (1/|C|) Σ_(i∈C) H(softmax(z̃_i / τ))              same dParallel certainty term, β = {b:g}
  + λ · Σ_i w_i · BCE(p_i, y_i) / Σ_i w_i                 calibration, λ = {lam:g}

p_i = max softmax(z̃_i)        top-1 probability with [MASK] suppressed: the decoder's commit confidence
y_i = 1[ argmax z̃_i = T_i ]    did that commit turn out right
w_i = 1                        [MASK] slots and confirms
    = {ow:g}                        override slots (draft present and wrong)
    = {cw:g}                        override slots in the code pool</div>
<p>The BCE term is a proper scoring rule on the exact scalar the decoder thresholds at inference, so a slot that is committed at 0.97 is trained to be right 97% of the time, in both directions.
The weights move the CE mass toward the decisions that break programs. Code: <code>code/draftalign/losses.py</code>; trainer flags
<code>--beta {b:g} --calib {lam:g} --override-weight {ow:g} --code-override-weight {cw:g} --init-adapter {esc(meta["init"] or "logs/draftalign/eval_ckpts/step000700")}</code>
(<code>{esc(meta["init_n"] or "?")}</code> tensors loaded; optimiser and cosine schedule restarted from step 0; config hash <code>{esc(json.load(open(os.path.join(TRAIN_B, "config.json")))["config_hash"])}</code>).</p>"""

    # ---- status
    status = f"""
<div class="statusbox"><h3>Status</h3><ul>
<li><strong>r3_A</strong> stopped for good at step 2,000 (2.82 epochs) at 11:44 IST. Its last logged row is step {lastA:,}. No further training on that recipe.</li>
<li><strong>r3_B</strong> launched 11:48 IST from <code>r3_A/step000700</code> (epoch-1 weights) with a fresh optimiser and schedule, {cfg["steps"]:,} steps = one more epoch.
It was restarted once at step {meta["resume"] or 300} (12:55 IST) only to add the calibration logging, and finished step {lastB:,} = epoch 2 at 14:31 IST
({'<code>done</code> marker present' if meta["done"] else 'no <code>done</code> marker yet'}; {meta["n_saves"]} checkpoint saves in the log). Last row {esc(lastB_utc or "—")} UTC.</li>
<li><strong>Same recipe otherwise:</strong> LoRA r{cfg["lora_r"]}/α{cfg["alpha"]:.0f} on q,k,v,o + draft flag, lr {cfg["lr_lora"]:.0e}, {cfg["world_size"]}×{cfg["grad_accum"]} = 32 states/step, holdout eval every {cfg["eval_every"]} steps, calibration probe every 100 from step 300.</li>
<li><strong>Downstream evals</strong> of <code>r3_B/step000710</code> + DES48 on all six sets are in the tables below (snapshot {esc(ev["generated_ist"]) if ev else "—"}); the r3_A epoch-1/epoch-2 arms and stock are re-summarised on the same item universe.</li>
</ul>
<p class="kick" style="margin:.8rem 0 .2rem">Where things are</p>
<ul>
<li>HF (public, Anupam): <a href="{HF}">Nemotron-split-moe-train</a> · <code>r3_B/checkpoints</code>, <code>r3_A/checkpoints</code>, <code>round2_t2b_union_flag/step003057</code>, <code>reference_evals</code>, <code>r3_B/analysis</code></li>
<li>GitHub: branch <a href="{GH}"><code>round3-2026-09-14</code></a> · <code>docs/results/r3_B/</code>, <code>docs/MODELS_AND_RESULTS_2026-09-15.md</code></li>
<li>W&amp;B: <a href="{WB_B}">runs/r3_B</a> · <a href="{WB_A}">runs/r3_A</a> (every chart on this page is the per-step series those runs logged, re-read from <code>train.rank0.jsonl</code>)</li>
</ul></div>"""

    # ---- evals
    ev_html = []
    if ev is None:
        ev_html.append('<div class="placeholder">No <code>probes_r3/r3_B/eval_summary.json</code>. Run <code>code/draftalign/eval_summary_r3B.py --step step000710</code>.</div>')
    else:
        ds = ev["datasets"]
        n_complete = sum(1 for d in ds.values() for a in COV_ROWS if d["rows"][a]["status"] == "complete")
        n_part = sum(1 for d in ds.values() for a in COV_ROWS if d["rows"][a]["status"] == "partial")
        n_pend = sum(1 for d in ds.values() for a in COV_ROWS if d["rows"][a]["status"] not in ("complete", "partial"))
        ev_html.append(f'<p class="lede">Snapshot generated {esc(ev["generated_ist"])} from <code>{esc(ev["run_dir"])}/{esc(ev["step"])}</code>: '
                       f'{n_complete} complete, {n_part} partial and {n_pend} pending cells over {len(COV_ROWS)} arms × {len(ds)} datasets. '
                       'Every Δ is paired on the items both arms scored; brackets are bootstrap 95% CIs; speedup is stock expert loads per token ÷ the arm’s, on the shared items (the drafter is not charged).</p>')
        ev_html.append("<h3>Coverage</h3>")
        ev_html.append('<p>Cell = items scored / items in the set. Partial cells are r3_A’s 100-item GSM8K and 50-item MATH subsets, and the team mirror’s incomplete stock + DES48 on GPQA and AIME; their Δs are on that subset only.</p>')
        ev_html.append(coverage_matrix(Hm, ev))
        # scatter, small multiples with shared axes
        ev_html.append("<h3>Accuracy cost against speedup</h3>")
        ev_html.append('<p>One point per arm and dataset: Δacc vs stock (whisker = 95% CI) against the speedup in expert loads per token vs stock. Up and right is better; the zero line is “as accurate as stock”. Hollow = partial cell.</p>')
        leg = "".join(f'<li><span class="sw" style="background:{ARM_COLOR[a]}"></span>{esc(ev["labels"][a])}</li>' for a in FOCUS
                      if any(d["rows"][a]["n_scored"] for d in ds.values()))
        ev_html.append(f'<ul class="legend">{leg}</ul>')
        allp = []
        for k, d in ds.items():
            for a in FOCUS:
                c = d["rows"][a]; v = c.get("vs_stock")
                if v and v.get("d_acc") is not None and v.get("speedup_exp"):
                    allp.append((k, a, v, c))
        if allp:
            xr = (min(v["speedup_exp"] for _, _, v, _ in allp) - 0.05, max(v["speedup_exp"] for _, _, v, _ in allp) + 0.05)
            yr = (min(v["ci"][0] for _, _, v, _ in allp) - 1, max(v["ci"][1] for _, _, v, _ in allp) + 1)
            sf = []
            for k, d in ds.items():
                pts = [{"x": v["speedup_exp"], "y": v["d_acc"], "lo": v["ci"][0], "hi": v["ci"][1], "color": ARM_COLOR[a], "partial": c["status"] == "partial",
                        "tip": f"{ev['labels'][a]} · {d['label']}: Δacc {num(v['d_acc'], 1, True)} [{num(v['ci'][0], 1, True)}, {num(v['ci'][1], 1, True)}] · {v['speedup_exp']:.2f}× · n={v['n_common_scored']}"}
                       for kk, a, v, c in allp if kk == k]
                sf.append(fig(d["label"], f"{len(pts)} arms", tradeoff_chart(Hm, f"bs-{k}", f"{d['label']} Δacc vs speedup", pts, xr, yr), ""))
            ev_html.append(f'<div class="grid2">{"".join(sf)}</div>')
        ev_html.append("<h3>Headline table</h3>")
        ev_html.append(headline_tables(Hm, ev))
        # interpretation, conditional on numbers
        def g(k, a, key="vs_stock"):
            v = ds[k]["rows"][a].get(key)
            return v if v and v.get("d_acc") is not None else None
        li = []
        hb, ha2 = g("humaneval", "r3B_judge_des48"), g("humaneval", "r3A_e2_judge_des48")
        if hb and ha2:
            li.append(f"<strong>HumanEval:</strong> r3_B + DES48 is {num(hb['d_acc'], 1, True)} pts vs stock [{num(hb['ci'][0], 1, True)}, {num(hb['ci'][1], 1, True)}] at {hb['speedup_exp']:.2f}×, "
                      f"against r3_A epoch 2's {num(ha2['d_acc'], 1, True)} at {ha2['speedup_exp']:.2f}×: "
                      + ("most of the code gap is recovered at the same cost, and the CI now includes zero." if hb["d_acc"] > ha2["d_acc"] and not sig(hb["ci"]) else "see the table."))
        mb48 = g("mbpp", "r3B_judge_des48", "vs_stock_des48"); mb = g("mbpp", "r3B_judge_des48"); ma2 = g("mbpp", "r3A_e2_judge_des48")
        if mb48 and mb:
            li.append(f"<strong>MBPP:</strong> {num(mb['d_acc'], 1, True)} vs stock, {num(mb48['d_acc'], 1, True)} vs stock + DES48 [{num(mb48['ci'][0], 1, True)}, {num(mb48['ci'][1], 1, True)}]"
                      + (", within the CI of the same-pool baseline" if not sig(mb48["ci"]) else "") + f", at {mb['speedup_exp']:.2f}×"
                      + (f" (r3_A epoch 2: {num(ma2['d_acc'], 1, True)} at {ma2['speedup_exp']:.2f}×)." if ma2 else "."))
        gb = g("gsm8k500", "r3B_judge_des48")
        if gb:
            li.append(f"<strong>GSM8K-500:</strong> {num(gb['d_acc'], 1, True)} [{num(gb['ci'][0], 1, True)}, {num(gb['ci'][1], 1, True)}] vs stock at {gb['speedup_exp']:.2f}×"
                      + (": matches stock at about 1.5× on the full 500." if not sig(gb["ci"]) and gb["speedup_exp"] >= 1.4 else "."))
        mt = g("math500", "r3B_judge_des48"); mt48 = g("math500", "r3B_judge_des48", "vs_stock_des48")
        if mt and mt48:
            li.append(f"<strong>MATH-500:</strong> {num(mt['d_acc'], 1, True)} [{num(mt['ci'][0], 1, True)}, {num(mt['ci'][1], 1, True)}] vs stock, {num(mt48['d_acc'], 1, True)} [{num(mt48['ci'][0], 1, True)}, {num(mt48['ci'][1], 1, True)}] vs stock + DES48"
                      + (": statistically level with the same-pool baseline" if not sig(mt48["ci"]) else "") + f", {mt['speedup_exp']:.2f}×.")
        gp = g("gpqa", "r3B_judge_des48")
        if gp:
            li.append(f"<strong>GPQA-Diamond:</strong> {num(gp['d_acc'], 1, True)} [{num(gp['ci'][0], 1, True)}, {num(gp['ci'][1], 1, True)}] vs stock"
                      + (", still clearly below stock. Science multiple-choice is not in the training mix (code / GSM8K / IF), and every judge arm loses here." if sig(gp["ci"]) and gp["d_acc"] < 0 else "."))
        ai = g("aime", "r3B_judge_des48")
        if ai:
            li.append(f"<strong>AIME:</strong> {num(ai['d_acc'], 1, True)} [{num(ai['ci'][0], 1, True)}, {num(ai['ci'][1], 1, True)}] on 90 items where stock scores {num(ds['aime']['rows']['stock']['acc'], 1)}%: at floor, uninformative.")
        slower = [(k, g(k, "r3B_judge_des48"), g(k, "r3A_e2_judge_des48")) for k in ("mbpp", "math500", "humaneval", "gsm8k500")]
        slower = [(k, b, a) for k, b, a in slower if b and a and b["speedup_exp"] < a["speedup_exp"]]
        if slower:
            li.append("<strong>The calibration trade-off:</strong> r3_B commits fewer tokens per pass than r3_A epoch 2, so its speedup is lower on "
                      + ", ".join(f"{ds[k]['label']} ({b['speedup_exp']:.2f}× vs {a['speedup_exp']:.2f}×)" for k, b, a in slower)
                      + ". Confidence it cannot back is confidence it no longer spends.")
        ev_html.append("<h3>Reading</h3><ul>" + "".join(f"<li>{x}</li>" for x in li) + "</ul>")
        ev_html.append('<p class="muted" style="font-size:.9rem">Tables as markdown: <code>logs/draftalign/probes_r3/r3_B/eval_summary.md</code>. Records: <code>logs/draftalign/probes_r3/r3_B/step000710/&lt;ds&gt;/&lt;arm&gt;/</code>.</p>')

    sec = Hm.sec
    parts = [
        '<div class="partband" id="r3b"><p class="kick">Round 3b</p><h2 class="parttitle">Calibrated retrain (r3_B)</h2>'
        '<p class="deck">One more epoch from r3_A’s epoch-1 weights, with a weaker certainty term, a calibration loss on the commit confidence and extra weight on override slots. '
        'The r3_A material (idea, data, method, plateau diagnosis) follows further down; this part covers only what changed and what it did.</p></div>',
        sec("status", "Where r3_B stands", status, "b-status"),
        sec("why", "Why a new recipe", why, "b-why"),
        sec("loss", "The loss, before and after", loss, "b-loss"),
        sec("training", "Training curves, r3_B against r3_A by epoch",
            f'<p class="lede">x is epochs of the same {22734:,}-example split: r3_A step N is N·32/22,734, r3_B step N is 1 + N·32/22,734 because it starts from r3_A’s step-700 weights. '
            f'These are the per-step series logged to W&amp;B (<a href="{WB_B}">r3_B</a>, <a href="{WB_A}">r3_A</a>), read from each run’s <code>train.rank0.jsonl</code>.</p>'
            + t_legend + f'<div class="grid2">{"".join(tfigs)}</div>', "b-training"),
        sec("holdout", "Holdout over epochs 1–2", f'<p class="lede">The trainer’s 64-state holdout eval (<code>draft_pass1</code>), r3_B every 100 steps against r3_A every 200, on the epoch axis. Differences of about one point are noise.</p>'
            + h_legend + f'<div class="grid2">{"".join(hfigs)}</div>', "b-holdout"),
        sec("calibration", "Calibration probes", "".join(calib_html), "b-calibration"),
        sec("evals", "Downstream evals, r3_B against every earlier arm", "".join(ev_html), "b-evals"),
        '<div class="partband" id="r3a"><p class="kick">Round 3a</p><h2 class="parttitle">The original run (r3_A), kept for the record</h2>'
        '<p class="deck">Everything below is the r3_A report as it stood: the idea, the data split, the method, the curves, the plateau diagnosis that motivated r3_B, and r3_A’s own eval wave.</p></div>',
    ]
    return "\n".join(parts), assets


CSS_EXTRA = r"""
:root{--r3b:#2a78d6;--r3a:#4a3aa7;--r3a-light:#9085e9;--r3a-dark:#2d2270;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#4a3aa7;
  --cv-ok:#E4F3E4;--cv-part:#FDF1D6;--cv-pend:#ECEFF1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--r3b:#3987e5;--r3a:#9085e9;--r3a-light:#b5adf2;--r3a-dark:#6c5fd0;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#9085e9;
  --cv-ok:#15301A;--cv-part:#35290C;--cv-pend:#20272E}}
:root[data-theme="dark"]{--r3b:#3987e5;--r3a:#9085e9;--r3a-light:#b5adf2;--r3a-dark:#6c5fd0;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#9085e9;
  --cv-ok:#15301A;--cv-part:#35290C;--cv-pend:#20272E}
.partband{padding:2.4rem 0 .6rem;border-top:3px solid var(--ink)}
.partband .parttitle{font-family:"IBM Plex Serif",Georgia,serif;font-size:clamp(1.6rem,3.2vw,2.2rem);font-weight:600;margin:.2rem 0 .6rem;text-wrap:balance}
table.cov2 td.cv{text-align:right;font-variant-numeric:tabular-nums;vertical-align:middle}
table.cov2 td.cv-ok{background:var(--cv-ok)} table.cov2 td.cv-part{background:var(--cv-part)} table.cov2 td.cv-pend{background:var(--cv-pend);color:var(--ink-3)}
table.cov2 thead th .nn{display:block;font-weight:400}
svg .whisk{stroke-width:1.5;opacity:.6}
.pngs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,26rem),1fr));gap:1.2rem 1.6rem;margin:1.2rem 0}
figure.png img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:#fff}
figure.png figcaption .mono{display:block;font-size:.72rem;margin-top:.1rem}
"""
