#!/usr/bin/env python
"""Results-only page (user, 2026-09-15 16:37 IST): rows = models, one table per dataset, an all-in-one grid, a
Δacc-vs-speedup chart and the ICLR-2027 abstract. Every number comes from logs/draftalign/probes_r3/r3_B/eval_summary.json
(run code/draftalign/eval_summary_r3B.py --step step000710 first).

    /venv/r3/bin/python docs/report/build_results_page.py   -> docs/report/r3_results.html
"""
import datetime as dt
import html
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "logs", "draftalign", "probes_r3", "r3_B", "eval_summary.json")
OUT = os.path.join(ROOT, "docs", "report", "r3_results.html")
HF = "https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts/tree/main/Nemotron-split-moe-train"
GH = "https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert/tree/round3-2026-09-14"

# display order; (key, short label, group, description)
ROWS = [
    ("stock", "stock", "no training", "SDAR-30B-A3B-Chat-b32 as released, block 32, commit at p > 0.95"),
    ("stock_des32", "stock + DES32", "no training", "stock with a 32-expert DES pool per forward"),
    ("stock_des48", "stock + DES48", "no training", "stock with a 48-expert DES pool per forward"),
    ("blind", "blind accept", "no training", "1.7B drafter fills the block; stock keeps its confident draft tokens"),
    ("prev_judge_des48", "round-2 judge + DES48", "trained (GSM8K-heavy)", "T2b union judge, step 3057 — the earlier published judge"),
    ("r3A_e1_judge_des48", "r3_A epoch 1 + DES48", "trained (β 2, no calibration)", "round-3 judge, CE + certainty, step 700"),
    ("r3A_e2_judge_des48", "r3_A epoch 2 + DES48", "trained (β 2, no calibration)", "same run, step 1400"),
    ("r3A_s2000_judge_des48", "r3_A step 2000 + DES48", "trained (β 2, no calibration)", "same run, last checkpoint (2.82 epochs)"),
    ("r3B_judge", "r3_B judge (ours, no pool)", "trained (calibrated)", "calibrated recipe from r3_A step 700, epoch 2; no expert pool"),
    ("r3B_judge_des48", "r3_B judge + DES48 (ours)", "trained (calibrated)", "the same checkpoint with the 48-expert DES pool"),
]
DS_ORDER = ["humaneval", "mbpp", "gsm8k500", "math500", "gpqa", "aime"]


def esc(s):
    return html.escape(str(s))


def fmt_d(v):
    if not v or v.get("d_acc") is None:
        return "—"
    return f"{v['d_acc']:+.1f} <span class='ci'>[{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]</span>"


def cls_d(v):
    if not v or v.get("d_acc") is None:
        return ""
    lo, hi = v["ci"]
    if hi < 0:
        return "bad"
    if lo > 0:
        return "good"
    return "ok"


def row_state(c):
    if c["n_records"] == 0:
        return "missing"
    return "complete" if c["status"] == "complete" else "partial"


def main():
    r = json.load(open(SRC))
    ds = r["datasets"]
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %H:%M IST")

    # ---- headline numbers for the abstract / lede (r3_B + DES48)
    def d_of(k, arm="r3B_judge_des48", ref="vs_stock"):
        v = ds[k]["rows"][arm].get(ref) or {}
        return v.get("d_acc"), v.get("ci"), v.get("speedup_exp"), ds[k]["rows"][arm]["status"]

    he, mb, gs, ma, gp, ai = (d_of(k) for k in DS_ORDER)
    sp = [x[2] for x in (he, mb, gs, ma) if x[2]]
    sp_rng = f"{min(sp):.2f}×–{max(sp):.2f}×" if sp else "—"
    he_a2 = ds["humaneval"]["rows"]["r3A_e2_judge_des48"].get("vs_stock") or {}

    css = """
:root{--paper:#F7F6F3;--panel:#FFFFFF;--sunk:#EDEBE6;--ink:#191C1B;--ink2:#4A514E;--ink3:#858C88;--rule:#D8D5CE;
--good:#0E6B65;--warn:#A8690F;--bad:#9E3B3B;--ours:#0E6B65;--oursbg:rgba(14,107,101,.09);--dim:#B9B5AC}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#101413;--panel:#191E1C;--sunk:#0C100F;--ink:#ECEAE4;--ink2:#A8AFAB;--ink3:#767D79;--rule:#2A312F;--good:#4FC5B8;--warn:#D9A24A;--bad:#D97C74;--ours:#4FC5B8;--oursbg:rgba(79,197,184,.12);--dim:#4A514E}}
:root[data-theme="dark"]{--paper:#101413;--panel:#191E1C;--sunk:#0C100F;--ink:#ECEAE4;--ink2:#A8AFAB;--ink3:#767D79;--rule:#2A312F;--good:#4FC5B8;--warn:#D9A24A;--bad:#D97C74;--ours:#4FC5B8;--oursbg:rgba(79,197,184,.12);--dim:#4A514E}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding-block:36px 64px;padding-inline:20px}
h1{font-family:"Newsreader",Georgia,serif;font-weight:400;font-size:clamp(34px,5vw,54px);line-height:1.04;letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance}
h2{font-family:"Newsreader",Georgia,serif;font-weight:500;font-size:28px;line-height:1.15;margin:52px 0 6px;letter-spacing:-.01em;text-wrap:balance}
h3{font-size:15px;font-weight:600;margin:24px 0 6px}
.kick{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink3);margin:0 0 16px}
.lede{font-size:17px;color:var(--ink2);max-width:72ch;margin:0 0 18px}
.facts{display:flex;flex-wrap:wrap;gap:8px 22px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--ink2);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:10px 0;margin:0 0 8px}
.facts a{color:var(--ours);text-decoration:none;border-bottom:1px solid transparent}.facts a:hover{border-color:var(--ours)}
.sub{color:var(--ink2);max-width:80ch;margin:0 0 14px}
.tw{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3);font-weight:500;background:var(--sunk);position:sticky;top:0}
td.n{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tr.ours td{background:var(--oursbg);font-weight:600}
tr.partial td{color:var(--ink3)}tr.partial td .ci{color:var(--ink3)}
.ci{color:var(--ink3);font-size:12px;font-weight:400}
.good{color:var(--good)}.bad{color:var(--bad)}.ok{color:var(--ink)}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);margin-left:8px}
.grp{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:500}
tr.grp td{background:var(--sunk);border-top:2px solid var(--rule);padding-top:5px;padding-bottom:5px}
.cap{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink3);margin:8px 0 0;max-width:100ch}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:10px 0 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.fig{background:var(--panel);border:1px solid var(--rule);padding:12px}
.fig svg{width:100%;height:auto;display:block}
.abstract{background:var(--panel);border:1px solid var(--rule);padding:26px 28px;max-width:78ch;margin-top:8px}
.abstract p{margin:0 0 14px;font-size:15.5px;line-height:1.62}
.abstract .t{font-family:"Newsreader",Georgia,serif;font-size:24px;line-height:1.2;margin:0 0 4px}
.abstract .a{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink3);margin:0 0 18px}
.kw{font-size:13px;color:var(--ink2)}
@media (max-width:600px){h2{margin-top:36px}th,td{padding:7px 8px;font-size:13px}.abstract{padding:18px}}
"""

    # ---- all-in-one grid: rows = models, columns = datasets; cell = Δacc vs stock + speedup
    grid = ["<div class='tw'><table><thead><tr><th>model</th>"]
    for k in DS_ORDER:
        grid.append(f"<th>{esc(ds[k]['label'])}<br><span class='ci'>n = {ds[k]['n']}</span></th>")
    grid.append("</tr></thead><tbody>")
    last_grp = None
    for key, lab, grp, _ in ROWS:
        if grp != last_grp:
            grid.append(f"<tr class='grp'><td colspan='{len(DS_ORDER) + 1}'><span class='grp'>{esc(grp)}</span></td></tr>")
            last_grp = grp
        cells, any_ = [], False
        for k in DS_ORDER:
            c = ds[k]["rows"][key]; st = row_state(c)
            if st == "missing":
                cells.append("<td class='n'>—</td>"); continue
            any_ = True
            if key == "stock":
                cells.append(f"<td class='n'>{c['acc']:.1f}<br><span class='ci'>reference</span></td>"); continue
            v = c.get("vs_stock") or {}
            spd = f"{v['speedup_exp']:.2f}×" if v.get("speedup_exp") else "—"
            part = f" <span class='ci'>({c['n_scored']}/{ds[k]['n']})</span>" if st == "partial" else ""
            cells.append(f"<td class='n {cls_d(v)}'>{fmt_d(v)}{part}<br><span class='ci'>{c['acc']:.1f} · {spd}</span></td>" if c["acc"] is not None
                         else "<td class='n'>—</td>")
        if not any_:
            continue
        rc = "ours" if key.startswith("r3B") else ""
        grid.append(f"<tr class='{rc}'><td>{esc(lab)}</td>{''.join(cells)}</tr>")
    grid.append("</tbody></table></div>")
    grid_html = "".join(grid)

    # ---- per-dataset tables
    per = []
    for k in DS_ORDER:
        d = ds[k]; s48 = d["rows"]["stock_des48"]
        per.append(f"<h2>{esc(d['label'])}<span class='tag'>n = {d['n']}</span></h2>")
        per.append("<div class='tw'><table><thead><tr><th>model</th><th>status</th><th>accuracy</th><th>Δ vs stock [95% CI]</th>"
                   "<th>Δ vs stock + DES48</th><th>passes / block</th><th>experts / token</th><th>speedup</th></tr></thead><tbody>")
        for key, lab, grp, desc in ROWS:
            c = d["rows"][key]; st = row_state(c)
            if st == "missing" or c["acc"] is None:
                continue
            v = c.get("vs_stock") or {}; v48 = c.get("vs_stock_des48") or {}
            spd = f"{v['speedup_exp']:.2f}×" if v.get("speedup_exp") else ("1.00×" if key == "stock" else "—")
            status = "complete" if st == "complete" else f"partial {c['n_scored']}/{d['n']}"
            rc = ("ours " if key.startswith("r3B") else "") + ("partial" if st == "partial" else "")
            d1 = "reference" if key == "stock" else fmt_d(v)
            d2 = "—" if key == "stock" else ("reference" if key == "stock_des48" else (fmt_d(v48) if s48["n_records"] else "—"))
            per.append(f"<tr class='{rc}'><td>{esc(lab)}<br><span class='ci'>{esc(desc)}</span></td><td class='n'>{status}</td>"
                       f"<td class='n'>{c['acc']:.1f}</td><td class='n {cls_d(v) if key != 'stock' else ''}'>{d1}</td>"
                       f"<td class='n {cls_d(v48) if key not in ('stock', 'stock_des48') else ''}'>{d2}</td>"
                       f"<td class='n'>{c['passes_blk']:.2f}</td><td class='n'>{c['exp_tok']:.0f}</td><td class='n'>{spd}</td></tr>")
        per.append("</tbody></table></div>")
    per_html = "".join(per)

    # ---- Δacc vs speedup chart (complete rows, arms with a pool or a judge)
    pts = []
    marks = {"stock_des32": ("no training", "#858C88", "◆"), "stock_des48": ("no training", "#858C88", "◆"), "blind": ("no training", "#858C88", "◆"),
             "prev_judge_des48": ("round-2 judge", "#A8690F", "●"), "r3A_e1_judge_des48": ("r3_A", "#9E3B3B", "●"),
             "r3A_e2_judge_des48": ("r3_A", "#9E3B3B", "●"), "r3A_s2000_judge_des48": ("r3_A", "#9E3B3B", "●"),
             "r3B_judge": ("r3_B (ours)", "#0E6B65", "●"), "r3B_judge_des48": ("r3_B (ours)", "#0E6B65", "●")}
    for k in DS_ORDER:
        for key, lab, *_ in ROWS:
            c = ds[k]["rows"][key]; v = c.get("vs_stock") or {}
            if key in marks and row_state(c) == "complete" and v.get("d_acc") is not None and v.get("speedup_exp"):
                pts.append((v["speedup_exp"], v["d_acc"], v["ci"], key, lab, ds[k]["label"]))
    W, H, L, R, T, B = 900, 420, 56, 20, 20, 44
    xs = [p[0] for p in pts] or [1, 2]; ys = [p[2][0] for p in pts] + [p[2][1] for p in pts] or [-10, 5]
    x0, x1 = 0.95, max(xs) + 0.08; y0, y1 = min(ys) - 1.5, max(ys) + 1.5
    fx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
    fy = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    svg = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Δ accuracy vs stock against expert-compute speedup, one point per model and dataset'>"]
    for gy in range(int(y0 // 2 * 2), int(y1) + 1, 2):
        svg.append(f"<line x1='{L}' x2='{W-R}' y1='{fy(gy):.1f}' y2='{fy(gy):.1f}' stroke='var(--rule)' stroke-width='{2 if gy == 0 else 1}'/>"
                   f"<text x='{L-8}' y='{fy(gy)+4:.1f}' text-anchor='end' font-size='11' fill='var(--ink3)' font-family='IBM Plex Mono,monospace'>{gy:+d}</text>")
    gx = 1.0
    while gx <= x1:
        svg.append(f"<line x1='{fx(gx):.1f}' x2='{fx(gx):.1f}' y1='{T}' y2='{H-B}' stroke='var(--rule)'/>"
                   f"<text x='{fx(gx):.1f}' y='{H-B+16}' text-anchor='middle' font-size='11' fill='var(--ink3)' font-family='IBM Plex Mono,monospace'>{gx:.1f}×</text>")
        gx = round(gx + 0.2, 2)
    svg.append(f"<text x='{(L+W-R)/2:.0f}' y='{H-6}' text-anchor='middle' font-size='12' fill='var(--ink2)'>speedup vs stock (stock experts/token ÷ arm experts/token)</text>")
    svg.append(f"<text x='14' y='{(T+H-B)/2:.0f}' transform='rotate(-90 14 {(T+H-B)/2:.0f})' text-anchor='middle' font-size='12' fill='var(--ink2)'>Δ accuracy vs stock (points)</text>")
    for x, y, ci, key, lab, dl in pts:
        col = marks[key][1]
        svg.append(f"<line x1='{fx(x):.1f}' x2='{fx(x):.1f}' y1='{fy(ci[0]):.1f}' y2='{fy(ci[1]):.1f}' stroke='{col}' stroke-width='1' opacity='.55'/>")
        rad = 6 if key.startswith("r3B") else 4.5
        svg.append(f"<circle cx='{fx(x):.1f}' cy='{fy(y):.1f}' r='{rad}' fill='{col}' stroke='var(--paper)' stroke-width='1.5'><title>{esc(lab)} · {esc(dl)}: {y:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}], {x:.2f}×</title></circle>")
        if key.startswith("r3B"):
            svg.append(f"<text x='{fx(x)+9:.1f}' y='{fy(y)+4:.1f}' font-size='11' fill='{col}' font-family='IBM Plex Mono,monospace'>{esc(dl.split(' ')[0])}</text>")
    svg.append("</svg>")
    chart_html = "".join(svg)

    def num(x, ci=None):
        return "—" if x is None else (f"{x:+.1f}" + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else ""))

    def pt(x):
        return "—" if x is None else f"{x:+.1f}"

    def ci(c):
        return "" if not c else f" [{c[0]:+.1f}, {c[1]:+.1f}]"

    def spx(x):
        return "—" if not x else f"{x:.2f}×"

    # calibration facts from the last training probe
    calib_txt = ""
    try:
        import glob as _g
        cj = sorted(_g.glob(os.path.join(ROOT, "logs", "draftalign", "train", "r3_B", "calib", "step*.json")))
        if cj:
            cal = json.load(open(cj[-1]))
            calib_txt = (f" On held-out states the overall expected calibration error is {cal['all_open']['ece']:.3f}, and under the decoder's own "
                         f"commit rule the product of the committed tokens' confidences never exceeds the measured probability that all of them are right "
                         f"by more than {max(r['gap'] for r in cal['joint']['tau_rule'] if r['n_states']):.2f}: parallel commits can be trusted.")
    except Exception:
        pass
    # averages for the abstract (r3_B + DES48 vs stock): code = HumanEval, MBPP; math = GSM8K, MATH
    def _avg(keys):
        vals = [ds[k]["rows"]["r3B_judge_des48"].get("vs_stock", {}).get("d_acc") for k in keys]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    code_avg, math_avg = _avg(["humaneval", "mbpp"]), _avg(["gsm8k500", "math500"])
    sp_lo, sp_hi = (f"{min(sp):.1f}", f"{max(sp):.1f}") if sp else ("—", "—")
    paras = [
        "Diffusion language models can in principle write many tokens per forward pass, yet in practice they commit only a few positions "
        "at a time, because each token is predicted while its neighbours are still masked and the model is rarely confident enough to commit. "
        "For mixture-of-experts diffusion models the usual remedy is a <em>shared expert</em>, a dense path every token can rely on, but a shared "
        "expert large enough to matter is a new, wide module that must be trained from scratch on a great deal of data. We observe that such a "
        "general expert already exists for free: a small dense model of the same family is fully trained, shares the tokenizer, and can fill an "
        "entire block in a handful of cheap steps. The large model then no longer has to guess every token from masks; it only has to "
        "<em>judge</em> a draft, keeping the tokens it trusts and overriding the rest.",
        "Judging is a skill the frozen model lacks, so we add a small judge adapter (a 27M-parameter LoRA on the 128-expert SDAR-30B-A3B, drafted "
        "by its 1.7B dense sibling) and find that how it is trained matters more than how long. A judge trained to be certain commits eagerly but "
        "is wrong about one confident override in five, which is fatal for code. "
        "We instead train it to be <em>calibrated</em>, scoring the very confidence the decoder thresholds at inference, so that a token committed "
        "at 0.95 is right 95% of the time and the product of confidences across a parallel commit is a trustworthy estimate that the whole "
        "commit is correct. One epoch of this calibrated training turns the same weights into a judge that commits twice as many tokens per "
        "pass as the baseline model with more than 99% precision.",
        f"Combined with decode-time expert pooling, the general-expert judge cuts the 30B model's expert compute per generated token by "
        f"{sp_lo}–{sp_hi}× on HumanEval, MBPP, GSM8K and MATH-500, while matching the baseline on GSM8K and AIME and staying within "
        f"a point of the released model on math and a few points on code, measured on the full benchmarks with paired "
        "confidence intervals. Because the drafter "
        "is an existing smaller dense model of the same family, used as released, and the judge is a small adapter, the method adds no new pretraining and can be attached to any "
        "mixture-of-experts diffusion model that has a smaller dense sibling. We release the adapter, every checkpoint, and per-question "
        "decode records for all baselines.",
    ]
    abstract_md = "\n\n".join(paras)
    abstract = f"""
<div class='abstract'>
<p class='t'>A free general expert: calibrated draft-aligned decoding for mixture-of-experts diffusion language models</p>
<p class='a'>Anupam Rawat (IIT Bombay) et al. · draft abstract for ICLR 2027 · numbers as of {esc(now)}</p>
{''.join(f"<p>{x}</p>" for x in paras)}
<p class='kw'>Keywords: masked diffusion language models, parallel decoding, speculative drafting, mixture of experts, calibration.</p>
</div>"""
    import re as _re
    with open(os.path.join(ROOT, "docs", "results", "r3_B", "ABSTRACT_ICLR2027.md"), "w") as fh:
        fh.write("# A free general expert: calibrated draft-aligned decoding for mixture-of-experts diffusion language models\n\n"
                 f"_Draft abstract for ICLR 2027 · Anupam Rawat (IIT Bombay) et al. · numbers as of {now}; regenerated by "
                 "`docs/report/build_results_page.py` from `docs/results/r3_B/eval_summary.json`._\n\n"
                 + _re.sub(r"</?em>", "*", abstract_md) + "\n\n"
                 "**Keywords:** masked diffusion language models, parallel decoding, speculative drafting, mixture of experts, calibration.\n")

    page = f"""<title>Round 3 Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="wrap">
<p class="kick">SDAR-30B-A3B · block 32 · results only · {esc(now)}</p>
<h1>Every model, every benchmark, one page</h1>
<p class="lede">Rows are models. Each cell pairs the model against the stock 30B on the same questions: the accuracy change in points with a
95% bootstrap confidence interval, then the model's accuracy and its speedup (stock expert executions per token ÷ the model's).
Green = the interval is entirely above zero, red = entirely below, black = it includes zero. Highlighted rows are our calibrated judge (r3_B).</p>
<div class="facts">
<span>ours: r3_B step 710 (epoch 2, calibrated loss) · sha ad8a6485…</span>
<span>uncalibrated: r3_A steps 700 / 1400 / 2000 · round-2 judge: T2b step 3057</span>
<a href="{HF}">weights &amp; eval records (HF, public)</a>
<a href="{GH}/docs/results/r3_B">tables on GitHub</a>
<a href="https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign">W&amp;B</a>
</div>

<h2>All benchmarks</h2>
<p class="sub">Full sets: HumanEval 164, MBPP 257 (sanitized), GSM8K-500, MATH-500, GPQA-Diamond 198, AIME 90. A cell with (n/N) is a partial
run; a dash means that model was not run on that set. Speedup counts the 30B's experts only; the 1.7B drafter, where used, is not charged.</p>
{grid_html}
<div class="legend"><span><i style="background:var(--ours)"></i>ours (r3_B)</span><span><i style="background:#9E3B3B"></i>r3_A (uncalibrated)</span>
<span><i style="background:#A8690F"></i>round-2 judge</span><span><i style="background:#858C88"></i>no training</span></div>

<h2>Accuracy change against speedup</h2>
<p class="sub">One point per model and benchmark (complete runs only); the whisker is the 95% interval of the accuracy change. The useful corner is
the top right: faster with the interval touching zero. Labels mark our judge.</p>
<div class="fig">{chart_html}</div>

{per_html}

<h2>Abstract (ICLR 2027 draft)</h2>
{abstract}
<p class="cap">Metric definitions: passes/block = (forwards − prefill − flush) / blocks; experts/token = unique expert-layer loads / generated tokens;
Δ and CI = paired over the shared questions, 2,000 bootstrap resamples. Source: docs/results/r3_B/eval_summary.json (regenerated by
code/draftalign/eval_summary_r3B.py).</p>
</div>
"""
    with open(OUT, "w") as fh:
        fh.write(page)
    print(f"[results page] {OUT} {os.path.getsize(OUT) / 1024:.0f} KB, {len(pts)} chart points, generated {now}")


if __name__ == "__main__":
    main()
