#!/usr/bin/env python
"""'Why the curves flattened' -- the plateau diagnosis section of docs/report/r3_report.html.

Imported by build_r3_report.py (render(H, tf, ev, specs) -> (section_html, summary_li)); also runnable on its own
to print the numbers:  /venv/r3/bin/python docs/report/plateau_r3.py

Sources (read-only, re-read on every render, so the section fills in as data lands):
  logs/draftalign/train/r3_A/train.rank0.jsonl   train rows every 10 steps. A resumed run appends: a step that is <= the
                                                 previous row's step starts a new segment. The merged trajectory keeps
                                                 the LAST occurrence of each step (the resumed run); phase fits for the
                                                 original run use the original segment only.
  logs/draftalign/plateau_probe_r3/<ckpt>.json   code/draftalign/probe_plateau_r3.py, one per checkpoint
                                                 (stock = step 0, 200, 400, 700, 1000, 1400, 1500); groups keyed
                                                 <set>|<pool>|<kind> with n, acc, ce, top1, keep_draft, commit_share,
                                                 precision (tau 0.95).
Accuracy on train rows is POOLED (sum n_correct / sum n_rows), not the logged per-rank-mean `acc`. `top1` in the logs
is the mean top-1 probability (a confidence), not top-1 accuracy.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_JSONL = os.path.join(ROOT, "logs", "draftalign", "train", "r3_A", "train.rank0.jsonl")
PROBE_DIR = os.path.join(ROOT, "logs", "draftalign", "plateau_probe_r3")
CLIP = 1.0
PEAK_LR, MIN_LR = 2e-5, 1e-5
CKPTS = [("stock", 0), ("step000200", 200), ("step000400", 400), ("step000700", 700),
         ("step001000", 1000), ("step001400", 1400), ("step001500", 1500)]
PHASES = [("200–700", 200, 700), ("700–1,100", 700, 1100), ("1,100–1,560", 1100, 1560), ("700–1,560 overall", 700, 1560)]
KINDS = [("draft_ok", "confirm: draft is right", "var(--k-ok)", "holdout_pass1"),
         ("draft_bad", "override: draft is wrong", "var(--k-bad)", "holdout_pass1"),
         ("mask", "masked slot", "var(--k-mask)", "holdout_mixed")]


# ---------------------------------------------------------------------------------------------------------------
# train log
# ---------------------------------------------------------------------------------------------------------------
def pooled_acc(r):
    return 100.0 * r["n_correct"] / r["n_rows"]


METRICS = [  # key, label, unit, value fn, weight by n_rows, decimals
    ("acc", "Accuracy on open slots (pooled)", "pts", pooled_acc, True, 2),
    ("gap", "Overconfidence gap: mean top-1 probability − accuracy", "pts", lambda r: 100.0 * r["top1"] - pooled_acc(r), True, 2),
    ("ce", "Cross-entropy", "nats", lambda r: r["ce"], True, 3),
    ("cert", "Certainty term (entropy on right slots)", "", lambda r: r["cert"], True, 4),
    ("grad_norm", "Gradient norm, pre-clip", "", lambda r: r["grad_norm"], False, 3),
]


def load_train(path=TRAIN_JSONL):
    segs, prev = [], None
    try:
        fh = open(path)
    except OSError:
        return {"segments": [], "original": [], "resumed": [], "merged": []}
    for line in fh:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" not in r or not r.get("n_rows"):
            continue
        if prev is None or r["step"] <= prev:
            segs.append([])
        r = dict(r, _seg=len(segs) - 1)
        segs[-1].append(r); prev = r["step"]
    merged = {}
    for s in segs:
        for r in s:
            merged[r["step"]] = r
    return {"segments": segs, "original": segs[0] if segs else [], "resumed": [r for s in segs[1:] for r in s],
            "merged": [merged[k] for k in sorted(merged)]}


def fit(rows, lo, hi, f, weighted):
    pts = [(r["step"], f(r), r["n_rows"] if weighted else 1.0) for r in rows if lo <= r["step"] <= hi]
    if len(pts) < 6:
        return None
    x = np.array([p[0] for p in pts], float) / 100.0; y = np.array([p[1] for p in pts], float)
    w = np.array([p[2] for p in pts], float)
    X = np.c_[np.ones_like(x), x]; XtW = X.T * w
    b = np.linalg.solve(XtW @ X, XtW @ y); res = y - X @ b
    s2 = float((w * res ** 2).sum() / (len(x) - 2)); cov = s2 * np.linalg.inv(XtW @ X)
    return {"slope": float(b[1]), "ci": 1.96 * math.sqrt(cov[1, 1]), "n": len(x), "lo": int(pts[0][0]), "hi": int(pts[-1][0]),
            "mean": float((w * y).sum() / w.sum())}


def windows(rows, width=100):
    by = {}
    for r in rows:
        by.setdefault((r["step"] - 1) // width, []).append(r)
    out = []
    for k in sorted(by):
        rs = by[k]; nr = sum(r["n_rows"] for r in rs)
        post = [r for r in rs if r["step"] > 100]
        out.append({"lo": k * width, "hi": (k + 1) * width, "n": len(rs),
                    "acc": 100.0 * sum(r["n_correct"] for r in rs) / nr,
                    "top1": 100.0 * sum(r["top1"] * r["n_rows"] for r in rs) / nr,
                    "ce": sum(r["ce"] * r["n_rows"] for r in rs) / nr, "cert": sum(r["cert"] * r["n_rows"] for r in rs) / nr,
                    "gn": sum(r["grad_norm"] for r in rs) / len(rs),
                    "clipped": (sum(r["grad_norm"] > CLIP for r in post) / len(post)) if post else None,
                    "lr": rs[-1]["lr"], "resumed": any(r["_seg"] > 0 for r in rs)})
    return out


# ---------------------------------------------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------------------------------------------
def load_probes(d=PROBE_DIR):
    out = {}
    for name, _ in CKPTS:
        try:
            out[name] = json.load(open(os.path.join(d, name + ".json")))
        except (OSError, json.JSONDecodeError):
            out[name] = None
    return out


def grp(P, ck, s, pool, kind):
    d = P.get(ck)
    if not d:
        return None
    g = d.get("groups", {}).get(f"{s}|{pool}|{kind}")
    return g if g and g.get("n") else None


def pool_label(k):
    kl = k.lower()
    if "code" in kl:
        return "code (OpenCodeInstruct)"
    if "gsm8k" in kl or kl == "train":          # the GSM8K pool's split tag is "train" (openai/gsm8k train split)
        return "maths (GSM8K)"
    if kl.startswith("if"):
        return "verifiable IF"
    if kl == "all":
        return "all pools"
    return k


def pools_in(P):
    ks = set()
    for d in P.values():
        if d:
            ks |= {key.split("|")[1] for key in d.get("groups", {})}
    order = lambda k: (0 if "code" in k else 1 if ("gsm8k" in k or k == "train") else 2 if k.startswith("if") else 4 if k == "all" else 3, k)
    return sorted(ks, key=order)


# ---------------------------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------------------------
def render(H, tf, ev, specs):
    esc, num = H.esc, H.num
    T = load_train(); P = load_probes()
    have = [ck for ck, _ in CKPTS if P[ck]]; missing = [ck for ck, _ in CKPTS if not P[ck]]
    step_of = dict(CKPTS)
    orig, res, merged = T["original"], T["resumed"], T["merged"]

    def pct(g, key="acc", nd=1):
        return "—" if not g or g.get(key) is None else f"{100 * g[key]:.{nd}f}"

    late_ck = next((ck for ck in ("step001500", "step001400") if P[ck]), None)

    def pool_moves(s, kind, a="step000700"):
        out = []
        for pool in pools_in(P):
            if pool == "all" or not late_ck:
                continue
            ga, gb = grp(P, a, s, pool, kind), grp(P, late_ck, s, pool, kind)
            if ga and gb:
                out.append((pool, ga, gb, 100 * (gb["acc"] - ga["acc"])))
        return out

    def legend(items):
        return '<ul class="legend">' + "".join(f'<li><span class="lk" style="border-color:{c}"></span>{esc(t)}</li>' for c, t in items) + "</ul>"

    # ---- 1. phase fits ------------------------------------------------------------------------------------------
    phases = list(PHASES)
    if len(res) >= 6:
        phases.append((f"resumed {res[0]['step']:,}–{res[-1]['step']:,}", res[0]["step"], res[-1]["step"]))
    fits = {}
    for key, label, unit, f, wtd, dec in METRICS:
        for pl, lo, hi in phases:
            rows = res if pl.startswith("resumed") else orig
            fits[(key, pl)] = fit(rows, lo, hi, f, wtd)

    def chip(ft, key):
        if not ft:
            return '<span class="chip">pending</span>'
        lo, hi = ft["slope"] - ft["ci"], ft["slope"] + ft["ci"]
        if lo <= 0 <= hi:
            return '<span class="chip">flat: CI spans 0</span>'
        up = ft["slope"] > 0
        good = {"acc": up, "gap": not up, "ce": not up, "cert": None, "grad_norm": None}[key]
        word = "rising" if up else "falling"
        return f'<span class="chip {"c-good" if good else "c-bad" if good is False else "c-info"}">{word}</span>'

    head = "".join(f'<th scope="col" class="num">{esc(pl)}</th>' for pl, _, _ in phases)
    body = []
    for key, label, unit, f, wtd, dec in METRICS:
        cells = []
        for pl, _, _ in phases:
            ft = fits[(key, pl)]
            if ft:
                cells.append(f'<td class="num">{num(ft["slope"], dec, True)} <span class="ci">± {ft["ci"]:.{dec}f}</span>'
                             f'<span class="nn">{chip(ft, key)} · mean {num(ft["mean"], dec)} · n={ft["n"]}</span></td>')
            else:
                cells.append('<td class="num muted">pending</td>')
        body.append(f'<tr><th scope="row">{esc(label)}<span class="nn">{esc((unit + " ") if unit else "")}per 100 steps</span></th>{"".join(cells)}</tr>')
    fit_table = (f'<div class="tscroll"><table class="data"><thead><tr><th scope="col">train-log metric</th>{head}</tr></thead>'
                 f'<tbody>{"".join(body)}</tbody></table></div>')

    W = windows(merged)
    wrows = "".join(
        f'<tr><th scope="row">{w["lo"]:,}–{w["hi"]:,}{" <span class=\"chip c-info\">resumed</span>" if w["resumed"] else ""}</th>'
        f'<td class="num">{w["n"]}</td><td class="num strong">{w["acc"]:.1f}</td><td class="num">{w["top1"]:.1f}</td>'
        f'<td class="num">{w["top1"] - w["acc"]:.1f}</td><td class="num">{w["ce"]:.3f}</td><td class="num">{w["cert"]:.3f}</td>'
        f'<td class="num">{w["gn"]:.2f}</td><td class="num">{"—" if w["clipped"] is None else f"{100 * w["clipped"]:.0f}%"}</td>'
        f'<td class="num">{w["lr"] * 1e5:.2f}</td></tr>' for w in W)
    win_table = ('<div class="tscroll"><table class="data"><thead><tr><th scope="col">steps</th><th scope="col" class="num">rows</th>'
                 '<th scope="col" class="num">acc %</th><th scope="col" class="num">mean top-1 prob %</th><th scope="col" class="num">gap pts</th>'
                 '<th scope="col" class="num">CE</th><th scope="col" class="num">certainty</th><th scope="col" class="num">grad norm</th>'
                 '<th scope="col" class="num">steps clipped</th><th scope="col" class="num">lr ×10⁻⁵ (end)</th></tr></thead>'
                 f"<tbody>{wrows}</tbody></table></div>")

    wmid = lambda w: w["hi"]
    svg1, sp = H.line_chart("pl-acc", "Windowed accuracy and confidence", [
        {"name": "accuracy (pooled)", "pts": [(wmid(w), w["acc"]) for w in W if w["hi"] > 100], "cls": "eval", "color": "var(--k-ok)"},
        {"name": "mean top-1 probability", "pts": [(wmid(w), w["top1"]) for w in W if w["hi"] > 100], "cls": "eval", "color": "var(--k-mask)"}],
        dec=1, suffix="%", cap100=True)
    specs.append(sp)
    svg2, sp = H.line_chart("pl-gap", "Overconfidence gap", [
        {"name": "top-1 prob − accuracy", "pts": [(wmid(w), w["top1"] - w["acc"]) for w in W if w["hi"] > 200], "cls": "eval", "color": "var(--k-bad)"}],
        dec=1, suffix=" pts")
    specs.append(sp)
    svg3, sp = H.line_chart("pl-gn", "Gradient norm", [
        {"name": "pre-clip grad norm", "pts": [(wmid(w), w["gn"]) for w in W if w["hi"] > 100], "cls": "eval", "color": "var(--s-stock)"}], dec=2)
    specs.append(sp)
    first_w = next((w for w in W if w["hi"] == 800), None); last_w = W[-1] if W else None
    gap_txt = (f"{first_w['top1'] - first_w['acc']:.1f} pts in steps 700–800 to {last_w['top1'] - last_w['acc']:.1f} pts in steps "
               f"{last_w['lo']:,}–{last_w['hi']:,}") if first_w and last_w else "—"
    post = [r for r in merged if r["step"] > 100]
    clipped_all = sum(r["grad_norm"] > CLIP for r in post) / len(post) if post else None
    gmin = min((r["grad_norm"] for r in post), default=None)

    fa = fits[("acc", "700–1,560 overall")]; fg = fits[("gap", "700–1,560 overall")]; fc = fits[("cert", "700–1,560 overall")]
    fce = fits[("ce", "700–1,560 overall")]; f0 = fits[("acc", "200–700")]
    lr_last = merged[-1]["lr"] if merged else None

    part1 = f"""
<h3>1 · Is it learning at all?</h3>
<p>Train rows are logged every 10 steps, and each carries its own <code>n_correct</code> and <code>n_rows</code>. Accuracy below is <strong>pooled</strong>
(Σ n_correct ÷ Σ n_rows). The logged <code>acc</code> field is a mean of per-rank ratios and reads up to 11 pts higher on single rows. <code>top1</code> in the log is the
<strong>mean top-1 probability</strong>, a confidence and not an accuracy, so top-1 minus accuracy measures overconfidence. Slopes are weighted least squares on
the 10-step rows, weighted by n_rows (unweighted for grad norm). They are reported in units per 100 steps with ± 1.96 SE, which treats the rows as independent. The pre-stop phases use the original run only.
{"The resumed run is fitted separately." if len(res) >= 6 else "The resumed run gets its own column once it has logged at least 60 steps."}</p>
{fit_table}
<p><strong>Reading.</strong> {f"Accuracy rose {num(f0['slope'], 2, True)} ± {f0['ci']:.2f} pts per 100 steps over steps 200–700." if f0 else ""}
{f"From step 700 to the stop the slope is {num(fa['slope'], 2, True)} ± {fa['ci']:.2f}, flat, with a 95% interval that rules out even half a point of gain per 100 steps." if fa and fa['slope'] + fa['ci'] < 0.5 else ""}
{f"Over the same stretch CE is flat ({num(fce['slope'], 3, True)} ± {fce['ci']:.3f}), while the certainty term keeps falling ({num(fc['slope'], 4, True)} ± {fc['ci']:.4f}) and the overconfidence gap keeps widening ({num(fg['slope'], 2, True)} ± {fg['ci']:.2f}): {esc(gap_txt)}." if fc and fg and fce else ""}
In short, after epoch 1 the model stopped getting more slots right, but kept getting more confident about the slots it was already predicting.</p>
<div class="grid2">{H.fig("Accuracy vs confidence, 100-step windows", "", legend([("var(--k-ok)", "accuracy (pooled)"), ("var(--k-mask)", "mean top-1 probability")]) + svg1, "Each dot is one 100-step window, plotted at its last step.")}
{H.fig("Overconfidence gap", "", svg2, "Mean top-1 probability minus pooled accuracy, per window.")}
{H.fig("Gradient norm before clipping", "", svg3, esc(f"Clip threshold {CLIP}. {'' if clipped_all is None else f'{100 * clipped_all:.0f}% of logged steps after step 100 exceed it (minimum {gmin:.2f}).'}"))}</div>
<p><strong>About the grad norm.</strong> The pre-clip norm drifts up from about 4.4 to about 5 while the clip is 1.0.
{'' if clipped_all is None else f'Every logged step after step 100 is clipped ({100 * clipped_all:.0f}%).'}
Clipping multiplies the gradient by 1/‖g‖. AdamW then divides by the running root-mean-square of past gradients, so a roughly constant rescale largely cancels: it does <em>not</em> shrink the effective step the way a lower learning rate would.
What it does do is re-weight steps, with large-gradient batches counting relatively less. A rising pre-clip norm next to a flat loss is consistent with the optimiser moving along a noisy, flat valley rather than descending. It is not evidence that the steps were too small.</p>
<details class="cmd"><summary>100-step windows (merged trajectory, last occurrence kept)</summary>{win_table}</details>"""

    # ---- 2. train vs holdout --------------------------------------------------------------------------------------
    miss_txt = (f'<p class="muted">Probe JSONs present: {len(have)}/7 ({esc(", ".join(have) or "none")}). '
                f'Pending: {esc(", ".join(missing))}. These come from <code>code/draftalign/probe_plateau_r3.py</code>, run by <code>resume_r3_after_evals.sh</code> before training resumes.</p>') if missing else ""
    tv = [(ck, grp(P, ck, "train_pass1", "all", "all"), grp(P, ck, "holdout_pass1", "all", "all")) for ck in have]
    if tv:
        svg, sp = H.line_chart("pl-tvh", "train vs holdout pass-1 accuracy", [
            {"name": "train split, pass 1", "pts": [(step_of[ck], t["acc"]) for ck, t, h in tv if t], "cls": "eval", "color": "var(--k-ok)"},
            {"name": "holdout, pass 1", "pts": [(step_of[ck], h["acc"]) for ck, t, h in tv if h], "cls": "eval", "color": "var(--k-bad)"}],
            scale=100.0, dec=1, suffix="%", cap100=True)
        specs.append(sp)
        trows = "".join(f'<tr><th scope="row"><span class="mono">{ck}</span></th><td class="num">{pct(t)}</td><td class="num">{pct(h)}</td>'
                        f'<td class="num">{num(100 * (t["acc"] - h["acc"]), 1, True) if t and h else "—"}</td>'
                        f'<td class="num">{(t or {}).get("n", "—")} / {(h or {}).get("n", "—")}</td></tr>' for ck, t, h in tv)
        tv_html = (H.fig("Pass-1 accuracy: train split vs holdout", "", legend([("var(--k-ok)", "train split"), ("var(--k-bad)", "holdout")]) + svg,
                       "Stock is plotted at step 0: the adapter starts as an exact identity.")
                   + '<div class="tscroll"><table class="data"><thead><tr><th scope="col">checkpoint</th><th scope="col" class="num">train acc %</th>'
                     '<th scope="col" class="num">holdout acc %</th><th scope="col" class="num">train − holdout</th><th scope="col" class="num">slots (train / holdout)</th>'
                     f"</tr></thead><tbody>{trows}</tbody></table></div>")
        latest = next((ck for ck in ("step001500", "step001400", "step001000") if P[ck]), None)
        t, h = (grp(P, latest, "train_pass1", "all", "all"), grp(P, latest, "holdout_pass1", "all", "all")) if latest else (None, None)
        e7 = grp(P, "step000700", "train_pass1", "all", "all")
        gap_tv = 100 * (t["acc"] - h["acc"]) if t and h else None
        tv_read = (f"At <span class=\"mono\">{latest}</span>, train-split accuracy is {pct(t)}% against {pct(h)}% on the holdout, a gap of {num(gap_tv, 1, True)} pts. "
                   + ("Both sit close together, so the adapter is not memorising its 22.7k training examples: this is a plateau, not overfitting. " if gap_tv is not None and abs(gap_tv) < 3 else
                      "The gap is large enough to suggest some memorisation; compare the trend across checkpoints. " if gap_tv is not None else "")
                   + (f"Train-split accuracy moved {num(100 * (t['acc'] - e7['acc']), 1, True)} pts from step 700." if t and e7 else "")) if latest else ""
    else:
        tv_html, tv_read = '<div class="placeholder">pending: no probe JSONs yet.</div>', ""
    part2 = f"""
<h3>2 · Train vs holdout: plateau or overfitting?</h3>
<p>The plateau probe scores the same deterministic pass-1 states (all open slots hold the draft, the inference state) on 160 training-split examples and on 160 holdout examples, for every checkpoint.
If the model were memorising, train accuracy would keep rising while the holdout stays flat. If both are flat, the objective has stopped yielding signal.</p>
{miss_txt}{tv_html}<p>{tv_read}</p>"""

    # ---- 3. where it stalls ---------------------------------------------------------------------------------------
    pools = pools_in(P)
    if have and pools:
        mfigs = []
        for pool in pools:
            sers = []
            for kind, klabel, color, s in KINDS:
                pts = [(step_of[ck], grp(P, ck, s, pool, kind)["acc"]) for ck in have if grp(P, ck, s, pool, kind)]
                if pts:
                    sers.append({"name": klabel, "pts": pts, "cls": "eval", "color": color})
            if not sers:
                continue
            svg, sp = H.line_chart(f"pl-k-{pool.replace(':', '_')}", f"{pool_label(pool)} accuracy by slot kind", sers, scale=100.0, dec=1, suffix="%", cap100=True)
            specs.append(sp)
            mfigs.append(H.fig(pool_label(pool), "", svg, ""))
        kleg = legend([(c, f"{l}{' (mixed states)' if s == 'holdout_mixed' else ' (pass 1)'}") for _, l, c, s in KINDS])
        hdr = "".join(f'<th scope="col" class="num"><span class="mono">{"stock" if ck == "stock" else f"{step_of[ck]:,}"}</span></th>' for ck in have)
        acc_rows, keep_rows, prec_rows = [], [], []
        for pool in pools:
            for kind, klabel, _, s in KINDS:
                gs = [grp(P, ck, s, pool, kind) for ck in have]
                if not any(gs):
                    continue
                acc_rows.append(f'<tr><th scope="row">{esc(pool_label(pool))} · {esc(klabel)}<span class="nn">n slots ≈ {max((g["n"] for g in gs if g), default=0):,}</span></th>'
                                + "".join(f'<td class="num">{pct(g)}</td>' for g in gs) + "</tr>")
                if kind == "draft_bad":
                    keep_rows.append(f'<tr><th scope="row">{esc(pool_label(pool))}</th>' + "".join(f'<td class="num">{pct(g, "keep_draft")}</td>' for g in gs) + "</tr>")
                if pool == "all":
                    prec_rows.append(f'<tr><th scope="row">{esc(klabel)}</th>' + "".join(
                        f'<td class="num">{pct(g, "precision")}<span class="nn">commits {pct(g, "commit_share")}%</span></td>' for g in gs) + "</tr>")
        tbl = lambda rows, first: (f'<div class="tscroll"><table class="data"><thead><tr><th scope="col">{first}</th>{hdr}</tr></thead>'
                                   f'<tbody>{"".join(rows)}</tbody></table></div>') if rows else '<div class="placeholder">pending</div>'
        b_all = [(ck, grp(P, ck, "holdout_pass1", "all", "draft_bad")) for ck in have]
        o_all = [(ck, grp(P, ck, "holdout_pass1", "all", "draft_ok")) for ck in have]
        share_bad = None
        for (ck, gb), (_, go) in zip(b_all, o_all):
            if gb and go:
                share_bad = gb["n"] / (gb["n"] + go["n"])
        part3 = f"""
<h3>3 · Where it stalls: pool × slot kind</h3>
<p>Each open slot is classed by what the judge should do. <strong>Confirm</strong>: the draft token equals the target. <strong>Override</strong>: the draft is wrong and the judge must
replace it. <strong>Masked</strong>: no draft token, only in the mixed training-regime states. Stock appears at step 0.
{f"On holdout pass-1 states, {100 * share_bad:.0f}% of drafted slots are overrides." if share_bad is not None else ""}
Stock scores low on drafted slots because at a filled slot it echoes the next token instead of judging (see “Why a judge”). Its draft-kind numbers show that echo, not a weak judge.</p>
{miss_txt}{kleg}<div class="grid2">{"".join(mfigs)}</div>
<h3 class="kick">Accuracy % by pool, kind and checkpoint (holdout)</h3>{tbl(acc_rows, "pool · kind")}
<h3 class="kick">Sycophancy: on override slots, how often the argmax keeps the wrong draft token (%)</h3>{tbl(keep_rows, "pool")}
<h3 class="kick">Precision at τ 0.95 by kind (all pools), with the share of slots committed</h3>{tbl(prec_rows, "kind")}"""
    else:
        part3 = f"""
<h3>3 · Where it stalls: pool × slot kind</h3>
<p>Once the probe lands, this part splits accuracy into <strong>confirm</strong> slots (draft right), <strong>override</strong> slots (draft wrong) and <strong>masked</strong> slots, per pool and checkpoint,
with the keep-the-wrong-draft rate and precision at τ 0.95. Stock sits at step 0.</p>{miss_txt}<div class="placeholder">pending: no probe JSONs yet.</div>"""

    # ---- 4. mechanisms --------------------------------------------------------------------------------------------
    late = next((ck for ck in ("step001500", "step001400") if P[ck]), None)
    b7, bl = grp(P, "step000700", "holdout_pass1", "all", "draft_bad"), grp(P, late, "holdout_pass1", "all", "draft_bad") if late else None
    o7, ol = grp(P, "step000700", "holdout_pass1", "all", "draft_ok"), grp(P, late, "holdout_pass1", "all", "draft_ok") if late else None
    bs, os_ = grp(P, "stock", "holdout_pass1", "all", "draft_bad"), grp(P, "stock", "holdout_pass1", "all", "draft_ok")
    ms, m7, ml = (grp(P, "stock", "holdout_mixed", "all", "mask"), grp(P, "step000700", "holdout_mixed", "all", "mask"),
                  grp(P, late, "holdout_mixed", "all", "mask") if late else None)

    ifk = next((p_ for p_ in pools_in(P) if p_.startswith("if")), None); codek = next((p_ for p_ in pools_in(P) if "code" in p_), None)
    if_bad = grp(P, late, "holdout_pass1", ifk, "draft_bad") if late and ifk else None
    if_mask = grp(P, late, "holdout_mixed", ifk, "mask") if late and ifk else None
    code_bad = grp(P, late, "holdout_pass1", codek, "draft_bad") if late and codek else None
    code_bad7 = grp(P, "step000700", "holdout_pass1", codek, "draft_bad") if codek else None

    def verdict(text, cls):
        return f'<span class="chip {cls}">{esc(text)}</span>'

    if b7 and bl:
        dbad = 100 * (bl["acc"] - b7["acc"]); dok = 100 * (ol["acc"] - o7["acc"]) if o7 and ol else None
        mv = pool_moves("holdout_pass1", "draft_bad"); rising = [m for m in mv if m[3] >= 3]
        if abs(dbad) < 3 and rising:
            va = verdict("partly: flat overall, but " + " and ".join(pool_label(m[0]).split(" (")[0] for m in rising) + " overrides still improve", "c-warn")
        elif abs(dbad) < 3:
            va = verdict("consistent with the data", "c-warn")
        else:
            va = verdict("against: overrides still improving", "c-good")
        ea = (f"Override accuracy is {pct(bs)}% for stock, {pct(b7)}% at step 700 and {pct(bl)}% at {late} ({num(dbad, 1, True)} pts after epoch 1). "
              f"Confirm accuracy is {pct(os_)}% / {pct(o7)}% / {pct(ol)}%{f' ({num(dok, 1, True)})' if dok is not None else ''}. "
              f"On override slots the argmax keeps the wrong draft {pct(b7, 'keep_draft')}% of the time at step 700 and {pct(bl, 'keep_draft')}% at {late}.")
        if mv:
            ea += (" The aggregate hides a split. Per pool, override accuracy from step 700 to " + late + ": "
                   + "; ".join(f"{pool_label(p_)} {pct(ga)}% → {pct(gb)}% ({num(d_, 1, True)} pts, n={gb['n']:,})" for p_, ga, gb, d_ in mv) + ". "
                   + ("So the skill has not saturated everywhere: code overrides keep improving after epoch 1, while the largest group, IF overrides, sits near its floor. "
                      "That reads more like target ambiguity in free-form prose (c) than a capacity cap on the adapter." if rising else ""))
    else:
        va, ea = verdict("pending probe", ""), "Needs override (draft_bad) and confirm (draft_ok) accuracy per checkpoint from the plateau probe."
    vb = (verdict("consistent with the data", "c-warn") if fc and fg and fce and fc["slope"] + fc["ci"] < 0 and fg["slope"] - fg["ci"] > 0
          and fce["slope"] - fce["ci"] <= 0 <= fce["slope"] + fce["ci"] else verdict("mixed", "") if fc else verdict("pending", ""))
    tvg = None
    if late:
        t_, h_ = grp(P, late, "train_pass1", "all", "all"), grp(P, late, "holdout_pass1", "all", "all")
        tvg = 100 * (t_["acc"] - h_["acc"]) if t_ and h_ else None
    vd = (verdict("against memorisation; data variety still open", "c-good") if tvg is not None and abs(tvg) < 3
          else verdict("partly supported", "c-warn") if tvg is not None else verdict("partly testable, probe pending", ""))

    dl = []
    if ev:
        for dsk in ("mbpp", "humaneval"):
            for nm in ("judge", "judge_des48"):
                e = ev["direction"][dsk][nm]["e2_vs_e1"]
                if e and e.get("d_acc") is not None:
                    lab = next(d["label"] for d in ev["datasets"] if d["key"] == dsk)
                    dl.append(f"{lab} {ev['arm_labels'][nm]} {num(e['d_acc'], 1, True)} pts [{num(e['ci'][0], 1, True)}, {num(e['ci'][1], 1, True)}], "
                              f"exp/token {num(e['ref_exp_tok'], 0)} → {num(e['exp_tok'], 0)}")
    down = ("; ".join(dl) + ".") if dl else "Downstream epoch-1 vs epoch-2 pairs are not scored yet."

    hyp = f"""
<h3>4 · Mechanisms: hypotheses and the evidence for each</h3>
<p>None of these is proven by a controlled ablation. Each is tied to what the current data shows, and its verdict updates as the probe and the resumed run land.</p>
<div class="hyp"><h4>(a) The override skill saturates, or needs signal the adapter cannot reach {va}</h4>
<p>The trainable part is a LoRA on attention only (q, k, v, o; r 32; 26.7 M parameters) plus the draft flag. Experts, router gates, embeddings and the head are frozen. Confirming a right draft
mostly needs attention to trust the slot. Overriding a wrong one needs the frozen expert path to produce a better token from the changed context, which is a harder function to move with attention alone.</p>
<p><strong>Evidence:</strong> {ea}</p>
<p><strong>Would falsify it:</strong> override accuracy still climbing after epoch 1 while overall accuracy is flat, which would mean confirms had regressed instead. Only a LoRA-on-MLP or higher-rank arm tests the capacity half directly.</p></div>
<div class="hyp"><h4>(b) The certainty term keeps sharpening after CE stops improving {vb}</h4>
<p>The loss adds β = 2 × the entropy (at τ 0.5) of slots whose argmax is already right. Once CE is flat, the remaining gradient comes largely from this term, which makes right slots more peaked without making more slots right.
Because the argmax is right on only about 60% of training slots, the confidence it teaches can spill onto wrong slots.</p>
<p><strong>Evidence:</strong> {f"after step 700 CE is flat ({num(fce['slope'], 3, True)} ± {fce['ci']:.3f} per 100 steps), the certainty term still falls ({num(fc['slope'], 4, True)} ± {fc['ci']:.4f}), and the overconfidence gap widens ({num(fg['slope'], 2, True)} ± {fg['ci']:.2f} pts; {esc(gap_txt)})." if fc and fg and fce else "pending train rows."}
The training-time holdout shows the same pattern: the confident fraction keeps inching up while argmax accuracy is flat (section above). Precision at τ 0.95 is what turns this into downstream errors. {f"The probe shows it directly: from step 700 to {late}, the share of confirm slots committed at τ 0.95 grew from {pct(o7, 'commit_share')}% to {pct(ol, 'commit_share')}%, while the precision of those commits fell from {pct(o7, 'precision')}% to {pct(ol, 'precision')}%. Override-slot commits are {pct(bl, 'commit_share')}% of override slots at {pct(bl, 'precision')}% precision. More confidence without more accuracy is exactly what this mechanism predicts." if o7 and ol and bl else "Per-kind precision is pending the probe."}</p>
<p><strong>Would falsify it:</strong> a β = 0 continuation from step 700 whose accuracy also stays flat. That run would show the plateau is not caused by the term, even if the term inflates confidence.</p></div>
<div class="hyp"><h4>(c) An irreducible floor: stock-decode targets and masked-future ambiguity {verdict("plausible for IF prose, not measured", "")}</h4>
<p>Targets are the final tokens of the stock pipeline’s own correct decodes. At pass 1 many slots are genuinely under-determined: several continuations are fine, and the stock trajectory picked one only after later passes had committed its neighbours.
For reference, on 100 GSM8K train prompts the stock model’s own pass-1 argmax already equals the final token at 58.4% of slots (<code>docs/EXPLAINER_DRAFT_ALIGNED_TRAINING.md</code> §1). {f"Pooled train accuracy flattens near {fa['mean']:.0f}%." if fa else ""}
That proximity is suggestive, not proof: the judge also sees the draft, so its ceiling should be higher than stock’s mask-only ceiling.</p>
<p><strong>Evidence:</strong> {f"masked-slot accuracy is {pct(ms)}% for stock, {pct(m7)}% at step 700 and {pct(ml)}% at {late}. At {late}, confirm slots are {pct(ol)}% right and override slots {pct(bl)}%, so most of the remaining error sits in override slots." if ms and ml and ol and bl else "the per-kind split (pending probe) shows where the residual error sits; a floor argument needs the override slots to be the ambiguous ones."}
{f"That error concentrates in IF. IF supplies {if_bad['n']:,} of {bl['n']:,} holdout override slots, and they stay at {pct(if_bad)}% right (masked IF slots {pct(if_mask)}%), against {pct(code_bad)}% for code. Free-form IF prose admits many valid continuations, so a target-ambiguity floor is most plausible there and least plausible for code, whose override accuracy is still rising." if if_bad and bl and code_bad else ""}</p>
<p><strong>Would test it:</strong> re-decoding the holdout prompts with the stock pipeline under a different seed or threshold and measuring how often the final tokens themselves disagree.</p></div>
<div class="hyp"><h4>(d) Data: 22.7k examples reused for 2+ epochs; code coverage {vd}</h4>
<p>The index has 22,734 training examples from 4,294 train items (4,444 minus the 150 held out). By step 1,560 the model has seen each about 2.2 times. The state is re-drawn on every visit (pass choice, mask ratio), so exact repeats are rare, but the targets and prefixes repeat.
By items the code pool is the <em>largest</em> (2,344 items, 8,297 blocks), and IF the smallest (690 items, 6,861 blocks).</p>
<p><strong>Evidence:</strong> {f"train-split minus holdout accuracy at {late} is {num(tvg, 1, True)} pts (probe section 2)." if tvg is not None else "train vs holdout from the probe is pending."} A small gap argues against memorisation.
It does not rule out that the <em>variety</em> of override situations is what runs out. That would show up per pool, which is why code, still improving, and IF, flat, are worth separating.</p></div>
<div class="hyp"><h4>(e) The learning rate decayed too far {verdict("against", "c-good")}</h4>
<p>The schedule is cosine from 2.0e-5 to a floor of 0.5× (1.0e-5) at step 2,175. The plateau starts around step 700, when the LR was still about 1.8e-5 (90% of peak), and at the stop it is still {f"{lr_last:.2e}" if lr_last else "—"}.
Decay cannot explain a stall that began near peak LR. The opposite is open: an LR too <em>high</em> for the late phase would also produce a flat, noisy loss with a rising gradient norm. A lower-LR tail is cheap to test.</p></div>
<h3>Why code accuracy can improve while the holdout is flat</h3>
<p>Downstream, epoch 2 vs epoch 1 on code (paired, same items): {esc(down)} Most of these intervals still include zero, but the direction on MBPP is positive, and expert work fell for every arm listed.
{f"The probe points the same way: code override accuracy rose {num(100 * (code_bad['acc'] - code_bad7['acc']), 1, True)} pts from step 700 to {late} ({pct(code_bad7)}% → {pct(code_bad)}%, n={code_bad['n']:,} slots). The all-pool holdout barely moved, because IF override slots, the largest group, stayed flat." if code_bad and code_bad7 else ""}
The two measurements answer different questions.</p>
<ul><li><strong>The holdout counts argmax accuracy on 64 pass-1 states</strong>, every slot weighted equally. Downstream decoding only commits slots above τ 0.95, and it repeats passes until the block is done. What matters there is <em>precision of the committed set</em> and how errors compound along the trajectory.</li>
<li>Sharper confidence on already-right slots (mechanism b) leaves argmax accuracy unchanged, but it changes <em>which</em> slots cross 0.95 and when. That can reduce passes (exp/token fell epoch 1 → 2) and shift errors between committed and deferred slots. For code, one wrong committed token can fail a test, so small precision changes move pass@1 much more than they move slot accuracy.</li>
<li>The holdout mixes code, GSM8K and IF slots, 64 states, noise ±~1 pt. A per-pool gain on code can hide inside that noise; part 3 splits it out.</li></ul>"""

    # ---- 5. what to try next --------------------------------------------------------------------------------------
    nxt = f"""
<h3>5 · What to try next, ranked</h3>
<ol>
<li><strong>Stricter commit τ, or a low-agreement fallback.</strong> This needs no retraining and targets the precision of the committed set, which is what downstream code accuracy depends on.
Sweep τ ∈ {{0.95, 0.97, 0.99}} on MBPP/HumanEval at <code>step001400</code>, and fall back to stock denoising in blocks where the judge overrides many draft tokens.
{f"Evidence: override-slot precision at τ 0.95 is {pct(bl, 'precision')}% against {pct(ol, 'precision')}% on confirm slots." if bl and ol else "Evidence: per-kind precision (probe) will show how much stricter τ buys."}</li>
<li><strong>Lower β, or drop the certainty term after epoch 1.</strong> After step 700 the term is the only loss component still moving, and the overconfidence gap grows with it (mechanism b). A cheap continuation from <code>step000700</code> with β = 0 or 0.5 for 300–500 steps tests it.</li>
<li><strong>More adapter capacity: LoRA on the MLP/expert projections, or a higher rank.</strong> Attention-only LoRA may cap the override skill (mechanism a). This costs memory and a fresh run, so try it after 1–2.</li>
<li><strong>More, and more varied, code data.</strong> Code is where downstream damage lives{f", and it is the pool whose override accuracy is still moving ({pct(code_bad7)}% → {pct(code_bad)}% from step 700)" if code_bad and code_bad7 else ""}. More distinct override situations (new prompts, not more epochs over the same 2,344 items) addresses the data half of (a)/(d).</li>
<li><strong>Per-kind loss weighting that up-weights override slots.</strong> {f"Overrides are {100 * share_bad:.0f}% of drafted holdout slots but only {pct(bl)}% right, against {pct(ol)}% on confirms." if have and pools and share_bad is not None and bl and ol else "Override slots are far less accurate than confirms."} Weighting override slots 2–4× pushes the gradient toward the skill that is stuck. Weight by pool as well: code overrides are still learnable, while IF overrides look ambiguous (c), and up-weighting those may teach noise. Watch confirm precision and the keep-draft rate.</li>
</ol>"""

    lede = (f'<p class="lede">Accuracy climbed through epoch 1 and then stopped: {num(fa["slope"], 2, True)} ± {fa["ci"]:.2f} pts per 100 steps from step 700 to the stop. '
            f'The loss did not get worse and the model did not memorise; it kept becoming more confident without becoming more correct. '
            'Below: the trend statistics, a per-slot-kind probe of seven checkpoints, five candidate mechanisms with their evidence, and a ranked list of what to try.</p>') if fa else ""
    html_ = lede + part1 + part2 + part3 + hyp + nxt
    li = (f'<li>Why it flattened: from step 700 to the stop, train accuracy moved {num(fa["slope"], 2, True)} ± {fa["ci"]:.2f} pts per 100 steps while the overconfidence gap '
          f'widened {num(fg["slope"], 2, True)} ± {fg["ci"]:.2f}. See the <a href="#plateau">diagnosis</a> (probe panels: {len(have)}/7 checkpoints in).</li>') if fa and fg else ""
    return html_, li


if __name__ == "__main__":
    T = load_train()
    print("segments", [len(s) for s in T["segments"]], "merged", len(T["merged"]))
    for key, label, unit, f, wtd, dec in METRICS:
        print(label, [(pl, None if not (ft := fit(T["original"], lo, hi, f, wtd)) else (round(ft["slope"], dec), round(ft["ci"], dec))) for pl, lo, hi in PHASES])
    print("probes:", {k: bool(v) for k, v in load_probes().items()})
