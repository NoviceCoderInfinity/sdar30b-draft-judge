"""Build the Phase 3 results page (Table 1 in the Phase 2 artifact's layout) from extrap + status outputs.
Usage (repo root): /venv/p2/bin/python -I build_artifact.py latest.json status.md out.html"""
import glob, json, os, sys, html, datetime as dt
EX, STATUS, OUT = sys.argv[1:4]
ex = json.load(open(EX)); p2 = json.load(open("logs/phase3/extrap/p2_metrics.json"))
SETS = [("humaneval", "HumanEval", 164), ("mbpp", "MBPP", 257), ("gsm8k500", "GSM8K-500", 500), ("ifeval541", "IFEval-541", 541),
        ("math500", "MATH-500", 500), ("gpqa", "GPQA-Diamond", 198), ("aime", "AIME", 90)]
# Phase 2 artifact Table 1 (old 8x box): acc, tok/s for stock, SDAR-1.7B, CoDA-1.7B, Qwen3-0.6B
T1 = {"humaneval": [(78.0, 11.6), (74.4, 14.6), (76.2, 15.1), (78.0, 14.8)], "mbpp": [(65.4, 6.6), (66.1, 8.5), (67.3, 8.2), (68.5, 7.9)],
      "gsm8k500": [(90.2, 10.0), (91.6, 12.6), (91.0, 11.5), (91.6, 11.8)], "ifeval541": [(52.7, 7.1), (51.0, 8.9), (49.9, 8.9), (51.4, 8.3)],
      "math500": [(72.4, 13.3), (71.6, 15.7), (69.8, 14.2), (71.2, 14.7)], "gpqa": [(37.4, 8.1), (36.9, 10.0), (38.9, 9.5), (34.8, 9.6)],
      "aime": [(6.7, 11.9), (5.6, 14.4), (7.8, 12.8), (6.7, 13.7)], "all": [(65.5, 9.7), (65.0, 11.8), (64.7, 11.2), (65.4, 11.2)]}
ARMS = ["rd1", "rd2"]
# the Phase 2 page's printed speedups for SDAR-1.7B, CoDA-1.7B, Qwen3-0.6B (copied, so the reference columns match exactly)
T1X = {"humaneval": [1.26, 1.30, 1.28], "mbpp": [1.29, 1.24, 1.19], "gsm8k500": [1.26, 1.15, 1.18], "ifeval541": [1.25, 1.25, 1.16],
       "math500": [1.19, 1.07, 1.11], "gpqa": [1.23, 1.17, 1.19], "aime": [1.22, 1.08, 1.16], "all": [1.22, 1.15, 1.15]}
cells = ex["cells"]
st_lines = [l for l in open(STATUS) if l.startswith("| rd") or l.startswith("| base")]
prog = {}
for l in st_lines:
    p = [x.strip() for x in l.strip().strip("|").split("|")]
    prog[(p[0], p[1])] = p
upd = [l for l in open(STATUS) if l.startswith("Progress at")][0].replace("Progress at", "").replace("IST", "").strip()
fin = [l for l in open(STATUS) if l.startswith("Run finished")][0].split("~")[-1].replace("IST", "").strip()
ndone = sum(1 for p in prog.values() if p[-1] == "done")
e = html.escape
def cellv(a, s):
    c = cells.get(f"{a}/{s}")
    return c if c and c.get("complete") else None

def fmt_acc(v, ref): d = v - ref; return f"{v:.1f} <span class=ci>({'+' if d > 0 else '&minus;' if d < 0 else ''}{abs(d):.1f})</span>"
def recs(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if l.strip(): r = json.loads(l); out[int(r["idx"])] = r
    return out
def agg(Rs):
    Rs = list(Rs)
    if not Rs: return None
    g = sum(r["n_gen_to_eos"] for r in Rs); sec = sum(r["seconds"] for r in Rs)
    return {"n": len(Rs), "tok": g, "sec": sec, "tps": g / sec, "acc": 100 * sum(bool(r.get("correct")) for r in Rs) / len(Rs),
            "passes": sum(r["n_forward"] - r["n_prefill"] - r["n_flush"] for r in Rs) / sum(r["n_blocks"] for r in Rs),
            "exp": sum(r["unique_expert_layer_loads"] for r in Rs) / g, "dfwd": sum(sum(r["drafter_n_by_kind"].values()) for r in Rs) / g}
M = {}   # measured on this box: per set -> judge / rd1 / rd2 aggregates and same-items speedups
for s, name, n in SETS:
    Jf = recs(f"logs/phase3/decode/judge/{s}/*.jsonl"); Js = recs(f"logs/phase3/decode/judge_sample/{s}/*.jsonl")
    full = len(Jf) >= n and os.path.exists(f"logs/phase3/decode/judge/{s}/.complete")     # a base run still in progress is not a full baseline
    J = Jf if full else (Js if Js else Jf)
    m = {"full": full, "judge": agg(J.values()), "judge_label": "full set" if full else f"{len(J)}-item sample"}
    for a in ARMS:
        R = recs(f"logs/phase3/decode/{a}/{s}/*.jsonl"); c = cells.get(f"{a}/{s}")
        m[a] = agg(R.values()) if R else None
        m[a + "_done"] = bool(c and c.get("complete"))
        com = [i for i in R if i in J]
        if com and m["judge"]:
            tr = agg(R[i] for i in com); tj = agg(J[i] for i in com)
            m[a + "_sp"] = tr["tps"] / tj["tps"]; m[a + "_sp_n"] = len(com)
    M[s] = m
def sp_txt(m, a):
    if m.get(a + "_sp") is None: return ""
    if m["full"]: return f"{m[a+'_sp']:.2f}&times; <span class=ci>vs base</span>"
    return f"<span class=flag title='speedup measured on only {m[a+'_sp_n']} items'>{m[a+'_sp']:.2f}&times;&dagger;</span>"
rows = []
for s, name, n in SETS:
    base = list(T1[s])
    for k, key in enumerate(["stock", "S0_nopool", "F2_nopool", "S1_nopool"]):
        if p2.get(s, {}).get(key, {}).get("acc") is not None: base[k] = (p2[s][key]["acc"], base[k][1])
    ref_acc = base[0][0]; m = M[s]
    acc = [b[0] for b in base[1:]]; pend = {}
    for a in ARMS:
        if m[a + "_done"] and m[a]: acc.append(m[a]["acc"])
        else:
            p = prog.get((a, s)); acc.append(None); pend[a] = f"{p[-1]} {p[2]}" if p else "queued"
    best = max(v for v in acc if v is not None)
    tds = "".join(f"<td class=pending>{e(pend[ARMS[k-3]])}</td>" if v is None else f"<td{' class=win' if round(v,1)==round(best,1) else ''}>{fmt_acc(v, ref_acc)}</td>" for k, v in enumerate(acc))
    ja = f"{m['judge']['acc']:.1f}" if m["full"] else "&mdash;"
    bench = f"<td class='l bench' rowspan=2>{name}<span class=n>n = {n}</span></td>"
    parts = tds.split("</td>")[:-1]
    p2tds = "</td>".join(parts[:3]) + "</td>"; rdtds = "</td>".join(parts[3:]) + "</td>"
    rows.append(f"<tr class='first sep'>{bench}<td class='l m'>Accuracy, % (&Delta; pts)</td><td class=ref>{ref_acc:.1f}</td>{p2tds}<td class=ref>{ja}</td>{rdtds}</tr>")
    # speed row, measured on this box
    jt = m["judge"]
    jcell = f"{jt['tps']:.2f} <span class=ci>({'all ' if m['full'] else ''}{jt['n']} items)</span>"
    tt = {a: (m[a]["tps"] if m[a + "_done"] and m[a] else None) for a in ARMS}
    bt = max([jt["tps"]] + [v for v in tt.values() if v])
    def tcell(a):
        if tt[a] is None: return f"<td class=pending>{e(pend[a])}</td>"
        return f"<td{' class=win' if tt[a] == bt else ''}>{tt[a]:.2f} <span class=ci>({sp_txt(m, a)})</span></td>"
    rows.append(f"<tr class=last><td class='l m'>Tokens / s on this box (&times; base)</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=na>&mdash;</td>"
                f"<td{' class=win' if jt['tps'] == bt else ''}>{jcell}</td>{tcell('rd1')}{tcell('rd2')}</tr>")
# pooled row: benchmarks with a full judge baseline and both arms complete
ds = [s for s, _, _ in SETS if M[s]["full"] and all(M[s][a + "_done"] and M[s][a] for a in ARMS)]
allrow = ""
if ds:
    P = {k: [M[s][k] for s in ds] for k in ("judge", "rd1", "rd2")}
    pt = {k: sum(x["tok"] for x in v) / sum(x["sec"] for x in v) for k, v in P.items()}
    pa = {k: sum(x["acc"] * x["n"] for x in v) / sum(x["n"] for x in v) for k, v in P.items()}
    stock_acc = sum(T1[s][0][0] * n for s, _, n in SETS if s in ds) / sum(n for s, _, n in SETS if s in ds)
    nit = sum(n for s, _, n in SETS if s in ds)
    lab = ", ".join(nm for s, nm, _ in SETS if s in ds)
    allrow = (f"<tbody class=all><tr class=first><td class='l bench' rowspan=2>Fully measured ({len(ds)} of 7)<span class=n>{nit:,} items</span></td><td class='l m'>Accuracy, % (&Delta; pts)</td>"
              f"<td class=ref>{stock_acc:.1f}</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=ref>{pa['judge']:.1f}</td>"
              f"<td>{fmt_acc(pa['rd1'], stock_acc)}</td><td>{fmt_acc(pa['rd2'], stock_acc)}</td></tr>"
              f"<tr class=last><td class='l m'>Tokens / s on this box (&times; base)</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=na>&mdash;</td><td class=na>&mdash;</td>"
              f"<td>{pt['judge']:.2f}</td><td>{pt['rd1']:.2f} <span class=ci>({pt['rd1']/pt['judge']:.2f}&times;)</span></td><td>{pt['rd2']:.2f} <span class=ci>({pt['rd2']/pt['judge']:.2f}&times;)</span></td></tr></tbody>")
    ALLNOTE = lab
else: ALLNOTE = ""
# ---- EXTRAPOLATED section: the Phase 1 box (8x). Every extrapolated cell carries a bold tag; the base row is the MEASURED Phase 1 value ----
proj = ex.get("projection_min", {})
XT = "<b class=xt>extrapolated</b>"; XTM = "<b class=xt>extrapolated (modelled)</b>"; MT = "<b class=mt>measured on the Phase 1 box</b>"
def sg(v, d=1): return ("+" if v > 0 else "&minus;" if v < 0 else "") + f"{abs(v):.{d}f}"
xrows = []
for s, name, n in SETS:
    ref = p2[s]["S0_nopool"]; oldt = 1000 / ref["ms_tok"]; grp = []
    for a in ARMS:
        c = cells.get(f"{a}/{s}")
        if not c:
            p = prog.get((a, s)); st_txt = f"{p[5]} {p[2]}" if p else "queued"
            grp.append(f"<td class=l>{a}</td><td colspan=5 class=pending>{e(st_txt)}: nothing to extrapolate yet</td>"); continue
        x = c["tps_headline"]; oj = c["tps_S0_old"]; dag = "" if c.get("samebox_method") == "full baseline" else "&dagger;"
        part = "" if c["complete"] else f" <span class=pt>partial, {c['n']} items</span>"
        w8 = proj.get(f"{a}/{s}", {}).get("old8")
        wall = f"{w8:.0f} min{XTM}" if (w8 and c["complete"]) else "&middot;"
        grp.append(f"<td class=l>{a}{part}</td><td class=calc>{oj:.1f} &times; {x/oj:.3f}{dag}</td><td class=xc>{x:.1f}{dag}{XT}</td>"
                   f"<td class=xc>{sg(x-oj)} ({sg(100*(x/oj-1),0)}%){dag}{XT}</td><td class=xc>{x/c['tps_stock_table1']:.2f}&times;{dag}{XT}</td><td class=xc>{wall}</td>")
    grp.append(f"<td class=l>base pipeline</td><td class=calc>Phase 2 record</td><td class='xc basecell'>{oldt:.1f}{MT}</td><td class=xc>reference</td>"
               f"<td class='xc basecell'>{T1X[s][0]:.2f}&times;{MT}</td><td class=xc>&middot;</td>")
    for i, g in enumerate(grp):
        lead = f"<td class='l bench' rowspan=3>{name}</td>" if i == 0 else ""
        xrows.append(f"<tr class='{'first sep' if i == 0 else ''}{' last' if i == 2 else ''}'>{lead}{g}</tr>")
XTABLE = "".join(xrows)
refrows = []
for s, name, n in SETS:
    ref = T1[s]; tx = T1X[s]
    refrows.append(f"<tr><td class=l>{name}</td><td>{ref[0][1]:.1f}</td><td>{ref[1][1]:.1f} <span class=ci>({tx[0]:.2f}&times;)</span></td><td>{ref[2][1]:.1f} <span class=ci>({tx[1]:.2f}&times;)</span></td>"
                   f"<td>{ref[3][1]:.1f} <span class=ci>({tx[2]:.2f}&times;)</span></td></tr>")
REF2 = "".join(refrows)
# progress table
prow_html = "".join(f"<tr><td class=l>{e(p[0]) if p[0] != 'base' else 'base pipeline'}</td><td class=l>{e(dict((s,nm) for s,nm,_ in SETS)[p[1]])}</td><td>{e(p[2])}</td><td>{e(p[3])}</td><td class=l><span class='chip {p[5]}'><i></i>{e(p[5])}</span></td></tr>" for p in prog.values())

dg = []
for s, name, _ in SETS:
    st = p2[s]["stock"]; m = M[s]
    for key, lab in (("judge", "base pipeline"), ("rd1", "rd1"), ("rd2", "rd2")):
        x = m[key]
        if not x: continue
        part = "" if key == "judge" or m[key + "_done"] else f" <span class=pt>partial</span>"
        if key == "judge": nlab = f"{x['n']}" + ("" if m["full"] else " <span class=pt>sample</span>")
        else: nlab = f"{x['n']}"
        spc = "&mdash;" if key == "judge" else (sp_txt(m, key) or "&mdash;")
        dg.append(f"<tr><td class=l>{name}</td><td class=l>{lab}{part}</td><td>{nlab}</td><td>{x['passes']:.2f} <span class=ci>({st['passes_blk']:.2f})</span></td>"
                  f"<td>{x['exp']:.0f} <span class=ci>({st['exp_tok']:.0f})</span></td><td>{x['dfwd']:.2f}</td><td><b>{x['tps']:.2f}</b></td><td>{spc}</td></tr>")
# figure data
FIG = []
REFK = ["stock", "S0_nopool", "F2_nopool", "S1_nopool"]
for s_, name, n in SETS:
    m = M[s_]; pts = []
    for k, (lab, tiny, shape, col) in enumerate([("Stock 30B", "Stock", "diamond", "--s-stock"), ("SDAR-1.7B + judge (Phase 2 records)", "SDAR", "circle", "--s-sdar"),
                                                  ("CoDA-1.7B + judge (Phase 2 records)", "CoDA", "square", "--s-coda"), ("Qwen3-0.6B + judge (Phase 2 records)", "Qwen3", "triangle", "--s-qwen")]):
        accv = p2.get(s_, {}).get(REFK[k], {}).get("acc", T1[s_][k][0])
        pts.append({"label": lab, "tiny": tiny, "shape": shape, "color": col, "acc": accv, "err": None, "note": "Phase 2 records (not measured on this box)"})
    pts.insert(1, {"label": "Base pipeline (this box)", "tiny": "Base", "shape": "circle", "color": "--s-sdar", "sp": 1.0, "acc": p2[s_]["S0_nopool"]["acc"], "err": None,
                   "tps": m["judge"]["tps"], "note": f"measured here, {m['judge']['n']} items ({m['judge_label']}); = reference 1x"})
    del pts[2]
    for a, shape, col in (("rd1", "pent", "--s-rd1"), ("rd2", "star", "--s-rd2")):
        x = m[a]; c = cellv(a, s_)
        if c and x:
            d = c.get("delta_vs_S0"); b0 = p2[s_]["S0_nopool"]["acc"]
            pts.append({"label": a + (" (re-draft every pass)" if a == "rd1" else " (re-draft every 2 passes)"), "tiny": a, "shape": shape, "color": col,
                        "sp": m.get(a + "_sp"), "flag": (not m["full"]), "acc": c["acc"], "err": [b0 + d[1], b0 + d[2]] if d else None, "tps": x["tps"],
                        "note": (f"measured here on {m[a+'_sp_n']} identical items vs the base pipeline" if m["full"] else f"speedup vs the 12-item base baseline, only {m.get(a+'_sp_n')} items: not fully measured") + (f"; acc Δ vs SDAR {d[0]:+.1f} pts [{d[1]:+.1f}, {d[2]:+.1f}]" if d else "")})
        else:
            pts.append({"label": a, "tiny": a, "shape": shape, "color": col, "sp": None, "acc": None, "err": None, "tps": None, "note": "pending"})
    FIG.append({"name": name, "n": n, "pts": pts, "live": any(pp.get("sp") is not None for pp in pts if pp["tiny"].startswith("rd"))})

# ---- unified table (rows from unified.py, same data; column groups are read from its header) ----
import subprocess, re as _re
_u = subprocess.run([sys.executable, "-I", "logs/phase3/extrap/unified.py", EX, STATUS], capture_output=True, text=True, check=True).stdout.splitlines()
_t = [[x.strip() for x in l.strip().strip("|").split("|")] for l in _u if l.startswith("| ")]
_hdr, _rows = _t[0], _t[1:]
def _grp(h): return "m" if h.startswith("MEASURED") else "x" if h.startswith("EXTRAPOLATED") else ""
_G = [_grp(h) for h in _hdr]
def _label(h):
    g = _grp(h); t = h.split(": ", 1)[1] if g and ": " in h else h; tl = t.lower()
    if g == "":
        if tl.startswith("accuracy"): return "Accuracy %<span class=sub>(&Delta; vs base pipeline)</span>"
        if tl.startswith("passes"): return "Passes / block"
        if tl.startswith("expert"): return "30B expert loads / token"
        return e(t)
    if "your pipeline tok/s" in tl: return "Base tokens / s<span class=sub>(same items)</span>"
    if tl.startswith("tok/s (all"): return "rd tokens / s<span class=sub>(all items)</span>"
    if tl.startswith("tok/s"): return "rd tokens / s"
    if tl.startswith("&delta;") or t.startswith("Δ"): return "&Delta; tokens / s<span class=sub>vs base</span>"
    if tl.startswith("speedup"): return "Speedup vs base<span class=sub>(items)</span>"
    if tl.startswith("wall"): return "Wall<span class=sub>" + e(t[t.index("("):]) + "</span>"
    if "stock" in tl: return "&times; stock"
    return e(t)
def _cell(v):
    v = e(v).replace("+ (", "&dagger; (")
    v = _re.sub(r"(\d)x(\b|&)", r"\1&times;\2", v)
    v = _re.sub(r"(?<![\w;])-(?=\d)", "&minus;", v)
    if v.endswith("+"): v = v[:-1] + "&dagger;"
    if " (" in v:
        h, r_ = v.split(" (", 1); v = f"{h} <span class=ci>({r_}</span>"
    return v
_u_rows = []; _prev = None
from collections import Counter as _Counter
_keep = [k for k, g in enumerate(_G) if g != "x"]          # MEASURED columns only: every extrapolated column lives in section 6
_cnt = _Counter(r[0] for r in _rows); _seen = _Counter()
for r in _rows:
    first = r[0] != _prev; _prev = r[0]; _seen[r[0]] += 1; lastrow = _seen[r[0]] == _cnt[r[0]]
    tds = []
    for k in _keep:
        v = r[k]
        if k == 0: tds.append(f"<td class='l bench' rowspan={_cnt[r[0]]}>{e(v)}</td>" if first else ""); continue
        if k == 1: tds.append(f"<td class=l>{'base pipeline' if v.startswith('base') else e(v)}</td>"); continue
        if k == 2:
            _w = "".join(ch for ch in v if ch.isalpha()); _w = "queued" if _w == "todo" else _w
            tds.append(f"<td class=l><span class='chip {e(_w)}'><i></i>{e(_w)}</span></td>"); continue
        tds.append(f"<td>{_cell(v) if v else '&middot;'}</td>")
    _u_rows.append(f"<tr class='{'first sep' if first else ''}{' last' if lastrow else ''}{' baserow' if r[1].startswith('base') else ''}'>{''.join(tds)}</tr>")
_thead = (f"<tr><th colspan=4></th><th colspan={len(_keep) - 5} class=gm-h>Measured on this 4-GPU box</th><th></th></tr><tr>"
          + "".join(f"<th{' class=l' if k < 3 else ''}>{_label(_hdr[k])}</th>" for k in _keep) + "</tr>")
UNI = f"<thead>{_thead}</thead><tbody>{''.join(_u_rows)}</tbody>"
page = open("logs/phase3/extrap/page_template.html").read()
page = page.replace("{{FIGDATA}}", json.dumps(FIG).replace("</", "<\\/"))
for k, v in {"{{UPDATED}}": e(upd), "{{NDONE}}": str(ndone), "{{NTOTAL}}": str(len(prog)), "{{FINISH}}": e(fin), "{{ROWS}}": "".join(rows), "{{ALLROW}}": allrow,
             "{{PROGRESS}}": prow_html, "{{DIAG}}": "".join(dg), "{{XTABLE}}": XTABLE, "{{REF2}}": REF2, "{{UNI}}": UNI, "{{ALLNOTE}}": e(ALLNOTE)}.items():
    page = page.replace(k, v)
open(OUT, "w").write(page)
print("wrote", OUT, "measured-full sets", ds)
