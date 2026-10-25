"""Build the FINAL Phase 3 results page. Every number comes from results/phase3/final_numbers.json (written by final_report.py from the raw
per-item records), plus the per-item records themselves (progress table) and Phase 2's p2_metrics.json (reference accuracies of
Stock / CoDA-1.7B / Qwen3-0.6B in Figure 2). Prose is rendered from results/phase3/REPORT.md and code/phase3/README.md, so the page and the
report cannot disagree.

Usage (repo root):
  /venv/p2/bin/python -I logs/phase3/extrap/final_report.py
  /venv/p2/bin/python -I logs/phase3/extrap/build_final.py results/phase3/final_numbers.json results/phase3/REPORT.md logs/phase3/extrap/phase3_redraft.html
"""
import glob, html, json, os, re, sys, datetime as dt

NUM, REPORT, OUT = sys.argv[1:4]
N = json.load(open(NUM)); B = N["benchmarks"]; S = N["samples"]
P2 = json.load(open("logs/phase3/extrap/p2_metrics.json"))
RT = open(REPORT).read()
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
e = html.escape
ORDER = ["humaneval", "mbpp", "gsm8k500", "ifeval541", "math500", "gpqa", "aime"]          # the Phase 2 page's order
NAMES = {"humaneval": "HumanEval", "mbpp": "MBPP", "gsm8k500": "GSM8K-500", "ifeval541": "IFEval-541", "math500": "MATH-500",
         "gpqa": "GPQA-Diamond", "aime": "AIME"}
XSENT = ("EXTRAPOLATED — based on the extrapolation using the measured numbers on 4× GPUs for this dataset for this model, using reference "
         "from the runs from this model (the base pipeline measured on the Phase 1 8× box).")
XT = "<b class=xt>extrapolated</b>"; XTA = "<b class=xt>extrapolated, drift-adjusted</b>"; MT = "<b class=mt>measured on the Phase 1 box</b>"


def n_exp(s): return P2[s]["S0_nopool"]["n"]                       # benchmark size
def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def sg(v, d=1): return ("&minus;" if v < 0 else "+") + f"{abs(v):.{d}f}"          # same sign convention as REPORT.md (+0.0 stays +0.0)
def dash(): return "&middot;"
def wall(m): return f"{m:.1f} min" if m else dash()


# ---------------------------------------------------------------- tiny markdown renderer (REPORT.md / README.md prose)
def inline(t):
    codes = []
    def stash(m): codes.append(m.group(1)); return f"\x00{len(codes) - 1}\x00"
    t = re.sub(r"`([^`]+)`", stash, t)                              # code spans first, so ** inside them (e.g. `decode/**`) is left alone
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: "<code>" + html.escape(codes[int(m.group(1))], quote=False) + "</code>", t)


def md_blocks(lines):
    out, para, i = [], [], 0
    def flush():
        if para: out.append(("p", " ".join(para))); para.clear()
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            flush(); j = i + 1; code = []
            while j < len(lines) and not lines[j].startswith("```"): code.append(lines[j]); j += 1
            out.append(("code", "\n".join(code))); i = j + 1; continue
        if l.startswith("|") or l.startswith("#"): flush(); i += 1; continue          # tables are rebuilt from final_numbers.json
        m = re.match(r"^(-|\d+\.)\s+(.*)$", l)
        if m:
            flush(); kind = "ol" if m.group(1)[0].isdigit() else "ul"; items = [m.group(2)]; i += 1
            while i < len(lines):
                m2 = re.match(r"^(-|\d+\.)\s+(.*)$", lines[i])
                if m2: items.append(m2.group(2)); i += 1
                elif lines[i].startswith("  ") and lines[i].strip(): items[-1] += " " + lines[i].strip(); i += 1
                else: break
            out.append((kind, items)); continue
        if not l.strip(): flush(); i += 1; continue
        para.append(l.strip()); i += 1
    flush(); return out


def md_html(blocks):
    h = []
    for kind, v in blocks:
        if kind == "p": h.append(f"<p>{inline(v)}</p>")
        elif kind in ("ul", "ol"): h.append(f"<{kind}>" + "".join(f"<li>{inline(x)}</li>" for x in v) + f"</{kind}>")
        elif kind == "code": h.append(f"<pre class=code><code>{html.escape(v)}</code></pre>")
    return "".join(h)


def sections(text):
    secs, cur = {}, None
    for l in text.splitlines():
        m = re.match(r"^## (\d+)\. ", l)
        if m: cur = int(m.group(1)); secs[cur] = []
        elif cur is not None: secs[cur].append(l)
    return secs


RS = sections(RT)


def pick(sec, *prefixes):
    """paragraphs of a REPORT section whose text starts with one of the prefixes (in the order given)"""
    bl = md_blocks(RS.get(sec, [])); out = []
    for pf in prefixes:
        hit = [b for b in bl if b[0] == "p" and b[1].startswith(pf)]
        if hit: out.append(hit[0])
        else: print(f"WARNING: REPORT.md section {sec}: no paragraph starting with {pf!r}", file=sys.stderr)
    return md_html(out)


# ---------------------------------------------------------------- records (progress table)
def recinfo(pat):
    n, last = 0, None
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except ValueError: continue
            n += 1; u = r.get("utc")
            if u and (last is None or u > last): last = u
    return n, last
def ist(u): return dt.datetime.strptime(u, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(IST).strftime("%d %b %H:%M") if u else dash()
NOW = dt.datetime.now(dt.timezone.utc)
def alive(u): return bool(u) and (NOW - dt.datetime.strptime(u, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)).total_seconds() < 20 * 60
def chip(kind, txt): return f"<span class='chip {kind}'><i></i>{e(txt)}</span>"


def state_of(done_marker, n, total, last):
    if os.path.exists(done_marker) and n >= total: return chip("done", "done")
    if n == 0: return chip("stopped", "not run: stopped by the lead")
    return chip("running", "running") if alive(last) else chip("stopped", f"stopped at {n}/{total}")


prog = []
for s in ORDER:
    total = n_exp(s); full = s in B
    for a in ("rd1", "rd2", "base"):
        if a == "base":
            pat, mk = (f"logs/phase3/decode/judge/{s}/*.jsonl", f"logs/phase3/decode/judge/{s}/.complete") if full else (f"logs/phase3/decode/judge_sample/{s}/*.jsonl", None)
        else:
            pat, mk = f"logs/phase3/decode/{a}/{s}/*.jsonl", f"logs/phase3/decode/{a}/{s}/.complete"
        n, last = recinfo(pat)
        if a == "base" and not full:
            st = chip("done", "done (first items only)") if n else chip("stopped", "not run"); tot = f"{n}/{total}"
        else:
            st = state_of(mk, n, total, last); tot = f"{n}/{total}"
        if full: w = (B[s]["wall_base_min"] if a == "base" else B[s]["arms"][a]["wall_here_min"])
        else: w = None
        label = "base pipeline" if a == "base" else a
        prog.append(f"<tr><td class=l>{label}</td><td class=l>{NAMES[s]}</td><td>{tot}</td><td>{wall(w)}</td><td>{ist(last) if n else dash()}</td><td class=l>{st}</td></tr>")
PROGRESS = "".join(prog)

# ---------------------------------------------------------------- status banner and summary
complete, partial = [], []
for s, b in B.items():
    n = n_exp(s)
    if b["full_base_run"] and b["base_here"]["n"] == n and all(x["all_items"]["n"] == n for x in b["arms"].values()): complete.append(b["name"])
    else:
        bits = [f"{a} {x['all_items']['n']}/{n}" for a, x in b["arms"].items() if x["all_items"]["n"] != n]
        if not (b["full_base_run"] and b["base_here"]["n"] == n): bits.append(f"base pipeline {b['base_here']['n']}/{n}")
        partial.append(f"{b['name']} ({', '.join(bits)})")
notrun = [v["name"] for v in S.values()]
is_final = not partial
banner = (f"<p class=banner><b>{'FINAL.' if is_final else 'NOT FINAL YET.'}</b> The project lead stopped the run after GSM8K-500. "
          f"Finished on every item with all three setups (base pipeline, rd1, rd2): <b>{', '.join(complete) or 'none'}</b>. "
          + (f"Still incomplete: {e('; '.join(partial))}. " if partial else "")
          + f"Not run: rd1 and rd2 on <b>{', '.join(notrun)}</b> (the base pipeline was decoded on their first items only, Table 2).</p>")
gen = re.search(r"^Generated (.*?) by ", RT, re.M)
GENERATED = f"numbers computed {gen.group(1)}." if gen else ""
STATUSLINE = (f"<span>Updated <b>{NOW.astimezone(IST).strftime('%d %b %H:%M')} IST</b></span>"
              f"<span><b>{'Final' if is_final else 'Not final'}</b>: stopped by the project lead after GSM8K-500</span>"
              f"<span><b>{len(complete)}</b> of {len(ORDER)} benchmarks fully measured</span>")

sm = []
for s, b in B.items():
    base = b["base_here"]; bits = []
    for a, x in b["arms"].items():
        m = x["all_items"]
        bits.append(f"{a} {f3(x['speedup_raw'])}&times; (passes per block {f2(m['passes_blk'])}, expert loads per token {m['loads_tok']:.0f}, accuracy change vs base {sg(x['delta_acc_vs_base'][0])} pts)")
    sm.append(f"<li><b>{b['name']}</b>: raw speedup over the base pipeline " + "; ".join(bits) +
              f"; the base pipeline has passes per block {f2(base['passes_blk'])} and expert loads per token {base['loads_tok']:.0f}.</li>")
SUMMARY = ("<p class=caption><b>At a glance</b> (measured, raw, all from Table 1; the host-speed drift of section 3 is not removed from the speedups):</p>"
           "<ul class=prose>" + "".join(sm) + "</ul>")

# ---------------------------------------------------------------- Table 1 (measured)
mrows = []
for s in ORDER:
    if s not in B: continue
    b = B[s]; base = b["base_here"]; n = n_exp(s); grp = []
    part = "" if (b["full_base_run"] and base["n"] == n) else f" <span class=pt>running {base['n']}/{n}</span>"
    grp.append(f"<td class=l>base pipeline</td><td>{base['n']}{part}</td><td>{f1(base['acc'])}</td><td class=na>reference</td><td>{f2(base['passes_blk'])}</td><td>{base['loads_tok']:.0f}</td>"
               f"<td>{f2(base['drafter_fwd_tok'])}</td><td>{f2(base['tok_s'])}</td><td>{f1(base['ms_tok'])}</td><td>{wall(b['wall_base_min'])}</td><td>1.00 <span class=ci>(reference)</span></td>")
    for a, x in b["arms"].items():
        m = x["all_items"]; d = x["delta_acc_vs_base"]
        pb = "" if x["paired_items"] == n else " <span class=pt>partial base</span>"
        grp.append(f"<td class=l>{a}</td><td>{m['n']}</td><td>{f1(m['acc'])}</td><td>{sg(d[0])} <span class=ci>[{sg(d[1])}, {sg(d[2])}]</span></td><td>{f2(m['passes_blk'])}</td><td>{m['loads_tok']:.0f}</td>"
                   f"<td>{f2(m['drafter_fwd_tok'])}</td><td>{f2(m['tok_s'])}</td><td>{f1(m['ms_tok'])}</td><td>{wall(x['wall_here_min'])}</td>"
                   f"<td><b>{f3(x['speedup_raw'])}&times;</b> <span class=ci>({x['paired_items']} paired items){pb}</span></td>")
    for i, g in enumerate(grp):
        lead = f"<td class='l bench' rowspan=3>{b['name']}</td>" if i == 0 else ""
        mrows.append(f"<tr class='{'first sep' if i == 0 else ''}{' last' if i == 2 else ''}'>{lead}{g}</tr>")
MTABLE = "".join(mrows)

# ---------------------------------------------------------------- Table 2 (base pipeline vs Phase 1)
brows = []
for s in ORDER:
    if s in B:
        b = B[s]; n = n_exp(s); i, t = b["base_identical_to_phase1"]
        brows.append(f"<tr><td class=l>{b['name']}</td><td>{b['base_here']['n']}{'' if b['base_here']['n'] == n else f' of {n}'}</td><td>{i}/{t}</td><td>{f2(b['base_here']['tok_s'])}</td><td>{f3(b['box_ratio_base'])}</td></tr>")
    else:
        v = S[s]
        brows.append(f"<tr><td class=l>{v['name']} <span class=pt>first items only</span></td><td>{v['items']}</td><td>{v['identical_to_phase1']}/{v['items']}</td><td>{f2(v['base_tps_here'])}</td><td>{f3(v['box_ratio'])}</td></tr>")
BASETABLE = "".join(brows)

# ---------------------------------------------------------------- Tables 3, 4 (drift)
hrows, arows = [], []
for s in ORDER:
    if s not in B: continue
    for a, x in B[s]["arms"].items():
        hrows.append(f"<tr><td class=l>{B[s]['name']}</td><td class=l>{a}</td><td>{f3(x['host_index_30b_prefill'])}</td><td>{f3(x['host_index_drafter_prefill'])}</td>"
                     f"<td>{f3(x['host_index_first_half'])}</td><td>{f3(x['host_index_second_half'])}</td></tr>")
        arows.append(f"<tr><td class=l>{B[s]['name']}</td><td class=l>{a}</td><td><b>{f3(x['speedup_raw'])}&times;</b></td><td>{f3(x['host_index_30b_prefill'])}</td>"
                     f"<td class=ax>{f3(x['speedup_drift_adjusted'])}&times;<b class=adj>adjusted, NOT a measurement</b></td></tr>")
HTABLE, ADJTABLE = "".join(hrows), "".join(arows)

# ---------------------------------------------------------------- Table 5 (extrapolated)
xrows = []
for s in ORDER:
    if s not in B: continue
    b = B[s]
    for k, (a, x) in enumerate(b["arms"].items()):
        lead = f"<td class='l bench' rowspan=2>{b['name']}</td>" if k == 0 else ""
        xrows.append(f"<tr class='{'first sep' if k == 0 else 'last'}'>{lead}<td class=l>{a}</td>"
                     f"<td class='xc basecell'>{f1(b['base_phase1_box']['tok_s'])} <span class=ci>(stock {f1(b['stock_tps_phase1_box'])})</span>{MT}</td>"
                     f"<td class=xc><b>{f1(x['extrap_tps_raw'])}</b>{XT}</td><td class=xc>{f2(x['extrap_x_stock_raw'])}&times;{XT}</td>"
                     f"<td class=xc>{f1(x['extrap_tps_adjusted'])} <span class=ci>({f2(x['extrap_x_stock_adjusted'])}&times; stock)</span>{XTA}</td>"
                     f"<td class=xc>{f1(x['extrap_wall_8gpu_min'])} min{XT}</td></tr>")
XTABLE = "".join(xrows)
XNOTRUN = (f"<p class=caption>Not run, so nothing to extrapolate: rd1 and rd2 on {', '.join(notrun)}.</p>")

# ---------------------------------------------------------------- figures
FIG = []
for s in ORDER:
    nm = NAMES[s]; n = n_exp(s)
    if s not in B:
        FIG.append({"name": nm, "n": n, "pts": [], "live": False, "msg": "rd1 and rd2 not run (stopped after GSM8K-500)"}); continue
    b = B[s]; base = b["base_here"]; pts = []
    pts.append({"label": "Stock 30B (Phase 2 records)", "tiny": "Stock", "shape": "diamond", "color": "--s-stock", "acc": P2[s]["stock"]["acc"], "err": None,
                "note": "Phase 2 records, not measured here"})
    pts.append({"label": "Base pipeline (this box)", "tiny": "Base", "shape": "circle", "color": "--s-sdar", "sp": 1.0, "acc": base["acc"], "err": None,
                "tps": base["tok_s"], "note": f"measured here, {base['n']} items; reference = 1x"})
    pts.append({"label": "CoDA-1.7B + judge (Phase 2 records)", "tiny": "CoDA", "shape": "square", "color": "--s-coda", "acc": P2[s]["F2_nopool"]["acc"], "err": None,
                "note": "Phase 2 records, not measured here"})
    pts.append({"label": "Qwen3-0.6B + judge (Phase 2 records)", "tiny": "Qwen3", "shape": "triangle", "color": "--s-qwen", "acc": P2[s]["S1_nopool"]["acc"], "err": None,
                "note": "Phase 2 records, not measured here"})
    for a, shape, col in (("rd1", "pent", "--s-rd1"), ("rd2", "star", "--s-rd2")):
        x = b["arms"][a]; m = x["all_items"]; d = x["delta_acc_vs_base"]
        pts.append({"label": a + (" (re-draft every pass)" if a == "rd1" else " (re-draft every 2 passes)"), "tiny": a, "shape": shape, "color": col,
                    "sp": x["speedup_raw"], "acc": m["acc"], "err": [m["acc"] + (d[1] - d[0]), m["acc"] + (d[2] - d[0])], "tps": m["tok_s"],
                    "note": f"raw, measured here on {x['paired_items']} paired items vs the base pipeline (host drift not removed); accuracy delta vs base {d[0]:+.1f} pts [{d[1]:+.1f}, {d[2]:+.1f}]"})
    FIG.append({"name": nm, "n": n, "pts": pts, "live": True})

# ---------------------------------------------------------------- prose
SEC1 = md_html(md_blocks(RS[1])).replace("(section 6)", "(Table 2)")
_gf = re.search(r"`gpu_frac` ([0-9.]+)", RT)
if _gf:
    SEC1 += (f"<p><b>Shared host.</b> The box is a shared, rented server: <code>gpu_frac</code> is {_gf.group(1)} (we rent part of the machine's GPUs; another tenant uses the rest). "
             "Section 3 shows what that does to the measured speed.</p>")
rd = open("code/phase3/README.md").read().splitlines()
i0 = next(i for i, l in enumerate(rd) if l.startswith("## Method")); i1 = next(i for i in range(i0 + 1, len(rd)) if rd[i].startswith("## "))
METHOD = md_html(md_blocks(rd[i0 + 1:i1]))
DRIFT1 = pick(3, "**Evidence the host is shared:**", "This is a shared, rented server", "**How it is measured:**")
DRIFT2 = pick(3, "**Reading:**")
DRIFT3 = pick(4, "**How to avoid this next time:**")
xs = pick(4, "Formula:")
XFORMULA = xs
SEC5 = md_html(md_blocks(RS[5]))
TIMING = ("<p><b>Timing in detail.</b> Per-item <code>seconds</code> is set in <code>decode_item</code> (<code>code/draftalign/da_decode.py</code>): "
          "<code>torch.cuda.synchronize()</code>, then <code>time.perf_counter()</code>, then the whole decode, then <code>torch.cuda.synchronize()</code> and "
          "<code>time.perf_counter()</code> again. The timed span contains prompt prefill, every judge, drafter, re-draft and KV-flush forward, and sampling; "
          "it excludes tokenisation (<code>encode_prompt</code>), scoring, trace saving and writing the record (<code>code/draftalign/run_da_decode.py</code>). "
          "The per-forward component times (<code>ms_by_kind</code>, <code>drafter_ms_by_kind</code>, <code>SeqCache._timed</code> in <code>code/draftalign/seqcache.py</code>) also synchronize "
          "before and after each forward, identically in every setup. There are no busy-polling loops, CUDA-event polling or sleeps in the timed path. "
          "Wall in Tables 0 and 1 is the run's start-to-done time from <code>logs/phase3/run.log</code>, model load included.</p>")
SEC7 = md_html(md_blocks(RS[7]))
SEC8 = md_html(md_blocks(RS[8]))
REBUILD = ("<p>Rebuild this page from the report numbers:</p><pre class=code><code>python logs/phase3/extrap/final_report.py\n"
           "python logs/phase3/extrap/build_final.py results/phase3/final_numbers.json results/phase3/REPORT.md logs/phase3/extrap/phase3_redraft.html</code></pre>")

FM = [
    ("logs/phase3/run.log", "logs/phase3/run.log", "start / done / skip / pause lines of the whole run (UTC, time of day only)."),
    ("logs/phase3/decode/rd1/&lt;benchmark&gt;/, decode/rd2/&lt;benchmark&gt;/", "logs/phase3/decode/rd1", "rd1 and rd2: one JSON record per item (<code>*.jsonl</code>, one file per shard), <code>.jsonl.done</code> shard markers, <code>.complete</code> when every shard finished."),
    ("logs/phase3/decode/judge/&lt;benchmark&gt;/", "logs/phase3/decode/judge", "base pipeline (SDAR-1.7B drafter + trained 128M judge), full-set runs."),
    ("logs/phase3/decode/judge_sample/&lt;benchmark&gt;/", "logs/phase3/decode/judge_sample", "base pipeline on the first items only (the benchmarks the run was stopped before)."),
    ("logs/phase3/jobs/&lt;setup&gt;/&lt;benchmark&gt;/", "logs/phase3/jobs", "stdout of every shard process (<code>shardK.attemptN.log</code>)."),
    ("logs/phase3/setup/", "logs/phase3/setup", "setup checks: <code>checks.json</code>, <code>tests.log</code>, <code>check_stock/</code>, <code>check_judge/</code> (identity checks against the Phase 1 records)."),
    ("logs/phase3/analysis/, analysis.log, analysis_snapshot.log", "logs/phase3/analysis", "output of <code>code/phase3/analyze_p3.py</code> (<code>TABLE.md</code>, <code>metrics.json</code>)."),
    ("logs/phase3/extrap/", "logs/phase3/extrap", "<code>final_report.py</code> (the report and <code>final_numbers.json</code>), <code>build_final.py</code> and <code>final_template.html</code> (this page), <code>extrap.py</code>, <code>status.py</code>, <code>unified.py</code>, <code>build_artifact.py</code> (the live page while the run was going), <code>p2_metrics.json</code> (Phase 2 page metrics), and the shell scripts that ordered the run."),
    ("logs/phase3/backup.log, snapshot.log, switch.log", "logs/phase3/backup.log", "backup uploads, the periodic snapshot loop, and the run-order switches."),
    ("logs/phase3/STOP", "logs/phase3/STOP", "empty marker file: the stop signal, checked by <code>run_p3.sh</code> before it starts the next benchmark."),
    ("logs/phase3/STOPPED_setup_tests.md", "logs/phase3/STOPPED_setup_tests.md", "note from the first stop, at setup, before any decoding (two CPU tests failed on a missing <code>drafter</code> field in their fake arguments); the run was resumed afterwards."),
    ("results/phase3/", "results/phase3", "<code>REPORT.md</code> and <code>final_numbers.json</code> (this page's source), <code>TABLE.md</code>, <code>metrics.json</code>, <code>EXTRAPOLATION_PHASE1_BOX.md</code>, <code>RESUME.md</code>, <code>setup_checks.json</code>."),
    ("code/phase3/", "code/phase3", "<code>run_p3.sh</code> (the driver), <code>setup_p3.sh</code>, <code>analyze_p3.py</code>, <code>backup_p3.py</code>, <code>README.md</code> (method), <code>extrap/</code> (copies of the page and report scripts)."),
    ("code/draftalign/", "code/draftalign", "the decoder: <code>run_da_decode.py</code> (runner), <code>da_decode.py</code> (decode loop, timing, records), <code>seqcache.py</code> (per-forward timers), <code>drafter.py</code>."),
    ("weights/c1b_128M/ (Hugging Face dataset)", None, "the trained 128M judge LoRA adapter used by every setup (weights hash 35805974)."),
    ("ref/phase2/S0_128M_nopool_by_set/ (Hugging Face dataset)", None, "the Phase 1 box's base-pipeline per-item records, used as the reference for token identity and for the Phase 1 box tokens per second."),
]
filemap = []
for disp, chk, desc in FM:
    if chk is not None and not os.path.exists(chk): print(f"WARNING: file map path missing: {chk}", file=sys.stderr); continue
    filemap.append(f"<tr><td class=l><code>{disp}</code></td><td class=l style='white-space:normal'>{desc}</td></tr>")
FILEMAP = "".join(filemap)

# ---------------------------------------------------------------- assemble
tpl = open("logs/phase3/extrap/page_template.html").read()
head = tpl[:tpl.index('<main class="wrap">')]
tail = tpl[tpl.index('<script type="application/json" id="fig-data">'):]
extra = """<style>
.prose { display: grid; gap: 10px; max-width: 112ch; font-size: 0.95rem; line-height: 1.55; color: var(--ink-2); }
.prose p, .prose ul, .prose ol { margin: 0; } .prose ul, .prose ol { padding-left: 1.3em; display: grid; gap: 6px; } .prose b { color: var(--ink); font-weight: 600; }
code { font-family: var(--font-mono); font-size: 0.86em; background: color-mix(in srgb, var(--ink) 6%, transparent); padding: 0 0.25em; border-radius: 3px; overflow-wrap: anywhere; }
pre.code { margin: 0; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.5; background: var(--card); border: 1px solid var(--hair); border-radius: 6px; padding: 12px 14px; overflow-x: auto; }
pre.code code { background: none; padding: 0; }
.banner { margin: 0; padding: 12px 16px; border-left: 4px solid var(--accent); background: var(--card); font-family: var(--font-sans); font-size: 0.95rem; line-height: 1.5; color: var(--ink-2); max-width: 112ch; }
.banner b { color: var(--ink); }
section.msec { border-left: 4px solid var(--pick); padding-left: 16px; }
section.xsec { border-left: 4px solid var(--warn); padding-left: 16px; }
.chip.stopped { color: var(--warn); box-shadow: inset 0 0 0 1px var(--warn); }
b.adj { display: block; font-family: var(--font-sans); font-size: 0.6rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; line-height: 1.25; color: var(--warn); white-space: nowrap; }
b.adjtxt { color: var(--warn); }
.xsent { font-size: 1.05rem; } .xsent b { color: var(--warn); }
table.sm td.ax { background: color-mix(in srgb, var(--warn) 7%, transparent); }
table.sm td.na { color: var(--muted); font-style: italic; }
table.sm td.l code { white-space: normal; }
</style>
"""
body = open("logs/phase3/extrap/final_template.html").read()
rep = {"{{STATUSLINE}}": STATUSLINE, "{{BANNER}}": banner, "{{SEC1}}": SEC1, "{{METHOD}}": METHOD, "{{PROGRESS}}": PROGRESS, "{{SUMMARY}}": SUMMARY,
       "{{MTABLE}}": MTABLE, "{{BASETABLE}}": BASETABLE, "{{DRIFT1}}": DRIFT1, "{{HTABLE}}": HTABLE, "{{DRIFT2}}": DRIFT2, "{{ADJTABLE}}": ADJTABLE,
       "{{DRIFT3}}": DRIFT3, "{{XSENT}}": e(XSENT), "{{XFORMULA}}": XFORMULA, "{{XTABLE}}": XTABLE, "{{XNOTRUN}}": XNOTRUN, "{{SEC5}}": SEC5,
       "{{TIMING}}": TIMING, "{{SEC7}}": SEC7, "{{SEC8}}": SEC8, "{{REBUILD}}": REBUILD, "{{FILEMAP}}": FILEMAP, "{{GENERATED}}": e(GENERATED)}
for k, v in rep.items():
    assert k in body, k
    body = body.replace(k, v)
tail = tail.replace("{{FIGDATA}}", json.dumps(FIG).replace("</", "<\\/"))
page = head + extra + body + tail
assert "{{" not in page.replace("{{FIGDATA}}", ""), re.findall(r"\{\{\w+\}\}", page)
open(OUT, "w").write(page)
print("wrote", OUT, "| final:", is_final, "| complete:", complete, "| partial:", partial)
