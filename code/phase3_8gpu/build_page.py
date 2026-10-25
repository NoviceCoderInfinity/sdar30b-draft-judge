#!/usr/bin/env python
"""Results page of the 8-GPU Phase 3 re-run: results/phase3_8gpu/page.html, built from final_numbers.json (run
report_8gpu.py first) and the run's event log. Static HTML plus one small script: the wall-clock / GPU-only speedup
switch at the top (every speedup cell carries both values; without the script the wall-clock values show). Claude
republishes it as the artifact.

    /venv/p2/bin/python code/phase3_8gpu/build_page.py
"""
import datetime as dt
import html
import json
import os

import redraft_why   # sections 7-8 (why re-drafting gains so little; feeding the judge's verdict back), from its own JSON

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
L = "logs/phase3_8gpu"
OUT = "results/phase3_8gpu/page.html"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ORDER = [("ifeval541", "IFEval-541"), ("math500", "MATH-500"), ("gpqa", "GPQA-Diamond"), ("aime", "AIME"),
         ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500")]
E = html.escape

CSS = """
/* Layout: the Phase 2/3 preprint look, continued: one 1200px column of prose and wide tables; tables scroll inside their own box. */
:root {
  --paper: #f6f6f9; --card: #ffffff; --ink: #16172a; --ink-2: #46485c; --muted: #6c6e82; --hair: #e0e1e9;
  --rule: #16172a; --accent: #3b3f9a; --pick: #1e7f5c; --warn: #b4540a; --s-rd1: #4a3aa7; --s-rd2: #c2477a;
  --font-serif: "STIX Two Text", "STIX Two", "Times New Roman", Times, serif;
  --font-sans: "Instrument Sans", "Helvetica Neue", Arial, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark; --paper: #14151f; --card: #1b1c29; --ink: #e8e9f1; --ink-2: #b6b8c9; --muted: #8c8ea3;
    --hair: #2a2c3c; --rule: #d8d9e5; --accent: #a9adf3; --pick: #4cba8e; --warn: #f0a050; --s-rd1: #9085e9; --s-rd2: #e07aa4;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark; --paper: #14151f; --card: #1b1c29; --ink: #e8e9f1; --ink-2: #b6b8c9; --muted: #8c8ea3;
  --hair: #2a2c3c; --rule: #d8d9e5; --accent: #a9adf3; --pick: #4cba8e; --warn: #f0a050; --s-rd1: #9085e9; --s-rd2: #e07aa4;
}
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font-family: var(--font-serif); font-size: 1.0625rem; line-height: 1.55; }
.wrap { max-width: 1200px; margin: 0 auto; padding-inline: clamp(16px, 4vw, 44px); padding-block: 44px 72px; display: grid; gap: 48px; }
section { display: grid; gap: 16px; min-width: 0; }
section > * { min-width: 0; }
p { margin: 0; }
h1, h2 { margin: 0; font-weight: 600; text-wrap: balance; }
h1 { font-size: 2.4rem; line-height: 1.1; letter-spacing: -0.012em; }
h2 { font-size: 1.45rem; line-height: 1.25; display: flex; gap: 12px; align-items: baseline; }
h2 .no { font-family: var(--font-sans); font-size: 0.85rem; font-weight: 500; color: var(--accent); }
.kicker { font-family: var(--font-sans); font-size: 0.75rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); }
.masthead { display: grid; gap: 12px; padding-bottom: 24px; border-bottom: 1px solid var(--hair); }
.lede { font-size: 1.2rem; line-height: 1.45; color: var(--ink-2); max-width: 66ch; }
.status { font-family: var(--font-sans); font-size: 0.875rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 18px; font-variant-numeric: tabular-nums; }
.status b { color: var(--ink); font-weight: 600; }
.banner { padding: 12px 16px; border-left: 4px solid var(--warn); background: var(--card); font-family: var(--font-sans); font-size: 0.95rem; line-height: 1.5; color: var(--ink-2); max-width: 112ch; }
.banner.ok { border-left-color: var(--pick); }
.banner b { color: var(--ink); }
.caption { font-size: 0.95rem; color: var(--ink-2); max-width: 112ch; }
.caption b { color: var(--ink); }
.prose { display: grid; gap: 8px; font-size: 0.95rem; color: var(--ink-2); max-width: 112ch; }
.prose ul { margin: 0; padding-left: 1.3em; display: grid; gap: 5px; }
.prose b { color: var(--ink); }
code { font-family: var(--font-mono); font-size: 0.85em; background: color-mix(in srgb, var(--ink) 6%, transparent); padding: 0 0.25em; border-radius: 3px; overflow-wrap: anywhere; }
.tbl { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-family: var(--font-sans); font-size: 0.86rem; font-variant-numeric: tabular-nums;
  border-top: 1.5px solid var(--rule); border-bottom: 1.5px solid var(--rule); }
thead { border-bottom: 0.8px solid var(--rule); }
th { font-weight: 500; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); text-align: right; padding: 8px 9px; vertical-align: bottom; }
td { padding: 6px 9px; text-align: right; border-top: 1px solid var(--hair); color: var(--ink-2); white-space: nowrap; }
.l { text-align: left; }
td.l { color: var(--ink); }
td.bench { font-weight: 600; vertical-align: top; }
tr.ref td { font-style: italic; color: var(--muted); }
td.na { color: var(--muted); }
.ci { color: var(--muted); }
.tag { display: block; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--warn); white-space: nowrap; }
.tag.m { color: var(--pick); }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 1px 10px 1px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }
.chip i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.chip.done { background: var(--ink); color: var(--paper); }
.chip.run { color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.chip.queued { color: var(--muted); box-shadow: inset 0 0 0 1px var(--hair); }
.chip.queued i { background: transparent; box-shadow: inset 0 0 0 1.5px currentColor; }
.arm-rd1 { color: var(--s-rd1); font-weight: 600; }
.arm-rd2 { color: var(--s-rd2); font-weight: 600; }
.sumtbl td.acc { font-weight: 600; color: var(--ink); }
.ok { color: var(--pick); font-weight: 600; }
.bad { color: var(--warn); font-weight: 600; }
footer { border-top: 1px solid var(--hair); padding-top: 14px; font-family: var(--font-sans); font-size: 0.8rem; color: var(--muted); }
/* speedup switch: every speedup cell holds a wall-clock and a GPU-only value; main[data-mode] picks one */
.modebar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px;
  padding: 10px 0; background: var(--paper); border-bottom: 1px solid var(--hair); font-family: var(--font-sans); font-size: 0.85rem; color: var(--muted); }
.modebar .seg { display: inline-flex; border-radius: 999px; box-shadow: inset 0 0 0 1px var(--accent); padding: 2px; gap: 2px; }
.modebar button { font: inherit; font-weight: 500; border: 0; border-radius: 999px; padding: 5px 14px; background: transparent; color: var(--accent); cursor: pointer; }
.modebar button[aria-pressed="true"] { background: var(--accent); color: var(--paper); }
.modebar button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.modebar .what { max-width: 70ch; }
.v-gpu { display: none; }
main[data-mode="gpu"] .v-gpu { display: inline; }
main[data-mode="gpu"] .v-wall { display: none; }
.ast { color: var(--warn); font-weight: 700; margin-left: 2px; }
th .mode { display: block; font-size: 0.6rem; color: var(--accent); }
"""

SCRIPT = """<script>(function () {
  var m = document.querySelector("main.wrap"), bs = document.querySelectorAll(".modebar button");
  function set(v) {
    m.setAttribute("data-mode", v);
    bs.forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-m") === v ? "true" : "false"); });
    try { localStorage.setItem("p3g-speedup-mode", v); } catch (e) {}
  }
  bs.forEach(function (b) { b.addEventListener("click", function () { set(b.getAttribute("data-m")); }); });
  var s = location.hash === "#gpu" ? "gpu" : null;
  if (!s) { try { s = localStorage.getItem("p3g-speedup-mode"); } catch (e) {} }
  if (s === "gpu") set("gpu");
})();</script>"""

MODE_HEAD = '<span class="mode"><span class="v-wall">wall-clock</span><span class="v-gpu">GPU-only</span></span>'
AST = '<span class="v-gpu ast" title="No GPU-time record exists for one factor of this row; its wall-clock value is used">*</span>'
P2_STOCK_MEASURED = {"ifeval541"}   # Phase 2 Table 1: stock timed on the Phase 1 machine type for IFEval only; the rest computed


def sp(w, g, p=3):
    """One speedup cell: the wall-clock value, and the GPU-only value shown when the switch is on GPU-only."""
    return f'<span class="v-wall">{f(w, p, "&times;")}</span><span class="v-gpu">{f(g, p, "&times;")}</span>'


def f(v, p=2, suf=""):
    return "·" if v is None else f"{v:,.{p}f}{suf}"


def sd(v, p=1):
    if v is None:
        return "·"
    s = f"{abs(v):.{p}f}"
    return s if float(s) == 0 else ("+" if v > 0 else "−") + s


def utc_to_ist(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(IST).strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return "·"


def running_now():
    st = {}
    if os.path.exists(f"{L}/events.tsv"):
        for line in open(f"{L}/events.tsv"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[2] in ("start_run", "end_run"):
                k = tuple(p[3].split()[:2])
                if p[2] == "start_run":
                    st[k] = p[1]
                else:
                    st.pop(k, None)
    return st


ARMS = ("rd1", "rd2", "base")
P1_URL = "https://claude.ai/artifact/8MHrYQCGnd3FZTrWa6KUvj"      # Phase 1 Judge Report
P2_URL = "https://claude.ai/artifact/V1Gtiz8t48YrsvWDCGPUvX"      # Phase 2 Drafter Swap
P3_4GPU_URL = "https://claude.ai/artifact/MDo497vHUmVX3zCDAbV9ny"  # Phase 3 Re-draft, first run (4 GPUs, shared host)
STOCK_ACC = {"ifeval541": 52.7, "math500": 72.4, "gpqa": 37.4, "aime": 6.7, "humaneval": 78.0, "mbpp": 65.4, "gsm8k500": 90.2}  # Phase 1 / 2 pages


def arm_obj(e, arm):
    return (e.get("arms") or {}).get(arm) if arm != "base" else e.get("base_here")


def pooled_here(B):
    """All 7 benchmarks pooled, this machine: rd over base = (sum rd tokens / sum rd seconds) / (the same for base), on the
    paired items, wall-clock and GPU-active; None unless every benchmark has its base run and both rd runs."""
    out = {"n": 0}
    base_n = base_c = 0
    for arm in ("rd1", "rd2"):
        t = {"rt": 0.0, "rs": 0.0, "rg": 0.0, "bt": 0.0, "bs": 0.0, "bg": 0.0, "n": 0, "c": 0}
        for s, _ in ORDER:
            v = (((B.get(s) or {}).get("arms") or {}).get(arm) or {}).get("vs_base_here")
            if not v or v["rd"].get("gpu_active_s") is None or v["base"].get("gpu_active_s") is None:
                return None
            r, b = v["rd"], v["base"]
            t["rt"] += r["tokens"]; t["rs"] += r["seconds"]; t["rg"] += r["gpu_active_s"]
            t["bt"] += b["tokens"]; t["bs"] += b["seconds"]; t["bg"] += b["gpu_active_s"]
            t["n"] += r["n"]; t["c"] += r["correct"]
            if arm == "rd1":
                base_n += b["n"]; base_c += b["correct"]
        out[arm] = {"wall": (t["rt"] / t["rs"]) / (t["bt"] / t["bs"]), "gpu": (t["rt"] / t["rg"]) / (t["bt"] / t["bg"]),
                    "acc": 100 * t["c"] / t["n"], "n": t["n"], "rd_gpu_tok_s": t["rt"] / t["rg"], "base_gpu_tok_s": t["bt"] / t["bg"]}
    out["n"] = base_n
    out["base_acc"] = 100 * base_c / base_n
    return out


def backup_line():
    try:
        st = json.load(open(f"{L}/backup_status.json"))
    except (OSError, ValueError):
        return ""
    gh, hf = st.get("github_ok_utc"), st.get("hf_ok_utc")
    last = st.get("last_status", "·")
    cls = "ok" if last == "ok" else "bad"
    return (f'<span>Backups: GitHub <b>{utc_to_ist(gh) if gh else "·"}</b>, Hugging Face <b>{utc_to_ist(hf) if hf else "·"}</b> IST '
            f'(last attempt <span class="{cls}">{E(last)}</span>)</span>')


def main():
    R = json.load(open("results/phase3_8gpu/final_numbers.json"))
    now = dt.datetime.now(dt.timezone.utc)
    run = running_now()
    B = R["benchmarks"]
    total_runs = len(ARMS) * len(ORDER)
    done_runs = sum(1 for s, _ in ORDER for a in ARMS if (arm_obj(B.get(s, {}), a) or {}).get("complete"))
    H = []
    a = H.append
    a("<title>Phase 3 Re-draft on 8 GPUs</title>")
    a('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&display=swap">')
    a(f"<style>{CSS}{redraft_why.css()}</style>")
    a('<main class="wrap">')
    # ---------------------------------------------------------------- masthead
    cur = "; ".join(f"{k[0]} {k[1]} since {utc_to_ist(v)} IST" for k, v in run.items()) or "none"
    a('<header class="masthead"><p class="kicker">Phase 3 &middot; 8&times; RTX PRO 6000 Server &middot; rd1, rd2 and base measured on one machine</p>')
    a("<h1>Phase 3 Re-draft on 8 GPUs</h1>")
    a('<p class="lede">rd1 and rd2 re-draft the slots the judge rejects, after every judge pass (rd1) or every second pass (rd2). '
      'Both ran on all seven benchmarks, every item, next to the base pipeline (SDAR-1.7B drafter + the 128M judge) on the same unshared machine, '
      'so rd1, rd2 and base compare directly. Speedups over the plain 30B (stock) are chained through the Phase 2 page and labelled as such.</p>')
    a(f'<p class="status"><span>Updated <b>{now.astimezone(IST).strftime("%d %b %H:%M")} IST</b></span>'
      f'<span><b>{done_runs}</b> of {total_runs} runs finished</span><span>Running: <b>{E(cur)}</b></span>{backup_line()}</p></header>')
    a('<div class="modebar" role="group" aria-label="Speedup measure"><span>Speedups shown as</span><span class="seg">'
      '<button type="button" id="mode-wall" data-m="wall" aria-pressed="true">Speedup (wall-clock)</button>'
      '<button type="button" id="mode-gpu" data-m="gpu" aria-pressed="false">GPU-only speedup</button></span>'
      '<span class="what v-wall">Tokens per second by wall-clock: CPU and GPU time together, what a user waits for.</span>'
      '<span class="what v-gpu">Tokens per GPU-active second (NVML counters): CPU launch overhead removed. '
      '<span class="ast">*</span> marks rows where one factor has no GPU-time record, so its wall-clock value is used.</span></div>')
    # ---------------------------------------------------------------- background
    a('<section><h2><span class="no">0</span>Background</h2><div class="prose">')
    a(f'<p><b>Earlier pages:</b> <a href="{P1_URL}">Phase 1 Judge Report</a> &middot; <a href="{P2_URL}">Phase 2 Drafter Swap</a> &middot; '
      f'<a href="{P3_4GPU_URL}">Phase 3 Re-draft, first run on a shared 4-GPU host</a> (its timings are void; superseded by this page).</p>')
    a('<ul><li><b>The system.</b> The target is the frozen MoE <b>SDAR-30B-A3B-Chat-b32</b> (48 layers &times; 128 experts, 8 per token; text is written in blocks of 32). '
      'A small drafter, <b>SDAR-1.7B-Chat-b32</b>, fills every slot of a block (8 steps); a <b>judge</b> &mdash; the same frozen 30B with a trained LoRA &mdash; reads the block and commits the slots it is confident about (p &gt; 0.95), keeps agreed-but-unsure drafts and re-masks the rest, pass after pass, until the block is committed.</li>'
      f'<li><b>Phase 1</b> (<a href="{P1_URL}">report</a>) trained that judge LoRA (r32 on q,k,v,o; 128M tokens). Without expert pooling its accuracy matches the plain 30B within noise on 7 of 7 benchmarks, '
      'while the 30B needs 1.30&ndash;1.41&times; fewer passes per block. That setup &mdash; SDAR-1.7B drafter + frozen 30B with the trained 128M judge, no pool &mdash; is the <b>base</b> on this page.</li>'
      f'<li><b>Phase 2</b> (<a href="{P2_URL}">report</a>) swapped the drafter (CoDA-1.7B, Qwen3-0.6B) with the same judge. On the Phase 1 machine the base ran 1.22&times; faster than the plain 30B in tokens per second (all 7 benchmarks); '
      'neither swap beat SDAR-1.7B (0.89&ndash;1.03&times; the base).</li>'
      '<li><b>Phase 3</b> (this page) keeps the base and adds <b>re-drafting</b>: after every judge pass (rd1) or every second pass (rd2) the drafter re-fills the slots the judge rejected, seeing only the prompt and committed tokens.</li>'
      '<li><b>Why the base was decoded again here.</b> The first Phase 3 run used a rented 4-GPU host shared with another tenant (gpu_frac 0.5), so its times are void. '
      'On this unshared 8-GPU machine, identical base work takes 2.2&times; as long as on the Phase 1 machine: same GPU model, but batch-1 decoding waits on one CPU core and this CPU is slower. '
      'A speedup only means something when both setups are timed on the same machine, so the base was re-decoded here with the same code and flags (minus the re-draft). '
      'Its outputs match the Phase 1 records token for token, so it is the same base, now timed next to rd1 and rd2.</li>'
      '<li><b>Which baseline the speedups use.</b> Tables 1 and 3 are against the <b>trained setup</b> (SDAR-1.7B drafter + the 30B with the trained 128M judge), measured on this machine. '
      'Table 2 is against the plain 30B MoE (&ldquo;stock&rdquo;): stock was not decoded here (13 GSM8K items only), so its speedups multiply two numbers, '
      'Phase 2&rsquo;s base over stock (Phase 1 machine) and rd over base (this machine).</li></ul></div>')
    p2b = R.get("phase2_drafters") or {}
    if p2b:
        a('<p class="caption"><b>Table 0.</b> Earlier phases at a glance, from the Phase 1 and Phase 2 records. Accuracy, passes and expert loads do not depend on the machine. '
          '&ldquo;Base identical&rdquo; = items whose base output on this machine equals the Phase 1 record token for token. Speedups over stock are in Table 2.</p>')
        a('<div class="tbl"><table style="min-width:880px"><thead><tr><th class="l">Benchmark</th><th>Stock accuracy</th><th>Base accuracy</th><th>Stock passes / block</th><th>Base passes / block</th>'
          '<th>Stock expert loads / token</th><th>Base expert loads / token</th><th>Base identical here</th></tr></thead><tbody>')
        for s_, name in ORDER:
            q = (p2b.get(s_) or {}).get("setups") or {}
            if not q:
                continue
            st, bs = q["stock"], q["S0_nopool"]
            ident = ((B.get(s_) or {}).get("base_here") or {}).get("identical_to_phase1")
            idt = f"{ident[0]}/{ident[1]}" if ident else "·"
            a(f'<tr><td class="l">{name}</td><td>{f(STOCK_ACC.get(s_), 1)}</td><td>{f(bs["acc"], 1)}</td><td>{f(st["passes_blk"])}</td><td>{f(bs["passes_blk"])}</td>'
              f'<td>{f(st["exp_tok"], 0)}</td><td>{f(bs["exp_tok"], 0)}</td><td>{idt}</td></tr>')
        a("</tbody></table></div>")
    a("</section>")
    # ---------------------------------------------------------------- machine banner
    mv = R.get("machine_vs_phase1")
    if mv:
        a(f'<p class="banner"><b>This machine is {mv["time_ratio"]:.2f}&times; slower than the Phase 1 machine on identical work.</b> '
          f'The base pipeline on {mv["items"]} HumanEval items gave token-identical outputs ({mv["token_identical"]}/{mv["items"]}) in {mv["seconds_here"]:.1f} s here '
          f'against {mv["seconds_phase1"]:.1f} s there. The GPUs are the same model; batch-1 decoding waits on one CPU core (the GPU is busy about 27% of the time), '
          'and this CPU is slower. Speedups are therefore taken against the base pipeline run on this machine.</p>')
    # ---------------------------------------------------------------- 1. summary
    a('<section><h2><span class="no">1</span>Summary</h2>')
    a('<p class="caption"><b>Table 1.</b> Accuracy per benchmark and setup, with &Delta; = rd minus base on the same items (95% bootstrap interval, 2,000 resamples). '
      'Speedup = rd tokens per second &divide; base tokens per second, both decoded on this machine on the same items: a direct measurement, nothing chained. '
      'The switch at the top picks wall-clock or GPU-active seconds. rd1 and rd2 compare directly with each other the same way.</p>')
    a(f'<div class="tbl"><table class="sumtbl" style="min-width:820px"><thead><tr><th class="l">Benchmark</th><th>Base accuracy</th>'
      f'<th>rd1 accuracy (&Delta;)</th><th>rd2 accuracy (&Delta;)</th><th>rd1 speedup over base{MODE_HEAD}</th><th>rd2 speedup over base{MODE_HEAD}</th></tr></thead><tbody>')
    for s, name in ORDER:
        e = B.get(s)
        if not e:
            continue
        b1 = e.get("base_phase1_all") or {}
        bh = (e.get("base_here") or {})
        base_acc = f'{f(bh["all"]["acc"], 1)}%' if bh.get("complete") else (f'{f(b1.get("acc"), 1)}%<span class="tag">Phase 1 records</span>' if b1 else "·")
        cells = []
        for arm in ("rd1", "rd2"):
            x = (e.get("arms") or {}).get(arm)
            if not x:
                cells.append(('<td class="na">queued</td>', '<td class="na">·</td>'))
                continue
            m = x["rd"]; v = x.get("vs_base_here") if bh.get("complete") else None
            v1 = x.get("vs_phase1_base") or {}
            d = (v or v1).get("delta_acc")
            part = "" if x.get("complete") else f'<span class="tag">running, {x["items_done"]}/{e["items_total"]}</span>'
            acc = f'<td class="acc">{f(m["acc"], 1)}% <span class="ci">({sd(d[0])} [{sd(d[1])}, {sd(d[2])}])</span>{part}</td>' if d else f'<td class="acc">{f(m["acc"], 1)}%{part}</td>'
            if v and x.get("complete"):
                spd = f'<td class="acc arm-{arm}">{sp(v["speedup_tok_s"], v.get("speedup_gpu_tok_s"))}</td>'
            else:
                spd = '<td class="na">after the base run</td>'
            cells.append((acc, spd))
        a(f'<tr><td class="l bench">{name}<span class="n" style="display:block;font-weight:400;font-size:0.78rem;color:var(--muted)">n = {e["items_total"]}</span></td>'
          f'<td>{base_acc}</td>{cells[0][0]}{cells[1][0]}{cells[0][1]}{cells[1][1]}</tr>')
    pool = pooled_here(B)
    if pool:
        a(f'<tr class="ref" style="font-style:normal"><td class="l bench">All 7 benchmarks<span style="display:block;font-weight:400;font-size:0.78rem;color:var(--muted)">'
          f'{pool["n"]:,} items, pooled</span></td><td>{f(pool["base_acc"], 1)}%</td>'
          + "".join(f'<td class="acc">{f(pool[arm]["acc"], 1)}%</td>' for arm in ("rd1", "rd2"))
          + "".join(f'<td class="acc arm-{arm}">{sp(pool[arm]["wall"], pool[arm]["gpu"])}</td>' for arm in ("rd1", "rd2")) + "</tr>")
    a("</tbody></table></div>")
    a('<p class="caption">Pooled = all tokens &divide; all seconds over the 2,250 items, so benchmarks with long answers weigh more. '
      'Total-time ratios on the same items (which also count how long the answers are) are in Table 6.</p></section>')
    # ---------------------------------------------------------------- 2. against stock
    vs = R.get("vs_stock") or {}
    if vs.get("benchmarks"):
        a('<section><h2><span class="no">2</span>Against the plain 30B MoE (stock)</h2>')
        a('<p class="caption"><b>Table 2.</b> rd1, rd2 and the base against the vanilla SDAR-30B-A3B (stock: no drafter, no judge). '
          'Stock was never decoded on this machine except 13 GSM8K items, so the speedup over stock multiplies two numbers: '
          '<b>A</b> = base over stock on the Phase 1 machine (Phase 2 page, Table 1) and <b>B</b> = rd over base, measured on this machine. '
          '<b>rd over stock = A &times; B.</b> Accuracy (paired with the stock records, 95% bootstrap interval) and 30B work (stock&rsquo;s 30B expert loads per token &divide; the setup&rsquo;s) '
          'are counts and need no chaining.</p>')
        a(f'<div class="tbl"><table style="min-width:1280px"><thead><tr><th class="l">Benchmark</th><th>Stock accuracy</th><th>rd1 accuracy (&Delta; vs stock)</th><th>rd2 accuracy (&Delta; vs stock)</th>'
          '<th>A: base over stock<span class="mode">Phase 1 machine, wall-clock</span></th>'
          f'<th>B: rd1 over base{MODE_HEAD}</th><th>B: rd2 over base{MODE_HEAD}</th>'
          f'<th>rd1 over stock, A&times;B{MODE_HEAD}</th><th>rd2 over stock, A&times;B{MODE_HEAD}</th>'
          '<th>Less 30B work than stock, rd1</th><th>Less 30B work than stock, rd2</th></tr></thead><tbody>')
        def acc_cell(y):
            d = y.get("delta_acc_vs_stock")
            return f'{f(y.get("acc"), 1)}% <span class="ci">({sd(d[0])} [{sd(d[1])}, {sd(d[2])}])</span>' if d else f'{f(y.get("acc"), 1)}%'
        for s_, name in ORDER:
            x = vs["benchmarks"].get(s_)
            if not x:
                continue
            A = x["base_over_stock_phase2"]
            src = '<span class="tag m">measured</span>' if s_ in P2_STOCK_MEASURED else '<span class="tag">stock computed</span>'
            cells = []
            for arm in ("rd1", "rd2"):
                v = (((B.get(s_) or {}).get("arms") or {}).get(arm) or {}).get("vs_base_here") or {}
                bw, bg = v.get("speedup_tok_s"), v.get("speedup_gpu_tok_s")
                cells.append((f'<td>{sp(bw, bg)}</td>',
                              f'<td class="acc arm-{arm}">{sp(A * bw if bw else None, A * bg if bg else None)}</td>'))
            r1, r2 = x["arms"].get("rd1", {}), x["arms"].get("rd2", {})
            a(f'<tr><td class="l">{name}{AST}</td><td>{f(x["stock_acc"], 1)}%</td><td>{acc_cell(r1)}</td><td>{acc_cell(r2)}</td>'
              f'<td>{f(A, 3)}&times;{src}</td>{cells[0][0]}{cells[1][0]}{cells[0][1]}{cells[1][1]}'
              f'<td>{f(r1.get("work_ratio_loads"), 2)}&times;</td><td>{f(r2.get("work_ratio_loads"), 2)}&times;</td></tr>')
        al = vs.get("all") or {}
        pool = pooled_here(B)
        if al and pool:
            A = al["base_over_stock_phase2"]
            r1, r2 = al["arms"].get("rd1", {}), al["arms"].get("rd2", {})
            a(f'<tr class="ref" style="font-style:normal"><td class="l"><b>All 7 benchmarks</b>{AST}<span style="display:block;font-size:0.78rem;color:var(--muted)">2,250 items, pooled</span></td>'
              f'<td>{f(al["stock_acc"], 1)}%</td><td>{f(r1.get("acc"), 1)}%</td><td>{f(r2.get("acc"), 1)}%</td>'
              f'<td>{f(A, 3)}&times;<span class="tag">mostly computed</span></td>'
              f'<td>{sp(pool["rd1"]["wall"], pool["rd1"]["gpu"])}</td><td>{sp(pool["rd2"]["wall"], pool["rd2"]["gpu"])}</td>'
              f'<td class="acc arm-rd1">{sp(A * pool["rd1"]["wall"], A * pool["rd1"]["gpu"])}</td><td class="acc arm-rd2">{sp(A * pool["rd2"]["wall"], A * pool["rd2"]["gpu"])}</td>'
              f'<td>{f(r1.get("work_ratio_loads"), 2)}&times;</td><td>{f(r2.get("work_ratio_loads"), 2)}&times;</td></tr>')
        ck = vs.get("same_machine_check_gsm8k")
        if ck:
            a(f'<tr class="ref"><td class="l">GSM8K, {ck["items"]} items<span style="display:block;font-size:0.78rem">stock timed on this machine</span></td><td>·</td><td>·</td><td>·</td>'
              f'<td><span class="v-wall">{f(ck["base_over_stock"], 3)}&times;</span><span class="v-gpu">·</span><span class="tag m">measured here</span></td>'
              f'<td><span class="v-wall">{f(ck["rd1_over_stock"] / ck["base_over_stock"], 3)}&times;</span><span class="v-gpu">·</span></td>'
              f'<td><span class="v-wall">{f(ck["rd2_over_stock"] / ck["base_over_stock"], 3)}&times;</span><span class="v-gpu">·</span></td>'
              f'<td><span class="v-wall">{f(ck["rd1_over_stock"], 3)}&times;</span><span class="v-gpu">·</span></td>'
              f'<td><span class="v-wall">{f(ck["rd2_over_stock"], 3)}&times;</span><span class="v-gpu">·</span></td><td>·</td><td>·</td></tr>')
        a("</tbody></table></div>")
        p2gsm = (vs["benchmarks"].get("gsm8k500") or {}).get("base_over_stock_phase2")
        a('<div class="prose"><p><b>Which machine each number comes from.</b></p><ul>'
          '<li><b>A, base over stock</b>, is the Phase 2 page&rsquo;s figure for the Phase 1 machine (RTX PRO 6000 Server, 29&ndash;30 Sep). Base was timed there on every benchmark. '
          'Stock was timed there only on <b>IFEval-541</b> (its time per expert load, 88.6 &micro;s, matches the base run&rsquo;s 88.2 &micro;s). '
          'On the other six, stock&rsquo;s time is <b>computed</b>: its 30B expert loads per token (a count, the same on any machine) &times; the Phase 1 machine&rsquo;s time per expert load '
          '(0.085&ndash;0.089 ms, from the base rows) plus drafter time. The Phase 2 page checked this on IFEval: 139.9 ms per token computed against 140.3 measured. '
          'The raw stock times for those six come from a third machine (Workstation Edition, 5&ndash;8 Sep, 0.17&ndash;0.18 ms per expert load) and are not used.</li>'
          '<li><b>B, rd over base</b>, is measured on this machine (RTX PRO 6000 Server, Xeon Gold 6430), 2.17&times; slower than the Phase 1 machine on identical base work.</li>'
          '<li><b>A &times; B</b> assumes base over stock is the same on both machines. The 13 stock items timed here test that on GSM8K: base over stock is '
          f'<b>{f((ck or {}).get("base_over_stock"), 3)}&times;</b> measured here against <b>{f(p2gsm, 3)}&times;</b> for A, so the chained GSM8K figures may be about '
          f'{f(100 * (p2gsm / ck["base_over_stock"] - 1) if ck and p2gsm else None, 0)}% high. Thirteen items is a small sample.</li>'
          '<li><b>GPU-only</b> (switch at the top): A has no GPU-time version, because no GPU-active time was recorded for stock on any machine. '
          'In GPU-only mode A &times; B uses the wall-clock A with the GPU-only B; those rows carry <span class="ast">*</span>.</li></ul></div>')
        a("</section>")
    # ---------------------------------------------------------------- 3. phase 2 comparison
    p2 = R.get("phase2_drafters") or {}
    if p2:
        a('<section><h2><span class="no">3</span>Re-drafting vs swapping the drafter</h2>')
        a('<p class="caption">rd1 and rd2 keep the base drafter, <b>SDAR-1.7B-Chat-b32</b>, and add re-drafting of the slots the judge rejects. '
          'Phase 2 tried the other lever: it swapped the drafter (CoDA-1.7B, Qwen3-0.6B) and did not re-draft. Same frozen 30B, same 128M judge, no expert pool throughout.</p>')
        a('<p class="caption"><b>Table 3.</b> Speedup over the SDAR-1.7B + judge base. rd1 and rd2: measured on this machine against the base run here. '
          'CoDA and Qwen3: from the Phase 2 page, tokens per second on the Phase 1 machine, where only the base was timed directly and the other setups were computed from their expert loads; '
          'they have no GPU-time record, so in GPU-only mode they keep their wall-clock values and the row carries <span class="ast">*</span>. Read them as indicative next to the rd columns.</p>')
        a(f'<div class="tbl"><table style="min-width:760px"><thead><tr><th class="l">Benchmark</th><th>CoDA-1.7B<span class="mode">Phase 2, computed</span></th>'
          f'<th>Qwen3-0.6B<span class="mode">Phase 2, computed</span></th><th>rd1{MODE_HEAD}</th><th>rd2{MODE_HEAD}</th></tr></thead><tbody>')
        for s_, name in ORDER:
            q = (p2.get(s_) or {}).get("setups") or {}
            e = B.get(s_) or {}
            if not q:
                continue
            cells = []
            for arm in ("rd1", "rd2"):
                v = ((e.get("arms") or {}).get(arm) or {}).get("vs_base_here")
                cells.append(f'<td class="acc arm-{arm}">{sp(v["speedup_tok_s"], v.get("speedup_gpu_tok_s"))}</td>' if v else '<td class="na">after the base run</td>')
            a(f'<tr><td class="l">{name}{AST}</td><td>{f(q["F2_nopool"]["speedup_vs_base"], 3)}&times;</td><td>{f(q["S1_nopool"]["speedup_vs_base"], 3)}&times;</td>{"".join(cells)}</tr>')
        a("</tbody></table></div>")
        a('<p class="caption"><b>Table 4.</b> Work and accuracy, which do not depend on the machine and compare exactly: 30B expert loads per token (unique (layer, expert) loads summed over the 48 layers), '
          'judge passes per block and accuracy, for the plain 30B (stock, no drafter), the base, the two Phase 2 drafters and the two re-draft setups. Stock and base figures are the Phase 1 / Phase 2 records.</p>')
        a('<div class="tbl"><table style="min-width:1320px"><thead><tr><th class="l" rowspan="2">Benchmark</th>'
          '<th colspan="6" style="text-align:center">Expert loads / token</th><th colspan="6" style="text-align:center">Passes / block</th><th colspan="6" style="text-align:center">Accuracy %</th></tr><tr>'
          + "".join(f"<th>{h}</th>" for h in ["stock", "base", "CoDA", "Qwen3", "rd1", "rd2"] * 3) + '</tr></thead><tbody>')
        for s_, name in ORDER:
            q = (p2.get(s_) or {}).get("setups") or {}
            e = B.get(s_) or {}
            if not q:
                continue
            r1 = (((e.get("arms") or {}).get("rd1")) or {}).get("rd") or {}
            r2 = (((e.get("arms") or {}).get("rd2")) or {}).get("rd") or {}
            row = []
            for key_p2, key_rd, p in (("exp_tok", "loads_tok", 0), ("passes_blk", "passes_blk", 2), ("acc", "acc", 1)):
                stock_v = STOCK_ACC.get(s_) if key_p2 == "acc" else q["stock"][key_p2]
                row += [f'<span class="na">{f(stock_v, p)}</span>', f(q["S0_nopool"][key_p2], p), f(q["F2_nopool"][key_p2], p), f(q["S1_nopool"][key_p2], p)]
                row += [f'<span class="arm-rd1">{f(r1.get(key_rd), p)}</span>', f'<span class="arm-rd2">{f(r2.get(key_rd), p)}</span>']
            a(f'<tr><td class="l">{name}</td>' + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        a("</tbody></table></div></section>")
    # ---------------------------------------------------------------- 4. full results (the old run list is merged in: Wall + Finished columns)
    a('<section><h2><span class="no">4</span>Full results</h2>')
    a('<p class="caption"><b>Table 5.</b> Every measured quantity per benchmark and setup (a running setup shows its partial count). Wall is a run&rsquo;s start-to-finish time on 8 GPUs, model load included; Finished is IST. Passes per block, expert counts and drafter forwards '
      'do not depend on the machine and compare directly with the Phase 1 records. Experts per 30B forward and per judge pass are unique (layer, expert) loads summed over the 48 layers. '
      'GPU-active time is read from the GPU&rsquo;s own counters (NVML GPM) just before and after each item, outside the timed span; GPU-idle time is the rest of the decode (CPU work, kernel launches, synchronisation).</p>')
    a('<div class="tbl"><table style="min-width:1300px"><thead><tr><th class="l">Benchmark</th><th class="l">Setup</th><th>Items</th><th>Accuracy %</th>'
      '<th>Passes / block</th><th>Expert loads / token</th><th>Experts / 30B forward</th><th>Experts / judge pass</th>'
      '<th>Drafter fwd / token</th><th>Tokens / s</th><th>ms / token</th><th>Total generation time</th><th>GPU-active time</th><th>GPU active %</th><th>GPU-only tokens / s</th><th>Wall (8 GPUs)</th><th>Finished (IST)</th><th>judge-diag share</th></tr></thead><tbody>')
    for s, name in ORDER:
        e = B.get(s)
        if not e:
            continue
        rows = []
        b = e.get("base_phase1_all")
        if b:
            rows.append(("base, Phase 1 records", b, None, True, None))
        bh = e.get("base_here") or {}
        if bh.get("all"):
            rows.append(("base, this machine", bh["all"], bh, False, None))
        for arm, x in (e.get("arms") or {}).items():
            rows.append((arm, x["rd"], x, False, x.get("judge_diag_cost_share_pct")))
        for k, (lab, m, x, ref, dshare) in enumerate(rows):
            first = f'<td class="l bench" rowspan="{len(rows)}">{name}</td>' if k == 0 else ""
            cnt = f'{m["n"]}' + ("" if (x is None or x.get("complete")) else f'<span class="tag">running, of {e["items_total"]}</span>')
            w = (x or {}).get("wall") or {}
            P1 = '<span class="tag">Phase 1 machine</span>' if ref else ""
            gpu = (f(m["gpu_active_s"] / 3600, 2) + " h") if m.get("gpu_active_s") is not None else '<span class="na">not recorded</span>'
            cls = ' class="ref"' if ref else ""
            armcls = f' arm-{lab}' if lab in ("rd1", "rd2") else ""
            a(f'<tr{cls}>{first}<td class="l{armcls}">{lab}</td><td>{cnt}</td><td>{f(m["acc"], 1)}</td><td>{f(m["passes_blk"])}</td><td>{f(m["loads_tok"], 0)}</td>'
              f'<td>{f(m["experts_per_30b_forward"], 0)}</td><td>{f(m["experts_per_judge_pass"], 0)}</td><td>{f(m["drafter_fwd_tok"])}</td>'
              f'<td>{f(m["tok_s"])}{P1}</td><td>{f(m["ms_tok"], 1)}</td><td>{f(m["seconds"] / 3600, 2)} h</td><td>{gpu}</td>'
              f'<td>{f(m.get("gpu_share_pct"), 1, "%")}</td><td>{f(m.get("gpu_tok_s"))}</td><td>{f(w.get("wall_min"), 1, " min")}</td>'
              f'<td>{utc_to_ist(w.get("end_utc")) if w.get("end_utc") else "·"}</td><td>{f(dshare, 3, "%")}</td></tr>')
    a("</tbody></table></div>")
    a('<p class="caption"><b>Table 6.</b> Speed ratios against the base pipeline on the same items, two ways. <b>Tokens per second</b> (as in Tables 1&ndash;3) measures decoding rate; '
      '<b>total time</b> = base seconds &divide; rd seconds, which also counts how long the answers are (rd1 writes 21% more tokens than base on IFEval and 25% more on HumanEval, '
      'so its total-time ratio there is below 1 although its rate is higher). Against the base on this machine these are the speedups of re-drafting; '
      'against the Phase 1 records they mix in the machine difference (H, the ratio of identical prompt-prefill work, measures it) and are shown only for reference; '
      'the Phase 1 records have no GPU-active time.</p>')
    a(f'<div class="tbl"><table style="min-width:860px"><thead><tr><th class="l">Benchmark</th><th class="l">Setup</th><th class="l">Base reference</th><th>Paired items</th>'
      f'<th>rd tokens &divide; base tokens</th><th>Speedup, tokens / s{MODE_HEAD}</th><th>Speedup, total time{MODE_HEAD}</th><th>H, 30B prefill</th></tr></thead><tbody>')
    for s, name in ORDER:
        for arm, x in ((B.get(s) or {}).get("arms") or {}).items():
            for key, lab in (("vs_base_here", "base on this machine"), ("vs_phase1_base", "Phase 1 records (other machine)")):
                v = x.get(key)
                if v:
                    tr = v["rd"]["tokens"] / v["base"]["tokens"] if v.get("rd") and v.get("base") else None
                    a(f'<tr><td class="l">{name}</td><td class="l arm-{arm}">{arm}</td><td class="l">{lab}</td><td>{v["paired_items"]}</td><td>{f(tr, 3)}</td>'
                      f'<td>{sp(v["speedup_tok_s"], v.get("speedup_gpu_tok_s"))}</td><td>{sp(v["speedup_time"], v.get("speedup_gpu_time"))}</td><td>{f(v.get("H_30b_prefill"), 3)}</td></tr>')
    a("</tbody></table></div></section>")
    # ---------------------------------------------------------------- 4. machine
    a('<section><h2><span class="no">5</span>Machine and measurement checks</h2>')
    a('<div class="prose"><ul><li><b>Not shared:</b> <code>gpu_frac</code> 1.0; all 8 GPUs and 128 of 128 CPU threads rented; before the run no other GPU process, 0 MiB and 0% utilisation on every GPU for 60 s; MIG off, no vGPU; idle load 0.03.</li>'
      '<li><b>Hardware:</b> 8&times; NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.84, 600 W limit); 2&times; Intel Xeon Gold 6430, busy cores at 2.6 GHz.</li>'
      f'<li><b>--judge-diag</b> (kept on, as specified): measured at {f(R.get("judge_diag_cost_us_per_pass"), 1)} &micro;s per judge pass, under 0.1% of decode time in every run (Table 5).</li>'
      '<li><b>Correctness:</b> the base pipeline reproduced the Phase 1 records token for token in the setup check (16/16).</li></ul></div>')
    a('<p class="caption"><b>Table 7.</b> During each run: cgroup CPU-quota throttling, the largest GPU clock-slowdown time on any GPU (driver counters read before and after), and the background-job check: '
      'judge-pass times of items that overlapped a backup or page build, against the other items, each normalised by the same item&rsquo;s Phase 1 record.</p>')
    a('<div class="tbl"><table style="min-width:760px"><thead><tr><th class="l">Benchmark</th><th class="l">Setup</th><th>CPU throttled periods</th><th>GPU slowdown, max s</th><th>Items overlapping a job</th><th>Overlapping vs other items</th></tr></thead><tbody>')
    for s, name in ORDER:
        for arm, x in ((B.get(s) or {}).get("arms") or {}).items():
            h = x.get("host_during_run") or {}
            o = (x.get("vs_phase1_base") or {}).get("background_overlap") or {}
            tot = (o.get("items_overlapping") or 0) + (o.get("items_other") or 0)
            dp = o.get("difference_pct")
            a(f'<tr><td class="l">{name}</td><td class="l arm-{arm}">{arm}</td><td>{h.get("cpu_nr_throttled", "·")}</td><td>{f(h.get("gpu_throttle_seconds_max"), 1)}</td>'
              f'<td>{o.get("items_overlapping", "·")} of {tot}</td><td>{(sd(dp, 2) + "%") if dp is not None else "·"}</td></tr>')
    a("</tbody></table></div></section>")
    # ---------------------------------------------------------------- 5. formulas
    a('<section><h2><span class="no">6</span>How each number is measured</h2><div class="prose"><ul>'
      '<li><b>Time per item:</b> <code>torch.cuda.synchronize()</code> + <code>time.perf_counter()</code> around the whole decode (prompt prefill, every judge, drafter, re-draft and KV-flush forward), as in every earlier phase; tokenisation, scoring and record writing are outside it.</li>'
      '<li><b>Tokens per second</b> = &Sigma; tokens &divide; &Sigma; seconds over items, never a mean of per-item rates. <b>Total generation time</b> = &Sigma; seconds. <b>Wall</b> = a run&rsquo;s start to finish on 8 GPUs.</li>'
      '<li><b>GPU-active time</b> = NVML GPM <code>GRAPHICS_UTIL</code> between two counter samples taken just before and after each item (outside the timed span) &times; that interval. <b>GPU-only tokens per second</b> = &Sigma; tokens &divide; &Sigma; GPU-active seconds.</li>'
      '<li><b>Speedup over base</b> (Tables 1, 3) = rd tokens per second &divide; base tokens per second on the same items, both on this machine: wall-clock seconds, or GPU-active seconds with the switch on GPU-only. '
      '<b>Speedup over stock</b> (Table 2) = A &times; B, A = base over stock on the Phase 1 machine (Phase 2 page), B = rd over base here. '
      'Pooled rows use &Sigma; tokens &divide; &Sigma; seconds over all 2,250 items.</li>'
      '<li><b>Passes per block</b> = &Sigma; (30B forwards &minus; prefill &minus; KV flushes) &divide; &Sigma; blocks. <b>Expert loads per token</b> = &Sigma; unique (layer, expert) loads &divide; &Sigma; tokens. <b>Experts per 30B forward</b> = &Sigma; unique loads &divide; &Sigma; 30B forwards; <b>per judge pass</b> = &Sigma; judge-pass loads &divide; &Sigma; judge passes.</li>'
      '<li><b>Settings:</b> base = <code>--drafter d32 --draft s8 --judge full --adapter &lt;128M&gt; --judge-diag</code>; rd1 / rd2 = base + <code>--full-redraft-every 1 | 2</code>. JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc, SDAR-1.7B-Chat-b32 @ 2c83c05, judge LoRA hash 35805974; block 32, threshold 0.95, no expert pool, batch 1, one process per GPU, 8 shards <code>items[k::8]</code>.</li>'
      '</ul></div></section>')
    if os.path.exists(redraft_why.OUT):   # written by code/phase3_8gpu/redraft_why.py from the per-item records
        a(redraft_why.sections_html(sec_no=7, table_no=8, fig_no=1))
    a('<footer>Records, logs and this report: GitHub branch <code>phase3-redraft-8gpu</code> (NoviceCoderInfinity/diffusion-moe-general-expert) and the Hugging Face dataset <code>Arushhh/phase3-redraft</code>, under <code>logs/phase3_8gpu/</code> and <code>results/phase3_8gpu/</code>.</footer>')
    a("</main>")
    a(SCRIPT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    open(tmp, "w").write("\n".join(H) + "\n")
    os.replace(tmp, OUT)
    print(f"[page {now.strftime('%H:%M:%S')} UTC] wrote {OUT}")


if __name__ == "__main__":
    main()
