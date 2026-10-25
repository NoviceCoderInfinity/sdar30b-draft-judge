#!/usr/bin/env python
"""Why re-drafting gains so little, and what feeding the judge's verdict back to the drafter would risk: the last two
sections of the 8-GPU Phase 3 page.

    /venv/p2/bin/python code/phase3_8gpu/redraft_why.py      # recompute results/phase3_8gpu/redraft_why.json (CPU only)

build_page.py renders the sections from that JSON (css() and sections_html(sec_no, table_no)); nothing is recomputed
at page time. Every number comes from the per-item records under logs/phase3_8gpu/decode/{base,rd1,rd2}/, on the items
all three setups decoded (all of them: 2,250 per setup). The SDAR-1.7B tokenizer is loaded only to tell word tokens
from whitespace / punctuation for the repeated-word count.
"""
import glob
import json
import os
import sys

ROOT = os.environ.get("P3G_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEC = os.path.join(ROOT, "logs/phase3_8gpu/decode")
OUT = os.environ.get("P3G_RW_OUT") or os.path.join(ROOT, "results/phase3_8gpu/redraft_why.json")
ORDER = [("ifeval541", "IFEval-541"), ("math500", "MATH-500"), ("gpqa", "GPQA-Diamond"), ("aime", "AIME"),
         ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k500", "GSM8K-500")]
SETUPS = ("base", "rd1", "rd2")
HARD = 10                    # a hard block needs more than this many judge passes
NGRAM = 12                   # loop detector: a token sits in a repeated n-gram if that n-gram appeared earlier
DRAFTER_EXPERT_EQUIV = 1_720_573_440 / 4_718_592   # = 364.64, code/draftalign/compare_table.py
EXAMPLE = ("mbpp", 172, 1)   # benchmark, item idx, block: one of the longest MBPP blocks with identical base / rd1 text


# ============================================================================ compute
def load(setup, bench):
    out = {}
    for fn in glob.glob(f"{DEC}/{setup}/{bench}/*.jsonl"):
        for line in open(fn):
            if line.strip():
                r = json.loads(line)
                out[r["idx"]] = r
    return out


def _word_token_fn():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    snap = glob.glob(os.path.join(ROOT, "hf_cache/hub/models--JetLM--SDAR-1.7B-Chat-b32/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    cache = {}

    def is_word(t):
        if t not in cache:
            s = tok.decode([t])
            cache[t] = s.strip() != "" and any(c.isalnum() for c in s)
        return cache[t]
    return is_word


def _ratio(num, den):
    return num / den if den else None


def setup_stats(rs, is_word):
    g = lambda fn: float(sum(fn(r) for r in rs))
    mk = lambda k: g(lambda r: r["ms_by_kind"].get(k, 0.0)) / 1000
    dk = lambda k: g(lambda r: (r.get("drafter_ms_by_kind") or {}).get(k, 0.0)) / 1000
    sec = g(lambda r: r["seconds"])
    judge, flush, pre, den = mk("judge"), mk("kvflush"), mk("prefill"), mk("denoise")
    dpre, ddraft, dred, dfl = dk("prefill"), dk("draft"), dk("redraft"), dk("flush")
    first = pre + dpre + ddraft + dfl                       # prompt prefill (both models) + first draft + drafter flush
    host = sec - (judge + flush + first + dred)             # host time between forwards (+ stock fall-back, 0 here)
    blocks, tokens, nj = g(lambda r: r["n_blocks"]), g(lambda r: r["n_gen_to_eos"]), g(lambda r: r["n_judge"])
    # pass mechanics
    n_blk = n_pass = 0
    hb = dict(nb=0, p=0, lp=0, c=0, f=0, t=0)
    for r in rs:
        for passes in r["full_passes_per_block"]:
            n_blk += 1; n_pass += len(passes)
            if len(passes) > HARD:
                hb["nb"] += 1; hb["p"] += len(passes)
                for k, p in enumerate(passes):
                    if k:
                        hb["lp"] += 1; hb["c"] += p["commit"]; hb["f"] += int(p["forced"]); hb["t"] += p["tentative"]
    red_slots = g(lambda r: sum(r.get("redraft_slots_per_block") or [0]))
    red_old = g(lambda r: sum(r.get("redraft_eq_old_per_block") or [0]))
    red_judge = g(lambda r: sum(r.get("redraft_eq_judge_per_block") or [0]))
    red_fwd = g(lambda r: sum(r.get("redraft_forwards_per_block") or [0]))
    fh = g(lambda r: r["judge_diag"]["copy_flag"][0]); fn = g(lambda r: r["judge_diag"]["copy_flag"][1])
    mh = g(lambda r: r["judge_diag"]["copy_mask"][0]); mn = g(lambda r: r["judge_diag"]["copy_mask"][1])
    remask = g(lambda r: sum(r["remask_per_block"]))
    commits = g(lambda r: sum(r["committed_judge_per_block"]))
    confirm = g(lambda r: sum(r["full_commit_eq_draft_per_block"]))
    dup = rep = loops = 0
    for r in rs:
        ids = r["gen_ids"][:r["n_gen_to_eos"]]
        dup += sum(1 for x, y in zip(ids, ids[1:]) if x == y and is_word(x))
        seen, rp = set(), 0
        for k in range(len(ids) - NGRAM + 1):
            t = tuple(ids[k:k + NGRAM])
            if t in seen:
                rp += 1
            else:
                seen.add(t)
        rep += rp
        if len(ids) > 2 * NGRAM and rp / (len(ids) - NGRAM + 1) > 0.5:
            loops += 1
    gpu = [r.get("gpu_active_s") for r in rs]
    return dict(
        items=len(rs), seconds=sec, tokens=tokens, blocks=blocks, judge_passes=nj,
        gpu_active_s=(sum(gpu) if all(v is not None for v in gpu) else None),
        hours=dict(judge=judge / 3600, redraft=dred / 3600, kvflush=flush / 3600, first=first / 3600, host=host / 3600, total=sec / 3600),
        ms_blk=dict(judge=1000 * judge / blocks, redraft=1000 * dred / blocks, fixed=1000 * (flush + first) / blocks, total=1000 * sec / blocks),
        passes_blk=nj / blocks,
        hard=dict(blocks=hb["nb"], pass_share=hb["p"] / n_pass, commit=_ratio(hb["c"], hb["lp"]), agree=_ratio(hb["t"], hb["lp"]),
                  forced=_ratio(hb["f"], hb["lp"])),
        redraft=dict(slots=red_slots, fwd_blk=red_fwd / blocks, same=_ratio(red_old, red_slots), judge=_ratio(red_judge, red_slots)),
        copy=dict(draft_hits=fh, draft_n=fn, mask_hits=mh, mask_n=mn, disagree=remask + commits - confirm,
                  draft=_ratio(fh, fn), mask=_ratio(mh, mn), of_disagree=_ratio(fh, remask + commits - confirm)),
        dup=dup, dup_per_k=1000 * dup / tokens, rep_share=rep / tokens, loops=loops,
        capped=sum(int(r["truncated"]) for r in rs), correct=sum(int(r.get("correct") or 0) for r in rs),
        acc=100 * sum(int(r.get("correct") or 0) for r in rs) / len(rs),
        loads_tok=g(lambda r: r["unique_expert_layer_loads"]) / tokens,
        drafter_fwd_tok=g(lambda r: sum(r["drafter_n_by_kind"].values())) / tokens)


def speedups(Db, Dr, items):
    """rd over base on `items`: time and tok/s, wall-clock and GPU-active (None if any item lacks a GPU record)."""
    bt = sum(Db[i]["n_gen_to_eos"] for i in items); rt = sum(Dr[i]["n_gen_to_eos"] for i in items)
    bs = sum(Db[i]["seconds"] for i in items); rs = sum(Dr[i]["seconds"] for i in items)
    bg = [Db[i].get("gpu_active_s") for i in items]; rg = [Dr[i].get("gpu_active_s") for i in items]
    gpu_ok = all(v is not None for v in bg + rg)
    out = dict(n=len(items), time=bs / rs, tok_s=(rt / rs) / (bt / bs), tokens_ratio=rt / bt, sums=dict(bt=bt, rt=rt, bs=bs, rs=rs))
    if gpu_ok:
        out.update(gpu_time=sum(bg) / sum(rg), gpu_tok_s=(rt / sum(rg)) / (bt / sum(bg)))
        out["sums"].update(bg=sum(bg), rg=sum(rg))
    return out


def compute():
    is_word = _word_token_fn()
    res = dict(benchmarks={}, pooled={})
    acc_pairs = {s: dict(all=dict(bt=0, rt=0, bs=0, rs=0, bg=0, rg=0), eos=dict(bt=0, rt=0, bs=0, rs=0, bg=0, rg=0)) for s in ("rd1", "rd2")}
    pooled_rs = {s: [] for s in SETUPS}
    for key, name in ORDER:
        D = {s: load(s, key) for s in SETUPS}
        common = sorted(set(D["base"]) & set(D["rd1"]) & set(D["rd2"]))
        e = dict(name=name, items=len(common), setups={})
        for s in SETUPS:
            rs = [D[s][i] for i in common]
            pooled_rs[s] += rs
            e["setups"][s] = setup_stats(rs, is_word)
        for s in ("rd1", "rd2"):
            eos = [i for i in common if not D["base"][i]["truncated"] and not D[s][i]["truncated"]]
            sa, se = speedups(D["base"], D[s], common), speedups(D["base"], D[s], eos)
            e["setups"][s]["vs_base"] = dict(all=sa, eos=se)
            for lab, sp in (("all", sa), ("eos", se)):
                for k in ("bt", "rt", "bs", "rs", "bg", "rg"):
                    acc_pairs[s][lab][k] += sp["sums"].get(k, float("nan"))
        res["benchmarks"][key] = e
    for s in SETUPS:
        res["pooled"][s] = setup_stats(pooled_rs[s], is_word)
    b = res["pooled"]["base"]
    for s in ("rd1", "rd2"):
        p = res["pooled"][s]
        A, Eo = acc_pairs[s]["all"], acc_pairs[s]["eos"]
        p["vs_base"] = dict(
            time=A["bs"] / A["rs"], gpu_time=A["bg"] / A["rg"],
            tok_s=(A["rt"] / A["rs"]) / (A["bt"] / A["bs"]), gpu_tok_s=(A["rt"] / A["rg"]) / (A["bt"] / A["bg"]),
            eos_tok_s=(Eo["rt"] / Eo["rs"]) / (Eo["bt"] / Eo["bs"]), eos_gpu_tok_s=(Eo["rt"] / Eo["rg"]) / (Eo["bt"] / Eo["bg"]),
            judge_pass_ratio=b["judge_passes"] / p["judge_passes"],
            judge_saved_h=b["hours"]["judge"] - p["hours"]["judge"],
            redraft_paid_h=p["hours"]["redraft"] - b["hours"]["redraft"])
        p["vs_base"]["paid_share"] = p["vs_base"]["redraft_paid_h"] / p["vs_base"]["judge_saved_h"]
    # capped outputs that loop
    for s in SETUPS:
        cap = [r for r in pooled_rs[s] if r["truncated"]]
        res["pooled"][s]["capped_looping"] = sum(1 for r in cap if _loops(r))
    # IFEval tok/s of capped vs EOS-ending items (base and rd1)
    res["ifeval_cap_vs_eos"] = {}
    for s in ("base", "rd1"):
        rs = load(s, "ifeval541").values()
        cap = [r for r in rs if r["truncated"]]; eos = [r for r in rs if not r["truncated"]]
        res["ifeval_cap_vs_eos"][s] = dict(capped_tok_s=sum(r["n_gen_to_eos"] for r in cap) / sum(r["seconds"] for r in cap),
                                           eos_tok_s=sum(r["n_gen_to_eos"] for r in eos) / sum(r["seconds"] for r in eos))
    # the example block
    bk, idx, blk = EXAMPLE
    Bx, Rx = load("base", bk)[idx], load("rd1", bk)[idx]
    assert Bx["gen_ids"] == Rx["gen_ids"], "example item: base and rd1 texts differ"
    res["example"] = dict(
        bench=bk, idx=idx, task=Bx.get("task_id"), block=blk, agree=round(Bx["draft_agreement_per_block"][blk] * 32),
        base=dict(passes=Bx["full_passes_per_block"][blk], s30=_block_ms(Bx, blk) / 1000, sdr=_block_drafter_ms(Bx, blk) / 1000),
        rd1=dict(passes=Rx["full_passes_per_block"][blk], s30=_block_ms(Rx, blk) / 1000, sdr=_block_drafter_ms(Rx, blk) / 1000,
                 rounds=Rx["redraft_rounds_per_block"][blk], slots=Rx["redraft_slots_per_block"][blk],
                 same=Rx["redraft_eq_old_per_block"][blk], judge=Rx["redraft_eq_judge_per_block"][blk],
                 fwd=Rx["redraft_forwards_per_block"][blk]))
    # Phase 2 page's metric (expert loads per token vs stock; drafter forwards charged at 364.64)
    m2 = json.load(open(os.path.join(ROOT, "docs/phase2_artifact/metrics.json")))
    res["phase2_metric"] = {}
    for key, _ in ORDER:
        st = m2[key]["stock"]["exp_tok"]
        row = dict(stock_loads=st)
        for s in SETUPS:
            x = res["benchmarks"][key]["setups"][s]
            row[s] = dict(loads=st / x["loads_tok"], charged=st / (x["loads_tok"] + x["drafter_fwd_tok"] * DRAFTER_EXPERT_EQUIV))
        nopool = [m2[key][a]["sp_judge"] for a in ("S0_nopool", "F2_nopool", "S1_nopool") if a in m2[key]]
        row["best_nopool_loads"] = max(nopool)
        if "S0" in m2[key]:
            row["pooled_arm"] = dict(loads=m2[key]["S0"]["sp_judge"], charged=m2[key]["S0"]["sp_full"], acc=m2[key]["S0"]["acc"])
        res["phase2_metric"][key] = row
    # machine check: the frozen cost models
    pj = json.load(open(os.path.join(ROOT, "results/phase3_8gpu/projection.json")))
    proj = [pj["sets"][k]["arms"][a]["proj_speedup_p1_time"] for k, _ in ORDER for a in ("rd1", "rd2")]
    res["machine"] = dict(
        judge_ms_here=pj["model_here_rd_only"]["kinds"]["judge"]["mean_ms"], judge_ms_p1=pj["model_phase1"]["kinds"]["judge"]["mean_ms"],
        drafter_ms_here=pj["model_here_rd_only"]["drafter_ms"]["redraft"], drafter_ms_p1=pj["model_phase1"]["drafter_ms"]["redraft"],
        proj_p1_min=min(proj), proj_p1_max=max(proj), projection_utc=pj.get("generated_utc"))
    # where stock's timings come from: the GPU and dates of the stock records Table 2 uses (same files as report_8gpu.py),
    # and of the only stock items decoded on this machine (the setup check)
    def gpu_dates(pattern):
        gpus, days, n = {}, [], 0
        for fn in glob.glob(os.path.join(ROOT, pattern)):
            for line in open(fn):
                if line.strip():
                    r = json.loads(line)
                    n += 1
                    gpus[r.get("gpu")] = gpus.get(r.get("gpu"), 0) + 1
                    if r.get("utc"):
                        days.append(r["utc"][:10])
        return dict(items=n, gpus=gpus, first=min(days) if days else None, last=max(days) if days else None)
    prefix = {"ifeval541": "ifeval", "gsm8k500": "gsm8k"}
    res["stock_sources"] = {k: gpu_dates(f"logs/phase2/ref/stock_flat/{prefix.get(k, k)}_B32_t95_stock.{k}.shard*.jsonl") for k, _ in ORDER}
    res["stock_sources"]["setup_check_here"] = gpu_dates("logs/phase3_8gpu/setup/check_stock/*.jsonl")
    res["n_items"] = sum(e["items"] for e in res["benchmarks"].values())
    return res


def _loops(r):
    ids = r["gen_ids"][:r["n_gen_to_eos"]]
    seen, rp = set(), 0
    for k in range(len(ids) - NGRAM + 1):
        t = tuple(ids[k:k + NGRAM])
        if t in seen:
            rp += 1
        else:
            seen.add(t)
    return len(ids) > 2 * NGRAM and rp / (len(ids) - NGRAM + 1) > 0.5


def _block_ms(r, k):
    """30B time of block k: its judge passes + its KV flush (forwards are logged in order, prefill first)."""
    fw, i = r["forwards"], 1
    for bi, passes in enumerate(r["full_passes_per_block"]):
        n = len(passes) + 1
        if bi == k:
            return sum(x[1] for x in fw[i:i + n])
        i += n


def _block_drafter_ms(r, k):
    """Drafter time of block k: first draft + re-drafts + its flush (prefill first)."""
    fw, i = r["drafter_forwards"], 1
    for bi in range(r["n_blocks"]):
        n = r["drafter_forwards_per_block"][bi] + r["redraft_forwards_per_block"][bi] + 1
        if bi == k:
            return sum(x[1] for x in fw[i:i + n])
        i += n


# ============================================================================ render
def css():
    return """
/* sections 8-9 (redraft_why.py): figure tokens, validated blue / orange pair + a neutral */
:root { --rw-judge: #2a78d6; --rw-redraft: #eb6834; --rw-fixed: #b9bccb; --rw-soft: #b7d3f6; --rw-onjudge: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --rw-judge: #3987e5; --rw-redraft: #d95926; --rw-fixed: #4f5268; --rw-soft: #24497a; --rw-onjudge: #0d0e16; }
}
:root[data-theme="dark"] { --rw-judge: #3987e5; --rw-redraft: #d95926; --rw-fixed: #4f5268; --rw-soft: #24497a; --rw-onjudge: #0d0e16; }
.rw-fig { background: var(--card); border: 1px solid var(--hair); border-radius: 6px; padding: 16px; display: grid; gap: 12px; min-width: 0; }
.rw-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0; padding: 0; list-style: none; font-family: var(--font-sans); font-size: 0.8rem; color: var(--ink-2); }
.rw-legend li { display: inline-flex; align-items: center; gap: 7px; }
.rw-sw { width: 12px; height: 12px; border-radius: 2px; flex: none; display: inline-block; }
.rw-svg { overflow-x: auto; }
.rw-svg svg { display: block; min-width: 560px; width: 100%; height: auto; font-family: var(--font-sans); }
.rw-strips { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; align-items: start; }
@media (max-width: 820px) { .rw-strips { grid-template-columns: minmax(0, 1fr); } }
.rw-strip { display: grid; gap: 8px; min-width: 0; }
.rw-strip h3 { margin: 0; font-family: var(--font-sans); font-size: 0.9rem; font-weight: 600; color: var(--ink); }
.rw-strip h3 span { font-weight: 400; color: var(--muted); }
.rw-rows { display: grid; gap: 3px; }
.rw-row { display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) 4.6rem; gap: 8px; align-items: center; font-family: var(--font-mono); font-size: 0.66rem; color: var(--muted); }
.rw-row .pn { text-align: right; font-variant-numeric: tabular-nums; }
.rw-row .gn { color: var(--ink); white-space: nowrap; }
.rw-cells { display: grid; grid-template-columns: repeat(32, minmax(0, 1fr)); gap: 2px; }
.rw-c { aspect-ratio: 1; border-radius: 1.5px; min-width: 0; }
.rw-c.s { background: var(--rw-fixed); } .rw-c.j { background: var(--rw-judge); } .rw-c.a { background: var(--rw-soft); }
.rw-c.h { box-shadow: inset 0 0 0 1px var(--muted); } .rw-c.r { background: var(--rw-redraft); }
.rw-note { font-family: var(--font-sans); font-size: 0.8rem; color: var(--ink-2); }
table.rw-rules td, table.rw-rules th { white-space: normal; text-align: left; vertical-align: top; line-height: 1.45; }
table.rw-rules th { text-transform: none; letter-spacing: 0; font-size: 0.8rem; color: var(--ink); font-weight: 600; }
table.rw-rules thead th { font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
"""


def _f(v, p=2, suf=""):
    return "·" if v is None else f"{v:,.{p}f}{suf}"


def _pc(v, p=1):
    return "·" if v is None else f"{100 * v:.{p}f}%"


def _sp(w, g, p=3):
    """A speedup cell that follows the page's wall-clock / GPU-only switch."""
    return f'<span class="v-wall">{_f(w, p, "&times;")}</span><span class="v-gpu">{_f(g, p, "&times;")}</span>'


MODE_HEAD = '<span class="mode"><span class="v-wall">wall-clock</span><span class="v-gpu">GPU-only</span></span>'


def _bar_svg(P):
    """Figure: decode time per setup, split into judge passes / re-draft forwards / everything else (static SVG)."""
    W, lab, right, top, row, bar, axis = 760, 52, 92, 6, 46, 22, 30
    plot = W - lab - right
    mx = 40.0
    X = lambda v: lab + v / mx * plot
    H = top + 3 * row + axis
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="rw-fig1-t"><title id="rw-fig1-t">Decode time in hours for base, rd1 and rd2, '
         'split into judge passes, re-draft forwards and everything else; the values are in the table above.</title>']
    for t in (0, 10, 20, 30, 40):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + 3 * row}" stroke="var(--hair)" stroke-width="1"/>')
        o.append(f'<text x="{X(t):.1f}" y="{top + 3 * row + 18}" text-anchor="{"start" if t == 0 else "middle"}" font-size="11" fill="var(--muted)">{t}{" h" if t == 40 else ""}</text>')
    for k, s in enumerate(SETUPS):
        h = P[s]["hours"]
        y = top + k * row + (row - bar) / 2
        o.append(f'<text x="{lab - 10}" y="{y + bar / 2 + 4:.1f}" text-anchor="end" font-size="12" fill="var(--ink-2)" font-family="var(--font-mono)">{s}</text>')
        segs = [(h["judge"], "var(--rw-judge)", "Judge passes"), (h["redraft"], "var(--rw-redraft)", "Re-draft forwards"),
                (h["kvflush"] + h["first"] + h["host"], "var(--rw-fixed)", "KV flush, first draft, prompt prefill, host")]
        segs = [z for z in segs if z[0] > 0]
        acc = 0.0
        for j, (v, col, nm) in enumerate(segs):
            x0, x1 = X(acc), X(acc + v)
            acc += v
            last = j == len(segs) - 1
            w = max(1.0, x1 - x0 - (0 if last else 2))
            tip = f"{s}, {nm}: {v:.2f} h ({100 * v / h['total']:.1f}% of {s}&#8217;s decode time)"
            if last:
                r = 4
                d = (f"M{x0:.1f},{y:.1f}H{x0 + w - r:.1f}A{r},{r} 0 0 1 {x0 + w:.1f},{y + r:.1f}V{y + bar - r:.1f}"
                     f"A{r},{r} 0 0 1 {x0 + w - r:.1f},{y + bar:.1f}H{x0:.1f}Z")
                o.append(f'<path d="{d}" fill="{col}"><title>{tip}</title></path>')
            else:
                o.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar}" fill="{col}"><title>{tip}</title></rect>')
            if nm == "Judge passes" and w > 60:
                o.append(f'<text x="{x0 + 8:.1f}" y="{y + bar / 2 + 4:.1f}" font-size="12" font-weight="600" fill="var(--rw-onjudge)">{v:.1f} h</text>')
        endx = X(h["total"]) + 8
        sp = None if s == "base" else P[s]["vs_base"]["time"]
        if sp is None:
            o.append(f'<text x="{endx:.1f}" y="{y + bar / 2 + 4:.1f}" font-size="12" font-weight="600" fill="var(--ink)">{h["total"]:.1f} h</text>')
        else:
            o.append(f'<text x="{endx:.1f}" y="{y + 9:.1f}" font-size="12" font-weight="600" fill="var(--ink)">{h["total"]:.1f} h</text>')
            o.append(f'<text x="{endx:.1f}" y="{y + 23:.1f}" font-size="11" fill="var(--ink-2)">{sp:.3f}&#215;</text>')
    o.append("</svg>")
    return "".join(o)


def _strip(passes, run):
    rows, before = [], 0
    for k, p in enumerate(passes):
        cells = ('<i class="rw-c s"></i>' * before + '<i class="rw-c j"></i>' * p["commit"] + '<i class="rw-c a"></i>' * p["tentative"]
                 + (f'<i class="rw-c {"h" if run == "base" else "r"}"></i>') * p["remask"])
        lab = (f"Pass {k + 1}: {p['commit']} committed{' (forced)' if p['forced'] else ''}, {p['tentative']} judge agrees at or below 0.95, "
               f"{p['remask']} {'re-masked' if run == 'base' else 're-drafted'}, {before} committed earlier")
        gain = f"+{p['commit']}" + (" forced" if p["forced"] else "")
        rows.append(f'<div class="rw-row" title="{lab}"><span class="pn">{k + 1}</span><span class="rw-cells" role="img" aria-label="{lab}">{cells}</span>'
                    f'<span class="gn">{gain}</span></div>')
        before += p["commit"]
    return "".join(rows)


def sections_html(sec_no=8, table_no=9, fig_no=1, path=OUT):
    """Sections `sec_no` and `sec_no + 1`, tables numbered from `table_no`, figures from `fig_no`."""
    R = json.load(open(path))
    B, P, M2, MC, EX = R["benchmarks"], R["pooled"], R["phase2_metric"], R["machine"], R["example"]
    b, r1, r2 = P["base"], P["rd1"], P["rd2"]
    v1, v2 = r1["vs_base"], r2["vs_base"]
    T = [table_no]
    Fg = [fig_no]

    def tn():
        T[0] += 1
        return T[0] - 1

    def fgn():
        Fg[0] += 1
        return Fg[0] - 1

    def grp(label, n=3):
        return f'<th colspan="{n}" style="text-align:center">{label}</th>'

    H = []
    a = H.append
    # ------------------------------------------------------------------------ section 8
    fixed_lo = min(B[k]["setups"]["base"]["ms_blk"]["fixed"] / B[k]["setups"]["base"]["ms_blk"]["total"] for k, _ in ORDER)
    fixed_hi = max(B[k]["setups"]["base"]["ms_blk"]["fixed"] / B[k]["setups"]["base"]["ms_blk"]["total"] for k, _ in ORDER)
    hc = lambda s, k: [B[x]["setups"][s]["hard"][k] for x, _ in ORDER]
    rise = lambda s: [B[x]["setups"][s]["hard"]["commit"] / B[x]["setups"]["base"]["hard"]["commit"] - 1 for x, _ in ORDER]
    ag = lambda s: [B[x]["setups"][s]["hard"]["agree"] / B[x]["setups"]["base"]["hard"]["agree"] for x, _ in ORDER]
    p2 = [M2[k] for k, _ in ORDER]
    pooled_arm = [x["pooled_arm"] for x in p2 if "pooled_arm" in x]
    pa_keys = [k for k, _ in ORDER if "pooled_arm" in M2[k]]
    rd2_charged_pa = [M2[k]["rd2"]["charged"] for k in pa_keys]
    ife = R["ifeval_cap_vs_eos"]
    ife_rd1 = B["ifeval541"]["setups"]["rd1"]["vs_base"]
    a(f'<section><h2><span class="no">{sec_no}</span>Why re-drafting gains so little</h2>')
    a(f'<p class="caption">rd1 and rd2 cut the judge passes by {v1["judge_pass_ratio"]:.2f}&times; and {v2["judge_pass_ratio"]:.2f}&times; (all 7 benchmarks pooled), '
      f'yet the total decode time on the same items is only {v1["time"]:.3f}&times; and {v2["time"]:.3f}&times; shorter. This section splits that time by forward kind '
      f'and follows what the re-draft does pass by pass, on all {R["n_items"]:,} items per setup. Every rd and base number is read from this machine&rsquo;s per-item records; the cost-model projection, '
      f'the Phase 2 figures and the probe figures are marked with their source. '
      f'The analysis is <code>code/phase3_8gpu/redraft_why.py</code>; its numbers are in <code>results/phase3_8gpu/redraft_why.json</code>.</p>')
    a('<div class="prose"><ul>'
      f'<li><b>The saving is real but small.</b> {v1["judge_pass_ratio"]:.2f}&times; fewer judge passes is a {100 * (1 - 1 / v1["judge_pass_ratio"]):.0f}% cut. '
      f'It frees {v1["judge_saved_h"]:.1f} h (rd1) and {v2["judge_saved_h"]:.1f} h (rd2) of the {b["hours"]["total"]:.1f} h the base needs for all seven benchmarks (Table {T[0]}, Figure {Fg[0]}).</li>'
      f'<li><b>Re-drafting spends much of it again.</b> rd1&rsquo;s re-draft forwards take {v1["redraft_paid_h"]:.2f} h, {100 * v1["paid_share"]:.0f}% of the judge time it saved; '
      f'rd2&rsquo;s take {v2["redraft_paid_h"]:.2f} h ({100 * v2["paid_share"]:.0f}%), because it re-drafts half as often for the same cut in passes ({v2["judge_pass_ratio"]:.2f}&times; against {v1["judge_pass_ratio"]:.2f}&times;).</li>'
      f'<li><b>Per-block fixed costs do not shrink.</b> Every block pays one KV flush (a full 30B forward, about {b["hours"]["kvflush"] * 3600000 / b["blocks"]:.0f} ms), the first 8-step draft and its share of the prompt prefill: '
      f'{100 * fixed_lo:.0f}&ndash;{100 * fixed_hi:.0f}% of base time per block. rd&rsquo;s longer outputs add blocks, so these grow slightly (Table {T[0] + 1}).</li>'
      f'<li><b>Little crosses from the judge to the drafter.</b> At a re-draft the drafter sees the prompt and the committed tokens only: tentative tokens are hidden as [MASK] '
      f'(<code>da_decode.py:776</code>), and the judge&rsquo;s choice, the rejected token and the probabilities are not passed in. '
      f'{100 * r1["redraft"]["same"]:.1f}% of rd1&rsquo;s re-drafted tokens are the token the judge had just rejected; {100 * r1["redraft"]["judge"]:.1f}% are the judge&rsquo;s own choice (Table {T[0] + 3}).</li>'
      f'<li><b>Agreement earns nothing at commit time.</b> A slot commits only when the judge&rsquo;s own probability is above 0.95 (<code>da_decode.py:670</code>). '
      f'Blocks that need more than {HARD} judge passes take {100 * min(hc("base", "pass_share")):.0f}&ndash;{100 * max(hc("base", "pass_share")):.0f}% of the base&rsquo;s passes. '
      f'In them, re-drafting multiplies the slots where the judge agrees with the draft but stays at or below 0.95 by {min(ag("rd1") + ag("rd2")):.1f}&ndash;{max(ag("rd1") + ag("rd2")):.1f}&times;, '
      f'while commits per pass rise only {100 * min(rise("rd1") + rise("rd2")):.0f}&ndash;{100 * max(rise("rd1") + rise("rd2")):.0f}%, and {100 * min(hc("rd1", "forced") + hc("rd2", "forced")):.0f}&ndash;'
      f'{100 * max(hc("rd1", "forced") + hc("rd2", "forced")):.0f}% of those passes still commit only the one forced slot (Table {T[0] + 4}).</li>'
      f'<li><b>Tokens per second flatters rd1.</b> rd1 sends {r1["capped"]} items to the length cap (base {b["capped"]}, rd2 {r2["capped"]}), and {r1["capped_looping"]} of those {r1["capped"]} loop. '
      f'Text that runs to the cap is cheap to decode: on IFEval, capped base items ran at {ife["base"]["capped_tok_s"]:.1f} tok/s against {ife["base"]["eos_tok_s"]:.1f} tok/s for items that ended at EOS. '
      f'On items that end at EOS in both runs, rd1&rsquo;s IFEval speedup is {ife_rd1["eos"]["tok_s"]:.3f}&times; instead of {ife_rd1["all"]["tok_s"]:.3f}&times; (Table {T[0] + 2}).</li>'
      f'<li><b>This machine does not set the ceiling.</b> A drafter step costs {MC["drafter_ms_here"]:.2f} ms here against {MC["judge_ms_here"]:.1f} ms per judge pass; '
      f'on the Phase 1 machine {MC["drafter_ms_p1"]:.2f} ms against {MC["judge_ms_p1"]:.1f} ms. Both ratios are '
      f'{100 * MC["drafter_ms_here"] / MC["judge_ms_here"]:.1f}%, and the frozen cost model projects {MC["proj_p1_min"]:.2f}&ndash;{MC["proj_p1_max"]:.2f}&times; '
      f'for rd over base on the Phase 1 machine (<code>results/phase3_8gpu/PROJECTION.md</code>).</li>'
      f'<li><b>Against Phase 2.</b> On the Phase 2 page&rsquo;s own measure (30B expert loads per token against stock) rd1 and rd2 reach '
      f'{min(x["rd2"]["loads"] for x in p2):.2f}&ndash;{max(x["rd1"]["loads"] for x in p2):.2f}&times;, above the base ({min(x["base"]["loads"] for x in p2):.2f}&ndash;{max(x["base"]["loads"] for x in p2):.2f}&times;) '
      f'and every no-pool Phase 2 arm on all seven benchmarks. Phase 2&rsquo;s pooled arm (the r3_B judge with the DES48 expert pool) stays ahead once drafter forwards are charged: '
      f'{min(x["charged"] for x in pooled_arm):.2f}&ndash;{max(x["charged"] for x in pooled_arm):.2f}&times; against rd2&rsquo;s {min(rd2_charged_pa):.2f}&ndash;{max(rd2_charged_pa):.2f}&times; '
      f'on the four benchmarks it ran. The pool cuts the experts in every forward, while re-drafting adds drafter forwards; Phase 3 runs without the pool by design (Table {T[0] + 5}).</li>'
      '</ul></div>')
    # Table: decode time by component + figure
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Decode time by forward kind, in hours summed over the {R["n_items"]:,} items of each setup (wall-clock; the GPU-active counters are per item, '
      'not per forward kind, so this split has no GPU-only version). First draft and prompt prefill = the 30B and drafter prompt prefill, the first 8-step draft and the drafter flush of each block; '
      'host = the rest of each item&rsquo;s decode time.</p>')
    a('<div class="tbl"><table style="min-width:560px"><thead><tr><th class="l">Component</th><th>base</th><th>rd1</th><th>rd2</th></tr></thead><tbody>')
    for lab, k in (("Judge passes", "judge"), ("Re-draft forwards", "redraft"), ("KV flush", "kvflush"), ("First draft and prompt prefill", "first"), ("Host time between forwards", "host")):
        a(f'<tr><td class="l">{lab}</td>' + "".join(f'<td>{P[s]["hours"][k]:.2f}</td>' for s in SETUPS) + "</tr>")
    a('<tr class="ref" style="font-style:normal"><td class="l"><b>Total</b></td>' + "".join(f'<td><b>{P[s]["hours"]["total"]:.2f}</b></td>' for s in SETUPS) + "</tr>")
    a(f'<tr><td class="l">Judge passes per block</td>' + "".join(f'<td>{P[s]["passes_blk"]:.2f}</td>' for s in SETUPS) + "</tr>")
    a(f'<tr><td class="l">Total time, base &divide; setup</td><td>·</td><td class="acc arm-rd1">{v1["time"]:.3f}&times;</td><td class="acc arm-rd2">{v2["time"]:.3f}&times;</td></tr>')
    a("</tbody></table></div>")
    f1 = fgn()
    a(f'<figure class="rw-fig" style="margin:0"><ul class="rw-legend"><li><span class="rw-sw" style="background:var(--rw-judge)"></span>Judge passes</li>'
      '<li><span class="rw-sw" style="background:var(--rw-redraft)"></span>Re-draft forwards</li>'
      '<li><span class="rw-sw" style="background:var(--rw-fixed)"></span>KV flush, first draft, prompt prefill, host</li></ul>'
      f'<div class="rw-svg">{_bar_svg(P)}</div>'
      f'<figcaption class="rw-note"><b>Figure {f1}.</b> Table {t} drawn to scale. The judge bar shrinks by the same {v1["judge_saved_h"]:.1f}&ndash;{v2["judge_saved_h"]:.1f} h in both rd runs; '
      'the orange re-draft bar takes back part of it. Hover a segment for its value.</figcaption></figure>')
    # Table: per block
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Wall-clock milliseconds per block, per benchmark: where rd saves (judge passes) and where it pays (re-draft forwards). '
      'Fixed = KV flush + first draft + prompt prefill. One judge pass costs about '
      f'{MC["judge_ms_here"]:.0f} ms on this machine and one drafter step about {MC["drafter_ms_here"]:.0f} ms.</p>')
    a('<div class="tbl"><table style="min-width:1120px"><thead><tr><th class="l" rowspan="2">Benchmark</th>' + grp("Judge passes / block") + grp("Judge ms / block")
      + grp("Re-draft ms / block", 2) + grp("Fixed ms / block") + grp("Total ms / block") + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["base", "rd1", "rd2"] * 2 + ["rd1", "rd2"] + ["base", "rd1", "rd2"] * 2) + "</tr></thead><tbody>")
    for k, name in ORDER:
        S = B[k]["setups"]
        a(f'<tr><td class="l">{name}</td>' + "".join(f'<td>{S[s]["passes_blk"]:.2f}</td>' for s in SETUPS)
          + "".join(f'<td>{S[s]["ms_blk"]["judge"]:,.0f}</td>' for s in SETUPS)
          + "".join(f'<td>{S[s]["ms_blk"]["redraft"]:,.0f}</td>' for s in ("rd1", "rd2"))
          + "".join(f'<td>{S[s]["ms_blk"]["fixed"]:,.0f}</td>' for s in SETUPS)
          + "".join(f'<td>{S[s]["ms_blk"]["total"]:,.0f}</td>' for s in SETUPS) + "</tr>")
    a("</tbody></table></div>")
    # Table: runaways and EOS-only speedups
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Outputs that run to the length cap, outputs that loop (more than half of the tokens inside a {NGRAM}-token sequence that already appeared earlier in the same output), '
      'and rd&rsquo;s tokens-per-second speedup over base on all items against only the items that end at EOS in both runs. '
      'The switch at the top picks wall-clock or GPU-active seconds for the speedups.</p>')
    a('<div class="tbl"><table style="min-width:1120px"><thead><tr><th class="l" rowspan="2">Benchmark</th>' + grp("Items at the length cap") + grp("Looping outputs")
      + grp("Accuracy %") + grp(f"tok/s speedup over base{MODE_HEAD}", 4) + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["base", "rd1", "rd2"] * 3 + ["rd1, all", "rd1, EOS", "rd2, all", "rd2, EOS"]) + "</tr></thead><tbody>")
    for k, name in ORDER:
        S = B[k]["setups"]
        sp = lambda s, w: _sp(S[s]["vs_base"][w]["tok_s"], S[s]["vs_base"][w].get("gpu_tok_s"))
        a(f'<tr><td class="l">{name}</td>' + "".join(f'<td>{S[s]["capped"]}</td>' for s in SETUPS) + "".join(f'<td>{S[s]["loops"]}</td>' for s in SETUPS)
          + "".join(f'<td>{S[s]["acc"]:.1f}</td>' for s in SETUPS)
          + f'<td class="arm-rd1">{sp("rd1", "all")}</td><td class="arm-rd1">{sp("rd1", "eos")}</td><td class="arm-rd2">{sp("rd2", "all")}</td><td class="arm-rd2">{sp("rd2", "eos")}</td></tr>')
    a(f'<tr class="ref" style="font-style:normal"><td class="l"><b>All 7 benchmarks</b></td>' + "".join(f'<td>{P[s]["capped"]}</td>' for s in SETUPS)
      + "".join(f'<td>{P[s]["loops"]}</td>' for s in SETUPS) + "".join(f'<td>{P[s]["acc"]:.1f}</td>' for s in SETUPS)
      + f'<td class="arm-rd1">{_sp(v1["tok_s"], v1["gpu_tok_s"])}</td><td class="arm-rd1">{_sp(v1["eos_tok_s"], v1["eos_gpu_tok_s"])}</td>'
      f'<td class="arm-rd2">{_sp(v2["tok_s"], v2["gpu_tok_s"])}</td><td class="arm-rd2">{_sp(v2["eos_tok_s"], v2["eos_gpu_tok_s"])}</td></tr>')
    a("</tbody></table></div>")
    # Figure: one block
    f2 = fgn()
    ex = EX
    forced_b = sum(1 for p in ex["base"]["passes"] if p["forced"])
    big = max(range(len(ex["rd1"]["passes"])), key=lambda k: ex["rd1"]["passes"][k]["commit"])
    a(f'<figure class="rw-fig" style="margin:0"><ul class="rw-legend"><li><span class="rw-sw" style="background:var(--rw-fixed)"></span>Committed earlier</li>'
      '<li><span class="rw-sw" style="background:var(--rw-judge)"></span>Committed this pass</li>'
      '<li><span class="rw-sw" style="background:var(--rw-soft)"></span>Judge agrees, at or below 0.95</li>'
      '<li><span class="rw-sw" style="box-shadow:inset 0 0 0 1px var(--muted)"></span>Re-masked (base)</li>'
      '<li><span class="rw-sw" style="background:var(--rw-redraft)"></span>Re-masked, then re-drafted (rd1)</li></ul>'
      '<div class="rw-strips">'
      f'<div class="rw-strip"><h3>base <span>{len(ex["base"]["passes"])} judge passes</span></h3><div class="rw-rows">{_strip(ex["base"]["passes"], "base")}</div>'
      f'<p class="rw-note">{forced_b} passes commit only the forced slot. 30B time {ex["base"]["s30"]:.1f} s, drafter {ex["base"]["sdr"]:.1f} s.</p></div>'
      f'<div class="rw-strip"><h3>rd1 <span>{len(ex["rd1"]["passes"])} judge passes</span></h3><div class="rw-rows">{_strip(ex["rd1"]["passes"], "rd1")}</div>'
      f'<p class="rw-note">{ex["rd1"]["rounds"]} re-draft rounds wrote {ex["rd1"]["slots"]} tokens in {ex["rd1"]["fwd"]} drafter forwards: {ex["rd1"]["same"]} were the token the judge '
      f'had just rejected, {ex["rd1"]["judge"]} the judge&rsquo;s own choice. Pass {big + 1} commits {ex["rd1"]["passes"][big]["commit"]} slots at once. '
      f'30B time {ex["rd1"]["s30"]:.1f} s, drafter {ex["rd1"]["sdr"]:.1f} s.</p></div></div>'
      f'<figcaption class="rw-note"><b>Figure {f2}.</b> One block, pass by pass: MBPP task {ex["task"]}, second block, where base and rd1 wrote the same 32 tokens, '
      f'so the two histories line up. The drafter&rsquo;s first draft matched {ex["agree"]} of them. Each row is one judge pass over the 32 slots, sorted by state (slot positions are not shown). '
      'Picked as one of the longest MBPP blocks with identical text: it shows the mechanism at its clearest and is not a typical block.</figcaption></figure>')
    # Table: what the re-draft writes
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> What the re-draft writes: drafter forwards spent on re-drafts per block, and each re-drafted token compared with the token the judge had just rejected '
      'at that slot and with the judge&rsquo;s choice at the pass before.</p>')
    a('<div class="tbl"><table style="min-width:900px"><thead><tr><th class="l" rowspan="2">Benchmark</th>' + grp("Re-draft forwards / block", 2)
      + grp("Same as the rejected token", 2) + grp("The judge&rsquo;s choice", 2) + grp("Another token", 2) + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["rd1", "rd2"] * 4) + "</tr></thead><tbody>")
    other = lambda x: None if x["same"] is None else 1 - x["same"] - x["judge"]
    for k, name in ORDER:
        S = B[k]["setups"]
        a(f'<tr><td class="l">{name}</td>' + "".join(f'<td>{S[s]["redraft"]["fwd_blk"]:.1f}</td>' for s in ("rd1", "rd2"))
          + "".join(f'<td>{_pc(S[s]["redraft"]["same"])}</td>' for s in ("rd1", "rd2")) + "".join(f'<td>{_pc(S[s]["redraft"]["judge"])}</td>' for s in ("rd1", "rd2"))
          + "".join(f'<td>{_pc(other(S[s]["redraft"]))}</td>' for s in ("rd1", "rd2")) + "</tr>")
    a(f'<tr class="ref" style="font-style:normal"><td class="l"><b>All 7 benchmarks</b></td>' + "".join(f'<td>{P[s]["redraft"]["fwd_blk"]:.1f}</td>' for s in ("rd1", "rd2"))
      + "".join(f'<td>{_pc(P[s]["redraft"]["same"])}</td>' for s in ("rd1", "rd2")) + "".join(f'<td>{_pc(P[s]["redraft"]["judge"])}</td>' for s in ("rd1", "rd2"))
      + "".join(f'<td>{_pc(other(P[s]["redraft"]))}</td>' for s in ("rd1", "rd2")) + "</tr>")
    a("</tbody></table></div>")
    # Table: hard blocks
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Hard blocks, the ones that need more than {HARD} judge passes: their share of the base&rsquo;s judge passes, and per judge pass after the first '
      '(in those blocks) the slots committed, the slots where the judge agrees with the draft at or below 0.95 (kept as tentative), and the share of passes where nothing cleared 0.95 '
      'so only the single most confident slot was committed.</p>')
    a('<div class="tbl"><table style="min-width:1000px"><thead><tr><th class="l" rowspan="2">Benchmark</th><th rowspan="2">Base passes in hard blocks</th>'
      + grp("Commits / pass") + grp("Judge agrees, &le; 0.95 / pass") + grp("Passes with only the forced commit") + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["base", "rd1", "rd2"] * 3) + "</tr></thead><tbody>")
    for k, name in ORDER:
        S = B[k]["setups"]
        a(f'<tr><td class="l">{name}</td><td>{_pc(S["base"]["hard"]["pass_share"], 0)}</td>'
          + "".join(f'<td>{S[s]["hard"]["commit"]:.2f}</td>' for s in SETUPS) + "".join(f'<td>{S[s]["hard"]["agree"]:.2f}</td>' for s in SETUPS)
          + "".join(f'<td>{_pc(S[s]["hard"]["forced"], 0)}</td>' for s in SETUPS) + "</tr>")
    a("</tbody></table></div>")
    # Table: Phase 2 metric
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> The Phase 2 page&rsquo;s measure: stock&rsquo;s 30B expert loads per token &divide; the setup&rsquo;s, alone and with every drafter forward charged at '
      f'{DRAFTER_EXPERT_EQUIV:.2f} expert loads (<code>compare_table.py</code>). Pooled arm = Phase 2&rsquo;s r3_B judge with the DES48 expert pool, run on four benchmarks. '
      'Counts, so the machine does not matter.</p>')
    a('<div class="tbl"><table style="min-width:1080px"><thead><tr><th class="l" rowspan="2">Benchmark</th>' + grp("Expert loads, stock &divide; setup", 5)
      + grp("With drafter forwards charged", 4) + grp("Accuracy %", 2) + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["best no-pool Phase 2", "pooled arm", "base", "rd1", "rd2", "pooled arm", "base", "rd1", "rd2", "pooled arm", "base"]) + "</tr></thead><tbody>")
    for k, name in ORDER:
        x = M2[k]
        pa = x.get("pooled_arm")
        a(f'<tr><td class="l">{name}</td><td>{x["best_nopool_loads"]:.3f}&times;</td><td>{_f(pa and pa["loads"], 3, "&times;")}</td>'
          + "".join(f'<td class="{"arm-" + s if s != "base" else ""}">{x[s]["loads"]:.3f}&times;</td>' for s in SETUPS)
          + f'<td>{_f(pa and pa["charged"], 3, "&times;")}</td>' + "".join(f'<td class="{"arm-" + s if s != "base" else ""}">{x[s]["charged"]:.3f}&times;</td>' for s in SETUPS)
          + f'<td>{_f(pa and pa["acc"], 1)}</td><td>{B[k]["setups"]["base"]["acc"]:.1f}</td></tr>')
    a("</tbody></table></div></section>")

    # ------------------------------------------------------------------------ section 9
    c_b, c_1, c_2 = b["copy"], r1["copy"], r2["copy"]
    dup_fell = sum(1 for k, _ in ORDER if B[k]["setups"]["rd1"]["dup_per_k"] < B[k]["setups"]["base"]["dup_per_k"])
    dup_rose = [n for k, n in ORDER if B[k]["setups"]["rd1"]["dup_per_k"] > B[k]["setups"]["base"]["dup_per_k"]]
    a(f'<section><h2><span class="no">{sec_no + 1}</span>Feeding the judge&rsquo;s verdict back to the drafter</h2>')
    a('<p class="caption">The obvious next step is to let the judge tell the drafter what it wanted: write the judge&rsquo;s choice into the re-masked slots, or give it to the drafter as context. '
      'In its raw form that would bring back the next-slot copy, the old failure where the 30B answers a filled slot i with the token of slot i+1. '
      'The drafter is not the risk: it only ever predicts at [MASK] slots, and its tokens are only drafts until the judge clears 0.95. The judge is.</p>')
    a('<div class="prose"><ul>'
      '<li><b>Where the copy comes from.</b> SDAR was trained to predict only at [MASK] slots. At a filled slot the stock 30B outputs the next slot&rsquo;s token 88&ndash;94% of the time '
      '(88% with the draft in the block, 94% with the final tokens, over slots that have a next slot; <code>docs/EXPLAINER_DRAFT_ALIGNED_TRAINING.md</code> section 3): Qwen3&rsquo;s next-token head showing through. '
      f'The trained judge removed most of it: at draft slots its choice equals the next slot&rsquo;s token {_pc(c_b["draft"])} of the time in base, {_pc(c_1["draft"])} in rd1 and {_pc(c_2["draft"])} in rd2.</li>'
      f'<li><b>Why a raw fill is still risky.</b> A copy always disagrees with the draft, so what is left of the habit sits in exactly the slots a fill would write: '
      f'{_pc(c_1["of_disagree"])} of the judge&rsquo;s disagreements at draft slots in rd1 name the next slot&rsquo;s token. Written into the block, each reads as a doubled token. '
      'In the AB probe (<code>docs/DRAFT_ALIGNED_DESIGN_2026-09-06.md</code> 7.2), given draft tokens as context, the untrained model confirmed 76% of the drafter&rsquo;s wrong tokens above 0.95, and the judge is a small LoRA on that model, '
      'so a later pass could confirm the double.</li>'
      f'<li><b>What the counters cannot separate.</b> At [MASK] slots, where SDAR was trained and has no copy habit, the judge names the next slot&rsquo;s token even more often: '
      f'{_pc(c_b["mask"])} in base and {_pc(c_2["mask"])} in rd2 (rd1 re-drafts after every pass, so its judge never reads a [MASK] slot). Most of these are probably correct, '
      'because a draft that is off by one position needs exactly that fix. A guard can therefore skip such a candidate but cannot tell a copy from a repair.</li>'
      f'<li><b>The loop risk is already visible.</b> rd1 raises the looping outputs from {b["loops"]} to {r1["loops"]} (rd2 {r2["loops"]}) without passing a single judge token to the drafter. '
      f'Single-token stutter did not rise the same way: adjacent repeated words per 1,000 tokens fell on {dup_fell} of 7 benchmarks under rd1 and rose on {" and ".join(dup_rose)} '
      f'({b["dup_per_k"]:.2f} &rarr; {r1["dup_per_k"]:.2f} pooled).</li></ul></div>')
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Next-slot copy, from the <code>--judge-diag</code> counters (<code>da_decode.py:693</code>): at open slots whose next slot holds a different real token, '
      'the share where the judge&rsquo;s choice equals that next token. Disagreements = re-masks plus commits that override the draft. '
      'Repeated words = adjacent identical tokens that contain a letter or digit, per 1,000 generated tokens.</p>')
    a('<div class="tbl"><table style="min-width:1000px"><thead><tr><th class="l" rowspan="2">Benchmark</th>' + grp("Judge names the next token, at draft slots")
      + '<th rowspan="2">Share of rd1&rsquo;s disagreements</th>' + grp("At [MASK] slots", 2) + grp("Repeated words / 1,000 tokens") + "</tr><tr>"
      + "".join(f"<th>{h}</th>" for h in ["base", "rd1", "rd2", "base", "rd2", "base", "rd1", "rd2"]) + "</tr></thead><tbody>")
    for k, name in ORDER:
        S = B[k]["setups"]
        a(f'<tr><td class="l">{name}</td>' + "".join(f'<td>{_pc(S[s]["copy"]["draft"])}</td>' for s in SETUPS) + f'<td>{_pc(S["rd1"]["copy"]["of_disagree"])}</td>'
          + "".join(f'<td>{_pc(S[s]["copy"]["mask"])}</td>' for s in ("base", "rd2")) + "".join(f'<td>{S[s]["dup_per_k"]:.2f}</td>' for s in SETUPS) + "</tr>")
    a(f'<tr class="ref" style="font-style:normal"><td class="l"><b>All 7 benchmarks</b></td>' + "".join(f'<td>{_pc(P[s]["copy"]["draft"])}</td>' for s in SETUPS)
      + f'<td>{_pc(c_1["of_disagree"])}</td>' + "".join(f'<td>{_pc(P[s]["copy"]["mask"])}</td>' for s in ("base", "rd2"))
      + "".join(f'<td>{P[s]["dup_per_k"]:.2f}</td>' for s in SETUPS) + "</tr>")
    a("</tbody></table></div>")
    t = tn()
    a(f'<p class="caption"><b>Table {t}.</b> Feedback the copy habit cannot reach: the judge vetoes, the drafter chooses, and no judge token is written straight into the block. '
      '<b>Not run; nothing is queued.</b> A paired run against base on the same items, checked on accuracy, length-cap count, looping outputs, repeated words and copy rate before speed, '
      'would show whether it closes the gap.</p>')
    a('<div class="tbl"><table class="rw-rules" style="min-width:900px"><thead><tr><th class="l">Rule</th><th class="l">What changes</th><th class="l">Guards against</th>'
      '<th class="l">Extra forwards</th></tr></thead><tbody>'
      '<tr><th class="l">Ban the rejected token</th><td class="l">At each re-masked slot the drafter takes its best token other than the one the judge just rejected. It still sees no judge token.</td>'
      f'<td class="l">The {_pc(r1["redraft"]["same"], 0)} of rd1 re-drafts that repeat a rejected token '
      f'({_pc(B["humaneval"]["setups"]["rd1"]["redraft"]["same"], 0)} on HumanEval, {_pc(B["mbpp"]["setups"]["rd1"]["redraft"]["same"], 0)} on MBPP).</td><td class="l">None (a logit mask)</td></tr>'
      '<tr><th class="l">Lift the ban when the rejection looks like a copy</th><td class="l">If the judge&rsquo;s choice at slot i equals the token at slot i+1, the drafter&rsquo;s options stay unchanged.</td>'
      '<td class="l">A correct draft token rejected because of the copy habit.</td><td class="l">None</td></tr>'
      '<tr><th class="l">Judge proposes, drafter decides</th><td class="l">The drafter picks its best token among the judge&rsquo;s top candidates, skipping any equal to the tokens at i&minus;1 or i+1; '
      'if none is in the drafter&rsquo;s own top-k, it drafts as today.</td><td class="l">The judge&rsquo;s copy residue reaching the block: two models must agree before a judge candidate becomes a draft.</td>'
      '<td class="l">None (both sets of probabilities already exist)</td></tr>'
      f'<tr><th class="l">Block loop continuations</th><td class="l">A re-draft may not complete a {NGRAM}-token sequence that already appears in the output.</td>'
      f'<td class="l">The extra loops rd1 already shows ({b["loops"]} &rarr; {r1["loops"]}).</td><td class="l">None (a CPU check)</td></tr>'
      '<tr><th class="l">Keep the commit rule</th><td class="l">Only the judge commits, still at p &gt; 0.95.</td><td class="l">Anything above reaching the output unverified.</td><td class="l">None</td></tr>'
      '</tbody></table></div></section>')
    a(_note_html(sec_no + 2, R))
    return "\n".join(H)


def _note_html(sec_no, R):
    """The last section: which numbers are measured on this machine, and what the source tags in Table 2 mean."""
    FN = json.load(open(os.path.join(ROOT, "results/phase3_8gpu/final_numbers.json")))
    vs, BN = FN["vs_stock"], FN["benchmarks"]
    ident = [(BN[k].get("base_here") or {}).get("identical_to_phase1") or [0, 0] for k, _ in ORDER]
    n_same, n_all = sum(x[0] for x in ident), sum(x[1] for x in ident)
    SS = R["stock_sources"]
    gpu = lambda d: ", ".join(g.replace("NVIDIA ", "") for g in d["gpus"])
    day = lambda s: f"{int(s[8:10])} {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(s[5:7]) - 1]}"
    span = lambda d: day(d["first"]) if d["first"] == d["last"] else (
        f"{int(d['first'][8:10])}&ndash;{day(d['last'])}" if d["first"][:7] == d["last"][:7] else f"{day(d['first'])}&ndash;{day(d['last'])}")
    others = [k for k, _ in ORDER if k != "ifeval541"]
    o_gpus = sorted({g for k in others for g in SS[k]["gpus"]})
    o_first, o_last = min(SS[k]["first"] for k in others), max(SS[k]["last"] for k in others)
    ck, A_all = vs["same_machine_check_gsm8k"], vs["all"]["base_over_stock_phase2"]
    A_gsm = vs["benchmarks"]["gsm8k500"]["base_over_stock_phase2"]
    A = {k: vs["benchmarks"][k]["base_over_stock_phase2"] for k, _ in ORDER}
    wall = {k: BN[k]["base_here"]["wall"]["wall_min"] for k, _ in ORDER}
    base_h, stock_h = sum(wall.values()) / 60, sum(wall[k] * A[k] for k in wall) / 60
    MC = R["machine"]
    o = [f'<section><h2><span class="no">{sec_no}</span>What is measured on this machine, and what the tags in Table 2 mean</h2><div class="prose"><ul>',
         f'<li><b>Every rd speedup against the base is measured on this machine.</b> rd1 and rd2 are compared with the Phase 1 base (SDAR-1.7B drafter + the 128M judge), '
         f're-decoded here on all {n_all:,} items; its outputs match the Phase 1 records token for token on {n_same:,} of {n_all:,} (Table 0). '
         'Table 1, the rd columns of Table 3, the &ldquo;base on this machine&rdquo; rows of Table 6 and sections 7&ndash;8 use only these runs: no older timings and no source tags.</li>',
         '<li><b>The tags in Table 2 concern stock only.</b> Stock is the plain 30B with no drafter and no judge, added in Table 2 as a second reference at the project lead&rsquo;s request. '
         'It is not the Phase 1 base, and it was not re-decoded on this machine.</li>',
         f'<li><b>Where stock&rsquo;s timings come from.</b> Stock was decoded on all seven benchmarks, but not on this machine (apart from the {ck["items"]} setup items below). IFEval-541 ran on an {gpu(SS["ifeval541"])} '
         f'({span(SS["ifeval541"])}), the GPU model the Phase 1 base ran on: its A is <b>MEASURED</b>. The other six ran on an {", ".join(g.replace("NVIDIA ", "") for g in o_gpus)} '
         f'({span(dict(first=o_first, last=o_last))}). Seconds from that box cannot be set against the Phase 1 machine&rsquo;s, so for those six the Phase 2 page estimated stock&rsquo;s time on the Phase 1 machine '
         'from stock&rsquo;s 30B expert-load count and that machine&rsquo;s measured time per expert load: <b>STOCK COMPUTED</b>. '
         f'<b>MOSTLY COMPUTED</b> marks the All-7 row ({A_all:.3f}&times;), where six of the seven inputs are such estimates.</li>',
         f'<li><b>The {ck["items"]} GSM8K items (MEASURED HERE) are not a baseline.</b> They are the only stock items decoded on this machine '
         f'({gpu(SS["setup_check_here"])}, {span(SS["setup_check_here"])}), from a setup check stopped after {ck["items"]} items, and serve only to spot-check A. '
         f'On them the base is {ck["base_over_stock"]:.3f}&times; faster than stock here, against {A_gsm:.3f}&times; for A on GSM8K, so the chained stock speedups may be about '
         f'{100 * (A_gsm / ck["base_over_stock"] - 1):.0f}% high; {ck["items"]} items is a small sample. That this {ck["base_over_stock"]:.3f}&times; and the All-7 {A_all:.3f}&times; '
         'both round to the same three decimals is a coincidence.</li>',
         '<li><b>Exact without any timing.</b> Stock&rsquo;s accuracy and its 30B expert loads per token do not depend on the machine, so those comparisons in Table 2 are exact.</li>',
         f'<li><b>Scaling to the Phase 1 machine is projected, not measured.</b> The frozen cost model gives {MC["proj_p1_min"]:.2f}&ndash;{MC["proj_p1_max"]:.2f}&times; for rd over base there '
         '(<code>results/phase3_8gpu/PROJECTION.md</code>).</li>',
         f'<li><b>Removing the tags</b> takes one of two things, and neither has been done: decode stock on this machine next to base, rd1 and rd2 '
         f'(about {stock_h:.1f} h on 8 GPUs, an estimate from the base&rsquo;s {base_h:.1f} h here scaled by A), or drop Table 2&rsquo;s speed columns and keep its accuracy and 30B-work columns.</li>',
         '</ul></div></section>']
    return "".join(o)


if __name__ == "__main__":
    out = compute()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print("wrote", OUT, "items per setup", out["n_items"])
