#!/usr/bin/env python
"""The canonical evaluation table: every decode cell under logs/draftalign/da_decode, scored from its record files,
plus the drafter gates, written to results/reports/EVAL_TABLE.md.  Re-run after any eval (the hourly sync does).

Definitions (all from the per-item records written by run_da_decode.py):
  acc          exact-match accuracy of the final answer (the branch's GSM8K/MATH/MBPP scorer), percent
  d_acc        paired accuracy difference vs the STOCK decode on the SAME items, percentage points, with a 95% paired
               bootstrap interval (2,000 resamples over items); "vs free" = the same against the training-free rule
  fwd/blk      MoE forwards per 32-token block, flush excluded, block-weighted = sum(passes) / sum(blocks) (analyze_da's pass/blk)
  fwd/blk(im)  the same as a mean over items of each item's own mean (the per-item definition used in chat tables)
  exp/tok      expert-layer executions per generated token = sum(unique_expert_layer_loads) / sum(generated tokens)
  ms/tok       wall-clock milliseconds per generated token = 1000 * sum(seconds) / sum(generated tokens)
  agree        drafter/final-token agreement, mean over blocks with a draft
  items        records on disk (a partial cell is scored on the items finished so far); shards = .done markers of 4 (or 1)
"""
from __future__ import annotations
import glob, json, os, re, statistics, subprocess, sys, time, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "logs", "draftalign", "da_decode")
OUT = os.path.join(ROOT, "results", "reports", "EVAL_TABLE.md")
DRAFTERS = {"a5250149": "fine-tuned d1 step 1,080"}   # _dr<hash8> tag -> checkpoint

def load_cells():
    cells = {}
    for d in sorted(glob.glob(f"{D}/*/")):
        for f in glob.glob(f"{d}*.shard*of*.jsonl"):          # the record file carries the cell name (it can differ from the dir, e.g. `_p32desf`)
            m = re.match(r"(.+)\.([a-z0-9]+)\.shard\d+of\d+\.jsonl$", os.path.basename(f))
            if not m:
                continue
            cell, split = m.group(1), m.group(2)
            recs = cells.setdefault((cell, split), {})
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("correct") is None:
                    continue
                recs[r["idx"]] = r
    done = {k: len(glob.glob(f"{D}/*/{k[0]}.{k[1]}.shard*.jsonl.done")) for k in cells}
    nsh = {k: (int(re.search(r"of(\d+)", os.path.basename(f)).group(1)) if (f := (glob.glob(f"{D}/*/{k[0]}.{k[1]}.shard*.jsonl") or [""])[0]) else 0) for k in cells}
    return cells, done, nsh

def judge_of(recs):
    ad = next(iter(recs.values())).get("adapter") or ""
    m = re.search(r"train/(t1_gsm8k_r1|t2a_gsm8k_flag|t2b_union_flag)/step0*(\d+)", ad)
    if not m:
        return "" if not ad else os.path.basename(ad)
    run = {"t1_gsm8k_r1": "T1", "t2a_gsm8k_flag": "T2a", "t2b_union_flag": "T2b"}[m.group(1)]
    return f"{run} step {int(m.group(2)):,}"

def rule_of(cell):
    if "_stock" in cell:
        return "stock" + ("" if cell.endswith("_stock") else " + " + cell.split("_stock_")[1])
    tail = cell.split("dv-s8_", 1)[1]
    tail = re.sub(r"_(ad|dr)[0-9a-f]{8}.*$", "", tail)
    return tail

def drafter_of(cell):
    m = re.search(r"_dr([0-9a-f]{8})", cell)
    return DRAFTERS.get(m.group(1), f"ckpt {m.group(1)}") if m else "stock"

def metrics(recs, ref=None, free=None, rng=None):
    idx = sorted(recs); n = len(idx)
    acc = 100 * sum(recs[i]["correct"] for i in idx) / n
    fb = sum(sum(recs[i]["passes_per_block"]) for i in idx) / max(1, sum(recs[i]["n_blocks"] for i in idx))
    fim = statistics.mean(statistics.mean(recs[i]["passes_per_block"]) for i in idx)
    gen = sum(max(1, recs[i]["n_gen_to_eos"]) for i in idx)
    exp = sum(recs[i]["unique_expert_layer_loads"] for i in idx) / gen
    ms = 1000 * sum(recs[i]["seconds"] for i in idx) / gen
    ag = [a for i in idx for a in (recs[i].get("draft_agreement_per_block") or []) if a is not None]
    agree = statistics.mean(ag) if ag else float("nan")
    def paired(other):
        if not other: return ""
        com = [i for i in idx if i in other]
        if len(com) < 5: return ""
        d = [recs[i]["correct"] - other[i]["correct"] for i in com]
        bs = sorted(100 * statistics.mean(rng.choice(d) for _ in d) for _ in range(2000))
        return f"{100*statistics.mean(d):+.1f} [{bs[50]:+.1f}, {bs[1949]:+.1f}] (n={len(com)})"
    return dict(n=n, acc=acc, d_stock=paired(ref), d_free=paired(free), fb=fb, fim=fim, exp=exp, ms=ms, agree=agree)

def main():
    rng = random.Random(0)
    cells, done, nsh = load_cells()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    L = [f"# Evaluation table (canonical)\n\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} "
         f"({time.strftime('%H:%M IST', time.localtime(time.time()+19800))}) by `code/draftalign/eval_table.py` at commit `{head}` "
         f"from the record files under `logs/draftalign/da_decode/`. Re-run the script after any evaluation; do not edit by hand.\n",
         "## Definitions\n", "```", __doc__.split("Definitions", 1)[1].strip("\n").replace("(all from the per-item records written by run_da_decode.py):", ""), "```\n"]
    splits = {"calibvalid": "GSM8K calibvalid = 100 frozen GSM8K TRAIN prompts (50 calibration + 50 validation), never in any training set",
              "gsm8k500": "GSM8K-500 = 500 frozen GSM8K TEST items (stock decode reference on this split: 90.2 %, 10.06 fwd/blk, 1,164 exp/tok, 198 ms/tok — the analyze_da reference cell, recorded outside da_decode/)", "math500": "MATH-500 = 500 test items", "mbpp": "MBPP sanitized test = 257 items"}
    for split, desc in splits.items():
        rows = [(k, v) for k, v in cells.items() if k[1] == split]
        if not rows: continue
        ref = next((v for k, v in rows if k[0].endswith("_stock")), None)
        free = next((v for k, v in rows if k[0].endswith("_dv-s8_blind_q50")), None)
        L.append(f"## {desc}\n")
        L.append("| judge | rule | drafter | items | shards | acc % | d_acc vs stock | d_acc vs free rule | fwd/blk | fwd/blk (im) | exp/tok | ms/tok | agree | cell |")
        L.append("|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
        def key(kv):
            cell = kv[0][0]; j = judge_of(kv[1]); m = re.search(r"step ([\d,]+)", j)
            return (0 if "_stock" in cell else 1 if not j else 2, {"T1": 0, "T2a": 1, "T2b": 2}.get(j.split(" ")[0], 9), int(m.group(1).replace(",", "")) if m else 0, rule_of(cell))
        for (cell, _), recs in sorted(rows, key=key):
            m = metrics(recs, ref if not cell.endswith("_stock") else None, free if not cell.endswith("_dv-s8_blind_q50") and not cell.endswith("_stock") else None, rng)
            j = judge_of(recs)
            L.append(f"| {j or '—'} | {rule_of(cell)} | {drafter_of(cell) if not cell.endswith('_stock') and '_stock_' not in cell else '—'} | {m['n']} | {done[(cell, split)]}/{nsh[(cell, split)]} | "
                     f"{m['acc']:.1f} | {m['d_stock']} | {m['d_free']} | {m['fb']:.2f} | {m['fim']:.2f} | {m['exp']:.0f} | {m['ms']:.0f} | "
                     f"{'' if m['agree'] != m['agree'] else f'{m['agree']:.3f}'} | `{cell}` |")
        L.append("")
    # drafter gates
    L.append("## Drafter fine-tune (SDAR-1.7B-Chat-b32, full fine-tune on the 30B's own answers) — gates\n")
    L.append("| drafter checkpoint | slot agreement vs stock (8-step draft replayed on the 30B's recorded blocks, paired held-out items) | standalone accuracy on calibvalid (drafter alone, block 32, thr .95) | passes/block alone |")
    L.append("|---|---|---|---:|")
    for tag, step in (("", "1,080"), ("_step1000", "1,000")):
        try:
            g = json.load(open(f"{ROOT}/results/metrics/draftalign/drafter_agreement_d1{tag}.json"))["paired"]
            a = json.load(open(f"{ROOT}/results/metrics/draftalign/drafter_accuracy_d1{tag}.json"))
            L.append(f"| d1 step {step} | {g['slot_agreement_ckpt']:.3f} vs {g['slot_agreement_baseline']:.3f} (item-mean {g['mean_item_delta']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}], better/worse {g['items_better']}/{g['items_worse']}, n={g['n_paired']}) | {100*a['accuracy']:.0f} / 100 | {a['passes_per_block_mean']:.2f} |")
        except FileNotFoundError:
            pass
    try:
        a = json.load(open(f"{ROOT}/results/metrics/draftalign/drafter_accuracy_stock.json"))
        L.append(f"| stock d32 | — (baseline) | {100*a['accuracy']:.0f} / 100 | {a['passes_per_block_mean']:.2f} |")
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}: {sum(1 for l in L if l.startswith('| '))-len(splits)-1} rows")

if __name__ == "__main__":
    main()
