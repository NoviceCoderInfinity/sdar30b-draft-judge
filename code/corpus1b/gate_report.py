#!/usr/bin/env python
"""The checkpoint-A gate (pre-registered in docs/CORPUS_1B_PLAN_2026-09-28.md), written to logs/c1b/gate_A.md.

    /venv/train/bin/python code/corpus1b/gate_report.py --out logs/c1b/gate_A.md

CONTINUE to wave B iff BOTH:
  (1) the held-out probe still improves from 32M to 128M in at least one of code / science / if:
      precision of confident overrides up >= 1 point, OR share of rows committed above tau up >= 5% (relative)
      while confident precision does not fall by more than 0.5 point;
  (2) GSM8K-500 (judge + DES48 at 128M) is within 1.5 points of the released model (90.2, RESULTS.md).
Otherwise STOP (the 128M judge still gets the final evaluation).
Also reported: every checkpoint-A benchmark vs the released model, and the 32M drafter-confidence test
(main tok_32M vs conf32 tok_32M at tau 0.90 / 0.95 / 0.98 on MBPP + HumanEval: keep the embedding only if
its accuracy-vs-passes curve is better).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402


def probe_at(path: str, tag: str) -> dict:
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    got = [r for r in rows if r.get("tag") == tag]
    return got[-1] if got else {}


def status_lines(ev_dir: str) -> list:
    """Every queued cell: complete (all shards .done), partial (cut by the deadline or still running), not run."""
    import glob
    import re
    L = ["", "## Evaluation jobs: finished, cut, skipped", "", "| queue | cell | shards done | items | status |",
         "|---|---|---:|---:|---|"]
    queues = sorted(glob.glob(os.path.join(ev_dir, "queue_*.txt.bak")))
    if not queues:
        return L + ["| — | — | | | no evaluation queue was written (evaluations never started) |"]
    for qb in queues:
        tag = os.path.basename(qb)[len("queue_"):-len(".txt.bak")]
        cells = {}
        for ln in open(qb):
            m = re.search(r"--out \S*/([^/\s]+) ", ln)
            n = re.search(r"--nshards (\d+)", ln)
            if m and n:
                cells[m.group(1)] = int(n.group(1))
        for cell, n in cells.items():
            d = os.path.join(ev_dir, cell)
            done = len(glob.glob(os.path.join(d, "*.jsonl.done")))
            items = sum(1 for f in glob.glob(os.path.join(d, "*.jsonl")) for l in open(f) if l.strip())
            st = ("complete" if done >= n else "partial: cut at DEADLINE or unfinished" if items else
                  "skipped: not launched before DEADLINE - 45 min")
            L.append(f"| {tag} | {cell} | {done}/{n} | {items} | {st} |")
    return L


def lr_lines(train_dir: str, main8: dict) -> list:
    """The 3-way LR test at 8M (lowest held-out probe all/ce wins); 8e-5 may be the main run's own 8M probe."""
    L = ["", "## Learning-rate test at 8M tokens (held-out probe all/ce, lower is better)", ""]
    rows = []
    for lr in ("4e-5", "8e-5", "1.6e-4"):
        r = probe_at(os.path.join(train_dir, f"lr_{lr}", "probe.jsonl"), "tok_000008000000") or \
            probe_at(os.path.join(train_dir, f"lr_{lr}", "probe.jsonl"), "final")
        src = f"lr_{lr}"
        if not r and lr == "8e-5" and main8:
            r, src = main8, "main run @ 8M (same schedule and data order)"
        if r:
            rows.append((r.get("all/ce"), lr, src, r.get("tokens", 0)))
    choice = os.path.join(train_dir, "lr_choice.txt")
    if not rows:
        skipped = os.path.exists(os.path.join(train_dir, "lr_test_skipped.txt"))
        return L + [f"Not run{' (no time before the main run: LR 8e-5 used)' if skipped else ''}."]
    L += ["| LR | probe all/ce | source | tokens |", "|---|---:|---|---:|"]
    for ce, lr, src, tok in sorted(rows, key=lambda x: (x[0] is None, x[0])):
        L.append(f"| {lr} | {'—' if ce is None else f'{ce:.4f}'} | {src} | {tok / 1e6:.2f}M |")
    if os.path.exists(choice):
        L += ["", f"LR choice (logs/c1b/train/lr_choice.txt): **{open(choice).read().strip()}**"]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/c1b/gate_A.md")
    ap.add_argument("--probe", default="logs/c1b/train/main/probe.jsonl")
    ap.add_argument("--evals", default="logs/c1b/eval/checkpointA/summary.json")
    a = ap.parse_args(argv)
    J = lambda p: p if os.path.isabs(p) else os.path.join(C.ROOT, p)
    p32 = probe_at(J(a.probe), "tok_000032000000")
    p128 = probe_at(J(a.probe), "tok_000128000000") or probe_at(J(a.probe), "final")
    ev = json.load(open(J(a.evals))) if os.path.exists(J(a.evals)) else {}
    L = ["# Checkpoint A gate", ""]
    ck_txt = J(os.path.join(os.path.dirname(a.evals), "checkpoint.txt"))
    if p128:
        L += [f"Checkpoint A = `{open(ck_txt).read().strip() if os.path.exists(ck_txt) else 'main'}` at "
              f"{p128.get('tokens', 0) / 1e6:.2f}M corpus tokens (probe tag `{p128.get('tag')}`).", ""]

    # (1) probe
    L += ["## Held-out probe, 32M -> 128M", "",
          "| domain | conf. precision | conf. share | override conf. precision | improving? |", "|---|---|---|---|---|"]
    improving = []
    for d in ("maths", "code", "science", "if", "chat"):
        g = lambda p, k: p.get(f"{d}/{k}")
        cp0, cp1 = g(p32, "conf_precision"), g(p128, "conf_precision")
        cf0, cf1 = g(p32, "conf_frac"), g(p128, "conf_frac")
        op0, op1 = g(p32, "ovr_conf_precision"), g(p128, "ovr_conf_precision")
        if None in (cp0, cp1, cf0, cf1):
            L.append(f"| {d} | missing | | | |")
            continue
        ok = False
        if op0 is not None and op1 is not None and op1 - op0 >= 0.01:
            ok = True
        if cf0 > 0 and (cf1 - cf0) / cf0 >= 0.05 and cp1 >= cp0 - 0.005:
            ok = True
        if d in ("code", "science", "if") and ok:
            improving.append(d)
        pc = lambda x: "—" if x is None else f"{100 * x:.1f}"
        L.append(f"| {d} | {pc(cp0)} -> {pc(cp1)} | {pc(cf0)} -> {pc(cf1)} | {pc(op0)} -> {pc(op1)} | "
                 f"{'yes' if ok else 'no'} |")
    # (2) GSM8K
    g = ev.get("gsm8k500_judge128M_des48", {})
    gsm_ok = g.get("acc") is not None and g["acc"] >= 90.2 - 1.5
    L += ["", f"GSM8K-500 judge+DES48 at checkpoint A: {g.get('acc', 'missing')} on n={g.get('n', 0)} "
              f"(released 90.2; must be >= 88.7) -> {'OK' if gsm_ok else 'NOT OK'}", ""]
    verdict = "CONTINUE to wave B" if (improving and gsm_ok) else "STOP (final evaluation on the 128M judge)"
    L += [f"**Verdict: {verdict}.** Improving domains: {', '.join(improving) or 'none'}.", ""]
    if not p32:
        # supplementary (NOT the pre-registered rule): the same improvement test between the probes this run has
        rows = [json.loads(l) for l in open(J(a.probe))] if os.path.exists(J(a.probe)) else []
        tags = [r for r in rows if r.get("tag", "").startswith("tok_") or r.get("tag") == "final"]
        if len(tags) >= 2:
            p0, p1 = tags[0], tags[-1]
            L += [f"> The pre-registered probe criterion (32M -> 128M) is not evaluable: training stopped at "
                  f"{p1.get('tokens', 0) / 1e6:.2f}M tokens on this rental. Supplementary, same test on the probes that "
                  f"exist ({p0['tag']} @ {p0['tokens'] / 1e6:.2f}M -> {p1['tag']} @ {p1['tokens'] / 1e6:.2f}M):", "",
                  "| domain | conf. precision | conf. share | override conf. precision | improving? |", "|---|---|---|---|---|"]
            sup = []
            for d in ("maths", "code", "science", "if", "chat"):
                gg = lambda p, k: p.get(f"{d}/{k}")
                cp0, cp1, cf0, cf1 = gg(p0, "conf_precision"), gg(p1, "conf_precision"), gg(p0, "conf_frac"), gg(p1, "conf_frac")
                op0, op1 = gg(p0, "ovr_conf_precision"), gg(p1, "ovr_conf_precision")
                if None in (cp0, cp1, cf0, cf1):
                    continue
                ok = (op0 is not None and op1 is not None and op1 - op0 >= 0.01) or \
                     (cf0 > 0 and (cf1 - cf0) / cf0 >= 0.05 and cp1 >= cp0 - 0.005)
                if d in ("code", "science", "if") and ok:
                    sup.append(d)
                pc = lambda x: "—" if x is None else f"{100 * x:.1f}"
                L.append(f"| {d} | {pc(cp0)} -> {pc(cp1)} | {pc(cf0)} -> {pc(cf1)} | {pc(op0)} -> {pc(op1)} | "
                         f"{'yes' if ok else 'no'} |")
            L += ["", f"Supplementary reading: improving in {', '.join(sup) or 'none'} of code / science / if; with the "
                      f"GSM8K criterion {'met' if gsm_ok else 'not met'}, the gate would read "
                      f"**{'CONTINUE' if (sup and gsm_ok) else 'STOP'}** on these probes (not the pre-registered one).", ""]

    # benchmarks
    L += ["## Checkpoint-A benchmarks", "", "| cell | n | acc % | vs released | passes/blk | x fewer passes |",
          "|---|---:|---:|---:|---:|---:|"]
    f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"
    for name, s in sorted(ev.items()):
        if "_c0." in name:
            continue
        L.append(f"| {name} | {s['n']} | {f(s.get('acc'))} | {f(s.get('d_acc_vs_released'))} | {f(s['ppb'], 2)} | "
                 f"{f(s.get('speedup_ppb_vs_released'), 2)} |")

    # 32M test
    L += ["", "## 32M test: drafter confidence as a judge input", "",
          "| set | tau | main 32M acc / passes | conf 32M acc / passes |", "|---|---|---|---|"]
    wins = 0
    for s in ("mbpp", "humaneval"):
        for t in ("0.90", "0.95", "0.98"):
            m, c = ev.get(f"{s}_main32M_c{t}", {}), ev.get(f"{s}_conf32M_c{t}", {})
            if m and c:
                better = (c.get("acc", 0) >= m.get("acc", 0) and c["ppb"] <= m["ppb"]) and \
                         (c.get("acc", 0) > m.get("acc", 0) or c["ppb"] < m["ppb"])
                wins += int(better)
                L.append(f"| {s} | {t} | {f(m.get('acc'))} / {f(m['ppb'], 2)} | {f(c.get('acc'))} / {f(c['ppb'], 2)}"
                         f"{' (better)' if better else ''} |")
    n32 = sum(1 for k in ev if "32M_c" in k)
    if n32 == 0:
        L += ["", "Not run (the 32M drafter-confidence test was cut on the 17 h rental; see the extra-work notes)."]
    else:
        L += ["", f"Confidence embedding better at {wins} of 6 points: "
                  f"{'train the 1B run WITH it (--conf-embed)' if wins >= 4 else 'keep the main recipe'}."]
    L += status_lines(J(os.path.dirname(a.evals)))
    L += lr_lines(J(os.path.dirname(os.path.dirname(a.probe))), probe_at(J(a.probe), "tok_000008000000"))
    out = J(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
