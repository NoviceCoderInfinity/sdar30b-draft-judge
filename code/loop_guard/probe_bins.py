#!/usr/bin/env python
"""Stage 2 probe A: the frozen judge's confidence bins on held-out pass-1 states, stored vs trained drafts.

    torchrun --standalone --nproc_per_node=8 code/loop_guard/probe_bins.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --holdout-pids logs/phase4/train/main/holdout_pids.json --drafts logs/phase4/gate1/drafts \
        --adapter logs/phase2/ref/c1b_128M/tok_000128000000 --out results/phase5_loop_guard/probe_bins.json

Same states, same judge forward and same sharding as Gate 2 (drafter_lora/judge_probe.py), but instead of one commit
share and one precision it keeps every open row: judge argmax a, top-1 probability p ([MASK] suppressed, fp32), whether
the drafted token D agrees with a, and whether a is the 30B's token T.  It reports, per confidence bin and separately
for agreeing and disagreeing rows, the row share and the precision, plus the answer to the question that decides the
judge round's design: if agreed drafts were committed at p > tau instead of 0.95 (the asymmetric rule that already
exists as --tau-commit tau --tau-override 0.95), how many more rows would commit and how precise would those extra
commits be.  Nothing is trained; the judge is never merged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b"), os.path.join(CODE, "drafter_lora")):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import torch                                                             # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign import train_fast as TF                                  # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler               # noqa: E402
from draftalign.train_drafter import dist_init                           # noqa: E402
from drafter_lora import data as D                                       # noqa: E402
from drafter_lora.judge_probe import SwappedDrafts, heldout_entries, load_drafts   # noqa: E402

EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]           # bins on p; the last is the commit bin
TAUS = (0.80, 0.85, 0.90)


def domain_of(sampler, k: int) -> str:
    i, _b = sampler.examples[k]
    return str(sampler.entries[i].get("domain") or "unknown")


def probe_rows(model, sampler: StateSampler, cfg, n_per_domain: int, block: int, seed: int, rank: int, world: int,
               device, conf_embed: bool, mask_id: int):
    """Per domain, a tensor [n_bins, 2 (disagree, agree), 2 (rows, correct)] of counts, plus totals."""
    model.eval()
    by_dom: Dict[str, List[int]] = {}
    for k in range(len(sampler.examples)):
        by_dom.setdefault(domain_of(sampler, k), []).append(k)
    doms = sorted(by_dom)
    nb = len(EDGES) - 1
    acc = torch.zeros(len(doms), nb, 2, 2, dtype=torch.float64, device=device)
    edges = torch.tensor(EDGES[1:], device=device)
    for di, d in enumerate(doms):
        g = torch.Generator().manual_seed(seed + 7 * di)                 # Gate 2's draw: same items, same order
        ks = [by_dom[d][j] for j in torch.randperm(len(by_dom[d]), generator=g)[:n_per_domain].tolist()]
        mine = [sampler.state(k, regime="draft_pass1") for k in ks[rank::world]]
        mine = [s for s in mine if s.meta.get("has_draft")]
        for j in range(0, len(mine), 32):
            sts = mine[j:j + 32]
            _, _, lgs = TF.micro_batch_loss(model, sts, cfg, block=block, checkpoint_layers=False,
                                            conf_embed=conf_embed, want_eval=True)
            for st, lg in zip(sts, lgs):
                lg = lg.float().clone()
                lg[:, mask_id] = float("-inf")
                pr = torch.softmax(lg, dim=-1)
                p, a = pr.max(dim=-1)
                lo = int(st.active_lo)
                inp = st.input_ids[lo:lo + block][st.open_mask].to(device)
                tg = st.targets[st.open_mask].to(device)
                agree = (inp != mask_id) & (a == inp)
                correct = (a == tg)
                b = torch.bucketize(p, edges, right=False).clamp(max=nb - 1)   # bin index per row
                for bi in range(nb):
                    m = (b == bi)
                    for ag in (0, 1):
                        mm = m & (agree if ag else ~agree)
                        acc[di, bi, ag, 0] += float(mm.sum()); acc[di, bi, ag, 1] += float((mm & correct).sum())
    if world > 1:
        torch.distributed.all_reduce(acc)
    return doms, acc.cpu()


def summarize(doms, acc) -> dict:
    out = {}
    nb = len(EDGES) - 1
    tot = acc.sum(0)
    for name, t in [(d, acc[i]) for i, d in enumerate(doms)] + [("all", tot)]:
        rows = float(t[:, :, 0].sum())
        bins = []
        for bi in range(nb):
            r_dis, c_dis = float(t[bi, 0, 0]), float(t[bi, 0, 1])
            r_ag, c_ag = float(t[bi, 1, 0]), float(t[bi, 1, 1])
            bins.append({"p_from": EDGES[bi], "p_to": min(EDGES[bi + 1], 1.0),
                         "share_pct": 100 * (r_dis + r_ag) / rows if rows else None,
                         "agree_rows": r_ag, "agree_precision": (c_ag / r_ag) if r_ag else None,
                         "disagree_rows": r_dis, "disagree_precision": (c_dis / r_dis) if r_dis else None})
        commit_rows = float(t[nb - 1, :, 0].sum()); commit_ok = float(t[nb - 1, :, 1].sum())
        eased = {}
        for tau in TAUS:
            sel = [bi for bi in range(nb - 1) if EDGES[bi] >= tau - 1e-9]       # agree rows with tau <= p < 0.95
            r = sum(float(t[bi, 1, 0]) for bi in sel); c = sum(float(t[bi, 1, 1]) for bi in sel)
            eased[f"agreed_at_{tau:.2f}"] = {"extra_commit_share_pct": 100 * r / rows if rows else None,
                                            "extra_commit_precision": (c / r) if r else None,
                                            "combined_commit_share_pct": 100 * (r + commit_rows) / rows if rows else None,
                                            "combined_precision": ((c + commit_ok) / (r + commit_rows)) if (r + commit_rows) else None}
        out[name] = {"rows": rows, "commit_share_pct": 100 * commit_rows / rows if rows else None,
                     "commit_precision": (commit_ok / commit_rows) if commit_rows else None, "bins": bins, "eased_rule": eased}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--holdout-pids", required=True)
    ap.add_argument("--drafts", required=True, help="the trained drafter's drafts (Phase 4 logs/phase4/gate1/drafts)")
    ap.add_argument("--adapter", default="logs/phase2/ref/c1b_128M/tok_000128000000")
    ap.add_argument("--model", default="b32")
    ap.add_argument("--n-per-domain", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=1856)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    info = dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    pids = D.load_holdout(args.holdout_pids)
    drafts = load_drafts(P.safe_out(args.drafts))
    entries = heldout_entries(args.index, pids)
    cfg = SamplerConfig(index_path="(phase4 held-out)", seed=args.seed, max_seq=args.max_seq)
    stored = StateSampler(cfg, entries)
    swapped = SwappedDrafts(cfg, entries, drafts)
    _tok, model = P.load_sdar(args.model, str(device))
    meta = A.load_adapter(model, P.safe_out(args.adapter))["meta"]
    model.eval()
    lcfg = TF.LossCfg()
    t0 = time.time()
    rep = {"adapter": os.path.abspath(args.adapter), "n_heldout_items": len(entries), "bins_p": EDGES[:-1] + [1.0]}
    for tag, s in (("stored", stored), ("trained", swapped)):
        doms, acc = probe_rows(model, s, lcfg, args.n_per_domain, 32, args.seed, rank, world, device,
                               bool(meta.get("conf_embed", False)), int(lcfg.mask_id))
        rep[tag] = summarize(doms, acc)
        if rank == 0:
            a = rep[tag]["all"]
            e = a["eased_rule"]["agreed_at_0.90"]
            print(f"[{tag}] commit share {a['commit_share_pct']:.2f}% precision {a['commit_precision']:.4f}; "
                  f"agreed drafts at 0.90-0.95: +{e['extra_commit_share_pct']:.2f}% rows at precision "
                  f"{(e['extra_commit_precision'] or float('nan')):.4f}", flush=True)
    if rank == 0:
        rep["seconds"] = round(time.time() - t0, 1)
        rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = P.safe_out(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rep, open(out, "w"), indent=1, sort_keys=True)
        print(f"[probe_bins] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
