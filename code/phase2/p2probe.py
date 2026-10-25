"""The Phase 2 calibration probe: how the judge adapter treats a drafter's drafts on held-out states.

Draft-filled pass-1 states (every open slot holds the draft: the state the judge meets first at inference) are
drawn per domain from a training index -- for a new drafter, the RE-DRAFTED index (`redraft.py`), so the drafts
are that drafter's. Per domain and overall it reports

    acc             argmax == the 30B model's final token, over open rows
    keep_rate       among rows whose draft is right: argmax == target (the judge keeps a correct draft)
    fix_rate        among rows whose draft is wrong: argmax == target (the judge fixes a wrong draft)
    conf_frac       share of rows with top-1 > tau (0.95): the rows the judge would commit
    conf_precision  their precision (pass/fail floor 0.99, rules.json)
    ovr_conf_*      the same on rows whose draft is wrong
    ece             expected calibration error of top-1 over 10 equal-width bins
    draft_acc       the draft's own agreement on these rows (separates "weaker drafter" from "misaligned judge")
    ce              mean cross-entropy

"right" means "equals the 30B model's token", not "the true answer": the probe diagnoses, only the benchmark
scores decide (Notion risk table).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as Fn

from draftalign import train_fast as F
from draftalign.dataset import StateSampler, domain_of_entry

KEYS = ("rows", "correct", "draft_right", "keep", "draft_wrong", "fix", "conf", "conf_correct",
        "conf_ovr", "conf_correct_ovr", "ce_sum")
NBINS = 10


@torch.no_grad()
def probe(model, sampler: StateSampler, cfg: "F.LossCfg", n_per_domain: int, block: int, seed: int,
          rank: int, world: int, device, conf_embed: bool = False, tau: float = 0.95,
          micro: int = 32) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    by_dom: Dict[str, List[int]] = {}
    for k, (i, _b) in enumerate(sampler.examples):
        by_dom.setdefault(domain_of_entry(sampler.entries[i]), []).append(k)
    doms = sorted(by_dom)
    acc = torch.zeros(len(doms), len(KEYS), dtype=torch.float64, device=device)
    bins = torch.zeros(len(doms), 3, NBINS, dtype=torch.float64, device=device)     # count, sum conf, sum correct
    for di, d in enumerate(doms):
        g = torch.Generator().manual_seed(seed + 7 * di)
        ks = [by_dom[d][j] for j in torch.randperm(len(by_dom[d]), generator=g)[:n_per_domain].tolist()]
        mine = [sampler.state(k, regime="draft_pass1") for k in ks[rank::world]]
        mine = [s for s in mine if s.meta.get("has_draft")]
        for j in range(0, len(mine), micro):
            chunk = mine[j:j + micro]
            _, _stats, logits = F.micro_batch_loss(model, chunk, cfg, block=block, checkpoint_layers=False,
                                                   conf_embed=conf_embed, want_eval=True)
            for st, lg in zip(chunk, logits):
                tg = st.targets[st.open_mask].to(lg.device)
                lo = int(st.active_lo)
                draft = st.input_ids[lo:lo + block][st.open_mask].to(lg.device)
                p = Fn.softmax(lg.float(), dim=-1)
                top_p, top_i = p.max(-1)
                ok = (top_i == tg).double()
                dr = (draft == tg).double()
                conf = (top_p > tau).double()
                ce = Fn.cross_entropy(lg.float(), tg, reduction="sum").double()
                acc[di] += torch.stack([ok.new_tensor(float(tg.numel())), ok.sum(), dr.sum(), (ok * dr).sum(),
                                        (1 - dr).sum(), (ok * (1 - dr)).sum(), conf.sum(), (ok * conf).sum(),
                                        (conf * (1 - dr)).sum(), (ok * conf * (1 - dr)).sum(), ce])
                b = torch.clamp((top_p.double() * NBINS).long(), max=NBINS - 1)
                bins[di, 0].index_add_(0, b, torch.ones_like(top_p, dtype=torch.float64))
                bins[di, 1].index_add_(0, b, top_p.double())
                bins[di, 2].index_add_(0, b, ok)
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(acc)
        dist.all_reduce(bins)
    if was_training:
        model.train()
    out: Dict[str, float] = {}
    nan = float("nan")
    rows = [(d, acc[i], bins[i]) for i, d in enumerate(doms)] + [("all", acc.sum(0), bins.sum(0))]
    for name, row, bn in rows:
        r = dict(zip(KEYS, row.tolist()))
        div = lambda a, b: (r[a] / r[b]) if r[b] else nan                      # noqa: E731
        out[f"{name}/rows"] = r["rows"]
        out[f"{name}/acc"] = div("correct", "rows")
        out[f"{name}/draft_acc"] = div("draft_right", "rows")
        out[f"{name}/keep_rate"] = div("keep", "draft_right")
        out[f"{name}/fix_rate"] = div("fix", "draft_wrong")
        out[f"{name}/conf_frac"] = div("conf", "rows")
        out[f"{name}/conf_precision"] = div("conf_correct", "conf")
        out[f"{name}/ovr_conf_frac"] = div("conf_ovr", "draft_wrong")
        out[f"{name}/ovr_conf_precision"] = div("conf_correct_ovr", "conf_ovr")
        out[f"{name}/ce"] = div("ce_sum", "rows")
        cnt, sconf, sok = bn[0].cpu().numpy(), bn[1].cpu().numpy(), bn[2].cpu().numpy()
        tot = cnt.sum()
        out[f"{name}/ece"] = float(np.abs(sconf - sok).sum() / tot) if tot else nan
        if name == "all":
            out["all/reliability"] = [[int(c), float(sc / c) if c else nan, float(so / c) if c else nan]
                                      for c, sc, so in zip(cnt, sconf, sok)]
    return out
