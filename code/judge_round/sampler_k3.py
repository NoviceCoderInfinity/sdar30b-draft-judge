"""Judge round: the state sampler with the decode-time pass >= 2 shapes (K1 / K2 / K3).

At decode time a block at pass >= 2 holds three kinds of open-or-not slots: COMMITTED (the 30B's token), TENTATIVE
(the drafter's first draft D0 where the judge agreed with it) and HOLES (the judge rejected D0 there).  In rd1 the
drafter re-fills the holes with a re-draft D1; in rd2 they are still [MASK] at the next pass.  draftalign/dataset.py's
StateSampler only knows D0.  Items written by swap_drafts.py --redrafts also carry, per block b:

    redraft      [nb, 32] int32   D1 at the hole slots, -1 elsewhere
    redraft_conf [nb, 32] uint8   D1's confidence
    redraft_pass [nb]     int16   the vanilla pass p the re-draft state was built at (0 = none)

For such a block, regime "sample" draws one kind with probabilities k_mix = (K1, K2, K3):

    K1  the pass-1 state: StateSampler.state(k, epoch, "draft_pass1") (nothing committed, every open slot holds D0);
    K2  rd2's pass-p shape: committed = committed[b, :p].any(0) (+ accepted, as the parent), open = valid & ~committed,
        tentative = open & (D0 == T) hold D0, holes = open & ~tentative hold [MASK];
    K3  the re-draft pass shape: as K2, but a hole holds D1 where redraft >= 0 ([MASK] where it is -1).

Loss rows are every open slot (tentative + holes), targets the trajectory's tokens, as in the parent.  draft_conf is D0's
confidence at tentative slots, D1's at filled holes, 0 elsewhere.  The trainer's draft flag needs nothing new:
train_fast.block_hidden sets it at open_mask & valid & (input != [MASK]), i.e. at every open slot holding a token.

Every other block, every other regime, and every block without a usable re-draft record go to the parent unchanged
(same seed, same state).  The kind is drawn from _seed(seed, epoch, "k3", split, idx, b), so a state is deterministic
given (seed, epoch, example) on every rank, like the parent's.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from draftalign.dataset import SamplerConfig, State, StateSampler, _seed   # noqa: E402

KINDS = ("K1", "K2", "K3")
DEFAULT_K_MIX = (0.15, 0.25, 0.60)


def parse_k_mix(spec) -> Tuple[float, float, float]:
    """"0.15,0.25,0.60" (or a 3-sequence) -> (K1, K2, K3) probabilities; non-negative, summing to 1."""
    vals = [float(x) for x in (spec.split(",") if isinstance(spec, str) else spec)]
    assert len(vals) == 3 and all(v >= 0 for v in vals), f"k_mix needs three non-negative numbers: {spec!r}"
    assert abs(sum(vals) - 1.0) < 1e-6, f"k_mix must sum to 1: {spec!r} sums to {sum(vals)}"
    return (vals[0], vals[1], vals[2])


class K3Sampler(StateSampler):
    """StateSampler that, in regime "sample", draws K1 / K2 / K3 states for blocks with a re-draft record."""

    def __init__(self, cfg: SamplerConfig, entries=None, root: Optional[str] = None,
                 k_mix: Sequence[float] = DEFAULT_K_MIX):
        self.k_mix = parse_k_mix(k_mix)
        super().__init__(cfg, entries, root)

    # -- splits stay K3Samplers (the parent builds plain StateSamplers); same held-out items, same seeds --
    def _like(self, other: StateSampler) -> "K3Sampler":
        return type(self)(self.cfg, other.entries, self.root, k_mix=self.k_mix)

    def split_holdout_by_domain(self, n_per_domain: int) -> Tuple["K3Sampler", "K3Sampler"]:
        tr, ev = super().split_holdout_by_domain(n_per_domain)
        return self._like(tr), self._like(ev)

    def split_holdout(self, n_items: int) -> Tuple["K3Sampler", "K3Sampler"]:
        tr, ev = super().split_holdout(n_items)
        return self._like(tr), self._like(ev)

    # -- the state ---------------------------------------------------------
    def _has_redraft(self, it: dict, b: int) -> bool:
        return ("redraft" in it and bool(it["has_draft"][b]) and int(it["redraft_pass"][b]) > 0
                and bool((np.asarray(it["redraft"][b]) >= 0).any()))

    def state(self, k: int, epoch: int = 0, regime: str = "sample") -> State:
        i, b = self.examples[k]
        it = self._item(i)
        if regime != "sample" or not self._has_redraft(it, b):
            return super().state(k, epoch, regime)
        cfg = self.cfg
        e = self.entries[i]
        g = torch.Generator().manual_seed(_seed(cfg.seed, epoch, "k3", e["split"], e["idx"], b))
        u = float(torch.rand((), generator=g))
        kind = "K1" if u < self.k_mix[0] else ("K2" if u < self.k_mix[0] + self.k_mix[1] else "K3")

        base = super().state(k, epoch, "draft_pass1")         # clean prefix, valid, targets; K1 as is
        if kind == "K1":
            meta = dict(base.meta, regime=regime, kind="K1", n_tentative=0, n_holes=0, n_redrafted=0)
            return base._replace(meta=meta)

        bl = cfg.block
        lo = base.active_lo
        valid, tgt = base.valid, base.targets
        p = int(it["redraft_pass"][b])
        committed = torch.from_numpy(np.asarray(it["committed"][b, :p], bool).any(0).copy())
        if cfg.committed_includes_accepted:                   # the parent's rule at a pass p > 0
            committed |= torch.from_numpy(np.asarray(it["accepted"][b], bool).copy())
        committed &= valid
        open_mask = valid & ~committed
        d0 = torch.from_numpy(np.asarray(it["draft"][b], np.int64).copy())
        tentative = open_mask & (d0 >= 0) & (d0 == tgt)
        holes = open_mask & ~tentative
        if not bool(holes.any()):                             # nothing rejected at this pass: not a K2/K3 state
            return super().state(k, epoch, regime)

        block_ids = tgt.clone()                               # committed + non-valid slots hold the truth
        block_ids[open_mask] = cfg.mask_id
        block_ids[tentative] = d0[tentative]
        filled = torch.zeros(bl, dtype=torch.bool)
        if kind == "K3":
            d1 = torch.from_numpy(np.asarray(it["redraft"][b], np.int64).copy())
            filled = holes & (d1 >= 0)
            block_ids[filled] = d1[filled]
        input_ids = base.input_ids.clone()
        input_ids[lo:lo + bl] = block_ids

        conf = None
        if "draft_conf" in it:
            conf = torch.zeros(bl, dtype=torch.uint8)
            conf[tentative] = torch.from_numpy(np.asarray(it["draft_conf"][b], np.uint8).copy())[tentative]
            if kind == "K3" and "redraft_conf" in it:
                conf[filled] = torch.from_numpy(np.asarray(it["redraft_conf"][b], np.uint8).copy())[filled]

        n_open = int(open_mask.sum())
        n_tok = int((tentative | filled).sum())
        meta = dict(base.meta, regime=regime, kind=kind, n_tentative=int(tentative.sum()), n_holes=int(holes.sum()),
                    n_redrafted=int(filled.sum()), m=(n_open - n_tok) / n_open, allmask=n_tok == 0, n_open=n_open,
                    n_draft_filled=n_tok)
        meta["pass"] = p
        return State(input_ids=input_ids, active_lo=lo, valid=valid, targets=tgt, open_mask=open_mask, meta=meta,
                     draft_conf=conf)
