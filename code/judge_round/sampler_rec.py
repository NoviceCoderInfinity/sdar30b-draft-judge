"""Judge round, piece A: the state sampler over the judge's RECORDED per-pass states.

judge_sglang.py --mode record decodes every trainset item with the whole judge pipeline (the 30B with the flag-gated
LoRA, the drafter, the commit rule, re-drafts, the ban, the loop guard), teacher-forced between blocks, and stores
every judge pass of every block (docs/PHASE5_SGLANG_JUDGE.md section 6).  index_states.py (piece A0) validates those
chunks into one memory-mapped index.  RecStateSampler serves them to corpus1b/train_c1b.py as dataset.State tuples,
through judge_round/train_c1b_rec.py; train_c1b, train_fast and the loss are unchanged.

  * One EXAMPLE is an (item, block) pair of the parent's (usable, within max_seq) that has a recorded pass.  Items
    without a record and blocks without a pass are counted in summary() and never sampled (require_record=True).
  * Regime "sample": ONE recorded pass of the block per epoch, drawn uniformly (pass_pick "uniform") among the
    passes that leave at least one loss row, from _seed(seed, epoch, "rec", split, idx, b): deterministic on every
    rank and rerun, a new draw each epoch.  One state per block keeps train_c1b's token accounting (tokens_share).
  * The state is the judge's view at that pass: the clean prefix (prompt + vanilla's earlier blocks, which is what
    teacher forcing put in the KV cache) and the block exactly as the judge saw it (`input`: its own commits, the
    tentative draft tokens, re-drafted tokens or [MASK] at the holes); targets = vanilla's tokens T; loss rows =
    the pass's open slots (tentative + holes), so a committed slot is context, never a row.
  * train_fast.block_hidden's flag (open_mask & valid & (input != [MASK])) then equals the recorded flag on the
    valid slots: A0 checks flag == open & (input != [MASK]) on every pass.
  * Post-EOS slots (after vanilla's first EOS, last block only): the recorder treats them as valid, the trainset
    does not (corpus1b/to_trainset.py).  post_eos="truth" (default): valid = the trainset's; those slots hold T, carry
    no flag and no loss row (the parent's convention); the view differs from the judge's only there.
    post_eos="exact": valid = the recorder's (slot >= the prompt tail); the recorded tokens and flag stay; those
    slots are loss rows with target T, which the trainset never trains.  The operator picks (plan question 3).
  * Every other regime ("draft_pass1", "allmask_pass1": the held-out probe) is the parent's state, bit for bit, and
    the held-out split keeps EVERY usable block (require_record=False), so the probe is the parent's probe on the
    same held-out items whether or not they were recorded.
  * No drafter confidence (draft_conf None): the 128M adapter has none; the wrapper refuses --conf-embed.

meta adds: kind "rec", "pass" (the recorded pass number), n_passes, n_eligible, forced, rec_row / rec_pass (global
block / pass rows of the index: the wrapper's step-0 check reads the recorded argmax there), post_eos, repeated
(the index row's flag: data the earlier rounds trained on).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from draftalign.dataset import SamplerConfig, State, StateSampler, _seed   # noqa: E402
from judge_round.index_states import StatesIndex                           # noqa: E402

PASS_PICKS = ("uniform",)
POST_EOS = ("truth", "exact")
STATE_KEYS = ("pass_no", "forced", "input", "open")      # what a state reads from the record


class RecStateSampler(StateSampler):
    """StateSampler whose regime "sample" states are the judge's recorded per-pass states."""

    def __init__(self, cfg: SamplerConfig, entries=None, root: Optional[str] = None, *, store: StatesIndex,
                 pass_pick: str = "uniform", post_eos: str = "truth", require_record: bool = True):
        assert pass_pick in PASS_PICKS, f"pass_pick {pass_pick!r} not in {PASS_PICKS}"
        assert post_eos in POST_EOS, f"post_eos {post_eos!r} not in {POST_EOS}"
        assert int(cfg.block) == int(store.block_size) == int(store.cfg.get("block_size", store.block_size)), \
            f"block {cfg.block} != the records' {store.block_size}"
        assert int(cfg.mask_id) == int(store.mask_id), f"--mask-id {cfg.mask_id} != the records' {store.mask_id}"
        self.store, self.pass_pick, self.post_eos, self.require_record = store, pass_pick, post_eos, require_record
        self.skipped: Counter = Counter()
        super().__init__(cfg, entries, root)
        keys = [(str(e.get("split")), int(e["idx"])) for e in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("two index rows share (split, idx): train_fast.group_states keys an item's prefix on that "
                             "pair, so two items would share one prefix (build_fresh_index.py renumbers idx: use ONE "
                             "index)")

    # -- construction --------------------------------------------------------
    def _build_examples(self) -> None:
        super()._build_examples()                       # usable, max_seq, require_draft: the parent's rules
        keep = []
        for i, b in self.examples:
            pid = str(self.entries[i]["pid"])
            row = self.store.block_row(pid, b)
            if row is None:
                self.skipped["item_without_record" if self.store.n_blocks(pid) == 0 else "block_without_record"] += 1
            elif self.store.n_passes(row) == 0:
                self.skipped["block_without_pass"] += 1
            else:
                keep.append((i, b))
                continue
            if not self.require_record:
                keep.append((i, b))
        self.n_recorded = sum(1 for i, b in keep if self._recorded(i, b))
        self.examples = keep

    def _recorded(self, i: int, b: int) -> bool:
        row = self.store.block_row(str(self.entries[i]["pid"]), b)
        return row is not None and self.store.n_passes(row) > 0

    # -- splits stay RecStateSamplers; the held-out side keeps every usable block --
    def _like(self, other: StateSampler, require_record: bool) -> "RecStateSampler":
        return type(self)(self.cfg, other.entries, self.root, store=self.store, pass_pick=self.pass_pick,
                          post_eos=self.post_eos, require_record=require_record)

    def split_holdout_by_domain(self, n_per_domain: int) -> Tuple["RecStateSampler", "RecStateSampler"]:
        tr, ev = super().split_holdout_by_domain(n_per_domain)
        return self._like(tr, self.require_record), self._like(ev, False)

    def split_holdout(self, n_items: int) -> Tuple["RecStateSampler", "RecStateSampler"]:
        tr, ev = super().split_holdout(n_items)
        return self._like(tr, self.require_record), self._like(ev, False)

    # -- the state -------------------------------------------------------------
    def state(self, k: int, epoch: int = 0, regime: str = "sample") -> State:
        if regime != "sample":
            return super().state(k, epoch, regime)
        i, b = self.examples[k]
        e = self.entries[i]
        rec = self.store.block(str(e["pid"]), b, keys=STATE_KEYS)
        if rec is None or rec.n == 0:
            if self.require_record:
                raise KeyError(f"{e['pid']} block {b}: no recorded pass")
            return super().state(k, epoch, regime)          # a held-out block without a record: the parent's
        cfg = self.cfg
        bl = int(cfg.block)
        base = super().state(k, epoch, "draft_pass1")      # the clean prefix, the trainset's valid, T, the meta
        lo = int(base.active_lo)
        T = base.targets.numpy()
        if rec.lo != lo or not np.array_equal(np.asarray(rec.teacher, np.int64), T):
            raise ValueError(f"{e['pid']} block {b}: the record's start or teacher tokens differ from the trainset "
                             f"item's (states of another index?)")
        v_tr = base.valid.numpy()
        v_rec = np.arange(bl) >= rec.start
        if np.any(v_tr & ~v_rec):
            raise ValueError(f"{e['pid']} block {b}: a trainset-valid slot was not valid for the recorder")
        valid = v_tr if self.post_eos == "truth" else v_rec
        opn = np.asarray(rec.passes["open"], bool) & valid[None, :]
        elig = np.flatnonzero(opn.any(1))
        if len(elig) == 0:                                  # pass 0 is open at every valid slot: cannot happen
            raise ValueError(f"{e['pid']} block {b}: no recorded pass with a loss row")
        g = torch.Generator().manual_seed(_seed(cfg.seed, epoch, "rec", e["split"], e["idx"], b))
        j = int(elig[int(torch.randint(len(elig), (1,), generator=g))])

        blk = np.where(valid, np.asarray(rec.passes["input"][j], np.int64), T)
        input_ids = base.input_ids.clone()
        input_ids[lo:lo + bl] = torch.from_numpy(blk)
        open_mask = torch.from_numpy(opn[j].copy())
        n_open = int(open_mask.sum())
        n_tok = int((open_mask & torch.from_numpy(blk != cfg.mask_id)).sum())
        meta = dict(base.meta, regime=regime, kind="rec", n_open=n_open, n_draft_filled=n_tok,
                    m=(n_open - n_tok) / n_open, allmask=n_tok == 0, n_passes=rec.n, n_eligible=int(len(elig)),
                    forced=bool(rec.passes["forced"][j]), rec_row=rec.row, rec_pass=rec.pass0 + j,
                    post_eos=self.post_eos, repeated=bool(e.get("repeated", False)))
        meta["pass"] = int(rec.passes["pass_no"][j])
        return State(input_ids=input_ids, active_lo=lo, valid=torch.from_numpy(valid.copy()), targets=base.targets,
                     open_mask=open_mask, meta=meta, draft_conf=None)

    def summary(self) -> dict:
        s = super().summary()
        pids = {str(e["pid"]) for e in self.entries}
        with_rec = sum(1 for p in pids if self.store.n_blocks(p) > 0)
        s.update({"sampler": "rec", "records": self.store.path, "pass_pick": self.pass_pick,
                  "post_eos": self.post_eos, "require_record": self.require_record,
                  "recorded_examples": self.n_recorded, "items_with_record": with_rec,
                  "items_without_record": len(pids) - with_rec, "skipped": dict(self.skipped),
                  "repeated_examples": sum(1 for i, _b in self.examples if self.entries[i].get("repeated")),
                  "records_outside_these_entries": sum(1 for r in self.store.rids if r not in pids)})
        return s
