"""State sampler for draft-aligned training (TRAINER_SPEC_v0 section 4, design 3.3).

One example is an (item, block) pair. Its STATE is drawn fresh each time it is
visited, from the corruption process the design fixes:

  * with probability `p_pass1` (0.4) nothing is committed -- the pass-1 state the
    judge actually meets at inference;
  * otherwise a recorded denoising pass p is drawn uniformly and the committed set
    is everything the trajectory had committed BEFORE that pass (committed[0..p-1]);
  * every open (valid, uncommitted) slot then holds the drafter's token with
    probability 1 - m and [MASK] with probability m, m ~ U(0, 1);
  * with probability `p_allmask` (0.2) the whole open set is [MASK] instead -- the
    stock regime, which keeps the adapter from forgetting mask-init decoding.
    A block with no recorded draft can only ever be sampled in that regime.

The prefix is clean: the prompt plus the FINAL tokens of the earlier blocks, which
is exactly what the inference KV cache holds. The prompt tail inside the first
generated block is fixed and is not a valid slot, so it is never corrupted and
never contributes a loss row.

Loss rows = the open valid slots. Everything is deterministic given (seed, epoch,
example): the same call returns the same state on every rank and every rerun.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

MASK_ID = 151669
BLOCK = 32
REGIMES = ("sample", "draft_pass1", "allmask_pass1")


class State(NamedTuple):
    input_ids: torch.Tensor        # [S] long, S = active_lo + block
    active_lo: int                 # absolute start of the active block
    valid: torch.Tensor            # [block] bool, the generated slots
    targets: torch.Tensor          # [block] long, the trajectory's final tokens
    open_mask: torch.Tensor        # [block] bool, the loss rows (valid & not committed)
    meta: dict
    draft_conf: Optional[torch.Tensor] = None   # [block] uint8 drafter confidence (1B corpus items only; else None)


# 1B-corpus items name their domain in the index (`domain`); older items get one from their split name
def domain_of_entry(entry: dict) -> str:
    d = entry.get("domain")
    if d:
        return str(d)
    s = str(entry.get("split", ""))
    if s.startswith("c1b_"):
        return s[4:]
    if s.startswith(("opencode", "mbpp", "humaneval")):
        return "code"
    if s.startswith("if"):
        return "if"
    return "maths"


@dataclass
class SamplerConfig:
    index_path: str
    seed: int = 0
    p_pass1: float = 0.4
    p_allmask: float = 0.2
    p_draftfull: float = 0.0      # extra mass on m = 0 (all open slots draft-filled); round 1 used 0.0
    mask_id: int = MASK_ID
    block: int = BLOCK
    max_seq: Optional[int] = 1200
    only_correct: bool = False
    require_draft: bool = False                 # keep only blocks that have a draft
    committed_includes_accepted: bool = True    # judge-accepted slots were committed context at inference (default True, 2026-09-06)
    cache_items: int = 64

    def to_json(self) -> dict:
        return asdict(self)


def _seed(*parts) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 63 - 1)


class StateSampler:
    """Index-backed sampler over (item, block) examples."""

    def __init__(self, cfg: SamplerConfig, entries: Optional[List[dict]] = None,
                 root: Optional[str] = None):
        self.cfg = cfg
        self.root = root if root is not None else os.path.dirname(os.path.abspath(cfg.index_path))
        if entries is None:
            entries = [json.loads(l) for l in open(cfg.index_path) if l.strip()]
        self.entries = entries
        self._cache: Dict[int, dict] = {}
        self._order: List[int] = []
        self.examples: List[Tuple[int, int]] = []
        self._build_examples()

    # -- construction ------------------------------------------------------
    def _build_examples(self) -> None:
        for i, e in enumerate(self.entries):
            if self.cfg.only_correct and not e.get("correct"):
                continue
            if "usable" in e:                       # per-block metadata from build_train_set
                usable, has_draft, start = e["usable"], e["has_draft"], e["block_start"]
            else:                                   # fall back to the npz itself
                it = self._item(i)
                usable, has_draft, start = it["usable"], it["has_draft"], it["block_start"]
            for b in range(len(usable)):
                if not bool(usable[b]):
                    continue
                if self.cfg.max_seq and int(start[b]) + self.cfg.block > self.cfg.max_seq:
                    continue
                if self.cfg.require_draft and not bool(has_draft[b]):
                    continue
                self.examples.append((i, b))

    def _item(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        path = self.entries[i]["file"]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        d = dict(np.load(path).items())
        if len(self._cache) >= max(1, self.cfg.cache_items):
            self._cache.pop(next(iter(self._cache)))
        self._cache[i] = d
        return d

    def __len__(self) -> int:
        return len(self.examples)

    # -- ordering ----------------------------------------------------------
    def epoch_order(self, epoch: int) -> torch.Tensor:
        """A deterministic permutation of the examples for one epoch."""
        g = torch.Generator().manual_seed(_seed(self.cfg.seed, "order", epoch))
        return torch.randperm(len(self.examples), generator=g)

    def global_state(self, step_index: int, epoch: int = 0, regime: str = "sample") -> State:
        """The `step_index`-th example of `epoch` in shuffled order (wraps around)."""
        order = self.epoch_order(epoch)
        return self.state(int(order[step_index % len(order)]), epoch=epoch, regime=regime)

    # -- the state ---------------------------------------------------------
    def state(self, k: int, epoch: int = 0, regime: str = "sample") -> State:
        assert regime in REGIMES, regime
        i, b = self.examples[k]
        it = self._item(i)
        cfg = self.cfg
        bl = cfg.block
        g = torch.Generator().manual_seed(
            _seed(cfg.seed, epoch, regime, self.entries[i]["split"], self.entries[i]["idx"], b))

        lo = int(it["block_start"][b])
        clean = np.concatenate([np.asarray(it["prompt_ids"], np.int64)[:int(it["block_start"][0])],
                                np.asarray(it["tokens"], np.int64).reshape(-1)])
        input_ids = torch.from_numpy(clean[:lo + bl].copy())
        valid = torch.from_numpy(np.asarray(it["valid"][b], bool).copy())
        targets = torch.from_numpy(np.asarray(it["tokens"][b], np.int64).copy())
        draft = torch.from_numpy(np.asarray(it["draft"][b], np.int64).copy())
        has_draft = bool(it["has_draft"][b])

        # 1. committed set
        n_pass = int(it["n_pass"][b])
        committed = torch.zeros(bl, dtype=torch.bool)
        pass_used = 0
        if regime == "sample" and n_pass > 1 and float(torch.rand((), generator=g)) >= cfg.p_pass1:
            p = int(torch.randint(1, n_pass, (1,), generator=g))
            cum = np.asarray(it["committed"][b, :p], bool).any(0)
            committed = torch.from_numpy(cum.copy())
            pass_used = p
        if cfg.committed_includes_accepted and pass_used:
            committed |= torch.from_numpy(np.asarray(it["accepted"][b], bool).copy())
        committed &= valid
        open_mask = valid & ~committed
        if not bool(open_mask.any()):                       # never train on a finished block
            committed[:] = False
            open_mask = valid.clone()
            pass_used = 0

        # 2. fill the open slots
        if regime == "allmask_pass1" or not has_draft:
            allmask, m = True, 1.0
        elif regime == "draft_pass1":
            allmask, m = False, 0.0
        else:
            u = float(torch.rand((), generator=g))
            if u < cfg.p_allmask:                       # stock regime: every open slot [MASK]
                allmask, m = True, 1.0
            elif u < cfg.p_allmask + cfg.p_draftfull:   # inference pass-1 regime: every open slot holds the draft
                allmask, m = False, 0.0
            else:                                       # mixed: each open slot MASK with prob m ~ U(0,1)
                allmask, m = False, float(torch.rand((), generator=g))
        keep_draft = (torch.rand(bl, generator=g) >= m) & open_mask & (draft >= 0)
        if allmask:
            keep_draft[:] = False

        block_ids = targets.clone()                         # committed + non-valid slots hold the truth
        block_ids[open_mask] = cfg.mask_id
        block_ids[keep_draft] = draft[keep_draft]
        input_ids[lo:lo + bl] = block_ids

        meta = {"item": i, "block": b, "idx": int(self.entries[i]["idx"]),
                "split": self.entries[i]["split"], "example": int(k), "epoch": int(epoch),
                "regime": regime, "pass": pass_used, "m": m, "allmask": bool(allmask),
                "has_draft": has_draft, "n_open": int(open_mask.sum()),
                "n_draft_filled": int(keep_draft.sum()), "seq_len": lo + bl,
                "correct": self.entries[i].get("correct"),
                "domain": domain_of_entry(self.entries[i]),
                "tokens_share": self.tokens_share(i)}
        conf = None
        if "draft_conf" in it:
            conf = torch.from_numpy(np.asarray(it["draft_conf"][b], np.uint8).copy())
            conf[~keep_draft] = 0                               # confidence only where a draft token sits
        return State(input_ids=input_ids, active_lo=lo, valid=valid, targets=targets,
                     open_mask=open_mask, meta=meta, draft_conf=conf)

    # -- item-grouped access (the two-stream trainer, code/draftalign/train_fast.py) --
    def item_examples(self) -> Dict[int, List[int]]:
        """item index -> its example indices (usable blocks), in block order."""
        if getattr(self, "_by_item", None) is None:
            by: Dict[int, List[int]] = {}
            for k, (i, _b) in enumerate(self.examples):
                by.setdefault(i, []).append(k)
            self._by_item = by
        return self._by_item

    def item_order(self, epoch: int) -> List[int]:
        """A deterministic permutation of the items that have at least one example."""
        items = sorted(self.item_examples())
        g = torch.Generator().manual_seed(_seed(self.cfg.seed, "item_order", epoch))
        return [items[j] for j in torch.randperm(len(items), generator=g).tolist()]

    def tokens_share(self, i: int) -> float:
        """Corpus tokens this item's example stands for: the item's corpus tokens (prompt + kept answer)
        spread over its usable examples, so one epoch counts exactly the index's corpus tokens."""
        e = self.entries[i]
        n = max(1, len(self.item_examples().get(i, ())))
        tot = e.get("corpus_tokens")
        if tot is None:
            tot = int(e.get("seq_len", 0))
        return float(tot) / n

    def split_holdout_by_domain(self, n_per_domain: int) -> Tuple["StateSampler", "StateSampler"]:
        """Deterministic ITEM-level split with `n_per_domain` held-out items in EVERY domain."""
        by_dom: Dict[str, List[int]] = {}
        for j, e in enumerate(self.entries):
            by_dom.setdefault(domain_of_entry(e), []).append(j)
        hold = set()
        for d, js in sorted(by_dom.items()):
            g = torch.Generator().manual_seed(_seed(self.cfg.seed, "holdout_dom", d, n_per_domain))
            perm = torch.randperm(len(js), generator=g).tolist()
            hold.update(js[p] for p in perm[:max(0, int(n_per_domain))])
        tr = [e for j, e in enumerate(self.entries) if j not in hold]
        ev = [e for j, e in enumerate(self.entries) if j in hold]
        return (StateSampler(self.cfg, tr, self.root), StateSampler(self.cfg, ev, self.root))

    # -- held-out split ----------------------------------------------------
    def split_holdout(self, n_items: int) -> Tuple["StateSampler", "StateSampler"]:
        """Deterministic ITEM-level split (no block of an eval item appears in training)."""
        g = torch.Generator().manual_seed(_seed(self.cfg.seed, "holdout", n_items))
        perm = torch.randperm(len(self.entries), generator=g).tolist()
        hold = set(perm[:max(0, int(n_items))])
        tr = [e for j, e in enumerate(self.entries) if j not in hold]
        ev = [e for j, e in enumerate(self.entries) if j in hold]
        return (StateSampler(self.cfg, tr, self.root), StateSampler(self.cfg, ev, self.root))

    def summary(self) -> dict:
        n_draft = sum(1 for i, b in self.examples
                      if bool((self.entries[i].get("has_draft") or self._item(i)["has_draft"])[b]))
        return {"items": len(self.entries), "examples": len(self.examples), "draft_examples": n_draft}
