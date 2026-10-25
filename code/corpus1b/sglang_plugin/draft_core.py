"""The drafter's static fill of ONE block, as a stepper: draftalign/drafter.py::draft_block (static settings) without
the SeqCache, so the SGLang plugin c1b_judge_draft.py (one batched forward per step) and the HF harness run the same
rule.  Pure torch; no sglang import.

    df = DraftFill(mask_id).begin(block_ids, fill, steps, ban=ban, guard=guard)
    while not df.done:
        df.step(logits_of(df.cur))         # logits [bl, V] of the block df.cur, any float dtype
    res = df.result()

`fill_block` is that loop around a `logits_fn`.  Semantics = draft_block(seq, final=block_ids, valid=fill, ...):
  * the fill slots are set to [MASK] (draftalign.drafter.masked_base); every other slot is fixed, including [MASK]
    slots outside `fill` (a re-draft's hidden tentative slots), which stay [MASK] and are never filled;
  * steps = min(steps, n_fill) forwards; step k commits the nt[k] most confident still-open slots,
    nt = get_num_transfer_tokens(n_fill, steps) (sums to n_fill);
  * per step, fp32 numerics: the [MASK] column is suppressed, then the ban (phase 4) and the loop guard (phase 5,
    draftalign.loop_guard.LoopGuard, called as drafter.py calls it) set their (slot, token) logits to -inf before
    softmax, argmax and the confidence ranking;
  * counters: raw_mask_argmax (open slots whose raw fp32 argmax was [MASK], summed over forwards), guard_slots
    (slots with a non-empty guard ban, summed over forwards), guard_changed (committed slots whose token the guard
    changed).  n_fill == 0 runs no forward.

Tested against draft_block on the CPU fakes of code/tests/test_da_decode.py:
    python3 -m pytest -q code/corpus1b/tests/test_draft_core.py code/corpus1b/tests/test_judge_draft_plugin.py
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def get_num_transfer_tokens(n: int, steps: int) -> torch.Tensor:
    """== sdar_harness.decode.get_num_transfer_tokens (copied: the plugin must not import the harness)."""
    base = n // steps
    remainder = n % steps
    nt = torch.zeros(steps, dtype=torch.int64) + base
    nt[:remainder] += 1
    return nt


class DraftFill:
    """One block's fill, one forward at a time (see the module docstring)."""

    def __init__(self, mask_id: int):
        self.mask_id = int(mask_id)

    def begin(self, block_ids: torch.Tensor, fill: torch.Tensor, steps: int,
              ban: Optional[Union[torch.Tensor, Iterable[Sequence[int]]]] = None, guard=None,
              valid_for_guard: Optional[torch.Tensor] = None) -> "DraftFill":
        """block_ids [bl] (any int dtype, kept), fill [bl] bool.  `ban`: a [bl] long tensor (drafter.py's form: slot j
        may not take ban[j] when ban[j] >= 0) or (slot, token) pairs.  `guard`: a LoopGuard already at this block
        (start_item + begin_block done).  `valid_for_guard`: the `fill` argument of guard.bans, default `fill`."""
        dev = block_ids.device
        self.fill = fill.to(device=dev, dtype=torch.bool).clone()
        self.cur = block_ids.clone()
        self.cur[self.fill] = self.mask_id                          # draftalign.drafter.masked_base
        self.open_ = self.fill.clone()
        self.n_fill = int(self.fill.sum())
        bl = self.cur.numel()
        self.p_commit = torch.zeros(bl, dtype=torch.float32, device=dev)
        self.raw_mask = torch.zeros((), dtype=torch.long, device=dev)
        self.ninf = torch.tensor(float("-inf"), device=dev)
        self.n_fwd = self.g_slots = self.g_changed = 0
        if self.n_fill:
            steps = min(int(steps), self.n_fill)
            assert steps >= 1, steps
            self.nt = [int(k) for k in get_num_transfer_tokens(self.n_fill, steps)]
        else:
            self.nt = []
        self.ban_rows = self.ban_cols = None
        if ban is not None:
            if torch.is_tensor(ban):
                ban = ban.to(dev)
                rows = torch.nonzero(ban >= 0).flatten()
                cols = ban[rows]
            else:
                pairs = [(int(s), int(t)) for s, t in ban]
                rows = torch.tensor([s for s, _ in pairs], dtype=torch.long, device=dev)
                cols = torch.tensor([t for _, t in pairs], dtype=torch.long, device=dev)
            if rows.numel():
                self.ban_rows, self.ban_cols = rows, cols
        self.guard = guard
        self.guard_fill = (self.fill if valid_for_guard is None else valid_for_guard.to(torch.bool)).cpu()
        return self

    @property
    def n_steps(self) -> int:
        return len(self.nt)

    @property
    def done(self) -> bool:
        return self.n_fwd >= len(self.nt)

    @torch.no_grad()
    def step(self, logits: torch.Tensor) -> bool:
        """Apply one step from `logits` [bl, V], the forward of the block as `self.cur` holds it.  Returns done."""
        assert not self.done, "step() after the last step"
        mask_id, dev = self.mask_id, self.cur.device
        lg = logits.to(torch.float32, copy=True)                    # fp32 copy (the MASK logit is overwritten)
        self.raw_mask += (lg.argmax(-1) == mask_id)[self.open_].sum()
        lg[:, mask_id] = float("-inf")
        if self.ban_rows is not None:
            lg[self.ban_rows, self.ban_cols] = float("-inf")
        x0_free = None
        if self.guard is not None:
            g_rows, g_cols = self.guard.bans(self.cur.cpu(), self.open_.cpu(), self.guard_fill)
            if g_rows:
                x0_free = lg.argmax(-1)                             # the unconstrained choice, for the counters
                self.g_slots += len(set(g_rows))
                lg[torch.tensor(g_rows, device=dev), torch.tensor(g_cols, device=dev)] = float("-inf")
        probs = F.softmax(lg, dim=-1)
        conf, x0 = probs.max(dim=-1)
        conf_open = torch.where(self.open_, conf, self.ninf)
        transfer = torch.zeros_like(self.open_)
        transfer[torch.topk(conf_open, self.nt[self.n_fwd]).indices] = True
        if x0_free is not None:
            self.g_changed += int(((x0 != x0_free) & transfer).sum())
        self.cur = torch.where(transfer, x0.to(self.cur.dtype), self.cur)
        self.p_commit = torch.where(transfer, conf, self.p_commit)
        self.open_ = self.open_ & ~transfer
        self.n_fwd += 1
        if self.done:
            assert int(self.open_.sum()) == 0, f"{int(self.open_.sum())} slots left open after {self.n_fwd} steps"
            if self.guard is not None:
                self.guard.n_changed += self.g_changed
        return self.done

    def result(self) -> dict:
        assert self.done, "result() before the last step"
        return {"tokens": self.cur, "p_commit": self.p_commit, "n_fwd": self.n_fwd,
                "raw_mask_argmax": int(self.raw_mask), "guard_slots": self.g_slots, "guard_changed": self.g_changed}


@torch.no_grad()
def fill_block(block_ids: torch.Tensor, fill: torch.Tensor, steps: int,
               logits_fn: Callable[[torch.Tensor], torch.Tensor], mask_id: int,
               ban=None, guard=None, valid_for_guard: Optional[torch.Tensor] = None) -> dict:
    """Fill the `fill` slots of `block_ids` [bl]; `logits_fn(ids [bl]) -> logits [bl, V]` runs the drafter (it must not
    modify `ids`).  Returns {tokens [bl], p_commit [bl] fp32 (0 where not filled), n_fwd, raw_mask_argmax,
    guard_slots, guard_changed}; see the module docstring."""
    df = DraftFill(mask_id).begin(block_ids, fill, steps, ban=ban, guard=guard, valid_for_guard=valid_for_guard)
    while not df.done:
        df.step(logits_fn(df.cur))
    return df.result()
