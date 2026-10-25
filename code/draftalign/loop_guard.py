"""Phase 5: the loop guard (Phase 3 report, Table 15, "block loop continuations").

A drafter fill may not complete an n-token sequence that already appears in the committed output.  At every
drafter step, for each slot still to be filled whose n-1 preceding tokens are all known (not [MASK]) in the
drafter's working state, every token t such that (those n-1 tokens, t) already occurs in the committed output is
banned: its logit is set to -inf before tokens are chosen and confidences ranked, exactly like the phase-4 ban of the
rejected token.  The judge is untouched: only the judge commits, still at p > tau_commit.  No extra forward; one CPU
dictionary lookup per open slot per drafter step.

"Committed output" = every generated token of the finished earlier blocks, plus the committed slots of the current
block (fixed, non-[MASK], not being filled by this call; tentative slots are [MASK] in the re-drafter's input, so
they never count).  Prompt tokens are never indexed, but they may serve as the left context of a slot.

Off (n = 0) nothing is touched: the decode is byte-identical to phase 4.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import torch


class LoopGuard:
    def __init__(self, n: int, mask_id: int):
        assert int(n) >= 2, n
        self.n = int(n)
        self.mask_id = int(mask_id)
        self.prompt_length = 0
        self.prefix: List[int] = []                 # x[0, :lo]: prompt + finished blocks (final tokens)
        self.lo = 0
        self.index: Dict[Tuple[int, ...], Set[int]] = {}   # (n-1 tokens) -> next tokens seen in the committed output
        self._indexed_upto = 0                      # prefix positions < this have their n-gram (ending there) indexed
        self.n_slots = 0                            # (step, slot) pairs with a non-empty ban set
        self.n_changed = 0                          # transferred slots whose token differs from the unconstrained argmax

    # -- per item / per block ------------------------------------------------------------------------------------
    def start_item(self, prompt_length: int) -> None:
        self.prompt_length = int(prompt_length)
        self.prefix, self.lo, self.index, self._indexed_upto = [], 0, {}, 0
        self.n_slots = self.n_changed = 0

    def begin_block(self, x_row: Sequence[int], lo: int) -> None:
        """`x_row[:lo]` is the final sequence before the block: the prompt and every finished block."""
        lo = int(lo)
        self.prefix = [int(t) for t in x_row[:lo]]
        self.lo = lo
        n = self.n
        for end in range(max(self._indexed_upto, self.prompt_length + n - 1), lo):   # n-gram = prefix[end-n+1 .. end]
            key = tuple(self.prefix[end - n + 1:end])
            self.index.setdefault(key, set()).add(self.prefix[end])
        self._indexed_upto = max(self._indexed_upto, lo)

    # -- per drafter step ----------------------------------------------------------------------------------------
    def bans(self, state: torch.Tensor, open_: torch.Tensor, fill: torch.Tensor) -> Tuple[List[int], List[int]]:
        """state [bl] long = the drafter's working block; open_ [bl] bool = slots still to fill at this step;
        fill [bl] bool = every slot this call fills.  Returns (rows, cols): slot j may not take token cols[k]
        where rows[k] == j."""
        n, lo, pl, mask = self.n, self.lo, self.prompt_length, self.mask_id
        blk = [int(t) for t in state.tolist()]
        op = [bool(b) for b in open_.tolist()]
        fl = [bool(b) for b in fill.tolist()]
        bl = len(blk)
        # committed slots of this block: a real token, not being filled by this call, not part of the prompt
        comm = [blk[j] != mask and not fl[j] and lo + j >= pl for j in range(bl)]

        def tok(pos: int):
            if pos < lo:
                return self.prefix[pos] if pos >= 0 else None
            t = blk[pos - lo]
            return None if t == mask else t

        def committed(pos: int) -> bool:
            if pos < pl:
                return False
            return True if pos < lo else comm[pos - lo]

        local: Dict[Tuple[int, ...], Set[int]] = {}
        for j in range(bl):                     # n-grams of the committed output that end inside this block
            end = lo + j
            if not comm[j] or end - n + 1 < pl:
                continue
            if all(committed(p) for p in range(end - n + 1, end)):
                key = tuple(tok(p) for p in range(end - n + 1, end))
                local.setdefault(key, set()).add(blk[j])
        rows: List[int] = []
        cols: List[int] = []
        for j in range(bl):
            if not op[j]:
                continue
            ctx = [tok(p) for p in range(lo + j - n + 1, lo + j)]
            if any(t is None for t in ctx):
                continue                        # the n-1 preceding tokens are not all known yet
            key = tuple(ctx)
            banned = self.index.get(key, set()) | local.get(key, set())
            if banned:
                self.n_slots += 1
                for t in sorted(banned):
                    rows.append(j); cols.append(int(t))
        return rows, cols
