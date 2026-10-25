"""CPU tests for phase 5: the n-gram loop guard on the drafter's fills (--loop-guard n).

  * n = 0 runs exactly the phase-4 loop (same tokens, passes, records; no loop-guard record fields);
  * with the guard a drafter that writes a period-12 cycle cannot continue it into the next block: no 12-gram of
    block 2 appears in block 1, the guard counters say where it acted, and the judge's rule is untouched;
  * the guard needs the full judge and a drafter;
  * LoopGuard.bans unit cases: an unknown left context bans nothing; a known one bans exactly the continuations
    seen in the committed output (earlier blocks and the block's own committed slots), never from the prompt alone.

No real model is loaded; the fakes are tests/test_da_decode.py's.
Run from the repo root:  /venv/p2/bin/python -m pytest -q code/tests/test_da_decode_loopguard.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign import da_decode as DA                                                     # noqa: E402
from draftalign.da_decode import da_generate                                               # noqa: E402
from draftalign.loop_guard import LoopGuard                                                # noqa: E402
from tests.test_da_decode import (ScriptedLM, onehot_row, logprob_row, prompt_ids, meter_for,   # noqa: E402
                                  item, FakeTok, MASK, EOS_IDS, BL)

PLEN, GEN, N = 32, 64, 12            # one prefill block, two decode blocks (32..63, 64..95), a 12-gram guard


def cyc(pos: int) -> int:
    """The cycle token at `pos`: 1..12 with period 12."""
    return 1 + (pos % N)


def alt(pos: int) -> int:
    """The runner-up at `pos`: 20..31, never equal to any cycle token."""
    return 20 + (pos % N)


def cycle_drafter_fn(pos, tok, is_mask, ids, cached, pkv):
    """Proposes the cycle at every [MASK] slot, runner-up `alt`; confidence falls with position so the s8 draft
    fills left to right (4 slots per step).  A ban sets the cycle token's logit to -inf and the softmax renormalises,
    so the banned slot's confidence becomes alt / (1 - p): the runner-up's share is chosen so that this (.35 / .40 =
    .875) still ranks above every unbanned slot (< .60) and the banned slot is filled in the step its context became
    known, not deferred to the last step (which the phase-4 ban mechanism otherwise does when the runner-up is weak)."""
    if not is_mask:
        return onehot_row(tok)
    p = .60 - .0005 * (pos - PLEN)
    return logprob_row({cyc(pos): p, alt(pos): .35, 0: 1.0 - p - .35})


def echo_judge_fn(pos, tok, is_mask, ids, cached, pkv):
    """Commits whatever real token a slot holds (p ~ 1); a [MASK] slot would get the cycle token."""
    if pos < PLEN:
        return onehot_row(tok)
    return onehot_row(cyc(pos) if is_mask else tok)


def run(**kw):
    moe = meter_for(echo_judge_fn)
    dr = SeqCache(ScriptedLM(cycle_drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, **kw)
    return moe, dr, res


def ngrams(ids, n=N):
    return {tuple(ids[k:k + n]) for k in range(len(ids) - n + 1)}


def test_guard_off_is_the_phase4_loop():
    for kw in ({}, {"full_redraft_every": 1, "redraft_ban_rejected": True}):
        moe_a, dr_a, a = run(**kw)
        moe_b, dr_b, b = run(loop_guard=0, **kw)
        assert torch.equal(a.x, b.x)
        assert [blk.full_passes for blk in a.blocks] == [blk.full_passes for blk in b.blocks]
        assert b.loop_guard == 0 and all(blk.n_loopguard_slots == 0 for blk in b.blocks)
        rec = DA.make_record(b, FakeTok(), item(), prompt_ids(PLEN), moe_b, dr_b, EOS_IDS, MASK, 0.0)
        assert "loop_guard" not in rec and "loopguard_changed_per_block" not in rec
    # without the guard the cycle runs through both blocks: every 12-gram of block 2 is already in block 1
    _, _, a = run()
    gen = a.generated_ids()[0].tolist()
    assert gen == [cyc(p) for p in range(PLEN, PLEN + GEN)]
    assert ngrams(gen[BL:]) <= ngrams(gen[:BL])


def test_guard_breaks_the_cycle_in_the_next_block_and_counts_it():
    moe, dr, res = run(loop_guard=N)
    assert res.loop_guard == N
    gen = res.generated_ids()[0].tolist()
    b1, b2 = gen[:BL], gen[BL:]
    assert b1 == [cyc(p) for p in range(PLEN, PLEN + BL)]           # nothing committed yet: the first block is free
    assert not (ngrams(b2) & ngrams(b1))                             # no 12-gram of block 2 appears in block 1
    # the guard acts exactly where the 11 preceding tokens were known and cycle-shaped: slots 64, 76, 88
    changed = [p for p in range(PLEN + BL, PLEN + GEN) if gen[p - PLEN] != cyc(p)]
    assert changed == [64, 76, 88]
    assert all(gen[p - PLEN] == alt(p) for p in changed)             # the runner-up took the banned slot
    assert [blk.n_loopguard_changed for blk in res.blocks] == [0, 3]
    assert res.blocks[0].n_loopguard_slots == 0 and res.blocks[1].n_loopguard_slots >= 3
    # the judge's rule is untouched: every slot is committed by the judge on pass 1 in both blocks
    assert all(len(blk.full_passes) == 1 and blk.full_passes[0]["commit"] == BL for blk in res.blocks)
    rec = DA.make_record(res, FakeTok(), item(), prompt_ids(PLEN), moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["loop_guard"] == N and rec["loopguard_changed_per_block"] == [0, 3]
    assert rec["gen_ids"][:GEN] == gen


def test_guard_also_covers_redrafts():
    """With re-drafting on, the re-draft calls get the same guard (the counters still fire in block 2)."""
    _, _, res = run(loop_guard=N, full_redraft_every=1, redraft_ban_rejected=True)
    gen = res.generated_ids()[0].tolist()
    assert not (ngrams(gen[BL:]) & ngrams(gen[:BL]))
    assert res.redraft_ban_rejected is True and res.loop_guard == N


def test_guard_needs_the_full_judge_and_a_drafter():
    moe = meter_for(echo_judge_fn)
    dr = SeqCache(ScriptedLM(cycle_drafter_fn), BL, "drafter")
    with pytest.raises(NotImplementedError):
        da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                    draft="s8", judge="halves", tau_commit=.95, loop_guard=N)
    with pytest.raises(NotImplementedError):
        da_generate(moe, None, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                    draft=None, judge="full", tau_commit=.95, loop_guard=N)


def test_loop_guard_bans_unit():
    g = LoopGuard(4, MASK)                      # 4-grams keep the hand cases short
    g.start_item(prompt_length=8)
    # prompt 0..7 = [9]*8 ; output 8..19 = 1 2 3 4 1 2 3 4 5 6 7 8  (block of 12 at lo = 8 already finished)
    row = [9] * 8 + [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8]
    g.begin_block(row, lo=20)
    assert g.index[(1, 2, 3)] == {4} and g.index[(2, 3, 4)] == {1, 5}
    assert (9, 9, 9) not in g.index             # the prompt is never indexed
    bl = 8
    state = torch.full((bl,), MASK, dtype=torch.long)
    fill = torch.ones(bl, dtype=torch.bool)
    open_ = fill.clone()
    # slot 0's context is the prefix's last three tokens (6 7 8): nothing followed them -> no ban
    rows, cols = g.bans(state, open_, fill)
    assert rows == [] and cols == []
    # slots 0..2 hold 2 3 4 (filled earlier in this call): slot 3 may not take 1 or 5; slot 4 is unknown-context
    state[:3] = torch.tensor([2, 3, 4]); open_[:3] = False
    rows, cols = g.bans(state, open_, fill)
    assert list(zip(rows, cols)) == [(3, 1), (3, 5)]
    # a committed slot of the block (not being filled) is indexed: 5 6 7 -> 1 inside the block
    state = torch.tensor([5, 6, 7, 1, MASK, MASK, MASK, MASK])
    fill = torch.tensor([False, False, False, False, True, True, True, True])
    open_ = fill.clone()
    rows, cols = g.bans(state, open_, fill)
    assert (4, 2) not in zip(rows, cols)         # 6 7 1 -> 2 was never seen
    state = torch.tensor([5, 6, 7, 8, 5, 6, 7, MASK])
    fill = torch.tensor([False, False, False, False, True, True, True, True])
    open_ = torch.tensor([False, False, False, False, False, False, False, True])
    rows, cols = g.bans(state, open_, fill)
    assert list(zip(rows, cols)) == [(7, 8)]     # 5 6 7 -> 8 appears in the block's committed slots (and the prefix)
    assert g.n_slots == 2
