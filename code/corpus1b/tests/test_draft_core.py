"""draft_core.fill_block (the stepper the SGLang drafter plugin c1b_judge_draft.py runs) == the harness's
draftalign/drafter.py::draft_block, bit for bit: tokens, p_commit, n_fwd, raw_mask_argmax, guard counters, and the
very same drafter forwards.  Three cases on the CPU fakes of code/tests/test_da_decode.py:

  a  first draft: every valid slot of the first decode block masked (5-slot prompt tail fixed), s8;
  b  re-draft: committed tokens visible, tentative slots masked but NOT filled, holes filled in min(8, ceil(holes/4))
     steps, the old draft banned at the holes;
  c  the phase-5 loop guard (n = 12) on the cycle drafter of code/tests/test_da_decode_loopguard.py.

    python3 -m pytest -q code/corpus1b/tests/test_draft_core.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign.drafter import draft_block                                                 # noqa: E402
from draftalign.loop_guard import LoopGuard                                                # noqa: E402
from sdar_harness.decode import get_num_transfer_tokens as harness_nt                      # noqa: E402
from corpus1b.sglang_plugin.draft_core import DraftFill, fill_block, get_num_transfer_tokens   # noqa: E402
from tests.test_da_decode import ScriptedLM, seeded_fn, prompt_ids, MASK, BL               # noqa: E402
from tests.test_da_decode_loopguard import cycle_drafter_fn, cyc, alt, N as GUARD_N        # noqa: E402


# --------------------------------------------------------------------------
# cases (shared with test_judge_draft_plugin.py)
# --------------------------------------------------------------------------
def masky_fn(seed):
    """seeded_fn, but the raw argmax is [MASK] at masked slots with pos % 7 == 0 (raw_mask_argmax counts them)."""
    base = seeded_fn(seed)

    def fn(pos, tok, is_mask, ids, cached, pkv):
        row = base(pos, tok, is_mask, ids, cached, pkv)
        if is_mask and pos % 7 == 0:
            row[MASK] = 40.0
        return row
    return fn


def drafter_seq(fn, prefix: torch.Tensor):
    """A drafter SeqCache over ScriptedLM(fn) with the full blocks of `prefix` [1, L] prefilled; returns (seq, lo)."""
    seq = SeqCache(ScriptedLM(fn), BL, "drafter")
    return seq, seq.prefill(prefix)


def first_draft_case(seed: int, plen: int, steps: int = 8):
    ids = prompt_ids(plen)
    lo = (plen // BL) * BL
    block = torch.full((BL,), MASK, dtype=torch.long)
    block[:plen - lo] = ids[0, lo:plen]
    return SimpleNamespace(name=f"first{seed}", fn=masky_fn(seed), prefix=ids, lo=lo, block=block,
                           fill=block == MASK, steps=steps, ban=None, guard=None)


def redraft_case(seed: int = 1, plen: int = 37):
    """Slots of the valid part: %3 == 0 committed (a token unlike the old draft), %3 == 1 tentative (hidden as
    [MASK], not filled), %3 == 2 holes (filled, the old draft D banned there), as da_decode.judge_full_block."""
    a = first_draft_case(seed, plen)
    seq, lo = drafter_seq(a.fn, a.prefix)
    D = draft_block(seq, a.block.clone(), a.fill, lo, "s8", MASK)["tokens"]
    slot = torch.arange(BL)
    valid = a.fill
    committed, tent, holes = valid & (slot % 3 == 0), valid & (slot % 3 == 1), valid & (slot % 3 == 2)
    T = (torch.arange(lo, lo + BL) * 11 + 5) % 60
    inp = torch.where(committed, T, a.block)
    inp = torch.where(tent | holes, torch.full_like(inp, MASK), inp)
    n_h = int(holes.sum())
    ban = torch.where(holes, D, torch.full_like(D, -1))
    return SimpleNamespace(name="redraft", fn=a.fn, prefix=a.prefix, lo=lo, block=inp, fill=holes,
                           steps=min(8, -(-n_h // 4)), ban=ban, guard=None, old=D, committed=committed, tent=tent)


GUARD_PLEN = 32


def guard_case():
    """Block 2 (lo = 64) of the loop-guard test: prefix = prompt + the cycle in block 1, every slot masked."""
    prefix = torch.cat([prompt_ids(GUARD_PLEN), torch.tensor([[cyc(p) for p in range(32, 64)]])], dim=1)
    guard = {"n": GUARD_N, "prompt_length": GUARD_PLEN, "prefix_ids": prefix[0].tolist()}
    block = torch.full((BL,), MASK, dtype=torch.long)
    return SimpleNamespace(name="guard", fn=cycle_drafter_fn, prefix=prefix, lo=64, block=block, fill=block == MASK,
                           steps=8, ban=None, guard=guard)


def make_guard(case):
    if case.guard is None:
        return None
    g = LoopGuard(case.guard["n"], MASK)
    g.start_item(case.guard["prompt_length"])
    g.begin_block(case.guard["prefix_ids"], case.lo)
    return g


def reference(case, static_steps="case"):
    """draft_block on a fresh drafter cache; returns (result, seq, guard)."""
    seq, lo = drafter_seq(case.fn, case.prefix)
    assert lo == case.lo
    guard = make_guard(case)
    kw = {} if static_steps is None else {"static_steps": case.steps}
    res = draft_block(seq, case.block.clone(), case.fill, lo, "s8", MASK, ban=case.ban, guard=guard, **kw)
    return res, seq, guard


CASES = {"first": lambda: first_draft_case(0, 37), "redraft": redraft_case, "guard": guard_case}


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_num_transfer_tokens_is_the_harness_schedule():
    for n in range(0, 41):
        for steps in range(1, 9):
            assert torch.equal(get_num_transfer_tokens(n, steps), harness_nt(n, steps))


@pytest.mark.parametrize("name", sorted(CASES))
def test_fill_block_equals_draft_block(name):
    case = CASES[name]()
    ref, seq_ref, g_ref = reference(case)
    seq, lo = drafter_seq(case.fn, case.prefix)
    guard = make_guard(case)
    got = fill_block(case.block.clone(), case.fill, case.steps,
                     lambda ids: seq.block_forward(ids.unsqueeze(0), lo, store_kv=False, kind="draft")[0],
                     MASK, ban=case.ban, guard=guard)
    assert torch.equal(got["tokens"], ref["tokens"]) and torch.equal(got["p_commit"], ref["p_commit"])
    for k in ("n_fwd", "raw_mask_argmax", "guard_slots", "guard_changed"):
        assert got[k] == ref[k], k
    # the very same drafter forwards, in the same order
    a, b = seq_ref.model.calls, seq.model.calls
    assert len(a) == len(b) == got["n_fwd"] + 1                  # + the prefill
    assert all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, b))
    if guard is not None:
        assert (guard.n_slots, guard.n_changed) == (g_ref.n_slots, g_ref.n_changed)
    # each case exercises what it is there for
    tok = got["tokens"]
    if name == "first":
        assert got["n_fwd"] == 8 and got["raw_mask_argmax"] > 0 and not bool((tok == MASK).any())
        assert torch.equal(tok[~case.fill], case.block[~case.fill])
    elif name == "redraft":
        assert got["n_fwd"] == case.steps == 3
        assert bool((tok[case.tent] == MASK).all())              # tentative slots stay hidden
        assert torch.equal(tok[case.committed], case.block[case.committed])
        assert not bool((tok[case.fill] == case.old[case.fill]).any())   # the ban held at every hole
        assert not bool((tok[case.fill] == MASK).any())
        assert bool((got["p_commit"][~case.fill] == 0).all())
    else:
        changed = [s for s in range(BL) if int(tok[s]) != cyc(64 + s)]
        assert changed == [0, 12, 24] and all(int(tok[s]) == alt(64 + s) for s in changed)
        assert got["guard_changed"] == 3 and got["guard_slots"] >= 3


def test_ban_pairs_equal_ban_tensor_and_bf16_logits():
    """The plugin passes the ban as (slot, token) pairs; the stepper also takes bf16 logits (upcast like draft_block)."""
    case = redraft_case()
    pairs = [(int(s), int(case.ban[s])) for s in range(BL) if int(case.ban[s]) >= 0]
    out = []
    for ban in (case.ban, pairs):
        seq, lo = drafter_seq(case.fn, case.prefix)
        out.append(fill_block(case.block.clone(), case.fill, case.steps,
                              lambda ids: seq.block_forward(ids.unsqueeze(0), lo, store_kv=False, kind="d")[0],
                              MASK, ban=ban))
    assert torch.equal(out[0]["tokens"], out[1]["tokens"]) and torch.equal(out[0]["p_commit"], out[1]["p_commit"])
    seq, lo = drafter_seq(case.fn, case.prefix)
    bf = lambda ids: seq.block_forward(ids.unsqueeze(0), lo, store_kv=False, kind="d")[0].to(torch.bfloat16)  # noqa: E731
    seq2, _ = drafter_seq(case.fn, case.prefix)

    class BF16Seq:                                                # draft_block's seq, logits rounded to bf16
        timings = []

        def block_forward(self, ids, lo_, store_kv, kind):
            return seq2.block_forward(ids, lo_, store_kv=store_kv, kind=kind).to(torch.bfloat16)
    ref = draft_block(BF16Seq(), case.block.clone(), case.fill, lo, "s8", MASK, static_steps=case.steps, ban=case.ban)
    got = fill_block(case.block.clone(), case.fill, case.steps, bf, MASK, ban=case.ban)
    assert torch.equal(got["tokens"], ref["tokens"]) and torch.equal(got["p_commit"], ref["p_commit"])


def test_nothing_to_fill_runs_no_forward():
    case = first_draft_case(0, 37)
    calls = []
    got = fill_block(case.block.clone(), torch.zeros(BL, dtype=torch.bool), 8, lambda ids: calls.append(1), MASK)
    assert calls == [] and got["n_fwd"] == 0 and torch.equal(got["tokens"], case.block)
    assert bool((got["p_commit"] == 0).all()) and got["raw_mask_argmax"] == 0
    df = DraftFill(MASK).begin(case.block, torch.zeros(BL, dtype=torch.bool), 8)
    assert df.done and df.n_steps == 0


def test_steps_capped_at_the_fill_size():
    case = first_draft_case(4, 60)                                # lo = 32, 28-slot tail -> 4 open slots
    assert int(case.fill.sum()) == 4
    seq, lo = drafter_seq(case.fn, case.prefix)
    got = fill_block(case.block.clone(), case.fill, 8,
                     lambda ids: seq.block_forward(ids.unsqueeze(0), lo, store_kv=False, kind="d")[0], MASK)
    ref, _, _ = reference(case, static_steps=None)
    assert got["n_fwd"] == ref["n_fwd"] == 4 and torch.equal(got["tokens"], ref["tokens"])
