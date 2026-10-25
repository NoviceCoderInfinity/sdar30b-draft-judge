"""CPU tests for phase 3: the full judge with a re-draft between judge passes (--full-redraft-every K)
and the judge diagnostic (--judge-diag).

  * K = 0 runs exactly the original loop (same tokens, passes and per-pass records);
  * after a judge pass the drafter re-fills ONLY the [MASK] slots, sees ONLY committed tokens (the
    tentative slots are [MASK] in its input and keep their draft), uses ceil(#[MASK] / 4) steps, and its
    forwards are logged as kind "redraft";
  * every committed token is the judge's argmax; the block always ends on a judge pass;
  * K = 2 waits two passes before re-drafting;
  * the diagnostic counts next-slot copies at flagged and at [MASK] slots.

No real model is loaded; the fakes are tests/test_da_decode.py's.
Run from the repo root:  /venv/p2/bin/python -m pytest -q code/tests/test_da_decode_redraft.py
"""
from __future__ import annotations

import os
import sys

import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign.da_decode import da_generate, drafter_records, DRAFTER_KIND_CODE         # noqa: E402
from tests.test_da_decode import (ScriptedLM, onehot_row, logprob_row, prompt_ids, meter_for,   # noqa: E402
                                  a_tok, g_tok, MASK, EOS_IDS, BL, PLEN, GEN)

F_COMMIT = {52, 55}            # pass 1: p ~ 1, argmax a != D      -> committed
F_KEEP = {53, 56}              # pass 1: argmax == D at p .60        -> tentative (keeps the draft g)
VALID = range(PLEN, 64)        # one decode block: positions 32..63, valid 52..63 (12 slots)


def _has_a(ids):
    """True once any valid slot holds its a-token (a commit or a re-draft happened)."""
    return any(ids[p - 32] == a_tok(p) for p in VALID)


def judge_fn(pos, tok, is_mask, ids, cached, pkv):
    """Pass 1 (block = the first draft g everywhere): commit F_COMMIT, keep F_KEEP, re-mask the rest.
    Later passes: a real token in the slot is confirmed with p ~ 1; a [MASK] slot gets a at p .60, i.e. it
    stays re-masked (the slow path the re-draft is meant to replace)."""
    if pos < PLEN:
        return onehot_row(tok)
    if not _has_a(ids):
        if pos in F_COMMIT:
            return onehot_row(a_tok(pos))
        if pos in F_KEEP:
            return logprob_row({g_tok(pos): .60, a_tok(pos): .40})
        return logprob_row({a_tok(pos): .60, g_tok(pos): .05, 0: .35})
    if is_mask:
        return logprob_row({a_tok(pos): .60, g_tok(pos): .05, 0: .35})
    return onehot_row(tok)


def drafter_fn(pos, tok, is_mask, ids, cached, pkv):
    """First draft: g everywhere.  A re-draft (committed a-tokens visible) proposes the judge's a."""
    if not is_mask:
        return onehot_row(tok)
    seen = any(ids[p - 32] == a_tok(p) for p in F_COMMIT)
    return onehot_row(a_tok(pos) if seen else g_tok(pos))


def run(**kw):
    moe = meter_for(judge_fn)
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, **kw)
    return moe, dr, res


def test_k0_is_the_original_loop():
    _, _, a = run()
    _, _, b = run(full_redraft_every=0)
    assert torch.equal(a.x, b.x)
    assert [blk.full_passes for blk in a.blocks] == [blk.full_passes for blk in b.blocks]
    assert all("redraft" not in p for blk in b.blocks for p in blk.full_passes)
    assert b.blocks[0].n_redraft_rounds == 0 and b.full_redraft_every == 0


def test_k1_refills_only_mask_slots_from_committed_tokens():
    moe, dr, res = run(full_redraft_every=1)
    b = res.blocks[0]
    # pass 1: 2 commits, 2 tentative, 8 re-masked -> one re-draft of the 8 holes in ceil(8 / 4) = 2 steps
    assert b.full_passes[0] == {"commit": 2, "tentative": 2, "remask": 8, "forced": False, "redraft": 8}
    assert b.n_redraft_rounds == 1 and b.n_redraft_slots == 8 and b.n_redraft_fwd == 2
    assert b.n_redraft_eq_judge == 8 and b.n_redraft_eq_old == 0
    kinds = [c for c, _ in drafter_records(dr)]
    assert kinds.count(DRAFTER_KIND_CODE["redraft"]) == 2
    # the drafter's re-draft input: committed a-tokens visible, tentative slots hidden as [MASK]
    re_calls = [c["input_ids"][0] for c in dr.model.calls
                if any(int(c["input_ids"][0][p - 32]) == a_tok(p) for p in F_COMMIT) and not c["store_kv"]]
    assert re_calls, "no re-draft call seen"
    for ids in re_calls:
        for p in F_KEEP:
            assert int(ids[p - 32]) == MASK, "the drafter saw a tentative token"
        for p in F_COMMIT:
            assert int(ids[p - 32]) == a_tok(p)
    # pass 2: kept drafts g at F_KEEP, re-drafts a elsewhere, no [MASK]; everything is confirmed -> 2 passes
    judge_calls = [c["input_ids"][0] for c in moe.seq.model.calls if not c["store_kv"] and c["cached"] == 32]
    second = judge_calls[1]
    for p in VALID:
        want = g_tok(p) if p in F_KEEP else a_tok(p)
        assert int(second[p - 32]) == want
    assert len(b.full_passes) == 2 and b.full_passes[1]["commit"] == 10
    for p in VALID:                                   # every committed token is the judge's argmax
        assert int(b.tokens[p - 32]) == (g_tok(p) if p in F_KEEP else a_tok(p))


def test_k2_waits_two_passes_and_beats_k0_on_passes():
    _, _, r2 = run(full_redraft_every=2)
    b = r2.blocks[0]
    assert b.full_passes[0]["redraft"] == 0 and b.full_passes[1]["redraft"] == 8
    assert b.n_redraft_rounds == 1 and len(b.full_passes) == 3
    _, _, r0 = run()
    assert len(r0.blocks[0].full_passes) > len(b.full_passes)      # the masks alone take forced passes
    assert torch.equal(r0.x, r2.x)                                  # same final tokens in this script


def copy_judge_fn(pos, tok, is_mask, ids, cached, pkv):
    """Pass 1: every valid slot outputs the NEXT slot's token with p ~ 1 (the untrained copy)."""
    if pos < PLEN:
        return onehot_row(tok)
    i = pos - 32
    nxt = ids[i + 1] if i + 1 < len(ids) else tok
    return onehot_row(nxt if nxt != MASK else tok)


def test_judge_diag_counts_next_slot_copies():
    moe = meter_for(copy_judge_fn)
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, judge_diag=True)
    b = res.blocks[0]
    # pass 1: 11 flagged slots (52..62) have a different next token; all 11 copy it; slot 63 has no next slot
    assert (b.copy_flag_hit, b.copy_flag_n) == (11, 11)
    assert b.copy_mask_n == 0 and res.judge_diag is True
