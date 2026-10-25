"""CPU tests for the two OFF-by-default decode-rule variants of draftalign.da_decode:

  (A) ANCHORED DRAFTING for the blind rule (`draft_after_pass=N`, runner `--draft-after-pass N`,
      cell suffix `_da{N}`): N STOCK passes first, then the drafter fills only the STILL-OPEN slots
      with those commits held fixed as anchors, then the usual blind acceptance on the next stock
      pass.  N = 0 (the default) is today's rule: draft before pass 1.
  (B) the ONE-PASS HYBRID for the full judge (`judge_passes=K`, runner `--judge-passes K`, cell
      suffix `_jp{K}`): K judge passes, then every uncommitted slot is RE-MASKED (no tentative draft
      context survives) and the STOCK loop finishes the block.  K = 0 is today's unbounded loop.

(1) here is the regression guard: at both flags' defaults the stock twin, the blind decoder and the
full judge reproduce, value for value, what tests/test_da_decode.py and tests/test_da_decode_full.py
already expect of them (and their trace NPZ keeps exactly the arrays it had -- `anchored` is written
only by an anchored run).

No real model is loaded; the fakes/fixtures are tests/test_da_decode.py's and test_da_decode_full.py's.
Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_da_decode_anchored.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from sdar_harness.decode import block_diffusion_generate                                   # noqa: E402
from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign import da_decode as DA                                                     # noqa: E402
from draftalign.da_decode import (da_generate, save_trace, load_trace, CODE_ACCEPT_Q,      # noqa: E402
                                  CODE_COMMIT, CODE_FORCED, FULL_MAX_PASSES)

from tests.test_da_decode import (ScriptedLM, seeded_fn, prompt_ids, item, FakeTok, meter_for,   # noqa: E402
                                  a_tok, g_tok, scripted_drafter_fn, blind_judge_fn, blind_drafter_fn,
                                  run_blind, MASK, EOS_IDS, BL, PLEN, GEN)
from tests.test_da_decode_full import full_judge_fn, flat_judge_fn, run_full, F_COMMIT, F_KEEP  # noqa: E402

B_COMMIT_A, B_COMMIT_D, B_ACC, B_EDGE, B_GATE = 52, 53, 54, 55, 56     # tests/test_da_decode.py's blind slots


# --------------------------------------------------------------------------
# runners (the fixtures' own, with the new keyword arguments threaded through)
# --------------------------------------------------------------------------
def run_blind_anchored(judge_fn=None, drafter_fn=blind_drafter_fn, **kw):
    moe = meter_for(judge_fn or blind_judge_fn())
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    ids = prompt_ids(PLEN)
    res = da_generate(moe, dr, ids, mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95, draft="s8",
                      judge="blind", stopping_criteria_idx=EOS_IDS, **{"tau_acc_q": .30, **kw})
    return moe, dr, ids, res


def run_full_blocks(judge_fn=full_judge_fn, gen_length=GEN, **kw):
    moe = meter_for(judge_fn)
    dr = SeqCache(ScriptedLM(scripted_drafter_fn), BL, "drafter")
    ids = prompt_ids(PLEN)
    res = da_generate(moe, dr, ids, mask_id=MASK, gen_length=gen_length, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, **kw)
    return moe, dr, ids, res


# wall-clock / ms fields of a record (never reproducible); everything else must match exactly
TIMED = {"seconds", "moe_seconds", "drafter_seconds", "ms_by_kind", "drafter_ms_by_kind",
         "forwards", "drafter_forwards", "peak_mem_gib"}


def stable(rec: dict) -> dict:
    """The record without its timings; the forward records keep everything but their `ms`."""
    out = {k: v for k, v in rec.items() if k not in TIMED}
    out["forwards_no_ms"] = [[r[0]] + list(r[2:]) for r in rec["forwards"]]
    out["drafter_forwards_no_ms"] = [[r[0]] for r in rec["drafter_forwards"]]
    return out


def same_calls(m1, m2):
    assert len(m1.seq.model.calls) == len(m2.seq.model.calls)
    for c1, c2 in zip(m1.seq.model.calls, m2.seq.model.calls):
        assert torch.equal(c1["input_ids"], c2["input_ids"]) and torch.equal(c1["pos"], c2["pos"])
        assert torch.equal(c1["mask"], c2["mask"]) and c1["store_kv"] == c2["store_kv"] and c1["cached"] == c2["cached"]


# ==========================================================================
# (1) both flags at their defaults: the existing decoders, unchanged
# ==========================================================================
@pytest.mark.parametrize("seed,plen,gen_length,threshold", [(0, 37, 64, 0.95), (3, 40, 64, 1.0)])
def test_defaults_keep_the_stock_twin_bit_identical(seed, plen, gen_length, threshold):
    """The --stock path still is sdar_harness.decode.block_diffusion_generate, pass for pass."""
    ids = prompt_ids(plen)
    ref_model = ScriptedLM(seeded_fn(seed))
    ref = block_diffusion_generate(ref_model, ids, mask_id=MASK, gen_length=gen_length, block_length=BL,
                                   denoising_steps=BL, temperature=1.0, top_k=1, top_p=1.0, threshold=threshold,
                                   stopping_criteria_idx=EOS_IDS, greedy_exact=True)
    moe = meter_for(seeded_fn(seed))
    dv = da_generate(moe, None, ids, mask_id=MASK, gen_length=gen_length, block_length=BL, denoising_steps=BL,
                     threshold=threshold, draft=None, stopping_criteria_idx=EOS_IDS,
                     draft_after_pass=0, judge_passes=0)
    assert torch.equal(ref.x, dv.x) and ref.n_forward == dv.n_forward == len(moe.records)
    assert [b.n_steps for b in ref.blocks] == [b.n_steps for b in dv.blocks]
    for a, b in zip(ref.blocks, dv.blocks):
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.valid, b.valid)
        for s in range(a.n_steps):
            assert torch.equal(a.committed[s], b.committed[s]) and torch.equal(a.proposed[s], b.proposed[s])
            assert torch.equal(a.confidence[s], b.confidence[s])
    assert len(ref_model.calls) == len(moe.seq.model.calls)
    for c1, c2 in zip(ref_model.calls, moe.seq.model.calls):
        assert torch.equal(c1["input_ids"], c2["input_ids"]) and c1["store_kv"] == c2["store_kv"]
    assert dv.draft_after_pass == 0 and dv.judge_passes == 0
    assert all(b.anchored is None and b.n_anchor == 0 for b in dv.blocks)


def test_defaults_reproduce_the_blind_decoder(tmp_path):
    """tests/test_da_decode.py::test_blind_reads_q_from_the_stock_pass_and_never_overrides_commits,
    value for value -- and passing the new flags explicitly at 0 changes nothing at all."""
    moe, dr, ids, res = run_blind(blind_judge_fn())                          # the fixture's own runner
    b = res.blocks[0]
    assert b.n_judge == 0 and b.judge_mode == "blind" and res.judge == "blind"
    assert [int(i) + 32 for i in torch.nonzero(b.committed[0]).flatten()] == [B_COMMIT_A, B_COMMIT_D]
    assert b.tokens[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) and b.tokens[B_COMMIT_D - 32] == g_tok(B_COMMIT_D)
    assert b.judge_code[B_ACC - 32] == CODE_ACCEPT_Q and b.judge_code[B_GATE - 32] == CODE_ACCEPT_Q
    assert b.judge_code[B_EDGE - 32] == 0 and b.judge_code[B_COMMIT_A - 32] == 0
    assert b.n_blind_accept == 2 and int(b.accepted.sum()) == 2 and not b.forced
    assert b.n_steps == 9 and b.n_forward == 10 and [r[0] for r in moe.records] == [0] + [1] * 9 + [2]
    # the draft still happens BEFORE pass 1: every valid slot is drafted, nothing is anchored
    assert b.anchored is None and b.n_anchor == 0
    assert (moe.seq.model.calls[1]["input_ids"][0][b.valid] == MASK).all()    # pass 1 sees the all-mask block
    assert (dr.model.calls[1]["input_ids"][0][b.valid] == MASK).all()         # the drafter saw 12 open slots

    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["judge"] == "blind" and rec["n_judge"] == 0 and rec["n_denoise"] == 9
    assert rec["blind_accept_per_block"] == [2] and rec["accepted_per_block"] == [2]
    assert rec["passes_per_block"] == [9] and rec["stock_passes_per_block"] == [9]
    # the new fields, at their defaults, on an old-path record
    assert rec["draft_after_pass"] == 0 and rec["judge_passes"] == 0 and rec["anchor_per_block"] == [0]

    # explicit zeros == the defaults: same record, same forwards, same trace arrays (no `anchored` key)
    moe2, dr2, ids2, res2 = run_blind_anchored(draft_after_pass=0, judge_passes=0)
    rec2 = DA.make_record(res2, FakeTok(), item(), ids2, moe2, dr2, EOS_IDS, MASK, 0.0)
    assert stable(rec) == stable(rec2)
    same_calls(moe, moe2)
    p1, p2 = str(tmp_path / "a" / "0.npz"), str(tmp_path / "b" / "0.npz")
    save_trace(p1, res, BL); save_trace(p2, res2, BL)
    t1, t2 = load_trace(p1), load_trace(p2)
    assert set(t1) == set(t2) and "anchored" not in t1
    for k in t1:
        assert np.array_equal(t1[k], t2[k], equal_nan=np.issubdtype(t1[k].dtype, np.floating))


def test_defaults_reproduce_the_full_judge():
    """tests/test_da_decode_full.py::test_full_judge_reads_every_open_slot_... , value for value."""
    moe, dr, ids, res = run_full()                                           # the fixture's own runner
    b = res.blocks[0]
    assert b.full_passes[0] == {"commit": 2, "tentative": 2, "remask": 8, "forced": False}
    assert b.n_tentative == 2 and b.n_remask == 8 and b.full_commit_eq_draft == 0 and b.full_fallback == 0
    assert b.n_judge == 2 and b.n_steps == 0 and b.n_forward == 3
    assert [r[0] for r in moe.records] == [0, 3, 3, 2]
    second = moe.seq.model.calls[2]["input_ids"][0]
    for p in F_KEEP:                                                         # tentative context survives
        assert second[p - 32] == g_tok(p)
    assert int((second[b.valid] == MASK).sum()) == 8
    assert int(b.committed_judge.sum()) == 12 and (b.judge_code[b.valid] == CODE_COMMIT).all()

    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["n_judge"] == 2 and rec["n_denoise"] == 0 and rec["passes_per_block"] == [2]
    assert rec["tentative_per_block"] == [2] and rec["remask_per_block"] == [8] and rec["full_fallback_per_block"] == [0]
    assert rec["draft_after_pass"] == 0 and rec["judge_passes"] == 0 and rec["anchor_per_block"] == [0]

    moe2, dr2, ids2, res2 = run_full_blocks(judge_passes=0, draft_after_pass=0)
    assert stable(rec) == stable(DA.make_record(res2, FakeTok(), item(), ids2, moe2, dr2, EOS_IDS, MASK, 0.0))
    same_calls(moe, moe2)


# ==========================================================================
# (2) A -- anchored drafting
# ==========================================================================
def test_anchored_drafting_drafts_only_the_open_slots_after_the_stock_pass():
    """--draft-after-pass 1 on the blind fixture: stock pass 1 commits 52 and 53, the drafter is then
    called with valid = the 10 still-open slots and those two commits fixed as anchors."""
    moe, dr, ids, res = run_blind_anchored(draft_after_pass=1)
    b = res.blocks[0]
    valid = b.valid
    anchor_pos = [int(i) + 32 for i in torch.nonzero(b.anchored).flatten()]
    open_pos = [p for p in range(PLEN, 64) if p not in anchor_pos]

    # the anchors ARE exactly the slots the stock pass 1 committed
    assert [int(i) + 32 for i in torch.nonzero(b.committed[0]).flatten()] == [B_COMMIT_A, B_COMMIT_D]
    assert anchor_pos == [B_COMMIT_A, B_COMMIT_D] and b.n_anchor == 2
    assert bool((b.anchored & ~valid).sum() == 0)                      # never the prompt tail

    # pass 1 is the STOCK pass on the all-mask block (identical to the stock decoder's first pass)
    first = moe.seq.model.calls[1]["input_ids"][0]
    assert (first[valid] == MASK).all() and torch.equal(first[~valid], ids[0, 32:PLEN])
    assert [r[0] for r in moe.records][:2] == [0, 1]                    # prefill, then a DENOISE pass

    # the drafter ran ONCE, after that pass, over the open slots only: anchors hold the committed tokens
    d0 = dr.model.calls[1]["input_ids"][0]                              # calls[0] is the drafter prefill
    assert [int(i) + 32 for i in torch.nonzero(d0 == MASK).flatten()] == open_pos
    assert d0[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) and d0[B_COMMIT_D - 32] == g_tok(B_COMMIT_D)
    assert torch.equal(d0[~valid], ids[0, 32:PLEN])                     # the prompt tail is still fixed
    assert dr.n_by_kind["draft_s8"] == 8 and dr.n_by_kind["prefill"] == 1 and dr.n_by_kind["flush"] == 1
    assert b.n_draft_fwd == 8

    # `draft_tokens` holds the ANCHORED draft: the committed tokens at the anchors, and they are never
    # re-drafted -- the final tokens at the anchor slots are the pass-1 tokens
    assert b.draft_tokens[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) == b.tokens[B_COMMIT_A - 32]
    assert b.draft_tokens[B_COMMIT_D - 32] == g_tok(B_COMMIT_D) == b.tokens[B_COMMIT_D - 32]
    assert torch.equal(b.draft_tokens[b.anchored], b.tokens[b.anchored])
    assert torch.equal(b.draft_tokens[~valid], ids[0, 32:PLEN])

    # the blind acceptance rule then runs on the NEXT stock pass (pass 2), exactly as it does today
    assert b.judge_code[B_ACC - 32] == CODE_ACCEPT_Q and b.judge_code[B_GATE - 32] == CODE_ACCEPT_Q
    assert b.judge_code[B_EDGE - 32] == 0 and b.n_blind_accept == 2 and int(b.accepted.sum()) == 2
    assert b.tokens[B_ACC - 32] == g_tok(B_ACC) and b.tokens[B_GATE - 32] == g_tok(B_GATE)
    assert b.judge_argmax[B_COMMIT_A - 32] == -1 and torch.isnan(b.judge_q[B_COMMIT_A - 32])   # not read again

    # counters: the stock pass-1 forward is a denoise pass, no judge forward exists at all
    assert b.n_judge == 0 and b.n_rejudge == 0 and b.n_steps == 9 and b.n_forward == 10
    assert [r[0] for r in moe.records] == [0] + [1] * 9 + [2]
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["draft_after_pass"] == 1 and rec["anchor_per_block"] == [2] and rec["judge_passes"] == 0
    assert rec["n_denoise"] == 9 and rec["n_judge"] == 0 and rec["stock_passes_per_block"] == [9]
    assert rec["drafter_forwards_per_block"] == [8] and rec["blind_accept_per_block"] == [2]


def test_anchored_drafting_differs_from_the_default_only_in_the_draft():
    """The same fixture at draft_after_pass=0 drafts all 12 valid slots before any MoE pass."""
    _, dr0, _, res0 = run_blind_anchored(draft_after_pass=0)
    _, dr1, _, res1 = run_blind_anchored(draft_after_pass=1)
    b0, b1 = res0.blocks[0], res1.blocks[0]
    assert int((dr0.model.calls[1]["input_ids"][0] == MASK).sum()) == 12      # every valid slot drafted
    assert int((dr1.model.calls[1]["input_ids"][0] == MASK).sum()) == 10      # the two anchors are fixed
    assert b0.anchored is None and b0.n_anchor == 0 and b1.n_anchor == 2
    assert res0.draft_after_pass == 0 and res1.draft_after_pass == 1


def test_anchored_drafting_trace_and_two_stock_passes():
    """--draft-after-pass 2 anchors both stock passes' commits, and the NPZ carries `anchored`."""
    import tempfile
    moe, dr, ids, res = run_blind_anchored(draft_after_pass=2)
    b = res.blocks[0]
    committed_first_two = (b.committed[0] | b.committed[1])
    assert torch.equal(b.anchored, committed_first_two & b.valid) and b.n_anchor == int(committed_first_two.sum())
    assert b.n_anchor >= 2 and b.n_judge == 0
    d0 = dr.model.calls[1]["input_ids"][0]
    assert not bool((d0[b.anchored] == MASK).any())                          # anchors are never re-drafted
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "traces", "00007.npz")
        save_trace(p, res, BL)
        tr = load_trace(p)
    assert "anchored" in tr and tr["anchored"].shape == (1, BL) and tr["anchored"].dtype == bool
    assert np.array_equal(tr["anchored"][0], b.anchored.numpy())
    assert np.array_equal(tr["draft_tokens"][0], b.draft_tokens.numpy())


def test_anchored_drafting_is_blind_only():
    with pytest.raises(NotImplementedError):
        run_full_blocks(draft_after_pass=1)
    with pytest.raises(NotImplementedError):                                  # halves
        moe = meter_for(blind_judge_fn())
        dr = SeqCache(ScriptedLM(blind_drafter_fn), BL, "drafter")
        da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, draft="s8",
                    judge="halves", draft_after_pass=1, stopping_criteria_idx=EOS_IDS)
    with pytest.raises(NotImplementedError):                                  # stock (no drafter)
        da_generate(meter_for(seeded_fn(0)), None, prompt_ids(37), mask_id=MASK, gen_length=64,
                    block_length=BL, draft=None, draft_after_pass=1, stopping_criteria_idx=EOS_IDS)


# ==========================================================================
# (3) B -- the one-pass hybrid
# ==========================================================================
def test_judge_passes_one_runs_one_judge_forward_then_the_stock_loop():
    """--judge-passes 1: one judge pass with today's commit rule, then every uncommitted slot --
    the TENTATIVE ones included -- is [MASK] and the stock decoder finishes the block."""
    moe, dr, ids, res = run_full_blocks(judge_passes=1)
    b = res.blocks[0]
    # exactly one judge forward, and it is the same first pass as the unbounded loop's
    assert b.n_judge == 1 and len(b.full_passes) == 1
    assert b.full_passes[0] == {"commit": 2, "tentative": 2, "remask": 8, "forced": False}
    assert [r[0] for r in moe.records] == [0, 3, 1, 2]                  # prefill, judge, denoise, kvflush
    judge_in = moe.seq.model.calls[1]["input_ids"][0]
    assert (judge_in != MASK).all() and torch.equal(judge_in[b.valid], b.draft_tokens[b.valid])

    # no tentative slot survives into the stock phase: all 10 uncommitted slots are re-masked
    den0 = moe.seq.model.calls[2]["input_ids"][0]
    assert int((den0[b.valid] == MASK).sum()) == 10 and b.full_fallback == 10
    for p in F_KEEP:                                                    # kept as context by the full loop
        assert den0[p - 32] == MASK
    for p in F_COMMIT:                                                  # the judge's commits stand
        assert den0[p - 32] == a_tok(p)
    assert torch.equal(den0[~b.valid], ids[0, 32:PLEN])

    # counters: n_judge == 1 per block, the stock passes after it are denoise passes
    assert b.n_steps == 1 and b.n_forward == 1 + 1 + 1 and (b.tokens != MASK).all()
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["judge_passes"] == 1 and rec["n_judge"] == 1 and rec["n_denoise"] == 1
    assert rec["full_passes_per_block"] == [[{"commit": 2, "tentative": 2, "remask": 8, "forced": False}]]
    assert rec["stock_passes_per_block"] == [1] and rec["passes_per_block"] == [2]
    assert rec["full_fallback_per_block"] == [10] and rec["draft_after_pass"] == 0
    assert rec["anchor_per_block"] == [0] and all(b2.anchored is None for b2 in res.blocks)


def test_judge_passes_one_judge_forward_per_block():
    """Two decode blocks: exactly one judge forward in each, the rest stock."""
    moe, dr, ids, res = run_full_blocks(gen_length=44, judge_passes=1)
    assert len(res.blocks) == 2
    assert [b.n_judge for b in res.blocks] == [1, 1]
    assert [len(b.full_passes) for b in res.blocks] == [1, 1]
    assert [r[0] for r in moe.records] == [0, 3, 1, 2, 3, 1, 2]
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["n_judge"] == 2 and rec["n_denoise"] == 2 and rec["judge_forwards_per_block"] == [1, 1]
    assert rec["judge_passes"] == 1


def test_judge_passes_two_runs_two_judge_passes_then_the_stock_loop():
    """K = 2 on the fixture where nothing ever passes tau_commit: 2 forced judge commits, then stock."""
    moe, dr, ids, res = run_full_blocks(judge_fn=flat_judge_fn, judge_passes=2)
    b = res.blocks[0]
    assert b.n_judge == 2 and len(b.full_passes) == 2 and all(p["forced"] for p in b.full_passes)
    assert int((b.judge_code == CODE_FORCED).sum()) == 2 and b.full_fallback == 10
    assert [r[0] for r in moe.records] == [0, 3, 3] + [1] * b.n_steps + [2]
    assert b.n_steps == 10 and (b.tokens != MASK).all()
    # K larger than the block never fires: the judge loop finishes on its own, as today
    b_big = run_full_blocks(judge_fn=flat_judge_fn, judge_passes=FULL_MAX_PASSES)[3].blocks[0]
    assert b_big.n_judge == 12 and b_big.n_steps == 0 and b_big.full_fallback == 0
    assert run_full(judge_fn=flat_judge_fn)[3].blocks[0].n_judge == 12          # == the unbounded loop


def test_judge_passes_is_full_only():
    with pytest.raises(NotImplementedError):
        run_blind_anchored(judge_passes=1)
    with pytest.raises(NotImplementedError):
        da_generate(meter_for(seeded_fn(0)), None, prompt_ids(37), mask_id=MASK, gen_length=64,
                    block_length=BL, draft=None, judge_passes=1, stopping_criteria_idx=EOS_IDS)


# ==========================================================================
# (4) the runner: cell names, parser defaults, validation, config
# ==========================================================================
def _ns(**kw):
    return SimpleNamespace(**{"stock": False, "draft": "s8", "threshold": .95, "tau_commit": .95, "tau_acc": .70,
                              "tau_acc_q": .50, "rejudge": False, "rejudge_k": 8, "judge": "halves",
                              "tau_draft": 0.0, "pool": 0, "pool_rule": "full", "pool_flush": False,
                              "adapter": None, "full_remask_only": False, "tau_override": None,
                              "full_max_passes": FULL_MAX_PASSES, "drafter": "d32",
                              "draft_after_pass": 0, "judge_passes": 0, **kw})


def test_cell_name_carries_the_new_flags_only_when_they_are_set():
    import draftalign.run_da_decode as R
    # unchanged for every existing flag combination
    assert R.cell_name("gsm8k", _ns(stock=True)) == "gsm8k_B32_t95_stock"
    assert R.cell_name("gsm8k", _ns()) == "gsm8k_B32_t95_dv-s8_c95_a70_q50"
    assert R.cell_name("gsm8k", _ns(judge="blind", tau_acc_q=.50)) == "gsm8k_B32_t95_dv-s8_blind_q50"
    assert R.cell_name("gsm8k", _ns(judge="full", tau_commit=.90, tau_override=.99)) == \
        "gsm8k_B32_t95_dv-s8_full_c90_o99"
    # ... and the same names when the flags are left at 0, even if the attribute is missing entirely
    bare = SimpleNamespace(**{k: v for k, v in vars(_ns(judge="blind", tau_acc_q=.50)).items()
                              if k not in ("draft_after_pass", "judge_passes")})
    assert R.cell_name("gsm8k", bare) == "gsm8k_B32_t95_dv-s8_blind_q50"
    # the new suffixes
    assert R.cell_name("gsm8k", _ns(judge="blind", tau_acc_q=.50, draft_after_pass=1)) == \
        "gsm8k_B32_t95_dv-s8_blind_q50_da1"
    assert R.cell_name("gsm8k", _ns(judge="blind", tau_acc_q=.30, draft_after_pass=2, rejudge=True)) == \
        "gsm8k_B32_t95_dv-s8_blind_q30_rj_da2"
    assert R.cell_name("gsm8k", _ns(judge="full", tau_commit=.90, tau_override=.99, judge_passes=1)) == \
        "gsm8k_B32_t95_dv-s8_full_c90_o99_jp1"
    # combinable with the pool / an adapter, and never mixed with an existing cell's name
    assert R.cell_name("gsm8k", _ns(judge="full", tau_commit=.90, judge_passes=1, pool=32, pool_rule="des",
                                    adapter="x"), "deadbeefcafe") == \
        "gsm8k_B32_t95_dv-s8_full_c90_jp1_p32des_addeadbeef"
    assert "." not in R.cell_name("gsm8k", _ns(judge="blind", draft_after_pass=1, threshold=.9))


def test_parser_defaults_and_validation():
    import draftalign.run_da_decode as R
    a = R.resolve_args(R.build_parser().parse_args([]))
    assert a.draft_after_pass == 0 and a.judge_passes == 0
    b = R.resolve_args(R.build_parser().parse_args(["--judge", "blind", "--draft-after-pass", "1"]))
    assert b.draft_after_pass == 1 and b.judge_passes == 0 and b.judge == "blind"
    c = R.resolve_args(R.build_parser().parse_args(["--judge", "full", "--judge-passes", "1",
                                                    "--tau-commit", ".90", "--tau-override", ".99"]))
    assert c.judge_passes == 1 and c.draft_after_pass == 0
    with pytest.raises(NotImplementedError):
        R.resolve_args(R.build_parser().parse_args(["--judge", "full", "--draft-after-pass", "1"]))
    with pytest.raises(NotImplementedError):
        R.resolve_args(R.build_parser().parse_args(["--judge", "blind", "--judge-passes", "1"]))
    with pytest.raises(AssertionError):
        R.resolve_args(R.build_parser().parse_args(["--stock", "--draft-after-pass", "1"]))
    with pytest.raises(AssertionError):
        R.resolve_args(R.build_parser().parse_args(["--judge", "blind", "--draft-after-pass", "-1"]))
