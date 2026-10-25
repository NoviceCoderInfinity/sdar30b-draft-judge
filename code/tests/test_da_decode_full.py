"""CPU tests for the three additions to draftalign.da_decode:

  (i)   judge="full" -- the trained regime: one draft-filled MoE forward per pass, read at EVERY open
        slot, commit > tau, keep a_i == D_i as tentative context else re-mask, repeat until every valid
        slot is committed; >= 1 commit per pass; a pass cap that falls back to the STOCK rule;
        --full-remask-only (never keep a tentative draft token).
  (ii)  the per-forward expert pool (PoolCtrl / enable_pool_moe): bit-identical at M >= E, every block
        row restricted to the pool at small M, the "des" rule == sdar_harness.desvote's Algorithm-3
        vote (and the whole restricted forward == desvote's), the prefill never restricted, the kv
        flush restricted only with pool_flush.
  (iii) the adapter path (load_adapter + merge_lora) reproducing the un-merged LoRA / GateFP32 model.
  (iv)  the --stock twin still bit-identical to sdar_harness.decode.block_diffusion_generate through
        the new code path (with a PoolCtrl attached but no pool, and no adapter).

No real model is loaded; the fakes are tests/test_da_decode.py's and tests/test_draftalign.py's.
Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_da_decode_full.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from sdar_harness.decode import block_diffusion_generate                                   # noqa: E402
from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign import da_decode as DA                                                     # noqa: E402
from draftalign.da_decode import (MoEMeter, da_generate, decode_item, save_trace, load_trace,   # noqa: E402
                                  POOL, PoolCtrl, build_pool, full_commit_rule, enable_pool_moe,
                                  disable_pool_moe, CODE_MASK, CODE_COMMIT, CODE_FORCED)

from tests.test_da_decode import (ScriptedLM, seeded_fn, onehot_row, logprob_row, prompt_ids,   # noqa: E402
                                  item, FakeTok, meter_for, a_tok, g_tok, scripted_drafter_fn,
                                  V, MASK, EOS_IDS, BL, PLEN, GEN)
from tests.test_desvote import E, K, H, L, SEED                                            # noqa: E402
from tests.test_draftalign import FakeMoEModel                                             # noqa: E402


@pytest.fixture(autouse=True)
def _clean_pool():
    """The pool controller is a module global (one decode per GPU): reset it around every test."""
    POOL.configure(None, "full"); POOL.valid = None; POOL.enabled = False
    POOL.force_restrict = False; POOL.sizes = []
    yield
    POOL.configure(None, "full"); POOL.valid = None; POOL.enabled = False
    POOL.force_restrict = False; POOL.sizes = []


# ==========================================================================
# (i) the FULL judge
# ==========================================================================
# one decode block: positions 32..63, valid 52..63 (PLEN = 52, GEN = 12)
F_COMMIT = {52, 55}            # p ~ .97 at a != D            -> committed at pass 1
F_KEEP = {53, 56}              # argmax == D at p .60         -> tentative, keeps the draft token
# every other valid slot: argmax = a != D at p .60            -> re-masked


def full_judge_fn(pos, tok, is_mask, ids, cached, pkv):
    """Pass 1 (the block holds the draft everywhere, no masks) scripts commit / keep / remask;
    from pass 2 on (some slot is [MASK]) every open slot is confident, so the block finishes."""
    n_mask = sum(1 for t in ids if t == MASK)
    if n_mask > 0:
        return onehot_row(a_tok(pos))                                    # pass 2+: commit everything
    if pos in F_COMMIT:
        return onehot_row(a_tok(pos))                                    # p ~ 1 > .95 -> commit a
    if pos in F_KEEP:
        return logprob_row({g_tok(pos): .60, a_tok(pos): .40})           # a == D, p < .95 -> tentative
    return logprob_row({a_tok(pos): .60, g_tok(pos): .05, 0: .35})       # a != D, p < .95 -> re-mask


def flat_judge_fn(pos, tok, is_mask, ids, cached, pkv):
    """No slot ever passes tau_commit; a != D everywhere; p rises with the position (best = 63).
    The prompt tail (pos < PLEN) echoes itself, as the prefill and the fixed slots require."""
    if pos < PLEN:
        return onehot_row(tok)
    r = .50 + .01 * (pos - PLEN)
    return logprob_row({a_tok(pos): r, g_tok(pos): .05, 0: 1 - r - .05})


def run_full(judge_fn=full_judge_fn, drafter_fn=scripted_drafter_fn, **kw):
    moe = meter_for(judge_fn)
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    ids = prompt_ids(PLEN)
    res = da_generate(moe, dr, ids, mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, **kw)
    return moe, dr, ids, res


def test_full_judge_reads_every_open_slot_and_applies_the_tentative_remask_rule():
    moe, dr, ids, res = run_full()
    b = res.blocks[0]
    pos = torch.arange(32, 64)
    valid = pos >= PLEN
    assert b.judge_mode == "full" and res.judge == "full" and int(valid.sum()) == 12

    # pass 1: the whole block holds the draft, NO masks anywhere; the judge reads at every valid slot
    calls = moe.seq.model.calls
    first = calls[1]["input_ids"][0]
    assert (first != MASK).all() and torch.equal(first[valid], b.draft_tokens[valid])
    assert torch.equal(first[~valid], ids[0, 32:PLEN]) and calls[1]["cached"] == 32
    assert (b.judge_argmax[valid] >= 0).all() and (b.judge_argmax[~valid] == -1).all()
    assert torch.isnan(b.judge_p[~valid]).all() and torch.isnan(b.judge_q[~valid]).all()
    for p in F_KEEP:
        assert b.judge_argmax[p - 32] == g_tok(p) and abs(float(b.judge_q[p - 32]) - .60) < 1e-4
    for p in F_COMMIT:
        assert b.judge_argmax[p - 32] == a_tok(p) and float(b.judge_p[p - 32]) > .99

    # the rule: 2 commits, 2 tentative (kept the draft), 8 re-masked
    assert b.full_passes[0] == {"commit": 2, "tentative": 2, "remask": 8, "forced": False}
    assert b.n_tentative == 2 and b.n_remask == 8 and b.full_commit_eq_draft == 0 and b.full_fallback == 0

    # pass 2 sees committed tokens / tentative draft tokens / [MASK], and finishes the block
    second = calls[2]["input_ids"][0]
    for p in F_COMMIT:
        assert second[p - 32] == a_tok(p)
    for p in F_KEEP:
        assert second[p - 32] == g_tok(p)
    assert int((second[valid] == MASK).sum()) == 8 and (second[~valid] == ids[0, 32:PLEN]).all()
    assert b.n_judge == 2 and b.n_steps == 0 and b.n_forward == 2 + 1        # 2 judge + the flush
    assert [r[0] for r in moe.records] == [0, 3, 3, 2]                       # prefill, judge, judge, kvflush
    assert (b.tokens != MASK).all() and moe.seq.cached_length() == 64 == dr.cached_length()
    assert torch.equal(calls[-1]["input_ids"][0], b.tokens) and calls[-1]["store_kv"]
    # the tentative slots were still re-decided at pass 2 (nothing is committed by being tentative)
    assert all(int(b.tokens[p - 32]) == a_tok(p) for p in F_KEEP)
    assert all(int(b.tokens[p - 32]) == a_tok(p) for p in F_COMMIT)

    # every valid slot is committed by the judge, none by a stock pass
    assert int(b.committed_judge.sum()) == 12 and int(b.accepted.sum()) == 0 and not b.forced
    assert (b.judge_code[valid] == CODE_COMMIT).all() and (b.judge_code[~valid] == CODE_MASK).all()

    # record + analyzer fields
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0, adapter=None)
    assert rec["judge_mode"] == "full" and rec["judge"] == "full" and rec["n_judge"] == 2 and rec["n_denoise"] == 0
    assert rec["passes_per_block"] == [2] and rec["stock_passes_per_block"] == [0]
    assert rec["tentative_per_block"] == [2] and rec["remask_per_block"] == [8]
    assert rec["full_fallback_per_block"] == [0] and rec["full_commit_eq_draft_per_block"] == [0]
    assert rec["pool_M"] is None and rec["pool_rule"] is None and rec["adapter"] is None


def test_full_judge_remask_only_never_keeps_a_tentative_draft_token():
    moe, dr, ids, res = run_full(full_remask_only=True)
    b = res.blocks[0]
    assert b.n_tentative == 0 and b.n_remask == 10 and b.full_passes[0]["tentative"] == 0
    second = moe.seq.model.calls[2]["input_ids"][0]
    for p in F_KEEP:
        assert second[p - 32] == MASK                       # kept in the default run, re-masked here
    assert int((second[b.valid] == MASK).sum()) == 10 and b.n_judge == 2 and b.n_steps == 0
    # the tokens are unchanged (pass 2 commits a_tok everywhere either way), only the context differs
    assert (b.tokens != MASK).all()


def test_full_judge_progress_guarantee_forces_one_commit_per_pass():
    """Nothing ever passes tau_commit: exactly one slot (the highest p) is committed per pass."""
    moe, dr, ids, res = run_full(judge_fn=flat_judge_fn)
    b = res.blocks[0]
    assert b.n_judge == 12 and b.n_steps == 0 and b.n_forward == 12 + 1
    assert all(p["commit"] == 1 and p["forced"] for p in b.full_passes)
    assert int((b.judge_code == CODE_FORCED).sum()) == 12 and b.forced
    assert torch.equal(b.tokens[b.valid], torch.tensor([a_tok(p) for p in range(PLEN, 64)]))
    # every uncommitted slot was re-masked (a != D), so the judge state degrades to the stock state
    assert b.n_tentative == 0 and b.n_remask == sum(range(12))       # 11 + 10 + ... + 0
    assert [r[0] for r in moe.records] == [0] + [3] * 12 + [2]


def test_full_judge_pass_cap_falls_back_to_the_stock_rule():
    """--full-max-passes 1: one judge pass, then the remaining open slots are [MASK] and the STOCK
    denoising loop finishes the block (stock passes, stock commits, the same flush)."""
    moe, dr, ids, res = run_full(judge_fn=flat_judge_fn, full_max_passes=1)
    b = res.blocks[0]
    assert b.n_judge == 1 and b.full_fallback == 11 and len(b.full_passes) == 1
    assert b.n_steps == 11 and b.n_forward == 1 + 11 + 1
    assert [r[0] for r in moe.records] == [0, 3] + [1] * 11 + [2]
    # the first stock pass saw the judge's one commit and 11 masked slots
    den0 = moe.seq.model.calls[2]["input_ids"][0]
    assert int((den0[b.valid] == MASK).sum()) == 11 and int((b.judge_code == CODE_FORCED).sum()) == 1
    assert (b.tokens != MASK).all()
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["passes_per_block"] == [12] and rec["stock_passes_per_block"] == [11] and rec["n_judge"] == 1
    assert rec["full_fallback_per_block"] == [11]
    # the cap never fires at its default (>= 1 commit per pass, and a block has at most 32 valid slots)
    assert run_full(judge_fn=flat_judge_fn)[3].blocks[0].full_fallback == 0


def test_full_commit_rule_hand_cases():
    bl = 8
    cur = torch.full((bl,), MASK); cur[:2] = torch.tensor([1, 2])
    open_ = torch.tensor([False, False, True, True, True, True, True, True])
    a = torch.tensor([-1, -1, 10, 12, 13, 15, 17, 20])
    D = torch.tensor([1, 2, 11, 12, 14, 16, 17, 20])
    p = torch.tensor([float("nan"), float("nan"), .97, .80, .80, .80, .60, .95])
    new, commit, tent, remask, forced = full_commit_rule(cur, a, p, D, open_, .95, MASK)
    assert commit.tolist() == [False] * 2 + [True] + [False] * 5 and not forced      # .95 is NOT > .95
    assert tent.tolist() == [False, False, False, True, False, False, True, True]    # a == D
    assert remask.tolist() == [False, False, False, False, True, True, False, False]
    assert new.tolist() == [1, 2, 10, 12, MASK, MASK, 17, 20]
    # remask_only
    _, _, t2, r2, _ = full_commit_rule(cur, a, p, D, open_, .95, MASK, remask_only=True)
    assert not t2.any() and r2.tolist() == [False, False, False, True, True, True, True, True]
    # progress guarantee: nothing above tau -> the single best p among the OPEN slots
    p3 = torch.tensor([float("nan"), float("nan"), .50, .40, .61, .30, .20, .10])
    new3, c3, t3, r3, f3 = full_commit_rule(cur, a, p3, D, open_, .95, MASK)
    assert f3 and c3.tolist() == [False] * 4 + [True] + [False] * 3 and new3[4] == 13
    # a high p OUTSIDE the open set is never committed and never touched
    p4 = p3.clone(); p4[0] = .99
    new4, c4, _, _, f4 = full_commit_rule(cur, a, p4, D, open_, .95, MASK)
    assert f4 and not bool(c4[0]) and new4[0] == 1 and bool(c4[4])


def test_full_judge_rejects_rejudge_and_trace_roundtrips(tmp_path):
    with pytest.raises(NotImplementedError):
        run_full(rejudge=True)
    _, _, _, res = run_full()
    path = str(tmp_path / "t.npz")
    save_trace(path, res, BL)
    tr = load_trace(path)
    b = res.blocks[0]
    assert tr["n_pass"].tolist() == [0] and np.array_equal(tr["judge_code"][0], b.judge_code.numpy())
    assert np.array_equal(tr["committed_judge"][0], b.committed_judge.numpy()) and not tr["accepted"].any()
    assert np.array_equal(tr["tokens"][0], b.tokens.numpy())


# ==========================================================================
# (ii) the per-forward expert pool
# ==========================================================================
def pooled_model(n_layers=L, seed=SEED, vocab=V):
    m = FakeMoEModel(vocab=vocab, n_layers=n_layers, seed=seed)
    assert enable_pool_moe(m) == n_layers
    return m


def run_one_block(model, blk_ids, lo=32, plen=40, valid=None, M=None, rule="full", force=False,
                  pool_flush=False, kind="judge", store_kv=False):
    """prefill + one block forward through a MoEMeter with the pool configured; returns (meter, logits, cap)."""
    from sdar_harness.moe_instr import SINK
    POOL.configure(M, rule); POOL.force_restrict = force
    moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL if M is not None else None, pool_flush=pool_flush)
    moe.prefill(prompt_ids(plen))
    moe.set_valid(valid)
    logits = moe.block(blk_ids, lo, store_kv=store_kv, kind=kind)
    return moe, logits


def block_state():
    blk = torch.full((BL,), MASK)
    blk[:8] = torch.arange(1, 9)                    # a prompt tail: fixed, does NOT vote
    valid = torch.arange(BL) >= 8
    return blk, valid


@pytest.mark.parametrize("rule", ["full", "top8", "des"])
@pytest.mark.parametrize("M,force", [(E, False), (E, True), (E + 64, True)])
def test_pool_at_M_ge_num_experts_is_bit_identical(rule, M, force):
    """M >= E (even through the RESTRICTED arithmetic, `force_restrict`): identical logits, identical
    selected experts, identical unique-expert counts."""
    blk, valid = block_state()
    ref_model = pooled_model()
    moe_ref, lg_ref = run_one_block(ref_model, blk, valid=valid, M=None)
    cap_ref = moe_ref.last_cap

    model = pooled_model()
    moe, lg = run_one_block(model, blk, valid=valid, M=M, rule=rule, force=force)
    cap = moe.last_cap

    assert torch.equal(lg_ref, lg)                                           # bit-identical logits
    assert len(cap.ids) == len(cap_ref.ids) == L
    for a, b in zip(cap_ref.ids, cap.ids):
        assert torch.equal(a, b)                                             # the same experts, same order
    assert moe.records[-1][2] == moe_ref.records[-1][2] > 0                   # unique experts
    assert moe.records[-1][5] == E and moe_ref.records[-1][5] == 0
    disable_pool_moe(model); disable_pool_moe(ref_model)


@pytest.mark.parametrize("rule", ["full", "top8", "des"])
def test_small_pool_restricts_every_block_row_and_is_counted(rule):
    """M = 6 (>= top_k = 4): every row of the block forward routes inside the voted pool, the SINK sees
    the RESTRICTED ids, and the recorded pool size is M."""
    from draftalign.moe_probe import RouterCapture
    from sdar_harness.moe_instr import SINK
    M = 6
    blk, valid = block_state()
    model = pooled_model()
    rc = RouterCapture(model, BL)                       # hooks blk.gate: the very same fp32 softmax
    POOL.configure(M, rule)
    moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL)
    moe.prefill(prompt_ids(40))
    moe.set_valid(valid)
    rc.arm()
    moe.block(blk, 32, store_kv=False, kind="denoise")
    cap = moe.last_cap
    rcap = rc.disarm(); rc.close()

    assert moe.records[-1][5] == M and moe.records[-1][2] <= L * M
    for layer in range(L):
        w = rcap.full[layer]                            # [BL, E] fp32 softmax, as the forward computes it
        pool = build_pool(w, valid, M, rule, K)
        assert int(pool.sum()) == M
        ids = cap.ids[layer].to(torch.int64)            # [BL, K] the dispatched experts
        assert bool(pool[ids.reshape(-1)].all()), f"layer {layer}: a row routed outside the pool"
        assert int(torch.unique(ids).numel()) <= M
    disable_pool_moe(model)


def test_des_rule_matches_desvote_algorithm3_on_a_random_router_tensor():
    """build_pool(..., "des") == sdar_harness.desvote's vote (UNnormalised top-k weights, desvote.py:96-99)
    -- and it differs from the renormalised "top8" rule and from the full-softmax "full" rule."""
    from sdar_harness.desvote import _topk_masked
    g = torch.Generator().manual_seed(7)
    n_diff_top8, n_diff_full = 0, 0
    for trial in range(20):
        logits = torch.randn(BL, 32, generator=g) * 2.0
        w = F.softmax(logits, dim=1, dtype=torch.float)
        valid = torch.zeros(BL, dtype=torch.bool); valid[torch.randperm(BL, generator=g)[:20]] = True
        for M in (4, 8, 12, 24):
            # a direct re-implementation of desvote.vote_moe_forward:121-135
            pv = _topk_masked(w, 4)
            votes = pv[valid].sum(0)
            order = torch.sort(votes, descending=True, stable=True).indices
            want = torch.zeros(32, dtype=torch.bool); want[order[:M]] = True
            assert torch.equal(build_pool(w, valid, M, "des", 4), want), (trial, M)
            n_diff_top8 += int(not torch.equal(want, build_pool(w, valid, M, "top8", 4)))
            n_diff_full += int(not torch.equal(want, build_pool(w, valid, M, "full", 4)))
    assert n_diff_top8 > 0 and n_diff_full > 0          # the three rules are genuinely different rules
    # only the VALID rows vote
    w = F.softmax(torch.randn(BL, 32, generator=g), dim=1, dtype=torch.float)
    v1 = torch.zeros(BL, dtype=torch.bool); v1[:4] = True
    v2 = torch.zeros(BL, dtype=torch.bool); v2[:8] = True
    assert not torch.equal(build_pool(w, v1, 6, "des", 4), build_pool(w, v2, 6, "des", 4))


def test_stock_pool_des_forward_equals_the_desvote_forward():
    """`--stock --pool M --pool-rule des` IS DES-Vote on this model: the same coreset, the same
    dispatched experts, the same output (the routing differs only by an fp32 softmax denominator --
    desvote.py:136-138 re-softmaxes the coreset's logits, we zero the probabilities outside it)."""
    import sdar_harness.desvote as DV
    from sdar_harness.moe_instr import SINK
    M = 6
    blk, valid = block_state()

    from draftalign.moe_probe import RouterCapture
    ours = pooled_model()
    rc = RouterCapture(ours, BL)
    POOL.configure(M, "des")
    moe = MoEMeter(SeqCache(ours, BL, "moe"), SINK, pool=POOL)
    moe.prefill(prompt_ids(40)); moe.set_valid(valid)
    rc.arm()
    lg_ours = moe.block(blk, 32, store_kv=False, kind="denoise")
    cap = moe.last_cap
    rcap = rc.disarm(); rc.close()

    theirs = FakeMoEModel(vocab=V, n_layers=L, seed=SEED)          # the same weights (same seed)
    assert DV.enable_vote_moe(theirs) == L
    DV.CTRL.reset_cost()
    DV.CTRL.M = None; DV.CTRL.valid = valid; DV.CTRL.masked = None
    DV.CTRL.use_recorded = False; DV.CTRL.recording = False
    seq = SeqCache(theirs, BL, "moe")
    seq.prefill(prompt_ids(40))                                    # the prefill is unrestricted
    DV.CTRL.M = M; DV.CTRL.log = []
    lg_theirs = seq.block_forward(blk, 32, store_kv=False, kind="denoise")

    assert torch.allclose(lg_ours, lg_theirs, rtol=1e-4, atol=1e-5)
    assert len(DV.CTRL.log) == L
    for layer, (li, core, hits) in enumerate(DV.CTRL.log):
        core_t = torch.from_numpy(core)
        assert torch.equal(core_t, build_pool(rcap.full[layer], valid, M, "des", K))   # the same coreset
        ours_ids = cap.ids[layer].to(torch.int64)
        assert sorted({int(x) for x in hits.tolist()}) == sorted({int(x) for x in ours_ids.reshape(-1).tolist()})
        assert bool(core_t[ours_ids.reshape(-1)].all())
    DV.disable_vote_moe(theirs); disable_pool_moe(ours)
    DV.CTRL.M = None; DV.CTRL.valid = None; DV.CTRL.log = None


def test_prefill_is_never_pooled_and_the_flush_only_with_pool_flush():
    from sdar_harness.moe_instr import SINK
    M = 6
    blk, valid = block_state()
    filled = torch.arange(1, BL + 1)                     # a finished block for the flush
    for pool_flush in (False, True):
        model = pooled_model()
        POOL.configure(M, "des")
        moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL, pool_flush=pool_flush)
        moe.prefill(prompt_ids(40))
        moe.set_valid(valid)
        moe.block(blk, 32, store_kv=False, kind="denoise")
        moe.block(filled, 32, store_kv=True, kind="kvflush")
        kinds = [r[0] for r in moe.records]
        pools = [r[5] for r in moe.records]
        assert kinds == [DA.MOE_KIND_CODE["prefill"], DA.MOE_KIND_CODE["denoise"], DA.MOE_KIND_CODE["kvflush"]]
        assert pools[0] == 0, "the prefill must never be restricted"
        assert pools[1] == M
        assert pools[2] == (M if pool_flush else 0)
        disable_pool_moe(model)
    # an unrestricted flush loads more experts than a pooled one on the same block
    uniq = {}
    for pool_flush in (False, True):
        model = pooled_model()
        POOL.configure(M, "des")
        moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL, pool_flush=pool_flush)
        moe.prefill(prompt_ids(40)); moe.set_valid(valid)
        moe.block(filled, 32, store_kv=True, kind="kvflush")
        uniq[pool_flush] = moe.records[-1][2]
        disable_pool_moe(model)
    assert uniq[True] <= L * M < uniq[False], uniq          # 12 <= 12 < 32 on this fake


def test_pool_end_to_end_through_da_generate_records_the_pool_size():
    """A whole --stock decode with a pool: every denoise forward pooled, prefill and flush not."""
    from sdar_harness.moe_instr import SINK
    M = 6
    model = pooled_model()
    POOL.configure(M, "full")
    moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL)
    res = da_generate(moe, None, prompt_ids(40), mask_id=MASK, gen_length=32, block_length=BL,
                      draft=None, stopping_criteria_idx=EOS_IDS)
    rec = DA.make_record(res, FakeTok(), item(), prompt_ids(40), moe, None, EOS_IDS, MASK, 0.0)
    assert rec["pool_M"] == M and rec["pool_rule"] == "full" and rec["pool_flush"] is False
    inv = {v: k for k, v in DA.MOE_KIND_CODE.items()}
    for f in rec["forwards"]:
        assert f[5] == (M if inv[f[0]] == "denoise" else 0)
        assert f[2] <= L * M if inv[f[0]] == "denoise" else True
    assert rec["uniq_per_layer_forward_denoise"] <= M
    disable_pool_moe(model)


# ==========================================================================
# (iii) the adapter: load_adapter + merge_lora
# ==========================================================================
class TinyMoE(nn.Module):
    """gate + experts, the stock SDAR MoE forward (returns (out, router_logits))."""

    def __init__(self, h, e, k, dtype=torch.float32):
        super().__init__()
        self.num_experts, self.top_k, self.norm_topk_prob = e, k, True
        self.gate = nn.Linear(h, e, bias=False, dtype=dtype)
        self.experts = nn.ModuleList([nn.Linear(h, h, bias=False, dtype=dtype) for _ in range(e)])

    def forward(self, hidden_states):
        B, S, Hd = hidden_states.shape
        hs = hidden_states.view(-1, Hd)
        router_logits = self.gate(hs)
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(w, self.top_k, dim=-1)
        rw = (rw / rw.sum(dim=-1, keepdim=True)).to(hs.dtype)
        out = torch.zeros_like(hs)
        mask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
        for e in torch.unique(sel).tolist():
            idx, top_x = torch.where(mask[e])
            out.index_add_(0, top_x, (self.experts[e](hs[None, top_x].reshape(-1, Hd)) * rw[top_x, idx, None]).to(hs.dtype))
        return out.reshape(B, S, Hd), router_logits


class TinyAttn(nn.Module):
    def __init__(self, h, dtype=torch.float32):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(h, h, bias=False, dtype=dtype))

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x)) * self.k_proj(x) + self.v_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, h, e, k, dtype=torch.float32):
        super().__init__()
        self.self_attn = TinyAttn(h, dtype)
        self.mlp = TinyMoE(h, e, k, dtype)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)[0]


class TinyInner(nn.Module):
    def __init__(self, h, e, k, n_layers, dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(h, e, k, dtype) for _ in range(n_layers)])


class TinyModel(nn.Module):
    """A registered nn.Module tree with `model.layers[i].self_attn.{q,k,v,o}_proj` and `.mlp.gate`."""

    def __init__(self, vocab=32, h=H, e=E, k=K, n_layers=2, dtype=torch.float32, seed=3):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, h, dtype=dtype)
        self.model = TinyInner(h, e, k, n_layers, dtype)
        self.lm_head = nn.Linear(h, vocab, dtype=dtype)
        from types import SimpleNamespace
        self.config = SimpleNamespace(hidden_size=h, num_hidden_layers=n_layers, num_experts=e,
                                      num_experts_per_tok=k, vocab_size=vocab)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


def test_gatefp32_to_linear_is_bit_identical_in_bf16():
    """The claim the runner relies on: merging the fp32 gate master back to a bf16 nn.Linear is the
    SAME arithmetic training saw, because GateFP32 casts to bf16 BEFORE the matmul (adapter.py:103)."""
    from draftalign.adapter import GateFP32
    torch.manual_seed(0)
    base = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    g = GateFP32(base)
    with torch.no_grad():
        g.weight.add_(0.01 * torch.randn_like(g.weight))         # "training" moved the fp32 master
    x = torch.randn(6, 16, dtype=torch.bfloat16)
    lin = g.to_linear()
    assert lin.weight.dtype == torch.bfloat16 and isinstance(lin, nn.Linear)
    assert torch.equal(g(x), lin(x))                              # bit-identical, not just allclose


def test_adapter_load_and_merge_reproduces_the_unmerged_lora(tmp_path):
    from draftalign.adapter import (attach_lora, freeze_base, save_adapter, load_adapter, merge_lora,
                                    LoRALinear, GateFP32, adapter_metadata)
    ids = torch.arange(12).unsqueeze(0) % 32

    trained = TinyModel()
    stock_out = trained(ids).clone()
    counts = attach_lora(trained, r=2, alpha=4.0)
    freeze_base(trained)
    assert counts == {"lora": 8, "gates": 2}
    assert torch.equal(trained(ids), stock_out), "an initialised adapter must be the identity"
    torch.manual_seed(11)
    with torch.no_grad():                                        # simulate a training run
        for m in trained.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.add_(0.05 * torch.randn_like(m.lora_B))
            elif isinstance(m, GateFP32):
                m.weight.add_(0.02 * torch.randn_like(m.weight))
    want = trained(ids).clone()
    assert not torch.equal(want, stock_out)
    meta = adapter_metadata(trained)
    d = str(tmp_path / "ad")
    save_adapter(trained, d)
    assert os.path.exists(os.path.join(d, "adapter_config.json"))

    # the inference path: a FRESH stock model -> load_adapter -> merge_lora
    infer = TinyModel()
    assert torch.equal(infer(ids), stock_out)
    info = load_adapter(infer, d)
    assert info["loaded"] == 8 * 2 + 2 and info["missing"] == 0 and info["unexpected"] == 0
    assert info["meta"]["config_hash"] == meta["config_hash"]
    assert torch.equal(infer(ids), want)                          # the un-merged adapter reproduces it
    merged = merge_lora(infer)
    assert merged == {"lora": 8, "gates": 2}
    for layer in infer.model.layers:                              # a stock module tree again
        assert isinstance(layer.mlp.gate, nn.Linear) and not isinstance(layer.mlp.gate, GateFP32)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(layer.self_attn, n), nn.Linear)
    got = infer(ids)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5) and not torch.equal(got, stock_out)
    # the gate is exactly the trained master (fp32 model -> the cast is the identity here)
    for a, b in zip(trained.model.layers, infer.model.layers):
        assert torch.equal(a.mlp.gate.weight.to(b.mlp.gate.weight.dtype), b.mlp.gate.weight)
    # the merged model still routes through moe_instr / the pool patch (gate is a plain Linear)
    from sdar_harness.routing import moe_blocks
    assert len(moe_blocks(infer)) == 2


def test_runner_adapter_and_pool_helpers(tmp_path):
    import draftalign.run_da_decode as R
    from types import SimpleNamespace
    from draftalign.adapter import attach_lora, freeze_base, save_adapter
    ns = lambda **kw: SimpleNamespace(**{"stock": False, "draft": "s8", "threshold": .95, "tau_commit": .95,
                                         "tau_acc": .70, "tau_acc_q": .50, "rejudge": False, "rejudge_k": 8,
                                         "drafter": "d32", "judge": "halves", "tau_draft": 0.0, "pool": 0, "pool_rule": "full",
                                         "pool_flush": False, "adapter": None, "full_remask_only": False,
                                         "full_max_passes": DA.FULL_MAX_PASSES, **kw})
    assert R.cell_name("gsm8k", ns(judge="full")) == "gsm8k_B32_t95_dv-s8_full_c95"
    assert R.cell_name("gsm8k", ns(judge="full", full_remask_only=True)) == "gsm8k_B32_t95_dv-s8_full_c95_ro"
    assert R.cell_name("gsm8k", ns(judge="full", full_max_passes=4)) == "gsm8k_B32_t95_dv-s8_full_c95_mp4"
    assert R.cell_name("gsm8k", ns(stock=True, pool=32, pool_rule="des")) == "gsm8k_B32_t95_stock_p32des"
    assert R.cell_name("gsm8k", ns(stock=True, pool=32, pool_rule="des", pool_flush=True)) == "gsm8k_B32_t95_stock_p32desf"
    assert R.cell_name("gsm8k", ns(judge="full", pool=32, pool_rule="top8", adapter="x"), "deadbeefcafe") == \
        "gsm8k_B32_t95_dv-s8_full_c95_p32top8_addeadbeef"
    assert "." not in R.cell_name("gsm8k", ns(judge="full", threshold=.9, pool=24))
    # parser defaults + validation
    a = R.resolve_args(R.build_parser().parse_args(["--judge", "full"]))
    assert a.full_max_passes == DA.FULL_MAX_PASSES and not a.full_remask_only and a.pool == 0 and a.pool_rule == "full"
    assert not a.pool_flush and a.adapter is None
    b = R.resolve_args(R.build_parser().parse_args(["--stock", "--pool", "32", "--pool-rule", "des"]))
    assert b.pool == 32 and b.pool_rule == "des" and b.stock and not b.pool_flush
    with pytest.raises(NotImplementedError):
        R.resolve_args(R.build_parser().parse_args(["--judge", "full", "--rejudge"]))
    with pytest.raises(AssertionError):
        R.resolve_args(R.build_parser().parse_args(["--pool", "4"]))
    # read_adapter_meta reads the hash the cell name uses
    m = TinyModel(); attach_lora(m, r=2, alpha=4.0); freeze_base(m)
    d = str(tmp_path / "ad2"); save_adapter(m, d)
    meta = R.read_adapter_meta(d)
    assert meta["config_hash"] and R.adapter_suffix(ns(adapter=d), meta["config_hash"]) == f"_ad{meta['config_hash'][:8]}"


# ==========================================================================
# (iv) the stock twin, through the new code path
# ==========================================================================
@pytest.mark.parametrize("seed,plen,gen_length,threshold,eos_at", [
    (0, 37, 64, 0.95, ()),
    (1, 64, 64, 0.95, ()),
    (2, 10, 96, 0.90, ()),
    (3, 40, 64, 1.0, ()),
    (4, 50, 128, 0.95, (90,)),
])
def test_stock_twin_still_matches_block_diffusion_generate(seed, plen, gen_length, threshold, eos_at):
    """--stock with NO pool and NO adapter: token for token, pass for pass, forward for forward."""
    ids = prompt_ids(plen)
    ref_model = ScriptedLM(seeded_fn(seed, eos_at=eos_at))
    ref = block_diffusion_generate(ref_model, ids, mask_id=MASK, gen_length=gen_length, block_length=BL,
                                   denoising_steps=BL, temperature=1.0, top_k=1, top_p=1.0,
                                   threshold=threshold, stopping_criteria_idx=EOS_IDS, greedy_exact=True)
    POOL.configure(None, "full")                       # a PoolCtrl attached but switched off
    moe = MoEMeter(SeqCache(ScriptedLM(seeded_fn(seed, eos_at=eos_at)), BL, "moe"), None, pool=POOL)
    dv = da_generate(moe, None, ids, mask_id=MASK, gen_length=gen_length, block_length=BL,
                     denoising_steps=BL, threshold=threshold, draft=None, stopping_criteria_idx=EOS_IDS)
    assert torch.equal(ref.x, dv.x) and ref.n_forward == dv.n_forward == len(moe.records)
    assert (ref.prompt_length, ref.total_length, ref.prefill_blocks, ref.num_blocks, ref.stopped_early) == \
           (dv.prompt_length, dv.total_length, dv.prefill_blocks, dv.num_blocks, dv.stopped_early)
    assert len(ref.blocks) == len(dv.blocks)
    for a, b in zip(ref.blocks, dv.blocks):
        assert (a.block_index, a.start, a.n_steps, a.n_forward) == (b.block_index, b.start, b.n_steps, b.n_forward)
        assert torch.equal(a.valid, b.valid) and torch.equal(a.tokens, b.tokens)
        for s in range(a.n_steps):
            assert torch.equal(a.committed[s], b.committed[s]) and torch.equal(a.proposed[s], b.proposed[s])
            assert torch.equal(a.confidence[s], b.confidence[s])
    for c1, c2 in zip(ref_model.calls, moe.seq.model.calls):
        assert torch.equal(c1["input_ids"], c2["input_ids"]) and torch.equal(c1["pos"], c2["pos"])
        assert torch.equal(c1["mask"], c2["mask"]) and c1["store_kv"] == c2["store_kv"] and c1["cached"] == c2["cached"]
    assert all(r[5] == 0 for r in moe.records)         # nothing was pooled
    rec = DA.make_record(dv, FakeTok(), item(), ids, moe, None, EOS_IDS, MASK, 0.0)
    assert rec["judge_mode"] == "stock" and rec["pool_M"] is None and rec["adapter"] is None
    assert rec["passes_per_block"] == rec["stock_passes_per_block"] == [b.n_steps for b in dv.blocks]


# ==========================================================================
# (v) analyze_da.py on full-judge / pooled records
# ==========================================================================
def test_analyzer_reports_judge_mode_pool_and_tentative_remask(tmp_path, capsys):
    import json
    import draftalign.analyze_da as A
    from sdar_harness.moe_instr import SINK
    cell = "gsm8k_B32_t95_dv-s8_full_c95_p6des"
    cdir = tmp_path / cell; cdir.mkdir()
    ref_dir = tmp_path / "ref"; ref_dir.mkdir()
    recs, refs = [], []
    for i, idx in enumerate([9, 12]):
        model = pooled_model(seed=SEED + i, vocab=V)
        POOL.configure(6, "des")
        moe = MoEMeter(SeqCache(model, BL, "moe"), SINK, pool=POOL)
        dr = SeqCache(ScriptedLM(scripted_drafter_fn), BL, "drafter")
        rec, res = decode_item(moe, dr, FakeTok(), item(idx), prompt_ids(PLEN), mask_id=MASK, eos_ids=EOS_IDS,
                               gen_length=GEN, block_length=BL, threshold=.95, draft="s8", judge="full",
                               tau_commit=.95, adapter="/tmp/ad", adapter_hash="abc123")
        rec.update({"cell": cell, "config_hash": "h", "model_revision": "r", "correct": i})
        recs.append(rec)
        disable_pool_moe(model)
        moe2 = meter_for(seeded_fn(70 + i))
        ref, _ = decode_item(moe2, None, FakeTok(), item(idx), prompt_ids(PLEN), mask_id=MASK, eos_ids=EOS_IDS,
                             gen_length=GEN, block_length=BL, threshold=.95, draft=None)
        ref.update({"cell": "gsm8k_B32_t95_stock", "correct": 1})
        refs.append(ref)
    with open(cdir / f"{cell}.calibvalid.shard0of1.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    open(cdir / f"{cell}.calibvalid.shard0of1.jsonl.done", "w").write("ok\n")
    with open(ref_dir / "gsm8k_B32_t95_stock.calibvalid.shard0of1.jsonl", "w") as fh:
        for r in refs:
            fh.write(json.dumps(r) + "\n")

    cells = A.load_cell(str(cdir))
    s = A.summarise(cells[(cell, "calibvalid")]["recs"], "gsm8k")
    assert s["judge_mode"] == ["full"] and s["pool_M"] == ["6"] and s["pool_rule"] == ["des"]
    assert s["adapter"] == ["/tmp/ad"] and s["adapter_hash"] == ["abc123"]
    assert s["pool_size_used"] == 6 and 0 < s["pooled_forward_frac"] < 1      # prefill / flush not pooled
    assert s["passes_per_block"] >= s["stock_passes_per_block"] >= 0
    assert s["tentative_per_block"] >= 0 and s["remask_per_block"] >= 0
    assert s["moe_forwards_per_token_by_kind"]["judge"] > 0
    A.main([str(cdir), "--ref-dir", str(ref_dir), "--out", str(tmp_path / "s.json")])
    out = capsys.readouterr().out
    assert "judge full" in out and "pool M 6 rule des" in out and "tentative" in out and "remask" in out
    j = json.load(open(tmp_path / "s.json"))
    assert j["cells"][f"{cell}.calibvalid"]["pool_M"] == ["6"]


def test_full_commit_rule_asymmetric_override_threshold():
    """Confirmations commit at tau_commit, overrides only at tau_override; None reproduces the symmetric rule."""
    import torch
    from draftalign.da_decode import full_commit_rule
    mask_id = 99
    cur = torch.tensor([5, 6, 7, 8, 9, 10])
    D = torch.tensor([5, 6, 7, 8, 9, 10])
    argmax = torch.tensor([5, 6, 1, 2, 9, 3])          # slots 0,1,4 agree with the draft; 2,3,5 override
    p = torch.tensor([0.96, 0.92, 0.96, 0.995, 0.50, 0.80])
    open_ = torch.ones(6, dtype=torch.bool)
    new_s, commit_s, tent_s, remask_s, forced_s = full_commit_rule(cur, argmax, p, D, open_, 0.95, mask_id)
    assert commit_s.tolist() == [True, False, True, True, False, False]      # symmetric: p > .95 regardless of agreement
    new_a, commit_a, tent_a, remask_a, forced_a = full_commit_rule(cur, argmax, p, D, open_, 0.95, mask_id, tau_override=0.99)
    assert commit_a.tolist() == [True, False, False, True, False, False]     # override at .96 no longer commits; .995 does
    assert new_a[2].item() == mask_id and new_a[3].item() == 2 and new_a[0].item() == 5
    assert not forced_a and tent_a.tolist() == [False, True, False, False, True, False]
    same = full_commit_rule(cur, argmax, p, D, open_, 0.95, mask_id, tau_override=0.95)
    assert same[1].tolist() == commit_s.tolist()
    # lower confirm threshold with strict override
    _, commit_b, *_ = full_commit_rule(cur, argmax, p, D, open_, 0.90, mask_id, tau_override=0.99)
    assert commit_b.tolist() == [True, True, False, True, False, False]
