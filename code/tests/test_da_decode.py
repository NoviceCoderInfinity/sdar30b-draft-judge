"""CPU tests for draftalign.da_decode (Draft-Verify decode), its runner helpers and analyzer.
No real model is loaded; fake cache-aware models follow tests/test_draftalign.py's pattern.

Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_da_decode.py
"""
from __future__ import annotations

import json, math, os, sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from sdar_harness.decode import block_diffusion_generate                                   # noqa: E402
from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign import da_decode as DA                                                     # noqa: E402
from draftalign.da_decode import (MoEMeter, da_generate, judge_halves, apply_accept_rule,  # noqa: E402
                                  save_trace, load_trace, decode_item, CODE_MASK, CODE_COMMIT,
                                  CODE_ACCEPT_EQ, CODE_ACCEPT_Q, CODE_FORCED, CODE_ACCEPT_LATE)

V, MASK, EOS_IDS = 64, 63, [62, 61]
SPECIAL = set(EOS_IDS) | {MASK}
BL = 32


# --------------------------------------------------------------------------
# fake models
# --------------------------------------------------------------------------
class ScriptedLM(nn.Module):
    """Cache-aware fake SDAR: logits at slot s = fn(pos, token at s, is_mask, block ids, cached len).
    Records every call; the cache stores K=V=ids (as test_state_spec._FakeCausal)."""

    def __init__(self, fn, vocab=V):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.fn, self.vocab, self.calls = fn, vocab, []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, store_kv=None, **kw):
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.calls.append(dict(input_ids=input_ids.clone(), pos=position_ids.clone(), mask=attention_mask.clone(),
                               store_kv=bool(store_kv), cached=cached))
        B, S = input_ids.shape
        assert B == 1
        ids = input_ids[0].tolist()
        logits = torch.empty(1, S, self.vocab)
        for s in range(S):
            logits[0, s] = self.fn(int(position_ids[0, s]), ids[s], ids[s] == MASK, ids, cached, past_key_values)
        if past_key_values is not None and store_kv:
            k = input_ids.float().view(B, 1, S, 1).expand(B, 2, S, 3).contiguous()
            for l in range(2):
                past_key_values.update(k, k, l)
        return SimpleNamespace(logits=logits)


def seeded_fn(seed, temp=5.0, eos_at=()):
    """Deterministic pseudo-random logits keyed by (seed, position, block ids, cache length, cache content):
    identical inputs -> identical logits, so two decoders taking the same path see the same model."""
    eos_at = set(eos_at)

    def fn(pos, tok, is_mask, ids, cached, pkv):
        ctx = int(pkv[0][0][0, 0, :, 0].sum().item()) if cached else 0
        g = torch.Generator().manual_seed(hash((seed, pos, tuple(ids), cached, ctx)) & 0x7FFFFFFF)
        row = torch.randn(V, generator=g) * temp
        row[EOS_IDS] = -30.0                     # EOS only where scripted
        if pos in eos_at:
            row[EOS_IDS[0]] = 30.0
        row[MASK] = -30.0
        return row
    return fn


def logprob_row(probs: dict, eps=1e-9):
    row = torch.full((V,), math.log(eps))
    for t, p in probs.items():
        row[t] = math.log(p)
    return row


def onehot_row(t, hi=30.0, bg=-8.0):
    row = torch.full((V,), bg); row[t] = hi
    return row


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i in SPECIAL))


def prompt_ids(plen):
    return (torch.arange(plen) % 50 + 1).unsqueeze(0)          # never MASK / EOS


def item(idx=7):
    return {"idx": idx, "rank": idx, "split": "train", "prompt": "p", "gold": "g"}


def meter_for(fn):
    return MoEMeter(SeqCache(ScriptedLM(fn), BL, "moe"))


# stock-record field names (run_research_baselines.decode_item + main), which analyze_research_baselines relies on
STOCK_FIELDS = {"idx", "rank", "split", "prompt_tokens", "pred", "gold", "gen_ids", "n_gen_nonmask", "n_gen_to_eos",
                "stopped_eos", "truncated", "n_blocks", "n_forward", "n_prefill", "n_denoise", "n_flush",
                "passes_per_block", "n_valid_per_block", "commits_per_pass", "unique_expert_layer_loads",
                "layer_forwards", "uniq_per_layer_forward_denoise", "uniq_per_layer_forward_all", "seconds",
                "ms_by_kind", "uniq_by_kind", "forwards", "peak_mem_gib"}


# --------------------------------------------------------------------------
# (i) the --stock path == sdar_harness.decode.block_diffusion_generate, pass for pass
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed,plen,gen_length,threshold,eos_at", [
    (0, 37, 64, 0.95, ()),          # prompt tail of 5 in the first block
    (1, 64, 64, 0.95, ()),          # no tail
    (2, 10, 96, 0.90, ()),          # no prefill at all (prompt shorter than a block)
    (3, 40, 64, 1.0, ()),           # threshold 1.0 -> the top-k branch
    (4, 50, 128, 0.95, (90,)),      # EOS inside the second decode block -> early stop
])
def test_stock_path_matches_block_diffusion_generate(seed, plen, gen_length, threshold, eos_at):
    ids = prompt_ids(plen)
    ref_model = ScriptedLM(seeded_fn(seed, eos_at=eos_at))
    ref = block_diffusion_generate(ref_model, ids, mask_id=MASK, gen_length=gen_length, block_length=BL,
                                   denoising_steps=BL, temperature=1.0, top_k=1, top_p=1.0, threshold=threshold,
                                   stopping_criteria_idx=EOS_IDS, greedy_exact=True)
    moe = meter_for(seeded_fn(seed, eos_at=eos_at))
    dv = da_generate(moe, None, ids, mask_id=MASK, gen_length=gen_length, block_length=BL, denoising_steps=BL,
                     threshold=threshold, draft=None, stopping_criteria_idx=EOS_IDS)
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
    # the very same forwards, in the same order, with the same inputs / masks / positions / cache state
    dv_model = moe.seq.model
    assert len(ref_model.calls) == len(dv_model.calls)
    for c1, c2 in zip(ref_model.calls, dv_model.calls):
        assert torch.equal(c1["input_ids"], c2["input_ids"]) and torch.equal(c1["pos"], c2["pos"])
        assert torch.equal(c1["mask"], c2["mask"]) and c1["store_kv"] == c2["store_kv"] and c1["cached"] == c2["cached"]
    kinds = [r[0] for r in moe.records]
    assert kinds.count(DA.MOE_KIND_CODE["kvflush"]) == len(dv.blocks) and kinds.count(DA.MOE_KIND_CODE["prefill"]) == (1 if plen >= BL else 0)
    assert all(b.draft_tokens is None and b.judge_code is None and b.n_judge == 0 for b in dv.blocks)
    cps = [int(c.sum()) for b in dv.blocks for c in b.committed]
    if threshold < 1.0:
        assert max(cps) > 1 and min(cps) == 1          # both stock branches were exercised
    else:
        assert set(cps) == {1}
    if eos_at:
        assert dv.stopped_early and len(dv.blocks) < dv.num_blocks - dv.prefill_blocks


# --------------------------------------------------------------------------
# (ii) the judge reads every slot from the forward in which it was MASKED
# --------------------------------------------------------------------------
def f_tok(pos):
    return (pos * 7 + 3) % 60


def g_tok(pos):
    return (pos * 11 + 5) % 60


def judge_echo_fn(pos, tok, is_mask, ids, cached, pkv):
    """Masked slot -> f(pos) with p~1; filled slot -> echoes its own token (the untrained-slot behaviour)."""
    return onehot_row(f_tok(pos)) if is_mask else onehot_row(tok)


@pytest.mark.parametrize("plen", [37, 63])
def test_judge_reads_each_slot_from_the_masked_forward(plen):
    moe = meter_for(judge_echo_fn)
    ids = prompt_ids(plen)
    assert moe.prefill(ids) == 32
    lo = 32
    pos = torch.arange(lo, lo + BL)
    valid = pos >= plen
    cur = torch.full((BL,), MASK); cur[~valid] = ids[0, lo:plen]
    D = torch.tensor([g_tok(int(p)) for p in pos])
    assert not any(f_tok(int(p)) == g_tok(int(p)) for p in pos[valid])
    n0 = len(moe.records)
    j = judge_halves(moe, cur, valid, D, lo, MASK)
    calls = [c for c in moe.seq.model.calls if not c["store_kv"]]
    even = (pos % 2 == 0)
    if plen == 37:
        assert j["n_fwd"] == 2 == len(calls) == len(moe.records) - n0
        a, b = calls
        assert torch.equal(a["input_ids"][0][valid & even], torch.full((int((valid & even).sum()),), MASK))
        assert torch.equal(a["input_ids"][0][valid & ~even], D[valid & ~even])
        assert torch.equal(b["input_ids"][0][valid & ~even], torch.full((int((valid & ~even).sum()),), MASK))
        assert torch.equal(b["input_ids"][0][valid & even], D[valid & even])
        for c in (a, b):
            assert torch.equal(c["input_ids"][0][~valid], cur[~valid]) and c["cached"] == lo and torch.equal(c["pos"][0], pos)
    else:                                   # a single valid slot (63, odd): only half_b has a masked slot
        assert int(valid.sum()) == 1 and j["n_fwd"] == 1 == len(calls)
        assert calls[0]["input_ids"][0][31] == MASK and torch.equal(calls[0]["input_ids"][0][:31], cur[:31])
    # every valid slot: the MASKED-forward prediction f(pos), never the echo of the draft
    assert j["argmax"][valid].tolist() == [f_tok(int(p)) for p in pos[valid]]
    assert (j["argmax"][~valid] == -1).all() and torch.isnan(j["p"][~valid]).all() and torch.isnan(j["q"][~valid]).all()
    assert (j["p"][valid] > 0.99).all() and (j["q"][valid] < 1e-3).all() and j["raw_mask_argmax"] == 0
    assert moe.seq.cached_length() == lo                       # judge forwards never write the cache
    assert all(r[0] == DA.MOE_KIND_CODE["judge"] for r in moe.records[n0:])


def test_judge_argmax_uses_fp32_with_mask_suppressed():
    def fn(pos, tok, is_mask, ids, cached, pkv):
        row = torch.full((V,), -8.0)
        row[MASK] = 9.0; row[5] = 8.0           # raw argmax would be [MASK]
        return row
    moe = meter_for(fn)
    cur = torch.full((BL,), MASK); valid = torch.ones(BL, dtype=torch.bool)
    D = torch.full((BL,), 5)
    j = judge_halves(moe, cur, valid, D, 0, MASK)
    assert (j["argmax"] == 5).all() and j["raw_mask_argmax"] == BL and j["p"].dtype == torch.float32
    assert torch.allclose(j["p"], j["q"]) and (j["p"] > 0.99).all()


# --------------------------------------------------------------------------
# (iii) accept rule + progress guarantee
# --------------------------------------------------------------------------
def test_accept_rule_hand_cases():
    bl = 8
    cur = torch.full((bl,), MASK); cur[:2] = torch.tensor([1, 2])
    slots = torch.tensor([False, False, True, True, True, True, True, True])
    a = torch.tensor([-1, -1, 10, 12, 13, 15, 17, 20])
    D = torch.tensor([1, 2, 11, 12, 14, 16, 17, 20])
    p = torch.tensor([float("nan"), float("nan"), .97, .80, .80, .80, .60, .95])
    q = torch.tensor([float("nan"), float("nan"), .01, .80, .60, .30, .60, .95])
    new, code = apply_accept_rule(cur, a, p, q, D, slots, .95, .70, .50)
    assert code.tolist() == [CODE_MASK, CODE_MASK, CODE_COMMIT, CODE_ACCEPT_EQ, CODE_ACCEPT_Q, CODE_MASK, CODE_ACCEPT_Q, CODE_ACCEPT_EQ]
    assert new.tolist() == [1, 2, 10, 12, 14, MASK, 17, 20]     # slot 7: p == .95 is NOT > .95 (strict), accepted via a == D
    # q rule disabled with tau_acc_q = 1.0
    new2, code2 = apply_accept_rule(cur, a, p, q, D, slots, .95, .70, 1.0)
    assert code2.tolist() == [0, 0, CODE_COMMIT, CODE_ACCEPT_EQ, CODE_MASK, CODE_MASK, CODE_MASK, CODE_ACCEPT_EQ]
    assert new2.tolist() == [1, 2, 10, 12, MASK, MASK, MASK, 20]
    # progress guarantee: nothing qualifies -> the single best p among the slots is committed to a_i
    p3 = torch.tensor([float("nan"), float("nan"), .50, .40, .61, .30, .20, .10])
    q3 = torch.full((bl,), .1)
    a3 = torch.tensor([-1, -1, 30, 31, 32, 33, 34, 35]); D3 = torch.tensor([1, 2, 40, 41, 42, 43, 44, 45])
    new3, code3 = apply_accept_rule(cur, a3, p3, q3, D3, slots, .95, .70, .50)
    assert code3.tolist() == [0, 0, 0, 0, CODE_FORCED, 0, 0, 0] and new3.tolist() == [1, 2, MASK, MASK, 32, MASK, MASK, MASK]
    new4, code4 = apply_accept_rule(cur, a3, p3, q3, D3, slots, .95, .70, .50, force_progress=False)
    assert code4.tolist() == [0] * bl and torch.equal(new4, cur)
    # a slot outside `slots` is never touched even with a high p
    p5 = p3.clone(); p5[0] = .99
    new5, code5 = apply_accept_rule(cur, a3, p5, q3, D3, slots, .95, .70, .50)
    assert new5[0] == 1 and code5[0] == 0 and code5[4] == CODE_FORCED


# --------------------------------------------------------------------------
# (iii b) end to end on scripted judge + drafter: judge state, stock passes, flushes, rejudge
# --------------------------------------------------------------------------
PLEN, GEN = 52, 12                                  # one decode block: positions 32..63, valid 52..63
COMMIT_SET, ACCEPT_SET, Q_SET = {52, 55}, {53, 58, 61}, {56}


def a_tok(pos):
    return (pos * 3 + 1) % 50


def scripted_judge_fn(pos, tok, is_mask, ids, cached, pkv):
    if not is_mask:
        return onehot_row(tok)                                          # echo at filled slots
    n_mask = sum(1 for t in ids if t == MASK)
    if pos in COMMIT_SET:
        return onehot_row(a_tok(pos))                                   # p ~ 1, a != D -> commit a
    if pos in ACCEPT_SET or n_mask <= 3:
        return logprob_row({g_tok(pos): .80, a_tok(pos): .15, 0: .05})          # a == D, .7 < p < .95 -> accept
    if pos in Q_SET:
        return logprob_row({g_tok(pos): .55, a_tok(pos): .45})                  # a == D, p = q = .55: not > .7, q > .5 -> accept via q
    r = .60 + .001 * (pos % 10)
    return logprob_row({a_tok(pos): r, g_tok(pos): .05, 0: 1 - r - .05})        # p < .7, q < .5 -> stays [MASK]


def scripted_drafter_fn(pos, tok, is_mask, ids, cached, pkv):
    return onehot_row(g_tok(pos)) if is_mask else onehot_row(tok)


def run_scripted(rejudge=False, rejudge_k=8, draft="s8", **kw):
    moe = meter_for(scripted_judge_fn)
    dr = SeqCache(ScriptedLM(scripted_drafter_fn), BL, "drafter")
    ids = prompt_ids(PLEN)
    res = da_generate(moe, dr, ids, mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95, draft=draft,
                      tau_commit=.95, tau_acc=.70, tau_acc_q=.50, rejudge=rejudge, rejudge_k=rejudge_k,
                      stopping_criteria_idx=EOS_IDS, **kw)
    return moe, dr, ids, res


def test_dv_end_to_end_scripted():
    moe, dr, ids, res = run_scripted()
    assert res.num_blocks == 2 and res.prefill_blocks == 1 and len(res.blocks) == 1
    b = res.blocks[0]
    pos = torch.arange(32, 64)
    valid = pos >= PLEN
    assert torch.equal(b.valid, valid) and int(valid.sum()) == 12
    # drafter: prefill (32) + 8 static steps over 12 open slots + flush of the FINAL block
    assert b.n_draft_fwd == 8 and dr.n_by_kind == {"prefill": 1, "draft_s8": 8, "flush": 1}
    assert torch.equal(b.draft_tokens[valid], torch.tensor([g_tok(int(p)) for p in pos[valid]]))
    assert torch.equal(b.draft_tokens[~valid], ids[0, 32:PLEN]) and (b.p_draft[valid] > .99).all()
    # judge outcome per slot
    exp_code = []
    for p in pos.tolist():
        if p < PLEN: exp_code.append(CODE_MASK)
        elif p in COMMIT_SET: exp_code.append(CODE_COMMIT)
        elif p in ACCEPT_SET: exp_code.append(CODE_ACCEPT_EQ)
        elif p in Q_SET: exp_code.append(CODE_ACCEPT_Q)
        else: exp_code.append(CODE_MASK)
    assert b.judge_code.tolist() == exp_code and b.n_judge == 2 and not b.forced
    assert int(b.accepted.sum()) == 4 and int(b.committed_judge.sum()) == 2
    for p in COMMIT_SET:
        assert b.judge_argmax[p - 32] == a_tok(p) and b.judge_p[p - 32] > .99 and b.tokens[p - 32] == a_tok(p)
    for p in ACCEPT_SET | Q_SET:
        assert b.tokens[p - 32] == g_tok(p)
    assert b.judge_argmax[56 - 32] == g_tok(56) and abs(float(b.judge_q[56 - 32]) - .55) < 1e-3 and abs(float(b.judge_p[56 - 32]) - .55) < 1e-3
    # 6 open slots left -> 6 stock passes of exactly one commit (p ~ .6 < .95), then the flush
    assert b.n_steps == 6 and [int(c.sum()) for c in b.committed] == [1] * 6
    rest = [p for p in pos[valid].tolist() if p not in COMMIT_SET | ACCEPT_SET | Q_SET]
    assert rest == [54, 57, 59, 60, 62, 63]
    # passes 1-3 (6, 5, 4 masks): the stock rule commits the best p = .60 + .001 * (pos % 10) -> 59, 57, 54 to a_tok;
    # passes 4-6 (<= 3 masks): the scripted rows favour the draft token -> 60, 62, 63 end as g_tok
    assert [int(torch.nonzero(c).item()) + 32 for c in b.committed[:3]] == [59, 57, 54]
    assert all(int(b.tokens[p - 32]) == a_tok(p) for p in (54, 57, 59))
    assert all(int(b.tokens[p - 32]) == g_tok(p) for p in (60, 62, 63))
    assert b.n_forward == 2 + 6 + 1 and res.n_forward == 1 + b.n_forward == len(moe.records)
    assert abs(b.draft_agreement - 7 / 12) < 1e-6
    # MoE calls: prefill (store), judge x2 (no store, cached 32), 6 denoise (no store), flush (store -> 64)
    calls = moe.seq.model.calls
    assert [c["store_kv"] for c in calls] == [True, False, False] + [False] * 6 + [True]
    assert all(c["cached"] == 32 for c in calls[1:]) and moe.seq.cached_length() == 64 == dr.cached_length()
    assert torch.equal(calls[-1]["input_ids"][0], b.tokens) and torch.equal(dr.model.calls[-1]["input_ids"][0], b.tokens)
    # denoise passes start from the judged state: the first denoise input holds the committed/accepted tokens
    first_den = calls[3]["input_ids"][0]
    assert (first_den[valid] == MASK).sum() == 6 and first_den[52 - 32] == a_tok(52) and first_den[53 - 32] == g_tok(53)
    # the drafter saw the block with every valid slot masked and the prompt tail fixed
    d0 = dr.model.calls[1]["input_ids"][0]
    assert (d0[valid] == MASK).all() and torch.equal(d0[~valid], ids[0, 32:PLEN])
    kinds = [r[0] for r in moe.records]
    assert kinds == [0, 3, 3] + [1] * 6 + [2]


def test_dv_progress_guarantee_end_to_end():
    def flat_judge(pos, tok, is_mask, ids, cached, pkv):
        if not is_mask:
            return onehot_row(tok)
        r = .50 + .01 * (pos - PLEN)
        return logprob_row({a_tok(pos): r, g_tok(pos): .05, 0: 1 - r - .05})     # best slot = 63
    moe = MoEMeter(SeqCache(ScriptedLM(flat_judge), BL, "moe"))
    dr = SeqCache(ScriptedLM(scripted_drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(PLEN), mask_id=MASK, gen_length=GEN, block_length=BL, draft="s8",
                      stopping_criteria_idx=EOS_IDS)
    b = res.blocks[0]
    assert b.forced and int(b.committed_judge.sum()) == 1 and int(b.accepted.sum()) == 0
    assert b.judge_code[63 - 32] == CODE_FORCED and b.tokens[63 - 32] == a_tok(63)
    assert b.n_steps == 11 and b.n_forward == 2 + 11 + 1


def test_dv_rejudge_redrafts_and_rejudges_open_slots_only():
    moe, dr, ids, res = run_scripted(rejudge=True, rejudge_k=4)
    b = res.blocks[0]
    # judge leaves 6 open; stock pass 1 commits one (5 open >= 4) -> redraft + rejudge accept all 5 (n_mask <= 3 rule)
    assert b.n_steps == 1 and b.rejudge_rounds == 1 and b.n_rejudge == 2 and b.rejudge_accepted == 5 and b.rejudge_committed == 0
    assert b.n_redraft_fwd == 5 and dr.n_by_kind == {"prefill": 1, "draft_s8": 8 + 5, "flush": 1}
    assert (b.tokens != MASK).all() and b.n_forward == 2 + 1 + 2 + 1
    rj = b.rejudges[0]
    open_ = rj["open"]
    assert int(open_.sum()) == 5 and (rj["code"][open_] == CODE_ACCEPT_EQ).all() and (rj["code"][~open_] == 0).all()
    assert torch.equal(b.tokens[open_], rj["draft"][open_])
    # the redraft call kept the committed slots fixed and masked only the open ones
    redraft_in = dr.model.calls[1 + 8][ "input_ids"][0]
    assert torch.equal(redraft_in == MASK, open_)
    ra, rb = moe.seq.model.calls[4]["input_ids"][0], moe.seq.model.calls[5]["input_ids"][0]   # rejudge halves
    even = torch.arange(BL) % 2 == 0
    assert torch.equal(ra == MASK, open_ & even) and torch.equal(rb == MASK, open_ & ~even)
    assert torch.equal(ra[~open_], b.tokens[~open_]) and torch.equal(rb[~open_], b.tokens[~open_])   # committed slots fixed
    assert torch.equal(ra[open_ & ~even], rj["draft"][open_ & ~even]) and torch.equal(rb[open_ & even], rj["draft"][open_ & even])
    assert [r[0] for r in moe.records] == [0, 3, 3, 1, 4, 4, 2]
    # below the k threshold nothing is redrafted
    moe2, dr2, _, res2 = run_scripted(rejudge=True, rejudge_k=6)
    assert res2.blocks[0].rejudge_rounds == 0 and res2.blocks[0].n_steps == 6


# --------------------------------------------------------------------------
# (iv) EOS trimming + record fields == run_research_baselines conventions
# --------------------------------------------------------------------------
def reference_fields(res, tok):
    """Verbatim from run_research_baselines.decode_item lines 138-142 / 150-157."""
    gen = res.generated_ids()[0].cpu()
    nonmask = gen[gen != MASK]
    eos_pos = [int(i) for i in torch.nonzero(torch.isin(nonmask, torch.tensor(EOS_IDS))).flatten().tolist()]
    n_to_eos = (eos_pos[0] + 1) if eos_pos else int(nonmask.numel())
    return {"pred": tok.decode(nonmask.tolist(), skip_special_tokens=True), "gen_ids": nonmask.tolist(),
            "n_gen_nonmask": int(nonmask.numel()), "n_gen_to_eos": n_to_eos, "stopped_eos": bool(res.stopped_early),
            "truncated": bool(not res.stopped_early and len(res.blocks) == res.num_blocks - res.prefill_blocks),
            "n_blocks": len(res.blocks), "n_forward": res.n_forward,
            # passes/block = every block forward except the kv flush (judge + rejudge + stock denoise);
            # for a stock / blind cell n_judge == n_rejudge == 0, so this is the reference cells' [b.n_steps]
            "passes_per_block": [b.n_judge + b.n_rejudge + b.n_steps for b in res.blocks],
            "stock_passes_per_block": [b.n_steps for b in res.blocks],
            "n_valid_per_block": [int(b.valid.sum()) for b in res.blocks],
            "commits_per_pass": [int(c.sum()) for b in res.blocks for c in b.committed]}


@pytest.mark.parametrize("draft", [None, "s8"])
def test_eos_trimming_and_record_fields(draft):
    tok = FakeTok()
    plen, gen_length = 37, 64
    eos_at = 57                                        # inside the first decode block (valid 37..63)
    moe = meter_for(seeded_fn(11, eos_at=(eos_at,)))
    dr = SeqCache(ScriptedLM(seeded_fn(12)), BL, "drafter") if draft else None
    ids = prompt_ids(plen)
    rec, res = decode_item(moe, dr, tok, item(), ids, mask_id=MASK, eos_ids=EOS_IDS, gen_length=gen_length,
                           block_length=BL, threshold=.95, draft=draft)
    ref = reference_fields(res, tok)
    for k, v in ref.items():
        assert rec[k] == v, k
    assert rec["stopped_eos"] and not rec["truncated"] and rec["n_blocks"] == 1
    assert rec["n_gen_nonmask"] == 27 and rec["n_gen_to_eos"] == eos_at - plen + 1 == 21
    assert rec["gen_ids"][20] == EOS_IDS[0] and len(rec["gen_ids"]) == 27      # tokens after EOS in the block are kept, as stock
    assert rec["pred"] == " ".join(str(i) for i in rec["gen_ids"] if i not in SPECIAL)
    assert rec["prompt_tokens"] == plen and rec["idx"] == 7 and rec["rank"] == 7 and rec["split"] == "train" and rec["gold"] == "g"
    assert STOCK_FIELDS <= set(rec) and set(rec["ms_by_kind"]) == set(DA.MOE_KIND_CODE) >= {"prefill", "denoise", "kvflush"}
    assert rec["n_prefill"] == 1 and rec["n_flush"] == 1 and rec["n_denoise"] == rec["stock_passes_per_block"][0]
    assert rec["n_denoise"] + rec["n_judge"] + rec["n_rejudge"] == rec["passes_per_block"][0]
    assert rec["n_forward"] == rec["n_prefill"] + rec["n_denoise"] + rec["n_flush"] + rec["n_judge"] + rec["n_rejudge"]
    assert len(rec["forwards"]) == rec["n_forward"] and all(len(f) == 6 for f in rec["forwards"])
    assert all(f[5] == 0 for f in rec["forwards"])                 # no pool -> pool size 0
    assert rec["unique_expert_layer_loads"] == 0 and rec["layer_forwards"] == 0 and rec["uniq_per_layer_forward_all"] is None
    assert rec["peak_mem_gib"] is None and rec["seconds"] >= rec["moe_seconds"] >= 0
    if draft is None:
        assert rec["draft"] is None and rec["n_judge"] == 0 and rec["accepted_per_block"] == [0] and rec["draft_agreement_per_block"] == [None]
        assert rec["drafter_n_by_kind"] == {k: 0 for k in DA.DRAFTER_KIND_CODE} and rec["drafter_forwards"] == []
    else:
        assert rec["draft"] == "s8" and rec["n_judge"] == 2 and rec["judge_forwards_per_block"] == [2]
        assert rec["drafter_n_by_kind"]["prefill"] == 1 and rec["drafter_n_by_kind"]["draft"] == 8 and rec["drafter_n_by_kind"]["flush"] == 1
        assert 0.0 <= rec["draft_agreement_per_block"][0] <= 1.0 and len(rec["drafter_forwards"]) == 10
        assert rec["moe_forwards_per_block"] == [rec["n_judge"] + rec["n_denoise"] + rec["n_flush"]]
    # analyze_research_baselines.summarise consumes the record as it consumes a stock record
    import analyze_research_baselines as ARB
    rec["correct"] = 1
    s = ARB.summarise([rec], "gsm8k")
    assert s["n"] == 1 and s["accuracy"] == 100.0 and s["passes_per_block"] == rec["passes_per_block"][0]
    # truncated: no EOS, every block used
    moe2 = meter_for(seeded_fn(13))
    dr2 = SeqCache(ScriptedLM(seeded_fn(14)), BL, "drafter") if draft else None
    rec2, res2 = decode_item(moe2, dr2, tok, {**item(3), "task_id": "t3"}, ids, mask_id=MASK, eos_ids=EOS_IDS,
                             gen_length=gen_length, block_length=BL, threshold=.95, draft=draft)
    assert rec2["truncated"] and not rec2["stopped_eos"] and rec2["n_blocks"] == 3 == res2.num_blocks - res2.prefill_blocks
    assert rec2["n_gen_nonmask"] == rec2["n_gen_to_eos"] == 128 - plen and rec2["task_id"] == "t3"
    for k, v in reference_fields(res2, tok).items():
        assert rec2[k] == v, k


# --------------------------------------------------------------------------
# (v) trace NPZ round trip (+ the stock subset == run_research_baselines.save_trace)
# --------------------------------------------------------------------------
def test_trace_npz_roundtrip(tmp_path):
    moe, dr, ids, res = run_scripted(rejudge=False)
    b = res.blocks[0]
    path = str(tmp_path / "traces" / "00007.npz")
    save_trace(path, res, BL)
    tr = load_trace(path)
    assert set(tr) == {"proposed", "confidence", "committed", "valid", "n_pass", "block_start", "tokens", "prompt_length",
                       "block_length", "draft_tokens", "p_draft", "judge_argmax", "judge_p", "judge_q", "judge_code",
                       "accepted", "committed_judge"}
    assert tr["proposed"].shape == (1, 6, BL) and tr["committed"].dtype == bool and tr["judge_code"].dtype == np.int8
    assert int(tr["prompt_length"]) == PLEN and int(tr["block_length"]) == BL and tr["n_pass"].tolist() == [6] and tr["block_start"].tolist() == [32]
    assert np.array_equal(tr["tokens"][0], b.tokens.numpy()) and np.array_equal(tr["valid"][0], b.valid.numpy())
    for s in range(6):
        assert np.array_equal(tr["committed"][0, s], b.committed[s].numpy()) and np.array_equal(tr["proposed"][0, s], b.proposed[s].numpy())
        assert np.allclose(tr["confidence"][0, s], b.confidence[s].numpy())
    assert np.array_equal(tr["draft_tokens"][0], b.draft_tokens.numpy()) and np.array_equal(tr["judge_argmax"][0], b.judge_argmax.numpy())
    assert np.array_equal(tr["judge_code"][0], b.judge_code.numpy()) and np.array_equal(tr["accepted"][0], b.accepted.numpy())
    assert np.array_equal(tr["committed_judge"][0], b.committed_judge.numpy())
    assert np.allclose(tr["judge_p"][0][b.valid.numpy()], b.judge_p.numpy()[b.valid.numpy()]) and np.isnan(tr["judge_p"][0][~b.valid.numpy()]).all()
    assert np.array_equal(tr["accepted"][0], np.isin(tr["judge_code"][0], [CODE_ACCEPT_EQ, CODE_ACCEPT_Q]))
    # replay: the generated ids are the valid tokens; committed masks over passes + judge cover every valid slot exactly once
    gen = np.concatenate([tr["tokens"][i][tr["valid"][i]] for i in range(tr["tokens"].shape[0])])
    assert gen.tolist() == res.generated_ids()[0][res.generated_ids()[0] != MASK].tolist()
    cover = tr["committed"][0].sum(0) + tr["committed_judge"][0] + tr["accepted"][0]
    assert np.array_equal(cover, tr["valid"][0].astype(int))
    # stock mode: the base arrays equal run_research_baselines.save_trace on the same result
    import run_research_baselines as RRB
    moe2 = meter_for(seeded_fn(21))
    res2 = da_generate(moe2, None, prompt_ids(37), mask_id=MASK, gen_length=64, block_length=BL, draft=None,
                       stopping_criteria_idx=EOS_IDS)
    p_ours, p_ref = str(tmp_path / "ours.npz"), str(tmp_path / "ref.npz")
    save_trace(p_ours, res2, BL); RRB.save_trace(p_ref, res2, BL)
    a, r = load_trace(p_ours), load_trace(p_ref)
    for k in r:
        assert k in a and a[k].shape == r[k].shape and a[k].dtype == r[k].dtype, k
        assert np.array_equal(a[k], r[k], equal_nan=True), k
    assert (a["draft_tokens"] == -1).all() and (a["judge_code"] == 0).all() and not a["accepted"].any() and np.isnan(a["judge_p"]).all()


# --------------------------------------------------------------------------
# MoE meter: unique experts per layer-forward from moe_instr's SINK
# --------------------------------------------------------------------------
def test_moe_meter_counts_unique_experts_via_moe_instr():
    from tests.test_draftalign import FakeMoEModel
    from tests.test_desvote import L, K
    from sdar_harness.moe_instr import enable_instrumented_moe, disable_instrumented_moe, SINK
    from draftalign.moe_probe import RouterCapture
    model = FakeMoEModel(vocab=V)
    assert enable_instrumented_moe(model) == L
    SINK.disarm(); SINK.verify = False
    try:
        rc = RouterCapture(model, BL)
        moe = MoEMeter(SeqCache(model, BL, "moe"), SINK)
        assert moe.prefill(prompt_ids(40)) == 32 and len(moe.records) == 1 and moe.records[0][0] == 0 and moe.records[0][4] == 32
        blk = torch.full((BL,), MASK); blk[:8] = 1
        rc.arm()
        moe.block(blk, 32, store_kv=False, kind="judge")
        cap = rc.disarm()
        code, ms, uniq, lf, npos, psize = moe.records[-1]
        assert psize == 0                                          # no PoolCtrl on this meter
        assert code == 3 and lf == L and npos == BL and ms >= 0
        assert uniq == int(cap.unique_per_layer().sum()) and L * K <= uniq <= L * model.config.num_experts
        assert SINK.active is None
        # not instrumented / no sink -> zeros, but the forward still runs and is timed
        moe0 = MoEMeter(SeqCache(model, BL, "moe0"), None)
        moe0.prefill(prompt_ids(40)); moe0.block(blk, 32, store_kv=False, kind="denoise")
        assert moe0.records[-1][2:4] == [0, 0] and moe0.records[-1][5] == 0
        by = moe.by_kind()
        assert by["judge"]["n"] == 1 and by["judge"]["uniq"] == uniq and by["prefill"]["lf"] == L
        rc.close()
    finally:
        disable_instrumented_moe(model)


# --------------------------------------------------------------------------
# runner helpers + analyzer on synthetic records
# --------------------------------------------------------------------------
def test_runner_helpers_and_analyzer(tmp_path):
    import draftalign.run_da_decode as R
    import draftalign.analyze_da as A
    ns = lambda **kw: SimpleNamespace(**{"stock": False, "draft": "s8", "threshold": .95, "tau_commit": .95, "tau_acc": .70,
                                         "tau_acc_q": .50, "rejudge": False, "rejudge_k": 8, "drafter": "d32", **kw})
    assert R.cell_name("gsm8k", ns(stock=True)) == "gsm8k_B32_t95_stock"
    assert R.cell_name("gsm8k", ns()) == "gsm8k_B32_t95_dv-s8_c95_a70_q50"
    assert R.cell_name("mbpp", ns(draft="dyn90", tau_acc_q=1.0, rejudge=True)) == "mbpp_B32_t95_dv-dyn90_c95_a70_q100_rj8"
    assert R.cell_name("gsm8k", ns(judge="blind", tau_acc_q=.30)) == "gsm8k_B32_t95_dv-s8_blind_q30"
    assert R.cell_name("gsm8k", ns(judge="blind", tau_acc_q=.30, tau_draft=.5, rejudge=True)) == "gsm8k_B32_t95_dv-s8_blind_q30_d50_rj"
    a = R.resolve_args(R.build_parser().parse_args(["--judge", "blind"]))
    assert a.tau_acc_q == .30 and a.tau_draft == 0.0 and not a.rejudge_redraft
    assert R.resolve_args(R.build_parser().parse_args([])).tau_acc_q == .50
    assert R.resolve_args(R.build_parser().parse_args(["--judge", "blind", "--tau-acc-q", ".4"])).tau_acc_q == .4
    with pytest.raises(NotImplementedError):
        R.resolve_args(R.build_parser().parse_args(["--judge", "blind", "--rejudge", "--rejudge-redraft"]))
    assert "." not in R.cell_name("gsm8k", ns(threshold=.9))
    # synthetic DV records + a synthetic stock reference on the same items
    tok = FakeTok()
    cell = "gsm8k_B32_t95_dv-s8_c95_a70_q50"
    cdir = tmp_path / cell; cdir.mkdir()
    ref_dir = tmp_path / "ref"; ref_dir.mkdir()
    recs, refs = [], []
    for i, idx in enumerate([9, 12, 15]):
        moe = meter_for(seeded_fn(30 + i, eos_at=(60,)))
        dr = SeqCache(ScriptedLM(seeded_fn(40 + i)), BL, "drafter")
        rec, res = decode_item(moe, dr, tok, item(idx), prompt_ids(37), mask_id=MASK, eos_ids=EOS_IDS, gen_length=64,
                               block_length=BL, threshold=.95, draft="s8")
        rec.update({"cell": cell, "config_hash": "h", "model_revision": "r", "correct": i % 2})
        moe_s = meter_for(seeded_fn(30 + i, eos_at=(60,)))
        ref, _ = decode_item(moe_s, None, tok, item(idx), prompt_ids(37), mask_id=MASK, eos_ids=EOS_IDS, gen_length=64,
                             block_length=BL, threshold=.95, draft=None)
        ref.update({"cell": "gsm8k_B32_t95_stock", "correct": 1})
        recs.append(rec); refs.append(ref)
    with open(cdir / f"{cell}.calibvalid.shard0of1.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    open(cdir / f"{cell}.calibvalid.shard0of1.jsonl.done", "w").write("ok\n")
    with open(ref_dir / "gsm8k_B32_t95_stock.calibvalid.shard0of1.jsonl", "w") as fh:
        for r in refs:
            fh.write(json.dumps(r) + "\n")
    loaded = R.load_ref_records(str(ref_dir), "gsm8k_B32_t95_stock", "calibvalid")
    assert sorted(loaded) == [9, 12, 15]
    m = R.ref_compare(refs[0], loaded[9])
    assert all(m.values()) and set(m) == {"gen_ids", "passes_per_block", "commits_per_pass", "n_forward", "prompt_tokens"}
    assert not R.ref_compare(recs[0], loaded[9])["n_forward"] or recs[0]["n_forward"] == loaded[9]["n_forward"]
    cells = A.load_cell(str(cdir))
    assert set(cells) == {(cell, "calibvalid")} and cells[(cell, "calibvalid")]["complete"]
    s = A.summarise(cells[(cell, "calibvalid")]["recs"], "gsm8k")
    assert s["n"] == 3 and abs(s["accuracy"] - 100 / 3) < 1e-9 and s["moe_forwards_per_token_by_kind"]["judge"] > 0
    assert s["drafter_forwards_per_token"] > 0 and 0 <= s["accepted_frac"] <= 1 and s["passes_per_block"] > 0
    pc = A.paired_accuracy(s["correct"], {r["idx"]: r["correct"] for r in refs})
    assert pc["n"] == 3 and abs(pc["delta"] - (100 / 3 - 100)) < 1e-9
    A.main([str(cdir), "--ref-dir", str(ref_dir), "--out", str(tmp_path / "summary.json")])
    out = json.load(open(tmp_path / "summary.json"))
    assert f"{cell}.calibvalid" in out["cells"] and "ref:gsm8k_B32_t95_stock.calibvalid" in out["cells"]
    assert out["paired_vs_ref"][f"{cell}.calibvalid"]["n"] == 3


# --------------------------------------------------------------------------
# blind judge: q read from the first STOCK pass, zero extra MoE forwards
# --------------------------------------------------------------------------
B_COMMIT_A, B_COMMIT_D, B_ACC, B_EDGE, B_GATE = 52, 53, 54, 55, 56


def blind_judge_fn(variant="normal"):
    def fn(pos, tok, is_mask, ids, cached, pkv):
        if not is_mask:
            return onehot_row(tok)
        n_mask = sum(1 for t in ids if t == MASK)
        if variant == "forced":
            # nothing passes .95: the stock rule force-commits the best slot (54 -> a, p .60) whose q(D) = .40 > .30
            if pos == B_ACC:
                return logprob_row({a_tok(pos): .60, g_tok(pos): .40})
            r = .50 + .001 * (pos % 10)
            return logprob_row({a_tok(pos): r, g_tok(pos): .05, 0: 1 - r - .05})
        if n_mask <= 3:
            return logprob_row({g_tok(pos): .50, a_tok(pos): .45, 0: .05})       # late passes: q = .50
        if pos == B_COMMIT_A:
            return logprob_row({a_tok(pos): .97, g_tok(pos): .02, 0: .01})       # stock commits a (!= D)
        if pos == B_COMMIT_D:
            return logprob_row({g_tok(pos): .96, a_tok(pos): .03, 0: .01})       # stock commits D itself
        if pos == B_ACC:
            return logprob_row({a_tok(pos): .60, g_tok(pos): .40})               # open; q .40 > .30 -> accept D
        if pos == B_EDGE:
            return logprob_row({a_tok(pos): .71, g_tok(pos): .29})               # q .29 -> stays [MASK]
        if pos == B_GATE:
            return logprob_row({a_tok(pos): .55, g_tok(pos): .35, 0: .10})       # q .35 -> accept unless tau_draft gates it
        r = .60 + .001 * (pos % 10)
        return logprob_row({a_tok(pos): r, g_tok(pos): .05, 0: 1 - r - .05})     # stays [MASK]
    return fn


def blind_drafter_fn(pos, tok, is_mask, ids, cached, pkv):
    if not is_mask:
        return onehot_row(tok)
    if pos == B_GATE:
        return logprob_row({g_tok(pos): .45, a_tok(pos): .40, 0: .15})           # D = g with p_draft .45
    return onehot_row(g_tok(pos))


def run_blind(judge_fn, drafter_fn=blind_drafter_fn, **kw):
    moe = meter_for(judge_fn)
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    ids = prompt_ids(PLEN)
    res = da_generate(moe, dr, ids, mask_id=MASK, gen_length=GEN, block_length=BL, threshold=.95, draft="s8",
                      judge="blind", stopping_criteria_idx=EOS_IDS, **{"tau_acc_q": .30, **kw})
    return moe, dr, ids, res


@pytest.mark.parametrize("seed,plen,gen_length", [(0, 37, 64), (5, 50, 96)])
def test_blind_disabled_reproduces_stock_twin(seed, plen, gen_length):
    """(i) judge='blind' with the draft disabled, or with tau_acc_q = 1.0 (also with rejudge): tokens, passes and
    the MoE forward sequence are the stock twin's, bit for bit; the drafter still runs and is accounted."""
    ids = prompt_ids(plen)
    stock = meter_for(seeded_fn(seed))
    rs = da_generate(stock, None, ids, mask_id=MASK, gen_length=gen_length, block_length=BL, draft=None,
                     stopping_criteria_idx=EOS_IDS)
    for draft, kw in ((None, {}), ("s8", {"tau_acc_q": 1.0}), ("s8", {"tau_acc_q": 1.0, "rejudge": True})):
        moe = meter_for(seeded_fn(seed))
        dr = SeqCache(ScriptedLM(seeded_fn(seed + 100)), BL, "drafter") if draft else None
        r = da_generate(moe, dr, ids, mask_id=MASK, gen_length=gen_length, block_length=BL, draft=draft, judge="blind",
                        stopping_criteria_idx=EOS_IDS, **kw)
        assert torch.equal(r.x, rs.x) and r.n_forward == rs.n_forward and r.stopped_early == rs.stopped_early
        assert [b.n_steps for b in r.blocks] == [b.n_steps for b in rs.blocks]
        for a, b in zip(rs.blocks, r.blocks):
            assert a.n_forward == b.n_forward and b.n_judge == 0 and b.n_rejudge == 0 and torch.equal(a.tokens, b.tokens)
            for st in range(a.n_steps):
                assert torch.equal(a.committed[st], b.committed[st]) and torch.equal(a.proposed[st], b.proposed[st])
                assert torch.equal(a.confidence[st], b.confidence[st])
        assert len(stock.seq.model.calls) == len(moe.seq.model.calls)
        for c1, c2 in zip(stock.seq.model.calls, moe.seq.model.calls):
            assert torch.equal(c1["input_ids"], c2["input_ids"]) and c1["store_kv"] == c2["store_kv"] and c1["cached"] == c2["cached"]
        assert [x[0] for x in moe.records] == [x[0] for x in stock.records]
        if draft is None:
            assert r.judge is None and all(b.judge_code is None for b in r.blocks)
        else:
            assert r.judge == "blind" and all(b.judge_mode == "blind" and b.n_blind_accept == 0 for b in r.blocks)
            assert all(b.judge_q is not None and b.judge_argmax is not None and not b.accepted.any() for b in r.blocks)
            assert dr.n_by_kind["draft_s8"] == 8 * len(r.blocks) and dr.n_by_kind["flush"] == len(r.blocks) and dr.n_by_kind["prefill"] == 1


def test_blind_reads_q_from_the_stock_pass_and_never_overrides_commits():
    """(ii) a / p / q come from the first stock pass's own forward; acceptance only at still-open slots."""
    moe, dr, ids, res = run_blind(blind_judge_fn())
    b = res.blocks[0]
    valid = b.valid
    assert b.n_judge == 0 and b.judge_mode == "blind" and res.judge == "blind"
    # pass 1: stock commits 52 -> a and 53 -> D (both > .95); blind accepts 54 (q .40) and 56 (q .35); 55 stays (q .29)
    assert [int(i) + 32 for i in torch.nonzero(b.committed[0]).flatten()] == [B_COMMIT_A, B_COMMIT_D]
    assert b.tokens[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) and b.tokens[B_COMMIT_D - 32] == g_tok(B_COMMIT_D)
    assert b.judge_code[B_ACC - 32] == CODE_ACCEPT_Q and b.judge_code[B_GATE - 32] == CODE_ACCEPT_Q
    assert b.judge_code[B_EDGE - 32] == 0 and b.judge_code[B_COMMIT_A - 32] == 0 and b.judge_code[B_COMMIT_D - 32] == 0
    assert b.tokens[B_ACC - 32] == g_tok(B_ACC) and b.tokens[B_GATE - 32] == g_tok(B_GATE)
    assert b.n_blind_accept == 2 and int(b.accepted.sum()) == 2 and int(b.committed_judge.sum()) == 0 and not b.forced
    # the arrays are the first pass's read over every slot masked at that forward (all 12 valid slots)
    assert (b.judge_argmax[valid] >= 0).all() and (b.judge_argmax[~valid] == -1).all() and torch.isnan(b.judge_q[~valid]).all()
    assert abs(float(b.judge_q[B_ACC - 32]) - .40) < 1e-4 and abs(float(b.judge_q[B_COMMIT_A - 32]) - .02) < 1e-4
    assert abs(float(b.judge_p[B_COMMIT_A - 32]) - .97) < 1e-4 and abs(float(b.judge_q[B_EDGE - 32]) - .29) < 1e-4
    assert b.judge_argmax[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) and b.judge_argmax[B_COMMIT_D - 32] == g_tok(B_COMMIT_D)
    # zero extra MoE forwards: prefill + stock passes + flush; pass-1 input had every valid slot masked
    assert [x[0] for x in moe.records] == [0] + [1] * b.n_steps + [2] and b.n_forward == b.n_steps + 1
    first = moe.seq.model.calls[1]["input_ids"][0]
    assert (first[valid] == MASK).all() and moe.seq.model.calls[1]["cached"] == 32
    # pass-2 input holds the stock commits AND the blind accepts; 8 slots remain open -> 8 single-commit passes
    second = moe.seq.model.calls[2]["input_ids"][0]
    assert second[B_COMMIT_A - 32] == a_tok(B_COMMIT_A) and second[B_COMMIT_D - 32] == g_tok(B_COMMIT_D)
    assert second[B_ACC - 32] == g_tok(B_ACC) and second[B_GATE - 32] == g_tok(B_GATE) and second[B_EDGE - 32] == MASK
    assert int((second[valid] == MASK).sum()) == 8
    assert b.n_steps == 9 and [int(c.sum()) for c in b.committed[1:]] == [1] * 8 and b.rejudge_rounds == 0
    assert dr.n_by_kind == {"prefill": 1, "draft_s8": 8, "flush": 1} and moe.seq.cached_length() == 64 == dr.cached_length()
    # forced-commit variant: the stock rule commits 54 -> a (best p .60); q(D) = .40 > .30 must NOT override it
    moe2, dr2, _, res2 = run_blind(blind_judge_fn("forced"))
    b2 = res2.blocks[0]
    assert [int(i) + 32 for i in torch.nonzero(b2.committed[0]).flatten()] == [B_ACC] and b2.tokens[B_ACC - 32] == a_tok(B_ACC)
    assert abs(float(b2.judge_q[B_ACC - 32]) - .40) < 1e-4 and b2.judge_code[B_ACC - 32] == 0 and b2.n_blind_accept == 0
    assert b2.n_steps == 12 and torch.equal(b2.tokens[valid], torch.tensor([a_tok(p) for p in range(PLEN, 64)]))
    # record + trace carry the blind fields
    rec = DA.make_record(res, FakeTok(), item(), ids, moe, dr, EOS_IDS, MASK, 0.0)
    assert rec["judge"] == "blind" and rec["n_judge"] == 0 and rec["blind_accept_per_block"] == [2] and rec["accepted_per_block"] == [2]
    assert rec["accepted_q_per_block"] == [2] and rec["accepted_late_per_block"] == [0] and rec["committed_judge_per_block"] == [0]
    assert rec["moe_forwards_per_block"] == [10] and rec["passes_per_block"] == [9] and rec["n_denoise"] == 9


def test_blind_rejudge_rereads_q_without_redraft(tmp_path):
    moe, dr, ids, res = run_blind(blind_judge_fn(), rejudge=True)
    b = res.blocks[0]
    # passes 2-6 commit one slot each (q .05 re-read, nothing accepted); at 3 open slots the rows give q = .50:
    # the stock rule force-commits one of them (to D) and the re-read accepts the other two -> done in 7 passes
    assert b.n_steps == 7 and b.rejudge_rounds == 6 and b.rejudge_accepted == 2 and b.n_blind_accept == 4
    late = b.judge_code == CODE_ACCEPT_LATE
    assert int(late.sum()) == 2 and int(b.accepted.sum()) == 4 and torch.equal(b.tokens[late], b.draft_tokens[late])
    assert b.n_forward == b.n_steps + 1 and b.n_judge == 0 and b.n_rejudge == 0 and b.n_redraft_fwd == 0
    assert dr.n_by_kind == {"prefill": 1, "draft_s8": 8, "flush": 1}                 # no redraft
    assert [x[0] for x in moe.records] == [0] + [1] * 7 + [2]
    assert (b.tokens != MASK).all()
    path = str(tmp_path / "t.npz")
    save_trace(path, res, BL); tr = load_trace(path)
    assert np.array_equal(tr["accepted"][0], np.isin(tr["judge_code"][0], [CODE_ACCEPT_Q, CODE_ACCEPT_LATE]))
    assert np.array_equal(tr["judge_q"][0][b.valid.numpy()], b.judge_q.numpy()[b.valid.numpy()]) and tr["n_pass"].tolist() == [7]
    with pytest.raises(NotImplementedError):
        run_blind(blind_judge_fn(), rejudge=True, rejudge_redraft=True)


def test_blind_tau_draft_gate():
    """(iii) p_draft at 56 is .45: tau_draft .5 blocks the accept, .4 / off allow it; strict > on both gates."""
    moe, dr, _, res = run_blind(blind_judge_fn(), tau_draft=0.5)
    b = res.blocks[0]
    assert abs(float(b.p_draft[B_GATE - 32]) - .45) < 1e-3 and float(b.p_draft[B_ACC - 32]) > .99
    assert b.judge_code[B_GATE - 32] == 0 and b.judge_code[B_ACC - 32] == CODE_ACCEPT_Q and b.n_blind_accept == 1
    second = moe.seq.model.calls[2]["input_ids"][0]                  # pass-2 input: the gated slot is still open
    assert second[B_GATE - 32] == MASK and second[B_ACC - 32] == g_tok(B_ACC) and b.committed[0][B_GATE - 32] == False
    assert run_blind(blind_judge_fn(), tau_draft=0.0)[3].blocks[0].n_blind_accept == 2
    assert run_blind(blind_judge_fn(), tau_draft=0.4)[3].blocks[0].n_blind_accept == 2
    cur = torch.tensor([MASK, MASK, 7, MASK]); D = torch.tensor([1, 2, 3, 4]); open_ = torch.tensor([True, True, False, True])
    q = torch.tensor([.30, .31, .90, float("nan")]); pd = torch.tensor([.5, .5, .5, .5])
    new, acc = DA.blind_accept(cur, q, D, open_, .30)
    assert acc.tolist() == [False, True, False, False] and new.tolist() == [MASK, 2, 7, MASK]
    assert not DA.blind_accept(cur, q, D, open_, .30, pd, .5)[1].any()
    assert DA.blind_accept(cur, q, D, open_, .30, pd, .49)[1].tolist() == [False, True, False, False]
