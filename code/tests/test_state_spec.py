"""CPU tests for sdar_harness.state_spec (transition contract, children, acceptance)
and sdar_harness.rowinv (fixed-row expert path, batch-padding wrapper).

Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_state_spec.py
"""
from __future__ import annotations

import os, sys
import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from sdar_harness.decode import sample_with_confidence, get_num_transfer_tokens   # noqa: E402
from sdar_harness.state_spec import (derive_transition, propose_children, child_accepted,   # noqa: E402
                                     transition_key, summarize_prev, MASK_ID)
from sdar_harness import rowinv as RI                                                        # noqa: E402
from sdar_harness.desvote import CTRL                                                        # noqa: E402
from tests.test_desvote import FakeMoE, stock_forward, E, K, H, N, SEED                      # noqa: E402


def reference_rule(logits, cur_x, mask_index, threshold, need):
    """Verbatim logic of decode.block_diffusion_generate's commit step (one row)."""
    x0, x0_p = sample_with_confidence(logits, temperature=1.0, top_k=1, top_p=1.0, greedy_exact=True)
    x0 = torch.where(mask_index, x0, cur_x)
    confidence = torch.where(mask_index, x0_p, torch.tensor(float("-inf"), dtype=x0_p.dtype))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(confidence.shape[0]):
        high_conf = confidence[j] > threshold
        if high_conf.sum() >= need:
            transfer_index[j] = high_conf
        else:
            _, idx = torch.topk(confidence[j], need)
            transfer_index[j, idx] = True
    nxt = cur_x.clone()
    nxt[transfer_index] = x0[transfer_index]
    return nxt, transfer_index


@pytest.mark.parametrize("seed", range(40))
def test_derive_transition_matches_reference_rule(seed):
    g = torch.Generator().manual_seed(seed)
    L, V = 32, 50
    cur = torch.randint(0, V, (1, L), generator=g)
    valid = torch.arange(L) >= int(torch.randint(0, 6, (1,), generator=g))
    open_mask = (torch.rand(1, L, generator=g) < 0.6) & valid.unsqueeze(0)
    if open_mask.sum() == 0:
        open_mask[0, L - 1] = True
    cur[open_mask] = MASK_ID
    logits = torch.randn(1, L, V, generator=g) * (6.0 if seed % 2 else 1.0)   # sharp / flat -> above / below threshold
    if seed % 5 == 0:                                                           # exact ties in confidence
        logits[0, 3] = logits[0, 5]
    thr = 0.95 if seed % 3 else 0.5
    need = 1
    tr = derive_transition(logits, cur, open_mask, need, thr)
    ref_next, ref_transfer = reference_rule(logits, cur, open_mask, thr, need)
    assert torch.equal(tr["next"], ref_next)
    assert torch.equal(tr["transfer"], ref_transfer)
    assert not (tr["transfer"] & ~open_mask).any(), "closed slot committed"
    assert len(tr["positions"]) >= 1
    assert tr["tokens"] == [int(tr["x0"][0, p]) for p in tr["positions"]]


def test_derive_transition_no_open_slots():
    cur = torch.zeros(1, 8, dtype=torch.long)
    tr = derive_transition(torch.randn(1, 8, 10), cur, torch.zeros(1, 8, dtype=torch.bool), 1, 0.9)
    assert tr["positions"] == [] and torch.equal(tr["next"], cur)


def test_num_transfer_is_one_per_pass_when_steps_equal_block():
    assert get_num_transfer_tokens(32, 32).tolist() == [1] * 32
    assert get_num_transfer_tokens(4, 4).tolist() == [1] * 4


def test_propose_children_ranking_and_skips():
    L = 8
    cur = torch.full((1, L), MASK_ID, dtype=torch.long); cur[0, :2] = 7          # slots 0,1 closed
    open_mask = cur == MASK_ID
    conf = torch.tensor([0.99, 0.99, 0.5, 0.9, 0.9, 0.1, 0.7, 0.2])            # slots 3 and 4 tie -> 3 first
    argmax = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    prev = {"conf": conf, "argmax": argmax, "step": 0}
    kids = propose_children(cur, open_mask, prev, ks=(1, 2, 3, 4, 7))
    assert [c["k"] for c in kids] == [1, 2, 3, 4]                              # k=7 > n_open=6 skipped
    assert kids[0]["slots"] == [3] and kids[0]["tokens"] == [4]
    assert kids[1]["slots"] == [3, 4] and kids[1]["tokens"] == [4, 5]
    assert kids[2]["slots"] == [3, 4, 6]
    assert kids[3]["slots"] == [3, 4, 6, 2]
    for c in kids:
        assert (c["state"][0, c["slots"]] == torch.tensor(c["tokens"])).all()
        assert not c["open_mask"][0, c["slots"]].any()
        assert torch.equal(c["state"][0, :2], cur[0, :2])
    # a MASK guess is never proposed
    argmax2 = argmax.clone(); argmax2[3] = MASK_ID
    kids2 = propose_children(cur, open_mask, {"conf": conf, "argmax": argmax2, "step": 0}, ks=(1, 2))
    assert kids2 == []


def test_child_accepted_requires_whole_transition():
    child = {"slots": [3, 4], "tokens": [4, 5]}
    assert child_accepted(child, {"positions": [4, 3], "tokens": [5, 4]})
    assert not child_accepted(child, {"positions": [3], "tokens": [4]})            # partial
    assert not child_accepted(child, {"positions": [3, 4], "tokens": [4, 6]})     # wrong token
    assert not child_accepted(child, {"positions": [3, 4, 5], "tokens": [4, 5, 6]})  # extra commit
    assert transition_key({"positions": [3, 4], "tokens": [4, 5]}) == ((3, 4), (4, 5))


# ---------------------------------------------------------------------------
# rowinv
# ---------------------------------------------------------------------------
def _fake_moe():
    torch.manual_seed(SEED)
    blk = FakeMoE(H, E, K)
    blk._layer_idx = 0
    return blk


def test_rowinv_moe_equals_stock_and_is_row_independent():
    blk = _fake_moe()
    CTRL.M = None; CTRL.recording = False; CTRL.log = None; CTRL.reset_cost()
    old = RI.CFG.m_e; RI.CFG.m_e = 4
    try:
        x = torch.randn(1, N, H); x2 = torch.randn(1, N, H)
        ref, rl = stock_forward(blk, x)
        y, rl2 = RI.rowinv_moe_forward(blk, x)
        assert torch.allclose(y, ref, atol=1e-5) and torch.equal(rl, rl2)
        yb, _ = RI.rowinv_moe_forward(blk, torch.cat([x, x2], 0))
        assert torch.allclose(yb[0:1], y, atol=1e-6)
        assert CTRL.layer_forwards == 2 and CTRL.unique_sum > 0
    finally:
        RI.CFG.m_e = old


def test_rowinv_expert_chunking_covers_all_rows():
    blk = _fake_moe()
    lin = blk.experts[0]
    x = torch.randn(11, H)
    y = RI._expert_padded(lin, x, 4)
    assert y.shape == (11, H) and torch.allclose(y, lin(x), atol=1e-6)
    assert torch.allclose(RI._expert_padded(lin, x[:0], 4).shape[0] * torch.ones(1), torch.zeros(1))


class _FakeCausal(torch.nn.Module):
    """Stand-in for the SDAR model: logits depend on the row; cache stores K=V=input rows."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.calls = []

    def forward(self, *, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=True, store_kv=False, **kw):
        from transformers.modeling_outputs import MoeCausalLMOutputWithPast
        B, L = input_ids.shape
        self.calls.append((B, L, store_kv, past_key_values.get_seq_length() if past_key_values is not None else None))
        if past_key_values is not None and store_kv:
            k = input_ids.float().view(B, 1, L, 1).expand(B, 2, L, 3).contiguous()
            for l in range(2):
                past_key_values.update(k, k, l)
        logits = (input_ids.float().unsqueeze(-1) * torch.arange(1, 5)).repeat(1, 1, 1)
        return MoeCausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


def test_rowinv_forward_wrapper_pads_and_slices():
    from transformers.cache_utils import DynamicCache
    m = _FakeCausal()
    RI.CFG.enabled = False
    m._rowinv_orig_forward = m.forward
    m.forward = RI._make_forward(m, m.forward)
    RI.CFG.mode = "pad"; RI.CFG.bpad = 3; RI.CFG.enabled = True
    try:
        cache = DynamicCache()
        x = torch.tensor([[5, 6, 7, 8]])
        out = m(input_ids=x, attention_mask=torch.ones(1, 4, 4), position_ids=torch.arange(4)[None], past_key_values=cache, store_kv=True)
        assert out.logits.shape[0] == 1 and m.calls[-1][0] == 3          # padded to 3 rows, sliced back
        assert cache.get_seq_length() == 4 and cache[0][0].shape[0] == 1   # only the real row stored
        assert torch.equal(cache[0][0][0, 0, :, 0], x[0].float())
        x2 = torch.tensor([[1, 2, 3, 4], [9, 9, 9, 9]])
        out2 = m(input_ids=x2, attention_mask=torch.ones(2, 4, 8), position_ids=torch.arange(4, 8)[None].expand(2, -1),
                 past_key_values=cache, store_kv=False)
        assert out2.logits.shape[0] == 2 and m.calls[-1][0] == 3 and m.calls[-1][3] == 4
        assert cache.get_seq_length() == 4                                  # untouched without store_kv
        assert torch.equal(out2.logits[1, :, 0], torch.full((4,), 9.0))
        with pytest.raises(AssertionError):
            m(input_ids=torch.zeros(4, 4, dtype=torch.long), past_key_values=cache)
    finally:
        RI.CFG.enabled = False


class _FakeAttn(torch.nn.Module):
    """Mimics SDARMoeAttention.forward's signature; output depends on the row's own cache row."""

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
        cos, sin = position_embeddings
        k, v = past_key_value[0]
        assert k.shape[0] == hidden_states.shape[0] == cos.shape[0]
        return hidden_states * cos.mean() + k.sum(dim=(1, 2, 3))[:, None, None], None


def test_rowinv_perrow_attention_loops_rows_with_row_caches():
    from transformers.cache_utils import DynamicCache
    RI.CFG.enabled = False
    fwd = RI._make_attn_forward(_FakeAttn.forward)
    a = _FakeAttn()
    cache = DynamicCache()
    cache.update(torch.arange(2.0).view(2, 1, 1, 1).expand(2, 2, 3, 4).contiguous(), torch.zeros(2, 2, 3, 4), 0)
    h = torch.randn(2, 5, 4); cos = torch.ones(2, 5, 4); sin = torch.zeros(2, 5, 4)
    ref = torch.cat([_FakeAttn.forward(a, h[r:r + 1], (cos[r:r + 1], sin[r:r + 1]), None, RI._RowView(cache, r))[0] for r in range(2)], 0)
    out, _ = fwd(a, h, (cos, sin), None, cache, None)
    assert torch.equal(out, ref)
    assert torch.allclose(out[1] - out[0], (h[1] - h[0]) + 24.0)       # row 1 saw its own cache row (sum 24), row 0 its own (sum 0)
    single, _ = fwd(a, h[:1], (cos[:1], sin[:1]), None, RI._RowView(cache, 0), None, store_kv=False)
    assert torch.equal(single, ref[:1])                                # B == 1 takes the stock path untouched
    with pytest.raises(RuntimeError):
        RI._RowView(cache, 0).update(None, None, 0)


def test_rowinv_perrow_lm_head_and_gate_loop_rows():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 6)
    f = RI._make_lm_head_forward(lin)
    x = torch.randn(3, 5, 4)
    assert torch.allclose(f(x), lin(x), atol=1e-6) and torch.equal(f(x[:1]), lin(x[:1]))
    blk = _fake_moe()
    CTRL.M = None; CTRL.recording = False; CTRL.log = None; CTRL.reset_cost()
    old = (RI.CFG.mode, RI.CFG.m_e); RI.CFG.mode, RI.CFG.m_e = "perrow", 4
    try:
        x = torch.randn(2, N, H)
        y, rl = RI.rowinv_moe_forward(blk, x)
        ref, rl_ref = stock_forward(blk, x)
        assert torch.allclose(y, ref, atol=1e-5) and torch.allclose(rl, rl_ref, atol=1e-6)
        y0, _ = RI.rowinv_moe_forward(blk, x[:1])
        assert torch.allclose(y[0:1], y0, atol=1e-6)
    finally:
        RI.CFG.mode, RI.CFG.m_e = old


def test_rowinv_perrow_norm_loops_rows():
    class N(torch.nn.Module):
        def forward(self, x):
            return x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
    f = RI._make_norm_forward(N.forward)
    n = N()
    x = torch.randn(3, 4, 8)
    ref = torch.cat([N.forward(n, x[r:r + 1]) for r in range(3)], 0)
    assert torch.equal(f(n, x), ref) and torch.equal(f(n, x[:1]), N.forward(n, x[:1]))
    x4 = torch.randn(2, 4, 3, 8)
    assert torch.equal(f(n, x4), N.forward(n, x4))                   # 4-D (q/k norm) untouched
