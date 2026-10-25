"""CPU tests for code/draftalign (AB probe). No real model is loaded.

Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_draftalign.py
"""
from __future__ import annotations

import json, os, sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from sdar_harness.decode import get_num_transfer_tokens                                  # noqa: E402
from draftalign.seqcache import SeqCache, block_tril_mask                                # noqa: E402
from draftalign.drafter import SETTINGS, draft_block, masked_base                        # noqa: E402
from draftalign.moe_probe import (Capture, RouterCapture, build_block_state, slot_metrics,   # noqa: E402
                                  top_m_pool, pool_coverage, unique_recall, coverage_all_kinds, KINDS)
from tests.test_desvote import FakeMoE, E, K, H, L, N, SEED                              # noqa: E402

MASK_ID = 151669
BL = 32


# --------------------------------------------------------------------------
# fake models (cache-aware, like test_state_spec._FakeCausal)
# --------------------------------------------------------------------------
class FakeSeqModel(nn.Module):
    """Records every call; stores K=V=input ids in the cache when store_kv. Subclasses define logits."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def _record(self, input_ids, attention_mask, position_ids, past_key_values, store_kv):
        self.calls.append(dict(input_ids=input_ids.clone(), mask=None if attention_mask is None else attention_mask.clone(),
                               pos=None if position_ids is None else position_ids.clone(), cache=past_key_values,
                               store_kv=store_kv, cached=past_key_values.get_seq_length() if past_key_values is not None else None))
        if past_key_values is not None and store_kv:
            B, S = input_ids.shape
            k = input_ids.float().view(B, 1, S, 1).expand(B, 2, S, 3).contiguous()
            for l in range(2):
                past_key_values.update(k, k, l)


class FakeDrafter(FakeSeqModel):
    """Logits favour token == absolute position; `strong` positions get p~1, others p~0."""

    def __init__(self, vocab=MASK_ID + 2, strong=None, mask_fav=()):
        super().__init__()
        self.vocab, self.strong, self.mask_fav = vocab, strong, set(mask_fav)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, store_kv=None, **kw):
        self._record(input_ids, attention_mask, position_ids, past_key_values, store_kv)
        S = input_ids.shape[1]
        pos = position_ids[0]
        logits = torch.zeros(1, S, self.vocab)
        for s in range(S):
            p = int(pos[s])
            hi = 20.0 if (self.strong is None or p in self.strong) else 0.5
            logits[0, s, p] = hi
            if p in self.mask_fav:
                logits[0, s, MASK_ID] = hi + 1.0          # raw argmax would be [MASK]
        return SimpleNamespace(logits=logits)


class FakeMoEModel(FakeSeqModel):
    """model.model.layers[i].mlp = FakeMoE (gate + experts); logits depend on the ids AND the cache."""

    def __init__(self, vocab=MASK_ID + 2, n_layers=L, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, H)
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=FakeMoE(H, E, K)) for _ in range(n_layers)])
        self.lm_head = nn.Linear(H, vocab)
        self.config = SimpleNamespace(num_experts=E, num_experts_per_tok=K, num_hidden_layers=n_layers)
        self.vocab = vocab

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, store_kv=None, **kw):
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        ctx = 0.0
        if cached:
            ctx = past_key_values[0][0][0, 0, :, 0].mean() / 1e5           # prefix-dependent term
        self._record(input_ids, attention_mask, position_ids, past_key_values, store_kv)
        h = self.embed(input_ids) + ctx
        for layer in self.model.layers:
            h = h + layer.mlp(h)[0]
        return SimpleNamespace(logits=self.lm_head(h))


def make_block(plen_tail: int, seed=0):
    """final / valid for a block whose first `plen_tail` positions are fixed prompt tokens."""
    g = torch.Generator().manual_seed(seed)
    final = torch.randint(1000, 5000, (BL,), generator=g)
    valid = torch.arange(BL) >= plen_tail
    return final, valid


# --------------------------------------------------------------------------
# 1. block states
# --------------------------------------------------------------------------
def test_build_block_state_modes_and_halves_parity():
    final, valid = make_block(3)
    fill = 200 + torch.arange(BL)
    ids, masked = build_block_state(final, valid, None, "mask", MASK_ID)
    assert torch.equal(masked, valid) and torch.equal(ids[~valid], final[~valid]) and (ids[valid] == MASK_ID).all()
    ids, masked = build_block_state(final, valid, fill, "full", MASK_ID)
    assert not masked.any() and torch.equal(ids[valid], fill[valid]) and torch.equal(ids[~valid], final[~valid])
    ia, ma = build_block_state(final, valid, fill, "half_a", MASK_ID)
    ib, mb = build_block_state(final, valid, fill, "half_b", MASK_ID)
    even = torch.arange(BL) % 2 == 0
    assert torch.equal(ma, valid & even) and torch.equal(mb, valid & ~even)
    assert torch.equal(ma | mb, valid) and not (ma & mb).any()
    assert torch.equal(ia[valid & ~even], fill[valid & ~even]) and torch.equal(ib[valid & even], fill[valid & even])
    assert torch.equal(ia[~valid], final[~valid]) and torch.equal(ib[~valid], final[~valid])
    # a fill token that IS the mask id is reported as masked
    fill2 = fill.clone(); fill2[5] = MASK_ID
    ids, masked = build_block_state(final, valid, fill2, "full", MASK_ID)
    assert masked.sum() == 1 and masked[5]
    assert torch.equal(masked_base(final, valid, MASK_ID), build_block_state(final, valid, None, "mask", MASK_ID)[0])


# --------------------------------------------------------------------------
# 2. SeqCache: prefill geometry, block forwards, flush
# --------------------------------------------------------------------------
def test_seqcache_prefill_block_flush_geometry():
    m = FakeDrafter()
    seq = SeqCache(m, BL, "fake")
    prompt = torch.arange(70).unsqueeze(0)
    assert seq.prefill(prompt) == 64 and seq.prefill_length == 64 and seq.cached_length() == 64
    c = m.calls[-1]
    assert torch.equal(c["input_ids"], prompt[:, :64]) and c["store_kv"] is True
    assert c["mask"].shape == (1, 64, 64) and c["mask"][0, 0, 40] == 0 and c["mask"][0, 40, 0] == 1 and c["mask"][0, 40, 40] == 1
    assert torch.equal(c["mask"], block_tril_mask(2, BL, torch.device("cpu")))
    assert torch.equal(c["pos"][0], torch.arange(64))
    blk = torch.full((BL,), MASK_ID); blk[:6] = prompt[0, 64:70]
    lg = seq.block_forward(blk, 64, store_kv=False, kind="probe")
    assert lg.shape == (1, BL, m.vocab) and seq.cached_length() == 64
    c = m.calls[-1]
    assert c["store_kv"] is False and torch.equal(c["pos"][0], torch.arange(64, 96)) and c["mask"].shape == (1, 32, 96) and bool(c["mask"].all())
    seq.flush(blk, 64)
    assert seq.cached_length() == 96 and m.calls[-1]["store_kv"] is True
    with pytest.raises(AssertionError):
        seq.block_forward(blk, 64, store_kv=False, kind="probe")       # cache moved on
    seq.block_forward(blk, 96, store_kv=False, kind="probe")
    assert seq.n_by_kind == {"prefill": 1, "probe": 2, "flush": 1} and len(seq.timings) == 4
    # short prompt: nothing prefilled, first block starts at 0
    seq.reset(); assert seq.prefill(torch.arange(10).unsqueeze(0)) == 0 and seq.cached_length() == 0
    seq.block_forward(blk, 0, store_kv=False, kind="probe")
    assert seq.cached_length() == 0


# --------------------------------------------------------------------------
# 3. drafter commit loop: static schedules and the dynamic rule
# --------------------------------------------------------------------------
def _run_draft(setting, plen_tail, strong=None, mask_fav=()):
    m = FakeDrafter(strong=strong, mask_fav=mask_fav)
    seq = SeqCache(m, BL, "fake")
    seq.prefill(torch.arange(32 + plen_tail).unsqueeze(0))
    final, valid = make_block(plen_tail)
    final[~valid] = torch.arange(32, 32 + plen_tail)
    out = draft_block(seq, final, valid, 32, setting, MASK_ID)
    return out, m, final, valid


@pytest.mark.parametrize("setting,plen_tail", [("s4", 8), ("s8", 8), ("s8", 29), ("s4", 0)])
def test_static_schedule_follows_get_num_transfer_tokens(setting, plen_tail):
    out, m, final, valid = _run_draft(setting, plen_tail)
    n_open = int(valid.sum()); K_ = SETTINGS[setting]["steps"]
    steps = min(K_, n_open)
    sched = get_num_transfer_tokens(n_open, steps).tolist()
    calls = [c for c in m.calls if c["store_kv"] is False]
    assert out["n_fwd"] == steps == len(calls) == len(out["ms"])
    remaining = n_open
    for s, c in enumerate(calls):
        assert int((c["input_ids"][0] == MASK_ID).sum()) == remaining
        assert torch.equal(c["input_ids"][0][~valid], final[~valid])
        assert torch.equal(c["pos"][0], torch.arange(32, 64)) and c["cached"] == 32
        remaining -= sched[s]
    assert remaining == 0
    assert torch.equal(out["tokens"][valid], torch.arange(32, 64)[valid]) and torch.equal(out["tokens"][~valid], final[~valid])
    assert (out["p_commit"][valid] > 0.99).all() and (out["p_commit"][~valid] == 0).all()
    # p_final_last: prob of `final` at every slot from the LAST forward (final != drafted token here -> tiny)
    assert (out["p_final_last"][valid] < 1e-3).all()
    assert out["raw_mask_argmax"] == 0


def test_dynamic_threshold_commits_high_conf_then_one_per_step():
    strong = {36, 40}
    out, m, final, valid = _run_draft("dyn90", 27, strong=strong)       # valid = 27..31 -> 5 open slots
    assert int(valid.sum()) == 5
    calls = [c for c in m.calls if c["store_kv"] is False]
    # positions 59..63 are open; none strong -> exactly one commit per step
    assert out["n_fwd"] == 5 == len(calls) and [int((c["input_ids"][0] == MASK_ID).sum()) for c in calls] == [5, 4, 3, 2, 1]
    assert torch.equal(out["tokens"][valid], torch.arange(32, 64)[valid]) and (out["p_commit"][valid] < 0.9).all()
    # now with strong slots: step 1 commits all of them at once
    m2 = FakeDrafter(strong={59, 61, 62}); seq = SeqCache(m2, BL, "fake"); seq.prefill(torch.arange(59).unsqueeze(0))
    final, valid = make_block(27)
    out2 = draft_block(seq, final, valid, 32, "dyn90", MASK_ID)
    calls = [c for c in m2.calls if c["store_kv"] is False]
    assert out2["n_fwd"] == 3 and [int((c["input_ids"][0] == MASK_ID).sum()) for c in calls] == [5, 2, 1]
    assert (out2["p_commit"][[27, 29, 30]] > 0.99).all() and (out2["p_commit"][[28, 31]] < 0.9).all()
    assert torch.equal(out2["tokens"][valid], torch.arange(32, 64)[valid])


def test_drafter_mask_logit_is_suppressed_and_counted():
    out, m, final, valid = _run_draft("s4", 8, mask_fav={40, 41})
    assert (out["tokens"] != MASK_ID).all() and torch.equal(out["tokens"][valid], torch.arange(32, 64)[valid])
    assert out["raw_mask_argmax"] >= 2            # both slots are open at the first forward; counted per forward while open


# --------------------------------------------------------------------------
# 4. router capture on a fake MoE
# --------------------------------------------------------------------------
def test_router_capture_matches_stock_router_math():
    model = FakeMoEModel()
    rc = RouterCapture(model, N)
    x = torch.randn(1, N, H)
    cap = rc.arm()
    outs = [layer.mlp(x) for layer in model.model.layers]
    cap2 = rc.disarm()
    assert cap is cap2 and rc.cur is None
    for i, (_, rl) in enumerate(outs):
        w_full = F.softmax(rl, dim=1, dtype=torch.float)
        w, ids = torch.topk(w_full, K, dim=-1)
        assert torch.equal(cap.full[i], w_full) and torch.equal(cap.ids[i], ids)
        assert torch.allclose(cap.top8w[i].gather(1, ids), w / w.sum(-1, keepdim=True))
        assert torch.allclose(cap.top8w[i].sum(1), torch.ones(N))
        assert (cap.top8w[i] != 0).sum(1).tolist() == [K] * N
    # disarmed: nothing recorded, no error
    model.model.layers[0].mlp(x)
    assert torch.equal(cap.full, cap2.full)
    # arm() then a forward with the wrong number of positions -> assertion
    rc.arm()
    with pytest.raises(AssertionError):
        model.model.layers[0].mlp(x[:, :2])
    rc.cur = None
    # arm without a forward -> disarm complains (hook never fired)
    rc.arm()
    with pytest.raises(AssertionError):
        rc.disarm()
    rc.close()
    assert all(not blk.gate._forward_hooks for blk in [l.mlp for l in model.model.layers])
    # unique count per layer
    u = cap.unique_per_layer()
    for i in range(L):
        assert int(u[i]) == len(set(cap.ids[i].flatten().tolist()))


# --------------------------------------------------------------------------
# 5. votes / coverage on a hand-computed router tensor
# --------------------------------------------------------------------------
def _cap(full_rows, ids_rows):
    """One layer, bl=4, E=6, K=2."""
    full = torch.tensor(full_rows, dtype=torch.float32).unsqueeze(0)             # [1, 4, 6]
    ids = torch.tensor(ids_rows, dtype=torch.long).unsqueeze(0)                    # [1, 4, 2]
    w = full.gather(2, ids); w = w / w.sum(-1, keepdim=True)
    top8w = torch.zeros_like(full).scatter_(2, ids, w)
    return Capture(full=full, top8w=top8w, ids=ids, filled=torch.ones(1, dtype=torch.bool))


def test_coverage_hand_computed():
    valid = torch.tensor([False, True, True, True])
    kind = _cap([[0, 0, 0, 0, 0, 1],                        # invalid position: must not vote
                 [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],
                 [0.10, 0.05, 0.50, 0.30, 0.05, 0.00],
                 [0.10, 0.10, 0.00, 0.05, 0.45, 0.30]],
                [[5, 4], [0, 1], [2, 3], [4, 5]])
    vf, vt = kind.votes(valid)
    assert torch.allclose(vf[0], torch.tensor([0.70, 0.45, 0.60, 0.40, 0.55, 0.30]))
    assert torch.allclose(vt[0], torch.tensor([0.625, 0.375, 0.625, 0.375, 0.60, 0.40]))
    assert set(torch.nonzero(top_m_pool(vf, 2)[0]).flatten().tolist()) == {0, 2}
    assert set(torch.nonzero(top_m_pool(vf, 3)[0]).flatten().tolist()) == {0, 2, 4}
    assert set(torch.nonzero(top_m_pool(vf, 4)[0]).flatten().tolist()) == {0, 1, 2, 4}
    assert set(torch.nonzero(top_m_pool(vt, 4)[0]).flatten().tolist()) == {0, 2, 4, 5}
    final = _cap([[1, 0, 0, 0, 0, 0]] * 4, [[0, 1], [0, 3], [2, 4], [1, 4]])       # R_final picks (valid): 0,3 | 2,4 | 1,4
    out = coverage_all_kinds({"P5": final, "X": kind}, valid, Ms=(2, 3, 4), final_kind="P5")
    cx = out["X"]["cov"]
    assert [round(float(cx["top8"][M][0]), 6) for M in (2, 3, 4)] == [round(v, 6) for v in (2 / 6, 4 / 6, 4 / 6)]
    assert [round(float(cx["full"][M][0]), 6) for M in (2, 3, 4)] == [round(v, 6) for v in (2 / 6, 4 / 6, 5 / 6)]
    # unique-expert recall: U_X (valid) = {0..5} covers U_final = {0,1,2,3,4} fully
    assert float(out["X"]["recall"][0]) == 1.0
    assert int(out["X"]["uniq"][0]) == 6 and int(out["P5"]["uniq"][0]) == 5                 # all 4 positions count
    small = _cap([[1, 0, 0, 0, 0, 0]] * 4, [[0, 1]] * 4)
    assert abs(float(unique_recall(small.expert_union(valid), final.expert_union(valid))[0]) - 2 / 5) < 1e-6
    # halves union: identical halves -> identical pools / coverage; disjoint halves -> summed votes
    out2 = coverage_all_kinds({"P5": final, "P2a": kind, "P2b": kind}, valid, Ms=(2, 3, 4), final_kind="P5")
    for M in (2, 3, 4):
        assert torch.equal(out2["P2"]["cov"]["top8"][M], out2["P2a"]["cov"]["top8"][M])
    a = _cap([[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], [[5, 0], [0, 1], [0, 1], [0, 1]])
    b = _cap([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]], [[5, 0], [5, 4], [5, 4], [5, 4]])
    out3 = coverage_all_kinds({"P5": final, "P2a": a, "P2b": b}, valid, Ms=(2,), final_kind="P5")
    assert set(torch.nonzero(top_m_pool(a.votes(valid)[0] + b.votes(valid)[0], 2)[0]).flatten().tolist()) == {0, 5}
    assert abs(float(out3["P2"]["cov"]["full"][2][0]) - 1 / 6) < 1e-6                       # only pick 0 (pos1) is in {0,5}
    assert abs(float(out3["P2"]["recall"][0]) - 3 / 5) < 1e-6                              # U = {0,1,4,5} & {0,1,2,3,4}
    # empty valid set -> nan coverage, not a crash
    assert torch.isnan(pool_coverage(top_m_pool(vf, 2), final.ids, torch.zeros(4, dtype=torch.bool))).all()


# --------------------------------------------------------------------------
# 6. slot metrics: MASK suppression, fp32 probabilities, bf16 greedy path
# --------------------------------------------------------------------------
def test_slot_metrics_mask_suppression():
    V, mask_id = 8, 5
    bl = 3
    logits = torch.zeros(1, bl, V)
    logits[0, 0, 5] = 9.0; logits[0, 0, 2] = 8.0          # raw argmax MASK -> suppressed argmax 2
    logits[0, 1, 3] = 6.0
    logits[0, 2, 1] = 2.0
    final = torch.tensor([2, 3, 4]); draft = torch.tensor([2, 0, 1]); fill = torch.tensor([7, 3, 1])
    valid = torch.tensor([True, True, False])
    m = slot_metrics(logits, final, draft, fill, mask_id, valid)
    assert m["argmax"].tolist() == [2, 3, 1] and m["argmax_bf16"].tolist() == [5, 3, 1] and int(m["raw_mask_argmax"]) == 1
    p = F.softmax(logits[0].float().masked_fill(torch.arange(V) == mask_id, float("-inf")), -1)
    assert torch.allclose(m["top1"], p.max(-1).values) and torch.allclose(m["p_final"], p[torch.arange(bl), final])
    assert torch.allclose(m["p_draft"], p[torch.arange(bl), draft]) and torch.allclose(m["p_fill"], p[torch.arange(bl), fill])
    assert m["top1"].dtype == torch.float32
    # the count only looks at valid slots
    logits[0, 2, 5] = 50.0
    assert int(slot_metrics(logits, final, draft, fill, mask_id, valid)["raw_mask_argmax"]) == 1


# --------------------------------------------------------------------------
# 7. end to end on fake models: run_item -> jsonl/npz -> analyze
# --------------------------------------------------------------------------
def _synthetic_trace(plen, nb, seed=0):
    g = torch.Generator().manual_seed(seed)
    pl = (plen // BL) * BL
    prompt = torch.randint(1000, 5000, (1, plen), generator=g)
    tokens = torch.randint(1000, 5000, (nb, BL), generator=g)
    valid = torch.zeros(nb, BL, dtype=torch.bool)
    for b in range(nb):
        valid[b] = torch.arange(pl + b * BL, pl + (b + 1) * BL) >= plen
    tokens[0][~valid[0]] = prompt[0, pl:]
    proposed = torch.randint(1000, 5000, (nb, 2, BL), generator=g)
    for b in range(nb):
        proposed[b, :, ~valid[b]] = tokens[b][~valid[b]]
    tr = {"tokens": tokens.numpy().astype(np.int32), "valid": valid.numpy(), "proposed": proposed.numpy().astype(np.int32),
          "n_pass": np.full(nb, 2, np.int32), "block_start": (pl + BL * np.arange(nb)).astype(np.int32),
          "prompt_length": np.int32(plen), "block_length": np.int32(BL)}
    return prompt, tr


def test_end_to_end_fake_models(tmp_path, monkeypatch):
    import draftalign.probe_ab as PA
    import draftalign.analyze_ab as AA
    moe, dr = FakeMoEModel(), FakeDrafter()
    moe_seq, dr_seq = SeqCache(moe, BL, "moe"), SeqCache(dr, BL, "drafter")
    rc = RouterCapture(moe, BL)
    prompt, tr = _synthetic_trace(plen=40, nb=2)
    blocks, item, arrays = PA.run_item(moe_seq, dr_seq, rc, prompt, tr)
    assert item["status"] == "ok" and item["n_blocks"] == 2 and item["prefill_length"] == 32
    assert item["moe_n_by_kind"]["prefill"] == 1 and item["moe_n_by_kind"]["flush"] == 2 and all(item["moe_n_by_kind"][k] == 2 for k, _, _ in KINDS)
    assert item["drafter_n_by_kind"]["draft_s4"] == 4 + 4 and item["drafter_n_by_kind"]["draft_s8"] == 8 + 8 and item["drafter_n_by_kind"]["flush"] == 2
    assert moe_seq.cached_length() == 96 == dr_seq.cached_length()
    # every probe forward saw the prefix cache at `lo` and did not write it; flushes hold the FINAL block
    probe_calls = [c for c in moe.calls if c["store_kv"] is False]
    assert len(probe_calls) == 2 * len(KINDS) and [c["cached"] for c in probe_calls] == [32] * 10 + [64] * 10
    flushes = [c for c in moe.calls if c["store_kv"] is True][1:]
    assert torch.equal(flushes[0]["input_ids"][0], torch.as_tensor(tr["tokens"][0]).long()) and torch.equal(flushes[1]["input_ids"][0], torch.as_tensor(tr["tokens"][1]).long())
    for b, rec in enumerate(blocks):
        v = np.asarray(rec["valid"], bool)
        assert rec["n_valid"] == int(v.sum()) and rec["b"] == b and rec["lo"] == 32 + 32 * b
        p0 = rec["moe"]["P0"]
        assert all(len(p0[k]) == BL for k in ("argmax", "top1", "p_final", "p_draft", "p_fill", "masked", "argmax_bf16"))
        assert np.asarray(p0["masked"], bool).tolist() == v.tolist()
        assert not any(np.asarray(rec["moe"]["P1"]["masked"], bool)) and (np.asarray(rec["moe"]["P2a"]["masked"], bool) | np.asarray(rec["moe"]["P2b"]["masked"], bool)).tolist() == v.tolist()
        # P5 fill == final -> p_fill == p_final; drafted s8 tokens == abs positions on valid slots
        assert rec["moe"]["P5"]["p_fill"] == rec["moe"]["P5"]["p_final"]
        assert np.asarray(rec["drafter"]["s8"]["tokens"])[v].tolist() == list(range(rec["lo"], rec["lo"] + BL))[int((~v).sum()):]
        ex = rec["experts"]
        assert set(ex) == {"P0", "P1", "P2a", "P2b", "P2", "P3", "P4a", "P4b", "P4", "P5", "P6a", "P6b", "P6"}
        assert len(ex["P0"]["recall"]) == L and set(ex["P0"]["cov"]["top8"]) == {"16", "24", "32", "48"}
        assert all(x == 1.0 for x in ex["P5"]["recall"])                    # R_final vs itself
        assert all(0 <= x <= 1 for M in ex["P1"]["cov"]["top8"].values() for x in M)
    assert arrays["top8_P0"].shape == (2, L, BL, K) and arrays["top8_P0"].dtype == np.int16
    # integrity: make the recorded pass-1 proposals equal P0's bf16 argmax -> a re-run reports 0 mismatches
    tr2 = {k: v.copy() for k, v in tr.items()}
    for b, rec in enumerate(blocks):
        tr2["proposed"][b, 0] = np.asarray(rec["moe"]["P0"]["argmax_bf16"])
        tr2["proposed"][b, 0][~tr["valid"][b]] = tr["tokens"][b][~tr["valid"][b]]
    blocks2, item2, _ = PA.run_item(moe_seq, dr_seq, rc, prompt, tr2)
    assert item2["p0_mismatch_total"] == 0 and item["p0_mismatch_total"] > 0
    assert [r["moe"]["P0"]["argmax"] for r in blocks2] == [r["moe"]["P0"]["argmax"] for r in blocks]   # deterministic replay
    # write a shard like the driver and run the analysis on it
    cell = tmp_path / "cell"; cell.mkdir()
    with open(cell / "shard0of1.jsonl", "w") as fh:
        for idx, (bl_, it_) in ((7, (blocks, item)), (8, (blocks2, item2))):
            common = {"idx": idx, "rank": idx, "config_hash": "x"}
            for r in bl_:
                fh.write(json.dumps({**common, **r}) + "\n")
            fh.write(json.dumps({**common, **it_}) + "\n")
        fh.write(json.dumps({"idx": 9, "rec": "item", "status": "skipped_trace_overwritten", "n_blocks": 0}) + "\n")
    items, blks = AA.load_cell(str(cell))
    assert len(items) == 3 and len(blks) == 4
    sa = AA.slot_arrays(blks[0], "P2")
    assert sa["argmax"].shape[0] == blks[0]["n_valid"]
    AA.main(["--cell", str(cell)])
    # trace validation helpers
    rec = {"prompt_tokens": 40, "gen_ids": np.concatenate([tr["tokens"][b][tr["valid"][b]] for b in range(2)]).tolist()}
    assert PA.trace_matches_record(tr, rec) and not PA.trace_matches_record(tr, {**rec, "prompt_tokens": 41})
    assert not PA.trace_matches_record(tr, {**rec, "gen_ids": rec["gen_ids"][:-1] + [1]})
    rc.close()
