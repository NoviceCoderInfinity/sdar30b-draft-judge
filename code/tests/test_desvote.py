"""CPU-only unit tests for the DES-Vote MoE forward and `draft_open_slots`.

Written from docs/DESVOTE_SPEC.md alone (section 2: interface + invariants,
section 4: draft_open_slots). The implementation under test is a black box
selected by environment variables:

    DESVOTE_IMPL    harness module exposing CTRL / enable_vote_moe / disable_vote_moe
                    (default "sdar_harness.desvote")
    DESVOTE_RUNNER  runner module exposing draft_open_slots (default "run_desvote")

Run from the repo root:
    DESVOTE_IMPL=sdar_harness.desvote DESVOTE_RUNNER=run_desvote \
        pytest -q code/tests/test_desvote.py

Everything is float32 on CPU with a fixed seed. No real model is ever loaded.
The "stock" reference is the STOCK SDAR loop from fastmoe.py's docstring
(visit ALL experts; softmax float32 -> topk -> renorm -> one_hot -> index_add_),
re-implemented here so that it never goes through a (possibly patched) forward.
"""
from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

IMPL = os.environ.get("DESVOTE_IMPL", "sdar_harness.desvote")
RUNNER = os.environ.get("DESVOTE_RUNNER", "run_desvote")

# fake-model geometry (small on purpose)
E, K, H, L, N = 16, 4, 8, 2, 6
SEED = 0
MASK_ID = 151669          # spec section 1


# --------------------------------------------------------------------------
# fake CPU model
# --------------------------------------------------------------------------
class FakeExpert(nn.Module):
    """Deterministic [n, H] -> [n, H] map that records non-empty calls."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.calls: list = []            # row counts of non-empty inputs

    def forward(self, x):
        if x.shape[0] > 0:
            self.calls.append(int(x.shape[0]))
        return torch.tanh(self.lin(x))


class FakeMoE(nn.Module):
    """Mimics SDARMoeSparseMoeBlock: gate + experts + the STOCK forward."""

    def __init__(self, hidden: int, n_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden
        self.num_experts = n_experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList([FakeExpert(hidden) for _ in range(n_experts)])

    def forward(self, hidden_states):
        return stock_forward(self, hidden_states)


def stock_forward(blk, hidden_states, core=None):
    """The stock SDAR MoE forward, optionally restricted to a coreset.

    core=None -> exactly the stock loop (fastmoe.py docstring).
    core=[E] bool -> spec 2.4: softmax over the coreset's logits only, then the
    identical topk / renorm / cast / dispatch.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)
    router_logits = blk.gate(hs)
    if core is None:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    else:
        restricted = router_logits.float().masked_fill(~core[None], float("-inf"))
        routing_weights = F.softmax(restricted, dim=1)
    routing_weights, selected = torch.topk(routing_weights, blk.top_k, dim=-1)
    if blk.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hs.dtype)
    final = torch.zeros((batch_size * sequence_length, hidden_dim),
                        dtype=hs.dtype, device=hs.device)
    expert_mask = F.one_hot(selected, num_classes=blk.num_experts).permute(2, 1, 0)
    for expert_idx in range(blk.num_experts):
        expert_layer = blk.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hs[None, top_x].reshape(-1, hidden_dim)
        cur = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final.index_add_(0, top_x, cur.to(hs.dtype))
    return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class _Layer:
    def __init__(self, mlp):
        self.mlp = mlp


class _Inner:
    def __init__(self, layers):
        self.layers = layers


class FakeModel:
    """model.model.layers[i].mlp is the MoE block, as routing.moe_blocks expects."""

    def __init__(self, layers):
        self.model = _Inner(layers)
        self.config = SimpleNamespace(num_experts=E, num_experts_per_tok=K,
                                      hidden_size=H, num_hidden_layers=L)
        self.device = torch.device("cpu")


def build_model(seed: int = SEED) -> FakeModel:
    torch.manual_seed(seed)
    return FakeModel([_Layer(FakeMoE(H, E, K)) for _ in range(L)])


def build_inputs(seed: int = SEED + 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(1, N, H, generator=g) for _ in range(L)]


XS = build_inputs()


def blocks_of(model):
    return [layer.mlp for layer in model.model.layers]


def clear_calls(model):
    for blk in blocks_of(model):
        for ex in blk.experts:
            ex.calls = []


def called(blk) -> set:
    return {i for i, ex in enumerate(blk.experts) if ex.calls}


# --------------------------------------------------------------------------
# hand computations (spec section 2)
# --------------------------------------------------------------------------
def w_full_of(router_logits):
    return F.softmax(router_logits, dim=1, dtype=torch.float)


def topk_masked(w_full, k=K):
    v, i = torch.topk(w_full, k, dim=-1)
    return torch.zeros_like(w_full).scatter_(1, i, v)


def ref_union(router_logits) -> set:
    return set(torch.topk(w_full_of(router_logits), K, dim=-1).indices.flatten().tolist())


def hand_core(router_logits, M, valid=None, masked=None, recorded_layer=None):
    own = topk_masked(w_full_of(router_logits))
    if masked is not None and recorded_layer is not None:
        votes_per_pos = torch.where(masked[:, None], recorded_layer, own)
    else:
        votes_per_pos = own
    votes = votes_per_pos[valid].sum(0) if valid is not None else votes_per_pos.sum(0)
    core = torch.zeros(router_logits.shape[1], dtype=torch.bool)
    core[torch.topk(votes, min(M, router_logits.shape[1])).indices] = True
    return core


def as_set(core) -> set:
    return set(torch.nonzero(core).flatten().tolist())


def reset_ctrl(CTRL):
    CTRL.M = None
    CTRL.force_restrict = False
    CTRL.valid = None
    CTRL.masked = None
    CTRL.recorded = None
    CTRL.use_recorded = False
    CTRL.recording = False
    CTRL.reset_cost()


def cost_get(cost: dict, key: str):
    """Exact key first; else the unique key containing it (spec names the
    counter but defers key names to analyze_desvote.py)."""
    if key in cost:
        return cost[key]
    cands = [k for k in cost if key in k or key.replace("_layer_forwards", "") in k]
    assert len(cands) == 1, f"cannot find counter {key!r} in cost() keys {sorted(cost)}"
    return cost[cands[0]]


def forward_all(model, xs=XS):
    return [blk(x) for blk, x in zip(blocks_of(model), xs)]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def impl():
    return importlib.import_module(IMPL)


@pytest.fixture
def env(impl):
    model = build_model()
    ref = build_model()             # identical weights, never patched
    reset_ctrl(impl.CTRL)
    n = impl.enable_vote_moe(model)
    assert n == L, f"enable_vote_moe patched {n} blocks, expected {L}"
    try:
        yield impl, model, ref
    finally:
        impl.disable_vote_moe(model)
        reset_ctrl(impl.CTRL)


# --------------------------------------------------------------------------
# 1. unrestricted == stock, bit-identical
# --------------------------------------------------------------------------
def test_1_unrestricted_bit_identical(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    CTRL.M = None
    assert CTRL.n_layers == L and CTRL.n_experts == E
    for i, (blk, rblk, x) in enumerate(zip(blocks_of(model), blocks_of(ref), XS)):
        out, rl = blk(x)
        sout, srl = stock_forward(rblk, x)
        assert torch.equal(rl, srl), f"layer {i}: router_logits differ from stock"
        assert torch.equal(out, sout), f"layer {i}: output differs from stock at M=None"
        assert out.shape == x.shape and rl.shape == (N, E)
    # enable_vote_moe must tag each patched block with its layer index
    for i, blk in enumerate(blocks_of(model)):
        assert getattr(blk, "_layer_idx", None) == i, f"layer {i}: _layer_idx not set by enable_vote_moe"


# --------------------------------------------------------------------------
# 2. force_restrict at M=E == stock, bit-identical
# --------------------------------------------------------------------------
def test_2_force_restrict_full_M_bit_identical(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    CTRL.M = E
    CTRL.force_restrict = True
    for valid in (None, torch.ones(N, dtype=torch.bool)):
        CTRL.valid = valid
        for i, (blk, rblk, x) in enumerate(zip(blocks_of(model), blocks_of(ref), XS)):
            out, rl = blk(x)
            sout, srl = stock_forward(rblk, x)
            assert torch.equal(rl, srl), f"layer {i}: router_logits differ (force_restrict, M=E)"
            assert torch.equal(out, sout), f"layer {i}: output differs (force_restrict, M=E, valid={valid})"
    assert cost_get(CTRL.cost(), "routing_override_rate") == 0.0


# --------------------------------------------------------------------------
# 3. DES vote at M<E: dispatch inside the coreset, restricted softmax weights
# --------------------------------------------------------------------------
def test_3_des_vote_restricts_to_coreset(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    M = 6
    CTRL.M = M
    CTRL.valid = torch.ones(N, dtype=torch.bool)
    CTRL.masked = None
    CTRL.use_recorded = False
    CTRL.reset_cost()
    excluded_somewhere = False
    for i, (blk, rblk, x) in enumerate(zip(blocks_of(model), blocks_of(ref), XS)):
        clear_calls(model)
        out, rl = blk(x)
        srl = rblk.gate(x.view(-1, H))
        assert torch.equal(rl, srl), f"layer {i}: router_logits must stay unrestricted"
        core = hand_core(rl, M, valid=CTRL.valid)
        assert core.sum().item() == M
        used = called(blk)
        assert used, f"layer {i}: no expert was dispatched to"
        assert used <= as_set(core), f"layer {i}: dispatched {used} outside coreset {as_set(core)}"
        hand_out, _ = stock_forward(rblk, x, core=core)
        assert torch.equal(out, hand_out), f"layer {i}: output != hand restricted forward"
        if not ref_union(rl) <= as_set(core):
            excluded_somewhere = True
            stock_out, _ = stock_forward(rblk, x)
            assert not torch.equal(out, stock_out), f"layer {i}: restricted output equals stock"
    assert excluded_somewhere, "test fixture is not discriminating: coreset never excluded a ref expert"
    c = CTRL.cost()
    assert cost_get(c, "routing_override_rate") > 0.0
    assert cost_get(c, "layer_forwards") == L


# --------------------------------------------------------------------------
# 4. an invalid position must not affect the coreset
# --------------------------------------------------------------------------
def test_4_invalid_position_does_not_vote(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    blk, rblk = blocks_of(model)[0], blocks_of(ref)[0]
    M = K                          # coreset == dispatched set exactly
    p = N - 1
    x = XS[0].clone()
    # baseline coreset from the other positions
    base_valid = torch.ones(N, dtype=torch.bool)
    base_valid[p] = False
    rl0 = rblk.gate(x.view(-1, H))
    core_wo = hand_core(rl0, M, valid=base_valid)
    outsider = min(set(range(E)) - as_set(core_wo))
    # make position p vote ~1.0 for `outsider`: hidden along that gate row
    wrow = rblk.gate.weight[outsider].detach()
    x[0, p] = 100.0 * wrow / (wrow @ wrow)
    rl = rblk.gate(x.view(-1, H))
    assert rl[p].argmax().item() == outsider
    core_all = hand_core(rl, M, valid=torch.ones(N, dtype=torch.bool))
    core_val = hand_core(rl, M, valid=base_valid)
    assert outsider in as_set(core_all), "precondition: p alone should pull outsider into top-M"
    assert outsider not in as_set(core_val), "precondition: without p outsider is out of top-M"

    CTRL.M = M
    CTRL.masked = None
    CTRL.use_recorded = False
    # (a) all valid -> outsider elected
    CTRL.valid = torch.ones(N, dtype=torch.bool)
    clear_calls(model)
    out_a, _ = blk(x)
    assert called(blk) == as_set(core_all)
    assert torch.equal(out_a, stock_forward(rblk, x, core=core_all)[0])
    # (b) p invalid -> outsider must not be elected; p is still routed inside the coreset
    CTRL.valid = base_valid
    clear_calls(model)
    out_b, _ = blk(x)
    assert outsider not in called(blk), "invalid position changed the coreset"
    assert called(blk) == as_set(core_val)
    assert torch.equal(out_b, stock_forward(rblk, x, core=core_val)[0])
    assert not torch.equal(out_a, out_b)


# --------------------------------------------------------------------------
# 5. blend: recording, take_recording, use_recorded on masked positions
# --------------------------------------------------------------------------
def test_5_recording_and_blend(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    # --- recording
    CTRL.M = None
    CTRL.recording = True
    outs = forward_all(model)
    rec = CTRL.take_recording()
    assert CTRL.recording is False
    assert tuple(rec.shape) == (L, N, E) == (CTRL.n_layers, N, CTRL.n_experts)
    assert rec.dtype == torch.float32
    assert (rec >= 0).all()
    assert ((rec != 0).sum(-1) <= K).all()
    assert (rec.sum(-1) <= 1 + 1e-6).all()
    for i, (_, rl) in enumerate(outs):
        assert torch.equal(rec[i], topk_masked(w_full_of(rl))), f"layer {i}: recorded != topk_masked(w_full)"

    # --- blend on layer 0
    blk, rblk, x = blocks_of(model)[0], blocks_of(ref)[0], XS[0]
    rl = rblk.gate(x.view(-1, H))
    M = K
    valid = torch.ones(N, dtype=torch.bool)
    own_core = hand_core(rl, M, valid=valid)
    outsiders = sorted(set(range(E)) - as_set(own_core))
    B, B2 = outsiders[0], outsiders[1]
    pm, pu = 1, 3                       # pm masked, pu unmasked
    masked = torch.zeros(N, dtype=torch.bool)
    masked[pm] = True
    rec = rec.clone()
    rec[0, pm] = 0.0
    rec[0, pm, B] = 1.0                 # recorded votes elect B
    rec[0, pu] = 0.0
    rec[0, pu, B2] = 1.0                # must be ignored: pu is not masked
    expected = hand_core(rl, M, valid=valid, masked=masked, recorded_layer=rec[0])
    assert B in as_set(expected) and B2 not in as_set(expected), "precondition on the hand coreset"

    CTRL.M = M
    CTRL.valid = valid
    CTRL.masked = masked
    CTRL.recorded = rec
    CTRL.use_recorded = True
    clear_calls(model)
    out, _ = blk(x)
    assert B in called(blk), "masked position did not vote from the recording"
    assert B2 not in called(blk), "unmasked position voted from the recording"
    assert called(blk) == as_set(expected)
    assert torch.equal(out, stock_forward(rblk, x, core=expected)[0])

    # use_recorded off -> own weights only, same masked
    CTRL.use_recorded = False
    clear_calls(model)
    out2, _ = blk(x)
    assert called(blk) == as_set(own_core)
    assert torch.equal(out2, stock_forward(rblk, x, core=own_core)[0])

    # use_recorded on but masked None -> own weights (spec 2.4)
    CTRL.use_recorded = True
    CTRL.masked = None
    clear_calls(model)
    out3, _ = blk(x)
    assert called(blk) == as_set(own_core)
    assert torch.equal(out3, out2)


# --------------------------------------------------------------------------
# 6. a recording forward is never restricted; loads go to readout counters
# --------------------------------------------------------------------------
def test_6_recording_forward_unrestricted_and_counted_as_readout(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    CTRL.M = K
    CTRL.valid = torch.ones(N, dtype=torch.bool)
    CTRL.recording = True
    CTRL.reset_cost()
    clear_calls(model)
    uniq = 0
    for i, (blk, rblk, x) in enumerate(zip(blocks_of(model), blocks_of(ref), XS)):
        out, rl = blk(x)
        sout, srl = stock_forward(rblk, x)
        assert torch.equal(out, sout), f"layer {i}: recording forward was restricted"
        assert torch.equal(rl, srl)
        assert called(blk) == ref_union(rl)
        uniq += len(called(blk))
    c = CTRL.cost()
    assert cost_get(c, "readout_layer_forwards") == L
    assert cost_get(c, "layer_forwards") == 0
    assert cost_get(c, "restricted_layer_forwards") == 0
    assert cost_get(c, "readout_unique_expert_layer_loads") == uniq
    assert cost_get(c, "unique_expert_layer_loads") == 0
    rec = CTRL.take_recording()
    assert tuple(rec.shape) == (L, N, E) and CTRL.recording is False


# --------------------------------------------------------------------------
# 7. cost() after a known sequence
# --------------------------------------------------------------------------
def test_7_cost_counters(env):
    impl, model, ref = env
    CTRL = impl.CTRL
    valid_all = torch.ones(N, dtype=torch.bool)

    # counts: 2 unrestricted + 1 force_restrict(M=E) + 1 readout + 1 restricted(M<E)
    CTRL.reset_cost()
    CTRL.M = None
    forward_all(model); forward_all(model)
    CTRL.M, CTRL.force_restrict, CTRL.valid = E, True, valid_all
    forward_all(model)
    CTRL.force_restrict = False
    CTRL.recording = True
    forward_all(model)
    rec = CTRL.take_recording()
    CTRL.M = K
    forward_all(model)
    c = CTRL.cost()
    assert cost_get(c, "layer_forwards") == 4 * L
    assert cost_get(c, "restricted_layer_forwards") == 2 * L
    assert cost_get(c, "readout_layer_forwards") == L
    assert cost_get(c, "unique_expert_layer_loads") > 0
    assert cost_get(c, "readout_unique_expert_layer_loads") > 0
    assert 0.0 <= cost_get(c, "zero_overlap_rate") <= 1.0

    # override rate: 0 when only M=E forwards were seen
    CTRL.reset_cost()
    CTRL.M, CTRL.force_restrict = E, True
    forward_all(model)
    assert cost_get(CTRL.cost(), "routing_override_rate") == 0.0
    CTRL.force_restrict = False

    # override rate > 0 with a small coreset (test 3 setting)
    CTRL.reset_cost()
    CTRL.M = 6
    forward_all(model)
    assert cost_get(CTRL.cost(), "routing_override_rate") > 0.0

    # vote_share_known_ahead = masked valid positions / valid positions, use_recorded on
    CTRL.reset_cost()
    valid = torch.tensor([True, True, True, True, True, False])
    masked = torch.tensor([True, True, False, False, False, True])   # 2 of the 5 valid
    CTRL.M, CTRL.valid, CTRL.masked = K, valid, masked
    CTRL.recorded, CTRL.use_recorded = rec, True
    forward_all(model)
    assert abs(cost_get(CTRL.cost(), "vote_share_known_ahead") - 2 / 5) < 1e-6

    # after reset everything is zero again
    CTRL.reset_cost()
    c = CTRL.cost()
    assert cost_get(c, "layer_forwards") == 0 and cost_get(c, "readout_layer_forwards") == 0


# --------------------------------------------------------------------------
# 8. disable_vote_moe restores the stock forward
# --------------------------------------------------------------------------
def test_8_disable_restores_stock(impl):
    model, ref = build_model(), build_model()
    reset_ctrl(impl.CTRL)
    assert impl.enable_vote_moe(model) == L
    impl.CTRL.M = K
    impl.CTRL.valid = torch.ones(N, dtype=torch.bool)
    out_r, _ = blocks_of(model)[0](XS[0])
    assert not torch.equal(out_r, stock_forward(blocks_of(ref)[0], XS[0])[0])
    assert impl.disable_vote_moe(model) == L
    for i, (blk, rblk, x) in enumerate(zip(blocks_of(model), blocks_of(ref), XS)):
        assert "forward" not in blk.__dict__ or blk.forward.__func__ is type(blk).forward
        out, rl = blk(x)          # CTRL.M is still K: a restored forward ignores it
        sout, srl = stock_forward(rblk, x)
        assert torch.equal(out, sout) and torch.equal(rl, srl), f"layer {i}: not restored"
    reset_ctrl(impl.CTRL)


# --------------------------------------------------------------------------
# 9. draft_open_slots on a fake drafter
# --------------------------------------------------------------------------
class FakeDrafter(nn.Module):
    """Dense drafter stand-in: logits favour token == absolute position at every slot."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=vocab)
        self.calls: list = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, store_kv=None, **kw):
        assert input_ids is not None and input_ids.dim() == 2 and input_ids.shape[0] == 1
        S = input_ids.shape[1]
        pos = position_ids[0] if position_ids is not None else torch.arange(S)
        self.calls.append(dict(input_ids=input_ids.detach().clone(),
                               position_ids=None if position_ids is None else position_ids.detach().clone(),
                               attention_mask=None if attention_mask is None else tuple(attention_mask.shape),
                               cache=past_key_values, store_kv=store_kv, use_cache=use_cache))
        logits = torch.full((1, S, self.vocab), -10.0)
        logits[0, torch.arange(S), pos] = 10.0
        return SimpleNamespace(logits=logits)


def _import_runner(monkeypatch):
    import transformers
    def boom(*a, **k):
        raise RuntimeError("model load attempted at import")
    for cls in ("AutoModelForCausalLM", "AutoModel", "AutoTokenizer"):
        if hasattr(transformers, cls):
            monkeypatch.setattr(getattr(transformers, cls), "from_pretrained", boom)
    monkeypatch.setattr(sys, "argv", [RUNNER])
    if RUNNER in sys.modules:
        return sys.modules[RUNNER]
    try:
        return importlib.import_module(RUNNER)
    except BaseException as e:            # SystemExit from argparse, CUDA, model loads
        pytest.skip(f"{RUNNER} cannot be imported on CPU without models: {e!r}")


def _run_draft(runner, x_ctx, lo, hi, passes, vocab=8192):
    drafter = FakeDrafter(vocab)
    before = x_ctx.clone()
    tokens, n_fwd = runner.draft_open_slots(drafter, x_ctx, lo, hi, passes)
    assert torch.equal(x_ctx, before), "draft_open_slots must not mutate x_ctx"
    tokens = torch.as_tensor(tokens).reshape(-1)
    assert tokens.numel() == hi - lo
    return tokens, int(n_fwd), drafter


def _check_draft(runner, x_ctx, lo, hi, passes):
    blk = x_ctx[0, lo:hi]
    open_ = blk == MASK_ID
    n_open = int(open_.sum())
    tokens, n_fwd, drafter = _run_draft(runner, x_ctx, lo, hi, passes)
    abs_pos = torch.arange(lo, hi)
    assert torch.equal(tokens[~open_], blk[~open_]), "committed / invalid slots were overwritten"
    assert torch.equal(tokens[open_], abs_pos[open_]), "open slots were not filled with the drafter's tokens"
    n_pass = min(passes, n_open)
    assert n_fwd == (1 if lo > 0 else 0) + n_pass
    assert len(drafter.calls) == n_fwd
    calls = drafter.calls
    if lo > 0:
        pre = calls.pop(0)
        assert torch.equal(pre["input_ids"], x_ctx[:, :lo]) and pre["store_kv"] is True
        assert pre["position_ids"] is None or torch.equal(pre["position_ids"][0], torch.arange(lo))
    assert len(calls) == n_pass
    sched = runner.get_num_transfer_tokens(n_open, n_pass) if hasattr(runner, "get_num_transfer_tokens") \
        else importlib.import_module("sdar_harness.decode").get_num_transfer_tokens(n_open, n_pass)
    remaining = n_open
    caches = {id(c["cache"]) for c in calls}
    assert len(caches) == 1 and None not in {c["cache"] for c in calls}, "one fresh cache for the whole draft"
    for s, call in enumerate(calls):
        ids = call["input_ids"][0]
        assert ids.shape[0] == hi - lo and call["store_kv"] is False
        assert call["position_ids"] is not None and torch.equal(call["position_ids"][0], abs_pos)
        assert torch.equal(ids[~open_], blk[~open_])
        assert int((ids == MASK_ID).sum()) == remaining, f"pass {s}: schedule not followed"
        remaining -= int(sched[s])
    assert remaining == 0
    return drafter


def test_9_draft_open_slots(monkeypatch):
    runner = _import_runner(monkeypatch)
    assert hasattr(runner, "draft_open_slots"), f"{RUNNER} has no draft_open_slots"
    plen, lo, hi = 40, 32, 64
    x = torch.full((1, hi), MASK_ID, dtype=torch.long)
    x[0, :plen] = 5000 + torch.arange(plen)                 # prompt (tail [32,40) is invalid)
    for p in (41, 45, 50, 60):
        x[0, p] = 1000 + p                                   # committed slots
    # 20 open slots, 8 passes -> 1 prefill + 8 forwards, schedule 3,3,3,3,2,2,2,2
    d1 = _check_draft(runner, x.clone(), lo, hi, 8)
    # fewer open slots than passes -> min(passes, n_open) forwards
    x2 = x.clone()
    x2[0, lo:hi] = torch.where(x2[0, lo:hi] == MASK_ID, 1000 + torch.arange(lo, hi), x2[0, lo:hi])
    for p in (44, 55, 63):
        x2[0, p] = MASK_ID
    d2 = _check_draft(runner, x2, lo, hi, 8)
    assert d1.calls[0]["cache"] is not d2.calls[0]["cache"], "each draft must use a fresh cache"
    # lo == 0: no prefill; the prompt tail occupies the first slots of the block
    x3 = torch.full((1, 32), MASK_ID, dtype=torch.long)
    x3[0, :3] = 5000 + torch.arange(3)
    _check_draft(runner, x3, 0, 32, 8)
    # passes larger than the block, all 32 open -> 32 forwards
    x4 = torch.full((1, 64), MASK_ID, dtype=torch.long)
    x4[0, :32] = 5000 + torch.arange(32)
    _check_draft(runner, x4, 32, 64, 40)
