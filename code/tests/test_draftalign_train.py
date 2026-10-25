"""CPU tests for the draft-aligned trainer (code/draftalign/{adapter,train_forward,losses,
build_train_set,dataset,train}.py).  No real model, no GPU.

The stand-in `TinyCausalLM` mirrors the remote SDAR module tree and arithmetic
(hf_cache/.../modeling_sdar_moe.py, line numbers cited inline) so that a parity test
between `training_forward` and the model's OWN stock forward is meaningful:

    model.model.layers[i].self_attn.{q,k,v,o}_proj / q_norm / k_norm
    model.model.layers[i].mlp.gate + .mlp.experts[j].{gate,up,down}_proj   (MoE layers)
    model.model.layers[i].mlp.{gate,up,down}_proj                          (dense layer)
    model.model.{embed_tokens, norm, rotary_emb} + model.lm_head

Run:  /venv/main/bin/python -m pytest -q code/tests/test_draftalign_train.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if os.path.join(CODE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(CODE, "compat"))

from draftalign import adapter as A                                       # noqa: E402
from draftalign import build_train_set as B                               # noqa: E402
from draftalign import train as T                                         # noqa: E402
from draftalign.dataset import REGIMES, SamplerConfig, StateSampler       # noqa: E402
from draftalign.losses import active_logits, masked_ce_and_certainty      # noqa: E402
from draftalign.train_forward import (block_tril_bool, expert_pool,       # noqa: E402
                                      stock_forward_reference, training_forward)

BLOCK = 4
HID = 32
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
N_EXPERTS, TOP_K, MOE_INTER = 8, 2, 16
N_LAYERS = 3
VOCAB = 40
MASK = 39
DENSE_LAYER = 1                      # one mlp_only layer, to exercise that branch


# ===========================================================================
# the stand-in model
# ===========================================================================
class TinyRMSNorm(nn.Module):
    """remote :328-347 (the pure-torch body the flash kernel replaces)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)


class TinyRotary(nn.Module):
    """remote :456-493, rope_type 'default'."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def _rotate_half(x):                                       # remote :68-72
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def _apply_rope(q, k, cos, sin):                           # remote :75-99
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class TinyAttention(nn.Module):
    """remote :143-242 (bool mask + `torch.all` branch + SDPA with enable_gqa)."""

    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_key_value_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = TinyRMSNorm(self.head_dim)
        self.k_norm = TinyRMSNorm(self.head_dim)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kw):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rope(q, k, cos, sin)
        attention_mask = attention_mask.bool() if attention_mask is not None else None
        if attention_mask is None or bool(torch.all(attention_mask)):          # remote :219
            o = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               scale=self.scaling, enable_gqa=True)
        else:                                                                  # remote :230
            o = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attention_mask,
                                               is_causal=False, scale=self.scaling, enable_gqa=True)
        o = o.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        return self.o_proj(o), None


class TinyMLP(nn.Module):
    """remote :245-264."""

    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = F.silu
        self.received = []                      # row indices this module was given (tests only)

    def forward(self, x):
        if x.dim() == 2:
            self.received.append(x.shape[0])
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyMoE(nn.Module):
    """remote :267-324, including the stock all-experts dispatch loop."""

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([TinyMLP(cfg.hidden_size, cfg.moe_intermediate_size)
                                      for _ in range(cfg.num_experts)])
        self.dispatch = None                   # {expert -> sorted row list} when recording

    def record(self, on=True):
        self.dispatch = {} if on else None

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        hs = hidden_states.view(-1, H)
        router_logits = self.gate(hs)                                          # remote :287
        w = F.softmax(router_logits, dim=1, dtype=torch.float)                 # remote :289
        rw, sel = torch.topk(w, self.top_k, dim=-1)                            # remote :290-291
        if self.norm_topk_prob:
            rw = rw / rw.sum(dim=-1, keepdim=True)                             # remote :292-293
        rw = rw.to(hs.dtype)
        final = torch.zeros((B * S, H), dtype=hs.dtype, device=hs.device)
        em = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)     # remote :303-304
        for e in range(self.num_experts):                                      # remote :307
            idx, top_x = torch.where(em[e])
            cur = self.experts[e](hs[None, top_x].reshape(-1, H)) * rw[top_x, idx, None]
            final.index_add_(0, top_x, cur.to(hs.dtype))
            if self.dispatch is not None and top_x.numel():
                self.dispatch.setdefault(e, []).extend(sorted(top_x.tolist()))
        return final.reshape(B, S, H), router_logits


class TinyLayer(nn.Module):
    """remote :353-453."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = TinyAttention(cfg)
        self.mlp = TinyMLP(cfg.hidden_size, cfg.intermediate_size) if layer_idx in cfg.mlp_only_layers \
            else TinyMoE(cfg)
        self.input_layernorm = TinyRMSNorm(cfg.hidden_size)
        self.post_attention_layernorm = TinyRMSNorm(cfg.hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kw):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        out = self.mlp(hidden_states)
        hidden_states = out[0] if isinstance(out, tuple) else out              # remote :437-441
        return (residual + hidden_states,)


class TinyInner(nn.Module):
    """remote :527-670."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TinyLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = TinyRMSNorm(cfg.hidden_size)
        self.rotary_emb = TinyRotary(cfg.head_dim)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        h = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pe = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_embeddings=pe)[0]
        return SimpleNamespace(last_hidden_state=self.norm(h))


class TinyCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = TinyInner(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                logits_to_keep=0, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        h = out.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=self.lm_head(h[:, sl, :]), last_hidden_state=h)


def tiny_config():
    return SimpleNamespace(hidden_size=HID, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
                           head_dim=HEAD_DIM, num_experts=N_EXPERTS, num_experts_per_tok=TOP_K,
                           norm_topk_prob=True, moe_intermediate_size=MOE_INTER,
                           intermediate_size=2 * MOE_INTER, num_hidden_layers=N_LAYERS,
                           vocab_size=VOCAB, mlp_only_layers=[DENSE_LAYER], rms_norm_eps=1e-6,
                           use_cache=False)


def build_model(seed: int = 0) -> TinyCausalLM:
    torch.manual_seed(seed)
    return TinyCausalLM(tiny_config()).eval()


def sample_ids(S: int, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, S), generator=g)


def moe_blocks(model):
    return [l.mlp for l in model.model.layers if isinstance(l.mlp, TinyMoE)]


# ===========================================================================
# adapter
# ===========================================================================
def test_lora_zero_init_is_the_identity():
    model = build_model()
    ids = sample_ids(3 * BLOCK)
    mask = block_tril_bool(3 * BLOCK, ids.device, BLOCK)
    with torch.no_grad():
        before = model(input_ids=ids, attention_mask=mask).logits.clone()
    counts = A.attach_lora(model, r=4, alpha=8.0)
    assert counts == {"lora": 4 * N_LAYERS, "gates": N_LAYERS - 1}     # the dense layer has no gate
    with torch.no_grad():
        after = model(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(before, after), (before - after).abs().max()


def test_gate_wrapper_is_bit_exact_in_bf16():
    """The fp32 master is cast BACK to bf16 for the matmul, so the router is unchanged."""
    torch.manual_seed(0)
    lin = nn.Linear(HID, N_EXPERTS, bias=False).to(torch.bfloat16)
    x = torch.randn(7, HID).to(torch.bfloat16)
    ref = lin(x)
    g = A.GateFP32(lin)
    assert g.weight.dtype == torch.float32 and g.compute_dtype == torch.bfloat16
    assert torch.equal(g(x), ref)
    assert torch.equal(g.to_linear()(x), ref)


def test_freeze_base_leaves_only_lora_and_gates():
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    names = [n for n, _ in A.trainable_parameters(model)]
    assert names, "nothing is trainable"
    for n in names:
        assert n.endswith(("lora_A", "lora_B")) or n.endswith("mlp.gate.weight"), n
    assert sum(1 for n in names if n.endswith("mlp.gate.weight")) == N_LAYERS - 1
    assert sum(1 for n in names if n.endswith("lora_A")) == 4 * N_LAYERS
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert any("experts" in n for n in frozen) and any("embed_tokens" in n for n in frozen)
    assert all("lora_" not in n for n in frozen)


def test_parameter_groups_split_lora_and_gate():
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    groups = A.parameter_groups(model, lr_lora=2e-5, lr_gate=1e-5)
    assert [g["name"] for g in groups] == ["lora", "gate"]
    assert groups[0]["lr"] == 2e-5 and groups[1]["lr"] == 1e-5
    assert len(groups[1]["params"]) == N_LAYERS - 1


def test_save_load_adapter_round_trip(tmp_path):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    with torch.no_grad():                                     # make B non-zero so it matters
        for n, p in A.trainable_parameters(model):
            p.add_(torch.randn_like(p) * 0.05)
    ids = sample_ids(2 * BLOCK)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        ref = training_forward(model, ids, BLOCK, valid, checkpoint_layers=False, block=BLOCK).hidden.clone()
    d = str(tmp_path / "ad")
    A.save_adapter(model, d)
    meta = json.load(open(os.path.join(d, A.ADAPTER_JSON)))
    assert meta["r"] == 4 and meta["alpha"] == 8.0 and meta["targets"] == list(A.ATTN_TARGETS)
    assert "config_hash" in meta and meta["model"]["num_experts"] == N_EXPERTS

    other = build_model()
    A.attach_lora(other, r=4, alpha=8.0)
    A.freeze_base(other)
    rep = A.load_adapter(other, d)
    assert rep["missing"] == 0 and rep["unexpected"] == 0
    with torch.no_grad():
        got = training_forward(other, ids, BLOCK, valid, checkpoint_layers=False, block=BLOCK).hidden
    assert torch.equal(ref, got)


def test_load_adapter_refuses_a_mismatched_rank(tmp_path):
    model = build_model(); A.attach_lora(model, r=4, alpha=8.0); A.freeze_base(model)
    d = str(tmp_path / "ad"); A.save_adapter(model, d)
    other = build_model(); A.attach_lora(other, r=2, alpha=8.0); A.freeze_base(other)
    with pytest.raises(AssertionError):
        A.load_adapter(other, d)


def test_merge_lora_matches_the_wrapped_model():
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, A.LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * 0.1)
    ids = sample_ids(2 * BLOCK)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        ref = training_forward(model, ids, BLOCK, valid, checkpoint_layers=False, block=BLOCK).hidden.clone()
    A.merge_lora(model)
    assert not any(isinstance(m, (A.LoRALinear, A.GateFP32)) for m in model.modules())
    with torch.no_grad():
        got = training_forward(model, ids, BLOCK, valid, checkpoint_layers=False, block=BLOCK).hidden
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (ref - got).abs().max()


# ===========================================================================
# training forward
# ===========================================================================
@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_training_forward_equals_the_stock_forward(n_blocks):
    """Pool off + LoRA at B=0 -> our forward is the remote forward, exactly."""
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    S = n_blocks * BLOCK
    ids = sample_ids(S)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    ref = stock_forward_reference(model, ids, block=BLOCK).last_hidden_state[0, S - BLOCK:]
    with torch.no_grad():
        got = training_forward(model, ids, S - BLOCK, valid, pool_M=None,
                               checkpoint_layers=False, block=BLOCK).hidden
    assert torch.equal(ref, got), (ref - got).abs().max()


def test_training_forward_with_checkpointing_matches():
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    S = 3 * BLOCK
    ids = sample_ids(S)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        off = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK).hidden
    on = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=True, block=BLOCK).hidden
    assert torch.equal(off, on.detach())
    assert on.requires_grad, "checkpointed forward must stay differentiable"


def test_block_tril_mask_is_block_causal():
    m = block_tril_bool(3 * BLOCK, torch.device("cpu"), BLOCK)[0, 0]
    assert m.shape == (3 * BLOCK, 3 * BLOCK)
    assert bool(m[:BLOCK, :BLOCK].all()) and not bool(m[:BLOCK, BLOCK:].any())
    assert bool(m[BLOCK:2 * BLOCK, :2 * BLOCK].all()) and not bool(m[BLOCK:2 * BLOCK, 2 * BLOCK:].any())
    with pytest.raises(AssertionError):
        block_tril_bool(5, torch.device("cpu"), BLOCK)


def test_training_forward_rejects_a_non_final_active_block():
    model = build_model()
    ids = sample_ids(3 * BLOCK)
    with pytest.raises(AssertionError):
        training_forward(model, ids, BLOCK, torch.ones(BLOCK, dtype=torch.bool), block=BLOCK)


def _dispatch(model, ids, lo, valid, pool_M, rule="full"):
    """(per-layer {expert -> rows}, ForwardOut) by recording what each expert received."""
    for blk in moe_blocks(model):
        blk.record(True)
    for blk in moe_blocks(model):
        for e in blk.experts:
            e.received.clear()
    # our forward does not go through TinyMoE.forward, so record from the expert calls
    hooks, seen = [], [{} for _ in moe_blocks(model)]

    def make(li, ei):
        def hook(mod, inp, out):
            seen[li].setdefault(ei, 0)
            seen[li][ei] += inp[0].shape[0]
        return hook
    for li, blk in enumerate(moe_blocks(model)):
        for ei, e in enumerate(blk.experts):
            hooks.append(e.register_forward_hook(make(li, ei)))
    with torch.no_grad():
        out = training_forward(model, ids, lo, valid, pool_M=pool_M, pool_rule=rule,
                               checkpoint_layers=False, return_pools=True, block=BLOCK)
    for h in hooks:
        h.remove()
    return seen, out


def test_pool_M_equal_to_E_is_identical_to_no_pool():
    model = build_model()
    S = 3 * BLOCK
    ids = sample_ids(S)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        a = training_forward(model, ids, S - BLOCK, valid, pool_M=None, checkpoint_layers=False, block=BLOCK)
        b = training_forward(model, ids, S - BLOCK, valid, pool_M=N_EXPERTS, checkpoint_layers=False, block=BLOCK)
        c = training_forward(model, ids, S - BLOCK, valid, pool_M=N_EXPERTS + 5, checkpoint_layers=False, block=BLOCK)
    assert torch.equal(a.hidden, b.hidden) and torch.equal(a.hidden, c.hidden)
    assert torch.equal(a.unique_experts, b.unique_experts)


@pytest.mark.parametrize("rule", ["full", "top8"])
def test_small_pool_restricts_active_rows_only(rule):
    model = build_model()
    S = 3 * BLOCK
    lo = S - BLOCK
    ids = sample_ids(S)
    valid = torch.tensor([False, True, True, True])
    M = 3

    # what the routing does with no pool: read the selections straight off the router
    ref_sel = _router_selection(model, ids, lo, valid, None, rule)
    res_sel = _router_selection(model, ids, lo, valid, M, rule)
    _, out = _dispatch(model, ids, lo, valid, M, rule)
    assert out.pools.shape == (N_LAYERS, N_EXPERTS)

    li = 0
    for layer_i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, TinyMoE):
            assert int(out.unique_experts[layer_i]) == 0
            continue
        pool = out.pools[layer_i]
        assert int(pool.sum()) == M
        act = res_sel[li][lo:]
        assert bool(pool[act.reshape(-1)].all()), "an active row escaped the pool"
        assert torch.equal(res_sel[li][:lo], ref_sel[li][:lo]), "a prefix row was restricted"
        assert int(out.unique_experts[layer_i]) == int(torch.unique(act).numel())
        assert int(out.unique_experts[layer_i]) <= M
        li += 1


def _router_selection(model, ids, lo, valid, pool_M, rule):
    """The top-k expert ids per row, per MoE layer, reproduced from the router hidden states."""
    sel = []
    caps = []

    def hook(mod, inp, out):
        caps.append(inp[0].detach().reshape(-1, HID))
    # our forward never calls TinyMoE.forward; hook the router gate, whose input is
    # exactly the flattened hidden states the routing is computed from
    hs = [blk.gate.register_forward_hook(hook) for blk in moe_blocks(model)]
    with torch.no_grad():
        training_forward(model, ids, lo, valid, pool_M=pool_M, pool_rule=rule,
                         checkpoint_layers=False, block=BLOCK)
    for h in hs:
        h.remove()
    for blk, x in zip(moe_blocks(model), caps):
        w = F.softmax(blk.gate(x), dim=1, dtype=torch.float)
        if pool_M is not None and pool_M < N_EXPERTS:
            pool = expert_pool(w[lo:], valid, pool_M, rule, TOP_K)
            keep = torch.ones_like(w)
            keep[lo:] = pool.to(w.dtype).unsqueeze(0)
            w = w * keep
        sel.append(torch.topk(w, TOP_K, dim=-1).indices)
    return sel


def test_pool_votes_only_from_valid_rows():
    """A slot marked invalid must not influence the pool."""
    torch.manual_seed(1)
    w = torch.zeros(4, N_EXPERTS)
    w[0, 0] = 1.0                                     # invalid row votes for expert 0
    w[1, 5] = 1.0
    w[2, 6] = 1.0
    w[3, 7] = 1.0
    valid = torch.tensor([False, True, True, True])
    pool = expert_pool(w, valid, M=3, rule="full", top_k=TOP_K)
    assert pool.tolist().count(True) == 3
    assert not bool(pool[0]) and bool(pool[5]) and bool(pool[6]) and bool(pool[7])


def test_pool_tie_break_is_the_lowest_expert_id():
    w = torch.zeros(1, N_EXPERTS)                     # all votes equal (zero)
    pool = expert_pool(w, torch.ones(1, dtype=torch.bool), M=3, rule="full", top_k=TOP_K)
    assert torch.equal(pool, torch.tensor([True, True, True] + [False] * (N_EXPERTS - 3)))


def test_unique_experts_matches_the_dispatch():
    model = build_model()
    S = 2 * BLOCK
    ids = sample_ids(S)
    valid = torch.ones(BLOCK, dtype=torch.bool)
    sel = _router_selection(model, ids, S - BLOCK, valid, None, "full")
    with torch.no_grad():
        out = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK)
    li = 0
    for layer_i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, TinyMoE):
            assert int(out.unique_experts[layer_i]) == int(torch.unique(sel[li][S - BLOCK:]).numel())
            li += 1


# ===========================================================================
# loss
# ===========================================================================
def test_loss_on_hand_computed_logits():
    V, beta, temp = 6, 2.0, 0.5
    mask_id = 5
    logits = torch.tensor([[9.0, 1.0, 0.0, 0.0, 0.0, 3.0],       # argmax = 0, MASK column non-trivial
                           [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])      # argmax = 1
    targets = torch.tensor([0, 3])                               # row 0 correct, row 1 wrong
    loss, st = masked_ce_and_certainty(logits, targets, beta=beta, temp=temp, mask_id=mask_id)

    ce = F.cross_entropy(logits, targets, reduction="mean")
    t = logits[0] / temp
    t = torch.cat([t[:mask_id], t[mask_id + 1:]])          # the MASK column is dropped
    p = torch.softmax(t, dim=-1)
    ent = -(p * torch.log(p)).sum()
    assert torch.allclose(st["ce"], ce)
    assert torch.allclose(st["cert"], ent, atol=1e-6)
    assert torch.allclose(loss, ce + beta * ent, atol=1e-6)
    assert float(st["n_rows"]) == 2 and float(st["n_correct"]) == 1 and float(st["acc"]) == 0.5
    assert torch.allclose(st["top1"], torch.softmax(logits, -1).max(-1).values.mean())


def test_certainty_ignores_the_mask_logit():
    """Raising the MASK logit must not change the certainty term."""
    mask_id = 5
    a = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, -5.0]])
    b = a.clone(); b[0, mask_id] = 2.9          # still below the argmax, so the row stays "correct"
    ta = torch.tensor([0])
    _, sa = masked_ce_and_certainty(a, ta, mask_id=mask_id)
    _, sb = masked_ce_and_certainty(b, ta, mask_id=mask_id)
    assert torch.allclose(sa["cert"], sb["cert"])


def test_loss_is_zero_certainty_when_nothing_is_correct():
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    loss, st = masked_ce_and_certainty(logits, torch.tensor([1]), beta=2.0, mask_id=2)
    assert float(st["cert"]) == 0.0 and float(st["n_correct"]) == 0
    assert torch.allclose(loss, st["ce"])


def test_loss_on_no_rows_is_finite_and_connected():
    logits = torch.zeros(0, 4, requires_grad=True)
    loss, st = masked_ce_and_certainty(logits, torch.zeros(0, dtype=torch.long), mask_id=3)
    assert torch.isfinite(loss) and float(st["n_rows"]) == 0
    loss.backward()


def test_active_logits_only_touches_the_loss_rows():
    model = build_model()
    h = torch.randn(BLOCK, HID)
    rows = torch.tensor([True, False, True, False])
    out = active_logits(model, h, rows)
    assert out.shape == (2, VOCAB) and out.dtype == torch.float32
    assert torch.allclose(out, model.lm_head(h[rows]).float())


# ===========================================================================
# build_train_set
# ===========================================================================
def synth_trace(nb=3, bl=BLOCK, prompt_length=6, n_pass=2, with_draft=True, seed=0):
    rng = np.random.default_rng(seed)
    lo0 = (prompt_length // bl) * bl
    tokens = rng.integers(0, VOCAB - 1, (nb, bl)).astype(np.int32)
    valid = np.ones((nb, bl), bool)
    tail = prompt_length - lo0
    valid[0, :tail] = False
    committed = np.zeros((nb, n_pass, bl), bool)
    for b in range(nb):
        v = np.where(valid[b])[0]
        committed[b, 0, v[: len(v) // 2]] = True
        committed[b, 1, v[len(v) // 2:]] = True
    draft = (tokens.copy() if with_draft else np.full((nb, bl), -1, np.int32))
    if with_draft:
        draft[:, 0] = (draft[:, 0] + 1) % (VOCAB - 1)          # make some draft tokens wrong
    prompt_ids = np.concatenate([rng.integers(0, VOCAB - 1, lo0).astype(np.int32),
                                 tokens[0, :tail]])
    trace = {"tokens": tokens, "valid": valid, "committed": committed,
             "n_pass": np.full(nb, n_pass, np.int32),
             "block_start": (lo0 + bl * np.arange(nb)).astype(np.int32),
             "prompt_length": np.int32(prompt_length), "block_length": np.int32(bl),
             "draft_tokens": draft, "accepted": np.zeros((nb, bl), bool)}
    return trace, prompt_ids


def write_source(tmp_path, n_items=3, split="train", with_draft=True, correct=None, nb=3):
    recs = tmp_path / "recs"; traces = tmp_path / "traces" / split
    recs.mkdir(parents=True, exist_ok=True); traces.mkdir(parents=True, exist_ok=True)
    prompts = {}
    with open(recs / f"cell.{split}.shard0of1.jsonl", "w") as fh:
        for i in range(n_items):
            tr, pids = synth_trace(nb=nb, with_draft=with_draft, seed=i)
            np.savez_compressed(traces / f"{i:05d}.npz", **tr)
            prompts[i] = pids.tolist()
            fh.write(json.dumps({"idx": i, "split": split, "cell": "cell", "config_hash": "x",
                                 "correct": (1 if correct is None else correct[i])}) + "\n")
    return str(recs), str(tmp_path / "traces"), (lambda rec: prompts[int(rec["idx"])])


def test_build_train_set_writes_items_and_index(tmp_path):
    recs, traces, pfn = write_source(tmp_path)
    out = str(tmp_path / "out")
    s = B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
                max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 3 and s["usable_blocks"] == 9 and s["draft_blocks"] == 9
    idx = [json.loads(l) for l in open(os.path.join(out, "index.jsonl"))]
    assert len(idx) == 3 and idx[0]["usable"] == [True] * 3
    d = np.load(os.path.join(out, idx[0]["file"]))
    assert d["tokens"].shape == (3, BLOCK) and d["draft"].shape == (3, BLOCK)
    assert bool(d["has_draft"].all())


def test_build_train_set_filters(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=4, correct=[1, 0, 1, 0])
    out = str(tmp_path / "out")
    s = B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
                only_correct=True, exclude_idxs={2}, max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 1                      # idx 0 kept; 1,3 wrong; 2 frozen
    assert s["dropped"]["incorrect"] == 2 and s["dropped"]["frozen"] == 1


def test_build_train_set_max_seq_marks_blocks_unusable(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=1, nb=4)
    out = str(tmp_path / "out")
    s = B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
                max_seq=2 * BLOCK + 4, block=BLOCK, verbose=False)     # block 0 hi=8, block 1 hi=12 > 8+4
    entry = json.loads(open(os.path.join(out, "index.jsonl")).readline())
    assert entry["usable"] == [True, True, False, False]
    assert s["usable_blocks"] == 2


def test_build_train_set_drops_a_test_split(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=2, split="test")
    s = B.build(B.read_records(recs), traces, str(tmp_path / "out"), layout="da",
                prompt_ids_fn=pfn, max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 0 and s["dropped"]["test_split"] == 2


def test_build_train_set_rejects_a_wrong_prompt(tmp_path):
    recs, traces, _ = write_source(tmp_path, n_items=1)
    bad = lambda rec: [1] * 6                                        # noqa: E731
    s = B.build(B.read_records(recs), traces, str(tmp_path / "out"), layout="da",
                prompt_ids_fn=bad, max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 0 and s["dropped"]["bad"] == 1


def test_build_train_set_without_drafts(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=1, with_draft=False)
    out = str(tmp_path / "out")
    s = B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
                max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 1 and s["draft_blocks"] == 0


# ===========================================================================
# dataset / StateSampler
# ===========================================================================
@pytest.fixture()
def sampler(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=4)
    out = str(tmp_path / "out")
    B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
            max_seq=None, block=BLOCK, verbose=False)
    cfg = SamplerConfig(index_path=os.path.join(out, "index.jsonl"), seed=7, mask_id=MASK,
                        block=BLOCK, max_seq=None)
    return StateSampler(cfg)


def test_sampler_enumerates_every_usable_block(sampler):
    assert len(sampler) == 4 * 3
    assert sampler.summary()["draft_examples"] == 12


def test_state_invariants(sampler):
    """committed slots hold the target, open slots hold a draft token or MASK, prefix is clean."""
    seen_regimes = set()
    for epoch in range(6):
        for k in range(len(sampler)):
            st = sampler.state(k, epoch=epoch)
            S = st.input_ids.shape[0]
            assert S == st.active_lo + BLOCK and st.meta["seq_len"] == S
            blk = st.input_ids[st.active_lo:]
            item = sampler._item(sampler.examples[k][0])
            b = sampler.examples[k][1]
            draft = torch.from_numpy(item["draft"][b].astype(np.int64))

            closed = st.valid & ~st.open_mask
            assert torch.equal(blk[closed], st.targets[closed]), "a committed slot lost its target"
            assert torch.equal(blk[~st.valid], st.targets[~st.valid]), "the prompt tail moved"
            for i in torch.nonzero(st.open_mask).flatten().tolist():
                assert int(blk[i]) in (MASK, int(draft[i])), (i, int(blk[i]))
            assert bool((st.open_mask & ~st.valid).sum() == 0), "loss rows must be valid slots"
            assert bool(st.open_mask.any()), "a state with no loss row is useless"
            # the prefix is the clean sequence
            lo0 = int(item["block_start"][0])
            clean = np.concatenate([item["prompt_ids"][:lo0], item["tokens"].reshape(-1)])
            assert np.array_equal(st.input_ids[:st.active_lo].numpy(), clean[:st.active_lo])
            seen_regimes.add(("allmask" if st.meta["allmask"] else "draft", st.meta["pass"] > 0))
    assert ("allmask", False) in seen_regimes and ("draft", False) in seen_regimes
    assert any(p for _, p in seen_regimes), "no state used a recorded committed set"


def test_state_is_deterministic(sampler):
    a = sampler.state(3, epoch=1)
    b = sampler.state(3, epoch=1)
    assert torch.equal(a.input_ids, b.input_ids) and a.meta == b.meta
    c = sampler.state(3, epoch=2)
    assert a.meta["epoch"] != c.meta["epoch"]
    other = StateSampler(SamplerConfig(index_path=sampler.cfg.index_path, seed=8, mask_id=MASK,
                                       block=BLOCK, max_seq=None))
    assert any(not torch.equal(sampler.state(k).input_ids, other.state(k).input_ids)
               for k in range(len(sampler)))


def test_allmask_regime_masks_every_open_slot(sampler):
    for k in range(len(sampler)):
        st = sampler.state(k, regime="allmask_pass1")
        blk = st.input_ids[st.active_lo:]
        assert torch.equal(blk[st.open_mask], torch.full((int(st.open_mask.sum()),), MASK))
        assert st.meta["pass"] == 0 and torch.equal(st.open_mask, st.valid)


def test_draft_pass1_regime_fills_every_open_slot(sampler):
    for k in range(len(sampler)):
        st = sampler.state(k, regime="draft_pass1")
        item = sampler._item(sampler.examples[k][0]); b = sampler.examples[k][1]
        draft = torch.from_numpy(item["draft"][b].astype(np.int64))
        blk = st.input_ids[st.active_lo:]
        assert torch.equal(blk[st.open_mask], draft[st.open_mask])
        assert MASK not in blk[st.open_mask].tolist()


def test_block_without_a_draft_is_all_mask(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=1, with_draft=False)
    out = str(tmp_path / "out")
    B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
            max_seq=None, block=BLOCK, verbose=False)
    s = StateSampler(SamplerConfig(index_path=os.path.join(out, "index.jsonl"), seed=1,
                                   mask_id=MASK, block=BLOCK, max_seq=None))
    for regime in REGIMES:
        for k in range(len(s)):
            st = s.state(k, regime=regime)
            blk = st.input_ids[st.active_lo:]
            assert torch.equal(blk[st.open_mask], torch.full((int(st.open_mask.sum()),), MASK))


def test_require_draft_drops_draftless_blocks(tmp_path):
    recs, traces, pfn = write_source(tmp_path, n_items=1, with_draft=False)
    out = str(tmp_path / "out")
    B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn,
            max_seq=None, block=BLOCK, verbose=False)
    cfg = SamplerConfig(index_path=os.path.join(out, "index.jsonl"), mask_id=MASK, block=BLOCK,
                        max_seq=None, require_draft=True)
    assert len(StateSampler(cfg)) == 0


def test_holdout_split_is_disjoint(sampler):
    tr, ev = sampler.split_holdout(2)
    assert len(ev.entries) == 2 and len(tr.entries) == 2
    assert not ({e["idx"] for e in tr.entries} & {e["idx"] for e in ev.entries})
    assert len(tr) + len(ev) == len(sampler)


def test_epoch_order_is_a_permutation(sampler):
    o = sampler.epoch_order(0)
    assert sorted(o.tolist()) == list(range(len(sampler)))
    assert not torch.equal(o, sampler.epoch_order(1))


# ===========================================================================
# train.py pieces + one real optimiser step
# ===========================================================================
def test_lr_schedule_warms_up_then_decays():
    assert T.lr_scale(0, 10, 100) == pytest.approx(0.1)
    assert T.lr_scale(9, 10, 100) == pytest.approx(1.0)
    assert T.lr_scale(10, 10, 100) == pytest.approx(1.0)
    assert T.lr_scale(99, 10, 100, 0.1) == pytest.approx(0.1, abs=1e-3)
    assert T.lr_scale(200, 10, 100, 0.1) == pytest.approx(0.1)


def test_pool_for_step_is_rank_independent_and_uses_every_M():
    Ms = T.parse_pool_Ms("2,3,4,128", N_EXPERTS)
    assert Ms == [2, 3, 4, None]
    a = [T.pool_for_step(s, Ms, seed=0) for s in range(200)]
    b = [T.pool_for_step(s, Ms, seed=0) for s in range(200)]
    assert a == b and set(a) == set(Ms)
    assert T.parse_pool_Ms("0", N_EXPERTS) == [None]


def test_one_optimiser_step_moves_only_the_adapter(sampler):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0)
    A.freeze_base(model)
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(A.parameter_groups(model, 1e-2, 1e-2))

    st = sampler.state(0)
    loss, stats, logits, rows = T.forward_state(model, st, pool_M=3, pool_rule="full",
                                                checkpoint_layers=True, beta=2.0, temp=0.5,
                                                mask_id=MASK, device=torch.device("cpu"), block=BLOCK)
    assert torch.isfinite(loss)
    assert logits.shape == (int(st.open_mask.sum()), VOCAB) and logits.dtype == torch.float32
    loss.backward()
    for n, p in A.trainable_parameters(model):
        assert p.grad is not None, n
    opt.step()

    moved = {n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert moved, "the step changed nothing"
    for n in moved:
        assert n.endswith(("lora_A", "lora_B")) or n.endswith("mlp.gate.weight"), n
    assert any(n.endswith("mlp.gate.weight") for n in moved), "the routers did not train"
    assert any(n.endswith("lora_B") for n in moved), "LoRA did not train"


def test_forward_state_uses_only_the_open_rows(sampler):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0); A.freeze_base(model)
    st = sampler.state(1)
    _, stats, logits, rows = T.forward_state(model, st, None, "full", False, 2.0, 0.5, MASK,
                                             torch.device("cpu"), block=BLOCK)
    assert int(stats["n_rows"]) == int(st.open_mask.sum()) == logits.shape[0]
    assert torch.equal(rows, st.open_mask)
    assert float(stats["uniq_experts"]) > 0


def test_evaluate_reports_both_regimes(sampler):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0); A.freeze_base(model)
    ev = T.evaluate(model, sampler, n=4, pool_M=None, pool_rule="full", mask_id=MASK,
                    device=torch.device("cpu"), block=BLOCK)
    for key in ("draft_pass1_acc", "allmask_pass1_acc", "draft_pass1_conf_precision",
                "allmask_pass1_conf_frac"):
        assert key in ev
    assert 0.0 <= ev["draft_pass1_acc"] <= 1.0


def test_checkpoint_round_trip(tmp_path, sampler):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0); A.freeze_base(model)
    opt = torch.optim.AdamW(A.parameter_groups(model, 1e-2, 1e-2))
    st = sampler.state(0)
    loss, _, _, _ = T.forward_state(model, st, None, "full", False, 2.0, 0.5, MASK,
                                    torch.device("cpu"), block=BLOCK)
    loss.backward(); opt.step()
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    T.save_checkpoint(model, opt, out, 7)
    assert open(os.path.join(out, "latest")).read().strip() == "step000007"

    fresh = build_model()
    A.attach_lora(fresh, r=4, alpha=8.0); A.freeze_base(fresh)
    opt2 = torch.optim.AdamW(A.parameter_groups(fresh, 1e-2, 1e-2))
    step = T.load_checkpoint(fresh, opt2, out, torch.device("cpu"))
    assert step == 7
    for (n, p), (n2, p2) in zip(A.trainable_parameters(model), A.trainable_parameters(fresh)):
        assert n == n2 and torch.equal(p.detach(), p2.detach())


def test_p_draftfull_puts_every_open_slot_on_the_draft(tmp_path):
    """With p_allmask=0 and p_draftfull=1 every sampled draft state holds the draft at EVERY open slot (m = 0)."""
    recs, traces, pfn = write_source(tmp_path, n_items=4)
    out = str(tmp_path / "out")
    B.build(B.read_records(recs), traces, out, layout="da", prompt_ids_fn=pfn, max_seq=None, block=BLOCK, verbose=False)
    cfg = SamplerConfig(index_path=os.path.join(out, "index.jsonl"), seed=3, mask_id=MASK, block=BLOCK, max_seq=None,
                        p_pass1=0.5, p_allmask=0.0, p_draftfull=1.0)
    S = StateSampler(cfg)
    n_open = 0
    for epoch in range(4):
        for k in range(len(S)):
            st = S.state(k, epoch=epoch)
            blk = st.input_ids[st.active_lo:st.active_lo + BLOCK]
            open_valid = st.open_mask & st.valid
            n_open += int(open_valid.sum())
            assert not bool((blk[open_valid] == MASK).any()), "an open slot held MASK although p_draftfull=1"
    assert n_open > 0
    # and the default (p_draftfull=0) still produces MASKed open slots somewhere
    S0 = StateSampler(SamplerConfig(index_path=os.path.join(out, "index.jsonl"), seed=3, mask_id=MASK, block=BLOCK, max_seq=None))
    assert any(bool((S0.state(k, epoch=e).input_ids[S0.state(k, epoch=e).active_lo:S0.state(k, epoch=e).active_lo + BLOCK][S0.state(k, epoch=e).open_mask] == MASK).any())
               for e in range(4) for k in range(len(S0)))
