"""Row-invariant execution for SDAR-30B-A3B (research_v1; a LABELLED departure).

Problem. cuBLAS picks a bf16 GEMM kernel per row count M, so a state evaluated
inside a batch of two gets different roundings than the same state evaluated
alone (kernel_shape_probe: 25/34 linears M-dependent, first divergence at layer
0; SDPA, RMSNorm, router softmax/top-k are NOT shape dependent). An exact
speculative decoder needs each row's result to be independent of how many
other rows share the forward.

Two remedies, both verified bit-exact end to end in kernel_shape_probe:

  mode "perrow" (default): dense GEMMs and the layer norms run PER ROW at their
     stock shapes (attention module, router gate, lm_head, RMSNorm loop over rows
     when B > 1; the fp32 norm reduction is itself row-count dependent below 32 rows), and
     every expert GEMM runs on a FIXED row count M_E over the concatenated rows'
     tokens (zero rows appended, sliced off before the weighted index_add_).
     A single-row forward differs from stock ONLY in the expert GEMM padding;
     a multi-row forward reproduces the single-row results bit for bit while
     each expert is still invoked once for all rows (the work sharing).
  mode "pad": every forward runs with a FIXED batch width B_PAD (dummy rows are
     copies of row 0, discarded) plus the fixed-M_E experts. Simpler, but every
     forward pays for B_PAD rows (~+20% in the reference loop).

Numerics. Neither mode is bit-identical to the stock path; each is a
different, equally valid bf16 evaluation of the same algorithm. Reference
cells compared with a rowinv speculative decoder must run in the same mode.

Counters / expert-set logs go through desvote.CTRL exactly as the vote forward
does (unrestricted only: CTRL.M must be None).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .desvote import CTRL
from .routing import moe_blocks


class RowInvCfg:
    def __init__(self):
        self.mode: str = "perrow"
        self.bpad: int = 1
        self.m_e: int = 160
        self.enabled = False
        self.n_forwards = 0
        self.n_padded_rows = 0
        self.n_row_loops = 0

CFG = RowInvCfg()


# ---------------------------------------------------------------------------
# experts on a fixed row count
# ---------------------------------------------------------------------------
def _expert_padded(expert_layer, x: torch.Tensor, m_e: int) -> torch.Tensor:
    """expert_layer on x [n, D] evaluated in chunks of exactly m_e rows (zero rows appended)."""
    n, d = x.shape
    outs = []
    for s in range(0, max(n, 1), m_e):
        chunk = x[s:s + m_e]
        c = chunk.shape[0]
        xp = torch.zeros(m_e, d, dtype=x.dtype, device=x.device)
        xp[:c] = chunk
        outs.append(expert_layer(xp)[:c])
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def rowinv_moe_forward(self, hidden_states: torch.Tensor):
    """desvote.vote_moe_forward's unrestricted branch with fixed-row expert GEMMs
    and (mode perrow) a per-row router gate GEMM."""
    assert CTRL.M is None and not CTRL.recording, "rowinv supports the unrestricted path only"
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if CFG.mode == "perrow" and batch_size > 1:
        router_logits = torch.cat([self.gate(hidden_states[r]) for r in range(batch_size)], 0)
        CFG.n_row_loops += 1
    else:
        router_logits = self.gate(hidden_states.view(-1, hidden_dim))
    hidden_states = hidden_states.view(-1, hidden_dim)
    w_full = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(w_full, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim),
                                      dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    hit = torch.unique(selected_experts).tolist()
    CTRL.unique_sum += len(hit); CTRL.layer_forwards += 1
    if CTRL.log is not None:
        CTRL.log.append((self._layer_idx, None, np.asarray(hit, dtype=np.uint8)))
    for expert_idx in hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        y = _expert_padded(expert_layer, current_state, CFG.m_e)
        current_hidden_states = y * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


# ---------------------------------------------------------------------------
# mode perrow: dense modules per row
# ---------------------------------------------------------------------------
class _RowView:
    """One batch row of a DynamicCache, read-only (store_kv=False paths only)."""

    def __init__(self, cache, r):
        self.cache, self.r = cache, r

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, l):
        k, v = self.cache[l]
        return k[self.r:self.r + 1], v[self.r:self.r + 1]

    def get_seq_length(self, layer_idx=0):
        return self.cache.get_seq_length(layer_idx)

    def update(self, *a, **k):
        raise RuntimeError("rowinv perrow: cache update inside a per-row attention call")


def _make_attn_forward(orig):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None,
                cache_position=None, **kwargs):
        B = hidden_states.shape[0]
        if B == 1 or kwargs.get("store_kv", False) or past_key_value is None:
            return orig(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        cos, sin = position_embeddings
        outs = []
        for r in range(B):
            am = attention_mask[r:r + 1] if attention_mask is not None else None
            o, _ = orig(self, hidden_states[r:r + 1], (cos[r:r + 1], sin[r:r + 1]), am, _RowView(past_key_value, r),
                        cache_position, **kwargs)
            outs.append(o)
        CFG.n_row_loops += 1
        return torch.cat(outs, 0), None
    return forward


def _make_norm_forward(orig):
    """RMSNorm per batch row when B > 1: torch's fp32 mean over D picks a different
    reduction order for very small row counts (4 vs 8 rows differ in ~50% of fp32 sums),
    which occasionally crosses a bf16 rounding boundary (narrow-row probe)."""
    def forward(self, x):
        if x.dim() == 3 and x.shape[0] > 1:
            CFG.n_row_loops += 1
            return torch.cat([orig(self, x[r:r + 1]) for r in range(x.shape[0])], 0)
        return orig(self, x)
    return forward


def _make_lm_head_forward(lin):
    def forward(x):
        if x.dim() == 3 and x.shape[0] > 1:
            CFG.n_row_loops += 1
            return torch.cat([F.linear(x[r:r + 1], lin.weight, lin.bias) for r in range(x.shape[0])], 0)
        return F.linear(x, lin.weight, lin.bias)
    return forward


# ---------------------------------------------------------------------------
# mode pad: fixed batch width
# ---------------------------------------------------------------------------
def _pad_batch(t: torch.Tensor, bpad: int):
    b = t.shape[0]
    if b == bpad:
        return t
    return torch.cat([t, t[:1].expand(bpad - b, *t.shape[1:])], 0)


def _padded_cache(cache, bpad: int, b: int):
    from transformers.cache_utils import DynamicCache
    pc = DynamicCache()
    for l in range(len(cache)):
        k, v = cache[l]
        if k.shape[0] == bpad:
            k2, v2 = k, v
        elif k.shape[0] == 1:
            k2, v2 = k.expand(bpad, -1, -1, -1), v.expand(bpad, -1, -1, -1)
        else:
            assert k.shape[0] == b, (k.shape, b)
            k2 = torch.cat([k, k[:1].expand(bpad - b, -1, -1, -1)], 0)
            v2 = torch.cat([v, v[:1].expand(bpad - b, -1, -1, -1)], 0)
        pc.update(k2, v2, l)
    return pc


def _make_forward(model, orig):
    def forward(*args, **kw):
        assert not args, "rowinv: call the model with keyword arguments"
        input_ids = kw["input_ids"]
        b = input_ids.shape[0]
        bpad = CFG.bpad
        assert 1 <= b <= bpad, f"batch {b} exceeds B_PAD {bpad}"
        cache = kw.get("past_key_values")
        store = bool(kw.get("store_kv", False))
        kw2 = dict(kw)
        kw2["input_ids"] = _pad_batch(input_ids, bpad)
        if kw.get("attention_mask") is not None:
            kw2["attention_mask"] = _pad_batch(kw["attention_mask"], bpad)
        if kw.get("position_ids") is not None:
            kw2["position_ids"] = _pad_batch(kw["position_ids"], bpad)
        s0 = cache.get_seq_length() if cache is not None else 0
        pc = _padded_cache(cache, bpad, b) if cache is not None else None
        kw2["past_key_values"] = pc
        out = orig(**kw2)
        CFG.n_forwards += 1; CFG.n_padded_rows += bpad - b
        if store and cache is not None:
            for l in range(len(pc)):
                k, v = pc[l]
                cache.update(k[:b, :, s0:, :].contiguous(), v[:b, :, s0:, :].contiguous(), l)
        out.logits = out.logits[:b]
        out.past_key_values = cache
        return out
    return forward


# ---------------------------------------------------------------------------
def enable_rowinv(model, mode: str = "perrow", bpad: int = 2, m_e: Optional[int] = None) -> dict:
    """Patch the model in place. perrow: m_e defaults to 160 (5 states x 32 slots, one chunk);
    pad: m_e defaults to bpad*32. Longer prefills are chunked, which keeps row-invariance."""
    assert not CFG.enabled, "rowinv already enabled"
    assert mode in ("perrow", "pad")
    CFG.mode = mode; CFG.enabled = True; CFG.n_forwards = 0; CFG.n_padded_rows = 0; CFG.n_row_loops = 0
    CFG.bpad = int(bpad) if mode == "pad" else 1
    CFG.m_e = int(m_e or (bpad * 32 if mode == "pad" else 160))
    blocks = moe_blocks(model)
    for i, blk in enumerate(blocks):
        assert getattr(blk, "norm_topk_prob", False)
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk._prev_forward = blk.forward
        blk._layer_idx = i
        blk.forward = rowinv_moe_forward.__get__(blk, type(blk))
    CTRL.n_layers = len(blocks); CTRL.n_experts = blocks[0].num_experts
    if mode == "pad":
        model._rowinv_orig_forward = model.forward
        model.forward = _make_forward(model, model.forward)
    else:
        attn_cls = type(model.model.layers[0].self_attn)
        model._rowinv_attn_cls = attn_cls
        model._rowinv_attn_orig = attn_cls.forward
        attn_cls.forward = _make_attn_forward(attn_cls.forward)
        model._rowinv_lm_orig = model.lm_head.forward
        model.lm_head.forward = _make_lm_head_forward(model.lm_head)
        norm_cls = type(model.model.layers[0].input_layernorm)
        model._rowinv_norm_cls = norm_cls
        model._rowinv_norm_orig = norm_cls.forward
        norm_cls.forward = _make_norm_forward(norm_cls.forward)
    return {"mode": mode, "bpad": CFG.bpad, "m_e": CFG.m_e, "patched_moe_blocks": len(blocks)}


def disable_rowinv(model) -> None:
    assert CFG.enabled
    for blk in moe_blocks(model):
        if hasattr(blk, "_prev_forward"):
            blk.forward = blk._prev_forward
            del blk._prev_forward
    if CFG.mode == "pad":
        model.forward = model._rowinv_orig_forward
        del model._rowinv_orig_forward
    else:
        model._rowinv_attn_cls.forward = model._rowinv_attn_orig
        model.lm_head.forward = model._rowinv_lm_orig
        model._rowinv_norm_cls.forward = model._rowinv_norm_orig
        del model._rowinv_attn_cls, model._rowinv_attn_orig, model._rowinv_lm_orig, model._rowinv_norm_cls, model._rowinv_norm_orig
    CFG.enabled = False
