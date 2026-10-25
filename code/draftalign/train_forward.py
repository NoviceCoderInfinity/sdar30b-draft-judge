"""Training forward for SDAR-30B-A3B-Chat-b32 (TRAINER_SPEC_v0 section 2).

Why this file exists at all -- three defects of the remote code
(hf_cache/.../c351bbc.../modeling_sdar_moe.py, cited by line below):

  * `SDARMoeForCausalLM.forward(labels=...)` computes a SHIFTED next-token loss
    (:1033-1036 -> `self.loss_function`), which is wrong for a diffusion model.
    We never pass `labels`; the loss is built from hidden states here.
  * the gradient-checkpointing branch (:622-634) is broken: it references an
    undefined `causal_mask` (the `_update_causal_mask` call is commented out at
    :604-606) and passes the layer's arguments positionally in the wrong order.
    We checkpoint per layer ourselves.
  * `SDARMoeSparseMoeBlock.forward` loops over all 128 experts (:307), i.e. 128
    `torch.where` syncs per layer. We use the hit-experts-only loop of
    `sdar_harness/fastmoe.py:44-50` -- bit-exact, because an expert with no rows
    contributes an empty `index_add_` -- keeping the SAME ascending expert order
    so the fp32/bf16 accumulation order is unchanged.

Everything else is a line-for-line mirror of the remote forward, so that with the
pool off and LoRA at B=0 `training_forward` == `stock_forward_reference`:

    mirror of                                     remote lines
    ---------------------------------------------------------
    rotate_half / apply_rotary_pos_emb            68-99
    SDARMoeAttention.forward                      190-242
    SDARMoeSparseMoeBlock.forward                 282-324
    SDARMoeDecoderLayer.forward                   415-443
    SDARMoeModel.forward (embed/rotary/layers)    592-658

Attention is the block-tril of the absolute 32-grid: every block attends
bidirectionally to itself and to all earlier blocks, which is exactly what the
inference KV cache implements (prefill with `seqcache.block_tril_mask`, then
per-block forwards against the cache with an all-ones mask).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from draftalign.adapter import flag_scope, get_draft_embed

BLOCK = 32
POOL_RULES = ("full", "top8")
_EPS = 1e-20                       # guards 0/0 only; a no-op for every normal router row


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def block_tril_bool(seq_len: int, device, block: int = BLOCK) -> torch.Tensor:
    """[1, 1, S, S] bool block-causal mask on the absolute `block`-grid.

    Same content as `seqcache.block_tril_mask` (which returns the [1, S, S] 0/1
    float form the decoder uses); 4-D and bool here so that the remote attention's
    `attention_mask.bool()` (:218) is a no-op and SDPA gets one canonical tensor.
    """
    assert seq_len % block == 0, f"S={seq_len} is not on the {block}-grid"
    nb = seq_len // block
    bm = torch.tril(torch.ones(nb, nb, dtype=torch.bool, device=device))
    return bm.repeat_interleave(block, 0).repeat_interleave(block, 1)[None, None]


# ---------------------------------------------------------------------------
# rotary (remote :68-99)
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# attention (remote :190-242, with past_key_value always None)
# ---------------------------------------------------------------------------
def attn_forward(attn, hidden_states: torch.Tensor, position_embeddings, attn_mask: Optional[torch.Tensor]):
    """`attn_mask` None <=> the remote `torch.all(attention_mask)` branch (:219-226).

    The branch is taken on the host (S == 32 <=> the block-tril is all ones) instead
    of by `if torch.all(mask)`, which would be a device sync in every layer.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    out = F.scaled_dot_product_attention(
        query=query_states, key=key_states, value=value_states,
        attn_mask=attn_mask, is_causal=False, scale=attn.scaling, enable_gqa=True)
    out = out.transpose(1, 2).contiguous()
    out = out.reshape(*input_shape, -1).contiguous()
    return attn.o_proj(out)


# ---------------------------------------------------------------------------
# MoE (remote :282-324 + fastmoe.py:44-50) with the dMoE / DES pool
# ---------------------------------------------------------------------------
def expert_pool(w_active: torch.Tensor, valid_active: torch.Tensor, M: int,
                rule: str = "full", top_k: int = 8) -> torch.Tensor:
    """Top-M pool [E] bool from the votes of the VALID active-block rows.

    `w_active` [n_active, E] are the fp32 router softmax rows of the active block.
    rule "full": votes = sum of the softmax (dMoE); "top8": sum of the renormalised
    top-k weights (DES-Vote's Algorithm-3 term, renormalised as the spec asks --
    `sdar_harness/desvote.py:96-99` votes with the UNnormalised top-k weights).
    Ties break on the lowest expert id (stable sort), as `desvote.py:133`.
    """
    assert rule in POOL_RULES, rule
    E = w_active.shape[1]
    if rule == "full":
        contrib = w_active
    else:
        v, i = torch.topk(w_active, top_k, dim=-1)
        v = v / v.sum(-1, keepdim=True).clamp_min(_EPS)
        contrib = torch.zeros_like(w_active).scatter_(1, i, v)
    votes = (contrib * valid_active.to(contrib.dtype).unsqueeze(1)).sum(0)          # [E]
    order = torch.sort(votes, descending=True, stable=True).indices
    pool = torch.zeros(E, dtype=torch.bool, device=w_active.device)
    pool[order[:min(M, E)]] = True
    return pool


def moe_forward(blk, hidden_states: torch.Tensor, n_active: int, valid_active: torch.Tensor,
                pool_M: Optional[int], pool_rule: str = "full"):
    """Differentiable hit-only MoE forward with an optional active-row expert pool.

    Returns (out [B,S,H], uniq_active [] int64, pool [E] bool).  With
    `pool_M is None` (or >= num_experts) the arithmetic is bit-identical to the
    stock block: the pool is applied by multiplying the fp32 softmax by a 0/1 mask
    (exact for 1.0) AFTER the softmax, so top-k and the renormalisation see the
    very same numbers the stock forward sees.
    """
    B, S, H = hidden_states.shape
    hs = hidden_states.view(-1, H)
    E, top_k = blk.num_experts, blk.top_k

    router_logits = blk.gate(hs)                                            # remote :287
    w = F.softmax(router_logits, dim=1, dtype=torch.float)                  # remote :289

    lo = hs.shape[0] - n_active
    pool = torch.zeros(E, dtype=torch.bool, device=hs.device)
    restrict = pool_M is not None and int(pool_M) < E
    if restrict:
        assert int(pool_M) >= top_k, f"pool M={pool_M} < top_k={top_k}"
        pool = expert_pool(w[lo:], valid_active, int(pool_M), pool_rule, top_k)
        keep = torch.ones_like(w)
        keep[lo:] = pool.to(w.dtype).unsqueeze(0)
        w_use = w * keep
    else:
        pool = torch.ones(E, dtype=torch.bool, device=hs.device)
        w_use = w

    routing_weights, selected = torch.topk(w_use, top_k, dim=-1)            # remote :290-291
    if blk.norm_topk_prob:                                                  # remote :292-293
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    routing_weights = routing_weights.to(hs.dtype)                          # remote :295

    final = torch.zeros((B * S, H), dtype=hs.dtype, device=hs.device)       # remote :297-299
    expert_mask = F.one_hot(selected, num_classes=E).permute(2, 1, 0)       # remote :303-304
    for expert_idx in torch.unique(selected).tolist():                      # fastmoe.py:44-45
        expert_layer = blk.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])                   # remote :309
        current_state = hs[None, top_x].reshape(-1, H)                      # remote :314
        cur = expert_layer(current_state) * routing_weights[top_x, idx, None]   # remote :315-316
        final.index_add_(0, top_x, cur.to(hs.dtype))                        # remote :320-321

    # unique experts of the ACTIVE rows, kept on device (no sync in the hot path)
    hit = torch.zeros(E, dtype=torch.bool, device=hs.device)
    hit.scatter_(0, selected[lo:].reshape(-1), True)
    return final.reshape(B, S, H), hit.sum(), pool


# ---------------------------------------------------------------------------
# decoder layer (remote :415-443)
# ---------------------------------------------------------------------------
def layer_forward(layer, hidden_states, attn_mask, position_embeddings, n_active,
                  valid_active, pool_M, pool_rule, num_experts):
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = attn_forward(layer.self_attn, hidden_states, position_embeddings, attn_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    mlp = layer.mlp
    if hasattr(mlp, "experts"):
        hidden_states, uniq, pool = moe_forward(mlp, hidden_states, n_active, valid_active,
                                                pool_M, pool_rule)
    else:                                                                   # dense mlp_only layer
        hidden_states = mlp(hidden_states)
        uniq = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
        pool = torch.zeros(num_experts, dtype=torch.bool, device=hidden_states.device)
    hidden_states = residual + hidden_states
    return hidden_states, uniq, pool


# ---------------------------------------------------------------------------
# the training forward
# ---------------------------------------------------------------------------
@dataclass
class ForwardOut:
    hidden: torch.Tensor                 # [32, H] the active block's rows, after the final norm
    unique_experts: torch.Tensor         # [L] int64, unique experts of the active rows per layer
    pools: Optional[torch.Tensor] = None  # [L, E] bool, only when return_pools


def training_forward(model, input_ids: torch.Tensor, block_lo: int, valid_slots: torch.Tensor,
                     pool_M: Optional[int] = None, pool_rule: str = "full",
                     checkpoint_layers: bool = True, return_pools: bool = False,
                     block: int = BLOCK, draft_flag: Optional[torch.Tensor] = None) -> ForwardOut:
    """One state -> the hidden states of the active block (mirror of remote :592-658).

    input_ids   [1, S], S a multiple of `block`; the active block is the LAST one.
    block_lo    absolute start of the active block (must be S - block).
    valid_slots [block] bool: the generated slots of the active block (the prompt
                tail of the first generated block is NOT valid and never votes).
    pool_M      None / >= num_experts -> no restriction (bit-identical to stock).
    draft_flag  [S] bool or None.  None = the pre-flag behaviour, bit-identical to
                round 1.  When given, `adapter.DraftEmbed` (if the model carries one)
                is added to the flagged positions' input embeddings and the LoRA delta
                is gated to those positions -- an all-False flag is then exactly the
                stock model (no delta, no embedding).

                The gate is armed INSIDE the per-layer closure, not around the whole
                forward: with `checkpoint_layers` the layer is re-run during backward,
                long after this function returned, and the recomputation must see the
                same context or the gradients would be those of an ungated delta.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, input_ids.shape
    S = int(input_ids.shape[1])
    assert S % block == 0, f"S={S} is not on the {block}-grid"
    assert block_lo == S - block, f"the active block must be the last one ({block_lo} != {S - block})"
    assert valid_slots.shape == (block,), valid_slots.shape
    assert pool_rule in POOL_RULES, pool_rule

    inner = model.model
    device = input_ids.device
    valid_active = valid_slots.to(device)
    num_experts = int(getattr(model.config, "num_experts", 0) or 0)

    flag = flag_any = None
    if draft_flag is not None:
        assert draft_flag.shape == (S,), f"draft_flag must be [S={S}], got {tuple(draft_flag.shape)}"
        flag_any = bool(draft_flag.any())                                   # on the caller's (cpu) tensor: no sync
        flag = draft_flag.to(device=device, dtype=torch.bool).view(1, S)

    inputs_embeds = inner.embed_tokens(input_ids)                           # remote :593
    if flag is not None:
        de = get_draft_embed(model)
        if de is not None:
            inputs_embeds = de.apply_to(inputs_embeds, flag)                # fp32 add, cast back
    position_ids = torch.arange(S, device=device).unsqueeze(0)              # remote :598-602
    position_embeddings = inner.rotary_emb(inputs_embeds, position_ids)     # remote :611
    attn_mask = None if S == block else block_tril_bool(S, device, block)   # remote :218-237

    hidden_states = inputs_embeds
    uniqs: List[torch.Tensor] = []
    pools: List[torch.Tensor] = []
    for layer in inner.layers:                                              # remote :618-650
        def run(h, _layer=layer):
            if flag is None:
                return layer_forward(_layer, h, attn_mask, position_embeddings, block,
                                     valid_active, pool_M, pool_rule, num_experts)
            with flag_scope(flag, flag_any):
                return layer_forward(_layer, h, attn_mask, position_embeddings, block,
                                     valid_active, pool_M, pool_rule, num_experts)
        if checkpoint_layers and torch.is_grad_enabled():
            hidden_states, uniq, pool = checkpoint(run, hidden_states, use_reentrant=False)
        else:
            hidden_states, uniq, pool = run(hidden_states)
        uniqs.append(uniq)
        if return_pools:
            pools.append(pool)

    hidden_states = inner.norm(hidden_states)                               # remote :658
    return ForwardOut(hidden=hidden_states[0, block_lo:],
                      unique_experts=torch.stack(uniqs),
                      pools=torch.stack(pools) if return_pools else None)


@torch.no_grad()
def stock_forward_reference(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                            logits_to_keep: int = 0, block: int = BLOCK):
    """A plain call of the REMOTE forward with the same block-tril mask (parity reference).

    Returns the model output; `logits_to_keep=32` makes the remote code slice the
    lm_head to the active block (:1029-1031) instead of materialising [S, 151936].
    """
    S = int(input_ids.shape[1])
    if attention_mask is None:
        attention_mask = block_tril_bool(S, input_ids.device, block)
    position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
    return model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                 use_cache=False, logits_to_keep=logits_to_keep)


@torch.no_grad()
def stock_hidden_reference(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                           block: int = BLOCK) -> torch.Tensor:
    """The remote `SDARMoeModel` forward's last hidden state [1, S, H] (no lm_head)."""
    S = int(input_ids.shape[1])
    if attention_mask is None:
        attention_mask = block_tril_bool(S, input_ids.device, block)
    position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
    out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=False)
    return out.last_hidden_state
