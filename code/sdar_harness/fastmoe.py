"""Bit-exact speedup for SDARMoeSparseMoeBlock.

The stock forward loops over ALL `num_experts` (128) every layer, every forward:

    for expert_idx in range(self.num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        ...
        final_hidden_states.index_add_(0, top_x, current_hidden_states...)

With B*S positions and top_k=8, at most min(128, B*S*8) experts can be non-empty
-- at block_length=4 that is <=32. For every other expert `top_x` is empty, so
`expert_layer(empty)` produces an empty tensor and `index_add_` is a no-op.
Skipping them is therefore BIT-EXACT, not an approximation; it only removes
launches (48 layers x 128 experts = 6144 per forward -> a few hundred).

`assert_bitexact()` verifies this on the live model rather than taking it on trust.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .routing import moe_blocks


def fast_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # ---- the only change: visit hit experts instead of all 128 ----
    hit = torch.unique(selected_experts).tolist()
    for expert_idx in hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def enable_fast_moe(model) -> int:
    """Patch every MoE block. Returns how many were patched."""
    blocks = moe_blocks(model)
    for blk in blocks:
        blk._stock_forward = blk.forward
        blk.forward = fast_moe_forward.__get__(blk, type(blk))
    return len(blocks)


def disable_fast_moe(model) -> int:
    blocks = moe_blocks(model)
    n = 0
    for blk in blocks:
        if hasattr(blk, "_stock_forward"):
            blk.forward = blk._stock_forward
            del blk._stock_forward
            n += 1
    return n


@torch.no_grad()
def assert_bitexact(model, n_positions: int = 32, seed: int = 0):
    """Run every MoE block both ways on the same input; require bit-identical output."""
    blocks = moe_blocks(model)
    g = torch.Generator(device="cpu").manual_seed(seed)
    H = model.config.hidden_size
    worst_out, worst_logits = 0.0, 0.0
    for blk in blocks:
        x = torch.randn(1, n_positions, H, generator=g).to(
            device=next(blk.parameters()).device, dtype=next(blk.parameters()).dtype)
        stock_out, stock_logits = type(blk).forward(blk, x)
        fast_out, fast_logits = fast_moe_forward(blk, x)
        worst_out = max(worst_out, (stock_out - fast_out).abs().max().item())
        worst_logits = max(worst_logits, (stock_logits - fast_logits).abs().max().item())
    return {"blocks": len(blocks), "max_abs_diff_output": worst_out,
            "max_abs_diff_router_logits": worst_logits,
            "bit_exact": worst_out == 0.0 and worst_logits == 0.0}
