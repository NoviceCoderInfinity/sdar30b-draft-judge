"""Block-level expert coresets (DES-Vote style) for SDARMoeSparseMoeBlock.

At step 1 of a block the router weights are summed over the block's valid
positions, per layer; the top-M experts become that block's **coreset**, and
every later step of the block routes only within it.

Restriction and renormalisation: the full softmax is masked to the coreset before
the top-k, and the selected weights are then renormalised by `norm_topk_prob` as
upstream does. That is identical to taking the softmax over the coreset's logits
alone -- softmax over a subset equals the full softmax restricted and rescaled,
and the subsequent top-k renormalisation cancels the common factor -- so there is
no ambiguity in "renormalise routing over the coreset".

The dispatch loop is the bit-exact hit-experts-only variant (`fastmoe`): experts
with no assigned tokens do an empty matmul and an `index_add_` of an empty
tensor, both no-ops. With a coreset of M experts at most M are ever hit, so the
saving is real rather than an artefact of the loop.

Vote sources
  mask   : the block's own step-1 forward, 32 [MASK]. This is DES-Vote as
           published. It cannot restrict step 1 -- step 1 IS the vote.
  draft  : one extra forward with the drafter's tokens placed in the block
           (`store_kv=False`, so the decode's cache is untouched). The coreset is
           known BEFORE step 1, so every step including the first is restricted.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .routing import moe_blocks


class CoresetCtrl:
    """Module-global controller. One decode runs single-threaded per GPU."""

    def __init__(self):
        self.voting = False
        self.valid: Optional[torch.Tensor] = None    # bool [N] over flattened positions
        self.accum: Optional[torch.Tensor] = None    # [n_layers, n_experts] fp32
        self.mask: Optional[torch.Tensor] = None     # [n_layers, n_experts] bool
        self.n_layers = 0
        self.n_experts = 0
        # cost counters: unique experts summed over (forward, layer)
        self.unique_sum = 0
        self.layer_forwards = 0

    # -- voting -----------------------------------------------------------
    def start_vote(self, valid: Optional[torch.Tensor], device):
        self.accum = torch.zeros(self.n_layers, self.n_experts,
                                 dtype=torch.float32, device=device)
        self.valid = valid
        self.voting = True

    def finish_vote(self, M: int) -> torch.Tensor:
        self.voting = False
        idx = torch.topk(self.accum, M, dim=1).indices
        m = torch.zeros_like(self.accum, dtype=torch.bool)
        m.scatter_(1, idx, True)
        self.valid = None
        return m

    # -- restriction ------------------------------------------------------
    def set_mask(self, m: Optional[torch.Tensor]):
        self.mask = m

    def clear(self):
        self.mask = None
        self.voting = False
        self.valid = None

    # -- cost -------------------------------------------------------------
    def reset_cost(self):
        self.unique_sum = 0
        self.layer_forwards = 0

    def cost(self) -> dict:
        return {"unique_expert_layer_loads": self.unique_sum,
                "layer_forwards": self.layer_forwards,
                "mean_unique_per_layer": (self.unique_sum / self.layer_forwards
                                          if self.layer_forwards else 0.0)}


CTRL = CoresetCtrl()


def coreset_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    w = F.softmax(router_logits, dim=1, dtype=torch.float)
    L = self._layer_idx

    if CTRL.voting:
        v = CTRL.valid
        CTRL.accum[L] += (w if v is None else w[v]).sum(0)

    if CTRL.mask is not None:
        w = w.masked_fill(~CTRL.mask[L].unsqueeze(0), float("-inf"))

    routing_weights, selected_experts = torch.topk(w, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    hit = torch.unique(selected_experts).tolist()
    CTRL.unique_sum += len(hit)
    CTRL.layer_forwards += 1

    for expert_idx in hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def enable_coreset_moe(model) -> int:
    blocks = moe_blocks(model)
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk._layer_idx = i
        blk.forward = coreset_moe_forward.__get__(blk, type(blk))
    CTRL.n_layers = len(blocks)
    CTRL.n_experts = blocks[0].num_experts if blocks else 0
    return len(blocks)
