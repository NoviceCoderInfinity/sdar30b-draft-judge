"""Instrumented SDARMoeSparseMoeBlock forward: logs the indices the experts
actually consumed.

Restated §6 criterion (2026-09-03): *the logged expert set must be IDENTICAL to
the set the experts actually computed.* The earlier implementation re-derived the
top-8 from `router_logits` after the fact, which made the log a reconstruction
and put it at the mercy of numeric agreement (it disagreed with an fp32 recompute
~10% of the time on strict set identity). This module instead captures
`selected_experts` -- the very tensor the dispatch loop indexes -- so identity
holds by construction, not by numeric luck.

The forward is also the bit-exact hit-experts-only variant from `fastmoe.py`:
experts with no assigned tokens do an empty matmul and an `index_add_` of an
empty tensor, both no-ops, so skipping them cannot change the result.

`verify_dispatch=True` additionally reconstructs, from inside the loop, which
positions each expert actually received (`torch.where(expert_mask[e])`) and
compares that against the logged sets. That is an end-to-end check of the claim
-- it does not consult the router at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from .routing import moe_blocks

TOP_K = 8


# ---------------------------------------------------------------------------
# capture sink
# ---------------------------------------------------------------------------
@dataclass
class MoECapture:
    """One armed forward, all layers."""
    tag: str
    step: int
    ids: List[torch.Tensor] = field(default_factory=list)     # per layer [N, 8] uint8
    probs: List[torch.Tensor] = field(default_factory=list)   # per layer [N, 8] fp16
    dispatched: List[torch.Tensor] = field(default_factory=list)  # per layer [N, E] bool (verify only)

    def stack_ids(self) -> torch.Tensor:
        return torch.stack(self.ids)


class _Sink:
    """Module-global arm/disarm target. One decode runs single-threaded per GPU."""
    def __init__(self):
        self.active: Optional[MoECapture] = None
        self.verify = False

    def arm(self, cap: MoECapture):
        self.active = cap

    def disarm(self):
        self.active = None


SINK = _Sink()


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def instrumented_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights_full = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights_full, self.top_k, dim=-1)

    # log BEFORE renormalisation: these are the true router probabilities, and
    # the ids here are the exact tensor the dispatch below indexes.
    cap = SINK.active
    if cap is not None:
        cap.ids.append(selected_experts.detach().to(torch.uint8).cpu().clone())
        cap.probs.append(routing_weights.detach().half().cpu().clone())

    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    verifying = cap is not None and SINK.verify
    if verifying:
        dispatched = torch.zeros((batch_size * sequence_length, self.num_experts),
                                 dtype=torch.bool, device=hidden_states.device)

    for expert_idx in torch.unique(selected_experts).tolist():
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        if verifying:
            # what this expert ACTUALLY received, read off the dispatch itself
            dispatched[top_x, expert_idx] = True

    if verifying:
        cap.dispatched.append(dispatched.detach().cpu().clone())

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def enable_instrumented_moe(model) -> int:
    blocks = moe_blocks(model)
    for blk in blocks:
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk.forward = instrumented_moe_forward.__get__(blk, type(blk))
    return len(blocks)


def disable_instrumented_moe(model) -> int:
    blocks = moe_blocks(model)
    n = 0
    for blk in blocks:
        if hasattr(blk, "_stock_forward"):
            blk.forward = blk._stock_forward
            del blk._stock_forward
            n += 1
    return n


# ---------------------------------------------------------------------------
# validation: logged set == dispatched set
# ---------------------------------------------------------------------------
def check_capture_identity(cap: MoECapture) -> dict:
    """Every logged top-8 set must equal the set of experts that received the token."""
    assert cap.dispatched, "capture has no dispatch record; set SINK.verify = True"
    positions = 0
    mismatched = 0
    wrong_count = 0
    for ids, disp in zip(cap.ids, cap.dispatched):
        a = torch.zeros_like(disp)
        a.scatter_(1, ids.to(torch.int64), True)      # logged set as a bool matrix
        eq = (a == disp).all(dim=1)
        positions += eq.numel()
        mismatched += int((~eq).sum().item())
        wrong_count += int((disp.sum(dim=1) != ids.shape[1]).sum().item())
    return {"positions": positions, "mismatched": mismatched,
            "positions_not_receiving_exactly_top_k": wrong_count,
            "identity": 1.0 - mismatched / max(positions, 1)}


# ---------------------------------------------------------------------------
# S1: reported statistic, bf16 gate vs fp32 recompute
# ---------------------------------------------------------------------------
class S1Probe:
    """Forward hooks recording the bf16-vs-fp32 router comparison as a STATISTIC.

    This is no longer a gate (see §10, 2026-09-03). It reports:
      - strict 8-set agreement
      - top-7 set agreement  (the rank-8 slot is where the instability lives)
      - the distribution of the rank-8 minus rank-9 softmax PROBABILITY margin
    """

    def __init__(self, model, top_k: int = TOP_K):
        self.blocks = moe_blocks(model)
        self.top_k = top_k
        self._handles = []
        self._armed = False
        self.positions = 0
        self.set8_match = 0
        self.set7_match = 0
        self.margins: List[torch.Tensor] = []

    def _hook(self, module, inputs, output):
        if not self._armed:
            return
        router_logits = output[1]
        p_bf16 = F.softmax(router_logits, dim=1, dtype=torch.float)
        hs = inputs[0]
        hs = hs.reshape(-1, hs.shape[-1]).float()
        p_fp32 = F.softmax(hs @ module.gate.weight.float().T, dim=1, dtype=torch.float)

        a = torch.topk(p_bf16, self.top_k, dim=-1).indices
        b = torch.topk(p_fp32, self.top_k, dim=-1).indices
        sa, sb = torch.sort(a, -1).values, torch.sort(b, -1).values
        self.set8_match += int((sa == sb).all(-1).sum().item())

        a7 = torch.sort(a[:, :self.top_k - 1], -1).values
        b7 = torch.sort(b[:, :self.top_k - 1], -1).values
        self.set7_match += int((a7 == b7).all(-1).sum().item())
        self.positions += a.shape[0]

        top9 = torch.topk(p_bf16, self.top_k + 1, dim=-1).values
        self.margins.append((top9[:, self.top_k - 1] - top9[:, self.top_k]).detach().cpu())

    def __enter__(self):
        for blk in self.blocks:
            self._handles.append(blk.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def arm(self):
        self._armed = True

    def disarm(self):
        self._armed = False

    def report(self) -> dict:
        if not self.positions:
            return {"positions": 0}
        m = torch.cat(self.margins)
        q = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        return {
            "positions": self.positions,
            "set8_agreement": self.set8_match / self.positions,
            "set7_agreement": self.set7_match / self.positions,
            "margin_rank8_minus_rank9": {
                "mean": float(m.mean()),
                "quantiles": {f"p{int(x*100)}": float(v)
                              for x, v in zip(q.tolist(), torch.quantile(m, q).tolist())},
                "frac_below_1e-3": float((m < 1e-3).float().mean()),
                "frac_below_1e-4": float((m < 1e-4).float().mean()),
            },
        }
