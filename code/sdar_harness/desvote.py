"""Per-forward DES-Vote (arXiv 2602.00879, Algorithm 1 + Algorithm 3) for
SDARMoeSparseMoeBlock, with an optional draft-informed vote at the positions
that are still [MASK].

Every MoE forward, every layer, in this order:

  1. router logits -> full softmax w (float32), exactly as the stock model.
  2. VOTE. Each valid position contributes its own top-K router weights and zeros
     elsewhere (DES's "saliency-aware" vote: weights outside the token's local
     top-K are masked out). In blend mode a position that holds [MASK] in THIS
     forward contributes instead the weights RECORDED for that slot in a readout
     forward where the drafter's token sat in it. Sum over positions -> [E];
     top-M -> the coreset C for this layer, this forward.
  3. ROUTE. sigma(I_n | i in C): softmax over the coreset's logits only, top-K,
     then the model's own top-K renormalisation. When C is every expert this is
     bit-identical to the stock path (verified by validate_desvote.py).
  4. DISPATCH. The hit-experts-only loop from fastmoe.py, bit-exact vs stock.

Nothing is carried between forwards except the recorded draft votes, which are
per-slot constants until re-recorded. This is the difference from coreset.py,
which elected once per block and froze the result.

Readout forward: set CTRL.recording = True, run the MoE once on the block with
the draft tokens in place (store_kv=False so the decode's cache is untouched),
then CTRL.take_recording() -> [L, N, E] per-slot votes. That forward is never
restricted and its output is discarded; its expert loads are counted separately.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .routing import moe_blocks


class VoteCtrl:
    """Module-global controller. One decode runs single-threaded per GPU."""

    def __init__(self):
        self.M: Optional[int] = None            # None -> unrestricted forward
        self.force_restrict = False             # run the restricted path even at M >= E (bit-exact test)
        self.valid: Optional[torch.Tensor] = None   # bool [N] over the block's flattened positions
        self.masked: Optional[torch.Tensor] = None  # bool [N]: positions holding [MASK] in this forward
        self.recorded: Optional[torch.Tensor] = None  # [L, N, E] fp32 per-slot votes from the readout
        self.use_recorded = False               # blend mode
        self.recording = False                  # this forward is a readout (record, do not restrict)
        self._rec_buf: Optional[torch.Tensor] = None
        self.n_layers = 0
        self.n_experts = 0
        # expert-set logging for the prefetch analysis: when a list, every layer
        # forward appends (layer, coreset bool[E] or None, hit-expert ids uint8)
        self.log: Optional[list] = None
        self.reset_cost()

    # ---- readout ---------------------------------------------------------
    def take_recording(self) -> torch.Tensor:
        self.recording = False
        r, self._rec_buf = self._rec_buf, None
        return r

    # ---- cost / behaviour counters ----------------------------------------
    def reset_cost(self):
        self.unique_sum = 0; self.layer_forwards = 0           # decode forwards (prefill + denoise + flush)
        self.readout_unique_sum = 0; self.readout_layer_forwards = 0
        self.restricted_layer_forwards = 0
        # statistics accumulate ON DEVICE (no per-layer host syncs); read in cost()
        self.moved_sum = None; self.moved_n = None             # share of a position's ref top-K moved by restriction
        self.zero_overlap = None; self.zero_overlap_n = None   # ref top-K has no expert inside the coreset
        self.ahead_sum = None; self.ahead_n = 0                # share of valid positions voting from the recording

    def _acc(self, name, value):
        cur = getattr(self, name)
        setattr(self, name, value if cur is None else cur + value)

    def cost(self) -> dict:
        f = lambda t: float(t.item()) if t is not None else 0.0
        mn, zn = f(self.moved_n), f(self.zero_overlap_n)
        return {
            "unique_expert_layer_loads": self.unique_sum,
            "layer_forwards": self.layer_forwards,
            "readout_unique_expert_layer_loads": self.readout_unique_sum,
            "readout_layer_forwards": self.readout_layer_forwards,
            "restricted_layer_forwards": self.restricted_layer_forwards,
            "routing_override_rate": (f(self.moved_sum) / mn) if mn else 0.0,
            "zero_overlap_rate": (f(self.zero_overlap) / zn) if zn else 0.0,
            "vote_share_known_ahead": (f(self.ahead_sum) / self.ahead_n) if self.ahead_n else 0.0,
        }


CTRL = VoteCtrl()


def _topk_masked(w_full: torch.Tensor, k: int) -> torch.Tensor:
    """Algorithm 3's per-token term: the row's top-k weights, zeros elsewhere. [N, E]."""
    vals, ids = torch.topk(w_full, k, dim=-1)
    return torch.zeros_like(w_full).scatter_(1, ids, vals)


def vote_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    w_full = F.softmax(router_logits, dim=1, dtype=torch.float)
    ref_w, ref_ids = torch.topk(w_full, self.top_k, dim=-1)
    L = self._layer_idx

    if CTRL.recording:
        pv = _topk_masked(w_full, self.top_k)
        if CTRL._rec_buf is None:
            CTRL._rec_buf = torch.zeros(CTRL.n_layers, *pv.shape, dtype=torch.float32,
                                        device=pv.device)
        CTRL._rec_buf[L] = pv

    restrict = (not CTRL.recording and CTRL.M is not None
                and (CTRL.M < self.num_experts or CTRL.force_restrict))
    if restrict:
        M = min(CTRL.M, self.num_experts)
        pv = _topk_masked(w_full, self.top_k)                  # every position votes for itself
        if CTRL.use_recorded and CTRL.recorded is not None and CTRL.masked is not None:
            pv = torch.where(CTRL.masked.unsqueeze(1), CTRL.recorded[L], pv)
            sel = CTRL.masked if CTRL.valid is None else (CTRL.masked & CTRL.valid)
            den = CTRL.valid.sum() if CTRL.valid is not None else torch.tensor(sel.numel(), device=sel.device)
            CTRL._acc("ahead_sum", sel.sum().float() / den.clamp(min=1).float())
        else:
            CTRL._acc("ahead_sum", torch.zeros((), device=w_full.device))
        CTRL.ahead_n += 1
        votes = (pv[CTRL.valid] if CTRL.valid is not None else pv).sum(0)      # [E]
        # top-M with a DETERMINISTIC tie-break (lowest expert id first). Ties occur
        # when fewer than M experts received any vote (tiny ragged block 0).
        order = torch.sort(votes, descending=True, stable=True).indices
        core = torch.zeros(self.num_experts, dtype=torch.bool, device=w_full.device)
        core[order[:M]] = True
        restricted = router_logits.float().masked_fill(~core.unsqueeze(0), float("-inf"))
        w = F.softmax(restricted, dim=1)
        routing_weights, selected_experts = torch.topk(w, self.top_k, dim=-1)
        # behaviour statistics, over valid positions, accumulated on device
        a = torch.zeros_like(w_full, dtype=torch.bool).scatter_(1, ref_ids, True)
        b = torch.zeros_like(w_full, dtype=torch.bool).scatter_(1, selected_experts, True)
        moved = (a & ~b).sum(1).float() / self.top_k
        zero = ((a & core.unsqueeze(0)).sum(1) == 0)
        if CTRL.valid is not None:
            moved, zero = moved[CTRL.valid], zero[CTRL.valid]
        CTRL._acc("moved_sum", moved.sum()); CTRL._acc("moved_n", torch.tensor(float(moved.numel()), device=moved.device))
        CTRL._acc("zero_overlap", zero.sum().float()); CTRL._acc("zero_overlap_n", torch.tensor(float(zero.numel()), device=zero.device))
        CTRL.restricted_layer_forwards += 1
    else:
        routing_weights, selected_experts = ref_w, ref_ids

    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    hit = torch.unique(selected_experts).tolist()
    if CTRL.recording:
        CTRL.readout_unique_sum += len(hit); CTRL.readout_layer_forwards += 1
    else:
        CTRL.unique_sum += len(hit); CTRL.layer_forwards += 1
    if CTRL.log is not None:
        CTRL.log.append((L, core.cpu().numpy() if restrict else None,
                         np.asarray(hit, dtype=np.uint8)))

    for expert_idx in hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def enable_vote_moe(model) -> int:
    blocks = moe_blocks(model)
    for i, blk in enumerate(blocks):
        # the restricted path (softmax over the coreset, top-K, then the model's
        # top-K renormalisation) equals the paper's sigma over the coreset only
        # when the model renormalises its top-K weights
        assert getattr(blk, "norm_topk_prob", False), "vote path assumes norm_topk_prob=True"
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk._layer_idx = i
        blk.forward = vote_moe_forward.__get__(blk, type(blk))
    CTRL.n_layers = len(blocks)
    CTRL.n_experts = blocks[0].num_experts if blocks else 0
    return len(blocks)


def disable_vote_moe(model) -> int:
    n = 0
    for blk in moe_blocks(model):
        if hasattr(blk, "_stock_forward"):
            blk.forward = blk._stock_forward
            del blk._stock_forward
            n += 1
    return n
