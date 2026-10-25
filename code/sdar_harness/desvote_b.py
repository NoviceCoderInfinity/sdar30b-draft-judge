"""Drafter-informed DES-Vote: the patched SDARMoeSparseMoeBlock forward.

Implementation B of docs/DESVOTE_SPEC.md section 2. Written independently of
`desvote.py`; the spec is the only shared authority.

Per forward, per MoE layer (DES-Vote, arXiv 2602.00879, Alg. 1 + Alg. 3):

  1. router_logits = gate(h); w_full = softmax(router_logits, fp32)   -- stock
  2. ref_w, ref_ids = topk(w_full, top_k)                              -- unrestricted choice
  3. recording forward  : store topk_masked(w_full) per layer, route with ref
  4. restricted forward : votes = sum over VALID positions of
                            recorded[layer][pos]  if pos is [MASK] and use_recorded
                            topk_masked(w_full)[pos] otherwise
                          core = topk(votes, M); softmax over core logits; topk
  5. otherwise          : route with ref
  6. norm_topk_prob, cast, dispatch with fastmoe's hit-experts-only loop

The dispatch section is a verbatim copy of `fastmoe.fast_moe_forward` (same
ops, same order) so that with `CTRL.M is None` the block is bit-identical to it.

Nothing is carried between forwards except `CTRL.recorded` (set by the caller
from `take_recording()`) and the counters. `CTRL._rec_buf` lives only for the
duration of one recording forward.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .routing import moe_blocks


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------
class _VoteCtrl:
    """Module-global state read by every patched MoE block. One decode per process."""

    def __init__(self):
        # --- routing control (set by the decode loop before every forward) ---
        self.M: Optional[int] = None            # coreset size; None = unrestricted
        self.force_restrict: bool = False       # restrict even when M >= E (G2 gate)
        self.valid: Optional[torch.Tensor] = None    # bool [N]: genuinely generated slots
        self.masked: Optional[torch.Tensor] = None   # bool [N]: slots that are [MASK] now
        self.recorded: Optional[torch.Tensor] = None # float32 [L, N, E] from the readout
        self.use_recorded: bool = False
        self.recording: bool = False
        # --- model geometry (filled by enable_vote_moe) ---
        self.n_layers: int = 48
        self.n_experts: int = 128
        # --- recording scratch (one forward only) ---
        self._rec_buf: Optional[torch.Tensor] = None
        self._rec_layers_seen: int = 0
        self.reset_cost()

    # -- recording -----------------------------------------------------------
    def take_recording(self) -> torch.Tensor:
        """Return the [L, N, E] recording of the forward just taken; clears `recording`."""
        buf = self._rec_buf
        seen = self._rec_layers_seen
        self._rec_buf = None
        self._rec_layers_seen = 0
        self.recording = False
        if buf is None:
            raise RuntimeError("take_recording() called but no recording forward was taken")
        if seen != self.n_layers:
            raise RuntimeError(f"recording covered {seen} layers, expected {self.n_layers}")
        return buf

    # -- counters ------------------------------------------------------------
    def reset_cost(self):
        self.unique_sum = 0                 # decode (non-recording) forwards
        self.layer_forwards = 0
        self.readout_unique_sum = 0         # recording forwards
        self.readout_layer_forwards = 0
        self.restricted_layer_forwards = 0
        self.moved_sum = 0.0                # sum over valid positions of moved fraction
        self.moved_n = 0
        self.zero_overlap = 0               # valid positions whose ref top-K misses the core
        self.zero_overlap_n = 0
        self.ahead_sum = 0                  # valid positions that voted from the recording
        self.ahead_n = 0

    def cost(self) -> dict:
        def rate(a, b):
            return (a / b) if b else None
        return {
            "unique_expert_layer_loads": int(self.unique_sum),
            "layer_forwards": int(self.layer_forwards),
            "readout_unique_expert_layer_loads": int(self.readout_unique_sum),
            "readout_layer_forwards": int(self.readout_layer_forwards),
            "restricted_layer_forwards": int(self.restricted_layer_forwards),
            "routing_override_rate": rate(self.moved_sum, self.moved_n),
            "zero_overlap_rate": rate(self.zero_overlap, self.zero_overlap_n),
            "vote_share_known_ahead": rate(self.ahead_sum, self.ahead_n),
            # raw tallies, so rates can be re-pooled across items
            "moved_sum": float(self.moved_sum), "moved_n": int(self.moved_n),
            "zero_overlap_sum": int(self.zero_overlap), "zero_overlap_n": int(self.zero_overlap_n),
            "ahead_sum": int(self.ahead_sum), "ahead_n": int(self.ahead_n),
        }


CTRL = _VoteCtrl()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _topk_masked(w_full: torch.Tensor, ref_w: torch.Tensor, ref_ids: torch.Tensor) -> torch.Tensor:
    """DES `Masked(.)`: each row's top-K weights at their expert ids, zeros elsewhere. [N, E] fp32."""
    return torch.zeros_like(w_full).scatter_(1, ref_ids, ref_w)


def _check_vec(t: Optional[torch.Tensor], n: int, name: str) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dtype != torch.bool or t.dim() != 1 or t.shape[0] != n:
        raise RuntimeError(f"CTRL.{name} must be bool [{n}], got {tuple(t.shape)} {t.dtype}")
    return t


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def vote_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    # 1-2. stock routing distribution and the unrestricted top-K (fastmoe-identical)
    w_full = F.softmax(router_logits, dim=1, dtype=torch.float)
    ref_w, ref_ids = torch.topk(w_full, self.top_k, dim=-1)

    N, E = w_full.shape
    layer = self._desvote_layer_idx
    M = CTRL.M

    if CTRL.recording:
        # 3. readout forward: remember every slot's top-K-masked weights; do NOT restrict
        if CTRL._rec_buf is None:
            CTRL._rec_buf = torch.zeros(CTRL.n_layers, N, E, dtype=torch.float32,
                                        device=w_full.device)
            CTRL._rec_layers_seen = 0
        if CTRL._rec_buf.shape[1] != N:
            raise RuntimeError(f"recording N changed within a forward: "
                               f"{CTRL._rec_buf.shape[1]} vs {N}")
        CTRL._rec_buf[layer] = _topk_masked(w_full, ref_w, ref_ids)
        CTRL._rec_layers_seen += 1
        routing_weights, selected_experts = ref_w, ref_ids
        restricted = False

    elif M is not None and (M < E or CTRL.force_restrict):
        # 4. dynamic per-forward, per-layer election (DES-Vote) with drafter-informed votes
        own = _topk_masked(w_full, ref_w, ref_ids)                       # [N, E]
        valid = _check_vec(CTRL.valid, N, "valid")
        masked = _check_vec(CTRL.masked, N, "masked")
        from_rec = None
        if CTRL.use_recorded and CTRL.recorded is not None and masked is not None:
            rec = CTRL.recorded[layer]
            if rec.shape != (N, E):
                raise RuntimeError(f"CTRL.recorded[{layer}] is {tuple(rec.shape)}, need {(N, E)}")
            votes_per_pos = torch.where(masked[:, None], rec, own)
            from_rec = masked
        else:
            votes_per_pos = own
        votes = votes_per_pos[valid].sum(0) if valid is not None else votes_per_pos.sum(0)  # [E]
        core_idx = torch.topk(votes, min(M, E)).indices
        core = torch.zeros(E, dtype=torch.bool, device=w_full.device)
        core[core_idx] = True
        # sigma(I_n | i in C): softmax over the coreset's logits only
        restricted_logits = router_logits.float().masked_fill(~core[None], float("-inf"))
        w = F.softmax(restricted_logits, dim=1)
        routing_weights, selected_experts = torch.topk(w, self.top_k, dim=-1)
        restricted = True

        # statistics over valid positions
        in_new = (ref_ids[:, :, None] == selected_experts[:, None, :]).any(-1)   # [N, K]
        moved = 1.0 - in_new.float().mean(-1)                                    # [N]
        zero_ov = ~core[ref_ids].any(-1)                                         # [N]
        if valid is not None:
            moved_v, zero_v = moved[valid], zero_ov[valid]
            n_valid = valid.sum()
            ahead = (from_rec & valid).sum() if from_rec is not None else torch.zeros((), device=w_full.device)
        else:
            moved_v, zero_v = moved, zero_ov
            n_valid = torch.tensor(N, device=w_full.device)
            ahead = from_rec.sum() if from_rec is not None else torch.zeros((), device=w_full.device)
        stats = torch.stack([moved_v.sum().float(), zero_v.sum().float(),
                             ahead.float(), n_valid.float()]).tolist()
        CTRL.moved_sum += stats[0]
        CTRL.zero_overlap += int(stats[1])
        CTRL.ahead_sum += int(stats[2])
        CTRL.moved_n += int(stats[3])
        CTRL.zero_overlap_n += int(stats[3])
        CTRL.ahead_n += int(stats[3])

    else:
        # 5. unrestricted
        routing_weights, selected_experts = ref_w, ref_ids
        restricted = False

    # 6. renormalise, cast, dispatch -- fastmoe.fast_moe_forward, same ops, same order
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)

    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    hit = torch.unique(selected_experts).tolist()
    for expert_idx in hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # counters
    if CTRL.recording:
        CTRL.readout_unique_sum += len(hit)
        CTRL.readout_layer_forwards += 1
    else:
        CTRL.unique_sum += len(hit)
        CTRL.layer_forwards += 1
        if restricted:
            CTRL.restricted_layer_forwards += 1

    return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------
def enable_vote_moe(model) -> int:
    """Patch every MoE block; records its layer index. Returns how many were patched."""
    blocks = moe_blocks(model)
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk._desvote_layer_idx = i
        blk.forward = vote_moe_forward.__get__(blk, type(blk))
    if blocks:
        CTRL.n_layers = len(blocks)
        CTRL.n_experts = int(blocks[0].num_experts)
    return len(blocks)


def disable_vote_moe(model) -> int:
    blocks = moe_blocks(model)
    n = 0
    for blk in blocks:
        if hasattr(blk, "_stock_forward"):
            blk.forward = blk._stock_forward
            del blk._stock_forward
            n += 1
        if hasattr(blk, "_desvote_layer_idx"):
            del blk._desvote_layer_idx
    return n
