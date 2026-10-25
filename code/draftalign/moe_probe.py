"""MoE probe pieces: block-state builders, router capture via forward hooks on each layer's
`mlp.gate`, per-slot logit metrics, and the coreset vote / coverage math.

Router capture (modeling_sdar_moe.py:287-295): router_logits = gate(h) [N, E] (bf16);
w_full = softmax(logits, fp32); top-8 -> (w, ids); w renormalised.  The hook reproduces
exactly that from the gate's output and accumulates ON DEVICE, per forward:
  full  [L, bl, E] fp32   the 128-way softmax               (dMoE-style votes)
  top8w [L, bl, E] fp32   renormalised top-8 weights, scattered (DES-Vote-style votes)
  ids   [L, bl, K] long   top-8 expert ids (rank order)
Nothing is written to disk except the int16 ids and the coverage summaries.

Vote pools (per layer): v_e = sum over VALID block positions of the vote row; top-M pool.
Coverage = fraction of (valid position, expert) picks of R_final that lie in the pool.
Votes are restricted to valid positions to match sdar_harness.desvote (CTRL.valid) -- the
prompt-tail positions of the first block hold the same tokens in every probe kind.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sdar_harness.routing import moe_blocks

MS = (16, 24, 32, 48)

# (kind, state mode, fill source); fill sources: draft = drafter s8, self = recorded pass-1 proposals, final
KINDS: List[Tuple[str, str, Optional[str]]] = [
    ("P0", "mask", None),
    ("P1", "full", "draft"), ("P2a", "half_a", "draft"), ("P2b", "half_b", "draft"),
    ("P3", "full", "self"), ("P4a", "half_a", "self"), ("P4b", "half_b", "self"),
    ("P5", "full", "final"), ("P6a", "half_a", "final"), ("P6b", "half_b", "final"),
]
UNIONS: Dict[str, Tuple[str, str]] = {"P2": ("P2a", "P2b"), "P4": ("P4a", "P4b"), "P6": ("P6a", "P6b")}
COVERAGE_KINDS = ["P0", "P1", "P2a", "P2b", "P2", "P3", "P4a", "P4b", "P4", "P5", "P6a", "P6b", "P6"]


# ---------------------------------------------------------------------------
# block states
# ---------------------------------------------------------------------------
def build_block_state(final: torch.Tensor, valid: torch.Tensor, fill: Optional[torch.Tensor],
                      mode: str, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block ids for one probe forward and the bool mask of slots holding [MASK].

    Non-valid positions (prompt tail) always hold `final` (== the prompt tokens).
      mask   : every valid slot [MASK]
      full   : every valid slot = fill
      half_a : even block positions [MASK], odd = fill   (parity of the block position, valid only)
      half_b : odd block positions [MASK], even = fill
    `masked` is read back from the ids, so a fill token that IS the mask id counts as masked.
    """
    bl = final.numel()
    ids = final.clone()
    parity = (torch.arange(bl, device=final.device) % 2 == 0)
    if mode == "mask":
        to_mask, to_fill = valid, torch.zeros_like(valid)
    elif mode == "full":
        to_mask, to_fill = torch.zeros_like(valid), valid
    elif mode == "half_a":
        to_mask, to_fill = valid & parity, valid & ~parity
    elif mode == "half_b":
        to_mask, to_fill = valid & ~parity, valid & parity
    else:
        raise ValueError(mode)
    if to_fill.any():
        assert fill is not None
        ids = torch.where(to_fill, fill, ids)
    ids = torch.where(to_mask, torch.full_like(ids, mask_id), ids)
    return ids, ids == mask_id


# ---------------------------------------------------------------------------
# router capture
# ---------------------------------------------------------------------------
@dataclass
class Capture:
    full: torch.Tensor      # [L, bl, E] fp32
    top8w: torch.Tensor     # [L, bl, E] fp32
    ids: torch.Tensor       # [L, bl, K] long
    filled: torch.Tensor    # [L] bool: layer hook fired

    def votes(self, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(full-softmax votes, renormalised top-K votes) summed over valid positions -> [L, E]."""
        return self.full[:, valid].sum(1), self.top8w[:, valid].sum(1)

    def expert_union(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """bool [L, E]: experts in the top-K of any of `positions` (all block positions if None)."""
        ids = self.ids if positions is None else self.ids[:, positions]
        L, E = self.full.shape[0], self.full.shape[2]
        u = torch.zeros(L, E, dtype=torch.bool, device=ids.device)
        if ids.numel():
            u.scatter_(1, ids.reshape(L, -1), True)
        return u

    def unique_per_layer(self) -> torch.Tensor:
        """Unique experts hit by the forward per layer (all block positions) -> [L] long."""
        return self.expert_union().sum(1)


class RouterCapture:
    """Forward hooks on every MoE layer's `mlp.gate`; records only between arm() and disarm()."""

    def __init__(self, model, block_length: int, n_experts: Optional[int] = None, top_k: Optional[int] = None):
        self.blocks = moe_blocks(model)
        self.L = len(self.blocks)
        cfg = getattr(model, "config", None)
        self.E = int(n_experts or getattr(cfg, "num_experts"))
        self.K = int(top_k or getattr(cfg, "num_experts_per_tok"))
        self.bl = int(block_length)
        self.dev = model.device
        self.cur: Optional[Capture] = None
        self._handles = [blk.gate.register_forward_hook(self._make_hook(i)) for i, blk in enumerate(self.blocks)]

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            cap = self.cur
            if cap is None:
                return
            logits = output.reshape(-1, output.shape[-1])
            assert logits.shape == (self.bl, self.E), f"gate output {tuple(logits.shape)} != ({self.bl}, {self.E})"
            p = F.softmax(logits, dim=-1, dtype=torch.float)            # as the model: softmax in fp32
            w, ids = torch.topk(p, self.K, dim=-1)
            w = w / w.sum(dim=-1, keepdim=True)                         # norm_topk_prob
            cap.full[layer] = p
            cap.top8w[layer].zero_().scatter_(1, ids, w)
            cap.ids[layer] = ids
            cap.filled[layer] = True
        return hook

    def arm(self) -> Capture:
        self.cur = Capture(full=torch.zeros(self.L, self.bl, self.E, dtype=torch.float32, device=self.dev),
                           top8w=torch.zeros(self.L, self.bl, self.E, dtype=torch.float32, device=self.dev),
                           ids=torch.zeros(self.L, self.bl, self.K, dtype=torch.long, device=self.dev),
                           filled=torch.zeros(self.L, dtype=torch.bool, device=self.dev))
        return self.cur

    def disarm(self) -> Capture:
        cap, self.cur = self.cur, None
        assert cap is not None and bool(cap.filled.all()), "a layer's gate hook did not fire"
        return cap

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.cur = None


# ---------------------------------------------------------------------------
# per-slot logit metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def slot_metrics(logits: torch.Tensor, final: torch.Tensor, draft: torch.Tensor, fill: torch.Tensor,
                 mask_id: int, valid: torch.Tensor) -> dict:
    """logits [1, bl, V] (bf16 on GPU). Returns device tensors:
      argmax_bf16 [bl]  argmax of the bf16 softmax, decode.py:78-81's greedy path (integrity check)
      argmax [bl]       fp32 argmax with the [MASK] logit suppressed
      top1 / p_final / p_draft / p_fill [bl] fp32 probabilities (MASK suppressed)
      raw_mask_argmax   count of valid slots whose unsuppressed fp32 argmax was [MASK]
    """
    lg16 = logits[0]
    argmax_bf16 = F.softmax(lg16, dim=-1).argmax(dim=-1)
    lg = lg16.to(torch.float32, copy=True)              # copy: never mutate the model output
    raw_mask = (lg.argmax(-1) == mask_id)[valid].sum()
    lg[:, mask_id] = float("-inf")
    p = F.softmax(lg, dim=-1)
    top1, argmax = p.max(dim=-1)
    g = lambda t: p.gather(1, t.view(-1, 1)).view(-1)
    return {"argmax_bf16": argmax_bf16, "argmax": argmax, "top1": top1, "p_final": g(final),
            "p_draft": g(draft), "p_fill": g(fill), "raw_mask_argmax": raw_mask}


# ---------------------------------------------------------------------------
# votes / coverage
# ---------------------------------------------------------------------------
def top_m_pool(votes: torch.Tensor, M: int) -> torch.Tensor:
    """votes [L, E] -> bool [L, E] with the top-M experts per layer (torch.topk tie order, as desvote)."""
    L, E = votes.shape
    idx = torch.topk(votes, min(M, E), dim=-1).indices
    return torch.zeros(L, E, dtype=torch.bool, device=votes.device).scatter_(1, idx, True)


def pool_coverage(pool: torch.Tensor, r_final: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pool bool [L, E]; r_final [L, bl, K] -> per-layer fraction of valid (position, expert) picks in the pool."""
    L = pool.shape[0]
    picks = r_final[:, valid].reshape(L, -1)
    if picks.shape[1] == 0:
        return torch.full((L,), float("nan"), device=pool.device)
    return pool.gather(1, picks).float().mean(1)


def unique_recall(u_kind: torch.Tensor, u_final: torch.Tensor) -> torch.Tensor:
    """|U_kind & U_final| / |U_final| per layer."""
    return (u_kind & u_final).sum(1).float() / u_final.sum(1).clamp(min=1).float()


@torch.no_grad()
def coverage_summary(votes_full: torch.Tensor, votes_top8: torch.Tensor, u_kind: torch.Tensor,
                     r_final: torch.Tensor, u_final: torch.Tensor, valid: torch.Tensor,
                     Ms: Sequence[int] = MS, u_all: Optional[torch.Tensor] = None) -> dict:
    """{"cov": {"top8": {M: [L]}, "full": {M: [L]}}, "recall": [L], "uniq": [L]} as device tensors."""
    out = {"cov": {"top8": {}, "full": {}}, "recall": unique_recall(u_kind, u_final)}
    if u_all is not None:
        out["uniq"] = u_all.sum(1)                       # experts loaded by the forward(s), all block positions
    for M in Ms:
        out["cov"]["top8"][int(M)] = pool_coverage(top_m_pool(votes_top8, M), r_final, valid)
        out["cov"]["full"][int(M)] = pool_coverage(top_m_pool(votes_full, M), r_final, valid)
    return out


@torch.no_grad()
def coverage_all_kinds(caps: Dict[str, Capture], valid: torch.Tensor, Ms: Sequence[int] = MS,
                       final_kind: str = "P5") -> Dict[str, dict]:
    """Coverage for every captured kind and the halves unions, against R_final = caps[final_kind]."""
    r_final = caps[final_kind].ids
    u_final = caps[final_kind].expert_union(valid)
    votes = {k: c.votes(valid) for k, c in caps.items()}
    unions = {k: c.expert_union(valid) for k, c in caps.items()}
    alls = {k: c.expert_union() for k, c in caps.items()}
    out = {}
    for k in caps:
        out[k] = coverage_summary(votes[k][0], votes[k][1], unions[k], r_final, u_final, valid, Ms, alls[k])
    for name, (a, b) in UNIONS.items():
        if a in caps and b in caps:
            out[name] = coverage_summary(votes[a][0] + votes[b][0], votes[a][1] + votes[b][1],
                                         unions[a] | unions[b], r_final, u_final, valid, Ms, alls[a] | alls[b])
    return out


def summary_to_lists(s: dict, nd: int = 5) -> dict:
    r = lambda t: [round(float(x), nd) for x in t.tolist()]
    out = {"recall": r(s["recall"]),
           "cov": {rule: {str(M): r(v) for M, v in d.items()} for rule, d in s["cov"].items()}}
    if "uniq" in s:
        out["uniq"] = [int(x) for x in s["uniq"].tolist()]
    return out
