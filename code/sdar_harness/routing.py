"""Router (top-8 expert) capture for SDAR-30B-A3B.

`SDARMoeSparseMoeBlock.forward` returns `(final_hidden_states, router_logits)`
with router_logits of shape [batch*seq, num_experts] (pre-softmax). The model
then does softmax(float32) -> topk(8). softmax is monotonic, so the top-8 SET is
identical whether taken over logits or over the softmax; we take it over the
logits and keep the softmax probabilities for the weights.

Expert ids are 0..127 so they fit in uint8, matching the log schema in
the router-numerics convention below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

TOP_K = 8


def moe_blocks(model) -> List[torch.nn.Module]:
    """Every SDARMoeSparseMoeBlock, in layer order."""
    out = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            out.append(mlp)
    return out


def router_top8(router_logits: torch.Tensor, top_k: int = TOP_K):
    """router_logits [N, E] -> (ids uint8 [N, top_k], probs fp32 [N, top_k]).

    Mirrors the model: softmax in float32, then topk, then (norm_topk_prob)
    renormalisation is applied by the caller if wanted. Ids are returned sorted
    by descending routing weight, as torch.topk gives them.
    """
    weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    probs, ids = torch.topk(weights, top_k, dim=-1)
    return ids.to(torch.uint8), probs


@dataclass
class Capture:
    """One armed forward's worth of routing, over all layers."""
    tag: str
    step: int
    ids: List[torch.Tensor] = field(default_factory=list)     # per layer [N, 8] uint8
    probs: List[torch.Tensor] = field(default_factory=list)   # per layer [N, 8] fp16
    logits: List[torch.Tensor] = field(default_factory=list)  # per layer [N, E] fp16 (opt)

    def stack_ids(self) -> torch.Tensor:
        return torch.stack(self.ids)          # [layers, N, 8]

    def stack_probs(self) -> torch.Tensor:
        return torch.stack(self.probs)        # [layers, N, 8]


class RouterRecorder:
    """Forward hooks on every MoE block; records only while armed.

    Usage:
        rec = RouterRecorder(model)
        with rec:
            rec.arm("denoise", step=0)
            model(...)                 # captured
            rec.disarm()
            model(...)                 # not captured
        rec.captures -> List[Capture]

    `check_fp32` additionally recomputes the gate in float32 from the block's
    input hidden states and records top-8 agreement -- the P1 numerics probe
    (see "Router numerics" above).
    """

    def __init__(self, model, keep_logits: bool = False, check_fp32: bool = False,
                 top_k: int = TOP_K):
        self.model = model
        self.blocks = moe_blocks(model)
        self.keep_logits = keep_logits
        self.check_fp32 = check_fp32
        self.top_k = top_k
        self._handles = []
        self._armed: Optional[Capture] = None
        self.captures: List[Capture] = []
        # fp32 agreement accounting
        self.fp32_positions = 0
        self.fp32_set_matches = 0          # identical top-8 SET
        self.fp32_slot_matches = 0         # identical id in identical rank slot
        self.fp32_slots = 0
        self.fp32_overlap_sum = 0          # sum of |A n B| over positions
        self.fp32_gap_sum = 0.0            # sum of (logit[7] - logit[8]) margins
        self.fp32_gap_n = 0

    # -- hook ------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self._armed is None:
                return
            router_logits = output[1]
            ids, probs = router_top8(router_logits, self.top_k)
            self._armed.ids.append(ids.detach().cpu().clone())
            self._armed.probs.append(probs.detach().half().cpu().clone())
            if self.keep_logits:
                self._armed.logits.append(router_logits.detach().half().cpu().clone())
            if self.check_fp32:
                hs = inputs[0]
                hs = hs.reshape(-1, hs.shape[-1]).float()
                w32 = module.gate.weight.float()
                ids32, _ = router_top8(hs @ w32.T, self.top_k)
                a = ids.to(torch.int64)
                b = ids32.to(torch.int64)
                self.fp32_slot_matches += int((a == b).sum().item())
                self.fp32_slots += a.numel()
                sa = torch.sort(a, dim=-1).values
                sb = torch.sort(b, dim=-1).values
                self.fp32_set_matches += int((sa == sb).all(dim=-1).sum().item())
                self.fp32_positions += a.shape[0]
                # |A n B| per position: both rows are sets of distinct ids
                eq = (sa.unsqueeze(-1) == sb.unsqueeze(-2))      # [N, 8, 8]
                self.fp32_overlap_sum += int(eq.any(dim=-1).sum().item())
                # decision margin: gap between the 8th and 9th router logit
                srt = torch.sort(router_logits.float(), dim=-1, descending=True).values
                self.fp32_gap_sum += float((srt[:, self.top_k - 1] - srt[:, self.top_k]).sum().item())
                self.fp32_gap_n += srt.shape[0]
        return hook

    # -- lifecycle -------------------------------------------------------
    def __enter__(self):
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._armed = None
        return False

    def arm(self, tag: str, step: int = -1) -> Capture:
        self._armed = Capture(tag=tag, step=step)
        self.captures.append(self._armed)
        return self._armed

    def disarm(self):
        self._armed = None

    # -- reporting -------------------------------------------------------
    def fp32_report(self) -> Dict[str, float]:
        if self.fp32_positions == 0:
            return {"positions": 0}
        return {
            "positions": self.fp32_positions,
            "set_agreement": self.fp32_set_matches / self.fp32_positions,
            "slot_agreement": self.fp32_slot_matches / self.fp32_slots,
            # the metric that actually bounds a recall measurement:
            "expert_overlap": self.fp32_overlap_sum / (self.fp32_positions * self.top_k),
            "mean_top8_margin": (self.fp32_gap_sum / self.fp32_gap_n) if self.fp32_gap_n else None,
        }
