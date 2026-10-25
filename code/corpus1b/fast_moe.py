"""Grouped-GEMM forward for SDAR-MoE sparse blocks, for TRAINING only (train_c1b.py --fast-moe).

The remote `SDARMoeSparseMoeBlock.forward` (modeling_sdar_moe.py) loops over its 128 experts in Python: per expert a
torch.where, a gather, three small matmuls and an index_add_, in every one of 48 layers, forward, layer-checkpoint
recompute and backward. At the trainer's 4-16 states per GPU per step that loop is launch-bound (GPUs ~50% busy).

`patch_moe(model)` swaps in the same math as three grouped matmuls per block:
  * routing is untouched: softmax (fp32) over the router logits, top-k, optional renormalisation, cast back;
  * the (token, expert) assignments are sorted by expert; the expert MLP down(silu(gate(x)) * up(x)) runs as
    torch._grouped_mm over the sorted rows when that kernel works on the device (probed once on the real device
    against a reference), else as torch.bmm over a zero-padded [experts, capacity, hidden] batch, in token chunks
    that bound the padding;
  * outputs are scaled by their routing weight and summed back with one index_add_.
Expert weights are stacked IN PLACE: one [E, out, in] tensor per projection per block, and every expert's
nn.Linear.weight becomes a view of its slice (the per-expert tensors are freed; nothing is duplicated). Experts and
router stay frozen (requires_grad False); gradients still flow through them to the inputs.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn.functional as F

CHUNK_TOKENS = 16384          # bmm path: tokens per chunk (bounds the padded [E, C, H] buffers)


def _stack_in_place(experts, attr: str) -> torch.Tensor:
    """[E, out, in] tensor holding every expert's `attr`.weight; each Linear.weight becomes a view of it."""
    ws = [getattr(e, attr).weight for e in experts]
    stack = torch.empty((len(ws),) + tuple(ws[0].shape), dtype=ws[0].dtype, device=ws[0].device)
    for i, e in enumerate(experts):
        lin = getattr(e, attr)
        stack[i].copy_(lin.weight.data)
        lin.weight = torch.nn.Parameter(stack[i], requires_grad=lin.weight.requires_grad)   # a view: no copy
    return stack


def _expert_mlp_grouped(blk, xs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    offs = torch.cumsum(counts, 0).to(torch.int32)
    g = torch._grouped_mm(xs, blk._w_gate.transpose(1, 2), offs=offs)
    u = torch._grouped_mm(xs, blk._w_up.transpose(1, 2), offs=offs)
    return torch._grouped_mm(blk._act(g) * u, blk._w_down.transpose(1, 2), offs=offs)


def _expert_mlp_bmm(blk, xs: torch.Tensor, sorted_sel: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    E, H = blk.num_experts, xs.shape[-1]
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(xs.shape[0], device=xs.device) - starts[sorted_sel]
    cap = int(counts.max().item()) if xs.shape[0] else 0
    xp = xs.new_zeros((E, max(cap, 1), H))
    xp = xp.index_put((sorted_sel, pos), xs)
    hp = blk._act(torch.bmm(xp, blk._w_gate.transpose(1, 2))) * torch.bmm(xp, blk._w_up.transpose(1, 2))
    yp = torch.bmm(hp, blk._w_down.transpose(1, 2))
    return yp[sorted_sel, pos]


def _fast_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    x = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(x)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    out = torch.zeros_like(x)
    k = self.top_k
    step = x.shape[0] if self._moe_mode == "grouped" else max(1, CHUNK_TOKENS)
    for t0 in range(0, x.shape[0], step):
        t1 = min(x.shape[0], t0 + step)
        sel = selected_experts[t0:t1].reshape(-1)
        order = torch.argsort(sel, stable=True)
        sorted_sel = sel[order]
        tok = order // k + t0
        counts = torch.bincount(sel, minlength=self.num_experts)
        xs = x[tok]
        if self._moe_mode == "grouped":
            y = _expert_mlp_grouped(self, xs, counts)
        else:
            y = _expert_mlp_bmm(self, xs, sorted_sel, counts)
        y = y * routing_weights[t0:t1].reshape(-1)[order, None]
        out = out.index_add(0, tok, y.to(out.dtype))
    return out.reshape(batch_size, sequence_length, hidden_dim), router_logits


def grouped_mm_works(device, dtype=torch.bfloat16) -> bool:
    """True if torch._grouped_mm runs (forward + backward) on `device` and matches a per-group matmul reference."""
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        E, K, N = 4, 64, 32
        counts = torch.tensor([5, 0, 11, 16])
        a = torch.randn(int(counts.sum()), K, generator=g).to(device, dtype).requires_grad_(True)
        b = (torch.randn(E, N, K, generator=g) / K ** 0.5).to(device, dtype)
        offs = torch.cumsum(counts, 0).to(device=device, dtype=torch.int32)
        y = torch._grouped_mm(a, b.transpose(1, 2), offs=offs)
        y.float().square().sum().backward()
        ref, i = [], 0
        for e, c in enumerate(counts.tolist()):
            ref.append(a[i:i + c].float() @ b[e].float().t()); i += c
        ref = torch.cat(ref)
        return bool(torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2)) and a.grad is not None
    except Exception:                                                    # noqa: BLE001 - unsupported arch / build
        return False


def patch_moe(model, mode: str = "auto") -> str:
    """Patch every SDAR sparse MoE block of `model` in place; returns the mode used ("grouped" or "bmm")."""
    blocks = [m for m in model.modules() if type(m).__name__.endswith("SparseMoeBlock") and hasattr(m, "experts")]
    assert blocks, "no SDAR sparse MoE block found"
    dev = next(blocks[0].parameters()).device
    dt = next(blocks[0].experts[0].parameters()).dtype
    if mode == "auto":
        mode = "grouped" if (dt == torch.bfloat16 and grouped_mm_works(dev, dt)) else "bmm"
    for blk in blocks:
        blk._w_gate = _stack_in_place(blk.experts, "gate_proj")
        blk._w_up = _stack_in_place(blk.experts, "up_proj")
        blk._w_down = _stack_in_place(blk.experts, "down_proj")
        blk._act = blk.experts[0].act_fn
        blk._moe_mode = mode
        blk.forward = types.MethodType(_fast_forward, blk)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()
    return mode
