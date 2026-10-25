"""The two-stream forward of the dense drafter and the phase 4 loss.

FORWARD.  The decoder layers are run here, one by one, with an explicit 4D boolean mask through SDPA -- the same
pieces as `draftalign/train_fast.py` (`_qkv`, `_attn_out`, `_mlp`), never `SDARForCausalLM.forward`.  That avoids
both remote-code pitfalls documented in `draftalign/train_drafter.py`: the `fuse_cross_entropy` branch that drops the
logits while training, and the `torch.all(mask)` fast path through `flash_attn_func`.  Every `LoRALinear` sits inside
the projections, so the adapter is applied wherever the layer calls them (both streams).

LOSS (runbook section 3a; H = the valid [MASK] slots of the micro-batch, T_i = the 30B's token):

    L = (1/|H|) sum_{i in H} -log q_theta,i(T_i)  +  w_kl (1/|H|) sum_{i in H : argmax q_0,i = T_i} KL(q_0,i || q_theta,i)

    q_theta = the drafter with the adapter; q_0 = the same forward with the adapter off, under no_grad.
    Logits are fp32 at the loss rows only (the bf16 lm_head output upcast, exactly what the drafter's decode reads),
    with the [MASK] logit at -inf in every distribution.  The KL is F.kl_div(log q_theta, log q_0, log_target=True)
    summed over the vocabulary; the [MASK] column is zeroed in both log-distributions first, because
    exp(-inf) * (-inf - -inf) would be NaN (its true contribution is 0).  One denominator |H| for both terms.
    At step 0 (lora_B = 0) the two forwards are bit-identical, so the KL term is exactly 0.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from draftalign.train_fast import _attn_out, _mlp, _qkv
from drafter_lora.data import KIND_FIRST, KIND_REDRAFT, Batch
from drafter_lora.lora import adapter_off

MASK_ID = 151669


def two_stream_hidden(model, ids: torch.Tensor, pos: torch.Tensor, attn: torch.Tensor,
                      checkpoint_layers: bool = True) -> torch.Tensor:
    """[B, L, H] final-norm hidden states under the explicit mask `attn` [B, 1, L, L] (True = attend)."""
    inner = model.model
    h = inner.embed_tokens(ids)
    cos, sin = inner.rotary_emb(h, pos)

    def run_layer(hh, li):
        layer = inner.layers[li]
        res = hh
        x = layer.input_layernorm(hh)
        q, k, v = _qkv(layer.self_attn, x, cos, sin)
        hh = res + _attn_out(layer.self_attn, q, k, v, attn, x.shape[:-1])
        return hh + _mlp(layer, layer.post_attention_layernorm(hh))

    for li in range(len(inner.layers)):
        if checkpoint_layers and torch.is_grad_enabled():
            h = checkpoint(run_layer, h, li, use_reentrant=False)
        else:
            h = run_layer(h, li)
    return inner.norm(h)


def _row_logits(model, h: torch.Tensor, rows: torch.Tensor, mask_id: int) -> torch.Tensor:
    z = model.lm_head(h[rows]).float()
    return z.index_fill(1, torch.tensor([mask_id], device=z.device), float("-inf"))


def drafter_loss(model, batch: Batch, *, kl_weight: float = 0.1, mask_id: int = MASK_ID,
                 checkpoint_layers: bool = True, autocast: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
    """(loss, stats) of one micro-batch.  `stats` holds sums (n_rows, ce_sum, kl_sum, ...) so ranks and micro-batches
    can be pooled exactly; `loss` is already normalised by this micro-batch's |H|."""
    dev = next(model.parameters()).device
    ids, pos = batch.ids.to(dev, non_blocking=True), batch.pos.to(dev, non_blocking=True)
    attn = batch.attn.to(dev, non_blocking=True)
    rows = batch.rows.to(dev)
    tgt = batch.targets.to(dev)[rows]
    kinds = batch.kinds.to(dev)[rows]
    n = int(tgt.numel())
    assert n > 0, f"micro-batch {batch.pids} has no loss row"
    use_ac = bool(autocast) and dev.type == "cuda"
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_ac):
        h = two_stream_hidden(model, ids, pos, attn, checkpoint_layers)
        z = _row_logits(model, h, rows, mask_id)
    with torch.no_grad():
        with adapter_off(tuple(ids.shape), dev):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_ac):
                h0 = two_stream_hidden(model, ids, pos, attn, checkpoint_layers=False)
                z0 = _row_logits(model, h0, rows, mask_id)
        del h0
        lp0 = F.log_softmax(z0, dim=-1)
        del z0
        pred0 = lp0.argmax(-1)
        in_kl = pred0 == tgt
    lp = F.log_softmax(z, dim=-1)
    ce = -lp.gather(1, tgt[:, None]).squeeze(1)
    mcol = torch.tensor([mask_id], device=dev)
    if bool(in_kl.any()):
        kl_rows = F.kl_div(lp[in_kl].index_fill(1, mcol, 0.0), lp0[in_kl].index_fill(1, mcol, 0.0),
                           reduction="none", log_target=True).sum(-1)
        kl_sum = kl_rows.sum()
    else:
        kl_sum = lp.new_zeros(())
    loss = (ce.sum() + float(kl_weight) * kl_sum) / n
    with torch.no_grad():
        pred = lp.argmax(-1)
        right, right0 = pred == tgt, pred0 == tgt
        first, red = kinds == KIND_FIRST, kinds == KIND_REDRAFT
        stats = {"n_rows": float(n), "ce_sum": float(ce.sum()), "kl_sum": float(kl_sum.detach()),
                 "n_kl": float(in_kl.sum()), "n_right": float(right.sum()), "n_right0": float(right0.sum()),
                 "n_kept": float((right & right0).sum()),
                 "n_first": float(first.sum()), "n_right_first": float((right & first).sum()),
                 "n_right0_first": float((right0 & first).sum()),
                 "n_redraft": float(red.sum()), "n_right_redraft": float((right & red).sum()),
                 "n_right0_redraft": float((right0 & red).sum()),
                 "n_items": float(len(batch.pids)), "n_pos": float(batch.n_pos)}
    return loss, stats


def summarize(tot: Dict[str, float]) -> Dict[str, float]:
    """Ratios from pooled sums (micro-batches and ranks)."""
    n = max(tot.get("n_rows", 0.0), 1.0)
    out = {"ce": tot.get("ce_sum", 0.0) / n, "kl": tot.get("kl_sum", 0.0) / n,
           "kl_share": tot.get("n_kl", 0.0) / n, "acc": tot.get("n_right", 0.0) / n,
           "acc0": tot.get("n_right0", 0.0) / n,
           "retention": tot.get("n_kept", 0.0) / max(tot.get("n_right0", 0.0), 1.0),
           "rows": tot.get("n_rows", 0.0), "items": tot.get("n_items", 0.0), "positions": tot.get("n_pos", 0.0)}
    out["loss"] = out["ce"] + tot.get("kl_weight", 0.1) * out["kl"]
    for k in ("first", "redraft"):
        m = tot.get(f"n_{k}", 0.0)
        out[f"acc_{k}"] = tot.get(f"n_right_{k}", 0.0) / m if m else float("nan")
        out[f"acc0_{k}"] = tot.get(f"n_right0_{k}", 0.0) / m if m else float("nan")
        out[f"share_{k}"] = m / n
    return out
