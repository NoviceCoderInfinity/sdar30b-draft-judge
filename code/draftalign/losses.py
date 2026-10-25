"""Draft-aligned loss (TRAINER_SPEC_v0 section 2, design 3.3): CE + certainty forcing.

    loss = mean_i CE(logits_i, T_i)  +  beta * mean_{i : argmax_i == T_i} H(softmax(logits_i / temp))

The entropy term is dParallel's certainty forcing (arXiv 2509.26488): it is applied
only where the model is ALREADY right, so it sharpens a correct distribution instead
of pushing a wrong one. The [MASK] logit is suppressed before the entropy softmax --
otherwise the model can lower the entropy by piling probability on [MASK], which is
never a legal output token.

The logits must be fp32 and must cover ONLY the loss rows: `active_logits` applies
`lm_head` to the ~32 active-block rows, so the [S, 151936] tensor is never built.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def active_logits(model, hidden: torch.Tensor, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`lm_head` on the given rows only, in fp32.  hidden [n, H] -> logits [k, V]."""
    h = hidden if rows is None else hidden[rows]
    return model.lm_head(h).float()


def masked_ce_and_certainty(logits_fp32: torch.Tensor, targets: torch.Tensor, beta: float = 2.0,
                            temp: float = 0.5, mask_id: int = 151669,
                            label_smoothing: float = 0.0, calib: float = 0.0,
                            row_weight: Optional[torch.Tensor] = None,
                            cert_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """logits [n, V] fp32, targets [n] long -> (loss, stats).

    Round-3b additions (both off by default, so the r3_A loss is unchanged):
      row_weight [n]  per-row weights for the CE and calibration means (e.g. 2x on OVERRIDE rows,
                      where the draft token is wrong); the certainty term stays an unweighted mean.
      calib           + calib * mean_i BCE(p_i, 1[argmax_i == T_i]), p_i = top-1 of the fp32 softmax
                      with [MASK] suppressed (the full judge's commit confidence). A confident WRONG row
                      costs -log(1 - p_i), which pulls `p > 0.95` back towards 95%-precise commits.

    1B-corpus addition (off by default: None leaves every earlier loss bit-identical):
      cert_mask [n]   bool; the certainty term averages only over rows that are correct AND in the mask.
                      Chat rows are left out (the 30B is often torn between words in prose, and pushing the
                      judge to be surer than the 30B there is false confidence; docs/CORPUS_1B_BUILD.md).

    stats are DETACHED 0-dim tensors (`ce`, `cert`, `n_rows`, `n_correct`, `acc`,
    `top1`); the caller turns them into floats once per step rather than syncing
    inside the hot path.
    """
    assert logits_fp32.dim() == 2 and targets.dim() == 1, (logits_fp32.shape, targets.shape)
    assert logits_fp32.shape[0] == targets.shape[0], (logits_fp32.shape, targets.shape)
    assert logits_fp32.dtype == torch.float32, logits_fp32.dtype
    n = logits_fp32.shape[0]
    if n == 0:
        z = logits_fp32.sum() * 0.0                       # keeps the graph connected
        zero = torch.zeros((), device=logits_fp32.device)
        return z, {"ce": zero, "cert": zero, "calib": zero, "n_rows": zero, "n_correct": zero,
                   "acc": zero, "top1": zero}

    if row_weight is None:
        ce = F.cross_entropy(logits_fp32, targets, reduction="mean", label_smoothing=label_smoothing)
    else:
        rw = row_weight.to(device=logits_fp32.device, dtype=logits_fp32.dtype)
        ce_rows = F.cross_entropy(logits_fp32, targets, reduction="none", label_smoothing=label_smoothing)
        ce = (ce_rows * rw).sum() / rw.sum().clamp_min(1e-6)

    correct = logits_fp32.argmax(dim=-1) == targets
    n_correct = correct.sum()
    # entropy on every row, averaged over the correct ones only. Written as a masked
    # mean rather than `logits[correct]` so there is no host sync on `correct` and the
    # graph stays connected when no row is correct (the term is then exactly 0).
    t = logits_fp32 / temp
    # "the MASK logit suppressed" = the column removed. Setting it to -inf instead
    # would make the entropy term 0 * log(0) = NaN (and a NaN gradient).
    t = torch.cat([t[:, :mask_id], t[:, mask_id + 1:]], dim=1)
    logp = F.log_softmax(t, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    w = correct.to(ent.dtype)
    if cert_mask is not None:
        w = w * cert_mask.to(device=w.device, dtype=w.dtype)
    cert = (ent * w).sum() / w.sum().clamp_min(1.0)

    loss = ce + beta * cert
    calib_t = torch.zeros((), device=logits_fp32.device)
    if calib > 0:
        z = torch.cat([logits_fp32[:, :mask_id], logits_fp32[:, mask_id + 1:]], dim=1)
        p_top, i_top = F.softmax(z, dim=-1).max(dim=-1)
        i_top = i_top + (i_top >= mask_id).long()           # back to vocab ids (the MASK column was removed)
        right = (i_top == targets).to(p_top.dtype)          # the token the judge would commit is the target
        bce = F.binary_cross_entropy(p_top.clamp(1e-6, 1.0 - 1e-6), right, reduction="none")
        calib_t = bce.mean() if row_weight is None else (bce * rw).sum() / rw.sum().clamp_min(1e-6)
        loss = loss + calib * calib_t
    with torch.no_grad():
        top1 = F.softmax(logits_fp32, dim=-1).max(dim=-1).values.mean()
        stats = {"ce": ce.detach(), "cert": cert.detach(), "calib": calib_t.detach(),
                 "n_rows": torch.tensor(float(n), device=logits_fp32.device),
                 "n_correct": n_correct.float(),
                 "acc": n_correct.float() / n, "top1": top1}
    return loss, stats


@torch.no_grad()
def eval_row_stats(logits_fp32: torch.Tensor, targets: torch.Tensor, tau: float = 0.95,
                   override: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Held-out diagnostics: argmax accuracy, and the precision of `top1 > tau` rows."""
    p = F.softmax(logits_fp32, dim=-1)
    top_p, top_i = p.max(dim=-1)
    correct = (top_i == targets).float()
    conf = top_p > tau
    ov = torch.zeros_like(correct) if override is None else override.to(device=p.device, dtype=correct.dtype)
    return {"n_rows": torch.tensor(float(targets.numel()), device=p.device),
            "acc": correct.mean() if targets.numel() else torch.zeros((), device=p.device),
            "n_conf": conf.sum().float(),
            "conf_correct": (correct * conf.float()).sum(),
            "n_ovr": ov.sum(), "ovr_correct": (correct * ov).sum(),
            "n_conf_ovr": (conf.float() * ov).sum(), "conf_correct_ovr": (correct * conf.float() * ov).sum(),
            "mean_top1": top_p.mean() if targets.numel() else torch.zeros((), device=p.device)}
