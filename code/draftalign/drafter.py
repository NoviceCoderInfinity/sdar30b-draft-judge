"""Persistent-cache drafter: block-32 greedy generation of ONE block given the teacher-forced
prefix cache (`SeqCache`), in the three settings of the spec:

  s4     static 4 steps   : schedule get_num_transfer_tokens(n_open, min(4, n_open))
  s8     static 8 steps   : schedule get_num_transfer_tokens(n_open, min(8, n_open))
  dyn90  dynamic thr .90  : commit every open slot with conf > .90, else the single best
                            (decode.py:241-251 with num_transfer == 1), cap 32 steps

Numerics: logits are upcast to fp32, the [MASK] logit is suppressed (-inf) and softmax /
argmax / confidence are fp32 (the spec asks for fp32 probabilities; decode.py:56-92 uses
bf16 -- see the report).  Open slots = the block's valid positions; every other position
(the prompt tail of the first block) is fixed.  All forwards use store_kv=False; the caller
flushes the MoE's FINAL tokens afterwards (teacher forcing).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from sdar_harness.decode import get_num_transfer_tokens

SETTINGS: Dict[str, dict] = {
    "s4": {"mode": "static", "steps": 4},
    "s8": {"mode": "static", "steps": 8},
    "dyn90": {"mode": "dynamic", "threshold": 0.90, "cap": 32},
}


def masked_base(final: torch.Tensor, valid: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Block ids with [MASK] on every valid slot and the fixed tokens elsewhere."""
    base = final.clone()
    base[valid] = mask_id
    return base


@torch.no_grad()
def draft_block(seq, final: torch.Tensor, valid: torch.Tensor, lo: int, setting: str,
                mask_id: int, threshold_strict: bool = True, static_steps: Optional[int] = None,
                kind: Optional[str] = None) -> dict:
    """Generate the valid slots of block [lo, lo+bl) with the drafter behind `seq`.

    final [bl] long (device), valid [bl] bool (device). `static_steps` (static settings only)
    replaces the setting's step count (still capped at the number of open slots); `kind`
    replaces the timing label (default draft_<setting>; the phase-3 re-draft passes "redraft").
    Both default to None, which is exactly the original call. Returns a dict of device tensors
    + python ints:
      tokens [bl]        the drafted block (fixed slots untouched)
      p_commit [bl]      fp32 confidence of each slot's token at the forward that committed it
      p_final_last [bl]  fp32 prob of the MoE's FINAL token at every slot, from the LAST forward
      n_fwd, n_steps, raw_mask_argmax (count of valid slots whose raw fp32 argmax was [MASK],
      summed over forwards), ms [list per forward]
    """
    cfg = SETTINGS[setting]
    dev = final.device
    bl = final.numel()
    cur = masked_base(final, valid, mask_id).unsqueeze(0)
    open_ = valid.clone()
    n_open = int(valid.sum())
    p_commit = torch.zeros(bl, dtype=torch.float32, device=dev)
    ninf = torch.tensor(float("-inf"), device=dev)
    raw_mask = torch.zeros((), dtype=torch.long, device=dev)
    t_start = len(seq.timings)
    last_probs = None
    n_steps = 0
    if n_open == 0:
        return {"tokens": cur[0], "p_commit": p_commit, "p_final_last": p_commit.clone(),
                "n_fwd": 0, "n_steps": 0, "raw_mask_argmax": 0, "ms": []}

    if cfg["mode"] == "static":
        steps = min(int(cfg["steps"] if static_steps is None else static_steps), n_open)
        assert steps >= 1, static_steps
        nt = get_num_transfer_tokens(n_open, steps)           # sums to n_open
        assert int(nt.sum()) == n_open
    else:
        if static_steps is not None:
            raise ValueError(f"static_steps applies to static settings only, not {setting}")
        steps = min(int(cfg["cap"]), n_open)                   # >= 1 commit per step
        nt = None
    label = kind or f"draft_{setting}"

    for step in range(steps):
        logits = seq.block_forward(cur, lo, store_kv=False, kind=label)
        lg = logits[0].to(torch.float32, copy=True)            # [bl, V] fp32 copy (MASK logit is overwritten)
        raw_mask += (lg.argmax(-1) == mask_id)[open_].sum()
        lg[:, mask_id] = float("-inf")
        probs = F.softmax(lg, dim=-1)
        conf, x0 = probs.max(dim=-1)
        last_probs = probs
        conf_open = torch.where(open_, conf, ninf)
        if nt is not None:
            k = int(nt[step])
            transfer = torch.zeros(bl, dtype=torch.bool, device=dev)
            transfer[torch.topk(conf_open, k).indices] = True
        else:
            thr = float(cfg["threshold"])
            high = (conf_open > thr) if threshold_strict else (conf_open >= thr)
            top1 = torch.zeros(bl, dtype=torch.bool, device=dev)
            top1[conf_open.argmax()] = True
            transfer = torch.where(high.any(), high, top1)
        cur[0] = torch.where(transfer, x0, cur[0])
        p_commit = torch.where(transfer, conf, p_commit)
        open_ = open_ & ~transfer
        n_steps += 1
        if nt is None and int(open_.sum()) == 0:
            break
    assert int(open_.sum()) == 0, f"{setting}: {int(open_.sum())} slots left open after {n_steps} steps"
    p_final_last = last_probs.gather(1, final.view(-1, 1)).view(-1)
    ms = [m for _, m in seq.timings[t_start:]]
    return {"tokens": cur[0], "p_commit": p_commit, "p_final_last": p_final_last,
            "n_fwd": n_steps, "n_steps": n_steps, "raw_mask_argmax": int(raw_mask), "ms": ms}


@torch.no_grad()
def draft_all_settings(seq, final, valid, lo, mask_id) -> Dict[str, dict]:
    return {name: draft_block(seq, final, valid, lo, name, mask_id) for name in SETTINGS}
