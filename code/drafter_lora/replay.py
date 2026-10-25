"""s8 first-draft replay of one corpus item, and the re-draft probe (Gate 1, the 25% check, the smoke parity check).

Same code path as `draftalign/eval_drafter_agreement.item_agreement` (a `SeqCache` holding the CLEAN 30B prefix, one
`drafter.draft_block(..., "s8")` per block from all-[MASK], then the 30B's tokens flushed), adapted to the 1B-corpus
item format and to the way the stored corpus drafts were made (`corpus1b/draft_sglang.py`):

  * the drafted (open) slots are every slot of the block except the prompt tail -- the post-EOS slots of the last
    block are drafted too, exactly like the stored drafts (which are -1 only on the prompt tail);
  * agreement counts the VALID slots only (the corpus' `valid`: no prompt tail, nothing after the EOS);
  * every block is replayed, the ragged last one included.

Re-draft probe, per block, after the first draft D: the slots where D equals T keep it, the other valid slots are
[MASK] (the holes; post-EOS slots keep the truth), and the drafter re-drafts the holes in min(8, ceil(holes / 4))
steps, with and without the ban on D at each hole.  Reported: re-draft agreement with T and how often the re-draft
repeats D (the rejected token).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from draftalign.drafter import draft_block
from draftalign.seqcache import SeqCache

MASK_ID = 151669
BLOCK = 32


def open_slots(item: dict, b: int) -> np.ndarray:
    """The slots the drafter fills in block b: all but the prompt tail."""
    o = np.ones(BLOCK, bool)
    if b == 0:
        pl = int(np.asarray(item["prompt_ids"]).shape[0])
        o[: pl - int(np.asarray(item["block_start"])[0])] = False
    return o


@torch.no_grad()
def replay_item(model, item: dict, device, *, setting: str = "s8", mask_id: int = MASK_ID,
                redraft_probe: bool = False, rate: int = 4) -> Dict[str, object]:
    """First drafts (+ optional re-draft probe) of every block.  Returns arrays and per-item counts."""
    prompt = torch.as_tensor(np.asarray(item["prompt_ids"], np.int64)).unsqueeze(0)
    tokens = np.asarray(item["tokens"], np.int64)
    valid = np.asarray(item["valid"], bool)
    starts = np.asarray(item["block_start"], np.int64)
    nb = tokens.shape[0]
    seq = SeqCache(model, BLOCK, "replay")
    seq.prefill(prompt)
    drafts = np.full((nb, BLOCK), -1, np.int64)
    conf = np.zeros((nb, BLOCK), np.uint8)
    right = np.zeros((nb, BLOCK), bool)
    rd = {"holes": 0, "agree": 0, "repeat": 0, "agree_ban": 0, "repeat_ban": 0, "fwd": 0, "fwd_ban": 0}
    n_fwd = 0
    for b in range(nb):
        lo = int(starts[b])
        final = torch.as_tensor(tokens[b], device=device)
        op = open_slots(item, b)
        opt = torch.as_tensor(op, device=device)
        out = draft_block(seq, final.clone(), opt, lo, setting, mask_id)
        D = out["tokens"]
        n_fwd += int(out["n_fwd"])
        drafts[b] = np.where(op, D.cpu().numpy(), -1)
        pc = torch.nan_to_num(out["p_commit"].float(), nan=0.0).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        conf[b] = np.where(op, pc.cpu().numpy(), 0)
        vt = torch.as_tensor(valid[b], device=device)
        right[b] = ((D == final) & vt).cpu().numpy()
        if redraft_probe:
            holes = vt & (D != final)
            n_h = int(holes.sum())
            if n_h:
                inp = torch.where(holes, torch.full_like(final, mask_id), final)
                steps = min(8, -(-n_h // int(rate)))
                ban = torch.where(holes, D, torch.full_like(D, -1))
                for tag, bn in (("", None), ("_ban", ban)):
                    r = draft_block(seq, inp.clone(), holes, lo, setting, mask_id, static_steps=steps,
                                    kind="redraft", ban=bn)
                    new = r["tokens"]
                    rd["agree" + tag] += int(((new == final) & holes).sum())
                    rd["repeat" + tag] += int(((new == D) & holes).sum())
                    rd["fwd" + tag] += int(r["n_fwd"])
                rd["holes"] += n_h
        seq.flush(final, lo)
    return {"pid": item.get("pid"), "domain": item.get("domain"), "drafts": drafts, "conf": conf, "right": right,
            "valid": valid, "n_fwd": n_fwd, "redraft": rd if redraft_probe else None}


def parity_with_stored(item: dict, drafts: np.ndarray) -> Dict[str, int]:
    """Replayed vs stored corpus drafts on the drafted slots (the smoke test's HF vs SGLang parity, >= 95%)."""
    stored = np.asarray(item["draft"], np.int64)
    m = stored >= 0
    return {"slots": int(m.sum()), "equal": int(((drafts == stored) & m).sum())}


def stored_agreement(item: dict) -> Dict[str, int]:
    """Agreement of the STORED corpus draft with T on the valid slots (the corpus' own untrained number)."""
    stored = np.asarray(item["draft"], np.int64)
    T = np.asarray(item["tokens"], np.int64)
    v = np.asarray(item["valid"], bool) & (stored >= 0)
    return {"slots": int(v.sum()), "agree": int(((stored == T) & v).sum())}


def eval_mode(model):
    """The replay calls `SDARForCausalLM.forward`, which drops the logits while training (fuse_cross_entropy)."""
    was = model.training
    model.eval()
    return was


def restore_mode(model, was: Optional[bool]):
    if was:
        model.train()
