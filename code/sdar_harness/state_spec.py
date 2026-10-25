"""Masked-state helpers for research_v1 (pure functions + a pass-by-pass driver).

Transition contract (handoff section 10). All logits-to-commit operations go
through the reference `sample_with_confidence` (bf16 softmax, argmax, gathered
confidence) and the reference commit rule of decode.py / run_desvote.decode:

    open slots above the threshold commit (strict >); if fewer than `need`
    are above it, the top-`need` open slots by confidence commit
    (torch.topk over -inf-masked confidence, exactly as the reference);
    closed / invalid slots never commit.

`reference_decode_states` re-implements the reference block loop (grid,
prefill, passes, kv flush, EOS at block boundaries) and calls `on_state`
before every denoising forward with the complete state, so a probe can run
extra forwards on the side. The real decode only ever advances with the
logits the callback returns (or with a fresh sequential forward when the
callback returns None); the driver never installs anything else. Its output
ids can be compared with the reference runner's records for the same prompt.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch

from .decode import get_num_transfer_tokens, sample_with_confidence

MASK_ID = 151669
EOS_IDS = [151645, 151643]


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------
def derive_transition(logits: torch.Tensor, cur: torch.Tensor, open_mask: torch.Tensor,
                      need: int, threshold: float):
    """logits [1, L, V], cur [1, L] token ids, open_mask [1, L] bool (open AND valid).
    Returns dict(x0 [1, L] proposed ids with cur at closed slots, conf [1, L] bf16
    confidence (-inf at closed), transfer [1, L] bool, above [L] bool, next [1, L])."""
    assert logits.shape[0] == 1 and cur.shape == open_mask.shape == logits.shape[:2]
    x0_raw, p = sample_with_confidence(logits, temperature=1.0, top_k=1, top_p=1.0, greedy_exact=True)
    x0 = torch.where(open_mask, x0_raw, cur)
    ninf = torch.tensor(float("-inf"), device=p.device, dtype=p.dtype)
    conf = torch.where(open_mask, p, ninf)
    transfer = torch.zeros_like(x0, dtype=torch.bool)
    above = conf[0] > threshold
    n_open = int(open_mask.sum())
    if n_open > 0:
        if int(above.sum()) >= need:
            transfer[0] = above
        else:
            _, idx = torch.topk(conf[0], min(need, n_open))
            transfer[0, idx] = True
    nxt = torch.where(transfer, x0, cur)
    return {"x0": x0, "conf": conf, "transfer": transfer, "above": above, "next": nxt,
            "positions": torch.nonzero(transfer[0]).flatten().tolist(),
            "tokens": x0[0, transfer[0]].tolist()}


def transition_key(tr) -> tuple:
    """Hashable complete transition: committed positions and their tokens."""
    return tuple(zip(tr["positions"], tr["tokens"]))


def summarize_prev(tr, step: int) -> dict:
    """What the next pass may use from this evaluation: per-slot argmax and confidence."""
    return {"argmax": tr["x0"][0].detach().clone(), "conf": tr["conf"][0].detach().clone(), "step": step}


def propose_children(cur: torch.Tensor, open_mask: torch.Tensor, prev: dict, ks=(1,)) -> List[dict]:
    """Rank open slots by previous-pass confidence (stable, lowest index first on
    ties); child k fills the top-k slots with their previous-pass argmax. Skips
    k > n_open and any child that would write [MASK]. Keeps k, slots, tokens and
    the pre-forward features in each record."""
    op = open_mask[0]
    n_open = int(op.sum())
    conf = torch.where(op, prev["conf"].float(), torch.full_like(prev["conf"], float("-inf"), dtype=torch.float))
    order = torch.sort(conf, descending=True, stable=True).indices
    ranked = [int(i) for i in order[:n_open]]
    top_conf = [float(conf[i]) for i in ranked[:4]]
    out = []
    for k in ks:
        if k > n_open:
            continue
        slots = ranked[:k]
        toks = [int(prev["argmax"][i]) for i in slots]
        if any(t == MASK_ID for t in toks):
            continue
        H = cur.clone()
        H[0, slots] = torch.tensor(toks, device=cur.device, dtype=cur.dtype)
        mH = open_mask.clone(); mH[0, slots] = False
        out.append({"k": k, "slots": slots, "tokens": toks, "state": H, "open_mask": mH,
                    "n_open_parent": n_open, "prev_top_conf": top_conf})
    return out


def child_accepted(child: dict, root_tr: dict) -> bool:
    """The child's ENTIRE guessed transition equals the root's reference transition."""
    return (sorted(root_tr["positions"]) == sorted(child["slots"])
            and dict(zip(root_tr["positions"], root_tr["tokens"])) == dict(zip(child["slots"], child["tokens"])))


# ---------------------------------------------------------------------------
# caches / grids
# ---------------------------------------------------------------------------
def make_batch_cache(prefix, rows: int, copy: bool = False):
    """A separate DynamicCache with every layer's prefix K/V repeated `rows` times
    (expand = zero-copy views, repeat = materialised). Never mutates `prefix`."""
    from transformers.cache_utils import DynamicCache
    bc = DynamicCache()
    for l in range(len(prefix)):
        k, v = prefix[l]
        if copy:
            k2, v2 = k.repeat(rows, 1, 1, 1), v.repeat(rows, 1, 1, 1)
        else:
            k2, v2 = k.expand(rows, -1, -1, -1), v.expand(rows, -1, -1, -1)
        bc.update(k2, v2, l)
    return bc


def snapshot_cache(cache):
    return [(k.clone(), v.clone()) for k, v in cache]


def cache_unchanged(cache, snap) -> bool:
    if len(cache) != len(snap):
        return False
    return all(torch.equal(k, k0) and torch.equal(v, v0) for (k, v), (k0, v0) in zip(list(cache), snap))


def block_grid(plen: int, gen_length: int, bl: int, device):
    num_blocks = (plen + gen_length + bl - 1) // bl
    total = num_blocks * bl
    bm = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = bm.repeat_interleave(bl, 0).repeat_interleave(bl, 1).unsqueeze(0)
    pos = torch.arange(total, dtype=torch.long, device=device).unsqueeze(0)
    return attn, pos, num_blocks, total


# ---------------------------------------------------------------------------
# reference driver with a per-state callback
# ---------------------------------------------------------------------------
@torch.no_grad()
def reference_decode_states(model, input_ids: torch.Tensor, *, bl: int, steps: int, threshold: float,
                            gen_length: int, on_state: Optional[Callable] = None,
                            max_blocks: Optional[int] = None, on_forward: Optional[Callable] = None):
    """Reference block-diffusion decode (token-identical to decode.block_diffusion_generate)
    that exposes every pre-forward state.

    on_state(ctx) -> None | logits. ctx has: cache, cur [1,bl], ca, cp, open_mask [1,bl],
    valid [bl], need, prev (None on the first pass), block, step, prefix_len, plen,
    n_open. If it returns logits they are used as THE sequential root evaluation of
    ctx['cur']; the driver then applies derive_transition itself.
    on_forward(kind) is called before each model forward the driver runs itself.
    Returns dict(x, generated ids, blocks: [{block, start, n_valid, passes: [{step, n_open, commits}], flushed}],
                 stopped_eos, n_forward, prompt_length)."""
    from transformers.cache_utils import DynamicCache
    dev = model.device
    ids = input_ids.to(dev)
    plen = ids.shape[1]
    attn, pos, num_blocks, total = block_grid(plen, gen_length, bl, dev)
    x = torch.full((1, total), MASK_ID, dtype=torch.long, device=dev)
    x[:, :plen] = ids
    pfb = plen // bl
    cache = DynamicCache()
    n_forward = 0
    if pfb:
        if on_forward:
            on_forward("prefill")
        model(input_ids=x[:, :pfb * bl], attention_mask=attn[:, :pfb * bl, :pfb * bl],
              position_ids=pos[:, :pfb * bl], past_key_values=cache, use_cache=True, store_kv=True)
        n_forward += 1
    nt = get_num_transfer_tokens(bl, steps)
    last = num_blocks if max_blocks is None else min(num_blocks, pfb + max_blocks)
    blocks, stopped = [], False
    hi_done = plen
    for nb in range(pfb, last):
        lo, hi = nb * bl, (nb + 1) * bl
        valid = torch.arange(lo, hi, device=dev) >= plen
        ca, cp = attn[:, lo:hi, :hi], pos[:, lo:hi]
        cur = x[:, lo:hi].clone()
        prev = None
        binfo = {"block": nb, "start": lo, "n_valid": int(valid.sum()), "passes": [], "flushed": False}
        for step in range(steps + 1):
            open_mask = (cur == MASK_ID) & valid.unsqueeze(0)
            n_open = int(open_mask.sum())
            if n_open == 0:
                assert cache.get_seq_length() == lo
                if on_forward:
                    on_forward("kvflush")
                model(input_ids=cur, attention_mask=ca, position_ids=cp,
                      past_key_values=cache, use_cache=True, store_kv=True)
                n_forward += 1
                assert cache.get_seq_length() == hi
                binfo["flushed"] = True
                break
            need = int(nt[min(step, len(nt) - 1)])
            ctx = {"cache": cache, "cur": cur, "ca": ca, "cp": cp, "open_mask": open_mask, "valid": valid,
                   "need": need, "prev": prev, "block": nb, "step": step, "prefix_len": lo, "plen": plen,
                   "n_open": n_open, "x": x}
            logits = on_state(ctx) if on_state is not None else None
            if logits is None:
                if on_forward:
                    on_forward("denoise")
                logits = model(input_ids=cur, attention_mask=ca, position_ids=cp,
                               past_key_values=cache, use_cache=True, store_kv=False).logits
                n_forward += 1
            tr = derive_transition(logits, cur, open_mask, need, threshold)
            cur = tr["next"]
            prev = summarize_prev(tr, step)
            binfo["passes"].append({"step": step, "n_open": n_open, "commits": len(tr["positions"])})
            del logits
        assert binfo["flushed"], "block ended without a flush (steps exhausted with open slots?)"
        x[:, lo:hi] = cur
        hi_done = hi
        blocks.append(binfo)
        if any((x[:, plen:] == e).any() for e in EOS_IDS):
            stopped = True
            break
    gen = x[0, plen:hi_done]
    return {"x": x, "generated": gen[gen != MASK_ID].cpu(), "blocks": blocks, "stopped_eos": stopped,
            "n_forward": n_forward, "prompt_length": plen, "cache": cache}
