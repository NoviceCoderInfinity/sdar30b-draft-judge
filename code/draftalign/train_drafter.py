#!/usr/bin/env python
"""Full fine-tuning of the DENSE drafter SDAR-1.7B-Chat-b32 on the MoE's own outputs.

Round 3 of the draft-aligned plan (EXPLAINER_DRAFT_ALIGNED_TRAINING.md section 19): the judge's
confirmations are capped by the drafter's 55.5% slot agreement with the big model (section 18),
so the next lever is the drafter itself.  Targets are SDAR-30B-A3B-Chat-b32's own generated
tokens -- knowledge distillation on outputs, exactly what EAGLE/Medusa do for speculative
decoding, and what section 12 names "the next cheap lever".

THE BUG THIS FILE WAS FIXED FOR (2026-09-08).  Until now every example noised EVERY block of
the response in ONE sequence and ran ONE block-causal forward, so every block was predicted from
a NOISED prefix: block b saw the partially-[MASK]ed blocks 0..b-1 as its context.  At inference
the prefix is always CLEAN -- the committed tokens of the earlier blocks sit in the KV cache
(seqcache.py) -- and only the CURRENT block is masked.  The 2026-09-07 run therefore trained,
and its held-out `evaluate()` measured, a condition the drafter never meets.  The probe is
unambiguous: the fine-tune moved the noised-prefix masked-slot accuracy 0.749 -> 0.804 and the
clean-prefix one 0.847 -> 0.857, with the fully-masked first draft step (the state that actually
starts every block at inference) 0.406 -> 0.410, i.e. nothing.

THE FIX: `--conditioning clean_prefix` (the default) mirrors the JUDGE trainer's construction in
`dataset.py` (`StateSampler`: "The prefix is clean: the prompt plus the FINAL tokens of the
earlier blocks", one active block per example, block-causal attention, loss only on the active
block).  The old construction is kept, reachable and bit-for-bit reproducible, as
`--conditioning noisy_all` -- that is what the 2026-09-07 run used.

THE OBJECTIVE (official SDAR block diffusion, as in LLaDA / dParallel)
    * the prompt is clean and is never masked and never a loss row;
    * a 32-token block of the RESPONSE draws its own mask ratio p ~ U(0, 1); each response
      slot of that block is replaced by [MASK] independently with probability p, and if the draw
      masked nothing one slot is forced, so every noised block has >= 1 masked token;
    * attention is block-causal on the ABSOLUTE 32-grid -- each block attends to itself
      bidirectionally and to every earlier block (`train_forward.block_tril_bool`), which is
      exactly what the inference KV cache implements (seqcache.py docstring);
    * the loss is cross-entropy at the MASKED positions only, against the MoE's tokens.

    --conditioning clean_prefix (DEFAULT, the fix)  ONE active block per example, drawn uniformly
        among the item's response blocks (fresh draw each visit).  The sequence is the prompt plus
        the teacher's CLEAN tokens of every earlier block plus the active block with its response
        slots masked at p ~ U(ratio_lo, ratio_hi); it is truncated right after the active block
        (block-causal attention means later blocks are never attended, so they are dropped instead
        of computed).  Loss rows = the masked response slots of the active block only.  This is
        exactly the inference state, and `evaluate()` scores exactly it.
    --conditioning noisy_all (the 2026-09-07 run)  EVERY block of the item noised in one sequence,
        one forward, loss on all of them -- the buggy condition, kept for provenance.

Blocks whose sequence would be a single 32-block (an item with `block_start == 0`, i.e. a prompt
shorter than 32 tokens, and the active block 0) are NOT eligible: `hidden_states` refuses a
one-block sequence (the block-tril would be all ones and the remote attention would take its
flash_attn fast path).  On the 2026-09-07 union index exactly ONE of the 8533 training items has
`block_start == 0` (idx 1227, 7 blocks), so it simply contributes 6 eligible blocks instead of 7;
an item with NO eligible block is skipped with a printed message.

TRAINING-TIME AGREEMENT (`--agree-every`, `--agree-items`).  Masked-slot accuracy is a proxy;
the number the whole round-3 lever is about is the slot AGREEMENT of the drafter with the MoE's
recorded blocks (`eval_drafter_agreement.item_agreement`, an 8-step replay from all-[MASK]
against the teacher-forced prefix).  Every `--agree-every` steps that exact function is called on
the in-memory model over `--agree-items` held-out items, and the STOCK drafter's agreement on the
SAME items is measured once at startup, so every eval row carries `agree_ckpt`, `agree_stock` and
`agree_delta`.  `--early-stop-patience` stops the run (after `--min-steps`) when `agree_ckpt` has
not beaten its best for that many consecutive agreement evals, and the best-agreement step is
recorded in a `best` marker next to `latest` and is never pruned.  The in-loop number is computed
with fp32 weights under bf16 autocast while `eval_drafter_agreement.py` loads a bf16 checkpoint,
so the two can differ in the last digit; `agree_stock` is measured the same way as that script
(bf16 weights), which is why `agree_delta` is the number to watch, not `agree_ckpt` alone.

Deviations from the official recipe, and why (each is a CLI flag):
    --mask-weighting mean (default)  The official code weights each masked token by 1/p_mask
        (the LLaDA ELBO estimator).  That is unbiased for the NLL but has unbounded variance:
        a block that drew p = 0.02 and masked one token gets weight 50 against weight 1 for a
        fully masked block, and with our batch of a handful of sequences per step that alone
        dominates the gradient.  We are not estimating a likelihood, we are distilling tokens,
        so plain mean CE over masked slots is the default; `--mask-weighting inv_p` restores the
        official weighting (normalised, so the loss scale is comparable).
    --min-blocks 2 (default)  A sequence of exactly ONE 32-block (a prompt under 32 tokens whose
        response also fits in one block) makes the block-tril mask ALL ONES, and the remote
        attention then takes its `torch.all(mask)` fast path through `flash_attn_func`
        (modeling_sdar.py:275-287), which does not honour autocast and would meet fp32 q/k (rope
        runs in the fp32 hidden dtype) against bf16 v.  `hidden_states` asserts against that
        case; `--min-blocks 2` additionally drops items whose whole response is one block, which
        carry at most 32 supervised slots.  On the GSM8K pool the smallest item has 3 blocks.

PRECISION.  Parameters are kept in FP32 and every matmul runs under `torch.autocast(bfloat16)`
(the standard HF `bf16=True` layout).  Plain bf16 AdamW is NOT an option here: bf16 has an 8-bit
mantissa (relative resolution 2^-8 = 3.9e-3) while one AdamW step moves a weight by about
lr = 1e-5 in absolute terms, i.e. ~5e-4 relative for a typical |w| ~ 0.02 -- an order of
magnitude below the rounding grid, so most updates would round away.  Cost: 6.8 GB of weights
+ 6.8 GB of gradients + 13.6 GB of AdamW moments = 27 GB per rank; the 1.7 B model leaves plenty
of room on a 96 GB card.  `--param-dtype bf16` exists only for a memory-starved emergency and
is documented as not recommended.

DISTRIBUTED.  One full model per rank, batch 1 sequence per micro-step, gradient accumulation,
and an explicit bucketed all-reduce of the gradients at the end of every optimiser step (the
same design as `train.py`, minus its 100 M-parameter special case: here every parameter is
trained, so the reduce is bucketed to bound the temporary buffer).  DDP is not used because the
forward calls `model.model(...)` and `model.lm_head(...)` separately, which DDP's hooks on
`model.forward` would never see.

    # 1 GPU
    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/torchrun --standalone --nproc_per_node=1 \
        code/draftalign/train_drafter.py --index logs/draftalign/drafterset_union/index.jsonl \
        --out logs/draftalign/train_drafter/d1 --steps 2000 --grad-accum 16 --lr 1e-5

    # 2 GPUs (same global batch of 16 sequences: halve the accumulation)
    CUDA_VISIBLE_DEVICES=0,1 /venv/main/bin/torchrun --standalone --nproc_per_node=2 \
        code/draftalign/train_drafter.py --index logs/draftalign/drafterset_union/index.jsonl \
        --out logs/draftalign/train_drafter/d1 --steps 2000 --grad-accum 8 --lr 1e-5

Checkpoints are FULL HF directories (`save_pretrained` + the tokenizer + the remote-code .py
files copied from the snapshot), so `research_protocol.load_sdar`-style loading works on them:

    AutoModelForCausalLM.from_pretrained(<ckpt>, trust_remote_code=True, local_files_only=True,
                                         torch_dtype=torch.bfloat16, attn_implementation="sdpa")
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402
from draftalign import wandb_log                                        # noqa: E402
from research_protocol import MASK_ID, LaunchLog, config_hash, load_sdar, safe_out   # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.distributed as dist                                         # noqa: E402
import torch.nn.functional as F                                          # noqa: E402
from torch.utils.checkpoint import checkpoint                            # noqa: E402

from draftalign.dataset import _seed                                     # noqa: E402  (same hash as the MoE sampler)
from draftalign.losses import active_logits                              # noqa: E402
from draftalign.train_forward import block_tril_bool                     # noqa: E402

BLOCK = 32
IGNORE = -100
WEIGHTINGS = ("mean", "inv_p")
CONDITIONINGS = ("clean_prefix", "noisy_all")


# ===========================================================================
# data
# ===========================================================================
class Example(NamedTuple):
    input_ids: torch.Tensor        # [S] long, the masked sequence
    targets: torch.Tensor          # [S] long, IGNORE (-100) everywhere but the masked response slots
    loss_mask: torch.Tensor        # [S] bool, the loss rows
    weights: torch.Tensor          # [S] float32, 1.0 or 1/p per loss row (0 elsewhere)
    resp: torch.Tensor             # [S] bool, the response slots (pad excluded)
    ratios: List[float]            # the drawn mask ratio of every response block
    meta: dict


@dataclass
class DrafterSetConfig:
    index_path: str
    seed: int = 0
    block: int = BLOCK
    mask_id: int = MASK_ID
    max_seq: Optional[int] = 1200
    min_blocks: int = 2
    only_correct: bool = True
    ratio_lo: float = 0.0
    ratio_hi: float = 1.0
    cache_items: int = 64
    # "noisy_all" is the DATACLASS default on purpose: every other importer of this module
    # (eval_drafter_agreement.py) builds a DrafterSetConfig positionally/by keyword without this
    # field and must keep the pre-2026-09-08 entry list, split and example semantics.  `main()`
    # passes `--conditioning` (default clean_prefix) explicitly.
    conditioning: str = "noisy_all"

    def to_json(self) -> dict:
        return asdict(self)


def mask_blocks(prompt_ids: np.ndarray, gen_ids: np.ndarray, n_blocks: int, block_start: int,
                g: torch.Generator, *, mask_id: int = MASK_ID, block: int = BLOCK,
                ratio_lo: float = 0.0, ratio_hi: float = 1.0,
                force_ratio: Optional[float] = None) -> Example:
    """Build one training example from a clean (prompt, generation) pair.

    S = block_start + block * n_blocks.  Positions [0, P) hold the prompt (clean, no loss);
    [P, P + G) hold the MoE's tokens; the remaining slots of the last block ("pad", beyond the
    EOS) hold [MASK] and are never loss rows -- that is what the last block of a real decode
    looks like the moment the EOS is written.

    `force_ratio` (evaluation) replaces the per-block draw by a fixed ratio; 1.0 gives the
    all-[MASK] block the drafter actually meets at inference.
    """
    P_ = int(prompt_ids.shape[0])
    G = int(gen_ids.shape[0])
    S = int(block_start) + int(block) * int(n_blocks)
    assert block_start == (P_ // block) * block, f"block_start {block_start} != (P//{block})*{block}"
    assert 0 < G <= S - P_, f"{G} generated tokens do not fit in [{P_}, {S})"

    ids = torch.full((S,), int(mask_id), dtype=torch.long)
    ids[:P_] = torch.from_numpy(np.asarray(prompt_ids, np.int64))
    ids[P_:P_ + G] = torch.from_numpy(np.asarray(gen_ids, np.int64))
    clean = ids.clone()

    resp = torch.zeros(S, dtype=torch.bool)
    resp[P_:P_ + G] = True
    loss_mask = torch.zeros(S, dtype=torch.bool)
    weights = torch.zeros(S, dtype=torch.float32)
    ratios: List[float] = []

    for b in range(int(n_blocks)):
        lo = int(block_start) + b * int(block)
        sl = slice(lo, lo + int(block))
        here = resp[sl]
        if not bool(here.any()):                              # can only happen for an empty response
            ratios.append(float("nan"))
            continue
        p = float(force_ratio) if force_ratio is not None else \
            float(ratio_lo + (ratio_hi - ratio_lo) * float(torch.rand((), generator=g)))
        pick = (torch.rand(int(block), generator=g) < p) & here
        if not bool(pick.any()):                              # the objective wants >= 1 masked slot
            cand = torch.nonzero(here).flatten()
            j = int(cand[int(torch.randint(cand.numel(), (1,), generator=g))])
            pick[j] = True
        ids[sl] = torch.where(pick, torch.full_like(ids[sl], int(mask_id)), ids[sl])
        loss_mask[sl] = pick
        # 1/p_mask of the official estimator; the forced token keeps the block's drawn p, and a
        # p of 0 is floored at 1/block (one masked slot out of `block` is the realised ratio).
        weights[sl] = torch.where(pick, torch.full_like(weights[sl], 1.0 / max(p, 1.0 / block)),
                                  torch.zeros_like(weights[sl]))
        ratios.append(p)

    targets = torch.where(loss_mask, clean, torch.full_like(clean, IGNORE))
    meta = {"seq_len": S, "prompt_length": P_, "n_gen": G, "n_blocks": int(n_blocks),
            "n_loss": int(loss_mask.sum()), "n_pad": S - P_ - G,
            "mean_ratio": float(np.nanmean(ratios)) if ratios else float("nan")}
    return Example(input_ids=ids, targets=targets, loss_mask=loss_mask, weights=weights,
                   resp=resp, ratios=ratios, meta=meta)


def eligible_blocks(prompt_length: int, n_gen: int, n_blocks: int, block_start: int,
                    block: int = BLOCK) -> List[int]:
    """The response blocks of an item that can be the ACTIVE block of a clean_prefix example.

    A block is eligible iff (a) its sequence -- everything up to and including it -- is longer
    than ONE block, because `hidden_states` refuses a single-block sequence (the block-tril would
    be all ones), and (b) it holds at least one response slot.  Only an item with
    `block_start == 0` (a prompt shorter than `block`) ever loses a block, and it loses only
    block 0.
    """
    P_, G = int(prompt_length), int(n_gen)
    out: List[int] = []
    for b in range(int(n_blocks)):
        lo = int(block_start) + b * int(block)
        hi = lo + int(block)
        if hi <= int(block):                       # a one-block sequence: hidden_states refuses it
            continue
        if min(hi, P_ + G) <= max(lo, P_):         # no response slot in this block
            continue
        out.append(b)
    return out


def mask_active_block(prompt_ids: np.ndarray, gen_ids: np.ndarray, n_blocks: int, block_start: int,
                      active: int, g: torch.Generator, *, mask_id: int = MASK_ID,
                      block: int = BLOCK, ratio_lo: float = 0.0, ratio_hi: float = 1.0,
                      force_ratio: Optional[float] = None) -> Example:
    """One CLEAN-PREFIX example: the inference state, as `dataset.StateSampler` builds it.

    S = block_start + block * (active + 1) -- the sequence stops right after the active block,
    because under the block-causal mask no later block is ever attended by it, so computing them
    would change nothing and only cost time (`dataset.py` truncates at `lo + bl` for the same
    reason).  Positions [0, P) hold the prompt and [P, P + G) the teacher's tokens, both CLEAN;
    inside the active block every response slot is [MASK]ed independently with probability
    p ~ U(ratio_lo, ratio_hi) (>= 1 forced), and the slots of a last block that lie beyond the
    response ("pad", past the EOS) hold [MASK] and are never loss rows.  Loss rows = the masked
    response slots of the active block, nothing else.
    """
    P_ = int(prompt_ids.shape[0])
    G = int(gen_ids.shape[0])
    S_item = int(block_start) + int(block) * int(n_blocks)
    assert block_start == (P_ // block) * block, f"block_start {block_start} != (P//{block})*{block}"
    assert 0 < G <= S_item - P_, f"{G} generated tokens do not fit in [{P_}, {S_item})"
    assert 0 <= int(active) < int(n_blocks), f"active block {active} outside [0, {n_blocks})"

    S = int(block_start) + int(block) * (int(active) + 1)
    assert S % block == 0, f"S={S} is not on the {block}-grid"
    assert S > block, (f"active block {active} of an item with block_start={block_start} makes a "
                       f"single-block sequence, which hidden_states refuses; use "
                       f"`eligible_blocks` to draw the active block")

    ids = torch.full((S,), int(mask_id), dtype=torch.long)
    n_prompt = min(P_, S)
    ids[:n_prompt] = torch.from_numpy(np.asarray(prompt_ids, np.int64)[:n_prompt])
    n_gen_kept = max(0, min(P_ + G, S) - P_)
    if n_gen_kept:
        ids[P_:P_ + n_gen_kept] = torch.from_numpy(np.asarray(gen_ids, np.int64)[:n_gen_kept])
    clean = ids.clone()

    resp = torch.zeros(S, dtype=torch.bool)
    resp[P_:P_ + n_gen_kept] = True
    loss_mask = torch.zeros(S, dtype=torch.bool)
    weights = torch.zeros(S, dtype=torch.float32)

    lo = int(block_start) + int(active) * int(block)
    sl = slice(lo, lo + int(block))
    here = resp[sl]
    assert bool(here.any()), f"active block {active} holds no response slot"
    p = float(force_ratio) if force_ratio is not None else \
        float(ratio_lo + (ratio_hi - ratio_lo) * float(torch.rand((), generator=g)))
    pick = (torch.rand(int(block), generator=g) < p) & here
    if not bool(pick.any()):                                  # the objective wants >= 1 masked slot
        cand = torch.nonzero(here).flatten()
        j = int(cand[int(torch.randint(cand.numel(), (1,), generator=g))])
        pick[j] = True
    ids[sl] = torch.where(pick, torch.full_like(ids[sl], int(mask_id)), ids[sl])
    loss_mask[sl] = pick
    weights[sl] = torch.where(pick, torch.full_like(weights[sl], 1.0 / max(p, 1.0 / block)),
                              torch.zeros_like(weights[sl]))

    targets = torch.where(loss_mask, clean, torch.full_like(clean, IGNORE))
    meta = {"seq_len": S, "prompt_length": P_, "n_gen": n_gen_kept, "n_gen_item": G,
            "n_blocks": int(active) + 1, "n_blocks_item": int(n_blocks),
            "active_block": int(active), "conditioning": "clean_prefix",
            "n_loss": int(loss_mask.sum()), "n_pad": S - P_ - n_gen_kept,
            "mean_ratio": float(p)}
    return Example(input_ids=ids, targets=targets, loss_mask=loss_mask, weights=weights,
                   resp=resp, ratios=[p], meta=meta)


class DrafterSet:
    """Index-backed set of items from `build_drafter_set.py`.

    conditioning="noisy_all"     one example = one whole item, every block noised (the old bug);
    conditioning="clean_prefix"  one example = one (item, active block) pair with a clean prefix.
    """

    def __init__(self, cfg: DrafterSetConfig, entries: Optional[List[dict]] = None,
                 root: Optional[str] = None):
        self.cfg = cfg
        self.root = root if root is not None else os.path.dirname(os.path.abspath(cfg.index_path))
        if entries is None:
            entries = [json.loads(l) for l in open(cfg.index_path) if l.strip()]
        assert cfg.conditioning in CONDITIONINGS, cfg.conditioning
        self.entries = [e for e in entries if self._keep(e)]
        self._cache: Dict[int, dict] = {}
        self._warned: set = set()          # items already reported as having no eligible block

    def _keep(self, e: dict) -> bool:
        if self.cfg.only_correct and not e.get("correct"):
            return False
        if int(e.get("n_blocks", 0)) < int(self.cfg.min_blocks):
            return False
        if self.cfg.max_seq and int(e.get("seq_len", 0)) > int(self.cfg.max_seq):
            return False
        return True

    def _item(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        path = self.entries[i]["file"]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        d = dict(np.load(path).items())
        if len(self._cache) >= max(1, self.cfg.cache_items):
            self._cache.pop(next(iter(self._cache)))
        self._cache[i] = d
        return d

    def arrays(self, k: int) -> dict:
        """The raw npz arrays of item `k` (prompt_ids, gen_ids, n_blocks, block_start, ...)."""
        return self._item(k)

    def __len__(self) -> int:
        return len(self.entries)

    def epoch_order(self, epoch: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(_seed(self.cfg.seed, "order", epoch))
        return torch.randperm(len(self.entries), generator=g)

    def item_blocks(self, k: int) -> List[int]:
        """The response blocks of item `k` that can be a clean_prefix ACTIVE block."""
        it = self._item(k)
        return eligible_blocks(int(np.asarray(it["prompt_ids"]).shape[0]),
                               int(np.asarray(it["gen_ids"]).shape[0]),
                               int(it["n_blocks"]), int(it["block_start"]), self.cfg.block)

    def eval_blocks(self, k: int, n: Optional[int] = None) -> List[int]:
        """A deterministic subset of `item_blocks(k)`, at most `n` of them (None/<=0 = all)."""
        blocks = self.item_blocks(k)
        if n is None or int(n) <= 0 or len(blocks) <= int(n):
            return blocks
        e = self.entries[k]
        g = torch.Generator().manual_seed(
            _seed(self.cfg.seed, "eval_blocks", e["split"], e["idx"], int(n)))
        pick = torch.randperm(len(blocks), generator=g)[:int(n)].tolist()
        return [blocks[i] for i in sorted(pick)]

    def global_example(self, step_index: int, epoch: int = 0, **kw) -> Example:
        order = self.epoch_order(epoch)
        n = len(order)
        for j in range(n):                      # skip (never-seen in practice) unusable items
            k = int(order[(step_index + j) % n])
            if self.cfg.conditioning == "clean_prefix" and not self.item_blocks(k):
                if k not in self._warned:
                    e = self.entries[k]
                    print(f"[skip] item {k} ({e['split']}/{e['idx']}) has no eligible active "
                          f"block (block_start=0 and a single 32-block response); "
                          f"clean_prefix cannot build an example for it", flush=True)
                    self._warned.add(k)
                continue
            return self.example(k, epoch=epoch, **kw)
        raise AssertionError("no item in this set has an eligible active block")

    def example(self, k: int, epoch: int = 0, force_ratio: Optional[float] = None,
                tag: str = "train", active: Optional[int] = None) -> Example:
        """One example of item `k`.

        clean_prefix: `active` names the active block; when it is None the block is drawn
        UNIFORMLY over `item_blocks(k)` from the example's own generator, i.e. a fresh draw on
        every visit (the epoch is part of the seed) that is reproducible for a given
        (seed, epoch, tag, item).
        """
        e = self.entries[k]
        it = self._item(k)
        g = torch.Generator().manual_seed(
            _seed(self.cfg.seed, epoch, tag, e["split"], e["idx"]))
        if self.cfg.conditioning == "clean_prefix":
            b = active
            if b is None:
                blocks = self.item_blocks(k)
                assert blocks, (f"item {k} ({e['split']}/{e['idx']}) has no eligible active block "
                                f"(block_start={int(it['block_start'])}, "
                                f"n_blocks={int(it['n_blocks'])})")
                b = int(blocks[int(torch.randint(len(blocks), (1,), generator=g))])
            ex = mask_active_block(np.asarray(it["prompt_ids"]), np.asarray(it["gen_ids"]),
                                   int(it["n_blocks"]), int(it["block_start"]), int(b), g,
                                   mask_id=self.cfg.mask_id, block=self.cfg.block,
                                   ratio_lo=self.cfg.ratio_lo, ratio_hi=self.cfg.ratio_hi,
                                   force_ratio=force_ratio)
        else:
            assert active is None, "`active` is a clean_prefix-only argument"
            ex = mask_blocks(np.asarray(it["prompt_ids"]), np.asarray(it["gen_ids"]),
                             int(it["n_blocks"]), int(it["block_start"]), g,
                             mask_id=self.cfg.mask_id, block=self.cfg.block,
                             ratio_lo=self.cfg.ratio_lo, ratio_hi=self.cfg.ratio_hi,
                             force_ratio=force_ratio)
            ex.meta["conditioning"] = "noisy_all"
        ex.meta.update({"item": int(k), "idx": int(e["idx"]), "split": e["split"],
                        "epoch": int(epoch), "correct": e.get("correct")})
        return ex

    def split_holdout(self, n_items: int) -> Tuple["DrafterSet", "DrafterSet"]:
        """Deterministic ITEM-level split; a held-out item never appears in training."""
        g = torch.Generator().manual_seed(_seed(self.cfg.seed, "holdout", n_items))
        perm = torch.randperm(len(self.entries), generator=g).tolist()
        hold = set(perm[:max(0, int(n_items))])
        tr = [e for j, e in enumerate(self.entries) if j not in hold]
        ev = [e for j, e in enumerate(self.entries) if j in hold]
        return DrafterSet(self.cfg, tr, self.root), DrafterSet(self.cfg, ev, self.root)

    def summary(self) -> dict:
        return {"items": len(self.entries),
                "blocks": sum(int(e["n_blocks"]) for e in self.entries),
                "gen_tokens": sum(int(e["n_gen"]) for e in self.entries)}


# ===========================================================================
# forward / loss
# ===========================================================================
def enable_grad_checkpointing(model) -> int:
    """Re-run each decoder layer during backward instead of keeping its activations.

    The remote `SDARModel.forward` has no checkpointing branch at all (modeling_sdar.py:528-545
    loops over the layers directly), so the layers are patched here.  The patch replaces the
    bound `forward` rather than wrapping the layer in a new Module on purpose: a wrapper would
    insert itself into the parameter names (`model.layers.0.layer.q_proj.weight`) and every
    checkpoint would then be unloadable by a stock `from_pretrained`.
    """
    n = 0
    for layer in model.model.layers:
        if getattr(layer, "_ckpt_patched", False):
            continue
        orig = layer.forward

        def fwd(*a, _orig=orig, **kw):
            if not torch.is_grad_enabled():
                return _orig(*a, **kw)
            return checkpoint(_orig, *a, use_reentrant=False, **kw)

        layer.forward = fwd
        layer._ckpt_patched = True
        n += 1
    return n


def hidden_states(model, input_ids: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Last hidden state [S, H] of the block-causal forward over the whole sequence.

    `model.model` is called directly, never `SDARForCausalLM.forward`: that wrapper drops the
    logits entirely when `config.fuse_cross_entropy and self.training` (modeling_sdar.py:838-843
    sets `logits = None`), and its `labels=` branch would then dereference the None.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, input_ids.shape
    S = int(input_ids.shape[1])
    assert S % block == 0, f"S={S} is not on the {block}-grid"
    assert S > block, ("a single-block sequence makes the block-tril all ones, which sends the "
                       "remote attention down its flash_attn fast path (modeling_sdar.py:275-287)")
    attn = block_tril_bool(S, input_ids.device, block)                     # [1,1,S,S] bool
    pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
    out = model.model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                      use_cache=False, store_kv=False)
    return out.last_hidden_state[0]


def forward_loss(model, ex: Example, device, *, weighting: str = "mean",
                 autocast: bool = True, block: int = BLOCK):
    """CE at the masked slots only, computed HERE from the logits.  Returns (loss, stats)."""
    assert weighting in WEIGHTINGS, weighting
    ids = ex.input_ids.unsqueeze(0).to(device, non_blocking=True)
    rows = ex.loss_mask.to(device)
    targets = ex.targets.to(device)[rows]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=bool(autocast) and device.type == "cuda"):
        h = hidden_states(model, ids, block)
    # `lm_head` runs OUTSIDE the autocast region, in fp32, as `train.py` does: with only the
    # masked rows selected it is a small matmul, and an fp32 [n_loss, 151936] tensor is cheaper
    # than a bf16 one plus the fp32 copy `active_logits` would have to make for the CE anyway.
    logits = active_logits(model, h, rows)                                 # [n_loss, V] fp32
    ce_row = F.cross_entropy(logits, targets, reduction="none")
    if weighting == "inv_p":
        w = ex.weights.to(device)[rows]
        loss = (ce_row * w).sum() / w.sum().clamp_min(1e-8)
    else:
        loss = ce_row.mean()
    with torch.no_grad():
        correct = (logits.argmax(-1) == targets)
        stats = {"ce": ce_row.mean().detach(), "n_rows": torch.tensor(float(targets.numel()),
                                                                      device=logits.device),
                 "acc": correct.float().mean(), "n_correct": correct.sum().float(),
                 "top1": F.softmax(logits, dim=-1).max(dim=-1).values.mean()}
    return loss, stats


@torch.no_grad()
def evaluate(model, ds: DrafterSet, n_items: int, device, *, weighting: str = "mean",
             regimes: Tuple[str, ...] = ("sampled", "allmask"), block: int = BLOCK,
             autocast: bool = True, blocks_per_item: int = 4) -> Dict[str, float]:
    """Held-out masked-token accuracy, in the CONDITION the set is configured for.

    "sampled"  the training distribution (mask ratio ~ U(0,1)), fixed by the item's own seed so
               the number is comparable across steps;
    "allmask"  every response slot of the block masked -- the state the drafter actually starts a
               block from at inference, so this tracks the agreement the
               `eval_drafter_agreement.py` replay measures, one forward instead of eight.

    clean_prefix (the fix): one example per (held-out item, response block), with the CLEAN
    teacher prefix and only that block masked -- exactly the inference condition.  At most
    `blocks_per_item` blocks per item (a deterministic subset, `DrafterSet.eval_blocks`) so the
    cost stays bounded.  noisy_all: the old whole-item example, i.e. the noised-prefix condition
    the 2026-09-07 run reported.

    Keys are unchanged in both modes: `<regime>_acc`, `_ce`, `_top1`, `_rows`, `_items`
    (+ `_blocks` in clean_prefix).
    """
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    n = min(int(n_items), len(ds))
    clean = ds.cfg.conditioning == "clean_prefix"
    for regime in regimes:
        tot = {"rows": 0.0, "correct": 0.0, "ce": 0.0, "top1": 0.0, "items": 0.0, "blocks": 0.0}
        for k in range(n):
            blocks = ds.eval_blocks(k, blocks_per_item) if clean else [None]
            if not blocks:
                continue
            tot["items"] += 1
            for b in blocks:
                ex = ds.example(k, epoch=0, tag=f"eval_{regime}" if b is None else
                                f"eval_{regime}_b{b}", active=b,
                                force_ratio=1.0 if regime == "allmask" else None)
                _, st = forward_loss(model, ex, device, weighting=weighting, autocast=autocast,
                                     block=block)
                r = float(st["n_rows"])
                tot["rows"] += r; tot["blocks"] += 1
                tot["correct"] += float(st["n_correct"])
                tot["ce"] += float(st["ce"]) * r
                tot["top1"] += float(st["top1"]) * r
        if tot["rows"]:
            out[f"{regime}_acc"] = tot["correct"] / tot["rows"]
            out[f"{regime}_ce"] = tot["ce"] / tot["rows"]
            out[f"{regime}_top1"] = tot["top1"] / tot["rows"]
            out[f"{regime}_rows"] = tot["rows"]
            out[f"{regime}_items"] = tot["items"]
            if clean:
                out[f"{regime}_blocks"] = tot["blocks"]
    if was_training:
        model.train()
    return out


@torch.no_grad()
def agreement(model, ds: DrafterSet, n_items: int, device, *, setting: str = "s8",
              autocast: bool = True, block: int = BLOCK) -> Dict[str, float]:
    """Slot agreement of `model` with the MoE's recorded blocks, on the first `n_items` of `ds`.

    Calls `eval_drafter_agreement.item_agreement` -- the SAME function the standalone script uses,
    not a reimplementation -- so the in-loop number is directly comparable to
    `results/metrics/draftalign/drafter_agreement_*.json`.  The model is put in eval mode for the
    replay (the remote `SDARForCausalLM.forward` returns `logits=None` while training when
    `fuse_cross_entropy` is set, and `SeqCache` calls exactly that forward) and restored
    afterwards.  Imported lazily: `eval_drafter_agreement` imports this module at import time.
    """
    from draftalign.eval_drafter_agreement import item_agreement                # noqa: E402

    was_training = model.training
    model.eval()
    n = min(int(n_items), len(ds))
    agree = slots = blocks = 0
    per_item: List[float] = []
    t0 = time.time()
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(autocast) and device.type == "cuda"):
            for k in range(n):
                it = ds.arrays(k)
                item = {key: it[key] for key in ("prompt_ids", "gen_ids", "block_start",
                                                 "prompt_length")}
                item["idx"] = ds.entries[k]["idx"]
                r = item_agreement(model, item, device, setting, mask_id=ds.cfg.mask_id,
                                   block=block)
                agree += int(r["agree"]); slots += int(r["slots"]); blocks += int(r["n_blocks"])
                per_item.append(float(r["agreement"]))
    finally:
        if was_training:
            model.train()
    return {"agree": float(agree), "slots": float(slots), "blocks": float(blocks),
            "items": float(n), "agreement": (agree / slots) if slots else float("nan"),
            "item_mean": float(np.mean(per_item)) if per_item else float("nan"),
            "seconds": round(time.time() - t0, 1)}


# ===========================================================================
# distributed
# ===========================================================================
def dist_init() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local)
    return {"rank": rank, "world": world, "local_rank": local}


def all_reduce_grads(params: List[torch.nn.Parameter], world: int, bucket_mb: int = 256) -> None:
    """Average gradients across ranks in <= `bucket_mb` buckets.

    `train.py` flattens ALL adapter gradients into one buffer; that is fine for 100 M
    parameters and would be a 6.8 GB temporary here, so the flatten is bucketed.
    """
    if world <= 1:
        return
    cap = int(bucket_mb) * 1024 * 1024
    bucket: List[torch.Tensor] = []
    nbytes = 0

    def flush():
        nonlocal bucket, nbytes
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        for g, s in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
            g.copy_(s)
        bucket, nbytes = [], 0

    for p in params:
        if p.grad is None:                         # a missing grad on one rank would desync
            p.grad = torch.zeros_like(p)
        if nbytes + p.grad.numel() * p.grad.element_size() > cap and bucket:
            flush()
        bucket.append(p.grad)
        nbytes += p.grad.numel() * p.grad.element_size()
    flush()


def reduce_stats(d: Dict[str, float], world: int, device) -> Dict[str, float]:
    if world <= 1:
        return d
    keys = sorted(d)
    t = torch.tensor([d[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world
    return {k: float(v) for k, v in zip(keys, t.tolist())}


def lr_scale(step: int, warmup: int, total: int, min_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_frac` of the peak (same shape as train.py)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))


# ===========================================================================
# checkpoints (full HF directories)
# ===========================================================================
CKPT_RE = re.compile(r"^step(\d{6})$")
EXTRA_SUFFIXES = (".py", ".jinja", ".json", ".txt")


def checkpoint_dirs(out: str) -> List[str]:
    if not os.path.isdir(out):
        return []
    got = [(int(m.group(1)), n) for n in os.listdir(out)
           for m in [CKPT_RE.match(n)] if m and os.path.isdir(os.path.join(out, n))]
    return [n for _, n in sorted(got)]


def marker(out: str, name: str) -> Optional[str]:
    """The step directory named by the `latest` / `best` marker file, if it exists."""
    fp = os.path.join(out, name)
    if not os.path.isfile(fp):
        return None
    val = open(fp).read().strip()
    return val or None


def write_marker(out: str, name: str, step_dir: str) -> str:
    """Write a marker file (`best`, like `latest`) naming a step directory."""
    with open(os.path.join(out, name), "w") as fh:
        fh.write(os.path.basename(step_dir))
    return os.path.basename(step_dir)


def prune_checkpoints(out: str, keep: int) -> List[str]:
    """Delete all but the `keep` newest step dirs; never the one named in `latest` or in `best`."""
    if keep is None or int(keep) <= 0:
        return []
    names = checkpoint_dirs(out)
    protected = {marker(out, "latest"), marker(out, "best")} - {None}
    removed = []
    for n in names[:max(0, len(names) - int(keep))]:
        if n in protected:
            continue
        shutil.rmtree(os.path.join(out, n), ignore_errors=True)
        removed.append(n)
    return removed


def snapshot_dir(model_key: str) -> Optional[str]:
    """The local HF snapshot of MODELS[model_key] (source of the remote-code / tokenizer files)."""
    m = P.MODELS[model_key]
    root = m["cache_dir"] or os.environ.get("HF_HOME", "")
    pat = os.path.join(root, "models--" + m["repo"].replace("/", "--"), "snapshots", m["revision"])
    return pat if os.path.isdir(pat) else None


def copy_model_files(src: Optional[str], dst: str, overwrite: bool = False) -> List[str]:
    """Copy the remote-code modules / tokenizer / template files a `from_pretrained` needs."""
    if not src or not os.path.isdir(src):
        return []
    done = []
    for path in sorted(glob.glob(os.path.join(src, "*"))):
        name = os.path.basename(path)
        if not name.endswith(EXTRA_SUFFIXES) or not os.path.isfile(path):
            continue
        target = os.path.join(dst, name)
        if os.path.exists(target) and not overwrite:
            continue
        shutil.copyfile(path, target)
        done.append(name)
    return done


def _localize_auto_map(path: str) -> None:
    """`save_pretrained` writes auto_map entries as `<hub repo>--module.Class`; a plain directory
    then cannot be loaded offline (transformers tries the hub).  Rewrite them to `module.Class`."""
    for name in ("config.json", "tokenizer_config.json"):
        fp = os.path.join(path, name)
        if not os.path.isfile(fp):
            continue
        d = json.load(open(fp)); am = d.get("auto_map")
        if not isinstance(am, dict):
            continue
        for k, v in list(am.items()):
            if isinstance(v, str):
                am[k] = v.split("--", 1)[1] if "--" in v else v
            elif isinstance(v, list):
                am[k] = [(x.split("--", 1)[1] if isinstance(x, str) and "--" in x else x) for x in v]
        json.dump(d, open(fp, "w"), indent=2, sort_keys=True)


def save_checkpoint(model, opt, out: str, step: int, *, tok=None, src: Optional[str] = None,
                    keep: int = 2, save_dtype: Optional[torch.dtype] = None,
                    extra: Optional[dict] = None) -> str:
    """A full `save_pretrained` directory + `trainer_state.pt`, then prune."""
    d = os.path.join(out, f"step{step:06d}")
    os.makedirs(d, exist_ok=True)
    sd = model.state_dict()
    if save_dtype is not None:
        sd = {k: (v.to(save_dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    model.save_pretrained(d, state_dict=sd, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(d)
    copy_model_files(src, d)                      # configuration_sdar.py / modeling_sdar.py / ...
    _localize_auto_map(d)                         # drop the `<hub repo>--` prefix so the dir loads offline
    torch.save({"step": int(step), "optimizer": opt.state_dict(),
                "extra": dict(extra or {})}, os.path.join(d, "trainer_state.pt"))
    with open(os.path.join(out, "latest"), "w") as fh:
        fh.write(os.path.basename(d))
    gone = prune_checkpoints(out, keep)           # only AFTER this save succeeded
    print(f"[save] {d}" + (f" (pruned {', '.join(gone)})" if gone else ""), flush=True)
    return d


def resolve_checkpoint(path: str) -> str:
    """`<out>` with a `latest` marker -> the step directory it names; otherwise `path` itself."""
    lp = os.path.join(path, "latest")
    if os.path.isfile(lp):
        name = open(lp).read().strip()
        if name and os.path.isdir(os.path.join(path, name)):
            return os.path.join(path, name)
    return path


def load_checkpoint(model, opt, path: str, device) -> int:
    """Load the weights of a checkpoint DIRECTORY into `model` (in place) + the optimiser state."""
    path = resolve_checkpoint(path)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if files:
        from safetensors.torch import load_file
        sd: Dict[str, torch.Tensor] = {}
        for f in files:
            sd.update(load_file(f, device="cpu"))
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu",
                        weights_only=True)
    ref = model.state_dict()
    sd = {k: v.to(dtype=ref[k].dtype) for k, v in sd.items() if k in ref}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if not m.endswith(".inv_freq")], f"missing weights: {missing[:5]}"
    st = torch.load(os.path.join(path, "trainer_state.pt"), map_location=device, weights_only=False)
    if opt is not None:
        opt.load_state_dict(st["optimizer"])
    return int(st["step"])


# ===========================================================================
# cli
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="build_drafter_set index.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="d32", help="research_protocol.MODELS key (the dense drafter)")
    ap.add_argument("--steps", type=int, default=2000, help="optimiser steps")
    ap.add_argument("--grad-accum", type=int, default=8, help="sequences per rank per step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--betas", default="0.9,0.95")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=1200)
    ap.add_argument("--min-blocks", type=int, default=2)
    ap.add_argument("--only-correct", dest="only_correct", action="store_true", default=True)
    ap.add_argument("--no-only-correct", dest="only_correct", action="store_false")
    ap.add_argument("--conditioning", choices=CONDITIONINGS, default="clean_prefix",
                    help="clean_prefix (default, the fix): ONE active block per example, the "
                         "prefix is the teacher's clean tokens, loss on that block only -- the "
                         "inference condition.  noisy_all: every block noised in one sequence, "
                         "the pre-2026-09-08 behaviour the 7 Sep run used (kept for provenance)")
    ap.add_argument("--mask-weighting", choices=WEIGHTINGS, default="mean")
    ap.add_argument("--ratio-lo", type=float, default=0.0)
    ap.add_argument("--ratio-hi", type=float, default=1.0)
    ap.add_argument("--param-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="fp32 = master weights + bf16 autocast (recommended); bf16 loses most "
                         "1e-5 updates to rounding")
    ap.add_argument("--no-autocast", dest="autocast", action="store_false", default=True)
    ap.add_argument("--grad-checkpointing", action="store_true", default=False)
    ap.add_argument("--holdout-items", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-items", type=int, default=64)
    ap.add_argument("--eval-blocks-per-item", type=int, default=4,
                    help="clean_prefix only: how many response blocks of each held-out item the "
                         "eval scores (a deterministic subset; <=0 = all).  Bounds the eval cost, "
                         "which is one forward per (item, block, regime)")
    ap.add_argument("--agree-every", type=int, default=None,
                    help="steps between in-loop agreement evals (default: --eval-every); 0 = off")
    ap.add_argument("--agree-items", type=int, default=50,
                    help="held-out items per agreement eval (the same split as --holdout-items)")
    ap.add_argument("--agree-setting", default="s8",
                    help="drafter.SETTINGS key replayed by eval_drafter_agreement.item_agreement")
    ap.add_argument("--agree-stock", dest="agree_stock", action="store_true", default=True,
                    help="measure the STOCK drafter's agreement on the same items once at startup")
    ap.add_argument("--no-agree-stock", dest="agree_stock", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=2,
                    help="stop after this many consecutive agreement evals without a new best "
                         "agree_ckpt (0 = never stop early)")
    ap.add_argument("--min-steps", type=int, default=200,
                    help="no early stop before this step")
    ap.add_argument("--disagree-weight", type=float, default=1.0,
                    help="TODO / NOT IMPLEMENTED (2026-09-08).  Intent: weight the loss rows where "
                         "the STOCK drafter's token differs from the target by W > 1.  The stock "
                         "draft tokens live in logs/draftalign/datagen/<cell>/traces/<split>/"
                         "<idx>.npz as draft_tokens [blocks, 32], but the drafterset_union index "
                         "cannot address them safely: 1622 of its 8638 entries share an item file "
                         "with another source (all four sources wrote items/<split>__<idx>.npz "
                         "into ONE directory), so entry.source/idx does not identify the npz that "
                         "is actually loaded and the weights would be silently attached to another "
                         "item's draft.  Any value != 1.0 raises")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--save-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="dtype of the saved weights; fp32 makes a resume bit-exact, bf16 halves "
                         "the checkpoint but should then be used for inference only")
    ap.add_argument("--resume", default=None, help="checkpoint directory (or --out with a `latest`)")
    ap.add_argument("--bucket-mb", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true", help="build everything, load no model")
    return ap


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.agree_every is None:
        args.agree_every = args.eval_every
    assert float(args.disagree_weight) == 1.0, (
        "--disagree-weight is documented as a TODO and is not implemented; see its help text "
        "(the union index cannot address the datagen draft_tokens unambiguously)")
    info = dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = safe_out(args.out)
    os.makedirs(out, exist_ok=True)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"world_size": world, "model_revision": P.MODELS[args.model]["revision"],
                "trainer": "draftalign_train_drafter_v0",
                "global_batch": world * args.grad_accum})
    chash = config_hash(cfg)
    if rank == 0:
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump({"config": cfg, "config_hash": chash}, fh, indent=2, sort_keys=True)

    # -- data ------------------------------------------------------------
    scfg = DrafterSetConfig(index_path=args.index, seed=args.seed, mask_id=MASK_ID,
                            max_seq=args.max_seq, min_blocks=args.min_blocks,
                            only_correct=args.only_correct, ratio_lo=args.ratio_lo,
                            ratio_hi=args.ratio_hi, conditioning=args.conditioning)
    full = DrafterSet(scfg)
    train_s, eval_s = full.split_holdout(args.holdout_items)
    per_epoch = max(1, len(train_s))
    if rank == 0:
        seqs = args.steps * args.grad_accum * world
        print(f"[data] {full.summary()} -> train {len(train_s)} / holdout {len(eval_s)} items; "
              f"{seqs} sequence visits = {seqs / per_epoch:.2f} epochs "
              f"[conditioning={args.conditioning}]", flush=True)
    if args.dry_run:
        return
    launch = LaunchLog(f"draftalign/train_drafter/{os.path.basename(out)}/rank{rank}", cfg)  # noqa: F841
    wb = wandb_log.init("drafter_" + os.path.basename(out), cfg, out, group="draftalign-round2b",
                        job_type="drafter-sft", tags=("drafter", args.model), rank=rank,
                        notes=(f"resume from {args.resume}" if args.resume else None))

    # -- model -----------------------------------------------------------
    torch.manual_seed(args.seed + rank)
    tok, model = load_sdar(args.model, str(device), dtype=DTYPES[args.param_dtype])
    # `model.config` is deliberately NOT mutated: the checkpoint's config.json must stay identical
    # to the stock one.  `hidden_states` passes use_cache=False / store_kv=False explicitly, and
    # the `fuse_cross_entropy` branch lives in `SDARForCausalLM.forward`, which we never call.
    model.train()
    if args.grad_checkpointing:
        n = enable_grad_checkpointing(model)
        if rank == 0:
            print(f"[model] gradient checkpointing on {n} layers", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        n_par = sum(p.numel() for p in params)
        print(f"[model] {args.model} rev {P.MODELS[args.model]['revision'][:8]}, "
              f"{n_par / 1e9:.3f} B trainable params in {args.param_dtype}", flush=True)

    b1, b2 = (float(x) for x in args.betas.split(","))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(b1, b2), eps=1e-8,
                            weight_decay=args.weight_decay)
    src = snapshot_dir(args.model)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, opt, safe_out(args.resume), device)
        if rank == 0:
            print(f"[resume] from step {start_step}", flush=True)

    # -- the stock drafter's agreement on the SAME held-out items, once -----
    do_agree = bool(args.agree_every) and int(args.agree_items) > 0 and len(eval_s) > 0
    agree_stock = float("nan")
    if do_agree and args.agree_stock:
        t0 = time.time()
        _, stock = load_sdar(args.model, str(device))     # bf16, exactly as eval_drafter_agreement loads it
        st = agreement(stock, eval_s, args.agree_items, device, setting=args.agree_setting,
                       autocast=args.autocast)
        agree_stock = float(st["agreement"])
        del stock
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"[agree stock] {args.model} {args.agree_setting} agreement {agree_stock:.4f} "
                  f"over {st['slots']:.0f} slots / {st['items']:.0f} items "
                  f"({time.time() - t0:.0f}s)", flush=True)

    logf = open(os.path.join(out, f"train.rank{rank}.jsonl"), "a")
    t_last, n_tok, agg = time.time(), 0, {}
    best_agree, best_step, bad_evals, stop = -1.0, -1, 0, False
    last_step = args.steps

    for step in range(start_step, args.steps):
        last_step = step + 1
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_scale(step, args.warmup, args.steps, args.min_lr_frac)
        opt.zero_grad(set_to_none=True)
        step_stats: Dict[str, float] = {}
        for j in range(args.grad_accum):
            gi = (step * args.grad_accum + j) * world + rank
            ex = train_s.global_example(gi, epoch=gi // per_epoch)
            loss, stats = forward_loss(model, ex, device, weighting=args.mask_weighting,
                                       autocast=args.autocast)
            (loss / args.grad_accum).backward()
            n_tok += int(ex.meta["seq_len"])
            for k, v in stats.items():
                step_stats[k] = step_stats.get(k, 0.0) + float(v) / args.grad_accum
            step_stats["loss"] = step_stats.get("loss", 0.0) + float(loss.detach()) / args.grad_accum
            step_stats["mean_ratio"] = step_stats.get("mean_ratio", 0.0) + \
                float(ex.meta["mean_ratio"]) / args.grad_accum

        all_reduce_grads(params, world, args.bucket_mb)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        step_stats["grad_norm"] = gnorm
        for k, v in step_stats.items():
            agg[k] = agg.get(k, 0.0) + v

        if args.log_every and (step + 1) % args.log_every == 0:
            row = {k: v / args.log_every for k, v in agg.items()}
            row = reduce_stats(row, world, device)
            dt = time.time() - t_last
            row.update({"step": step + 1, "lr": opt.param_groups[0]["lr"],
                        "epoch": ((step + 1) * args.grad_accum * world) / per_epoch,
                        "tokens_per_s": n_tok / max(dt, 1e-9), "sec_per_step": dt / args.log_every,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            if rank == 0:
                print(f"[step {step + 1}/{args.steps}] loss {row['loss']:.4f} ce {row['ce']:.4f} "
                      f"acc {row['acc']:.3f} top1 {row['top1']:.3f} rows {row['n_rows']:.0f} "
                      f"gnorm {row['grad_norm']:.2f} lr {row['lr']:.2e} "
                      f"{row['tokens_per_s']:.0f} tok/s", flush=True)
            logf.write(json.dumps(row) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(row, "train"), step=step + 1)
            agg, n_tok, t_last = {}, 0, time.time()

        ev: Dict[str, float] = {}
        did_eval = bool(args.eval_every and (step + 1) % args.eval_every == 0)
        did_agree = bool(do_agree and (step + 1) % args.agree_every == 0)
        new_best = False
        if did_eval:
            ev.update(evaluate(model, eval_s, args.eval_items, device,
                               weighting=args.mask_weighting, autocast=args.autocast,
                               blocks_per_item=args.eval_blocks_per_item))
        if did_agree:
            ag = agreement(model, eval_s, args.agree_items, device, setting=args.agree_setting,
                           autocast=args.autocast)
            ev.update({"agree_ckpt": float(ag["agreement"]), "agree_stock": agree_stock,
                       "agree_delta": float(ag["agreement"]) - agree_stock,
                       "agree_item_mean": float(ag["item_mean"]), "agree_items": ag["items"],
                       "agree_slots": ag["slots"], "agree_blocks": ag["blocks"],
                       "agree_seconds": ag["seconds"]})
        if ev:
            ev = reduce_stats(ev, world, device)
            ev.update({"step": step + 1, "event": "eval"})
            if rank == 0:
                print(f"[eval {step + 1}] " + " ".join(f"{k}={v:.4f}" for k, v in sorted(ev.items())
                                                       if isinstance(v, float)), flush=True)
            logf.write(json.dumps(ev) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(ev, "eval"), step=step + 1)

        if did_agree:                           # best-agreement bookkeeping + early stop
            a = float(ev["agree_ckpt"])
            if a > best_agree:
                best_agree, best_step, bad_evals, new_best = a, step + 1, 0, True
            else:
                bad_evals += 1
            if rank == 0 and args.early_stop_patience and (step + 1) >= args.min_steps \
                    and bad_evals >= int(args.early_stop_patience):
                stop = True
            if world > 1:                       # rank 0 decides, everybody obeys
                flag = torch.tensor([1.0 if stop else 0.0], device=device)
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())

        saving = bool(args.save_every and (step + 1) % args.save_every == 0) or new_best or stop
        if saving and rank == 0:
            d = save_checkpoint(model, opt, out, step + 1, tok=tok, src=src,
                                keep=args.keep_checkpoints, save_dtype=DTYPES[args.save_dtype],
                                extra={"config_hash": chash, "world_size": world,
                                       "agree_ckpt": (float(ev["agree_ckpt"]) if did_agree
                                                      else None),
                                       "agree_stock": agree_stock})
            if new_best:                        # `best` marker, like `latest`; never pruned
                write_marker(out, "best", d)
                print(f"[best] agree_ckpt {best_agree:.4f} (stock {agree_stock:.4f}, delta "
                      f"{best_agree - agree_stock:+.4f}) at step {best_step} -> "
                      f"{os.path.basename(d)}", flush=True)
        if saving:
            wb.log({"checkpoint/step": step + 1}, step=step + 1)
        if saving and world > 1:
            dist.barrier()                      # no rank may run ahead of a half-written checkpoint

        if stop:
            row = {"step": step + 1, "event": "early_stop", "agree_ckpt": float(ev["agree_ckpt"]),
                   "agree_best": best_agree, "agree_best_step": best_step,
                   "agree_stock": agree_stock, "agree_delta": best_agree - agree_stock,
                   "evals_without_improvement": bad_evals,
                   "patience": int(args.early_stop_patience), "min_steps": int(args.min_steps)}
            if rank == 0:
                print(f"[early-stop] step {step + 1}: agree_ckpt {row['agree_ckpt']:.4f} has not "
                      f"beaten the best {best_agree:.4f} (step {best_step}) for {bad_evals} "
                      f"consecutive agreement evals (patience {args.early_stop_patience}, "
                      f"min-steps {args.min_steps}); stock {agree_stock:.4f}, best delta "
                      f"{best_agree - agree_stock:+.4f}.  Final checkpoint saved at step "
                      f"{step + 1}; best is <out>/best -> "
                      f"{marker(out, 'best')}", flush=True)
            logf.write(json.dumps(row) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(row, "eval"), step=step + 1)
            break

    if rank == 0 and not stop:
        save_checkpoint(model, opt, out, last_step, tok=tok, src=src, keep=args.keep_checkpoints,
                        save_dtype=DTYPES[args.save_dtype],
                        extra={"config_hash": chash, "world_size": world})
    wb.finish()
    logf.close()
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
