"""The judge's block loop, batched, as pure torch: the FULL judge (draftalign/da_decode.py::judge_full_block), its
re-draft cadence (rd1 / rd2) with the rejected-token ban and the loop guard, the stock fallback after the pass cap,
and the closing (KV-storing) forward -- expressed over a BATCH of blocks with two callbacks, so SGLang's judge
algorithm (c1b_judge_loop.py) and the CPU tests (corpus1b/tests/test_judge_core.py) drive one implementation.

The decisions are the harness's, function by function:

    read at the open slots   blind_read / moe_probe.slot_metrics   fp32 softmax, [MASK] column suppressed, argmax, top-1 p, q = p(D)
    commit rule              full_commit_rule                      p > tau_commit strict; progress guarantee; tentative = argmax == D; else [MASK]
    flag per pass            judge_full_block                      open & (cur != [MASK]) -- the tentative draft slots of THIS pass
    re-draft at pass it      judge_full_block                      (it+1) % redraft_every == 0; holes = open & [MASK]; tentative hidden as
                                                                   [MASK] in the drafter's input; steps = min(8, ceil(n_h / rate)); ban = D at holes
    stock fallback           da_generate step 4                    after max_passes with open slots: re-mask them, then STOCK passes
                                                                   (bf16 confidence, strict >, 1 transfer per step) until no [MASK]
    closing forward          da_generate "kvflush"                 flag all False, stores the block's KV

Those four helpers are copied here VERBATIM (no import of da_decode: the SGLang process must not pull the HF harness's
transformers-side modules); test_judge_core.py pins each copy to the original with torch.equal.

Batch semantics.  All blocks of a batch advance in lock-step: one `forward_fn` call per pass serves every block
(a finished block is forwarded with an all-False flag and its final tokens; its logits are ignored), which is how
SGLang's dLLM algorithms must run.  Per block the sequence of (flag, decision) pairs is exactly the single-block
harness sequence, because a block's pass counter `it` only advances on the passes it took part in.

Callbacks:
    forward_fn(ids [B, bl] long, flag [B, bl] bool, kind: str) -> logits [B, bl, V]   (kind in judge | denoise | kvflush)
    draft_fn(reqs: List[DraftReq]) -> List[DraftRes]                                    (one batched drafter call)

The loop never touches the prefix: `lo` and the prompt tail are the caller's (the first decode block holds the prompt
tail at its non-valid slots; `valid` marks the generated slots).  Teacher forcing is `closing_ids`: the tokens the
closing forward stores and `BlockResult.tokens_out` reports, while `BlockResult.tokens` keeps the pipeline's own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

FULL_MAX_PASSES = 32          # da_decode.FULL_MAX_PASSES
DRAFT_STEPS = 8               # drafter.SETTINGS["s8"]["steps"]


# ---------------------------------------------------------------------------
# verbatim copies of the harness's decision helpers (pinned by test_judge_core.py)
# ---------------------------------------------------------------------------
def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    """== sdar_harness.decode.get_num_transfer_tokens."""
    base = block_length // steps
    remainder = block_length % steps
    n = torch.zeros(steps, dtype=torch.int64) + base
    n[:remainder] += 1
    return n


def read_open(logits: torch.Tensor, D: torch.Tensor, read: torch.Tensor, mask_id: int) -> dict:
    """== da_decode.blind_read(logits[None], D, read, mask_id) on logits [bl, V]: fp32 softmax with the [MASK] column
    suppressed, at the slots `read`: argmax (-1 elsewhere), p = top-1 and q = p(D) (nan elsewhere), raw_mask_argmax."""
    lg = logits.to(torch.float32, copy=True)
    raw_mask = int((lg.argmax(-1) == mask_id)[read].sum())
    lg[:, mask_id] = float("-inf")
    p = F.softmax(lg, dim=-1)
    top1, argmax = p.max(dim=-1)
    q = p.gather(1, D.view(-1, 1)).view(-1)
    nan = torch.full_like(top1, float("nan"))
    return {"argmax": torch.where(read, argmax, torch.full_like(argmax, -1)),
            "p": torch.where(read, top1, nan), "q": torch.where(read, q, nan), "raw_mask_argmax": raw_mask}


def full_commit_rule(cur: torch.Tensor, argmax: torch.Tensor, p: torch.Tensor, D: torch.Tensor,
                     open_: torch.Tensor, tau_commit: float, mask_id: int, remask_only: bool = False,
                     tau_override: Optional[float] = None):
    """== da_decode.full_commit_rule without the pool-coverage gate (never used by the judge round)."""
    pz = torch.nan_to_num(p, nan=-1.0)
    if tau_override is None or float(tau_override) == float(tau_commit):
        commit = open_ & (pz > tau_commit)
    else:
        agree = argmax == D
        commit = open_ & ((agree & (pz > tau_commit)) | (~agree & (pz > float(tau_override))))
    forced = False
    if bool(open_.any()) and not bool(commit.any()):
        best = torch.where(open_, pz, torch.full_like(pz, float("-inf"))).argmax()
        commit = commit.clone(); commit[best] = True
        forced = True
    rest = open_ & ~commit
    tentative = rest & (argmax == D) if not remask_only else torch.zeros_like(rest)
    remask = rest & ~tentative
    new = torch.where(commit, argmax,
                      torch.where(tentative, D,
                                  torch.where(remask, torch.full_like(cur, mask_id), cur)))
    return new, commit, tentative, remask, forced


def stock_transfer(logits: torch.Tensor, cur: torch.Tensor, mask_id: int, threshold: float, n_transfer: int):
    """== da_decode.stock_pass's decision on logits [bl, V] (sample_with_confidence greedy_exact / top_k 1 / T 1):
    x0 = argmax of the softmax IN THE LOGITS' DTYPE, confidence = that probability, transfer = every masked slot
    with confidence > threshold (strict) if at least n_transfer of them, else the top n_transfer by confidence."""
    probs = F.softmax(logits, dim=-1)
    x0 = probs.argmax(dim=-1)
    x0_p = probs.gather(-1, x0.view(-1, 1)).view(-1)
    mask_index = cur == mask_id
    x0 = torch.where(mask_index, x0, cur)
    confidence = torch.where(mask_index, x0_p, torch.tensor(float("-inf"), device=cur.device, dtype=x0_p.dtype))
    transfer = torch.zeros_like(mask_index)
    if threshold < 1.0:
        high = confidence > threshold
        if int(high.sum()) >= n_transfer:
            transfer = high
        else:
            _, idx = torch.topk(confidence, n_transfer)
            transfer[idx] = True
    else:
        _, idx = torch.topk(confidence, k=n_transfer)
        transfer[idx] = True
    return x0, x0_p, transfer


def masked_base(final: torch.Tensor, valid: torch.Tensor, mask_id: int) -> torch.Tensor:
    """== drafter.masked_base."""
    base = final.clone()
    base[valid] = mask_id
    return base


# ---------------------------------------------------------------------------
# the drafter interface
# ---------------------------------------------------------------------------
@dataclass
class DraftReq:
    key: int                          # index of the block in the batch
    block_ids: torch.Tensor           # [bl] the drafter's input block ([MASK] at the slots to fill AND at hidden slots)
    fill: torch.Tensor                # [bl] bool: the slots to fill
    steps: Optional[int]              # None = the setting's own (min(8, n_fill)); else min(steps, n_fill)
    ban: Optional[torch.Tensor]       # [bl] long, -1 = no ban (the rejected draft at each hole)
    guard: bool                       # the loop guard applies (the caller holds the guard state per item)
    kind: str                         # "draft" | "redraft" (timing label only)


@dataclass
class DraftRes:
    tokens: torch.Tensor              # [bl]
    p_commit: torch.Tensor            # [bl] fp32 (0 where not filled)
    n_fwd: int = 0
    guard_slots: int = 0
    guard_changed: int = 0


# ---------------------------------------------------------------------------
# per-block state and result
# ---------------------------------------------------------------------------
@dataclass
class BlockRun:
    cur: torch.Tensor                 # [bl] long (device): the block's working tokens
    valid: torch.Tensor               # [bl] bool (device): generated slots
    lo: int = 0
    committed: Optional[torch.Tensor] = None
    D: Optional[torch.Tensor] = None  # the CURRENT draft per slot (re-drafts replace it)
    it: int = 0                       # judge passes taken
    done: bool = False                # every valid slot committed (or the stock fallback finished)
    stock: bool = False               # in the stock fallback
    stock_step: int = 0
    # records
    draft_tokens: Optional[torch.Tensor] = None
    p_draft: Optional[torch.Tensor] = None
    passes: List[dict] = field(default_factory=list)
    states: List[dict] = field(default_factory=list)     # per pass: the recorder's view (cpu tensors)
    judge_argmax: Optional[torch.Tensor] = None
    judge_p: Optional[torch.Tensor] = None
    judge_q: Optional[torch.Tensor] = None
    judge_raw_mask_argmax: int = 0
    n_judge: int = 0
    n_draft_fwd: int = 0
    n_redraft_rounds: int = 0
    n_redraft_slots: int = 0
    n_redraft_fwd: int = 0
    n_redraft_eq_judge: int = 0
    n_redraft_eq_old: int = 0
    n_loopguard_slots: int = 0
    n_loopguard_changed: int = 0
    n_tentative: int = 0
    n_remask: int = 0
    full_commit_eq_draft: int = 0
    full_fallback: int = 0
    n_stock: int = 0
    stock_committed: List[torch.Tensor] = field(default_factory=list)
    stock_confidence: List[torch.Tensor] = field(default_factory=list)
    stock_proposed: List[torch.Tensor] = field(default_factory=list)
    tokens_out: Optional[torch.Tensor] = None            # the closing forward's tokens (teacher-forced when given)


@dataclass
class JudgeConfig:
    mask_id: int
    tau_commit: float = 0.95
    tau_override: Optional[float] = None
    remask_only: bool = False
    max_passes: int = FULL_MAX_PASSES
    redraft_every: int = 0            # 0 = no re-draft; 1 = rd1; 2 = rd2
    redraft_rate: int = 4
    redraft_ban: bool = False
    loop_guard: int = 0               # n of the n-gram guard (0 = off); the guard itself is the caller's
    threshold: float = 0.95           # the stock fallback's commit threshold
    draft_steps: int = DRAFT_STEPS
    record_states: bool = False       # keep the per-pass states (the recorder)


def _first_draft_req(key: int, b: BlockRun, cfg: JudgeConfig) -> DraftReq:
    return DraftReq(key=key, block_ids=masked_base(b.cur, b.valid, cfg.mask_id), fill=b.valid.clone(), steps=None,
                    ban=None, guard=cfg.loop_guard > 0, kind="draft")


def _redraft_req(key: int, b: BlockRun, holes: torch.Tensor, cfg: JudgeConfig) -> DraftReq:
    tent = b.valid & ~b.committed & (b.cur != cfg.mask_id)
    inp = torch.where(tent, torch.full_like(b.cur, cfg.mask_id), b.cur)     # the drafter never sees a tentative token
    n_h = int(holes.sum())
    steps = min(cfg.draft_steps, -(-n_h // int(cfg.redraft_rate)))
    ban = torch.where(holes, b.D, torch.full_like(b.D, -1)) if cfg.redraft_ban else None
    return DraftReq(key=key, block_ids=inp, fill=holes, steps=steps, ban=ban, guard=cfg.loop_guard > 0, kind="redraft")


def _apply_first_draft(b: BlockRun, d: DraftRes) -> None:
    b.D = d.tokens.clone()
    b.p_draft = d.p_commit.float().clone()
    b.n_draft_fwd = int(d.n_fwd)
    b.n_loopguard_slots += int(d.guard_slots); b.n_loopguard_changed += int(d.guard_changed)
    b.draft_tokens = b.D.cpu().clone()
    b.cur = torch.where(b.valid, b.D, b.cur)                 # every open valid slot = the draft, no masks
    b.committed = torch.zeros_like(b.valid)


def run_blocks(blocks: Sequence[BlockRun], cfg: JudgeConfig, forward_fn: Callable, draft_fn: Callable,
               closing_ids: Optional[Sequence[Optional[torch.Tensor]]] = None) -> List[BlockRun]:
    """Drive `blocks` (one per request, same block length) to completion; returns them with `cur` final.

    Every block enters with [MASK] at its valid slots (the prompt tail at the rest) and no draft; the first draft is
    requested here.  `closing_ids[i]` (optional, teacher forcing) replaces block i's tokens in the closing forward.
    """
    blocks = list(blocks)
    B = len(blocks)
    if B == 0:
        return blocks
    bl = int(blocks[0].cur.numel())
    dev = blocks[0].cur.device
    rde = int(cfg.redraft_every)
    assert rde >= 0 and int(cfg.redraft_rate) >= 1 and int(cfg.max_passes) >= 1

    # 1. first drafts (one batched drafter call)
    res = draft_fn([_first_draft_req(i, b, cfg) for i, b in enumerate(blocks)])
    assert len(res) == B
    for b, d in zip(blocks, res):
        _apply_first_draft(b, d)
        if not bool(b.valid.any()):
            b.done = True

    # 2. judge passes, lock-step
    while True:
        active = [i for i, b in enumerate(blocks) if not b.done and not b.stock]
        if not active:
            break
        ids = torch.stack([b.cur for b in blocks])
        flag = torch.zeros(B, bl, dtype=torch.bool, device=dev)
        opens: Dict[int, torch.Tensor] = {}
        for i in active:
            b = blocks[i]
            open_ = b.valid & ~b.committed
            if not bool(open_.any()):
                b.done = True
                continue
            opens[i] = open_
            flag[i] = open_ & (b.cur != cfg.mask_id)        # the tentative draft slots of THIS pass
        if not opens:
            continue
        logits = forward_fn(ids, flag, "judge")
        reqs: List[DraftReq] = []
        holes_of: Dict[int, torch.Tensor] = {}
        argmax_of: Dict[int, torch.Tensor] = {}
        for i, open_ in opens.items():
            b = blocks[i]
            r = read_open(logits[i], b.D, open_, cfg.mask_id)
            argmax_of[i] = r["argmax"]
            b.n_judge += 1
            before = b.cur.clone()
            new, commit, tentative, remask, forced = full_commit_rule(
                b.cur, r["argmax"], r["p"], b.D, open_, cfg.tau_commit, cfg.mask_id, cfg.remask_only,
                tau_override=cfg.tau_override)
            b.cur = new
            b.committed = b.committed | commit
            b.n_tentative += int(tentative.sum()); b.n_remask += int(remask.sum())
            b.full_commit_eq_draft += int((commit & (r["argmax"] == b.D)).sum())
            pass_rec = {"commit": int(commit.sum()), "tentative": int(tentative.sum()),
                        "remask": int(remask.sum()), "forced": bool(forced)}
            if cfg.record_states:
                b.states.append({"pass": b.it, "input": before.cpu(), "flag": flag[i].cpu(), "open": open_.cpu(),
                                 "draft": b.D.cpu().clone(), "commit": commit.cpu(), "tentative": tentative.cpu(),
                                 "remask": remask.cpu(), "forced": bool(forced), "argmax": r["argmax"].cpu(),
                                 "p": r["p"].cpu(), "q": r["q"].cpu()})
            if b.it == 0:
                b.judge_argmax = r["argmax"].cpu().clone(); b.judge_p = r["p"].cpu().clone()
                b.judge_q = r["q"].cpu().clone(); b.judge_raw_mask_argmax = r["raw_mask_argmax"]
            if rde:
                n_h = 0
                if (b.it + 1) % rde == 0:
                    holes = b.valid & ~b.committed & (b.cur == cfg.mask_id)
                    n_h = int(holes.sum())
                    if n_h:
                        holes_of[i] = holes
                        reqs.append(_redraft_req(i, b, holes, cfg))
                pass_rec["redraft"] = n_h
            b.passes.append(pass_rec)
            b.it += 1
        if reqs:
            res = draft_fn(reqs)
            assert len(res) == len(reqs)
            for q, d in zip(reqs, res):
                b = blocks[q.key]
                holes = holes_of[q.key]
                newtok = d.tokens
                b.n_redraft_rounds += 1; b.n_redraft_slots += int(holes.sum())
                b.n_loopguard_slots += int(d.guard_slots); b.n_loopguard_changed += int(d.guard_changed)
                b.n_redraft_fwd += int(d.n_fwd)
                b.n_redraft_eq_judge += int((holes & (newtok == argmax_of[q.key])).sum())
                b.n_redraft_eq_old += int((holes & (newtok == b.D)).sum())
                b.cur = torch.where(holes, newtok, b.cur)
                b.D = torch.where(holes, newtok, b.D)
                if cfg.record_states:
                    b.states[-1]["redraft"] = {"holes": holes.cpu(), "tokens": newtok.cpu().clone(),
                                               "p_commit": d.p_commit.float().cpu().clone(),
                                               "input": q.block_ids.cpu().clone(), "n_fwd": int(d.n_fwd)}
        for i in active:
            b = blocks[i]
            if b.done:
                continue
            open_ = b.valid & ~b.committed
            if not bool(open_.any()):
                b.done = True
            elif b.it >= int(cfg.max_passes):                 # cap reached -> fall back to the STOCK rule
                b.full_fallback = int(open_.sum())
                b.cur = torch.where(open_, torch.full_like(b.cur, cfg.mask_id), b.cur)
                b.stock = True

    # 3. stock fallback passes (da_generate step 4), lock-step over the blocks that need them
    num_transfer = get_num_transfer_tokens(bl, bl)
    while True:
        active = [i for i, b in enumerate(blocks) if b.stock and not b.done]
        if not active:
            break
        for i in active:
            b = blocks[i]
            if not bool((b.cur == cfg.mask_id).any()):
                b.done = True
        active = [i for i in active if not blocks[i].done]
        if not active:
            break
        ids = torch.stack([b.cur for b in blocks])
        flag = torch.zeros(B, bl, dtype=torch.bool, device=dev)
        logits = forward_fn(ids, flag, "denoise")
        for i in active:
            b = blocks[i]
            assert b.stock_step < bl + 1, "stock fallback did not converge"
            x0, x0_p, transfer = stock_transfer(logits[i], b.cur, cfg.mask_id, cfg.threshold,
                                                int(num_transfer[b.stock_step]))
            b.stock_committed.append(transfer.cpu().clone())
            b.stock_confidence.append(x0_p.float().cpu().clone())
            b.stock_proposed.append(x0.cpu().clone())
            b.cur = torch.where(transfer, x0, b.cur)
            b.stock_step += 1
            b.n_stock += 1

    # 4. the closing forward: flag all False, final tokens (teacher-forced tokens when given), stores the KV
    final = []
    for i, b in enumerate(blocks):
        t = b.cur
        if closing_ids is not None and closing_ids[i] is not None:
            t = closing_ids[i].to(device=dev, dtype=b.cur.dtype)
            assert t.numel() == bl
        final.append(t)
    forward_fn(torch.stack(final), torch.zeros(B, bl, dtype=torch.bool, device=dev), "kvflush")
    for b, t in zip(blocks, final):
        b.tokens_out = t
    return blocks


def block_summary(b: BlockRun) -> dict:
    """The counters of one finished block (python ints / lists), in DABlock's names where one exists."""
    return {"lo": int(b.lo), "n_judge": b.n_judge, "n_draft_fwd": b.n_draft_fwd, "passes": list(b.passes),
            "n_tentative": b.n_tentative, "n_remask": b.n_remask, "full_commit_eq_draft": b.full_commit_eq_draft,
            "n_redraft_rounds": b.n_redraft_rounds, "n_redraft_slots": b.n_redraft_slots,
            "n_redraft_fwd": b.n_redraft_fwd, "n_redraft_eq_judge": b.n_redraft_eq_judge,
            "n_redraft_eq_old": b.n_redraft_eq_old, "n_loopguard_slots": b.n_loopguard_slots,
            "n_loopguard_changed": b.n_loopguard_changed, "full_fallback": b.full_fallback, "n_stock": b.n_stock,
            "judge_raw_mask_argmax": b.judge_raw_mask_argmax,
            "tokens": b.cur.cpu().tolist(), "tokens_out": b.tokens_out.cpu().tolist(),
            "draft_tokens": None if b.draft_tokens is None else b.draft_tokens.tolist(),
            "p_draft": None if b.p_draft is None else [round(float(x), 6) for x in b.p_draft.cpu().tolist()]}
