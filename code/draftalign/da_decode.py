"""Training-free Draft-Verify (DV) block decoding for SDAR at block 32.

Per block (same grid / valid / prompt-tail conventions as sdar_harness.decode.block_diffusion_generate):
  1. the dense drafter generates the block (draftalign.drafter.draft_block, setting s4 | s8 | dyn90) -> D;
  2. the MoE JUDGES the draft in-distribution with two complementary-halves forwards (store_kv=False):
       A = even valid slots [MASK] + odd slots D,  B = the reverse;
     every slot is read from the forward in which it was MASKED: fp32 probs with the [MASK] logit
     suppressed (moe_probe.slot_metrics), argmax a_i, p_i = top-1, q_i = p(D_i);
  3. accept rule (apply_accept_rule): commit slot i to a_i if p_i > tau_commit (strict, as stock);
     else take the DRAFT token if (a_i == D_i and p_i > tau_acc) or q_i > tau_acc_q; else [MASK].
     If nothing was committed / accepted, commit the single best slot by p_i (progress guarantee);
  4. the STOCK denoising rule from that state (stock_pass: bf16 softmax, strict > threshold, >= 1
     commit per pass, decode.py:230-264 verbatim) until no mask remains, then the kv flush
     (store_kv=True); the finished block is also flushed into the drafter's cache.
     `rejudge`: after every stock pass that leaves >= rejudge_k open slots, redraft the open slots
     (committed slots fixed) and re-judge them with halves over the open slots only (no progress
     forcing: the stock pass already made progress).

`draft=None` runs the identical loop with NO drafter and NO judge: that is the stock decode, and it
must reproduce block_diffusion_generate token-for-token and pass-for-pass (tests/test_da_decode.py).

judge="blind" (zero extra MoE forwards; the halves judge confirmed 76% of WRONG draft tokens at > .95
in the AB probe): the block's first STOCK pass runs exactly as stock (bf16 softmax, strict > threshold,
>= 1 forced commit); from the SAME forward's fp32 softmax (MASK suppressed) read a_i, p_i and
q_i = p_i(D_i) for every slot masked at that forward, then -- after the stock commits -- accept D_i at
every STILL-OPEN slot with q_i > tau_acc_q (and p_draft_i > tau_draft when tau_draft > 0).  A stock
commit is never overridden; n_judge = 0; the pass counts as a stock pass.  With `rejudge`, every later
stock pass re-reads q_i for its open slots against the CURRENT draft (no redraft) and accepts the same
way (code CODE_ACCEPT_LATE).  tau_commit / tau_acc / rejudge_k are unused in blind mode.

judge="full" (the TRAINED regime; section 3.1 of the design doc): per block, ONE MoE forward whose
every open valid slot holds the drafter's token (no masks at all), read at EVERY open slot -- fp32
softmax with the [MASK] logit suppressed -> a_i, p_i = top-1, q_i = p(D_i).  Commit slot i to a_i if
p_i > tau_commit (strict).  Every uncommitted open slot keeps D_i as TENTATIVE context if a_i == D_i
and becomes [MASK] otherwise (`full_remask_only` -> always [MASK]: the ablation).  The same judge pass
is then repeated on the new state (committed slots hold their tokens, tentative slots hold D_i, masked
slots [MASK]; read at all open slots = tentative + masked) until every valid slot is committed; at
least one commit per pass (the highest p_i among the open slots) so it always terminates.  After
`full_max_passes` judge passes the remaining open slots fall back to the STOCK rule (they are masked
and the stock denoising loop finishes the block).  Then the flush (store_kv) and the drafter flush,
as in every mode.  These forwards are counted as "judge" forwards.

Two OFF-by-default decode-rule variants (every default path is byte-identical without them):

  ANCHORED DRAFTING (`draft_after_pass=N`, blind only; N = 0 is today's rule).  With N >= 1 the block
  starts as the STOCK block and its first N passes are ordinary stock passes (bf16 softmax, strict >
  threshold, >= 1 commit; counted as denoise passes).  Only then is the drafter run -- over the
  STILL-OPEN slots, with the N passes' commits held FIXED as anchors (`draft_block(valid=open slots,
  final=the committed block)`, the same clean drafter prefix) -- so the draft is conditioned on real
  MoE tokens.  The blind acceptance rule then runs on the NEXT stock pass (pass N+1: q_i read from its
  own forward, accept D_i at the still-open slots with q_i > tau_acc_q) and the block finishes with
  stock passes, exactly as the blind rule does today.  `DABlock.anchored` / `n_anchor` record the
  anchor slots; `draft_tokens` holds the committed tokens at them.

  ONE-PASS HYBRID (`judge_passes=K`, full only; K = 0 is today's unbounded loop).  The full judge runs
  K passes with the same commit rule (confirm at tau_commit, override at tau_override, tentative /
  remask); after them EVERY uncommitted slot is re-masked -- no tentative draft context survives --
  and the STOCK denoising loop finishes the block (the `full_max_passes` fall-back path).  The K
  forwards count as judge forwards (n_judge, `full_passes_per_block`), the stock passes after them as
  denoise passes.

Draft FLAG (round 2; only with a flag-gated adapter, `MoEMeter(draft_flag=True)` -- see
`adapter.py`'s module docstring and `run_da_decode.load_adapter_into`): every judge pass is run with
`adapter`'s per-position context armed at its TENTATIVE DRAFT slots (open and not [MASK]), which
both adds the trained `draft_embed` vector to those slots' input embeddings (passed as
`inputs_embeds`, or added by an `embed_tokens` hook when the model's forward cannot take them) and
gates the LoRA delta to those positions.  Every other forward -- prefill, kv flush, stock denoise,
the halves / blind judges -- is armed with an ALL-FALSE flag, under which the projections are
bit-identical to the frozen base and no vector is added, i.e. it is the stock model.  Without such
an adapter (`draft_flag=False`, every cell so far) nothing is armed at all.

Expert pool (`PoolCtrl` / `enable_pool_moe`): a per-forward top-M expert restriction for every BLOCK
forward the MoE does (judge / denoise / rejudge; NEVER the prefill, and the flush only with
`pool_flush`).  Per layer: the fp32 router softmax of all 32 block rows; the VALID slots vote; the
top-M pool is built with `train_forward.expert_pool` for rules "full" (dMoE, sum of the full softmax)
and "top8" (sum of the RENORMALISED top-k weights), or with the DES-Vote rule "des" (sum of the
UNNORMALISED top-k weights, `sdar_harness/desvote.py:96-99` = its Algorithm 3); probabilities outside
the pool are zeroed, then top-8 + renormalise (`train_forward.moe_forward`'s arithmetic, so M >= E is
bit-identical to the unrestricted forward), then the hit-only expert loop.  The forward writes the
RESTRICTED `selected_experts` into moe_instr's SINK, so the counted experts are the ones that ran.

Numerics: the stock rule keeps decode.py's bf16 softmax + strict > threshold untouched; fp32 is used
only inside the judge.  Expert loads per MoE forward are read from sdar_harness.moe_instr's SINK
(the exact `selected_experts` tensor the dispatch indexes) with the same per-layer unique count as
run_research_baselines' `unique_expert_layer_loads` (desvote.py:165).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sdar_harness.decode import BlockTrace, get_num_transfer_tokens, sample_with_confidence
from sdar_harness.moe_instr import SINK
from sdar_harness.routing import moe_blocks
from draftalign.adapter import draft_flag_scope
from draftalign.seqcache import SeqCache
from draftalign.drafter import SETTINGS, draft_block
from draftalign.moe_probe import build_block_state, slot_metrics
from draftalign.train_forward import expert_pool

MOE_KIND_CODE = {"prefill": 0, "denoise": 1, "kvflush": 2, "judge": 3, "rejudge": 4}
DRAFTER_KIND_CODE = {"prefill": 0, "draft": 1, "redraft": 2, "flush": 3}
# per-slot judge outcome codes
CODE_MASK, CODE_COMMIT, CODE_ACCEPT_EQ, CODE_ACCEPT_Q, CODE_FORCED, CODE_ACCEPT_LATE = 0, 1, 2, 3, 4, 5
JUDGE_MODES = ("halves", "blind", "full")
POOL_RULES = ("full", "top8", "des")
POOLED_KINDS = ("denoise", "judge", "rejudge")     # block forwards the pool restricts (never "prefill")
FULL_MAX_PASSES = 32


# ---------------------------------------------------------------------------
# per-forward expert pool (dMoE / DES-Vote), patched into SDARMoeSparseMoeBlock.forward
# ---------------------------------------------------------------------------
class PoolCtrl:
    """Module-global controller for `pool_moe_forward` (one decode runs single-threaded per GPU).

    `M` / `rule` are the budget; `valid` is the bool [N] mask of the block rows that VOTE (the
    generated slots -- the prompt tail of the first decode block never votes, as in
    `desvote.CTRL.valid` and `train_forward.training_forward`).  `enabled` is flipped per forward
    by `MoEMeter` (prefill and, unless `pool_flush`, the kv flush run unrestricted).
    `force_restrict` runs the restricted arithmetic even at M >= E, which must be bit-identical.
    """

    def __init__(self):
        self.M: Optional[int] = None
        self.rule: str = "full"
        self.valid: Optional[torch.Tensor] = None
        self.enabled = False
        self.force_restrict = False
        self.sizes: List[int] = []          # pool size per layer of the CURRENT forward
        self.track_cov = False              # accumulate per-row pool coverage (off => bit-identical)
        self.cov_sum: Optional[torch.Tensor] = None
        self.cov_n = 0

    def configure(self, M: Optional[int], rule: str = "full"):
        assert rule in POOL_RULES, rule
        self.M, self.rule = (None if M is None else int(M)), rule
        return self

    @property
    def on(self) -> bool:
        return self.M is not None

    def begin(self, enabled: bool):
        self.enabled = bool(enabled)
        self.sizes = []
        self.cov_sum, self.cov_n = None, 0

    def coverage(self) -> Optional[torch.Tensor]:
        """Mean over layers of each row's ORIGINAL top-k router mass that survived into the pool.

        1.0 => this row's routing was untouched by pooling, so its logits are what the unpooled
        forward would have produced; low => most of its experts were pruned and its prediction is
        a substitute computed from experts the router did not want.  None when not tracked.
        """
        if self.cov_sum is None or not self.cov_n:
            return None
        return self.cov_sum / float(self.cov_n)

    def end(self) -> int:
        """The pool size actually used by this forward (0 when no layer was pooled)."""
        self.enabled = False
        sizes = self.sizes
        self.sizes = []
        if not sizes:
            return 0
        assert len(set(sizes)) == 1, f"layers disagreed on the pool size: {sorted(set(sizes))}"
        return int(sizes[0])


POOL = PoolCtrl()


def build_pool(w: torch.Tensor, valid: torch.Tensor, M: int, rule: str, top_k: int) -> torch.Tensor:
    """Top-M pool [E] bool from the votes of the VALID rows of the fp32 router softmax `w` [N, E].

    "full" / "top8" are `train_forward.expert_pool` verbatim (dMoE sum-of-softmax, and DES's
    Algorithm-3 term RENORMALISED, which is what the trainer samples pools with).  "des" is
    `sdar_harness/desvote.py:96-99` exactly: the row's UNNORMALISED top-k weights, zeros elsewhere,
    summed over the valid rows.  Ties break on the lowest expert id (stable sort), as desvote.py:133.
    """
    assert rule in POOL_RULES, rule
    if rule != "des":
        return expert_pool(w, valid, M, rule, top_k)
    E = w.shape[1]
    vals, ids = torch.topk(w, top_k, dim=-1)
    contrib = torch.zeros_like(w).scatter_(1, ids, vals)                     # desvote._topk_masked
    votes = (contrib * valid.to(contrib.dtype).unsqueeze(1)).sum(0)          # [E]
    order = torch.sort(votes, descending=True, stable=True).indices
    pool = torch.zeros(E, dtype=torch.bool, device=w.device)
    pool[order[:min(int(M), E)]] = True
    return pool


def pool_moe_forward(self, hidden_states: torch.Tensor):
    """`moe_instr.instrumented_moe_forward` + the per-forward pool of `train_forward.moe_forward`.

    The pool is applied by multiplying the fp32 softmax by a 0/1 mask AFTER the softmax (exact for
    1.0), so with every expert in the pool the top-k, the renormalisation and the dispatch see the
    very same numbers the unrestricted forward sees -- bit-identical.  The ids written to the SINK
    are the RESTRICTED `selected_experts`, i.e. the experts the dispatch below actually indexes.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    w = F.softmax(router_logits, dim=1, dtype=torch.float)
    E = self.num_experts
    restrict = POOL.enabled and POOL.on and (POOL.M < E or POOL.force_restrict)
    if restrict:
        assert POOL.M >= self.top_k, f"pool M={POOL.M} < top_k={self.top_k}"
        valid = POOL.valid
        if valid is None:
            valid = torch.ones(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        assert valid.numel() == hidden_states.shape[0], \
            f"pool vote mask {valid.numel()} != {hidden_states.shape[0]} rows"
        pool = build_pool(w, valid, POOL.M, POOL.rule, self.top_k)
        if POOL.track_cov:                                     # per-row survival of the ORIGINAL routing
            ov, oi = torch.topk(w, self.top_k, dim=-1)
            cov = (ov * pool[oi].to(ov.dtype)).sum(1) / ov.sum(1).clamp_min(1e-9)
            POOL.cov_sum = cov if POOL.cov_sum is None else POOL.cov_sum + cov
            POOL.cov_n += 1
        w_use = w * pool.to(w.dtype).unsqueeze(0)
        POOL.sizes.append(int(pool.sum()))
    else:
        w_use = w
        if POOL.enabled:
            POOL.sizes.append(E)
    routing_weights, selected_experts = torch.topk(w_use, self.top_k, dim=-1)

    cap = SINK.active                                          # moe_instr.py:79-82, before renormalisation
    if cap is not None:
        cap.ids.append(selected_experts.detach().to(torch.uint8).cpu().clone())
        cap.probs.append(routing_weights.detach().half().cpu().clone())

    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim),
                                      dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = F.one_hot(selected_experts, num_classes=E).permute(2, 1, 0)
    for expert_idx in torch.unique(selected_experts).tolist():
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def enable_pool_moe(model) -> int:
    """Install `pool_moe_forward` on every MoE block (replaces moe_instr's patch; it captures too)."""
    blocks = moe_blocks(model)
    for blk in blocks:
        assert getattr(blk, "norm_topk_prob", False), "the pool path assumes norm_topk_prob=True"
        if not hasattr(blk, "_stock_forward"):
            blk._stock_forward = blk.forward
        blk.forward = pool_moe_forward.__get__(blk, type(blk))
    return len(blocks)


def disable_pool_moe(model) -> int:
    n = 0
    for blk in moe_blocks(model):
        if hasattr(blk, "_stock_forward"):
            blk.forward = blk._stock_forward
            del blk._stock_forward
            n += 1
    return n


# ---------------------------------------------------------------------------
# metering: one SeqCache + the moe_instr SINK -> per-forward [kind, ms, uniq, layer_forwards, npos]
# ---------------------------------------------------------------------------
class MoEMeter:
    """Wraps a SeqCache; arms `sink` (sdar_harness.moe_instr.SINK) around every forward and reads the
    hit-expert sets back: uniq = sum over layers of |unique(selected_experts)|, lf = layers seen.
    Without an instrumented model (sink None / not armed) uniq = lf = 0.

    `pool` (a `PoolCtrl`, normally the module-global `POOL`) is armed around every BLOCK forward whose
    kind is in POOLED_KINDS -- and around the kv flush only when `pool_flush` -- and never around the
    prefill.  `set_valid` gives it the block's voting rows.  Per forward the pool size actually used
    is stored as the 6th field of the record.

    `draft_flag` (on only for a FLAG-GATED adapter, `adapter.is_flag_gated`) arms `adapter`'s
    per-position context around every forward: the caller's flag for the full judge's draft-filled
    passes, an ALL-FALSE flag for everything else (prefill, kv flush, stock denoise, halves / blind
    judges).  All-False is the right "untouched" value, not `None`: such an adapter runs UN-merged,
    so a `None` (ungated) context would apply the LoRA delta at every position -- exactly what
    training excluded -- while under an all-False flag every projection is bit-identical to the
    frozen base and no `draft_embed` is added.  `self.flags` keeps the flag of every forward (cpu)
    for the tests and for auditing."""

    def __init__(self, seq: SeqCache, sink=None, pool: "Optional[PoolCtrl]" = None, pool_flush: bool = False,
                 draft_flag: bool = False):
        self.seq, self.sink = seq, sink
        self.pool, self.pool_flush = pool, bool(pool_flush)
        self.draft_flag = bool(draft_flag)
        self.records: List[list] = []
        self.flags: List[Optional[torch.Tensor]] = []      # per forward: the flag armed (None = ungated)
        self.last_cap = None                # the MoECapture of the most recent forward (or None)

    @property
    def bl(self) -> int:
        return self.seq.bl

    def reset(self):
        self.records = []
        self.flags = []
        self.last_cap = None

    @contextmanager
    def _flag_scope(self, flag: Optional[torch.Tensor], ids: torch.Tensor, npos: int,
                    conf: Optional[torch.Tensor] = None):
        """Arm the draft flag for one forward; yields the extra model kwargs (`{}` or inputs_embeds).
        `conf` [npos] uint8 = the drafter's confidence per slot, used only by an adapter with a confidence
        embedding (the 1B corpus's 32M test); every other adapter ignores it."""
        if not self.draft_flag:
            self.flags.append(None)
            yield {}
            return
        dev = ids.device if ids is not None else self.seq.dev
        f = (torch.zeros(1, int(npos), dtype=torch.bool, device=dev) if flag is None
             else flag.reshape(1, -1).to(device=dev, dtype=torch.bool))
        assert f.shape[1] == int(npos), f"flag {tuple(f.shape)} does not cover {npos} positions"
        self.flags.append(f[0].detach().cpu().clone())
        cu8 = None
        if conf is not None:
            cu8 = conf.reshape(1, -1).to(device=dev, dtype=torch.uint8)
        elif flag is not None:
            cu8 = torch.zeros(1, int(npos), dtype=torch.uint8, device=dev)
        with draft_flag_scope(self.seq.model, ids, f, conf_u8=cu8) as extra:
            yield extra

    def set_valid(self, valid: Optional[torch.Tensor]):
        """The block rows that vote for the pool (the generated slots)."""
        if self.pool is not None:
            self.pool.valid = valid

    def _pooled(self, kind: str) -> bool:
        if self.pool is None or not self.pool.on:
            return False
        return kind in POOLED_KINDS or (kind == "kvflush" and self.pool_flush)

    def _arm(self, kind: str):
        if self.sink is None:
            return None
        from sdar_harness.moe_instr import MoECapture
        cap = MoECapture(tag=kind, step=len(self.records))
        self.sink.arm(cap)
        return cap

    def _disarm(self):
        if self.sink is not None:
            self.sink.disarm()

    def _record(self, cap, kind: str, npos: int, pool_size: int = 0):
        """Called after a successful forward: seq.timings[-1] is that forward."""
        uniq = sum(int(torch.unique(ids).numel()) for ids in cap.ids) if cap is not None else 0
        lf = len(cap.ids) if cap is not None else 0
        self.last_cap = cap
        ms = self.seq.timings[-1][1]
        self.records.append([MOE_KIND_CODE[kind], round(ms, 3), uniq, lf, int(npos), int(pool_size)])

    def prefill(self, prompt_ids: torch.Tensor) -> int:
        cap = self._arm("prefill")
        if self.pool is not None:
            self.pool.begin(False)                    # the prefill is NEVER restricted
        pl_planned = (int(prompt_ids.shape[1]) // self.seq.bl) * self.seq.bl
        try:
            if pl_planned == 0:                       # no prompt block: no forward, so no flag record
                pl = self.seq.prefill(prompt_ids)
            else:
                with self._flag_scope(None, prompt_ids[:, :pl_planned].to(self.seq.dev), pl_planned) as extra:
                    pl = self.seq.prefill(prompt_ids, extra=extra)
        finally:
            self._disarm()
            if self.pool is not None:
                self.pool.end()
        if pl > 0:
            self._record(cap, "prefill", pl)
        return pl

    def block(self, block_ids: torch.Tensor, lo: int, store_kv: bool, kind: str,
              flag: Optional[torch.Tensor] = None, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        cap = self._arm(kind)
        pooled = self._pooled(kind)
        if self.pool is not None:
            self.pool.begin(pooled)
        ids = block_ids if block_ids.dim() == 2 else block_ids.unsqueeze(0)
        try:
            with self._flag_scope(flag, ids.to(self.seq.dev), self.seq.bl, conf=conf) as extra:
                logits = self.seq.block_forward(block_ids, lo, store_kv=store_kv, kind=kind, extra=extra)
        finally:
            self._disarm()
            psize = self.pool.end() if self.pool is not None else 0
        self._record(cap, kind, self.seq.bl, psize)
        return logits

    def by_kind(self) -> Dict[str, dict]:
        inv = {v: k for k, v in MOE_KIND_CODE.items()}
        out = {k: {"n": 0, "ms": 0.0, "uniq": 0, "lf": 0, "pool": 0} for k in MOE_KIND_CODE}
        for code, ms, u, lf, _, ps in self.records:
            k = inv[code]
            out[k]["n"] += 1; out[k]["ms"] += ms; out[k]["uniq"] += u; out[k]["lf"] += lf
            out[k]["pool"] += ps
        return out


def drafter_records(dr_seq: Optional[SeqCache]) -> List[list]:
    """[kind_code, ms] per drafter forward, in call order (kinds draft_<setting> -> 'draft')."""
    if dr_seq is None:
        return []
    out = []
    for kind, ms in dr_seq.timings:
        k = "draft" if kind.startswith("draft_") else kind
        out.append([DRAFTER_KIND_CODE[k], round(ms, 3)])
    return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class DABlock(BlockTrace):
    """BlockTrace (per STOCK pass: committed / confidence / proposed) + the draft/judge of this block."""
    draft_tokens: Optional[torch.Tensor] = None   # [bl] long, first draft (fixed slots = block tokens)
    p_draft: Optional[torch.Tensor] = None        # [bl] fp32 drafter confidence at commit
    judge_argmax: Optional[torch.Tensor] = None   # [bl] long, -1 outside valid
    judge_p: Optional[torch.Tensor] = None        # [bl] fp32 top-1 (MASK suppressed), nan outside valid
    judge_q: Optional[torch.Tensor] = None        # [bl] fp32 p(D_i), nan outside valid
    judge_code: Optional[torch.Tensor] = None     # [bl] int8: CODE_* per slot
    n_judge: int = 0                              # MoE judge forwards (first judge)
    n_draft_fwd: int = 0                          # drafter forwards (first draft)
    judge_raw_mask_argmax: int = 0
    rejudge_rounds: int = 0
    n_rejudge: int = 0                            # MoE re-judge forwards
    n_redraft_fwd: int = 0                        # drafter forwards spent on redrafts
    rejudge_committed: int = 0
    rejudge_accepted: int = 0
    rejudges: List[dict] = field(default_factory=list)   # per round: {open, draft, argmax, p, q, code} (cpu)
    judge_mode: str = "none"                      # none | halves | blind | full
    n_blind_accept: int = 0                       # blind mode: draft slots accepted (first pass + later passes)
    # -- anchored drafting (blind, --draft-after-pass N > 0) -----------------
    anchored: Optional[torch.Tensor] = None       # [bl] bool: slots committed by the N stock passes BEFORE the draft
    n_anchor: int = 0                             # int(anchored.sum()); 0 on every non-anchored path
    # -- full judge --------------------------------------------------------
    n_tentative: int = 0                          # full: uncommitted open slots that KEPT the draft token (a_i == D_i)
    n_remask: int = 0                             # full: uncommitted open slots set back to [MASK]
    full_commit_eq_draft: int = 0                 # full: committed slots whose a_i was the draft token
    full_fallback: int = 0                        # full: open slots handed to the stock rule at the pass cap
    full_passes: List[dict] = field(default_factory=list)   # per judge pass {commit, tentative, remask, forced}

    # -- derived ---------------------------------------------------------
    @property
    def accepted(self) -> torch.Tensor:
        """Draft tokens taken by the judge: halves = first judge (codes 2, 3); blind = every blind accept (3, 5)."""
        if self.judge_code is None:
            return torch.zeros(self.valid.numel(), dtype=torch.bool)
        return (self.judge_code == CODE_ACCEPT_EQ) | (self.judge_code == CODE_ACCEPT_Q) | (self.judge_code == CODE_ACCEPT_LATE)

    @property
    def committed_judge(self) -> torch.Tensor:
        if self.judge_code is None:
            return torch.zeros(self.valid.numel(), dtype=torch.bool)
        return (self.judge_code == CODE_COMMIT) | (self.judge_code == CODE_FORCED)

    @property
    def forced(self) -> bool:
        return self.judge_code is not None and bool((self.judge_code == CODE_FORCED).any())

    @property
    def block_forwards_ex_flush(self) -> int:
        """Every block forward of this block except the kv flush: judge + rejudge + stock denoise."""
        return int(self.n_judge) + int(self.n_rejudge) + int(self.n_steps)

    @property
    def draft_agreement(self) -> float:
        if self.draft_tokens is None or self.tokens is None or not bool(self.valid.any()):
            return float("nan")
        return float((self.draft_tokens == self.tokens)[self.valid].float().mean())


@dataclass
class DAResult:
    x: torch.Tensor
    prompt_length: int
    total_length: int
    block_length: int
    prefill_blocks: int
    num_blocks: int
    blocks: List[DABlock]
    stopped_early: bool
    n_forward: int                    # MoE forwards incl. prefill (== stock n_forward when draft is None)
    draft: Optional[str]
    judge: Optional[str] = None       # halves | blind | None (stock)
    draft_after_pass: int = 0         # blind: stock passes run BEFORE the draft (0 = draft first, the default)
    judge_passes: int = 0             # full: judge passes before the stock hand-off (0 = the unbounded loop)

    def generated_ids(self) -> torch.Tensor:
        return self.x[:, self.prompt_length:]


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------
@torch.no_grad()
def judge_halves(moe: MoEMeter, cur: torch.Tensor, slots: torch.Tensor, D: torch.Tensor, lo: int,
                 mask_id: int, kind: str = "judge") -> dict:
    """Two complementary-halves forwards over `slots` (bool [bl], must hold [MASK] in `cur`).
    Returns device tensors argmax [bl] long (-1 outside slots), p / q [bl] fp32 (nan outside),
    n_fwd, raw_mask_argmax.  A half with no masked slot is skipped."""
    bl = cur.numel()
    dev = cur.device
    argmax = torch.full((bl,), -1, dtype=torch.long, device=dev)
    p = torch.full((bl,), float("nan"), dtype=torch.float32, device=dev)
    q = torch.full((bl,), float("nan"), dtype=torch.float32, device=dev)
    n_fwd, raw = 0, 0
    for mode in ("half_a", "half_b"):
        ids, masked = build_block_state(cur, slots, D, mode, mask_id)
        read = masked & slots
        if not bool(read.any()):
            continue
        logits = moe.block(ids, lo, store_kv=False, kind=kind)
        m = slot_metrics(logits, D, D, D, mask_id, read)
        argmax = torch.where(read, m["argmax"], argmax)
        p = torch.where(read, m["top1"], p)
        q = torch.where(read, m["p_draft"], q)
        raw += int(m["raw_mask_argmax"])
        n_fwd += 1
        del logits
    assert bool((argmax[slots] >= 0).all()), "a judged slot was read from no forward"
    return {"argmax": argmax, "p": p, "q": q, "n_fwd": n_fwd, "raw_mask_argmax": raw}


def apply_accept_rule(cur: torch.Tensor, argmax: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                      D: torch.Tensor, slots: torch.Tensor, tau_commit: float, tau_acc: float,
                      tau_acc_q: float, force_progress: bool = True):
    """Returns (new block ids [bl], code [bl] int8).  Slots outside `slots` are untouched (code MASK=0).
    commit -> a_i if p_i > tau_commit; accept D_i if (a_i == D_i and p_i > tau_acc) or q_i > tau_acc_q;
    else [MASK].  force_progress: if nothing was committed/accepted, commit the best slot by p_i."""
    pz = torch.nan_to_num(p, nan=-1.0)
    qz = torch.nan_to_num(q, nan=-1.0)
    commit = slots & (pz > tau_commit)
    acc_eq = slots & ~commit & (argmax == D) & (pz > tau_acc)
    acc_q = slots & ~commit & ~acc_eq & (qz > tau_acc_q)
    code = torch.zeros(cur.numel(), dtype=torch.int8, device=cur.device)
    code = torch.where(commit, torch.full_like(code, CODE_COMMIT), code)
    code = torch.where(acc_eq, torch.full_like(code, CODE_ACCEPT_EQ), code)
    code = torch.where(acc_q, torch.full_like(code, CODE_ACCEPT_Q), code)
    if force_progress and bool(slots.any()) and not bool((commit | acc_eq | acc_q).any()):
        best = torch.where(slots, pz, torch.full_like(pz, float("-inf"))).argmax()
        commit = commit.clone(); commit[best] = True
        code[best] = CODE_FORCED
    new = torch.where(commit, argmax, torch.where(acc_eq | acc_q, D, cur))
    return new, code


@torch.no_grad()
def stock_pass(moe: MoEMeter, cur: torch.Tensor, lo: int, mask_id: int, threshold: float, n_transfer: int,
               temperature: float = 1.0, top_k: int = 1, top_p: float = 1.0, greedy_exact: bool = True,
               kind: str = "denoise"):
    """One STOCK denoising pass on cur [1, bl] (decode.py:228-258 verbatim; bf16 confidence, strict >).
    Returns (x0 [1, bl], x0_p [1, bl], transfer_index [1, bl] bool, logits [1, bl, V]); the caller writes
    the commits (and, in blind mode, reads q from the same logits)."""
    mask_index = (cur == mask_id)
    logits = moe.block(cur, lo, store_kv=False, kind=kind)
    x0, x0_p = sample_with_confidence(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                      greedy_exact=greedy_exact)
    x0 = torch.where(mask_index, x0, cur)
    confidence = torch.where(mask_index, x0_p, torch.tensor(float("-inf"), device=cur.device, dtype=x0_p.dtype))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    if threshold < 1.0:
        for j in range(confidence.shape[0]):
            high_conf = confidence[j] > threshold
            if high_conf.sum() >= n_transfer:
                transfer_index[j] = high_conf
            else:
                _, idx = torch.topk(confidence[j], n_transfer)
                transfer_index[j, idx] = True
    else:
        for j in range(confidence.shape[0]):
            _, idx = torch.topk(confidence[j], k=n_transfer)
            transfer_index[j, idx] = True
    return x0, x0_p, transfer_index, logits


@torch.no_grad()
def blind_read(logits: torch.Tensor, D: torch.Tensor, read: torch.Tensor, mask_id: int) -> dict:
    """fp32 softmax (MASK suppressed) of a stock pass's logits at the slots `read` (masked at that forward):
    argmax a_i (-1 elsewhere), p_i top-1 and q_i = p_i(D_i) (nan elsewhere)."""
    m = slot_metrics(logits, D, D, D, mask_id, read)
    nan = torch.full_like(m["top1"], float("nan"))
    return {"argmax": torch.where(read, m["argmax"], torch.full_like(m["argmax"], -1)),
            "p": torch.where(read, m["top1"], nan), "q": torch.where(read, m["p_draft"], nan),
            "raw_mask_argmax": int(m["raw_mask_argmax"])}


def blind_accept(cur: torch.Tensor, q: torch.Tensor, D: torch.Tensor, open_: torch.Tensor, tau_acc_q: float,
                 p_draft: Optional[torch.Tensor] = None, tau_draft: float = 0.0):
    """Accept D_i at the STILL-OPEN slots `open_` with q_i > tau_acc_q (and p_draft_i > tau_draft if tau_draft > 0).
    Returns (new block ids [bl], accept mask [bl])."""
    acc = open_ & (torch.nan_to_num(q, nan=-1.0) > tau_acc_q)
    if tau_draft > 0.0:
        assert p_draft is not None
        acc = acc & (p_draft > tau_draft)
    return torch.where(acc, D, cur), acc


# ---------------------------------------------------------------------------
# full judge (the trained regime): one draft-filled forward per pass, read at every OPEN slot
# ---------------------------------------------------------------------------
def full_commit_rule(cur: torch.Tensor, argmax: torch.Tensor, p: torch.Tensor, D: torch.Tensor,
                     open_: torch.Tensor, tau_commit: float, mask_id: int, remask_only: bool = False,
                     tau_override: Optional[float] = None, cov: Optional[torch.Tensor] = None,
                     tau_cov: float = 0.0):
    """One full-judge decision over the OPEN slots of `cur` [bl].

    commit -> a_i where p_i > tau_commit (strict, as stock).  If nothing passed, the single highest
    p_i among the open slots is committed anyway (progress guarantee -> the loop always terminates).
    Every other open slot keeps D_i as tentative context if a_i == D_i, else becomes [MASK];
    `remask_only` re-masks every uncommitted slot (the ablation).  `tau_override` (optional) is a
    separate, usually stricter threshold for commits that OVERRIDE the draft (a_i != D_i); confirmations
    keep tau_commit.  None or equal to tau_commit -> the symmetric rule above, unchanged.
    Returns (new ids [bl], commit [bl], tentative [bl], remask [bl], forced bool).
    """
    pz = torch.nan_to_num(p, nan=-1.0)
    if tau_override is None or float(tau_override) == float(tau_commit):
        commit = open_ & (pz > tau_commit)
    else:   # asymmetric rule: confirmations (a_i == D_i) at tau_commit, overrides (a_i != D_i) at tau_override
        agree = argmax == D
        commit = open_ & ((agree & (pz > tau_commit)) | (~agree & (pz > float(tau_override))))
    if cov is not None and tau_cov > 0.0:
        # POOL-COVERAGE GATE.  Pooling leaves the confidence scale intact while degrading the logits
        # of rows whose routing was pruned, so p alone cannot tell a clean prediction from a
        # substitute one.  Hold back commits on poorly-covered rows; they get another pass.
        commit = commit & (cov > float(tau_cov))
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


@torch.no_grad()
def judge_full_block(moe: MoEMeter, cur: torch.Tensor, valid: torch.Tensor, D: torch.Tensor, lo: int,
                     mask_id: int, tau_commit: float, bt: DABlock, max_passes: int = FULL_MAX_PASSES,
                     remask_only: bool = False, kind: str = "judge", tau_override: Optional[float] = None,
                     tau_pool_cov: float = 0.0, p_draft: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run the full judge on block `cur` [bl] until every valid slot is committed (or the cap fires).

    The block enters with EVERY valid slot holding the drafter's token -- no masks.  Returns the new
    block ids; slots left open at the cap are [MASK] and the caller's stock loop finishes them.
    `bt` is updated in place (n_judge / n_forward / judge_* arrays / tentative-remask counters).

    Each pass is flagged (`MoEMeter.block(..., flag=...)`) with its TENTATIVE DRAFT slots -- open and
    not [MASK], i.e. exactly what a trained state's flag marks: the committed slots, the re-masked
    slots and the prompt tail are unflagged.  With no flag-gated adapter loaded the flag is ignored.
    """
    cur = torch.where(valid, D, cur)                       # every open valid slot = the draft, no masks
    committed = torch.zeros_like(valid)
    # drafter confidence per slot as the 1B corpus stores it (uint8, round(255 p)); read only by an adapter
    # that has a confidence embedding (32M test), ignored by every other adapter
    conf_u8 = None if p_draft is None else torch.nan_to_num(p_draft.float(), nan=0.0).mul(255.0).round() \
        .clamp(0, 255).to(torch.uint8)
    for it in range(int(max_passes)):
        open_ = valid & ~committed
        if not bool(open_.any()):
            break
        flag = open_ & (cur != mask_id)                    # the tentative draft slots of THIS pass
        conf = None if conf_u8 is None else torch.where(flag, conf_u8, torch.zeros_like(conf_u8))
        logits = moe.block(cur.unsqueeze(0), lo, store_kv=False, kind=kind, flag=flag, conf=conf)
        cov = POOL.coverage() if tau_pool_cov > 0.0 else None   # [bl], aligned with this block's rows
        r = blind_read(logits, D, open_, mask_id)          # fp32, [MASK] logit suppressed, read at open slots
        del logits
        bt.n_judge += 1; bt.n_forward += 1
        new, commit, tentative, remask, forced = full_commit_rule(
            cur, r["argmax"], r["p"], D, open_, tau_commit, mask_id, remask_only, tau_override=tau_override,
            cov=cov, tau_cov=tau_pool_cov)
        cur = new
        committed = committed | commit
        # `forced` <=> nothing passed tau_commit, so `commit` holds exactly the one forced slot
        if bt.judge_code is not None:
            bt.judge_code[commit.cpu()] = CODE_FORCED if forced else CODE_COMMIT
        bt.n_tentative += int(tentative.sum()); bt.n_remask += int(remask.sum())
        bt.full_commit_eq_draft += int((commit & (r["argmax"] == D)).sum())
        bt.full_passes.append({"commit": int(commit.sum()), "tentative": int(tentative.sum()),
                               "remask": int(remask.sum()), "forced": bool(forced)})
        if it == 0:
            bt.judge_argmax = r["argmax"].cpu().clone(); bt.judge_p = r["p"].cpu().clone()
            bt.judge_q = r["q"].cpu().clone(); bt.judge_raw_mask_argmax = r["raw_mask_argmax"]
    open_ = valid & ~committed
    if bool(open_.any()):                                   # cap reached -> fall back to the STOCK rule
        bt.full_fallback = int(open_.sum())
        cur = torch.where(open_, torch.full_like(cur, mask_id), cur)
    return cur


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def da_generate(moe: MoEMeter, dr_seq: Optional[SeqCache], input_ids: torch.Tensor, *, mask_id: int,
                gen_length: int, block_length: int = 32, denoising_steps: Optional[int] = None,
                threshold: float = 0.95, draft: Optional[str] = "s8", tau_commit: float = 0.95,
                tau_acc: float = 0.70, tau_acc_q: float = 0.50, rejudge: bool = False, rejudge_k: int = 8,
                judge: str = "halves", tau_draft: float = 0.0, rejudge_redraft: bool = False,
                full_max_passes: int = FULL_MAX_PASSES, full_remask_only: bool = False,
                tau_override: Optional[float] = None, draft_after_pass: int = 0, judge_passes: int = 0,
                tau_pool_cov: float = 0.0,
                stopping_criteria_idx: Optional[Sequence[int]] = None, temperature: float = 1.0,
                top_k: int = 1, top_p: float = 1.0, greedy_exact: bool = True,
                max_new_blocks: Optional[int] = None) -> DAResult:
    """Draft-Verify decode of one prompt.  `draft=None` -> the stock decode (no drafter, no judge).
    judge: "halves" (2 judge forwards per block) | "blind" (q read from the first stock pass, 0 extra
    forwards) | "full" (draft-filled judge passes until every slot is committed; the trained regime)."""
    assert input_ids.shape[0] == 1, "single-sequence decoding only"
    assert judge in JUDGE_MODES, judge
    blind = (judge == "blind")
    full = (judge == "full")
    assert int(full_max_passes) >= 1, full_max_passes
    if blind and rejudge and rejudge_redraft:
        raise NotImplementedError("blind rejudge with redraft is not implemented; only the no-redraft variant")
    if full and rejudge:
        raise NotImplementedError("the full judge already re-judges every pass; --rejudge is halves/blind only")
    # -- the two OFF-by-default decode-rule variants ------------------------
    da_pass = int(draft_after_pass)
    assert da_pass >= 0, draft_after_pass
    if da_pass and not (blind and draft is not None):
        raise NotImplementedError("draft_after_pass (anchored drafting) is the blind rule's variant: "
                                  "it needs judge='blind' and a drafter")
    jp = int(judge_passes)
    assert jp >= 0, judge_passes
    if jp and not (full and draft is not None):
        raise NotImplementedError("judge_passes (the one-pass hybrid) is the full judge's variant: "
                                  "it needs judge='full' and a drafter")
    # the hybrid is the full judge capped at K passes: after them every uncommitted slot is re-masked
    # and the STOCK loop finishes the block (exactly what `full_max_passes` already does).
    eff_full_passes = min(int(full_max_passes), jp) if jp else int(full_max_passes)
    bl = int(block_length)
    assert moe.bl == bl
    steps = bl if denoising_steps is None else int(denoising_steps)
    assert bl % steps == 0 and bl >= steps
    if draft is not None:
        assert draft in SETTINGS, draft
        assert dr_seq is not None and dr_seq.bl == bl
        assert steps == bl, "DV assumes the stock 1-per-pass schedule (denoising_steps == block_length)"
    dev = moe.seq.dev
    input_ids = input_ids.to(dev)
    prompt_length = int(input_ids.shape[1])
    num_blocks = (prompt_length + gen_length + bl - 1) // bl
    total_length = num_blocks * bl
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=dev)
    x[:, :prompt_length] = input_ids

    moe.seq.reset(); moe.reset()
    pl = moe.prefill(input_ids)
    prefill_blocks = pl // bl
    assert prefill_blocks == prompt_length // bl
    if draft is not None:
        dr_seq.reset()
        assert dr_seq.prefill(input_ids) == pl, "drafter prefill geometry differs"
    num_transfer = get_num_transfer_tokens(bl, steps)
    stop_ids = list(stopping_criteria_idx or [])
    last_block = num_blocks if max_new_blocks is None else min(num_blocks, prefill_blocks + max_new_blocks)

    blocks: List[DABlock] = []
    stopped_early = False
    for nb in range(prefill_blocks, last_block):
        lo, hi = nb * bl, (nb + 1) * bl
        cur = x[:, lo:hi]                                   # view into x, as in decode.py
        valid = (torch.arange(lo, hi, device=dev) >= prompt_length)
        bt = DABlock(block_index=nb, start=lo, valid=valid.cpu())
        moe.set_valid(valid)                                # the pool votes over the generated slots only

        D = pD = None
        anchored = (draft is not None and da_pass > 0)     # blind: draft AFTER `da_pass` stock passes
        if draft is not None and not anchored:
            # 1. draft the valid slots (prompt tail fixed) against the drafter's prefix cache
            d = draft_block(dr_seq, cur[0].clone(), valid, lo, draft, mask_id)
            D, pD = d["tokens"], d["p_commit"].float()
            bt.n_draft_fwd = int(d["n_fwd"]); bt.judge_mode = judge
            bt.draft_tokens = D.cpu().clone(); bt.p_draft = pD.cpu().clone()
            bt.judge_code = torch.zeros(bl, dtype=torch.int8)
        elif anchored:
            # 1'. the draft is deferred to the end of stock pass `da_pass` (below); the block starts
            #     as the STOCK block (all-[MASK] valid slots) and its first passes are stock passes.
            bt.judge_mode = judge
            bt.judge_code = torch.zeros(bl, dtype=torch.int8)
        if draft is not None and full:
            # 2'. the FULL judge: draft-filled forwards, read at every open slot, until all committed
            cur[0] = judge_full_block(moe, cur[0], valid, D, lo, mask_id, tau_commit, bt,
                                      max_passes=eff_full_passes, remask_only=full_remask_only,
                                      tau_override=tau_override, tau_pool_cov=tau_pool_cov, p_draft=pD)
        elif draft is not None and not blind:
            # 2. judge with complementary halves; 3. accept rule (+ progress guarantee)
            j = judge_halves(moe, cur[0], valid, D, lo, mask_id, "judge")
            new, code = apply_accept_rule(cur[0], j["argmax"], j["p"], j["q"], D, valid,
                                          tau_commit, tau_acc, tau_acc_q, force_progress=True)
            cur[0] = new
            bt.n_judge = j["n_fwd"]; bt.n_forward += j["n_fwd"]; bt.judge_raw_mask_argmax = j["raw_mask_argmax"]
            bt.judge_argmax = j["argmax"].cpu().clone(); bt.judge_p = j["p"].cpu().clone()
            bt.judge_q = j["q"].cpu().clone(); bt.judge_code = code.cpu().clone()

        # 4. stock denoising from the current state (decode.py:217-264)
        finished = False
        for step in range(steps + 1):
            mask_index = (cur == mask_id)
            if mask_index.sum() == 0:
                moe.block(cur, lo, store_kv=True, kind="kvflush")
                bt.n_forward += 1
                finished = True
                break
            x0, x0_p, transfer_index, logits = stock_pass(moe, cur, lo, mask_id, threshold, int(num_transfer[step]),
                                                          temperature, top_k, top_p, greedy_exact, "denoise")
            bt.n_forward += 1
            bt.committed.append(transfer_index[0].detach().cpu().clone())
            bt.confidence.append(x0_p[0].detach().float().cpu().clone())
            bt.proposed.append(x0[0].detach().cpu().clone())
            cur[transfer_index] = x0[transfer_index]

            if blind and draft is not None and D is not None and (step == da_pass or rejudge):
                # read a_i / p_i / q_i from THIS forward at the slots it saw masked; accept only still-open slots
                # (with anchored drafting the accepting pass is pass `da_pass` + 1, i.e. step == da_pass)
                read = mask_index[0] & valid
                r = blind_read(logits, D, read, mask_id)
                open_ = (cur[0] == mask_id) & valid
                new, acc = blind_accept(cur[0], r["q"], D, open_, tau_acc_q, pD, tau_draft)
                cur[0] = new
                n_acc = int(acc.sum())
                bt.n_blind_accept += n_acc
                if step == da_pass:
                    bt.judge_argmax = r["argmax"].cpu().clone(); bt.judge_p = r["p"].cpu().clone(); bt.judge_q = r["q"].cpu().clone()
                    bt.judge_raw_mask_argmax = r["raw_mask_argmax"]
                    bt.judge_code[acc.cpu()] = CODE_ACCEPT_Q
                elif bool(read.any()):
                    bt.rejudge_rounds += 1; bt.rejudge_accepted += n_acc
                    bt.judge_code[acc.cpu()] = CODE_ACCEPT_LATE
            del logits

            if anchored and step == da_pass - 1:
                # ANCHORED DRAFT: the `da_pass` stock passes above have committed some slots; draft only
                # the STILL-OPEN ones with those commits held fixed as anchors (draft_block masks and
                # fills `valid`, and keeps `final`'s tokens everywhere else -- anchors and prompt tail).
                open_ = (cur[0] == mask_id) & valid
                anchors = valid & ~open_
                d = draft_block(dr_seq, cur[0].clone(), open_, lo, draft, mask_id)
                D, pD = d["tokens"], d["p_commit"].float()
                bt.n_draft_fwd = int(d["n_fwd"])
                bt.draft_tokens = D.cpu().clone(); bt.p_draft = pD.cpu().clone()
                bt.anchored = anchors.cpu().clone(); bt.n_anchor = int(anchors.sum())
                # defaults, so a block that is already finished here still has full judge arrays
                bt.judge_argmax = torch.full((bl,), -1, dtype=torch.long)
                bt.judge_p = torch.full((bl,), float("nan"), dtype=torch.float32)
                bt.judge_q = torch.full((bl,), float("nan"), dtype=torch.float32)

            if rejudge and draft is not None and not blind:
                open_ = (cur[0] == mask_id)
                if int(open_.sum()) >= rejudge_k:
                    d2 = draft_block(dr_seq, cur[0].clone(), open_, lo, draft, mask_id)
                    D2 = d2["tokens"]
                    j2 = judge_halves(moe, cur[0], open_, D2, lo, mask_id, "rejudge")
                    new, code2 = apply_accept_rule(cur[0], j2["argmax"], j2["p"], j2["q"], D2, open_,
                                                   tau_commit, tau_acc, tau_acc_q, force_progress=False)
                    cur[0] = new
                    bt.rejudge_rounds += 1; bt.n_rejudge += j2["n_fwd"]; bt.n_forward += j2["n_fwd"]
                    bt.n_redraft_fwd += int(d2["n_fwd"])
                    bt.rejudge_committed += int((code2 == CODE_COMMIT).sum())
                    bt.rejudge_accepted += int(((code2 == CODE_ACCEPT_EQ) | (code2 == CODE_ACCEPT_Q)).sum())
                    bt.rejudges.append({"open": open_.cpu().clone(), "draft": D2.cpu().clone(),
                                        "argmax": j2["argmax"].cpu().clone(), "p": j2["p"].cpu().clone(),
                                        "q": j2["q"].cpu().clone(), "code": code2.cpu().clone()})
        assert finished, "block did not reach the kv flush"

        bt.tokens = cur[0].detach().cpu().clone()
        if draft is not None:
            dr_seq.flush(cur[0], lo)
        blocks.append(bt)

        for sid in stop_ids:
            if (x[:, prompt_length:] == sid).any():
                stopped_early = True
                break
        if stopped_early:
            break

    return DAResult(x=x, prompt_length=prompt_length, total_length=total_length, block_length=bl,
                    prefill_blocks=prefill_blocks, num_blocks=num_blocks, blocks=blocks,
                    stopped_early=stopped_early, n_forward=len(moe.records), draft=draft,
                    judge=None if draft is None else judge,
                    draft_after_pass=(da_pass if draft is not None else 0),
                    judge_passes=(jp if draft is not None else 0))


# ---------------------------------------------------------------------------
# records (same fields as run_research_baselines.decode_item, plus the DV extras)
# ---------------------------------------------------------------------------
def make_record(res: DAResult, tok, item: dict, prompt_ids: torch.Tensor, moe: MoEMeter,
                dr_seq: Optional[SeqCache], eos_ids: Sequence[int], mask_id: int, wall_s: float,
                peak_mem_gib: Optional[float] = None, adapter: Optional[str] = None,
                adapter_hash: Optional[str] = None) -> dict:
    gen = res.generated_ids()[0].cpu()
    nonmask = gen[gen != mask_id]
    eos_pos = [int(i) for i in torch.nonzero(torch.isin(nonmask, torch.tensor(list(eos_ids)))).flatten().tolist()]
    n_to_eos = (eos_pos[0] + 1) if eos_pos else int(nonmask.numel())
    text = tok.decode(nonmask.tolist(), skip_special_tokens=True)
    by_kind = moe.by_kind()
    denoise_lf = by_kind["denoise"]["lf"]
    judge_lf = by_kind["judge"]["lf"] + by_kind["rejudge"]["lf"]
    uniq_all = sum(v["uniq"] for v in by_kind.values())
    lf_all = sum(v["lf"] for v in by_kind.values())
    blocks = res.blocks
    dr_by_kind: Dict[str, dict] = {k: {"n": 0, "ms": 0.0} for k in DRAFTER_KIND_CODE}
    dr_recs = drafter_records(dr_seq)
    inv = {v: k for k, v in DRAFTER_KIND_CODE.items()}
    for code, ms in dr_recs:
        dr_by_kind[inv[code]]["n"] += 1; dr_by_kind[inv[code]]["ms"] += ms
    moe_ms = sum(v["ms"] for v in by_kind.values())
    dr_ms = sum(v["ms"] for v in dr_by_kind.values())
    rec = {
        "idx": item["idx"], "rank": item.get("rank"), "split": item["split"],
        "prompt_tokens": int(prompt_ids.shape[1]),
        "pred": text, "gold": item["gold"], "gen_ids": nonmask.tolist(),
        "n_gen_nonmask": int(nonmask.numel()), "n_gen_to_eos": n_to_eos,
        "stopped_eos": bool(res.stopped_early),
        "truncated": bool(not res.stopped_early and len(blocks) == res.num_blocks - res.prefill_blocks),
        "n_blocks": len(blocks), "n_forward": res.n_forward,
        "n_prefill": by_kind["prefill"]["n"], "n_denoise": by_kind["denoise"]["n"], "n_flush": by_kind["kvflush"]["n"],
        "n_judge": by_kind["judge"]["n"], "n_rejudge": by_kind["rejudge"]["n"],
        # every block forward except the kv flush (judge + rejudge + stock denoise); in stock and
        # blind mode n_judge == n_rejudge == 0, so this is the stock passes/block of the reference cells
        "passes_per_block": [b.block_forwards_ex_flush for b in blocks],
        "stock_passes_per_block": [b.n_steps for b in blocks],
        "n_valid_per_block": [int(b.valid.sum()) for b in blocks],
        "commits_per_pass": [int(c.sum()) for b in blocks for c in b.committed],
        "unique_expert_layer_loads": uniq_all, "layer_forwards": lf_all,
        "uniq_per_layer_forward_denoise": (by_kind["denoise"]["uniq"] / denoise_lf) if denoise_lf else None,
        "uniq_per_layer_forward_judge": ((by_kind["judge"]["uniq"] + by_kind["rejudge"]["uniq"]) / judge_lf) if judge_lf else None,
        "uniq_per_layer_forward_all": (uniq_all / lf_all) if lf_all else None,
        "seconds": round(wall_s, 3),
        "moe_seconds": round(moe_ms / 1e3, 3), "drafter_seconds": round(dr_ms / 1e3, 3),
        "ms_by_kind": {k: round(v["ms"], 1) for k, v in by_kind.items()},
        "uniq_by_kind": {k: v["uniq"] for k, v in by_kind.items()},
        "forwards": moe.records,
        "peak_mem_gib": peak_mem_gib,
        # -- DV extras ----------------------------------------------------
        "draft": res.draft,
        "judge": res.judge,
        "judge_mode": res.judge or "stock",
        "pool_M": (moe.pool.M if (moe.pool is not None and moe.pool.on) else None),
        "pool_rule": (moe.pool.rule if (moe.pool is not None and moe.pool.on) else None),
        "pool_flush": bool(moe.pool_flush) if (moe.pool is not None and moe.pool.on) else False,
        "pool_by_kind": {k: v["pool"] for k, v in by_kind.items()},
        "adapter": adapter, "adapter_hash": adapter_hash,
        "tentative_per_block": [b.n_tentative for b in blocks],
        "remask_per_block": [b.n_remask for b in blocks],
        "full_commit_eq_draft_per_block": [b.full_commit_eq_draft for b in blocks],
        "full_fallback_per_block": [b.full_fallback for b in blocks],
        "blind_accept_per_block": [b.n_blind_accept for b in blocks],
        # -- the OFF-by-default decode-rule variants (0 / all-zero on every earlier cell) --
        "draft_after_pass": int(res.draft_after_pass),
        "judge_passes": int(res.judge_passes),
        "anchor_per_block": [b.n_anchor for b in blocks],
        "moe_forwards_per_block": [b.n_forward for b in blocks],
        "judge_forwards_per_block": [b.n_judge for b in blocks],
        "drafter_forwards_per_block": [b.n_draft_fwd for b in blocks],
        "accepted_per_block": [int(b.accepted.sum()) for b in blocks],
        "accepted_eq_per_block": [int((b.judge_code == CODE_ACCEPT_EQ).sum()) if b.judge_code is not None else 0 for b in blocks],
        "accepted_q_per_block": [int((b.judge_code == CODE_ACCEPT_Q).sum()) if b.judge_code is not None else 0 for b in blocks],
        "accepted_late_per_block": [int((b.judge_code == CODE_ACCEPT_LATE).sum()) if b.judge_code is not None else 0 for b in blocks],
        "committed_judge_per_block": [int(b.committed_judge.sum()) for b in blocks],
        "forced_per_block": [bool(b.forced) for b in blocks],
        "full_passes_per_block": [list(b.full_passes) for b in blocks],   # per judge pass: commit / tentative / remask / forced
        "draft_agreement_per_block": [None if np.isnan(b.draft_agreement) else round(b.draft_agreement, 5) for b in blocks],
        "judge_raw_mask_argmax": sum(b.judge_raw_mask_argmax for b in blocks),
        "rejudge_rounds_per_block": [b.rejudge_rounds for b in blocks],
        "rejudge_accepted_per_block": [b.rejudge_accepted for b in blocks],
        "rejudge_committed_per_block": [b.rejudge_committed for b in blocks],
        "redraft_forwards_per_block": [b.n_redraft_fwd for b in blocks],
        "drafter_n_by_kind": {k: v["n"] for k, v in dr_by_kind.items()},
        "drafter_ms_by_kind": {k: round(v["ms"], 1) for k, v in dr_by_kind.items()},
        "drafter_forwards": dr_recs,
    }
    if "task_id" in item:
        rec["task_id"] = item["task_id"]
    return rec


def save_trace(path: str, res: DAResult, block: int) -> None:
    """run_research_baselines.save_trace's arrays (per STOCK pass) + draft / judge arrays per block."""
    blocks = res.blocks
    nb = len(blocks)
    if nb == 0:
        return
    Pn = max(max(b.n_steps for b in blocks), 1)
    proposed = np.full((nb, Pn, block), -1, np.int32)
    conf = np.full((nb, Pn, block), np.nan, np.float32)
    committed = np.zeros((nb, Pn, block), bool)
    valid = np.zeros((nb, block), bool)
    n_pass = np.zeros(nb, np.int32); start = np.zeros(nb, np.int32)
    tokens = np.full((nb, block), -1, np.int32)
    draft_tokens = np.full((nb, block), -1, np.int32)
    judge_argmax = np.full((nb, block), -1, np.int32)
    judge_p = np.full((nb, block), np.nan, np.float32)
    judge_q = np.full((nb, block), np.nan, np.float32)
    judge_code = np.zeros((nb, block), np.int8)
    accepted = np.zeros((nb, block), bool)
    committed_judge = np.zeros((nb, block), bool)
    p_draft = np.full((nb, block), np.nan, np.float32)
    anchored = np.zeros((nb, block), bool)      # anchored drafting: slots committed BEFORE the draft
    for i, b in enumerate(blocks):
        n_pass[i] = b.n_steps; start[i] = b.start; valid[i] = b.valid.numpy()
        tokens[i] = b.tokens.numpy()
        for s in range(b.n_steps):
            proposed[i, s] = b.proposed[s].numpy(); conf[i, s] = b.confidence[s].numpy()
            committed[i, s] = b.committed[s].numpy()
        if b.draft_tokens is not None:
            draft_tokens[i] = b.draft_tokens.numpy(); p_draft[i] = b.p_draft.numpy()
            judge_argmax[i] = b.judge_argmax.numpy(); judge_p[i] = b.judge_p.numpy(); judge_q[i] = b.judge_q.numpy()
            judge_code[i] = b.judge_code.numpy(); accepted[i] = b.accepted.numpy(); committed_judge[i] = b.committed_judge.numpy()
        if b.anchored is not None:
            anchored[i] = b.anchored.numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # `anchored` is written ONLY by an anchored-drafting run, so every other cell's trace keeps
    # exactly the arrays (and the bytes) it has always had.
    extra = {"anchored": anchored} if any(b.anchored is not None for b in blocks) else {}
    np.savez_compressed(path, proposed=proposed, confidence=conf, committed=committed, valid=valid,
                        n_pass=n_pass, block_start=start, tokens=tokens,
                        prompt_length=np.int32(res.prompt_length), block_length=np.int32(block),
                        draft_tokens=draft_tokens, p_draft=p_draft, judge_argmax=judge_argmax, judge_p=judge_p,
                        judge_q=judge_q, judge_code=judge_code, accepted=accepted, committed_judge=committed_judge,
                        **extra)


def load_trace(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------------------
# one item, timed (used by the runner)
# ---------------------------------------------------------------------------
def decode_item(moe: MoEMeter, dr_seq: Optional[SeqCache], tok, item: dict, prompt_ids: torch.Tensor, *,
                mask_id: int, eos_ids: Sequence[int], gen_length: int,
                adapter: Optional[str] = None, adapter_hash: Optional[str] = None, **gen_kw):
    dev = moe.seq.dev
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    res = da_generate(moe, dr_seq, prompt_ids, mask_id=mask_id, gen_length=gen_length,
                      stopping_criteria_idx=list(eos_ids), **gen_kw)
    if cuda:
        torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    peak = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None
    rec = make_record(res, tok, item, prompt_ids, moe, dr_seq, eos_ids, mask_id, wall, peak,
                      adapter=adapter, adapter_hash=adapter_hash)
    return rec, res
