"""CPU tests for corpus1b/sglang_plugin/judge_core.py: the batched judge block loop reproduces the HF harness's decode
(draftalign/da_decode.py::da_generate with judge="full") on the test fakes, token for token and counter for counter.

  * the four copied helpers (read_open, full_commit_rule, stock_transfer, get_num_transfer_tokens) equal the harness's
    originals on random inputs (fp32 and bf16);
  * whole decodes: the core driven exactly like SGLang would drive it (one forward per pass for the whole batch, one
    batched drafter call per draft step) versus da_generate one item at a time, for rd0 / rd1 + ban + guard / rd2 +
    ban, a 2-item batch with different models (lock-step), an item that stops at EOS, and the pass cap (stock fallback);
  * teacher forcing: closing_ids replace the stored tokens, the pipeline's own tokens stay in the result.

No real model; the fakes are tests/test_da_decode.py's.
Run from the repo root:  python3 -m pytest -q code/corpus1b/tests/test_judge_core.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import List

import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign import da_decode as DA                                                     # noqa: E402
from draftalign.da_decode import da_generate                                               # noqa: E402
from draftalign.seqcache import SeqCache                                                   # noqa: E402
from draftalign.drafter import draft_block                                                 # noqa: E402
from draftalign.loop_guard import LoopGuard                                                # noqa: E402
from tests.test_da_decode import (ScriptedLM, seeded_fn, prompt_ids, meter_for,           # noqa: E402
                                  scripted_drafter_fn, V, MASK, EOS_IDS, BL)
from tests.test_da_decode_loopguard import cycle_drafter_fn, echo_judge_fn                # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "judge_core", os.path.join(CODE, "corpus1b", "sglang_plugin", "judge_core.py"))
JC = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = JC                      # dataclasses resolve the module through sys.modules (py3.12)
_spec.loader.exec_module(JC)

PLEN, GEN = 40, 88          # prefill block 0; decode blocks [32,64) (tail 8) , [64,96), [96,128)


# ---------------------------------------------------------------------------
# the copied helpers equal the originals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_read_open_equals_blind_read(dtype):
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        logits = (torch.randn(BL, V, generator=g) * 4).to(dtype)
        D = torch.randint(0, V, (BL,), generator=g)
        read = torch.rand(BL, generator=g) < 0.6
        a = JC.read_open(logits, D, read, MASK)
        b = DA.blind_read(logits.unsqueeze(0), D, read, MASK)
        assert torch.equal(a["argmax"], b["argmax"])
        for k in ("p", "q"):
            assert torch.equal(torch.nan_to_num(a[k], nan=-7.0), torch.nan_to_num(b[k], nan=-7.0))
        assert a["raw_mask_argmax"] == b["raw_mask_argmax"]


def test_full_commit_rule_equals_harness():
    g = torch.Generator().manual_seed(4)
    for forced in (False, True):
        for override in (None, 0.98):
            cur = torch.randint(0, V, (BL,), generator=g)
            argmax = torch.randint(0, V, (BL,), generator=g)
            D = torch.where(torch.rand(BL, generator=g) < 0.5, argmax, torch.randint(0, V, (BL,), generator=g))
            p = torch.rand(BL, generator=g) * (0.9 if forced else 1.0)
            open_ = torch.rand(BL, generator=g) < 0.7
            p[~open_] = float("nan")
            a = JC.full_commit_rule(cur, argmax, p, D, open_, 0.95, MASK, tau_override=override)
            b = DA.full_commit_rule(cur, argmax, p, D, open_, 0.95, MASK, tau_override=override)
            for x, y in zip(a[:4], b[:4]):
                assert torch.equal(x, y)
            assert a[4] == b[4]
    assert torch.equal(JC.get_num_transfer_tokens(32, 32), DA.get_num_transfer_tokens(32, 32))
    assert torch.equal(JC.get_num_transfer_tokens(13, 8), DA.get_num_transfer_tokens(13, 8))


def test_stock_transfer_equals_stock_pass():
    moe = meter_for(seeded_fn(11, temp=2.0))
    moe.prefill(prompt_ids(PLEN))
    lo = 32
    cur = torch.full((1, BL), MASK, dtype=torch.long)
    cur[0, :8] = prompt_ids(PLEN)[0, 32:40]
    cur[0, 20:24] = 7
    for n_transfer in (1, 3):
        x0, x0_p, transfer, logits = DA.stock_pass(moe, cur, lo, MASK, 0.95, n_transfer)
        a0, ap, at = JC.stock_transfer(logits[0], cur[0], MASK, 0.95, n_transfer)
        assert torch.equal(a0, x0[0]) and torch.equal(ap, x0_p[0]) and torch.equal(at, transfer[0])


# ---------------------------------------------------------------------------
# driving the core the way SGLang drives it
# ---------------------------------------------------------------------------
class FakeItem:
    def __init__(self, judge_fn, drafter_fn, plen, gen, guard_n):
        self.moe = meter_for(judge_fn)
        self.dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
        self.prompt = prompt_ids(plen)
        self.plen, self.gen = plen, gen
        self.guard = LoopGuard(guard_n, MASK) if guard_n else None
        self.blocks: List[JC.BlockRun] = []
        self.stopped = False


def core_generate(items: List[FakeItem], cfg: JC.JudgeConfig, closing=None):
    """Lock-step decode of `items` with judge_core: every block index is one batch over the live items."""
    for it in items:
        num_blocks = (it.plen + it.gen + BL - 1) // BL
        it.x = torch.full((num_blocks * BL,), MASK, dtype=torch.long)
        it.x[:it.plen] = it.prompt[0]
        it.moe.seq.reset(); it.moe.reset(); it.dr.reset()
        if it.guard is not None:
            it.guard.start_item(it.plen)
        it.moe.prefill(it.prompt); it.dr.prefill(it.prompt)
        it.prefill_blocks, it.num_blocks = it.plen // BL, num_blocks
    nb = min(it.prefill_blocks for it in items)
    while True:
        live = [it for it in items if not it.stopped and nb < it.num_blocks]
        if not live:
            break
        lo, hi = nb * BL, (nb + 1) * BL
        runs = []
        for it in live:
            if it.guard is not None:
                it.guard.begin_block(it.x[:lo].tolist(), lo)
            valid = torch.arange(lo, hi) >= it.plen
            it.moe.set_valid(valid)
            runs.append(JC.BlockRun(cur=it.x[lo:hi].clone(), valid=valid, lo=lo))

        def forward_fn(ids, flag, kind):
            outs = []
            for i, it in enumerate(live):
                outs.append(it.moe.block(ids[i].unsqueeze(0), lo, store_kv=(kind == "kvflush"), kind=kind,
                                         flag=flag[i])[0])
            return torch.stack(outs)

        def draft_fn(reqs):
            out = []
            for q in reqs:
                it = live[q.key]
                d = draft_block(it.dr, q.block_ids, q.fill, lo, "s8", MASK, static_steps=q.steps,
                                kind=None if q.kind == "draft" else "redraft", ban=q.ban,
                                guard=it.guard if q.guard else None)
                out.append(JC.DraftRes(tokens=d["tokens"], p_commit=d["p_commit"], n_fwd=int(d["n_fwd"]),
                                       guard_slots=int(d.get("guard_slots", 0)),
                                       guard_changed=int(d.get("guard_changed", 0))))
            return out

        cl = None if closing is None else [closing(it, nb) for it in live]
        JC.run_blocks(runs, cfg, forward_fn, draft_fn, closing_ids=cl)
        for it, b in zip(live, runs):
            it.x[lo:hi] = b.tokens_out
            it.dr.flush(b.tokens_out, lo)
            it.blocks.append(b)
            if any((it.x[it.plen:] == sid).any() for sid in EOS_IDS):
                it.stopped = True
        nb += 1
    return items


def harness_generate(judge_fn, drafter_fn, plen, gen, **kw):
    moe = meter_for(judge_fn)
    dr = SeqCache(ScriptedLM(drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(plen), mask_id=MASK, gen_length=gen, block_length=BL, threshold=.95,
                      draft="s8", judge="full", tau_commit=.95, stopping_criteria_idx=EOS_IDS, **kw)
    return res


def _nan_eq(a, b):
    return torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0))


def assert_same(item: FakeItem, res) -> None:
    gen_core = item.x[item.plen:]
    gen_h = res.generated_ids()[0]
    assert torch.equal(gen_core, gen_h), (gen_core.tolist(), gen_h.tolist())
    assert len(item.blocks) == len(res.blocks)
    for b, bt in zip(item.blocks, res.blocks):
        assert b.lo == bt.start
        assert torch.equal(b.cur, bt.tokens)
        assert b.passes == bt.full_passes
        assert b.n_judge == bt.n_judge and b.n_draft_fwd == bt.n_draft_fwd
        assert torch.equal(b.draft_tokens, bt.draft_tokens) and torch.equal(b.p_draft, bt.p_draft)
        assert torch.equal(b.judge_argmax, bt.judge_argmax)
        assert _nan_eq(b.judge_p, bt.judge_p) and _nan_eq(b.judge_q, bt.judge_q)
        assert b.judge_raw_mask_argmax == bt.judge_raw_mask_argmax
        for k in ("n_tentative", "n_remask", "full_commit_eq_draft", "n_redraft_rounds", "n_redraft_slots",
                  "n_redraft_fwd", "n_redraft_eq_judge", "n_redraft_eq_old", "n_loopguard_slots",
                  "n_loopguard_changed", "full_fallback"):
            assert getattr(b, k) == getattr(bt, k), (k, getattr(b, k), getattr(bt, k))
        assert b.n_stock == len(bt.committed)
        for x, y in zip(b.stock_committed, bt.committed):
            assert torch.equal(x, y)


CASES = {
    "rd0": dict(core=dict(), harness=dict()),
    "rd1_ban_guard": dict(core=dict(redraft_every=1, redraft_ban=True, loop_guard=12),
                          harness=dict(full_redraft_every=1, redraft_ban_rejected=True, loop_guard=12)),
    "rd2_ban": dict(core=dict(redraft_every=2, redraft_ban=True),
                    harness=dict(full_redraft_every=2, redraft_ban_rejected=True)),
    "cap3": dict(core=dict(max_passes=3, redraft_every=1, redraft_ban=True),
                 harness=dict(full_max_passes=3, full_redraft_every=1, redraft_ban_rejected=True)),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("judge_seed,drafter", [(1, "seeded"), (2, "cycle")])
def test_core_reproduces_the_harness_decode(case, judge_seed, drafter):
    c = CASES[case]
    judge_fn = seeded_fn(judge_seed, temp=3.0)
    drafter_fn = seeded_fn(100 + judge_seed, temp=2.0) if drafter == "seeded" else cycle_drafter_fn
    cfg = JC.JudgeConfig(mask_id=MASK, **c["core"])
    res = harness_generate(judge_fn, drafter_fn, PLEN, GEN, **c["harness"])
    item = core_generate([FakeItem(judge_fn, drafter_fn, PLEN, GEN, cfg.loop_guard)], cfg)[0]
    assert_same(item, res)
    assert sum(b.n_judge for b in item.blocks) > 0
    if case == "cap3":
        assert any(b.full_fallback > 0 for b in item.blocks) and any(b.n_stock > 0 for b in item.blocks)
    if "rd" in case and case != "rd0":
        assert any(b.n_redraft_rounds > 0 for b in item.blocks)


def test_batch_of_two_in_lock_step_equals_two_single_decodes_and_eos_stops_one_early():
    cfg = JC.JudgeConfig(mask_id=MASK, redraft_every=1, redraft_ban=True, loop_guard=12)
    j1, d1 = seeded_fn(5, temp=3.0, eos_at=(70,)), seeded_fn(55, temp=2.0)      # EOS in block 2 -> stops early
    j2, d2 = seeded_fn(6, temp=3.0), cycle_drafter_fn
    items = core_generate([FakeItem(j1, d1, PLEN, GEN, 12), FakeItem(j2, d2, PLEN, GEN, 12)], cfg)
    r1 = harness_generate(j1, d1, PLEN, GEN, full_redraft_every=1, redraft_ban_rejected=True, loop_guard=12)
    r2 = harness_generate(j2, d2, PLEN, GEN, full_redraft_every=1, redraft_ban_rejected=True, loop_guard=12)
    assert_same(items[0], r1)
    assert_same(items[1], r2)
    assert r1.stopped_early and items[0].stopped and len(items[0].blocks) < len(items[1].blocks)


def test_echo_judge_commits_everything_at_pass_one_and_the_guard_acts_on_first_drafts():
    cfg = JC.JudgeConfig(mask_id=MASK, redraft_every=1, redraft_ban=True, loop_guard=12)
    res = harness_generate(echo_judge_fn, cycle_drafter_fn, 32, 64, full_redraft_every=1,
                           redraft_ban_rejected=True, loop_guard=12)
    item = core_generate([FakeItem(echo_judge_fn, cycle_drafter_fn, 32, 64, 12)], cfg)[0]
    assert_same(item, res)
    assert all(len(b.passes) == 1 and b.passes[0]["commit"] == BL for b in item.blocks)
    assert item.blocks[1].n_loopguard_changed == 3


def test_teacher_forcing_stores_the_given_tokens_and_keeps_the_pipelines_own():
    cfg = JC.JudgeConfig(mask_id=MASK, redraft_every=1, redraft_ban=True, record_states=True)
    teacher = (torch.arange(4 * BL) % 37 + 2)                                   # never MASK / EOS (< 39)
    judge_fn, drafter_fn = seeded_fn(9, temp=3.0), seeded_fn(99, temp=2.0)

    def closing(it, nb):
        return teacher[nb * BL:(nb + 1) * BL]

    item = core_generate([FakeItem(judge_fn, drafter_fn, PLEN, GEN, 0)], cfg, closing=closing)[0]
    for b in item.blocks:
        assert torch.equal(b.tokens_out, teacher[b.lo:b.lo + BL])
        assert not torch.equal(b.cur, b.tokens_out)                              # the pipeline's own tokens differ
        assert len(b.states) == b.n_judge
        for s in b.states:
            assert s["flag"].sum() <= s["open"].sum()
            assert torch.equal(s["commit"] | s["tentative"] | s["remask"], s["open"])
    # the prefix the second block saw is the teacher's first block: a fresh harness run on the teacher-forced
    # prefix must produce the same block-2 decisions as the core's record (the states are "what the judge saw")
    assert torch.equal(item.x[32:64], teacher[32:64])
