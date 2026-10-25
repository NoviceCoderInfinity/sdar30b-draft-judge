"""CPU tests for the judge round's recorded-state sampler (code/judge_round/sampler_rec.py, piece A) and its train_c1b
wrapper (code/judge_round/train_c1b_rec.py).  No GPU.

The records are real recorder output (test_index_states.py's helpers: judge_core.run_blocks, teacher-forced, through
judge_sglang's checks and pack_states), indexed by index_states.py (piece A0).

THE EQUIVALENCE TESTS compare against the PLAIN HARNESS FORWARD (operator rule of 23 Oct: training may be sped up,
the reference may not): the states are recorded with the tiny SDAR stand-in's own forward over [prefix + block]
under the block-causal mask (train_forward.stock_forward_reference, pinned to the remote forward by
test_draftalign_train.py), with the flag and the draft embedding armed by adapter.draft_flag_scope exactly as
da_decode arms them for a judge pass; that is the computation of the harness's cached block forward.  The trainer's
sped-up two-stream path (train_fast.micro_batch_loss: prefix K/V once, then the block) on the sampled states must
reproduce the recorded argmax and top-1 p at every loss row (fp32, CPU).

Pins: examples are the parent's recorded blocks; a state is the judge's view (clean prefix, the recorded block,
targets T, loss rows = the pass's open slots) and the trainer's flag equals the recorded flag; post-EOS slots follow
--post-eos (truth: the trainset's convention; exact: the recorder's view); other regimes and the held-out split are the
parent's; one pass per block per epoch, deterministic, uniform; token accounting; refusals; the wrapper swaps the
sampler into train_c1b, lands its arguments in config.json, refuses --conf-embed, runs the step-0 check after
--continue-from and stops a run whose adapter did not record the states.

    python -m pytest -q code/tests/test_sampler_rec.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests"), os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_draftalign_train as TT                                        # noqa: E402  (the tiny SDAR stand-in)
import test_index_states as TIS                                           # noqa: E402  (records + helpers)
from draftalign import adapter as A                                       # noqa: E402
from draftalign import train_fast as F                                    # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                # noqa: E402
from draftalign.train_forward import stock_forward_reference              # noqa: E402
from judge_round import index_states as IS                                # noqa: E402
from judge_round import sampler_rec as SR                                 # noqa: E402

BL, MASK = TIS.BL, TIS.MASK
SEED = 3
EPOCHS = range(4)
quiet = TIS.quiet


def cfg_of(index, **kw):
    d = dict(index_path=index, seed=SEED, max_seq=1856, mask_id=MASK)
    d.update(kw)
    return SamplerConfig(**d)


def rec_of(store, s, k):
    i, b = s.examples[k]
    return store.block(str(s.entries[i]["pid"]), b)


def same_state(a, b, meta=True):
    assert torch.equal(a.input_ids, b.input_ids) and a.active_lo == b.active_lo
    assert torch.equal(a.valid, b.valid) and torch.equal(a.targets, b.targets) and torch.equal(a.open_mask, b.open_mask)
    assert (a.draft_conf is None) == (b.draft_conf is None)
    assert a.draft_conf is None or torch.equal(a.draft_conf, b.draft_conf)
    if meta:
        assert a.meta == b.meta


# ---------------------------------------------------------------------------
# a fake-judge world (the mapping tests): six recorded items + one item without a record
# ---------------------------------------------------------------------------
UNRECORDED = (9, 41, 2, None, "code", False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("rec"))
    index, items = TIS.write_trainset(os.path.join(root, "trainset"), TIS.SPECS + [UNRECORDED])
    jcfg, ycfg = TIS.judge_cfgs()
    recs = TIS.record_items(items[:-1], jcfg, TIS.FakeJudge(1), TIS.FakeDrafter(2))
    rec_root = os.path.join(root, "record")
    TIS.write_records(rec_root, 0, [recs[:4], recs[4:]], ycfg)
    a0 = os.path.join(root, "a0")
    IS.build_index([rec_root], a0, index_paths=[index], mask_id=MASK, log=quiet)
    return SimpleNamespace(root=root, index=index, items=items, a0=a0, store=IS.StatesIndex(a0))


def sampler(world, **kw):
    return SR.RecStateSampler(cfg_of(world.index), store=world.store, **kw)


def test_examples_are_the_parents_recorded_blocks(world):
    s, base = sampler(world), StateSampler(cfg_of(world.index))
    unrec = len(world.items) - 1
    assert set(s.examples) == {(i, b) for i, b in base.examples if i != unrec} and len(s) == s.n_recorded
    summ = s.summary()
    assert summ["items_with_record"] == len(TIS.SPECS) and summ["items_without_record"] == 1
    assert summ["skipped"] == {"item_without_record": 2} and summ["recorded_examples"] == len(s)
    assert summ["repeated_examples"] == 4                                       # the science item's 4 blocks
    # max_seq drops the parent's blocks first; they are never examples
    short = SR.RecStateSampler(cfg_of(world.index, max_seq=96), store=world.store)
    assert all(int(world.items[i]["block_start"][b]) + BL <= 96 for i, b in short.examples) and len(short) < len(s)


@pytest.mark.parametrize("post_eos", SR.POST_EOS)
def test_a_state_is_the_judges_view_at_one_recorded_pass(world, post_eos):
    s, base = sampler(world, post_eos=post_eos), StateSampler(cfg_of(world.index))
    for k in range(len(s)):
        i, b = s.examples[k]
        it = world.items[i]
        rec = rec_of(world.store, s, k)
        T = np.asarray(it["tokens"][b], np.int64)
        v_tr = np.asarray(it["valid"][b], bool)
        v_rec = np.arange(BL) >= rec.start
        valid = v_tr if post_eos == "truth" else v_rec
        for e in EPOCHS:
            st = s.state(k, e, "sample")
            j = st.meta["pass"]
            lo = st.active_lo
            assert lo == int(it["block_start"][b]) and st.meta["kind"] == "rec" and st.meta["regime"] == "sample"
            assert st.meta["rec_pass"] == rec.pass0 + j and st.meta["rec_row"] == rec.row and st.meta["n_passes"] == rec.n
            blk = st.input_ids[lo:lo + BL].numpy()
            assert np.array_equal(blk, np.where(valid, rec.passes["input"][j], T))       # the judge's block
            assert torch.equal(st.input_ids[:lo], base.state(k, e, "draft_pass1").input_ids[:lo])   # clean prefix
            assert np.array_equal(st.targets.numpy(), T) and np.array_equal(st.valid.numpy(), valid)
            assert np.array_equal(st.open_mask.numpy(), rec.passes["open"][j] & valid)   # loss rows
            assert st.open_mask.any() and st.draft_conf is None
            # train_fast.block_hidden's flag == the recorded flag on the valid slots
            flag = (st.open_mask & st.valid & (st.input_ids[lo:lo + BL] != MASK)).numpy()
            assert np.array_equal(flag, world.store.a["flag"][rec.pass0 + j] & valid)
            assert st.meta["n_draft_filled"] == int(flag.sum()) and st.meta["n_open"] == int(st.open_mask.sum())
            assert st.meta["post_eos"] == post_eos and st.meta["domain"] == str(it["domain"])
            assert st.meta["repeated"] == (it["pid"].startswith("science")) and st.meta["tokens_share"] > 0
            same_state(st, s.state(k, e, "sample"))                                      # deterministic


def test_truth_and_exact_differ_only_at_post_eos_slots(world):
    tr, ex = sampler(world, post_eos="truth"), sampler(world, post_eos="exact")
    n_diff = 0
    for k in range(len(tr)):
        i, b = tr.examples[k]
        post = (np.arange(BL) >= rec_of(world.store, tr, k).start) & ~np.asarray(world.items[i]["valid"][b], bool)
        for e in EPOCHS:
            a, c = tr.state(k, e), ex.state(k, e)
            assert torch.equal(a.input_ids[:a.active_lo], c.input_ids[:c.active_lo])
            if not post.any():
                same_state(a, c, meta=False)
                continue
            if a.meta["pass"] != c.meta["pass"]:              # the eligible passes can differ: compare one pass
                continue
            lo = a.active_lo
            d = (a.input_ids[lo:lo + BL] != c.input_ids[lo:lo + BL]).numpy() | (a.open_mask != c.open_mask).numpy()
            assert not np.any(d & ~post)
            n_diff += int(d.any())
            assert not np.any(a.open_mask.numpy() & post) and not np.any(a.valid.numpy() & post)
    assert n_diff > 0


def test_other_regimes_are_the_parents(world):
    s, base = sampler(world), StateSampler(cfg_of(world.index))
    rec_set = set(s.examples)
    for regime in ("draft_pass1", "allmask_pass1"):
        for k in range(len(s)):
            kb = base.examples.index(s.examples[k])
            for e in EPOCHS:
                a, b = s.state(k, e, regime), base.state(kb, e, regime)
                assert torch.equal(a.input_ids, b.input_ids) and torch.equal(a.open_mask, b.open_mask)
                assert torch.equal(a.draft_conf, b.draft_conf)
                assert {kk: v for kk, v in a.meta.items() if kk != "example"} == \
                       {kk: v for kk, v in b.meta.items() if kk != "example"}
    assert rec_set


def test_one_pass_per_block_per_epoch_deterministic_and_uniform(world):
    s = sampler(world)
    n_ep = 300
    multi = 0
    for k in range(len(s)):
        st0 = s.state(k, 0)
        n_el = st0.meta["n_eligible"]
        picks = [s.state(k, e).meta["pass"] for e in range(n_ep)]
        rec = rec_of(world.store, s, k)
        elig = set(np.flatnonzero((rec.passes["open"] & np.asarray(st0.valid.numpy())[None]).any(1)).tolist())
        assert set(picks) <= elig and len(elig) == n_el
        if n_el >= 3:
            multi += 1
            counts = np.bincount(picks, minlength=rec.n)[sorted(elig)]
            assert counts.min() > 0 and np.abs(counts / n_ep - 1 / n_el).max() < 0.1, counts
        same_state(s.state(k, 7), s.state(k, 7))
        s2 = SR.RecStateSampler(cfg_of(world.index), store=world.store)        # a fresh sampler: the same draw
        same_state(s.state(k, 7), s2.state(k, 7))
    assert multi >= 3


def test_splits_keep_the_class_and_the_held_out_side_keeps_every_block(world):
    s, base = sampler(world, post_eos="exact"), StateSampler(cfg_of(world.index))
    for got, want in ((s.split_holdout_by_domain(1), base.split_holdout_by_domain(1)),
                      (s.split_holdout(2), base.split_holdout(2))):
        tr, ev = got
        assert type(tr) is SR.RecStateSampler and type(ev) is SR.RecStateSampler
        assert tr.require_record and not ev.require_record and tr.post_eos == ev.post_eos == "exact"
        assert tr.entries == want[0].entries and ev.entries == want[1].entries and tr.root == want[0].root
        assert ev.examples == want[1].examples                                   # every usable held-out block
        assert set(tr.examples) <= set(want[0].examples) and all(tr._recorded(i, b) for i, b in tr.examples)
        for k in range(len(ev)):                                                 # the probe: the parent's states
            same_state(ev.state(k, 0, "draft_pass1"), want[1].state(k, 0, "draft_pass1"))


def test_states_group_one_prefix_per_item_and_count_the_corpus_tokens(world):
    s = sampler(world)
    sts = [s.state(k, 0) for k in range(len(s))]
    mb = F.group_states(sts, BL)
    assert len(mb.prefixes) == len(TIS.SPECS)
    for st, j in zip(sts, mb.item_of_state):
        assert torch.equal(mb.prefixes[j][:st.active_lo], st.input_ids[:st.active_lo])
    rows = [json.loads(l) for l in open(world.index)]
    want = sum(r["corpus_tokens"] for r in rows if r["pid"] != world.items[-1]["pid"])
    assert abs(sum(st.meta["tokens_share"] for st in sts) - want) < 1e-6


def test_refusals(world, tmp_path):
    with pytest.raises(AssertionError, match="mask-id"):
        SR.RecStateSampler(SamplerConfig(index_path=world.index, seed=SEED, mask_id=MASK + 1), store=world.store)
    with pytest.raises(AssertionError, match="post_eos"):
        sampler(world, post_eos="maybe")
    with pytest.raises(AssertionError, match="pass_pick"):
        sampler(world, pass_pick="last")
    rows = [json.loads(l) for l in open(world.index)]
    with pytest.raises(ValueError, match="share"):                              # two rows with one (split, idx)
        SR.RecStateSampler(cfg_of(world.index), entries=rows + [dict(rows[0], pid="other-pid")], store=world.store)
    # states of another item: the trainset item's tokens changed after recording
    ts = str(tmp_path / "ts")
    shutil.copytree(os.path.dirname(world.index), ts)
    pid = world.items[0]["pid"]
    pth = os.path.join(ts, "items", f"{pid}.npz")
    with np.load(pth) as z:
        it = {k: z[k] for k in z.files}
    it["tokens"] = it["tokens"].copy()
    it["tokens"][1, 0] = (it["tokens"][1, 0] + 1) % TIS.NTOK
    np.savez_compressed(pth, **it)
    s = SR.RecStateSampler(cfg_of(os.path.join(ts, "index.jsonl")), store=world.store)
    k = s.examples.index((0, 1))
    with pytest.raises(ValueError, match="teacher"):
        s.state(k, 0)


# ---------------------------------------------------------------------------
# the model world: records from the PLAIN HARNESS FORWARD of a tiny flag-gated judge
# ---------------------------------------------------------------------------
MODEL_SEED, LM_SCALE = 0, 12.0
MODEL_SPECS = [(10, 40, 2, None, "maths", False), (11, 36, 3, 12, "code", False),
               (12, 64, 2, None, "chat", False), (13, 45, 2, 7, "science", False)]


def base_model():
    """The tiny SDAR stand-in (3 layers, MoE + one dense layer, vocab 40), lm_head sharpened so the judge commits
    more than the one forced slot per pass."""
    m = TT.build_model(MODEL_SEED)
    with torch.no_grad():
        m.lm_head.weight.mul_(LM_SCALE)
    return m


def judge_model(perturb_seed):
    """base_model + a NON-identity flag-gated LoRA (r 4) and draft vector, as train_c1b attaches them."""
    m = base_model()
    A.attach_lora(m, r=4, alpha=8.0, train_gates=False, draft_flag=True)
    A.freeze_base(m)
    g = torch.Generator().manual_seed(perturb_seed)
    with torch.no_grad():
        for _, p in A.trainable_parameters(m):
            p.add_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


class PlainJudge:
    """The plain HF forward the harness decodes with: the model's own forward over [teacher prefix + block] under the
    block-causal mask (stock_forward_reference), the flag and the draft embedding armed by adapter.draft_flag_scope as
    da_decode arms them for a judge pass (flag False on the prefix).  The harness's cached block forward computes the
    same thing with the prefix's K/V in the cache."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, ctx, ids, flag, kind):
        lo = int(ctx["lo"])
        out = []
        for i in range(ids.shape[0]):
            seq = torch.cat([ctx["prefix"].long(), ids[i].long()])[None]
            f = torch.zeros_like(seq, dtype=torch.bool)
            f[0, lo:] = flag[i]
            with A.draft_flag_scope(self.model, seq, f) as extra:
                assert not extra                                 # no inputs_embeds: the embed_tokens hook path
                o = stock_forward_reference(self.model, seq, block=BL, logits_to_keep=BL)
            out.append(o.logits[0, -BL:].float())
        return torch.stack(out)


def save_ckpt(model, path):
    from judge_round import train_c1b_rec as W
    opt = torch.optim.AdamW(A.parameter_groups(model, 1e-4, 1e-4, 0.0), betas=(0.9, 0.95), eps=1e-8)
    W.TC.save_ckpt(model, opt, path, {"step": 0, "tokens": 0.0, "epoch": 0, "step_in_epoch": 0,
                                      "milestones_done": []})
    return path


@pytest.fixture(scope="module")
def mworld(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("model"))
    index, items = TIS.write_trainset(os.path.join(root, "trainset"), MODEL_SPECS)
    model = judge_model(11)
    jcfg, ycfg = TIS.judge_cfgs(tau=0.9)
    recs = TIS.record_items(items, jcfg, PlainJudge(model), TIS.FakeDrafter(5))
    rec_root = os.path.join(root, "record")
    TIS.write_records(rec_root, 0, [recs], ycfg)
    a0 = os.path.join(root, "a0")
    IS.build_index([rec_root], a0, index_paths=[index], mask_id=MASK, log=quiet)
    return SimpleNamespace(root=root, index=index, items=items, model=model, a0=a0, store=IS.StatesIndex(a0),
                           ckpt=save_ckpt(model, os.path.join(root, "ckpt_recorder")),
                           other=save_ckpt(judge_model(99), os.path.join(root, "ckpt_other")))


def test_model_records_are_not_degenerate(mworld):
    a = mworld.store.a
    com = np.asarray(a["commit"]).sum(1)
    assert (com >= 2).any() and np.asarray(a["forced"]).any() and np.asarray(a["rd_on"]).any()
    assert np.asarray(a["tentative"]).any() and np.diff(mworld.store.bp).max() >= 3


def plain_vs_trainer(model, store, states):
    """max |top-1 p - recorded p| and argmax agreement over the loss rows, the trainer's two-stream forward."""
    with torch.no_grad():
        _, _, logits = F.micro_batch_loss(model, states, F.LossCfg(mask_id=MASK), block=BL, checkpoint_layers=False,
                                          want_eval=True)
    agree = rows = 0
    max_dp = 0.0
    for st, lg in zip(states, logits):
        lg = lg.clone()
        lg[:, MASK] = float("-inf")
        top, am = torch.softmax(lg, -1).max(-1)
        r = st.meta["rec_pass"]
        ra = torch.from_numpy(np.array(store.a["argmax"][r], np.int64))[st.open_mask]
        rp = torch.from_numpy(np.array(store.a["p"][r], np.float32))[st.open_mask]
        agree += int((am == ra).sum())
        rows += int(st.open_mask.sum())
        max_dp = max(max_dp, float((top - rp).abs().max()))
    return agree, rows, max_dp


@pytest.mark.parametrize("post_eos", SR.POST_EOS)
def test_the_trainers_forward_reproduces_the_plain_harness_forward_on_every_state(mworld, post_eos):
    s = SR.RecStateSampler(cfg_of(mworld.index), store=mworld.store, post_eos=post_eos)
    ks = list(range(len(s)))
    if post_eos == "truth":       # truth changes the view at post-EOS slots by design: only blocks without them
        ks = [k for k in ks if np.array_equal(s.state(k, 0).valid.numpy(),
                                              np.arange(BL) >= rec_of(mworld.store, s, k).start)]
    assert len(ks) >= 4
    for e in range(3):
        sts = [s.state(k, e) for k in ks]
        agree, rows, max_dp = plain_vs_trainer(mworld.model, mworld.store, sts)
        assert agree == rows > 0 and max_dp < 1e-5, (agree, rows, max_dp)


def test_another_adapter_does_not_reproduce_the_records(mworld):
    s = SR.RecStateSampler(cfg_of(mworld.index), store=mworld.store, post_eos="exact")
    agree, rows, _ = plain_vs_trainer(judge_model(99), mworld.store, [s.state(k, 0) for k in range(len(s))])
    assert agree < 0.95 * rows


# ---------------------------------------------------------------------------
# the train_c1b wrapper
# ---------------------------------------------------------------------------
@pytest.fixture
def W(monkeypatch):
    from judge_round import train_c1b_rec as W
    monkeypatch.setattr(W.TC.P, "safe_out", lambda p: os.path.abspath(p))
    monkeypatch.setattr(W.TC.wandb_log, "init", lambda *a, **k: W.TC.wandb_log.Noop())   # never W&B in a test
    return W


def test_wrapper_dry_run_swaps_the_sampler_and_restores_train_c1b(world, tmp_path, monkeypatch, W):
    TC = W.TC
    orig = (TC.StateSampler, TC.build_parser, TC.load_ckpt)
    seen = []
    real_plan = TC.plan_epoch
    monkeypatch.setattr(TC, "plan_epoch", lambda smp, *a, **kw: seen.append(smp) or real_plan(smp, *a, **kw))
    out = str(tmp_path / "run")
    res = W.main(["--states", world.a0, "--post-eos", "exact", "--index", world.index, "--out", out, "--dry-run",
                  "--holdout-per-domain", "0", "--mask-id", str(MASK)])
    assert res["steps"] >= 0 and len(seen) == 1
    assert type(seen[0]) is SR.RecStateSampler and seen[0].post_eos == "exact" and len(seen[0]) > 0
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["states"] == world.a0 and cfg["post_eos"] == "exact" and cfg["pass_pick"] == "uniform"
    assert cfg["step0_states"] == 256 and cfg["step0_min_agree"] == 0.95
    assert (TC.StateSampler, TC.build_parser, TC.load_ckpt) == orig


def test_wrapper_refuses_conf_embed(world, tmp_path, W):
    with pytest.raises(SystemExit, match="conf-embed"):
        W.main(["--states", world.a0, "--index", world.index, "--out", str(tmp_path / "r"), "--dry-run",
                "--conf-embed"])


def _train_argv(mworld, out, ckpt, *extra):
    return ["--states", mworld.a0, "--index", mworld.index, "--out", out, "--continue-from", ckpt,
            "--post-eos", "exact", "--rank", "4", "--alpha", "8", "--mask-id", str(MASK), "--holdout-per-domain", "0",
            "--states-per-step", "0:4", "--max-states-micro", "4", "--no-checkpoint-layers", "--log-every", "1",
            "--save-at-tokens", "1e12", "--save-every-tokens", "1e12", "--step0-states", "64", *extra]


def test_wrapper_trains_on_recorded_states_after_the_step0_check(mworld, tmp_path, monkeypatch, W):
    checks = []
    real = W.step0_check
    monkeypatch.setattr(W, "step0_check", lambda *a, **k: checks.append(real(*a, **k)) or checks[-1])
    out = str(tmp_path / "run")
    res = W.main(_train_argv(mworld, out, mworld.ckpt), model_builder=base_model)
    assert len(checks) == 1 and checks[0]["agree"] == 1.0 and checks[0]["max_abs_dp"] < 1e-5 and checks[0]["rows"] > 0
    assert res["step"] >= 2 and res["tokens"] > 0 and os.path.isdir(os.path.join(out, "final"))
    rows = [json.loads(l) for l in open(os.path.join(out, "train.rank0.jsonl"))]
    assert len(rows) == res["step"] and all(np.isfinite(r["all/loss"]) for r in rows)


def test_wrapper_stops_before_training_when_the_adapter_did_not_record_the_states(mworld, tmp_path, W):
    out = str(tmp_path / "run")
    with pytest.raises(SystemExit, match="step0"):
        W.main(_train_argv(mworld, out, mworld.other), model_builder=base_model)
    assert not os.path.exists(os.path.join(out, "final")) and not os.path.exists(os.path.join(out, "train.rank0.jsonl"))
