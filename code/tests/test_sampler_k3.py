"""CPU tests for the judge round's K1 / K2 / K3 state sampler (code/judge_round/sampler_k3.py), its train_c1b wrapper
(train_c1b_k3.py) and swap_drafts.py --redrafts.  No model, no GPU.

Pins: K3 states hold the truth at committed slots, D0 at tentative slots, D1 at re-drafted holes ([MASK] where the
re-draft left a hole empty); K2 states hold [MASK] at every hole; K1 is the parent's draft_pass1 state; blocks and
items without a re-draft record, and every regime other than "sample", are the parent's states bit for bit; the
splits stay K3Samplers; the wrapper swaps the sampler into train_c1b; swap_drafts writes the three re-draft arrays.

    python -m pytest -q code/tests/test_sampler_k3.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests"), os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_drafter_lora as TDL                                           # noqa: E402  (fake 1B-corpus items)
from draftalign import train_fast as F                                    # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                # noqa: E402
from drafter_lora import data as D                                        # noqa: E402
from judge_round import sampler_k3 as K                                   # noqa: E402

BL = 32
MASK = TDL.MASK                                                           # 39; fake tokens are < 37
SEED = 3
EPOCHS = range(4)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def redraft_record(it, seed=0, skip_block=None):
    """A re-draft record as redraft_items.py defines it: per block a vanilla pass p in [1, n_pass), holes =
    valid & ~committed[:p].any(0) & (D0 != T); D1 right on about half the holes, wrong on the rest, and the first
    hole of a block with two or more left empty (-1)."""
    rng = np.random.default_rng(100 + seed)
    nb = it["tokens"].shape[0]
    rd = np.full((nb, BL), -1, np.int32)
    rc = np.zeros((nb, BL), np.uint8)
    pp = np.zeros(nb, np.int16)
    for b in range(nb):
        n_pass = int(it["n_pass"][b])
        if n_pass < 2 or b == skip_block:
            continue
        p = int(rng.integers(1, n_pass))
        holes = it["valid"][b] & ~it["committed"][b, :p].any(0) & (it["draft"][b] != it["tokens"][b])
        js = np.flatnonzero(holes)
        if len(js) == 0:
            continue
        pp[b] = p
        d1 = np.where(rng.random(BL) < 0.5, it["tokens"][b], (it["tokens"][b] + 1) % 37)
        rd[b, js] = d1[js]
        rc[b, js] = rng.integers(1, 256, len(js))
        if len(js) >= 2:
            rd[b, js[0]], rc[b, js[0]] = -1, 0
    return {"redraft": rd, "redraft_conf": rc, "redraft_pass": pp}


def index_row(it, pid, i, dom, rel):
    return {"file": rel, "pid": pid, "idx": i, "domain": dom, "split": f"c1b_{dom}", "correct": 1,
            "usable": [True] * len(it["usable"]), "has_draft": [True] * len(it["usable"]),
            "block_start": it["block_start"].tolist(), "n_blocks": len(it["usable"]),
            "seq_len": int(it["block_start"][-1]) + BL, "corpus_tokens": int(it["prompt_length"]) + BL * len(it["usable"])}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Item 0 (maths) carries a re-draft record on blocks 0-2 (block 3 has redraft_pass 0) and one judge-accepted
    hole; item 1 (code) has none."""
    root = str(tmp_path_factory.mktemp("k3"))
    os.makedirs(os.path.join(root, "items"))
    items = []
    with open(os.path.join(root, "index.jsonl"), "w") as fh:
        for i, dom in enumerate(("maths", "code")):
            it = TDL.fake_item(seed=i, pl=45, nb=4, domain=dom)
            pid = it.pop("pid")
            if i == 0:
                it.update(redraft_record(it, seed=i, skip_block=3))
                b = int(np.argmax((it["redraft"] >= 0).sum(1)))           # the block with the most re-drafted holes
                filled = np.flatnonzero(it["redraft"][b] >= 0)
                it["accepted"][b, filled[0]] = True                    # a hole the judge accepted: committed context
            rel = os.path.join("items", f"{pid}.npz")
            np.savez_compressed(os.path.join(root, rel), **it)
            fh.write(json.dumps(index_row(it, pid, i, dom, rel)) + "\n")
            items.append(it)
    return os.path.join(root, "index.jsonl"), items


def cfg_of(index):
    return SamplerConfig(index_path=index, seed=SEED, max_seq=1856, mask_id=MASK)


def eligible(s):
    out = []
    for k, (i, b) in enumerate(s.examples):
        it = s._item(i)
        if "redraft" in it and int(it["redraft_pass"][b]) > 0 and (it["redraft"][b] >= 0).any():
            out.append(k)
    return out


def expected(it, b, kind):
    """The K2 / K3 state of block b, computed straight from the item arrays."""
    T = it["tokens"][b].astype(np.int64)
    valid = it["valid"][b].astype(bool)
    d0 = it["draft"][b].astype(np.int64)
    d1 = it["redraft"][b].astype(np.int64)
    p = int(it["redraft_pass"][b])
    com = (it["committed"][b, :p].any(0) | it["accepted"][b]) & valid
    opn = valid & ~com
    ten = opn & (d0 == T)
    hol = opn & ~ten
    blk = T.copy()
    blk[ten] = d0[ten]
    blk[hol] = MASK
    conf = np.zeros(BL, np.uint8)
    conf[ten] = it["draft_conf"][b][ten]
    fill = np.zeros(BL, bool)
    if kind == "K3":
        fill = hol & (d1 >= 0)
        blk[fill] = d1[fill]
        conf[fill] = it["redraft_conf"][b][fill]
    return {"p": p, "com": com, "opn": opn, "ten": ten, "hol": hol, "fill": fill, "blk": blk, "conf": conf}


def same_state(a, b, meta=True):
    assert torch.equal(a.input_ids, b.input_ids) and a.active_lo == b.active_lo
    assert torch.equal(a.valid, b.valid) and torch.equal(a.targets, b.targets) and torch.equal(a.open_mask, b.open_mask)
    assert (a.draft_conf is None) == (b.draft_conf is None)
    assert a.draft_conf is None or torch.equal(a.draft_conf, b.draft_conf)
    if meta:
        assert a.meta == b.meta


def check_k23(s, base, k, epoch, kind):
    i, b = s.examples[k]
    it = s._item(i)
    st = s.state(k, epoch, "sample")
    ex = expected(it, b, kind)
    lo = st.active_lo
    blk = st.input_ids[lo:lo + BL].numpy()
    assert st.meta["kind"] == kind and st.meta["pass"] == ex["p"] and st.meta["regime"] == "sample"
    assert np.array_equal(blk, ex["blk"])
    assert np.array_equal(blk[ex["com"]], it["tokens"][b][ex["com"]])            # committed: the truth
    assert np.array_equal(blk[ex["ten"]], it["draft"][b][ex["ten"]])             # tentative: D0
    assert np.array_equal(st.open_mask.numpy(), ex["ten"] | ex["hol"])           # loss rows: tentative + holes
    assert np.array_equal(st.open_mask.numpy(), ex["opn"])
    assert np.array_equal(st.targets.numpy(), it["tokens"][b])
    assert np.array_equal(st.draft_conf.numpy(), ex["conf"])
    assert torch.equal(st.input_ids[:lo], base.state(k, epoch, "draft_pass1").input_ids[:lo])   # clean prefix
    # train_fast.block_hidden's draft flag: every open slot that holds a token
    flag = (st.open_mask & st.valid & (st.input_ids[lo:lo + BL] != MASK)).numpy()
    assert np.array_equal(flag, ex["ten"] | ex["fill"])
    assert st.meta["n_tentative"] == int(ex["ten"].sum()) and st.meta["n_holes"] == int(ex["hol"].sum())
    assert st.meta["n_draft_filled"] == int(flag.sum()) and st.meta["n_open"] == int(ex["opn"].sum())
    assert st.meta["has_draft"] and st.meta["domain"] == "maths" and st.meta["tokens_share"] > 0
    same_state(st, s.state(k, epoch, "sample"))                                  # deterministic
    return st, ex


# ---------------------------------------------------------------------------
# the fixture is what the tests need
# ---------------------------------------------------------------------------
def test_fixture_has_every_slot_kind(corpus):
    index, items = corpus
    s = K.K3Sampler(cfg_of(index), k_mix=(0, 0, 1))
    ks = eligible(s)
    assert len(ks) >= 2 and all(s.examples[k][0] == 0 for k in ks)
    tot = {"ten": 0, "fill": 0, "empty": 0, "acc": 0}
    for k in ks:
        i, b = s.examples[k]
        ex = expected(items[0], b, "K3")
        tot["ten"] += int(ex["ten"].sum())
        tot["fill"] += int(ex["fill"].sum())
        tot["empty"] += int((ex["hol"] & ~ex["fill"]).sum())
        tot["acc"] += int((items[0]["accepted"][b] & (items[0]["redraft"][b] >= 0)).sum())
    assert all(v > 0 for v in tot.values()), tot


# ---------------------------------------------------------------------------
# (a) K3, (b) K2, (c) K1
# ---------------------------------------------------------------------------
def test_k3_state_holds_truth_d0_and_d1(corpus):
    index, items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=(0, 0, 1)), StateSampler(cfg_of(index))
    for k in eligible(s):
        for e in EPOCHS:
            st, ex = check_k23(s, base, k, e, "K3")
            i, b = s.examples[k]
            d1 = items[0]["redraft"][b]
            lo = st.active_lo
            blk = st.input_ids[lo:lo + BL].numpy()
            assert np.array_equal(blk[ex["fill"]], d1[ex["fill"]])                # re-drafted holes: D1
            assert (blk[ex["hol"] & ~ex["fill"]] == MASK).all()                  # empty re-draft: still [MASK]
            acc = items[0]["accepted"][b] & (d1 >= 0)
            assert np.array_equal(blk[acc], items[0]["tokens"][b][acc])          # accepted hole: committed truth


def test_k2_state_masks_every_hole(corpus):
    index, _items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=(0, 1, 0)), StateSampler(cfg_of(index))
    for k in eligible(s):
        for e in EPOCHS:
            st, ex = check_k23(s, base, k, e, "K2")
            lo = st.active_lo
            assert (st.input_ids[lo:lo + BL].numpy()[ex["hol"]] == MASK).all()
            assert (st.draft_conf.numpy()[ex["hol"]] == 0).all()
            assert st.meta["n_redrafted"] == 0


def test_k1_is_the_parents_pass1_draft_state(corpus):
    index, _items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=(1, 0, 0)), StateSampler(cfg_of(index))
    for k in eligible(s):
        for e in EPOCHS:
            st = s.state(k, e, "sample")
            assert st.meta["kind"] == "K1" and st.meta["pass"] == 0
            same_state(st, base.state(k, e, "draft_pass1"), meta=False)


def test_default_mix_draws_all_three_kinds_in_proportion(corpus):
    index, _items = corpus
    s = K.K3Sampler(cfg_of(index))
    assert s.k_mix == K.DEFAULT_K_MIX
    n = 400
    for k in eligible(s):
        kinds = [s.state(k, e, "sample").meta["kind"] for e in range(n)]
        for kind, want in zip(K.KINDS, K.DEFAULT_K_MIX):
            assert abs(kinds.count(kind) / n - want) < 0.08, (kind, kinds.count(kind) / n)


# ---------------------------------------------------------------------------
# (d) no re-draft record -> the parent, (e) other regimes -> the parent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mix", [(0, 0, 1), (0, 1, 0), K.DEFAULT_K_MIX])
def test_blocks_without_a_redraft_are_the_parents(corpus, mix):
    index, items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=mix), StateSampler(cfg_of(index))
    assert s.examples == base.examples
    ks = [k for k in range(len(s)) if k not in set(eligible(s))]
    assert any(s.examples[k][0] == 1 for k in ks)                               # the item without a record
    assert any(s.examples[k] == (0, 3) for k in ks)                             # its block with redraft_pass 0
    for k in ks:
        for e in EPOCHS:
            same_state(s.state(k, e, "sample"), base.state(k, e, "sample"))


@pytest.mark.parametrize("regime", ["draft_pass1", "allmask_pass1"])
def test_other_regimes_are_the_parents(corpus, regime):
    index, _items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=(0, 0, 1)), StateSampler(cfg_of(index))
    assert {s.examples[k][0] for k in range(len(s))} == {0, 1}
    for k in range(len(s)):
        for e in EPOCHS:
            same_state(s.state(k, e, regime), base.state(k, e, regime))
            s.global_state(k, e, regime)


# ---------------------------------------------------------------------------
# splits, the trainer's grouping, the k_mix parser
# ---------------------------------------------------------------------------
def test_splits_stay_k3_with_the_parents_items(corpus):
    index, _items = corpus
    s, base = K.K3Sampler(cfg_of(index), k_mix=(0.2, 0.3, 0.5)), StateSampler(cfg_of(index))
    for got, want in ((s.split_holdout_by_domain(1), base.split_holdout_by_domain(1)),
                      (s.split_holdout(1), base.split_holdout(1))):
        for g, w in zip(got, want):
            assert type(g) is K.K3Sampler and g.k_mix == (0.2, 0.3, 0.5)
            assert g.entries == w.entries and g.examples == w.examples and g.root == w.root


def test_k3_states_group_like_the_parents(corpus):
    index, _items = corpus
    s = K.K3Sampler(cfg_of(index), k_mix=(0, 0, 1))
    sts = [s.state(k, 0, "sample") for k in range(len(s))]
    mb = F.group_states(sts, BL)
    assert len(mb.prefixes) == 2 and len(mb.item_of_state) == len(sts)


def test_parse_k_mix():
    assert K.parse_k_mix("0.15,0.25,0.60") == (0.15, 0.25, 0.60)
    for bad in ("0.5,0.5", "0.5,0.5,0.5", "-0.1,0.5,0.6"):
        with pytest.raises(AssertionError):
            K.parse_k_mix(bad)


# ---------------------------------------------------------------------------
# the train_c1b wrapper (dry run: data plan only, no model)
# ---------------------------------------------------------------------------
def test_wrapper_swaps_the_sampler_into_train_c1b(corpus, tmp_path, monkeypatch):
    from judge_round import train_c1b_k3 as W
    TC = W.TC
    orig = TC.StateSampler
    monkeypatch.setattr(TC.P, "safe_out", lambda p: p)
    seen = []
    real_plan = TC.plan_epoch
    monkeypatch.setattr(TC, "plan_epoch", lambda sampler, *a, **kw: seen.append(sampler) or real_plan(sampler, *a, **kw))
    index, _items = corpus
    out = str(tmp_path / "run")
    res = W.main(["--k-mix", "0.1,0.2,0.7", "--index", index, "--out", out, "--dry-run", "--holdout-per-domain", "0"])
    assert res["steps"] >= 0 and len(seen) == 1
    assert type(seen[0]) is K.K3Sampler and seen[0].k_mix == (0.1, 0.2, 0.7) and len(seen[0]) > 0
    assert json.load(open(os.path.join(out, "config.json")))["k_mix"] == "0.1,0.2,0.7"
    assert TC.StateSampler is orig                                              # restored after the run


# ---------------------------------------------------------------------------
# (f) swap_drafts.py --redrafts
# ---------------------------------------------------------------------------
def test_swap_drafts_writes_the_redraft_arrays(tmp_path):
    from judge_round import swap_drafts as SW
    idx = TDL.write_corpus(str(tmp_path / "c"), n=3)
    entries = D.load_entries([idx])
    pids = [str(e["pid"]) for e in entries]
    pj = str(tmp_path / "pids.json")
    json.dump({"pids": pids}, open(pj, "w"))
    rng = np.random.default_rng(0)
    drafts, records = {}, {}
    for j, e in enumerate(entries):
        it = D.load_item(e)
        dr = np.where(it["draft"] >= 0, np.where(rng.random(it["draft"].shape) < 0.5, it["tokens"],
                                                 (it["tokens"] + 2) % 37), -1).astype(np.int32)
        cf = np.where(dr >= 0, rng.integers(1, 256, dr.shape), 0).astype(np.uint8)
        drafts[e["pid"]] = (dr, cf)
        records[e["pid"]] = redraft_record(dict(it, draft=dr), seed=j)
    os.makedirs(tmp_path / "drafts")
    np.savez(tmp_path / "drafts" / "drafts.shard0of1.npz", pids=np.array(pids),
             nb=np.array([len(drafts[p][0]) for p in pids]), draft=np.concatenate([drafts[p][0] for p in pids]),
             conf=np.concatenate([drafts[p][1] for p in pids]))

    def write_redrafts(d, groups):
        os.makedirs(d)
        for si, grp in enumerate(groups):
            np.savez(os.path.join(d, f"redrafts.shard{si}of{len(groups)}.npz"), pids=np.array(grp),
                     nb=np.array([len(records[p]["redraft_pass"]) for p in grp]),
                     pass_p=np.concatenate([records[p]["redraft_pass"] for p in grp]),
                     redraft=np.concatenate([records[p]["redraft"] for p in grp]),
                     rconf=np.concatenate([records[p]["redraft_conf"] for p in grp]))

    with_rd = [pids[0], pids[2]]
    write_redrafts(str(tmp_path / "redrafts"), [[pids[0]], [pids[2]]])               # two shards
    out = str(tmp_path / "out")
    assert SW.main(["--index", idx, "--pids", pj, "--drafts", str(tmp_path / "drafts"),
                    "--redrafts", str(tmp_path / "redrafts"), "--out", out]) == 0
    summ = json.load(open(os.path.join(out, "build_summary.json")))
    assert summ["items"] == 3 and summ["with_redraft"] == 2 and summ["without_redraft"] == 1
    rows = {r["pid"]: r for r in map(json.loads, open(os.path.join(out, "index.jsonl")))}
    for pid in pids:
        z = np.load(os.path.join(out, rows[pid]["file"]))
        nb = z["draft"].shape[0]
        assert np.array_equal(z["draft"], drafts[pid][0]) and np.array_equal(z["draft_conf"], drafts[pid][1])
        if pid in with_rd:
            assert z["redraft"].shape == (nb, BL) and z["redraft"].dtype == np.int32
            assert z["redraft_conf"].shape == (nb, BL) and z["redraft_conf"].dtype == np.uint8
            assert z["redraft_pass"].shape == (nb,) and z["redraft_pass"].dtype == np.int16
            assert np.array_equal(z["redraft"], records[pid]["redraft"])
            assert np.array_equal(z["redraft_pass"], records[pid]["redraft_pass"])
        else:
            assert not {"redraft", "redraft_conf", "redraft_pass"} & set(z.files)

    # the trainer reads the swapped corpus: K3 states on re-drafted blocks, the parent's state elsewhere
    cfg = cfg_of(os.path.join(out, "index.jsonl"))
    s, base = K.K3Sampler(cfg, k_mix=(0, 0, 1)), StateSampler(cfg)
    ks = set(eligible(s))
    assert ks and {s.entries[s.examples[k][0]]["pid"] for k in ks} <= set(with_rd)
    for k in range(len(s)):
        st = s.state(k, 0, "sample")
        if k in ks:
            assert st.meta["kind"] == "K3"
        else:
            same_state(st, base.state(k, 0, "sample"))

    # without --redrafts: the old behaviour, every item counted as without a re-draft
    out2 = str(tmp_path / "out2")
    assert SW.main(["--index", idx, "--pids", pj, "--drafts", str(tmp_path / "drafts"), "--out", out2]) == 0
    summ2 = json.load(open(os.path.join(out2, "build_summary.json")))
    assert summ2["with_redraft"] == 0 and summ2["without_redraft"] == 3 and summ2["redrafts"] is None

    # a re-draft record whose block count differs from the item's is refused
    records[pids[1]] = {k_: v[:-1] for k_, v in records[pids[1]].items()}
    write_redrafts(str(tmp_path / "bad"), [[pids[1]]])
    with pytest.raises(AssertionError):
        SW.main(["--index", idx, "--pids", pj, "--drafts", str(tmp_path / "drafts"),
                 "--redrafts", str(tmp_path / "bad"), "--out", str(tmp_path / "out3")])
