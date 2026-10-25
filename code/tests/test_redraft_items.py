"""CPU tests for the judge round's re-draft dump (code/judge_round/redraft_items.py).  No real model, no GPU.

Pins: the pass-p recipe; the re-draft state (committed slots show T, tentative slots and holes are [MASK], invalid
slots show T, only the holes are filled); the ban (no re-draft repeats D0 at a hole); the clean prefix (every block's
flush is T); the shard files round-trip through load_redrafts; the CLI end to end with a fake drafter.
The fake drafter is tests/test_da_decode.py's cache-aware ScriptedLM.

Run from the repo root:  python -m pytest -q code/tests/test_redraft_items.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid

import numpy as np
import pytest

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_da_decode as TD                                              # noqa: E402  (ScriptedLM, V 64, MASK 63)
from drafter_lora.replay import open_slots                               # noqa: E402
from judge_round import redraft_items as RI                              # noqa: E402

BL = 32
NTOK = 37                       # item tokens live in [0, 37): never MASK (63), never EOS (61, 62)
FILLER = 40


def f_tok(pos):
    """The fake drafter's first choice at a [MASK] slot (also the wrong token of the test's first drafts)."""
    return (7 * np.asarray(pos) + 3) % NTOK


def g_tok(pos):
    """Its runner-up, always != f_tok."""
    g = (11 * np.asarray(pos) + 5) % NTOK
    return np.where(g == f_tok(pos), (g + 1) % NTOK, g)


def drafter_fn(pos, tok, is_mask, ids, cached, pkv):
    if not is_mask:
        return TD.onehot_row(tok)
    return TD.logprob_row({int(f_tok(pos)): .6, int(g_tok(pos)): .3, FILLER: .1})


def make_item(seed, pid, domain="maths", pl=35, nb=2, eos_slot=10, n_pass=6):
    """A 1B-corpus item (to_trainset format): prompt tail pl % 32 on block 0, EOS at `eos_slot` of the last block,
    a vanilla commit history, and D0 = right on ~half the slots, f_tok (the drafter's favourite) elsewhere."""
    rng = np.random.default_rng(seed)
    lo0 = (pl // BL) * BL
    tail = pl - lo0
    prompt = rng.integers(0, NTOK, pl)
    tokens = rng.integers(0, NTOK, (nb, BL))
    tokens[0, :tail] = prompt[lo0:]
    valid = np.ones((nb, BL), bool)
    valid[0, :tail] = False
    valid[-1, eos_slot + 1:] = False
    steps = rng.integers(0, n_pass, (nb, BL))
    committed = np.zeros((nb, n_pass, BL), bool)
    bi, sj = np.nonzero(valid)
    committed[bi, steps[bi, sj], sj] = True
    npass = (np.where(valid, steps, -1).max(1) + 1).astype(np.int32)
    starts = (lo0 + BL * np.arange(nb)).astype(np.int32)
    pos = starts[:, None] + np.arange(BL)[None, :]
    wrong = f_tok(pos)
    wrong = np.where(wrong == tokens, (wrong + 1) % NTOK, wrong)          # a wrong token is never T
    d0 = np.where(rng.random((nb, BL)) < 0.5, tokens, wrong)
    d0[0, :tail] = -1
    stored = np.where(rng.random((nb, BL)) < 0.5, tokens, rng.integers(0, NTOK, (nb, BL)))
    stored[0, :tail] = -1
    it = {"prompt_ids": prompt.astype(np.int32), "tokens": tokens.astype(np.int32), "valid": valid,
          "committed": committed, "n_pass": npass, "block_start": starts, "draft": stored.astype(np.int32),
          "draft_conf": np.full((nb, BL), 128, np.uint8), "has_draft": np.ones(nb, bool),
          "usable": np.ones(nb, bool), "domain": np.array(domain), "pid": pid}
    return it, d0.astype(np.int32)


def items():
    out = {}
    for s, pid, dom in ((3, "q00003", "maths"), (11, "q00011", "code")):
        it, d0 = make_item(s, pid, dom)
        out[pid] = (it, d0)
    return out


def redraft_calls(model, lo):
    """Inputs of the drafter's non-storing forwards on block lo (the re-draft steps)."""
    return [c["input_ids"][0].numpy() for c in model.calls if not c["store_kv"] and int(c["pos"][0, 0]) == lo]


def test_pass_recipe():
    pid = "q00003"
    h = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
    for b, K in ((0, 6), (1, 2), (3, 17)):
        want = int(np.random.default_rng([5, h, b]).integers(1, K))
        assert RI.sample_pass(5, pid, b, K) == want and 1 <= want <= K - 1
    assert RI.sample_pass(5, pid, 0, 1) == 0 and RI.sample_pass(5, pid, 0, 0) == 0


def test_redraft_item_state_ban_and_clean_prefix():
    seen = {"committed": 0, "tentative": 0, "holes": 0, "post_eos": 0, "banned_top": 0}
    for pid, (it, d0) in items().items():
        model = TD.ScriptedLM(drafter_fn)
        r = RI.redraft_item(model, it, d0, "cpu", seed=0, rate=4, mask_id=TD.MASK)
        T = it["tokens"].astype(np.int64)
        valid = it["valid"]
        assert r["redraft"].shape == (2, BL) and r["redraft"].dtype == np.int32
        assert r["rconf"].shape == (2, BL) and r["rconf"].dtype == np.uint8 and r["pass_p"].shape == (2,)
        fwd = 0
        for b in range(2):
            lo = int(it["block_start"][b])
            K = int(it["n_pass"][b])
            p = int(r["pass_p"][b])
            # (c) the pass is in range and is the documented recipe
            assert 0 <= p <= K - 1 and (p >= 1 or K <= 1)
            assert p == RI.sample_pass(0, pid, b, K)
            # the state, recomputed independently
            com = it["committed"][b, :p].any(0) & valid[b] if p else np.zeros(BL, bool)
            opn = valid[b] & ~com
            tent = opn & (d0[b] == T[b])
            holes = opn & (d0[b] != T[b]) & (d0[b] >= 0)
            assert np.array_equal(r["committed"][b], com) and np.array_equal(r["tentative"][b], tent)
            assert np.array_equal(r["holes"][b], holes)
            # (a) the re-draft is -1 exactly outside the holes; confidences likewise
            assert np.array_equal(r["redraft"][b] >= 0, holes)
            assert (r["rconf"][b][~holes] == 0).all() and (r["rconf"][b][holes] > 0).all()
            # (b) the ban: never D0 at a hole; where D0 is the drafter's favourite, the runner-up is taken
            rd = r["redraft"][b].astype(np.int64)
            pos = lo + np.arange(BL)
            assert not ((rd == d0[b]) & holes).any()
            want = np.where(d0[b] == f_tok(pos), g_tok(pos), f_tok(pos))
            assert np.array_equal(rd[holes], want[holes])
            seen["banned_top"] += int((holes & (d0[b] == f_tok(pos))).sum())
            # (d) tentative, committed and invalid slots are never filled, and the drafter never saw a tentative
            #     token: [MASK] there in every re-draft step; committed and invalid slots show T
            assert (r["redraft"][b][tent | com | ~valid[b]] == -1).all()
            calls = redraft_calls(model, lo)
            n_h = int(holes.sum())
            assert len(calls) == (min(8, -(-n_h // 4)) if n_h else 0)
            fwd += len(calls)
            if calls:
                assert np.array_equal(calls[0], np.where(opn, TD.MASK, T[b]))
            for ids in calls:
                assert (ids[tent] == TD.MASK).all()
                assert np.array_equal(ids[com | ~valid[b]], T[b][com | ~valid[b]])
            # the prefix stays clean: the block's flush is the 30B's tokens
            flush = [c["input_ids"][0].numpy() for c in model.calls if c["store_kv"] and int(c["pos"][0, 0]) == lo]
            assert len(flush) == 1 and np.array_equal(flush[0], T[b])
            # per-block counts
            assert int(r["n_holes"][b]) == n_h and int(r["n_repeat"][b]) == 0
            assert int(r["n_agree"][b]) == int(((rd == T[b]) & holes).sum())
            seen["committed"] += int(com.sum()); seen["tentative"] += int(tent.sum()); seen["holes"] += n_h
            seen["post_eos"] += int((~valid[b] & open_slots(it, b)).sum())
        c = r["counts"]
        assert c["items"] == 1 and c["blocks"] == 2 and c["repeat"] == 0 and c["forwards"] == fwd
        assert c["holes"] == int(r["holes"].sum()) and c["agree"] == int(r["n_agree"].sum())
    assert all(v > 0 for v in seen.values()), seen                       # every slot kind was exercised


def test_shard_npz_round_trips(tmp_path):
    its = items()
    pa, pb = sorted(its)
    drafts = {pa: its[pa][1]}                                            # pb falls back to its stored draft
    load = lambda pid: its[pid][0]                                       # noqa: E731
    res = RI.run_shard(TD.ScriptedLM(drafter_fn), [pb, pa], load, drafts, shard=0, nshards=1,
                       out_dir=str(tmp_path), device="cpu", seed=0, rate=4, mask_id=TD.MASK)
    npz, js = RI.shard_paths(str(tmp_path), 0, 1)
    assert os.path.isfile(npz) and os.path.isfile(js)
    got = RI.load_redrafts(str(tmp_path))
    assert sorted(got) == [pa, pb]
    for pid, d0 in ((pa, its[pa][1]), (pb, its[pb][0]["draft"])):
        pp, rd, rc = got[pid]
        assert pp.shape == (2,) and rd.shape == (2, BL) and rc.shape == (2, BL)
        assert pp.dtype == np.int16 and rd.dtype == np.int32 and rc.dtype == np.uint8
        ref = RI.redraft_item(TD.ScriptedLM(drafter_fn), its[pid][0], d0, "cpu", seed=0, mask_id=TD.MASK)
        assert np.array_equal(pp, ref["pass_p"]) and np.array_equal(rd, ref["redraft"])
        assert np.array_equal(rc, ref["rconf"])
    z = np.load(npz)
    assert z["pids"].tolist() == [pa, pb] and z["nb"].tolist() == [2, 2] and z["n_holes"].shape == (4,)
    rep = json.load(open(js))
    assert rep["items"] == 2 and rep["blocks"] == 4 and rep["fallback_items"] == 1 and rep["repeat"] == 0
    assert rep["holes"] == int(z["n_holes"].sum()) == sum(int((g[1] >= 0).sum()) for g in got.values())
    assert sorted(rep["per_domain"]) == ["code", "maths"]
    assert sum(v["holes"] for v in rep["per_domain"].values()) == rep["holes"]
    # sharding: shard k of n takes sorted(pids)[k::n]
    res1 = RI.run_shard(TD.ScriptedLM(drafter_fn), [pb, pa], load, drafts, shard=1, nshards=2,
                        out_dir=str(tmp_path / "s"), device="cpu", mask_id=TD.MASK)
    assert res1["items"] == 1 and list(RI.load_redrafts(str(tmp_path / "s"))) == [pb]
    assert res["seconds"] >= 0


def test_cli_end_to_end(tmp_path, monkeypatch):
    import research_protocol as P
    root = tmp_path / "corpus"
    (root / "items").mkdir(parents=True)
    rows, d0s = [], {}
    for s, dom in ((1, "maths"), (2, "code"), (4, "chat")):
        pid = f"c{s:05d}"
        it, d0 = make_item(s, pid, dom, pl=33 + 9 * s, nb=2 + s % 2, eos_slot=4 * s)
        it.pop("pid")
        np.savez_compressed(root / "items" / f"{pid}.npz", **it)
        rows.append({"file": f"items/{pid}.npz", "pid": pid, "domain": dom})
        d0s[pid] = d0
    (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "pids.json").write_text(json.dumps({"pids": [r["pid"] for r in rows]}))
    dd = tmp_path / "drafts"
    dd.mkdir()
    keep = ["c00001", "c00002"]                                          # c00004 falls back to its stored draft
    np.savez_compressed(dd / "drafts.shard0of1.npz", pids=np.array(keep),
                        nb=np.array([d0s[p].shape[0] for p in keep], np.int32),
                        draft=np.concatenate([d0s[p] for p in keep]).astype(np.int32),
                        conf=np.zeros((sum(d0s[p].shape[0] for p in keep), BL), np.uint8))
    monkeypatch.setattr(RI, "load_trained", lambda trained, device: (TD.ScriptedLM(drafter_fn), {"kind": "fake"}))
    out_rel = os.path.join("logs", "_pytest_redraft", uuid.uuid4().hex[:8])
    out = os.path.join(P.ROOT, out_rel)
    try:
        argv = ["--index", str(root / "index.jsonl"), "--pids", str(tmp_path / "pids.json"), "--drafts", str(dd),
                "--trained", "fake", "--device", "cpu", "--mask-id", str(TD.MASK), "--out", out_rel]
        assert RI.main(argv) == 0
        rep = json.load(open(os.path.join(out, "redrafts.shard0of1.json")))
        assert rep["items"] == 3 and rep["fallback_items"] == 1 and rep["repeat"] == 0 and rep["holes"] > 0
        got = RI.load_redrafts(out)
        assert sorted(got) == ["c00001", "c00002", "c00004"]
        for pid, (pp, rd, rc) in got.items():
            nb = d0s[pid].shape[0]
            assert pp.shape == (nb,) and rd.shape == (nb, BL) and rc.shape == (nb, BL)
        with pytest.raises(SystemExit):                                  # --drafts is required
            RI.build_parser().parse_args(["--index", "x", "--pids", "y", "--trained", "z", "--out", "o"])
    finally:
        shutil.rmtree(os.path.join(P.ROOT, "logs", "_pytest_redraft"), ignore_errors=True)


def test_block_masks_fold_accepted_like_the_sampler():
    """At a pass p > 0 an `accepted` slot is committed (shows T, never a hole), as StateSampler and K3Sampler count it;
    at p = 0 nothing is committed; without `accepted` the sets are unchanged."""
    valid = np.ones(BL, bool); valid[30:] = False
    hist = np.zeros((3, BL), bool); hist[0, :4] = True; hist[1, 4:8] = True
    T = np.arange(BL); D0 = T.copy(); D0[8:12] += 1; D0[12] = -1
    acc = np.zeros(BL, bool); acc[8] = True; acc[31] = True            # 31 is invalid: never committed
    m0 = RI.block_masks(valid, hist, T, D0, 2)
    m1 = RI.block_masks(valid, hist, T, D0, 2, acc)
    assert m0["committed"].sum() == 8 and m0["holes"][8] and m0["holes"].sum() == 4
    assert m1["committed"].sum() == 9 and m1["committed"][8] and not m1["holes"][8] and m1["holes"].sum() == 3
    assert not m1["committed"][31] and not m1["hidden"][31]
    assert np.array_equal(m1["tentative"], m0["tentative"]) and np.array_equal(m1["undrafted"], m0["undrafted"])
    assert RI.block_masks(valid, hist, T, D0, 0, acc)["committed"].sum() == 0
