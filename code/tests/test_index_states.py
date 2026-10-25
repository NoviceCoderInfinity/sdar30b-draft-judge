"""CPU tests for the judge round's states index (code/judge_round/index_states.py, piece A0).  No model, no GPU.

The records are REAL recorder output: fake trainset items (test_drafter_lora.fake_item) decoded by
corpus1b/sglang_plugin/judge_core.run_blocks with record_states=True and teacher forcing (closing_ids = the item's
tokens), a deterministic fake judge and a fake drafter that obeys the ban; the trace rows go through
judge_sglang.assemble_blocks / check_record_blocks / pack_states exactly as judge_sglang.py --mode record writes them.
The helpers here (write_trainset, record_items, write_records, the fakes) are shared with test_sampler_rec.py.

Pins: the index holds every chunk's arrays unchanged, item by item; duplicate rids keep the first copy; .tmp chunks
are never read; every layout and recorder invariant check fires on a corrupted chunk; mixed run.json / cfg and
records without the ban are refused; rids without an index row are dropped (or refused with strict); the trainset
checks catch a wrong teacher; the summary counts domains, fresh vs repeated, post-EOS passes and ban-unreachable
holes; the CLI writes through safe_out and refuses to overwrite.

    python -m pytest -q code/tests/test_index_states.py
"""
from __future__ import annotations

import hashlib
import importlib.util
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

import test_drafter_lora as TDL                                           # noqa: E402  (fake 1B-corpus items)
from judge_round import index_states as IS                                # noqa: E402
from judge_round import judge_sglang as JS                                # noqa: E402

BL = 32
MASK = TDL.MASK                                                           # 39
NTOK = MASK - 2                                                           # fake tokens are < 37
V = TDL.VOCAB                                                             # 40


def _load_judge_core():
    if "judge_core" in sys.modules:
        return sys.modules["judge_core"]
    spec = importlib.util.spec_from_file_location("judge_core", os.path.join(JS.PLUGIN, "judge_core.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses resolve the module through sys.modules
    spec.loader.exec_module(mod)
    return mod


JC = _load_judge_core()


def quiet(*_a, **_k):
    pass


# ---------------------------------------------------------------------------
# shared helpers: trainset, fakes, the recorder driver, the chunk writer
# ---------------------------------------------------------------------------
# (seed, prompt length, blocks, eos slot in the last block, domain, repeated)
SPECS = [(0, 45, 3, None, "maths", False),
         (1, 37, 2, 10, "code", False),          # post-EOS slots in its last block
         (2, 64, 2, None, "chat", False),        # prompt on the 32-grid: block 0 has no prompt tail
         (3, 33, 4, 20, "science", True),        # repeated (seen) science, with post-EOS slots
         (4, 50, 2, None, "if", False),
         (5, 70, 3, 5, "maths", False)]


def write_trainset(root, specs=SPECS):
    """Fake items (to_trainset format) + index.jsonl; returns (index path, items with "pid")."""
    os.makedirs(os.path.join(root, "items"), exist_ok=True)
    items = []
    with open(os.path.join(root, "index.jsonl"), "w") as fh:
        for n, (seed, pl, nb, eos, dom, rep) in enumerate(specs):
            it = TDL.fake_item(seed=seed, pl=pl, nb=nb, eos_slot=eos, domain=dom, pid=f"{dom}-t-{seed:04d}")
            pid = it.pop("pid")
            rel = os.path.join("items", f"{pid}.npz")
            np.savez_compressed(os.path.join(root, rel), **it)
            row = {"file": rel, "pid": pid, "idx": n, "domain": dom, "split": f"c1b_{dom}", "correct": 1,
                   "usable": [True] * nb, "has_draft": [True] * nb, "block_start": it["block_start"].tolist(),
                   "n_blocks": nb, "seq_len": int(it["block_start"][-1]) + BL,
                   "corpus_tokens": int(it["prompt_length"]) + BL * nb}
            if rep:
                row.update({"repeated": True, "wave": "A"})
            fh.write(json.dumps(row) + "\n")
            items.append(dict(it, pid=pid))
    return os.path.join(root, "index.jsonl"), items


def judge_cfgs(redraft_every=1, ban=True, tau=0.95):
    """(judge_core.JudgeConfig for the recorder, the yaml cfg judge_sglang stores in run.json and every chunk)."""
    jc = JC.JudgeConfig(mask_id=MASK, tau_commit=tau, max_passes=32, redraft_every=redraft_every, redraft_rate=4,
                        redraft_ban=ban, loop_guard=0, threshold=0.95, draft_steps=8, record_states=True)
    yc = {"block_size": BL, "tau_commit": float(tau), "tau_override": None, "max_passes": 32,
          "redraft_every": int(redraft_every), "redraft_rate": 4, "redraft_ban": bool(ban), "loop_guard": 0,
          "threshold": 0.95, "draft_steps": 8}
    return jc, yc


def _h(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest(),
                          "little") % (2 ** 63 - 1)


class FakeJudge:
    """Deterministic logits from (seed, block start, the block's tokens, its flag): each slot prefers the teacher
    token (p 0.7) or a random one by a random margin, so passes mix commits, tentatives and remasks."""

    def __init__(self, seed=1):
        self.seed = seed

    def __call__(self, ctx, ids, flag, kind):
        out = []
        for i in range(ids.shape[0]):
            g = torch.Generator().manual_seed(_h(self.seed, ctx["lo"], kind, ids[i].tolist(), flag[i].int().tolist()))
            lg = torch.randn(BL, V, generator=g)
            pref = torch.where(torch.rand(BL, generator=g) < 0.7, ctx["teacher"],
                               torch.randint(0, NTOK, (BL,), generator=g))
            lg[torch.arange(BL), pref] += 1.0 + 9.0 * torch.rand(BL, generator=g)
            out.append(lg)
        return torch.stack(out)


class FakeDrafter:
    """Fills the requested slots with the teacher token (p 0.6) or a random one; never re-proposes the banned token,
    or (obey_ban=False) always does."""

    def __init__(self, seed=2, obey_ban=True):
        self.seed, self.obey_ban = seed, obey_ban

    def __call__(self, ctx, reqs):
        out = []
        for q in reqs:
            g = torch.Generator().manual_seed(_h(self.seed, ctx["lo"], q.kind, q.block_ids.tolist(),
                                                 q.fill.int().tolist()))
            cand = torch.where(torch.rand(BL, generator=g) < 0.6, ctx["teacher"],
                               torch.randint(0, NTOK, (BL,), generator=g))
            if q.ban is not None and self.obey_ban:
                cand = torch.where((q.ban >= 0) & (cand == q.ban), (cand + 1) % NTOK, cand)
            elif q.ban is not None:                              # a leaky drafter: re-proposes every banned token
                cand = torch.where(q.ban >= 0, q.ban, cand)
            tok = torch.where(q.fill, cand, q.block_ids)
            pc = torch.where(q.fill, 0.05 + 0.9 * torch.rand(BL, generator=g), torch.zeros(BL))
            out.append(JC.DraftRes(tokens=tok, p_commit=pc, n_fwd=int(q.steps or min(8, int(q.fill.sum())))))
        return out


def trace_row(run, b, lo, start, rid):
    """A judge trace row as the SGLang plugin writes it (lists; NaN where not read)."""
    def plain(d):
        return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in d.items()}
    states = []
    for st in run.states:
        d = plain({k: v for k, v in st.items() if k != "redraft"})
        d["redraft"] = None if st.get("redraft") is None else plain(st["redraft"])
        states.append(d)
    return {"rid": rid, "block": b, "lo": lo, "start": start, "summary": JC.block_summary(run), "states": states}


def record_items(items, jcfg, judge, drafter):
    """judge_core.run_blocks over every block of every item, teacher-forced (closing_ids = the item's tokens), the
    prefix = the prompt + the item's earlier blocks; returns [(rid, blocks)] checked by judge_sglang."""
    out = []
    for it in items:
        req = JS.record_request(it)
        L = int(it["prompt_length"])
        lo0 = (L // BL) * BL
        tail = L - lo0
        prompt = np.asarray(it["prompt_ids"], np.int64)
        tokens = np.asarray(it["tokens"], np.int64)
        clean = np.concatenate([prompt[:lo0], tokens.reshape(-1)])
        rows = {}
        for b in range(tokens.shape[0]):
            lo = lo0 + BL * b
            cur = torch.full((BL,), MASK, dtype=torch.long)
            valid = torch.ones(BL, dtype=torch.bool)
            if b == 0 and tail:
                cur[:tail] = torch.from_numpy(prompt[lo0:L])
                valid[:tail] = False
            run = JC.BlockRun(cur=cur, valid=valid, lo=lo)
            ctx = {"lo": lo, "prefix": torch.from_numpy(clean[:lo].copy()), "teacher": torch.from_numpy(tokens[b].copy())}
            JC.run_blocks([run], jcfg, lambda ids, flag, kind, c=ctx: judge(c, ids, flag, kind),
                          lambda reqs, c=ctx: drafter(c, reqs), closing_ids=[ctx["teacher"]])
            rows[lo] = trace_row(run, b, lo, tail if b == 0 else 0, it["pid"])
        blocks = JS.assemble_blocks(rows, L)
        JS.check_record_blocks(blocks, req)
        out.append((it["pid"], blocks))
    return out


def write_records(root, gpu, groups, ycfg, adapter="adapter-A", drafter="drafter-D"):
    """<root>/gpu<gpu>/run.json + one states.NNNNN.npz per group + done.txt, as judge_sglang.py --mode record."""
    gd = os.path.join(root, f"gpu{gpu}")
    os.makedirs(gd, exist_ok=True)
    rj = os.path.join(gd, "run.json")
    if not os.path.exists(rj):
        with open(rj, "w") as fh:
            json.dump({"mode": "record", "cfg": ycfg, "judge_adapter": adapter, "drafter_dir": drafter,
                       "max_new_tokens": None}, fh, indent=1)
    for entries in groups:
        ci = JS.next_states_index(gd)
        np.savez_compressed(os.path.join(gd, f"states.{ci:05d}.npz"), **JS.pack_states(entries, ycfg))
        with open(os.path.join(gd, "done.txt"), "a") as fh:
            fh.write("".join(f"{r}\n" for r, _ in entries))
    return gd


def eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def item_rows(ix, rid):
    i = ix.pos[rid]
    b0, b1 = int(ix.ib[i]), int(ix.ib[i + 1])
    return slice(b0, b1), slice(int(ix.bp[b0]), int(ix.bp[b1]))


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("a0"))
    index, items = write_trainset(os.path.join(root, "trainset"))
    jcfg, ycfg = judge_cfgs()
    recs = record_items(items, jcfg, FakeJudge(1), FakeDrafter(2))
    rec_root = os.path.join(root, "record")
    write_records(rec_root, 0, [recs[0:2], recs[2:3]], ycfg)               # gpu0: two chunks
    write_records(rec_root, 1, [recs[3:6]], ycfg)                          # gpu1: one chunk
    out = os.path.join(root, "a0")
    summ = IS.build_index([rec_root], out, index_paths=[index], mask_id=MASK, log=quiet)
    return SimpleNamespace(root=root, index=index, items=items, recs=recs, jcfg=jcfg, ycfg=ycfg, rec_root=rec_root,
                           out=out, summ=summ, ix=IS.StatesIndex(out))


def test_fixture_covers_every_case(world):
    a = world.ix.a
    npb = np.diff(world.ix.bp)
    assert world.ix.n_items == len(SPECS) and int(npb.min()) >= 1 and int(npb.max()) >= 3
    assert np.asarray(a["forced"]).any() and np.asarray(a["rd_on"]).any()
    assert np.asarray(a["tentative"]).any() and np.asarray(a["remask"]).any() and np.asarray(a["commit"]).any()
    assert (~np.asarray(a["rd_on"])).any()                                   # passes without a re-draft (the last)
    assert world.summ["post_eos_passes"] > 0 and world.summ["redraft_holes"] > 0
    assert sorted(os.listdir(os.path.join(world.rec_root, "gpu0"))) == ["done.txt", "run.json", "states.00000.npz",
                                                                         "states.00001.npz"]


# ---------------------------------------------------------------------------
# the index is the chunks, item by item; the reader
# ---------------------------------------------------------------------------
def test_index_holds_every_chunk_array_unchanged(world):
    ix = world.ix
    assert ix.rids == [r for r, _ in world.recs]
    for rid, blocks in world.recs:
        exp = JS.pack_states([(rid, blocks)], world.ycfg)
        brows, prows = item_rows(ix, rid)
        for k in IS.BLK_KEYS:
            assert eq(ix.a[k][brows], exp[k]), k
            assert ix.a[k].dtype == exp[k].dtype, k
        for k in IS.PASS_KEYS:
            assert eq(ix.a[k][prows], exp[k]), k
            assert ix.a[k].dtype == exp[k].dtype, k
        assert np.array_equal(np.diff(ix.bp[brows.start:brows.stop + 1]), np.diff(exp["blk_pass"]))
    assert ix.cfg == world.ycfg and ix.mask_id == MASK and ix.counters == list(JS.COUNTERS)


def test_reader_returns_each_trainset_block(world):
    ix = world.ix
    for it in world.items:
        nb = it["tokens"].shape[0]
        assert ix.n_blocks(it["pid"]) == nb and ix.block(it["pid"], nb) is None and ix.block_row(it["pid"], -1) is None
        for b in range(nb):
            rb = ix.block(it["pid"], b)
            assert rb.lo == int(it["block_start"][b]) and np.array_equal(rb.teacher, it["tokens"][b])
            assert rb.start == (int(it["prompt_length"]) - int(it["block_start"][0]) if b == 0 else 0)
            assert rb.n == ix.n_passes(rb.row) == int(ix.a["blk_counters"][rb.row, list(JS.COUNTERS).index("n_judge")])
            assert np.array_equal(rb.passes["pass_no"], np.arange(rb.n))
            assert eq(rb.passes["input"], ix.a["input"][rb.pass0:rb.pass0 + rb.n])
    assert ix.block("no-such-pid", 0) is None and ix.n_blocks("no-such-pid") == 0


def test_summary_counts(world):
    s = world.summ
    npb = np.diff(world.ix.bp)
    assert s["items"] == len(SPECS) and s["blocks"] == len(npb) and s["passes"] == int(npb.sum())
    assert s["passes_per_block_hist"] == np.bincount(npb).tolist()
    assert s["items_by_domain"] == {"chat": 1, "code": 1, "if": 1, "maths": 2, "science": 1}
    assert s["items_fresh_vs_repeated"] == {"fresh": 5, "repeated": 1}
    assert s["duplicates"] == 0 and s["records_without_index_row"] == 0 and s["ban_violations"] == 0
    assert 0 <= s["ban_unreachable_holes"] <= s["redraft_holes"]
    # post-EOS: passes of the last blocks of items 1, 3, 5 with an open slot after the EOS
    want = 0
    for it in world.items:
        post = ~np.asarray(it["valid"][-1], bool)
        if it["block_start"].shape[0] == 1:
            continue
        rb = world.ix.block(it["pid"], it["tokens"].shape[0] - 1)
        want += int((np.asarray(rb.passes["open"], bool) & post).any(1).sum())
    assert s["post_eos_passes"] == want > 0
    assert json.load(open(os.path.join(world.out, "summary.json")))["passes"] == s["passes"]
    meta = json.load(open(os.path.join(world.out, "meta.json")))
    assert meta["version"] == IS.VERSION and meta["run"]["judge_adapter"] == "adapter-A"
    assert [len(src["chunks"]) for src in meta["sources"]] == [3]


# ---------------------------------------------------------------------------
# duplicates, crash leftovers, rows without an index row
# ---------------------------------------------------------------------------
def copy_root(world, dst):
    shutil.copytree(world.rec_root, dst)
    return dst


def test_a_rid_recorded_twice_keeps_the_first_copy(world, tmp_path):
    root = copy_root(world, str(tmp_path / "rec"))
    again = record_items([world.items[0]], world.jcfg, FakeJudge(7), FakeDrafter(8))       # a different decode
    write_records(root, 1, [again], world.ycfg)                                            # gpu1/states.00001.npz
    s = IS.build_index([root], str(tmp_path / "out"), index_paths=[world.index], mask_id=MASK, log=quiet)
    assert s["duplicates"] == 1 and s["items"] == len(SPECS)
    d = s["duplicate_examples"][0]
    assert d["rid"] == world.items[0]["pid"] and d["kept"].endswith("gpu0/states.00000.npz")
    assert d["dropped"].endswith("gpu1/states.00001.npz")
    ix = IS.StatesIndex(str(tmp_path / "out"))
    _, prows = item_rows(ix, world.items[0]["pid"])
    assert eq(ix.a["input"][prows], world.ix.a["input"][item_rows(world.ix, world.items[0]["pid"])[1]])


def test_a_crash_tmp_chunk_is_never_read(world, tmp_path):
    root = copy_root(world, str(tmp_path / "rec"))
    with open(os.path.join(root, "gpu0", "states.00002.tmp.npz"), "wb") as fh:
        fh.write(b"not an npz")
    s = IS.build_index([root], str(tmp_path / "out"), index_paths=[world.index], mask_id=MASK, log=quiet)
    assert s["items"] == len(SPECS) and s["chunks_read"] == 3


def test_rids_without_an_index_row_are_dropped_or_refused(world, tmp_path):
    rows = [json.loads(l) for l in open(world.index)]
    short = str(tmp_path / "trainset" / "index.jsonl")
    os.makedirs(os.path.dirname(short))
    with open(short, "w") as fh:
        for r in rows[1:]:
            fh.write(json.dumps(dict(r, file=os.path.join(os.path.dirname(world.index), r["file"]))) + "\n")
    s = IS.build_index([world.rec_root], str(tmp_path / "out"), index_paths=[short], mask_id=MASK, log=quiet)
    assert s["items"] == len(SPECS) - 1 and s["records_without_index_row"] == 1
    assert s["without_row_examples"] == [rows[0]["pid"]]
    with pytest.raises(IS.StatesError, match="no --index row"):
        IS.build_index([world.rec_root], str(tmp_path / "out2"), index_paths=[short], mask_id=MASK, strict=True,
                       log=quiet)


# ---------------------------------------------------------------------------
# refusals: configs, the ban, the trainset checks, an existing output
# ---------------------------------------------------------------------------
def test_mixed_configs_are_refused(world, tmp_path):
    jc2, yc2 = judge_cfgs(redraft_every=2)
    rd2 = str(tmp_path / "rd2")
    write_records(rd2, 0, [record_items(world.items[:1], jc2, FakeJudge(1), FakeDrafter(2))], yc2)
    with pytest.raises(IS.StatesError, match="run.json differs"):
        IS.build_index([world.rec_root, rd2], str(tmp_path / "out"), mask_id=MASK, log=quiet)
    other = copy_root(world, str(tmp_path / "other_adapter"))
    for g in ("gpu0", "gpu1"):
        p = os.path.join(other, g, "run.json")
        rj = json.load(open(p))
        json.dump(dict(rj, judge_adapter="adapter-B"), open(p, "w"))
    with pytest.raises(IS.StatesError, match="run.json differs"):
        IS.build_index([world.rec_root, other], str(tmp_path / "out2"), mask_id=MASK, log=quiet)


def test_records_without_the_ban_need_allow_no_ban(world, tmp_path):
    jc, yc = judge_cfgs(ban=False)
    root = str(tmp_path / "noban")
    write_records(root, 0, [record_items(world.items[:2], jc, FakeJudge(1), FakeDrafter(2))], yc)
    with pytest.raises(IS.StatesError, match="WITHOUT --redraft-ban"):
        IS.build_index([root], str(tmp_path / "out"), mask_id=MASK, log=quiet)
    s = IS.build_index([root], str(tmp_path / "out"), mask_id=MASK, allow_no_ban=True, log=quiet)
    assert s["items"] == 2


def test_ban_violations_are_counted_and_strict_refuses(world, tmp_path):
    root = str(tmp_path / "leaky")
    write_records(root, 0, [record_items(world.items, world.jcfg, FakeJudge(1), FakeDrafter(2, obey_ban=False))],
                  world.ycfg)
    s = IS.build_index([root], str(tmp_path / "out"), mask_id=MASK, log=quiet)
    assert s["ban_violations"] > 0
    with pytest.raises(IS.StatesError, match="banned"):
        IS.build_index([root], str(tmp_path / "out2"), mask_id=MASK, strict=True, log=quiet)


def test_trainset_checks_catch_a_wrong_teacher(world, tmp_path):
    ts = str(tmp_path / "trainset")
    shutil.copytree(os.path.dirname(world.index), ts)
    pid = world.items[1]["pid"]
    p = os.path.join(ts, "items", f"{pid}.npz")
    with np.load(p) as z:
        it = {k: z[k] for k in z.files}
    it["tokens"] = it["tokens"].copy()
    it["tokens"][1, 3] = (it["tokens"][1, 3] + 1) % NTOK
    np.savez_compressed(p, **it)
    with pytest.raises(IS.StatesError, match="teacher"):
        IS.build_index([world.rec_root], str(tmp_path / "out"), index_paths=[os.path.join(ts, "index.jsonl")],
                       mask_id=MASK, log=quiet)
    s = IS.build_index([world.rec_root], str(tmp_path / "out2"), index_paths=[os.path.join(ts, "index.jsonl")],
                       mask_id=MASK, item_checks=False, log=quiet)          # --no-item-checks skips them
    assert s["items"] == len(SPECS) and "post_eos_passes" not in s


def test_an_existing_output_needs_force(world, tmp_path):
    out = str(tmp_path / "out")
    IS.build_index([world.rec_root], out, mask_id=MASK, log=quiet)
    with pytest.raises(IS.StatesError, match="exists"):
        IS.build_index([world.rec_root], out, mask_id=MASK, log=quiet)
    IS.build_index([world.rec_root], out, mask_id=MASK, force=True, log=quiet)
    assert not os.path.exists(out + ".tmp") and IS.StatesIndex(out).n_items == len(SPECS)


def test_nothing_to_index_is_refused(tmp_path):
    with pytest.raises(IS.StatesError):
        IS.build_index([str(tmp_path)], str(tmp_path / "out"), mask_id=MASK, log=quiet)


# ---------------------------------------------------------------------------
# every chunk check fires on a corrupted copy
# ---------------------------------------------------------------------------
def _pass_with(a, cond):
    """The first pass row j (and slot s) where cond(a, j) holds per slot."""
    for j in range(a["pass_no"].shape[0]):
        s = np.flatnonzero(cond(a, j))
        if len(s):
            return j, int(s[0])
    raise AssertionError("the fixture chunk lacks the case")


def _next_in_block(a):
    bp = a["blk_pass"]
    for blk in range(len(bp) - 1):
        if bp[blk + 1] - bp[blk] >= 2:
            return int(bp[blk])
    raise AssertionError("no block with two passes")


def c_flag(a):
    j, s = _pass_with(a, lambda a, j: a["flag"][j])
    a["flag"][j, s] = False


def c_partition(a):
    j, s = _pass_with(a, lambda a, j: a["commit"][j])
    a["tentative"][j, s] = True


def c_argmax(a):
    j, s = _pass_with(a, lambda a, j: ~a["open"][j])
    a["argmax"][j, s] = 3


def c_p(a):
    j, s = _pass_with(a, lambda a, j: a["open"][j])
    a["p"][j, s] = np.nan


def c_n_judge(a):
    a["blk_counters"][0, list(JS.COUNTERS).index("n_judge")] += 1


def c_pass_no(a):
    j = _next_in_block(a)
    a["pass_no"][j], a["pass_no"][j + 1] = a["pass_no"][j + 1], a["pass_no"][j]


def c_blk_start(a):
    a["blk_start"][int(np.flatnonzero(a["blk_index"] == 1)[0])] = 3


def c_blk_lo(a):
    a["blk_lo"][0] += 1


def c_transition(a):
    j = _next_in_block(a)
    s = int(np.flatnonzero(~a["open"][j + 1] & (np.arange(BL) >= a["blk_start"][0]))[0])
    a["input"][j + 1, s] = (a["input"][j + 1, s] + 1) % NTOK


def c_rd_holes(a):
    j, s = _pass_with(a, lambda a, j: a["rd_holes"][j] & a["rd_on"][j])
    a["rd_holes"][j, s] = False


def c_rd_off(a):
    j, _s = _pass_with(a, lambda a, j: np.full(BL, not a["rd_on"][j]))
    a["rd_n_fwd"][j] = 1


def c_first_draft(a):
    s = int(np.flatnonzero(np.arange(BL) >= a["blk_start"][0])[0])
    a["blk_draft"][0, s] = (a["blk_draft"][0, s] + 1) % NTOK


def c_forced(a):
    j, _s = _pass_with(a, lambda a, j: np.full(BL, a["commit"][j].sum() >= 2))
    a["forced"][j] = True


def c_rid(a):
    a["rid"] = np.array(["zz"] + a["rid"].tolist()[1:])


@pytest.mark.parametrize("corrupt,match", [
    (c_flag, "flag"), (c_partition, "partition"), (c_argmax, "argmax"), (c_p, "p / q"), (c_n_judge, "n_judge"),
    (c_pass_no, "pass_no"), (c_blk_start, "blk_start"), (c_blk_lo, "32-grid"), (c_transition, "input of pass"),
    (c_rd_holes, "holes"), (c_rd_off, "without a re-draft"), (c_first_draft, "first draft"), (c_forced, "forced"),
    (c_rid, "rid array")])
def test_every_chunk_check_fires(world, tmp_path, corrupt, match):
    src = os.path.join(world.rec_root, "gpu0", "states.00000.npz")
    with np.load(src) as z:
        a = {k: z[k].copy() for k in z.files}
    corrupt(a)
    root = str(tmp_path / "bad")
    gd = os.path.join(root, "gpu0")
    os.makedirs(gd)
    shutil.copy(os.path.join(world.rec_root, "gpu0", "run.json"), gd)
    np.savez_compressed(os.path.join(gd, "states.00000.npz"), **a)
    with pytest.raises(IS.StatesError, match=match):
        IS.build_index([root], str(tmp_path / "out"), mask_id=MASK, log=quiet)
    assert not os.path.exists(str(tmp_path / "out")) and not os.path.exists(str(tmp_path / "out.tmp"))


def test_the_unchanged_chunk_passes(world, tmp_path):
    src = os.path.join(world.rec_root, "gpu0", "states.00000.npz")
    with np.load(src) as z:
        a = {k: z[k] for k in z.files}
    counts = IS.check_chunk(a, json.loads(str(a["index"])), MASK, src)
    assert counts["items"] == 2 and counts["redrafts"] > 0 and counts["ban_violations"] == 0


# ---------------------------------------------------------------------------
# the cli
# ---------------------------------------------------------------------------
def test_cli_writes_through_safe_out_and_refuses_to_overwrite(world, tmp_path, monkeypatch):
    import research_protocol as P
    monkeypatch.setattr(P, "safe_out", lambda p: os.path.abspath(p))
    out = str(tmp_path / "cli")
    argv = [world.rec_root, "--index", world.index, "--out", out, "--mask-id", str(MASK)]
    assert IS.main(argv) == 0 and IS.StatesIndex(out).n_items == len(SPECS)
    assert IS.main(argv) == 2
    assert IS.main(argv + ["--force"]) == 0
    with pytest.raises(AssertionError, match="refusing to write outside"):
        monkeypatch.undo()
        IS.main([world.rec_root, "--out", str(tmp_path / "outside"), "--mask-id", str(MASK)])
