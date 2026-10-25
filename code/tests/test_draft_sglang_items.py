"""CPU tests for judge_round/draft_sglang_items.py (no SGLang, no model).

  * item_requests: block 0 = the prompt (max_new = 32 - prompt tail), block b >= 1 = prompt + the 30B's tokens before
    block b (block 0's prompt tail skipped), max_new = 32; a prompt that ends on a block boundary has no tail;
  * pack_results + write_chunk + merge_shard: the plugin's trace -> draft_sglang's chunk format (-1 / conf 0 on the
    prompt tail, uint8 conf = round(255 p)), done.txt, and the merged shard file read by judge_probe.load_drafts;
  * main on a finished shard: nothing to draft, no engine (sglang is never imported), the shard file is (re)merged.

Run from the repo root:  python3 -m pytest -q code/tests/test_draft_sglang_items.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from judge_round import draft_sglang_items as DS                         # noqa: E402
from drafter_lora.judge_probe import load_drafts                         # noqa: E402

BL = 32


def make_item(pid: str = "maths-t-0001", L: int = 37, nb: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prompt = rng.integers(1000, 2000, size=L).astype(np.int32)
    lo0 = (L // BL) * BL
    tokens = rng.integers(3000, 4000, size=(nb, BL)).astype(np.int32)
    tokens[0, :L - lo0] = prompt[lo0:L]                          # the prompt tail duplicated in block 0
    return {"pid": pid, "prompt_ids": prompt, "tokens": tokens,
            "block_start": (lo0 + BL * np.arange(nb)).astype(np.int32), "prompt_length": np.int32(L)}


class FakeReader:
    """gen_sglang.TraceReader's take(): rid -> {lo: {"start", "ids", "conf"}}, popped once."""

    def __init__(self, rows):
        self.rows = rows

    def take(self, rid):
        return self.rows.pop(rid, {})


def trace_rows(item: dict, conf_row, drop=()):
    """What the plugin would record for every block of `item` (blocks in `drop` are missing)."""
    L = int(item["prompt_length"])
    rows = {}
    for q in DS.item_requests(item):
        if q["b"] in drop:
            continue
        start = BL - q["max_new"]
        ids = [int(t) + 7 for t in item["tokens"][q["b"]]]                # the draft: anything, here T + 7
        ids[:start] = [int(t) for t in item["prompt_ids"][q["lo"]:L]]     # fixed slots carry the prompt tail
        conf = [-1.0] * start + list(conf_row[start:])                    # -1.0 on never-masked slots
        rows[q["rid"]] = {q["lo"]: {"rid": q["rid"], "lo": q["lo"], "start": start, "ids": ids, "conf": conf}}
    return rows


def test_item_requests_prompt_tail():
    it = make_item(L=37, nb=3)
    assert int(it["block_start"][0]) == 32
    reqs = DS.item_requests(it)
    assert [q["rid"] for q in reqs] == ["maths-t-0001#0", "maths-t-0001#1", "maths-t-0001#2"]
    assert [len(q["input_ids"]) for q in reqs] == [37, 64, 96]
    assert [q["max_new"] for q in reqs] == [27, 32, 32]
    assert [q["lo"] for q in reqs] == [32, 64, 96]
    prompt = it["prompt_ids"].tolist()
    t = it["tokens"]
    assert reqs[0]["input_ids"] == prompt
    assert reqs[1]["input_ids"] == prompt + t[0, 5:].tolist()             # tail skipped, not duplicated
    assert reqs[2]["input_ids"] == prompt + t[0, 5:].tolist() + t[1].tolist()
    assert reqs[2]["input_ids"][-BL:] == t[1].tolist()
    assert all(type(x) is int for q in reqs for x in q["input_ids"])


def test_item_requests_no_tail():
    it = make_item(L=64, nb=2)
    reqs = DS.item_requests(it)
    assert [len(q["input_ids"]) for q in reqs] == [64, 96]
    assert [q["max_new"] for q in reqs] == [32, 32]
    assert reqs[1]["input_ids"] == it["prompt_ids"].tolist() + it["tokens"][0].tolist()


def test_shard_and_pending():
    pids = ["c", "a", "e", "b", "d"]
    assert DS.shard_pids(pids, 0, 2) == ["a", "c", "e"]
    assert DS.shard_pids(pids, 1, 2) == ["b", "d"]
    assert DS.pending(["a", "c", "e"], {"c"}) == ["a", "e"]
    assert DS.pending(["a", "c", "e"], set(), 2) == ["a", "c"]


def test_pack_write_merge(tmp_path):
    out_root = str(tmp_path / "drafts")
    gdir = os.path.join(out_root, "gpu0")
    os.makedirs(gdir)
    conf_row = [0.2] * BL
    conf_row[6], conf_row[31], conf_row[10] = 0.5, 1.0, 0.0
    a, b, c = make_item("a-1", 37, 3, 1), make_item("b-2", 70, 2, 2), make_item("c-3", 40, 2, 3)
    rows = {**trace_rows(a, conf_row), **trace_rows(b, conf_row), **trace_rows(c, conf_row, drop=(1,))}
    per_item, good, n_blocks, n_bad = DS.pack_results([a, b, c], FakeReader(rows))
    assert good == ["a-1", "b-2"] and n_blocks == 5 and n_bad == 1         # c-3 lacks block 1: incomplete

    da, ca = per_item["a-1"]["draft"], per_item["a-1"]["conf"]
    assert da.dtype == np.int32 and ca.dtype == np.uint8 and da.shape == (3, BL)
    assert (da[0, :5] == -1).all() and (ca[0, :5] == 0).all()             # prompt tail: not drafted
    assert (da[0, 5:] == a["tokens"][0, 5:] + 7).all() and (da[1:] == a["tokens"][1:] + 7).all()
    assert ca[0, 6] == 128 and ca[0, 31] == 255 and ca[0, 10] == 0 and ca[0, 5] == 51
    assert (ca[1] == np.array([51] * 6 + [128] + [51] * 3 + [0] + [51] * 20 + [255], np.uint8)).all()
    db = per_item["b-2"]["draft"]
    assert (db[0, :70 - 64] == -1).all() and (db[0, 6:] >= 0).all() and (db[1] >= 0).all()

    DS.write_chunk(gdir, DS.next_chunk_index(gdir), ["a-1"], per_item)
    DS.write_chunk(gdir, DS.next_chunk_index(gdir), ["b-2"], per_item)
    assert sorted(os.listdir(gdir)) == ["done.txt", "drafts.00000.npz", "drafts.00001.npz"]
    assert DS.read_done(gdir) == {"a-1", "b-2"}
    z = np.load(os.path.join(gdir, "drafts.00000.npz"), allow_pickle=True)   # draft_sglang's chunk format
    assert z["pids"].tolist() == ["a-1"] and z["nb"].tolist() == [3] and z["draft"].dtype == np.int32

    open(os.path.join(gdir, "drafts.00002.npz.tmp.npz"), "w").close()       # a crash's leftover is not a chunk
    assert DS.next_chunk_index(gdir) == 2
    info = DS.merge_shard(out_root, 0, 8)
    assert info["items"] == 2 and info["blocks"] == 5
    assert os.path.basename(info["path"]) == "drafts.shard0of8.npz"
    got = load_drafts(out_root)                                             # no allow_pickle there
    assert sorted(got) == ["a-1", "b-2"]
    for pid in got:
        dr, cf = got[pid]
        assert dr.dtype == np.int32 and cf.dtype == np.uint8
        assert np.array_equal(dr, per_item[pid]["draft"]) and np.array_equal(cf, per_item[pid]["conf"])
    kept = DS.merge_shard(out_root, 0, 8, keep={"b-2"})                     # only this shard's pids
    assert (kept["items"], kept["blocks"]) == (1, 2)


def test_main_nothing_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(DS.P, "safe_out", lambda p: p)
    tmp = str(tmp_path)
    items = [make_item("a-1", 37, 3, 1), make_item("b-2", 70, 2, 2)]
    with open(os.path.join(tmp, "index.jsonl"), "w") as fh:
        for it in items:
            fh.write(json.dumps({"file": f"items/{it['pid']}.npz", "pid": it["pid"], "domain": "maths"}) + "\n")
    with open(os.path.join(tmp, "pids.json"), "w") as fh:
        json.dump({"pids": ["b-2", "a-1"]}, fh)
    out_root = os.path.join(tmp, "drafts")
    gdir = os.path.join(out_root, "gpu0")
    os.makedirs(gdir)
    per_item = {it["pid"]: {"draft": np.zeros((it["tokens"].shape[0], BL), np.int32),
                            "conf": np.ones((it["tokens"].shape[0], BL), np.uint8)} for it in items}
    DS.write_chunk(gdir, 0, ["a-1", "b-2"], per_item)
    argv = ["--index", os.path.join(tmp, "index.jsonl"), "--pids", os.path.join(tmp, "pids.json"),
            "--drafter-dir", os.path.join(tmp, "no_such_dir"), "--out", out_root, "--gpu", "0", "--num-gpus", "1"]
    info = DS.main(argv)                                 # would fail importing sglang / on --drafter-dir if it ran
    assert "sglang" not in sys.modules
    assert info["items"] == 2 and os.path.exists(os.path.join(out_root, "drafts.shard0of1.npz"))
    assert sorted(load_drafts(out_root)) == ["a-1", "b-2"]
