"""The SGLang drafter algorithm DraftTracedV2 (code/corpus1b/sglang_plugin/c1b_judge_draft.py) on CPU, with fake
sglang modules and a fake model runner whose full_logits come from the ScriptedLM fakes of code/tests/test_da_decode.py
(one drafter cache per block of the batch).  Its trace rows equal the harness's draft_block on every block:

  * batch 1 (int64 ids): a re-draft (fill = holes, tentative slots hidden, ban, 3 steps) next to a loop-guarded
    first draft (8 steps) -- two settings in lock-step, one forward per step;
  * batch 2 (int32 ids): a first draft in 4 steps (side file) next to a block with no side file (plain s8);
  * no side file anywhere == StaticDraftTraced (same tokens, KV-storing forward, next_token_ids);
  * side files are deleted once read; a mask-free block (a prefill chunk) never consumes its side file;
  * a wrong C1B_CODE_DIR fails the import with a message naming it.

    python3 -m pytest -q code/corpus1b/tests/test_judge_draft_plugin.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
PLUGIN = os.path.join(CODE, "corpus1b", "sglang_plugin")
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign.drafter import draft_block                                               # noqa: E402
from tests.test_da_decode import MASK, BL                                                # noqa: E402
from test_draft_core import (first_draft_case, redraft_case, guard_case, drafter_seq,    # noqa: E402
                             reference)

ALG = "sglang.srt.dllm.algorithm"


# --------------------------------------------------------------------------
# fake sglang
# --------------------------------------------------------------------------
class DllmAlgorithm:
    def __init__(self, config):
        self.block_size = config.block_size
        self.mask_id = config.mask_id


def _leaf(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def load(mp, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mp.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sgl():
    """Fake sglang modules for the plugins' imports; removed again afterwards (other tests assert no sglang)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("C1B_CODE_DIR", raising=False)            # the plugin finds code/ relative to its repo file
        for name in ("sglang", "sglang.srt", "sglang.srt.dllm", ALG, "sglang.srt.layers", "sglang.srt.model_executor"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            mp.setitem(sys.modules, name, pkg)
        leaves = {
            f"{ALG}.base": {"DllmAlgorithm": DllmAlgorithm},
            f"{ALG}.low_confidence": {"LowConfidence": type("LowConfidence", (DllmAlgorithm,), {})},
            "sglang.srt.dllm.config": {"DllmConfig": type("DllmConfig", (), {})},
            "sglang.srt.layers.logits_processor": {"LogitsProcessorOutput": type("LogitsProcessorOutput", (), {})},
            "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": type("ForwardBatch", (), {})},
            "sglang.srt.model_executor.model_runner": {"ModelRunner": type("ModelRunner", (), {})},
        }
        for name, attrs in leaves.items():
            mp.setitem(sys.modules, name, _leaf(name, **attrs))
        load(mp, f"{ALG}.c1b_traced", os.path.join(PLUGIN, "c1b_traced.py"))
        static = load(mp, f"{ALG}.c1b_static_draft", os.path.join(PLUGIN, "c1b_static_draft.py"))
        judge = load(mp, f"{ALG}.c1b_judge_draft", os.path.join(PLUGIN, "c1b_judge_draft.py"))
        yield SimpleNamespace(mp=mp, judge=judge, static=static)


CONFIG = SimpleNamespace(block_size=BL, mask_id=MASK, algorithm_config={"block_size": BL, "steps": 8})


class FakeRunner:
    """model_runner.forward: full_logits [batch*32, V], block b from its own drafter cache (store_kv=False)."""

    def __init__(self, cases):
        self.seqs = [drafter_seq(c.fn, c.prefix)[0] for c in cases]
        self.los = [c.lo for c in cases]
        self.n_fwd = 0

    def forward(self, fb, pp_proxy_tensors=None):
        self.n_fwd += 1
        rows = [seq.block_forward(fb.input_ids[b * BL:(b + 1) * BL].long().unsqueeze(0), lo, store_kv=False,
                                  kind="sgl")[0] for b, (seq, lo) in enumerate(zip(self.seqs, self.los))]
        return SimpleNamespace(logits_output=SimpleNamespace(full_logits=torch.cat(rows)), can_run_graph=False)


def batch(cases, rids, dtype=torch.long):
    return SimpleNamespace(input_ids=torch.cat([c.block for c in cases]).to(dtype), rids=list(rids),
                           extend_prefix_lens_cpu=[c.lo for c in cases], batch_size=len(cases))


def run_alg(alg, cases, rids, dtype=torch.long):
    fb, runner = batch(cases, rids, dtype), FakeRunner(cases)
    logits_output, next_ids, graph = alg.run(runner, fb)
    return fb, runner, logits_output, next_ids, graph


def write_side(side_dir, rid, spec):
    os.makedirs(side_dir, exist_ok=True)
    with open(os.path.join(side_dir, f"{rid}.json"), "w") as fh:
        json.dump(spec, fh)


def read_trace(trace_dir):
    rows = {}
    for p in glob.glob(os.path.join(trace_dir, "trace.*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows[r["rid"]] = r
    return rows


def new_alg(cls, monkeypatch, trace_dir, side_dir):
    monkeypatch.setenv("C1B_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("C1B_SIDE_DIR", str(side_dir))
    return cls(CONFIG)


def assert_row(row, rid, case, ref):
    assert row["rid"] == rid and row["lo"] == case.lo
    assert row["start"] == BL - int((case.block == MASK).sum())
    assert row["ids"] == ref["tokens"].tolist()
    assert row["conf"] == torch.where(case.fill, ref["p_commit"], torch.tensor(-1.0)).tolist()   # exact fp32
    assert row["fill"] == torch.nonzero(case.fill).flatten().tolist()
    for k in ("n_fwd", "raw_mask_argmax", "guard_slots", "guard_changed"):
        assert row[k] == ref[k], k


def side_spec(case) -> dict:
    spec = {"fill": torch.nonzero(case.fill).flatten().tolist(), "steps": case.steps}
    if case.ban is not None:
        spec["ban"] = [[s, int(case.ban[s])] for s in range(BL) if int(case.ban[s]) >= 0]
    if case.guard is not None:
        spec["guard"] = case.guard
    return spec


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_redraft_and_guard_blocks_in_one_batch(sgl, tmp_path, monkeypatch):
    rd, gd = redraft_case(), guard_case()
    side = tmp_path / "side"
    write_side(side, "item7#0", side_spec(rd))
    write_side(side, "item9#2", {"guard": gd.guard})               # fill and steps default: every mask, 8
    alg = new_alg(sgl.judge.DraftTracedV2, monkeypatch, tmp_path / "trace", side)
    fb, runner, _, next_ids, graph = run_alg(alg, [rd, gd], ["item7#0", "item9#2"])
    assert graph is False and runner.n_fwd == 8 + 1                 # max steps + the KV-storing forward
    rows = read_trace(tmp_path / "trace")
    for rid, case in (("item7#0", rd), ("item9#2", gd)):
        ref, _, _ = reference(case)
        assert_row(rows[rid], rid, case, ref)
    assert rows["item7#0"]["n_fwd"] == 3 and rows["item9#2"]["guard_changed"] == 3
    # the closing forward saw the finished blocks; next_token_ids = the block from start
    ids = fb.input_ids.view(2, BL)
    for i, rid in enumerate(("item7#0", "item9#2")):
        assert ids[i].tolist() == rows[rid]["ids"]
        assert next_ids[i].tolist() == rows[rid]["ids"][rows[rid]["start"]:]
        assert runner.seqs[i].model.calls[-1]["input_ids"][0].tolist() == rows[rid]["ids"]
    assert os.listdir(side) == []                                   # side files are consumed


def test_four_step_draft_next_to_a_plain_s8_block(sgl, tmp_path, monkeypatch):
    a4 = first_draft_case(3, 70, steps=4)                           # lo = 64, 6-slot prompt tail
    plain = first_draft_case(0, 37)
    side = tmp_path / "side"
    write_side(side, "p#1", {"steps": 4})
    alg = new_alg(sgl.judge.DraftTracedV2, monkeypatch, tmp_path / "trace", side)
    fb, runner, _, next_ids, _ = run_alg(alg, [a4, plain], ["p#1", "q#0"], dtype=torch.int32)
    assert fb.input_ids.dtype == torch.int32 and runner.n_fwd == 8 + 1
    rows = read_trace(tmp_path / "trace")
    ref4, _, _ = reference(a4)                                      # draft_block(..., static_steps=4)
    assert_row(rows["p#1"], "p#1", a4, ref4)
    assert rows["p#1"]["n_fwd"] == 4
    seq, lo = drafter_seq(plain.fn, plain.prefix)                   # the plain s8 setting, no overrides
    ref8 = draft_block(seq, plain.block.clone(), plain.fill, lo, "s8", MASK)
    assert_row(rows["q#0"], "q#0", plain, ref8)
    assert rows["q#0"]["n_fwd"] == 8 and rows["q#0"]["raw_mask_argmax"] > 0


def test_no_side_file_equals_static_draft_traced(sgl, tmp_path, monkeypatch):
    cases = [first_draft_case(0, 37), guard_case()]                 # the guard case WITHOUT its guard
    rids = ["a#0", "b#2"]
    v2 = new_alg(sgl.judge.DraftTracedV2, monkeypatch, tmp_path / "t2", tmp_path / "empty")
    fb2, r2, _, next2, _ = run_alg(v2, cases, rids)
    st = new_alg(sgl.static.StaticDraftTraced, monkeypatch, tmp_path / "t1", tmp_path / "empty")
    fb1, r1, _, next1, _ = run_alg(st, cases, rids)
    assert torch.equal(fb1.input_ids, fb2.input_ids) and r1.n_fwd == r2.n_fwd == 9
    assert all(torch.equal(a, b) for a, b in zip(next1, next2))
    rows1, rows2 = read_trace(tmp_path / "t1"), read_trace(tmp_path / "t2")
    for rid, case in zip(rids, cases):
        assert rows1[rid]["ids"] == rows2[rid]["ids"] and rows1[rid]["start"] == rows2[rid]["start"]
        assert rows1[rid]["conf"] == [round(c, 5) for c in rows2[rid]["conf"]]
        seq, lo = drafter_seq(case.fn, case.prefix)
        ref = draft_block(seq, case.block.clone(), case.fill, lo, "s8", MASK)
        assert_row(rows2[rid], rid, case, ref)
    assert rows2["b#2"]["guard_slots"] == 0                          # no side file -> no guard


def test_prefill_chunk_keeps_its_side_file_and_fill_is_clipped_to_masks(sgl, tmp_path, monkeypatch):
    a = first_draft_case(0, 37)
    pre = SimpleNamespace(fn=a.fn, prefix=a.prefix[:, :0], lo=0, block=a.prefix[0, :BL].clone(), fill=None)
    side = tmp_path / "side"
    write_side(side, "a#0", {"fill": [0, 1, 6, 9, 40]})             # slots 0, 1 hold the prompt tail; 40 is out of range
    write_side(side, "pre#0", {"steps": 2})
    alg = new_alg(sgl.judge.DraftTracedV2, monkeypatch, tmp_path / "trace", side)
    fb, runner, _, next_ids, _ = run_alg(alg, [a, pre], ["a#0", "pre#0"])
    assert sorted(os.listdir(side)) == ["pre#0.json"]               # the mask-free block did not consume it
    rows = read_trace(tmp_path / "trace")
    assert rows["a#0"]["fill"] == [6, 9] and rows["a#0"]["n_fwd"] == 2
    fill = torch.zeros(BL, dtype=torch.bool)
    fill[[6, 9]] = True
    seq, lo = drafter_seq(a.fn, a.prefix)
    ref = draft_block(seq, a.block.clone(), fill, lo, "s8", MASK)
    assert_row(rows["a#0"], "a#0", SimpleNamespace(lo=a.lo, block=a.block, fill=fill), ref)
    assert rows["pre#0"]["fill"] == [] and rows["pre#0"]["n_fwd"] == 0 and next_ids[1].numel() == 0
    assert rows["pre#0"]["ids"] == pre.block.tolist()


def test_prefill_only_batch_is_one_forward(sgl, tmp_path, monkeypatch):
    a = first_draft_case(0, 37)
    pre = SimpleNamespace(fn=a.fn, prefix=a.prefix[:, :0], lo=0, block=a.prefix[0, :BL].clone())
    alg = new_alg(sgl.judge.DraftTracedV2, monkeypatch, tmp_path / "trace", tmp_path / "side")
    _, runner, _, next_ids, _ = run_alg(alg, [pre], ["pre#0"])
    assert runner.n_fwd == 1 and next_ids == [] and read_trace(tmp_path / "trace") == {}


def test_wrong_code_dir_fails_clearly(sgl, tmp_path, monkeypatch):
    monkeypatch.setenv("C1B_CODE_DIR", str(tmp_path))
    with pytest.raises(ImportError, match="C1B_CODE_DIR"):
        load(monkeypatch, f"{ALG}.c1b_judge_draft_badenv", os.path.join(PLUGIN, "c1b_judge_draft.py"))
