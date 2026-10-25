"""CPU tests for judge_round/judge_sglang.py (no SGLang, no model, no GPU).

  * the CLI defaults and argument checks; the yaml the judge plugin reads (round trip through yaml.safe_load);
    the memory-fraction conversion; the drafter command and the judge environment (every C1B_* name the plugin needs);
  * the judge side files: decode (prompt, max_new_tokens, EOS stop ids) and record (teacher blocks from a synthetic
    trainset item, the prompt tail inside block 0, max_new_tokens = 32 nb - tail);
  * the trace -> output: block ordering checks, gen_ids cut after the first EOS, the decode record's sums, the record
    mode checks, and the states npz (flat arrays + offsets, loads with allow_pickle=False);
  * resume: done.txt plus the rids already in records.jsonl; main on a finished shard never imports sglang.

Run from the repo root:  python3 -m pytest -q code/tests/test_judge_sglang_cli.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import yaml

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "compat"))

from judge_round import judge_sglang as JS                               # noqa: E402

BL = 32
EOS = JS.EOS_IDS


def args_of(*extra):
    return JS.build_parser().parse_args(["--mode", "decode", "--items", "gsm8k500", "--gpu", "3", *extra])


def make_item(pid="maths-t-0001", L=37, nb=3, seed=0):
    rng = np.random.default_rng(seed)
    prompt = rng.integers(1000, 2000, size=L).astype(np.int32)
    lo0 = (L // BL) * BL
    tokens = rng.integers(3000, 4000, size=(nb, BL)).astype(np.int32)
    tokens[0, :L - lo0] = prompt[lo0:L]                          # the prompt tail duplicated in block 0
    return {"pid": pid, "prompt_ids": prompt, "tokens": tokens,
            "block_start": (lo0 + BL * np.arange(nb)).astype(np.int32), "prompt_length": np.int32(L),
            "domain": "maths"}


def summary(tokens, n_judge=2, **kw):
    s = {k: 0 for k in JS.COUNTERS}
    s.update({"lo": 0, "n_judge": n_judge, "passes": [{}] * n_judge, "tokens": list(tokens), "tokens_out": list(tokens),
              "draft_tokens": [7] * BL, "p_draft": [0.5] * BL})
    s.update(kw)
    return s


def rows_for(prompt_len, blocks, **kw):
    """Judge trace rows {lo: row} for consecutive blocks starting at the prompt's block boundary."""
    lo0 = (prompt_len // BL) * BL
    return {lo0 + BL * k: {"rid": "r", "block": k, "lo": lo0 + BL * k, "start": prompt_len - lo0 if k == 0 else 0,
                           "summary": summary(b, **kw)} for k, b in enumerate(blocks)}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_parser_defaults():
    a = args_of()
    assert (a.redraft_every, a.redraft_rate, a.redraft_ban, a.loop_guard, a.tau_commit, a.max_passes) == \
        (1, 4, False, 12, 0.95, 32)
    assert (a.max_running, a.mem_fraction, a.drafter_mem_fraction, a.chunk, a.shard, a.nshards) == \
        (64, 0.62, 0.18, 256, 0, 1)
    assert a.judge_adapter == os.path.join("logs", "c1b", "train", "main_128m", "tok_000128000000")
    assert a.drafter_dir == os.path.join("logs", "phase4", "train", "main", "merged")
    assert a.max_new_tokens is None and a.attention_backend is None and a.limit == 0
    assert JS.drafter_port(a) == 30103 and JS.drafter_port(args_of("--drafter-port", "31000")) == 31000
    JS.check_args(a)
    r = JS.build_parser().parse_args(["--mode", "record", "--index", "a.jsonl", "--index", "b.jsonl",
                                      "--pids", "p.json", "--gpu", "0"])
    assert r.index == ["a.jsonl", "b.jsonl"]
    JS.check_args(r)


@pytest.mark.parametrize("argv", [
    ["--mode", "decode", "--gpu", "0"],                                            # no --items
    ["--mode", "record", "--index", "a.jsonl", "--gpu", "0"],                      # no --pids
    ["--mode", "decode", "--items", "x", "--gpu", "0", "--redraft-every", "0", "--redraft-ban"],
    ["--mode", "decode", "--items", "x", "--gpu", "0", "--loop-guard", "1"],
    ["--mode", "record", "--index", "a", "--pids", "p", "--gpu", "0", "--max-new-tokens", "64"],
])
def test_bad_arguments(argv):
    with pytest.raises(AssertionError):
        JS.check_args(JS.build_parser().parse_args(argv))


def test_yaml_round_trip(tmp_path):
    cfg = JS.judge_yaml_config(args_of("--redraft-ban", "--redraft-every", "2", "--loop-guard", "0",
                                       "--tau-commit", "0.9", "--max-passes", "8"))
    assert cfg == {"block_size": 32, "tau_commit": 0.9, "tau_override": None, "max_passes": 8, "redraft_every": 2,
                   "redraft_rate": 4, "redraft_ban": True, "loop_guard": 0, "threshold": 0.95, "draft_steps": 8}
    path = JS.write_yaml(str(tmp_path / "judge_3.yaml"), cfg)
    assert yaml.safe_load(open(path)) == cfg                                  # what SGLang's DllmConfig reads


def test_memory_fraction_conversion():
    f = JS.judge_sglang_fraction(0.62, 0.18)
    assert abs(f - 0.62 / 0.80) < 1e-12
    assert abs(f * (1 - 0.18 - JS.DRAFTER_OVERHEAD) - 0.62) < 1e-12              # = 0.62 of the whole GPU
    with pytest.raises(AssertionError):
        JS.judge_sglang_fraction(0.85, 0.18)


def test_drafter_command_and_judge_environment(tmp_path):
    cmd = JS.drafter_cmd("/m/drafter", 30103, 0.18, "triton")
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    pairs = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    assert pairs["--model-path"] == pairs["--tokenizer-path"] == "/m/drafter"
    assert pairs["--dllm-algorithm"] == "DraftTracedV2" and pairs["--port"] == "30103"
    assert pairs["--host"] == "127.0.0.1" and pairs["--mem-fraction-static"] == "0.1800"
    assert pairs["--max-running-requests"] == "512" and pairs["--context-length"] == "4096"
    assert pairs["--attention-backend"] == "triton"
    assert pairs["--dllm-algorithm-config"].endswith(os.path.join("sglang_plugin", "draft_s8.yaml"))
    for flag in ("--skip-tokenizer-init", "--disable-radix-cache", "--disable-cuda-graph", "--trust-remote-code"):
        assert flag in cmd
    dirs = {k: str(tmp_path / k) for k in JS.SIDE_DIRS}
    denv = JS.drafter_env(dirs, 3)
    assert denv["C1B_SIDE_DIR"] == dirs["drafter_side"] and denv["C1B_TRACE_DIR"] == dirs["drafter_trace"]
    assert denv["C1B_CODE_DIR"] == CODE
    jenv = JS.judge_env(dirs, "/a/adapter", "http://127.0.0.1:30103")
    plugin = open(os.path.join(CODE, "corpus1b", "sglang_plugin", "c1b_judge_loop.py")).read()
    for k in jenv:
        assert f'"{k}"' in plugin, k                                           # every name the plugin reads
    assert jenv["C1B_DRAFTER_SIDE_DIR"] == denv["C1B_SIDE_DIR"]
    assert jenv["C1B_DRAFTER_TRACE_DIR"] == denv["C1B_TRACE_DIR"]
    assert JS.sampling_params("decode", 1024) == {"temperature": 0.0, "max_new_tokens": 1024, "ignore_eos": False,
                                                  "stop_token_ids": EOS}
    assert JS.sampling_params("record", 91) == {"temperature": 0.0, "max_new_tokens": 91, "ignore_eos": True}


# ---------------------------------------------------------------------------
# side files
# ---------------------------------------------------------------------------
def test_decode_side_file(tmp_path):
    spec = JS.decode_side_spec(np.arange(5, 45), 1024)
    assert spec == {"prompt_ids": list(range(5, 45)), "mode": "decode", "max_new_tokens": 1024, "stop_ids": EOS}
    path = JS.write_side(str(tmp_path), "gsm8k500#12", spec)
    assert json.load(open(path)) == spec and os.listdir(tmp_path) == ["gsm8k500#12.json"]
    with pytest.raises(AssertionError):
        JS.write_side(str(tmp_path), "a/b", spec)
    assert JS.block_end(37, 1024) == 1088 and JS.block_end(64, 64) == 128 and JS.block_end(37, 27) == 64


@pytest.mark.parametrize("L,nb", [(37, 3), (64, 2), (70, 1)])
def test_record_side_file_from_a_trainset_item(L, nb):
    it = make_item(L=L, nb=nb)
    q = JS.record_request(it)
    lo0, tail = (L // BL) * BL, L - (L // BL) * BL
    assert q["rid"] == "maths-t-0001" and q["prompt_ids"] == it["prompt_ids"].tolist()
    assert q["max_new"] == BL * nb - tail and JS.block_end(L, q["max_new"]) == lo0 + BL * nb
    spec = q["spec"]
    assert spec["mode"] == "record" and spec["stop_ids"] == [] and spec["max_new_tokens"] == q["max_new"]
    assert spec["teacher"] == it["tokens"].tolist() and len(spec["teacher"]) == nb
    assert spec["teacher"][0][:tail] == it["prompt_ids"][lo0:].tolist()          # the prompt tail inside block 0
    assert all(len(b) == BL for b in spec["teacher"])
    json.dumps(spec)                                                              # plain ints


def test_record_request_refuses_a_bad_item():
    it = make_item(L=37, nb=2)
    it["tokens"][0, 0] += 1                                                       # block 0 no longer holds the tail
    with pytest.raises(AssertionError, match="prompt tail"):
        JS.record_request(it)
    it = make_item(L=37, nb=2)
    it["block_start"] = it["block_start"] + BL
    with pytest.raises(AssertionError):
        JS.record_request(it)


# ---------------------------------------------------------------------------
# trace -> output
# ---------------------------------------------------------------------------
def test_trim_eos():
    assert JS.trim_eos([5, 6, EOS[1], 7, EOS[0]]) == [5, 6, EOS[1]]
    assert JS.trim_eos([5, 6, 7]) == [5, 6, 7]
    assert JS.trim_eos([EOS[0]]) == [EOS[0]] and JS.trim_eos([]) == []


def test_assemble_blocks_checks_the_order():
    b = [[11] * BL, [12] * BL, [13] * BL]
    rows = rows_for(37, b)
    got = JS.assemble_blocks(rows, 37)
    assert [r["block"] for r in got] == [0, 1, 2] and JS.generated(got) == [11] * 27 + [12] * BL + [13] * BL
    with pytest.raises(JS.TraceError, match="lo0"):
        JS.assemble_blocks({k: v for k, v in rows.items() if k != 32}, 37)
    with pytest.raises(JS.TraceError, match="contiguous"):
        JS.assemble_blocks({k: v for k, v in rows.items() if k != 64}, 37)
    bad = rows_for(37, b)
    bad[32]["start"] = 0
    with pytest.raises(JS.TraceError, match="start"):
        JS.assemble_blocks(bad, 37)


def test_decode_record_from_trace_rows():
    blk0 = [9] * 5 + [100 + i for i in range(27)]                                 # prompt tail 5
    blk1 = [200 + i for i in range(BL)]
    blk1[10] = EOS[0]
    rows = rows_for(37, [blk0, blk1])
    rows[32]["summary"].update(n_judge=3, n_redraft_rounds=2, n_loopguard_slots=4, n_stock=0)
    rows[64]["summary"].update(n_judge=5, n_redraft_rounds=1, full_fallback=2, n_stock=3)
    blocks = JS.assemble_blocks(rows, 37)
    meta = {"rid": "gsm8k500#4", "idx": 4, "split": "test", "gold": "42"}
    rec = JS.decode_record(meta, list(range(37)), blocks, lambda ids: f"<{len(ids)}>", {"x": 1}, 1.23456,
                           output_ids=JS.generated(blocks)[:40])
    assert rec["gen_ids"] == blk0[5:] + blk1[:11] and rec["stopped_eos"] and rec["pred"] == "<38>"
    assert (rec["n_blocks"], rec["n_judge"], rec["n_redraft_rounds"], rec["n_loopguard_slots"]) == (2, 8, 3, 4)
    assert (rec["full_fallback"], rec["n_stock"], rec["passes_per_block"]) == (2, 3, [3, 5])
    assert (rec["idx"], rec["split"], rec["rid"], rec["gold"], rec["prompt_tokens"]) == (4, "test", "gsm8k500#4", "42", 37)
    assert rec["engine"] == "sglang" and rec["cfg"] == {"x": 1} and rec["wall_s"] == 1.235
    for k in JS.SUM_KEYS:
        assert k in rec
    no_eos = JS.decode_record(meta, list(range(37)), JS.assemble_blocks(rows_for(37, [blk0]), 37),
                              lambda ids: "", {}, 0.0)
    assert no_eos["gen_ids"] == blk0[5:] and not no_eos["stopped_eos"]
    with pytest.raises(JS.TraceError, match="returned"):
        JS.decode_record(meta, list(range(37)), blocks, lambda ids: "", {}, 0.0, output_ids=[1, 2, 3])


def test_record_mode_checks():
    it = make_item(L=37, nb=2)
    q = JS.record_request(it)
    st = {"pass": 0}
    rows = rows_for(37, q["teacher"], n_judge=1)
    for r in rows.values():
        r["states"] = [st]
    blocks = JS.assemble_blocks(rows, 37)
    JS.check_record_blocks(blocks, q, output_ids=JS.generated(blocks))
    with pytest.raises(JS.TraceError, match="blocks traced"):
        JS.check_record_blocks(blocks[:1], q)
    rows[64]["summary"]["tokens_out"] = [1] * BL
    with pytest.raises(JS.TraceError, match="teacher"):
        JS.check_record_blocks(JS.assemble_blocks(rows, 37), q)
    rows = rows_for(37, q["teacher"], n_judge=2)
    for r in rows.values():
        r["states"] = [st]
    with pytest.raises(JS.TraceError, match="states"):
        JS.check_record_blocks(JS.assemble_blocks(rows, 37), q)


def _state(k, redraft):
    open_ = [s >= 3 for s in range(BL)]
    return {"pass": k, "input": [k + s for s in range(BL)], "flag": [s >= 4 for s in range(BL)], "open": open_,
            "draft": [7] * BL, "commit": [s == 3 for s in range(BL)], "tentative": [s >= 4 for s in range(BL)],
            "remask": [False] * BL, "forced": k == 1, "argmax": [s if o else -1 for s, o in enumerate(open_)],
            "p": [0.5 if o else None for o in open_], "q": [0.25 if o else None for o in open_],
            "redraft": None if not redraft else {"holes": [s == 5 for s in range(BL)], "tokens": [8] * BL,
                                                 "p_commit": [0.0] * 5 + [0.75] + [0.0] * 26,
                                                 "input": [9] * BL, "n_fwd": 1}}


def test_pack_states_round_trip(tmp_path):
    a = rows_for(37, [[1] * BL, [2] * BL], n_judge=2)
    a[32]["states"] = [_state(0, True), _state(1, False)]
    a[64]["states"] = [_state(0, False), _state(1, True)]
    b = rows_for(64, [[3] * BL], n_judge=1, p_draft=None, draft_tokens=None)
    b[64]["states"] = [_state(0, False)]
    entries = [("item-a", JS.assemble_blocks(a, 37)), ("item-b", JS.assemble_blocks(b, 64))]
    path = str(tmp_path / "states.00000.npz")
    np.savez_compressed(path, **JS.pack_states(entries, {"loop_guard": 12}))
    z = np.load(path, allow_pickle=False)
    index = json.loads(str(z["index"]))
    assert index["cfg"] == {"loop_guard": 12} and index["counters"] == list(JS.COUNTERS)
    assert index["items"] == [{"rid": "item-a", "n_blocks": 2}, {"rid": "item-b", "n_blocks": 1}]
    assert z["rid"].tolist() == ["item-a", "item-b"] and z["item_block"].tolist() == [0, 2, 3]
    assert z["blk_pass"].tolist() == [0, 2, 4, 5] and z["blk_lo"].tolist() == [32, 64, 64]
    assert z["blk_start"].tolist() == [5, 0, 0] and z["blk_index"].tolist() == [0, 1, 0]
    assert z["blk_tokens"].shape == (3, BL) and (z["blk_teacher"][2] == 3).all()
    assert (z["blk_draft"][2] == -1).all() and np.isnan(z["blk_p_draft"][2]).all() and (z["blk_p_draft"][0] == 0.5).all()
    assert z["blk_counters"].shape == (3, len(JS.COUNTERS))
    assert z["blk_counters"][0, list(JS.COUNTERS).index("n_judge")] == 2
    assert z["pass_no"].tolist() == [0, 1, 0, 1, 0] and z["forced"].tolist() == [False, True, False, True, False]
    assert z["input"].dtype == np.int32 and z["flag"].dtype == bool and z["p"].dtype == np.float32
    assert z["input"][1].tolist() == [1 + s for s in range(BL)]
    assert np.isnan(z["p"][0, :3]).all() and (z["p"][0, 3:] == 0.5).all() and (z["q"][0, 3:] == 0.25).all()
    assert z["argmax"][0, 0] == -1 and z["commit"][0].sum() == 1
    assert z["rd_on"].tolist() == [True, False, False, True, False] and z["rd_n_fwd"].tolist() == [1, 0, 0, 1, 0]
    assert z["rd_holes"][0].sum() == 1 and z["rd_p_commit"][0, 5] == 0.75 and (z["rd_tokens"][1] == -1).all()
    assert np.isnan(z["rd_p_commit"][1]).all() and (z["rd_input"][3] == 9).all()
    empty = JS.pack_states([("x", JS.assemble_blocks(rows_for(37, [[1] * BL], n_judge=0), 37))], {})
    assert empty["input"].shape == (0, BL) and empty["blk_pass"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# resume / main
# ---------------------------------------------------------------------------
def test_read_done_counts_records(tmp_path):
    g = str(tmp_path)
    JS.append_lines(os.path.join(g, "done.txt"), ["a#1", "a#2"])
    with open(os.path.join(g, "records.jsonl"), "w") as fh:
        fh.write(json.dumps({"rid": "a#3"}) + "\n" + '{"rid": "a#4"')               # a truncated last line
    assert JS.read_done(g) == {"a#1", "a#2", "a#3"}
    open(os.path.join(g, "states.00004.npz.tmp.npz"), "w").close()
    open(os.path.join(g, "states.00002.npz"), "w").close()
    assert JS.next_states_index(g) == 3


def test_main_on_a_finished_shard_never_starts_an_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(JS.P, "safe_out", lambda p: p)
    tmp = str(tmp_path)
    with open(os.path.join(tmp, "index.jsonl"), "w") as fh:
        for pid in ("a-1", "b-2"):
            fh.write(json.dumps({"file": f"items/{pid}.npz", "pid": pid, "domain": "maths"}) + "\n")
    with open(os.path.join(tmp, "pids.json"), "w") as fh:
        json.dump({"pids": ["b-2", "a-1"]}, fh)
    for d, f in (("adapter", "adapter_config.json"), ("drafter", "config.json")):
        os.makedirs(os.path.join(tmp, d))
        open(os.path.join(tmp, d, f), "w").write("{}")
    out = os.path.join(tmp, "out")
    os.makedirs(os.path.join(out, "gpu0"))
    JS.append_lines(os.path.join(out, "gpu0", "done.txt"), ["a-1", "b-2"])
    argv = ["--mode", "record", "--index", os.path.join(tmp, "index.jsonl"), "--pids", os.path.join(tmp, "pids.json"),
            "--judge-adapter", os.path.join(tmp, "adapter"), "--drafter-dir", os.path.join(tmp, "drafter"),
            "--out", out, "--gpu", "0"]
    JS.main(argv)
    assert "sglang" not in sys.modules
    meta = json.load(open(os.path.join(out, "gpu0", "run.json")))
    assert meta["mode"] == "record" and meta["cfg"]["loop_guard"] == 12
    with pytest.raises(AssertionError, match="another configuration"):
        JS.main(argv + ["--loop-guard", "6"])                                       # a resume with another config
