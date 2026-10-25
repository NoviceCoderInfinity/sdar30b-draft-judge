"""Wave C plumbing in the three Phase 1 scripts: gen_sglang (--wave C, --prompt-dir, --extra-done), draft_sglang
(--wave C, --drafter-dir -> engine arguments + drafter.json) and to_trainset (--wave C defaults). CPU only, no SGLang.

    python -m pytest -q code/corpus1b/tests/test_wave_c_flags.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
C1B = os.path.abspath(os.path.join(HERE, ".."))
CODE = os.path.abspath(os.path.join(C1B, ".."))
for p in (os.path.join(CODE, "compat"), CODE, C1B):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                        # noqa: E402
import draft_sglang as D                                                  # noqa: E402
import gen_sglang as G                                                    # noqa: E402
import to_trainset as TS                                                  # noqa: E402


def write_list(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- gen_sglang
def test_gen_parser_accepts_wave_c_flags():
    a = G.build_parser().parse_args(["--wave", "C", "--gpu", "0", "--prompt-dir", "X",
                                     "--extra-done", "a", "--extra-done", "b"])
    assert a.wave == "C" and a.prompt_dir == "X" and a.extra_done == ["a", "b"]
    d = G.build_parser().parse_args(["--wave", "A", "--gpu", "0"])                # defaults are unchanged
    assert d.prompt_dir == C.PROMPT_DIR and d.extra_done == []
    with pytest.raises(SystemExit):
        G.build_parser().parse_args(["--wave", "D", "--gpu", "0"])


def test_load_rows_reads_the_prompt_dir_wave_c_only_minus_done(tmp_path):
    ngpu = 2
    c_rows = [{"pid": f"maths-c-{i:03d}", "domain": "maths", "wave": "C", "prompt": "p"} for i in range(40)]
    b_rows = [{"pid": f"maths-b-{i:03d}", "domain": "maths", "wave": "B", "prompt": "p"} for i in range(10)]
    write_list(tmp_path / "maths.jsonl", c_rows + b_rows)
    mine = [r for r in c_rows if G.shard_of(r["pid"], ngpu) == 1]
    assert 0 < len(mine) < len(c_rows)                                           # the shard split is non-trivial
    done = {mine[0]["pid"], mine[1]["pid"], c_rows[0]["pid"]}
    got = G.load_rows("C", ["maths"], 1, ngpu, done, prompt_dir=str(tmp_path))
    assert {r["pid"] for r in got} == {r["pid"] for r in mine} - done
    assert all(r["wave"] == "C" for r in got)
    with pytest.raises(AssertionError):                                          # a missing list is loud, in that dir
        G.load_rows("C", ["code"], 1, ngpu, set(), prompt_dir=str(tmp_path))


def test_load_extra_done_unions_a_glob_of_pid_files(tmp_path, capsys):
    for k, pids in enumerate((["a1", "a2"], ["a2", "b1"])):
        d = tmp_path / "gen" / f"wave{'AB'[k]}" / "gpu0"
        d.mkdir(parents=True)
        (d / "done.txt").write_text("\n".join(pids) + "\n")
    lone = tmp_path / "lone.txt"
    lone.write_text("x1\n")
    got = G.load_extra_done([str(tmp_path / "gen" / "wave*" / "gpu*" / "done.txt"), str(lone)])
    assert got == {"a1", "a2", "b1", "x1"}
    assert "extra done: 4 pids from 3 files" in capsys.readouterr().out
    assert G.load_extra_done([]) == set()
    # the guard end to end: a wave-C list that overlaps a wave-A pid never yields it
    write_list(tmp_path / "maths.jsonl", [{"pid": "a1", "domain": "maths", "wave": "C"},
                                          {"pid": "c1", "domain": "maths", "wave": "C"}])
    rows = G.load_rows("C", ["maths"], 0, 1, got, prompt_dir=str(tmp_path))
    assert [r["pid"] for r in rows] == ["c1"]


# ------------------------------------------------------------- draft_sglang
def test_draft_parser_and_engine_kwargs_use_the_drafter_override(monkeypatch, tmp_path):
    assert "sglang" not in sys.modules                                           # imported only inside main
    monkeypatch.setattr(D, "snapshot_dir", lambda key: f"/pinned/{key}")
    ap = D.build_parser()
    a = ap.parse_args(["--wave", "C", "--gpu", "0", "--drafter-dir", str(tmp_path)])
    assert a.wave == "C" and a.drafter_dir == str(tmp_path)
    kw = D.engine_kwargs(a)
    assert kw["model_path"] == kw["tokenizer_path"] == str(tmp_path)
    assert kw["dllm_algorithm"] == "StaticDraftTraced" and "attention_backend" not in kw
    stock = D.engine_kwargs(ap.parse_args(["--wave", "A", "--gpu", "0", "--attention-backend", "triton"]))
    assert stock["model_path"] == stock["tokenizer_path"] == "/pinned/d32"       # no override: today's behaviour
    assert stock["attention_backend"] == "triton"
    with pytest.raises(SystemExit):
        ap.parse_args(["--wave", "D", "--gpu", "0"])


def test_draft_main_records_the_drafter_and_refuses_a_mixed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(D.P, "safe_out", lambda p: os.path.join(str(tmp_path), p))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    base = ["--wave", "C", "--gpu", "0"]
    D.main(base + ["--drafter-dir", str(ckpt)])                                  # no kept records: returns before sglang
    meta = tmp_path / "logs" / "c1b" / "draft" / "waveC" / "gpu0" / "drafter.json"
    assert json.loads(meta.read_text()) == {"drafter_dir": str(ckpt), "wave": "C"}
    D.main(base + ["--drafter-dir", str(ckpt)])                                  # same drafter: a plain resume
    with pytest.raises(AssertionError):
        D.main(base)                                                             # stock d32 into a dir drafted by another
    with pytest.raises(AssertionError):
        D.main(base + ["--drafter-dir", str(tmp_path / "nope")])                 # not a checkpoint
    assert "sglang" not in sys.modules


# -------------------------------------------------------------- to_trainset
def test_to_trainset_wave_c_derives_its_default_dirs(tmp_path, monkeypatch):
    seen = []

    def safe_out(p):
        seen.append(p)
        return os.path.join(str(tmp_path), p)

    monkeypatch.setattr(TS.P, "safe_out", safe_out)
    s = TS.main(["--wave", "C"])                                                 # empty gen/draft dirs: writes nothing
    assert seen == [os.path.join("logs", "c1b", "gen", "waveC"), os.path.join("logs", "c1b", "draft", "waveC"),
                    os.path.join("logs", "c1b", "trainset", "waveC")]
    assert s["wave"] == "C" and s["out"] == os.path.join(str(tmp_path), "logs", "c1b", "trainset", "waveC")
    assert s["counts"]["written"] == 0
    with pytest.raises(SystemExit):
        TS.main(["--wave", "D"])
