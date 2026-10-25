"""CPU tests for the fresh judge round's index builder (code/judge_round/build_fresh_index.py).  Tiny synthetic indexes in
tmp_path with real (empty) item files, so os.path.relpath and the existence check are exercised for real.

Pins: every fresh row is in the output; repeated science rows come in common.order_key(pid) order and stop once the corpus-token
budget is reached (overshoot < one item), carry "repeated": true and keep their wave; excluded pids (.txt, .json list,
{"pids": ...}) are skipped; "file" resolves to the same existing item from the OUTPUT index's directory (also through the
trainer's own loader); idx is renumbered 0..n-1, fresh first; other fields are untouched; build_summary.json totals match
the rows; a budget the pool cannot meet is a reported shortfall (--strict: exit 3); duplicate pids and missing files are errors.

    python -m pytest -q code/tests/test_build_fresh_index.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                        # noqa: E402
from judge_round import build_fresh_index as B                            # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _write_index(root, wave, spec):
    """spec = [(domain, n_items, tokens_each)]; writes <root>/wave<W>/index.jsonl + items/<h2>/<pid>.npz; returns rows."""
    d = root / f"wave{wave}"
    rows = []
    for dom, n, tok in spec:
        for i in range(n):
            pid = f"{dom}-src-{wave}{i:04d}"
            rel = os.path.join("items", C.sha256_hex(pid)[:2], f"{pid}.npz")
            os.makedirs(d / os.path.dirname(rel), exist_ok=True)
            (d / rel).write_bytes(b"x")
            rows.append({"file": rel, "idx": len(rows), "split": f"c1b_{dom}", "source": "src", "domain": dom, "pid": pid,
                         "wave": wave, "correct": 1, "n_blocks": 4, "usable_blocks": 3, "draft_blocks": 3,
                         "usable": [True, True, True, False], "corpus_tokens": tok + i, "n_gen": tok // 2})
    with open(d / "index.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


@pytest.fixture
def waves(tmp_path):
    root = tmp_path / "trainset"
    c = _write_index(root, "C", [("maths", 5, 500), ("code", 4, 510), ("chat", 6, 360), ("if", 3, 315)])
    a = _write_index(root, "A", [("science", 8, 100), ("maths", 3, 490)])
    b = _write_index(root, "B", [("science", 7, 100), ("code", 2, 500)])
    return {"root": root, "C": c, "A": a, "B": b,
            "fresh": [str(root / "waveC" / "index.jsonl")],
            "rep": [str(root / "waveA" / "index.jsonl"), str(root / "waveB" / "index.jsonl")],
            "out": str(root / "fresh" / "index.jsonl")}


def _run(w, budget="science=450", exclude=(), extra=()):
    argv = ["--out", w["out"]]
    for f in w["fresh"]:
        argv += ["--fresh", f]
    for r in w["rep"]:
        argv += ["--repeat-index", r]
    if budget:
        argv += ["--repeat", budget]
    for e in exclude:
        argv += ["--exclude-pids", e]
    rc = B.main(argv + list(extra))
    rows = [json.loads(l) for l in open(w["out"]) if l.strip()]
    summ = json.load(open(os.path.join(os.path.dirname(w["out"]), "build_summary.json")))
    return rc, rows, summ


def _science_by_order(w):
    sci = [r for r in w["A"] + w["B"] if r["domain"] == "science"]
    return sorted(sci, key=lambda r: C.order_key(r["pid"]))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_all_fresh_rows_present_first_and_untouched(waves):
    rc, rows, _ = _run(waves)
    assert rc == 0
    n = len(waves["C"])
    assert [r["pid"] for r in rows[:n]] == [r["pid"] for r in waves["C"]]
    for out, src in zip(rows[:n], waves["C"]):
        assert "repeated" not in out
        for k, v in src.items():
            if k not in ("file", "idx"):
                assert out[k] == v, k                               # usable lists, corpus_tokens, wave, ... untouched
    assert all(r["wave"] == "C" for r in rows[:n])


def test_repeated_science_stops_at_budget_in_order_key_order(waves):
    rc, rows, summ = _run(waves, budget="science=450")
    rep = [r for r in rows if r.get("repeated")]
    ordered = _science_by_order(waves)
    # take rows in order until the running sum reaches 450: the stop row overshoots by less than one item
    exp, got = [], 0
    for r in ordered:
        if got >= 450:
            break
        exp.append(r["pid"])
        got += r["corpus_tokens"]
    assert [r["pid"] for r in rep] == exp
    assert 450 <= got < 450 + max(r["corpus_tokens"] for r in ordered)
    assert sum(r["corpus_tokens"] for r in rep) == got
    assert all(r["repeated"] is True and r["domain"] == "science" for r in rep)
    assert {r["wave"] for r in rep} <= {"A", "B"}
    assert {r["wave"] for r in rep} == {r["wave"] for r in ordered if r["pid"] in exp}     # original wave kept
    assert [r["pid"] for r in rows[len(waves["C"]):]] == exp                                # repeated rows come after the fresh ones
    assert summ["budgets"]["science"]["taken_tokens"] == got and summ["budgets"]["science"]["shortfall_tokens"] == 0
    assert not any(r["domain"] in ("maths", "code") and r.get("repeated") for r in rows)    # only the named domain is repeated


def test_exclusions_txt_json_list_and_json_pids(waves, tmp_path):
    ordered = _science_by_order(waves)
    txt = tmp_path / "ex.txt"
    txt.write_text("# holdout\n" + ordered[0]["pid"] + "\n\n" + ordered[1]["pid"] + "\n")
    js = tmp_path / "ex_list.json"
    js.write_text(json.dumps([ordered[2]["pid"], "not-a-real-pid"]))
    jp = tmp_path / "ex_pids.json"
    jp.write_text(json.dumps({"pids": [ordered[3]["pid"]]}))
    rc, rows, summ = _run(waves, budget="science=450", exclude=[str(txt), str(js), str(jp)])
    rep = [r["pid"] for r in rows if r.get("repeated")]
    skipped = {o["pid"] for o in ordered[:4]}
    assert not skipped & set(rep)
    rest = [r for r in ordered if r["pid"] not in skipped]
    exp, got = [], 0
    for r in rest:
        if got >= 450:
            break
        exp.append(r["pid"]); got += r["corpus_tokens"]
    assert rep == exp                                                    # the next rows in order_key order took their place
    ex = summ["exclusions"]
    assert ex["unique_pids"] == 5 and ex["repeat_pool_rows_excluded"] == {"science": 4}
    assert ex["files"][str(txt)] == 2 and ex["files"][str(js)] == 2 and ex["files"][str(jp)] == 1
    assert ex["fresh_rows_in_exclusion"] == 0


def test_fresh_row_in_exclusion_is_kept_and_counted(waves, tmp_path):
    ex = tmp_path / "ex.txt"
    ex.write_text(waves["C"][0]["pid"] + "\n")
    _, rows, summ = _run(waves, exclude=[str(ex)])
    assert waves["C"][0]["pid"] in {r["pid"] for r in rows}
    assert summ["exclusions"]["fresh_rows_in_exclusion"] == 1


def test_file_paths_resolve_from_the_output_dir(waves, tmp_path):
    waves["out"] = str(tmp_path / "elsewhere" / "deep" / "er" / "index.jsonl")      # a different depth than the inputs
    _, rows, _ = _run(waves)
    out_dir = os.path.dirname(waves["out"])
    src = {r["pid"]: os.path.join(waves["root"], f"wave{r['wave']}", r["file"]) for r in waves["C"] + waves["A"] + waves["B"]}
    for r in rows:
        assert not os.path.isabs(r["file"])
        full = os.path.join(out_dir, r["file"])
        assert os.path.isfile(full)
        assert os.path.samefile(full, src[r["pid"]])


def test_trainer_loader_resolves_every_row(waves):
    pytest.importorskip("torch")
    from drafter_lora import data as D
    _, rows, _ = _run(waves)
    ents = D.load_entries([waves["out"]])                                # asserts pids are unique, resolves _path
    assert len(ents) == len(rows) and all(os.path.isfile(e["_path"]) for e in ents)


def test_idx_renumbered_and_count(waves):
    _, rows, summ = _run(waves)
    assert [r["idx"] for r in rows] == list(range(len(rows)))
    assert len({r["pid"] for r in rows}) == len(rows)
    assert summ["totals"]["items"] == len(rows)


def test_summary_totals_match_rows(waves):
    _, rows, summ = _run(waves, budget="science=450")
    assert summ["totals"]["items"] == len(rows)
    assert summ["totals"]["corpus_tokens"] == sum(r["corpus_tokens"] for r in rows)
    fr = [r for r in rows if not r.get("repeated")]
    rp = [r for r in rows if r.get("repeated")]
    assert summ["totals"]["fresh_items"] == len(fr) == len(waves["C"])
    assert summ["totals"]["repeated_items"] == len(rp)
    assert summ["totals"]["fresh_corpus_tokens"] == sum(r["corpus_tokens"] for r in fr)
    assert summ["totals"]["repeated_corpus_tokens"] == sum(r["corpus_tokens"] for r in rp)
    for dom, v in summ["by_domain"].items():
        sel = [r for r in rows if r["domain"] == dom]
        assert v["items"] == len(sel) and v["corpus_tokens"] == sum(r["corpus_tokens"] for r in sel)
    assert abs(sum(v["token_share"] for v in summ["by_domain"].values()) - 1) < 1e-3
    for key, v in summ["by_wave_repeated"].items():
        wave, kind = key.split("/")
        sel = [r for r in rows if r["wave"] == wave and bool(r.get("repeated")) == (kind == "repeated")]
        assert v["items"] == len(sel) and v["corpus_tokens"] == sum(r["corpus_tokens"] for r in sel)
    assert sum(c["items"] for c in summ["cells"]) == len(rows)
    assert sum(c["corpus_tokens"] for c in summ["cells"]) == summ["totals"]["corpus_tokens"]
    assert summ["budgets"]["science"]["requested_tokens"] == 450.0
    txt = B.table(summ)
    assert "science" in txt and "TOTAL" in txt


def test_no_repeat_budget_gives_fresh_only(waves):
    rc, rows, summ = _run(waves, budget=None)
    assert rc == 0 and len(rows) == len(waves["C"]) and summ["budgets"] == {}
    assert not any(r.get("repeated") for r in rows)


def test_shortfall_reported_and_strict_exit(waves, capsys):
    total = sum(r["corpus_tokens"] for r in _science_by_order(waves))
    rc, rows, summ = _run(waves, budget=f"science={total + 1000}")
    assert rc == 0
    assert sum(1 for r in rows if r.get("repeated")) == 15                     # the whole pool
    assert summ["budgets"]["science"]["shortfall_tokens"] == 1000
    assert "budget not met" in capsys.readouterr().err
    rc, _, _ = _run(waves, budget=f"science={total + 1000}", extra=["--strict"])
    assert rc == 3


def test_parse_budgets():
    assert B.parse_budgets(["science=8.0e6"]) == {"science": 8e6}
    assert B.parse_budgets(["science=1e6, chat=2e6", "if=3"]) == {"science": 1e6, "chat": 2e6, "if": 3.0}
    with pytest.raises(SystemExit):
        B.parse_budgets(["science"])
    with pytest.raises(SystemExit):
        B.parse_budgets(["science=1", "science=2"])


def test_duplicate_pid_between_fresh_and_repeat_is_an_error(waves):
    sci = _science_by_order(waves)[0]
    with open(waves["root"] / "waveC" / "index.jsonl", "a") as fh:                    # a "fresh" row that was decoded before
        fh.write(json.dumps(dict(sci, file=os.path.join("..", f"wave{sci['wave']}", sci["file"]))) + "\n")
    with pytest.raises(SystemExit, match="appears twice"):
        _run(waves, budget="science=1e9")


def test_missing_item_file_is_an_error(waves):
    r = waves["C"][2]
    os.remove(waves["root"] / "waveC" / r["file"])
    with pytest.raises(SystemExit, match="do not exist"):
        _run(waves)


def test_repeat_without_repeat_index_is_an_error(waves):
    waves["rep"] = []
    with pytest.raises(SystemExit):
        _run(waves)
