"""CPU tests for corpus1b/checkers.py, corpus1b/eval_sets.py, the four new research_protocol.py loaders,
and the run_da_decode.py / run_research_baselines.py family-map additions (finqa/chartqa/chat, and the
confirmation that ifeval541 already falls into "ifeval" via the existing `base.startswith("if")` branch).
No network, no model, no GPU.

TODO (left for the GPU box; see the final report of the session that wrote this file): this file's logic
was validated with equivalent throwaway smoke scripts (python smoke1.py / smoke2.py /
smoke3_evalsets.py, run directly, all passing) run against this exact checkers.py / eval_sets.py /
research_protocol.py before a mid-task instruction stopped further pytest/network use on this Mac. This
file itself -- in its final pytest form below -- has NOT been executed with `pytest` past that point,
only `python3 -m py_compile`'d for syntax. Run `pytest code/corpus1b/tests -q` (and `pytest code/tests -q`
for the pre-existing suite) on the GPU box before trusting it further; if something here does not match
the smoke-script behaviour it is a transcription slip in THIS file, not a re-verified logic change.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))                    # code/corpus1b/tests
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))               # code/
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from corpus1b import checkers as C                                    # noqa: E402
from corpus1b import eval_sets as E                                   # noqa: E402
from corpus1b import common                                           # noqa: E402
import research_protocol as P                                         # noqa: E402

EOS = common.EOS_IDS[0]


def mkids(n=1, eos=EOS):
    return [1] * n + [eos]


@pytest.fixture()
def sbx(tmp_path):
    return str(tmp_path / "sandbox")


# ---------------------------------------------------------------------------
# checkers.py: the common gate
# ---------------------------------------------------------------------------
def test_gate_no_eos_uses_full_capped_window_for_loop(sbx):
    rep = [7, 8, 9] * 20                                              # no EOS anywhere in this window
    r = C.check({"type": "chat"}, rep, lambda ids: "unused", sandbox_dir=sbx, gen_cap=len(rep))
    assert r == {"ok": False, "reason": "no_eos", "loop": True, "eos": False, "n_gen": len(rep)}


def test_gate_eos_found_n_gen_includes_eos_and_loop_is_pre_eos_only():
    # n_gen = eos_idx + 1 (matches common.py's own GENERATION RECORD convention); loop is computed on the
    # PRE-eos slice only, so a loop that only appears at/after the eos index must not trip it.
    ids = [5, 5, 5, 5, EOS] + [5] * 40      # a real loop would need >= LOOP_SPAN+LOOP_REPEATS-1 repeats;
    g = C._gate(ids, lambda pre: "x", gen_cap=len(ids))
    assert g["eos"] is True and g["n_gen"] == 5 and g["text"] == "x"


# ---------------------------------------------------------------------------
# math (verified against the vendored MATHEvaluator(v2) -- NOT assumed equivalences; v2 normalises LaTeX
# spelling but does not evaluate fractions numerically, so \frac{1}{2} != 0.5 under v2. See module
# docstring of checkers.py.)
# ---------------------------------------------------------------------------
def test_math_dfrac_normalises_to_frac_pass(sbx):
    key = {"type": "math", "answer": r"\frac{1}{2}"}
    r = C.check(key, mkids(), lambda ids: r"We compute and get \boxed{\dfrac{1}{2}}.", sandbox_dir=sbx, gen_cap=64)
    assert r["ok"] and r["reason"] == "" and r["eos"] and not r["loop"]


def test_math_wrong_answer_fails(sbx):
    key = {"type": "math", "answer": "43"}
    r = C.check(key, mkids(), lambda ids: r"\boxed{42}", sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


def test_math_v2_does_not_equate_fraction_and_decimal():
    """Documents the verified v2 quirk directly against the vendored evaluator, independent of checkers.py,
    so a future change to the vendored copy is caught here too."""
    ev = C.MATHEvaluator(version="v2")
    assert ev.is_equiv(r"\frac{1}{2}", "0.5") is False
    assert ev.is_equiv(r"\frac{1}{2}", r"\frac{1}{2}") is True


def test_no_eos_gate_fires_before_any_scoring(sbx):
    key = {"type": "math", "answer": "1"}
    r = C.check(key, [1, 2, 3, 4], lambda ids: "unused", sandbox_dir=sbx, gen_cap=64)
    assert r["ok"] is False and r["reason"] == "no_eos" and r["eos"] is False


# ---------------------------------------------------------------------------
# free: numeric tolerance + units + boolean
# ---------------------------------------------------------------------------
def test_free_numeric_with_unit_and_times_ten(sbx):
    key = {"type": "free", "answer": "6.03e23", "answer_type": "Float"}
    text = r"roughly \boxed{6.02 \times 10^{23} \text{ atoms}}"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


def test_free_numeric_outside_tolerance_fails(sbx):
    key = {"type": "free", "answer": "100", "answer_type": "Float"}
    r = C.check(key, mkids(), lambda ids: r"\boxed{110}", sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


def test_free_boolean_yes_matches_true(sbx):
    key = {"type": "free", "answer": "true", "answer_type": "Boolean"}
    r = C.check(key, mkids(), lambda ids: r"\boxed{Yes}", sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


# ---------------------------------------------------------------------------
# mc: up to J, LaTeX-stripped boxed letter
# ---------------------------------------------------------------------------
def test_mc_boxed_text_e_of_ten(sbx):
    key = {"type": "mc", "answer": "E", "n_options": 10}
    r = C.check(key, mkids(), lambda ids: r"The answer is \boxed{\text{E}}", sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


def test_mc_letter_outside_n_options_not_extracted(sbx):
    key = {"type": "mc", "answer": "A", "n_options": 4}                 # only A-D admitted
    r = C.check(key, mkids(), lambda ids: r"\boxed{\text{E}}", sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "no_letter"


# ---------------------------------------------------------------------------
# asserts: pass / fail / timeout, inside a tmp sandbox
# ---------------------------------------------------------------------------
def test_asserts_pass(sbx):
    key = {"type": "asserts", "tests": "assert add(1,2) == 3\nassert add(-1,1) == 0"}
    text = "```python\ndef add(a,b):\n    return a+b\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


def test_asserts_wrong_answer(sbx):
    key = {"type": "asserts", "tests": "assert add(1,2) == 3"}
    text = "```python\ndef add(a,b):\n    return a-b\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


@pytest.mark.timeout_sensitive
def test_asserts_timeout(sbx):
    key = {"type": "asserts", "tests": "assert True"}
    text = "```python\nwhile True:\n    pass\n```"
    t0 = time.time()
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "timeout"
    assert time.time() - t0 < C.ASSERT_TIMEOUT + 5           # bounded, not hung


def test_humaneval_pass(sbx):
    key = {"type": "humaneval",
          "test": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(5) == 6",
          "entry_point": "inc"}
    text = "```python\ndef inc(x):\n    return x + 1\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


# ---------------------------------------------------------------------------
# stdio: pass / fail / float tolerance / timeout / checker protocol / no_checker_runtime
# ---------------------------------------------------------------------------
def test_stdio_pass_and_float_tolerance(sbx):
    key = {"type": "stdio", "cases": [{"input": "2\n3\n", "output": "5\n3.141590001\n"}]}
    text = "```python\na=int(input()); b=int(input())\nprint(a+b)\nprint(3.14159)\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


def test_stdio_float_outside_tolerance_fails(sbx):
    key = {"type": "stdio", "cases": [{"input": "2\n3\n", "output": "5\n3.14160\n"}]}   # diff 1e-5 > 1e-6
    text = "```python\na=int(input()); b=int(input())\nprint(a+b)\nprint(3.14159)\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


def test_stdio_wrong_answer(sbx):
    key = {"type": "stdio", "cases": [{"input": "2\n3\n", "output": "5\n"}]}
    text = "```python\na=int(input()); b=int(input())\nprint(a-b)\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


def test_stdio_no_code(sbx):
    key = {"type": "stdio", "cases": [{"input": "", "output": "5\n"}]}
    text = "I think the answer is probably five but am not fully sure."
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "no_code"


@pytest.mark.timeout_sensitive
def test_stdio_timeout(sbx):
    key = {"type": "stdio", "cases": [{"input": "", "output": "x\n"}]}
    text = "```python\nwhile True:\n    pass\n```"
    t0 = time.time()
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "timeout"
    assert time.time() - t0 < C.STDIO_PER_CASE_TIMEOUT + 5


def test_stdio_python_checker_protocol(sbx):
    checker_src = (
        "import sys\n"
        "o = open(sys.argv[2]).read().strip()\n"
        "a = open(sys.argv[3]).read().strip()\n"
        "print('1' if o == a else '0')\n"
    )
    key = {"type": "stdio", "cases": [{"input": "4\n", "output": "8\n"}],
          "checker": {"lang": "python", "source": checker_src}}
    text = "```python\nprint(int(input()) * 2)\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r
    key_bad = {"type": "stdio", "cases": [{"input": "4\n", "output": "9\n"}],
              "checker": {"lang": "python", "source": checker_src}}
    r = C.check(key_bad, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "wrong_answer"


def test_stdio_no_checker_runtime_when_gpp_missing(sbx, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "g++" else shutil.which(name))
    key = {"type": "stdio", "cases": [{"input": "4\n", "output": "8\n"}],
          "checker": {"lang": "cpp", "source": "int main(){return 0;}"}}
    text = "```python\nprint(int(input()) * 2)\n```"
    r = C.check(key, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "no_checker_runtime"


# ---------------------------------------------------------------------------
# ifeval: score_ifeval via a simple constraint that needs no NLTK data
# (punctuation:no_comma is pure regex -- see code/tests/test_if_scorer.py's own CONTROLS list; word/
# sentence-count constraints additionally need NLTK punkt_tab, not exercised here -- see checkers.py's
# module docstring / this session's final report for that limitation)
# ---------------------------------------------------------------------------
def test_ifeval_simple_constraint_pass_and_fail(sbx):
    gold = json.dumps({"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]})
    key = {"type": "ifeval", "gold": gold}
    r = C.check(key, mkids(), lambda ids: "I like apples and pears.", sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r
    r = C.check(key, mkids(), lambda ids: "I like apples, and pears.", sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"], r


# ---------------------------------------------------------------------------
# chat: loop / no_eos / not_english
# ---------------------------------------------------------------------------
def test_chat_loop_rejected(sbx):
    ids = mkids(n=40)
    r = C.check({"type": "chat"}, ids, lambda pre: ("hours " * 40).strip(), sandbox_dir=sbx, gen_cap=200)
    assert not r["ok"] and r["reason"] == "loop" and r["loop"]


def test_chat_not_english_rejected(sbx):
    text = ("éèàùç " * 10) + "bonjour a tous comment allez vous aujourdhui"
    r = C.check({"type": "chat"}, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert not r["ok"] and r["reason"] == "not_english"


def test_chat_english_passes(sbx):
    text = "Hello, this is a perfectly ordinary English sentence used for testing purposes today."
    r = C.check({"type": "chat"}, mkids(), lambda ids: text, sandbox_dir=sbx, gen_cap=64)
    assert r["ok"], r


# ---------------------------------------------------------------------------
# check_many: order preservation, cheap items in-process + code items pooled
# ---------------------------------------------------------------------------
def test_check_many_preserves_order(sbx):
    rep = [7, 8, 9] * 20
    items = [
        {"key": {"type": "math", "answer": "7"}, "gen_ids": mkids(), "tok_decode": lambda ids: r"\boxed{7}",
        "sandbox_dir": sbx, "gen_cap": 64},
        {"key": {"type": "asserts", "tests": "assert f() == 1"}, "gen_ids": mkids(),
        "tok_decode": lambda ids: "```python\ndef f():\n    return 1\n```", "sandbox_dir": sbx, "gen_cap": 64},
        {"key": {"type": "math", "answer": "8"}, "gen_ids": mkids(), "tok_decode": lambda ids: r"\boxed{7}",
        "sandbox_dir": sbx, "gen_cap": 64},
        {"key": {"type": "chat"}, "gen_ids": rep, "tok_decode": lambda ids: "unused", "sandbox_dir": sbx,
        "gen_cap": len(rep)},
    ]
    res = C.check_many(items, workers=2)
    assert res[0]["ok"] and res[0]["reason"] == ""
    assert res[1]["ok"]
    assert not res[2]["ok"] and res[2]["reason"] == "wrong_answer"
    assert not res[3]["ok"] and res[3]["reason"] == "no_eos"


# ===========================================================================
# eval_sets.py builders on synthetic rows shaped like the verified schemas
# (HF datasets-server, 2026-09-28 -- see eval_sets.py's module docstring)
# ===========================================================================
def test_ifeval_rows_strips_none_kwargs_and_orders_by_file():
    ds_rows = [
        {"key": 1000, "prompt": "Write about dogs.", "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{"num_highlights": None, "relation": None}]},
        {"key": 1001, "prompt": "Write about cats in 3 sections.",
        "instruction_id_list": ["detectable_format:multiple_sections"],
        "kwargs": [{"num_sections": 3, "section_spliter": "Section", "relation": None}]},
    ]
    rows = E._ifeval_rows(ds_rows)
    assert [r["idx"] for r in rows] == [0, 1]
    assert json.loads(rows[0]["gold"]) == {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    assert json.loads(rows[1]["gold"])["kwargs"][0] == {"num_sections": 3, "section_spliter": "Section"}


def test_finqa_rows_table_markdown_and_boxed_tail():
    ds_rows = [
        {"id": "AAA-1", "pre_text": ["Intro line."], "post_text": ["Outro line."],
        "question": "What is the net change?", "answer": "94",
        "table": [["", "amount"], ["2014", "$5735"], ["2015", "$5829"]]},
        {"id": "ZZZ-2", "pre_text": [], "post_text": [], "question": "What percent increase?",
        "answer": "14%", "table": []},
    ]
    rows = E._finqa_rows(ds_rows)
    assert len(rows) == 2 and [r["idx"] for r in rows] == [0, 1]
    assert all("Please reason step by step" in r["prompt"] for r in rows)
    assert any("| 2014 | $5735 |" in r["prompt"] for r in rows)
    assert {r["gold"] for r in rows} == {"94", "14%"}


def test_chartqa_rows_human_only_and_joined_on_imgname():
    table_rows = [{"imgname": "img1.png", "table_csv": "Country,Value\nA,1\nB,2\n"},
                 {"imgname": "img2.png", "table_csv": "X,Y\n1,2\n"}]
    qa_rows = [
        {"imgname": "img1.png", "query": "How many?", "label": "14", "type": "human"},
        {"imgname": "img1.png", "query": "auto question", "label": "9", "type": "augmented"},
        {"imgname": "img2.png", "query": "What is X?", "label": "1", "type": "human"},
        {"imgname": "missing.png", "query": "orphan", "label": "0", "type": "human"},   # no table -> dropped
    ]
    rows = E._chartqa_rows(table_rows, qa_rows)
    assert len(rows) == 2                                       # augmented and table-less rows excluded
    assert [r["idx"] for r in rows] == [0, 1]
    assert any("| Country | Value |" in r["prompt"] for r in rows)


def test_chat_rows_gold_always_empty_and_ordered_by_prompt_id():
    ds_rows = [{"prompt_id": "b" * 10, "prompt": "Second"}, {"prompt_id": "a" * 10, "prompt": "First"}]
    rows = E._chat_rows(ds_rows)
    assert len(rows) == 2 and all(r["gold"] == "" for r in rows)
    assert [r["idx"] for r in rows] == [0, 1]


def test_freeze_writes_jsonl_and_matching_sha256_sidecar(tmp_path):
    rows = [{"idx": 0, "prompt": "p0", "gold": "g0", "split": "test"},
           {"idx": 1, "prompt": "p1", "gold": "g1", "split": "test"}]
    summary = E._freeze("dummyset", rows, str(tmp_path))
    assert summary["n"] == 2
    assert os.path.exists(summary["path"]) and os.path.exists(summary["path"] + ".sha256")
    assert summary["sha256"] == common.sha256_file(summary["path"])
    sidecar = open(summary["path"] + ".sha256").read()
    assert sidecar == f"{summary['sha256']}  dummyset.jsonl\n"


# ===========================================================================
# research_protocol.py loaders: sha256-checked read-back, wired through items_for_split
# ===========================================================================
def test_research_protocol_c1b_loader_reads_frozen_set(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "C1B_EVAL_DIR", str(tmp_path))
    rows = [{"idx": 0, "prompt": "p0", "gold": "g0", "split": "test"},
           {"idx": 1, "prompt": "p1", "gold": "g1", "split": "test"}]
    E._freeze("finqa500", rows, str(tmp_path))
    items = P.items_for_split("finqa500")
    assert [it["idx"] for it in items] == [0, 1]
    assert items[0]["prompt"] == "p0" and items[0]["gold"] == "g0"
    assert P.GEN_LEN["finqa500"] == 1024


def test_research_protocol_c1b_loader_rejects_bad_sha256(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "C1B_EVAL_DIR", str(tmp_path))
    rows = [{"idx": 0, "prompt": "p0", "gold": "g0", "split": "test"}]
    summary = E._freeze("chat800", rows, str(tmp_path))
    with open(summary["path"] + ".sha256", "w") as fh:
        fh.write("0" * 64 + "  chat800.jsonl\n")
    with pytest.raises(AssertionError):
        P.items_for_split("chat800")


def test_c1b_eval_sets_registered_and_gen_len_present():
    for name in ("ifeval541", "finqa500", "chartqa300", "chat800"):
        assert name in P.C1B_EVAL_SETS
        assert P.GEN_LEN[name] == 1024


# ===========================================================================
# family maps in both runners: finqa/chartqa/chat map correctly, ifeval541 confirmed via "if" prefix.
# Parsed as text (not imported): both runner modules pull in torch/sdar_harness/CUDA-adjacent imports at
# module scope that do not belong in a CPU-only test process.
# ===========================================================================
RUNNERS = {
    "run_da_decode.py": os.path.join(CODE, "draftalign", "run_da_decode.py"),
    "run_research_baselines.py": os.path.join(CODE, "run_research_baselines.py"),
}


@pytest.mark.parametrize("runner_path", RUNNERS.values(), ids=RUNNERS.keys())
def test_family_map_covers_finqa_chartqa_chat_and_if(runner_path):
    src = open(runner_path, encoding="utf-8").read()
    assert '"finqa" if' in src and 'startswith("finqa")' in src
    assert '"chartqa" if' in src and 'startswith("chartqa")' in src
    assert '"chat" if' in src and 'startswith("chat")' in src
    assert '"ifeval" if' in src and 'startswith("if")' in src   # covers if_pool AND ifeval541


def test_analyze_s1b_has_finqa_chartqa_chat_branches():
    path = os.path.join(CODE, "analyze_s1b.py")
    src = open(path, encoding="utf-8").read()
    assert 'ds == "finqa"' in src and 'ds == "chartqa"' in src and 'ds == "chat"' in src
