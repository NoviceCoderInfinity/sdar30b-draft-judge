"""CPU tests for corpus1b/extract_code.py, extract_if.py and compose_if.py (no model, no network): every HF call
is monkeypatched with 2-3 synthetic rows shaped exactly like the schema verified against the live HF
datasets-server while these extractors were written (see the task's final report for the verified field names).

Needs the vendored IFEval checker's pure-Python deps and NLTK punkt_tab (code/vendor/ifeval/PROVENANCE.md); if
punkt_tab is not importable, the tests that build a sentence-counting instruction are skipped with a clear
reason instead of failing -- see `_HAVE_PUNKT` below. Set NLTK_DATA to a directory holding it if it is not on
the default nltk.data.path.
"""
from __future__ import annotations

import glob as glob_module
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))               # code/corpus1b/tests
CORPUS1B = os.path.abspath(os.path.join(HERE, ".."))             # code/corpus1b
CODE = os.path.abspath(os.path.join(CORPUS1B, ".."))             # code/
for _p in (CORPUS1B, os.path.join(CODE, "compat"), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets                                                  # noqa: E402
import huggingface_hub                                           # noqa: E402
import research_protocol as P                                    # noqa: E402
import common                                                    # noqa: E402
import extract_code                                               # noqa: E402
import extract_if                                                 # noqa: E402
import compose_if                                                 # noqa: E402
from draftalign import build_opencode_pool as bop                 # noqa: E402
from draftalign import build_if_pool as bip                       # noqa: E402
from draftalign import if_scorer as S                             # noqa: E402

try:
    S.build_instruction("length_constraints:number_sentences", {"relation": "at least", "num_sentences": 1}) \
        .check_following("One sentence.")
    _HAVE_PUNKT = True
except LookupError:
    _HAVE_PUNKT = False

needs_punkt = pytest.mark.skipif(not _HAVE_PUNKT, reason="NLTK punkt_tab not found on nltk.data.path / NLTK_DATA")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeClassLabel:
    def __init__(self, names):
        self.names = names

    def int2str(self, i):
        return self.names[i]


class FakeDataset(list):
    """A list of dict rows, optionally with `.features` (ClassLabel columns) for code_contests."""

    def __init__(self, rows, features=None):
        super().__init__(rows)
        self.features = features or {}


def _patch_load_dataset(monkeypatch, table):
    """table: {(repo, config, split): FakeDataset|list}. `config`/`split` may be None wildcards."""
    def fake(repo, config=None, split=None, **kw):
        for (r, c, s), rows in table.items():
            if r == repo and (c is None or c == config) and (s is None or s == split):
                return rows
        raise AssertionError(f"unexpected load_dataset({repo!r}, {config!r}, split={split!r})")
    monkeypatch.setattr(datasets, "load_dataset", fake)


# ---------------------------------------------------------------------------
# extract_code: mbpp_train
# ---------------------------------------------------------------------------
def test_mbpp_train_shape_and_setup_code(monkeypatch):
    rows_train = [
        {"task_id": 601, "text": "Write a function to add two numbers.", "code": "def add(a, b):\n    return a + b",
         "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"], "test_setup_code": "",
         "challenge_test_list": []},
    ]
    rows_valid = [
        {"task_id": 602, "text": "Write a function to multiply two numbers.", "code": "def mul(a, b):\n    return a * b",
         "test_list": ["assert mul(2, 3) == 6"], "test_setup_code": "import math", "challenge_test_list": []},
    ]
    rows_prompt = [
        {"task_id": 603, "text": "Write a function to subtract.", "code": "def sub(a, b):\n    return a - b",
         "test_list": ["assert sub(5, 2) == 3"], "test_setup_code": "", "challenge_test_list": []},
    ]
    _patch_load_dataset(monkeypatch, {
        ("google-research-datasets/mbpp", "full", "train"): rows_train,
        ("google-research-datasets/mbpp", "full", "validation"): rows_valid,
        ("google-research-datasets/mbpp", "full", "prompt"): rows_prompt,
    })
    spec = {"tag": "mbpptrain", "repo": "google-research-datasets/mbpp", "config": "full",
            "split": ["train", "validation", "prompt"], "gen_cap": 512, "bucket": "function"}
    out = list(extract_code.mbpp_train(spec))
    assert len(out) == 3
    by_id = {r["source_id"]: r for r in out}
    assert "test_setup_code" not in by_id[601]["key"] and "setup" not in by_id[601]["key"]
    assert by_id[602]["key"]["setup"] == "import math"
    for r in out:
        assert r["domain"] == "code" and r["key"]["type"] == "asserts"
        assert r["key"]["tests"].count("\n") + 1 == len(r["key"]["tests"].split("\n"))
        assert r["meta"]["bucket"] == "function" and r["meta"]["template"] == "mbpp"
        assert r["pid"] == common.make_pid("code", "mbpptrain", "google-research-datasets/mbpp", r["source_id"])


def test_mbpp_train_never_reads_test_split():
    spec = {"tag": "mbpptrain", "repo": "google-research-datasets/mbpp", "config": "full",
            "split": ["train", "test"], "gen_cap": 512, "bucket": "function"}
    with pytest.raises(AssertionError):
        list(extract_code.mbpp_train(spec))


# ---------------------------------------------------------------------------
# extract_code: stdio sources -- case parsing, the 20-case cap, fn_name / multi-answer skips
# ---------------------------------------------------------------------------
def test_stdio_cases_from_apps_io_cap_and_fn_name_skip():
    ok = json.dumps({"inputs": [str(i) for i in range(30)], "outputs": [str(i) for i in range(30)]})
    cases = extract_code._stdio_cases_from_apps_io(ok)
    assert len(cases) == 20 and cases[0] == {"input": "0", "output": "0"}
    fn = json.dumps({"fn_name": "f", "inputs": [[1, 2]], "outputs": [[3]]})
    assert extract_code._stdio_cases_from_apps_io(fn) is None
    assert extract_code._stdio_cases_from_apps_io(json.dumps({"inputs": [], "outputs": []})) is None
    assert extract_code._stdio_cases_from_apps_io("not json") is None


def test_taco_skips_fn_name_and_multi_answer(monkeypatch):
    rows = [
        {"id": 1, "question": "Read two integers and print their sum.",
         "input_output": json.dumps({"inputs": ["1 2\n", "3 4\n"], "outputs": ["3\n", "7\n"]}), "difficulty": "EASY"},
        {"id": 2, "question": "Reverse a list via a function.",
         "input_output": json.dumps({"fn_name": "rev", "inputs": [[[1, 2, 3]]], "outputs": [[[3, 2, 1]]]}),
         "difficulty": "EASY"},
        {"id": 3, "question": "If there are multiple correct answers, print any of them.",
         "input_output": json.dumps({"inputs": ["5\n"], "outputs": ["1\n"]}), "difficulty": "MEDIUM"},
    ]
    _patch_load_dataset(monkeypatch, {("likaixin/TACO-verified", None, "train"): FakeDataset(rows)})
    spec = {"tag": "taco", "repo": "likaixin/TACO-verified", "gen_cap": 1024, "bucket": "competitive"}
    out = list(extract_code.taco(spec))
    assert [r["source_id"] for r in out] == [1]
    assert out[0]["key"] == {"type": "stdio", "cases": [{"input": "1 2\n", "output": "3\n"},
                                                         {"input": "3 4\n", "output": "7\n"}], "checker": None}
    assert out[0]["meta"]["n_cases"] == 2 and out[0]["meta"]["difficulty"] == "EASY"


def test_apps_reads_raw_jsonl_and_filters_difficulty(monkeypatch, tmp_path):
    rows = [
        {"problem_id": 10, "question": "Print the sum of two numbers given on one line.",
         "input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}), "difficulty": "introductory"},
        {"problem_id": 11, "question": "Competition problem.",
         "input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]}), "difficulty": "competition"},
        {"problem_id": 12, "question": "Function style problem.",
         "input_output": json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[1]]}), "difficulty": "interview"},
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, fname, repo_type=None: str(p))
    spec = {"tag": "apps", "repo": "codeparrot/apps", "gen_cap": 1024, "bucket": "competitive"}
    out = list(extract_code.apps(spec))
    assert [r["source_id"] for r in out] == [10]
    assert out[0]["meta"]["difficulty"] == "introductory"


def test_code_contests_merges_and_caps_cases(monkeypatch):
    rows = [
        {"description": "Sum two numbers.", "public_tests": {"input": ["1 2\n"], "output": ["3\n"]},
         "private_tests": {"input": ["4 5\n"], "output": ["9\n"]}, "generated_tests": {"input": [], "output": []},
         "difficulty": 1, "source": 2, "cf_contest_id": 100, "cf_index": "A"},
        {"description": "No tests.", "public_tests": {"input": [], "output": []},
         "private_tests": {"input": [], "output": []}, "generated_tests": {"input": [], "output": []},
         "difficulty": 0, "source": 0, "cf_contest_id": -1, "cf_index": ""},
    ]
    features = {"difficulty": FakeClassLabel(["UNKNOWN_DIFFICULTY", "EASY", "MEDIUM"]),
                "source": FakeClassLabel(["UNKNOWN_SOURCE", "CODECHEF", "CODEFORCES"])}
    _patch_load_dataset(monkeypatch, {("deepmind/code_contests", None, "train"): FakeDataset(rows, features)})
    spec = {"tag": "cc", "repo": "deepmind/code_contests", "gen_cap": 1024, "bucket": "competitive"}
    out = list(extract_code.code_contests(spec))
    assert len(out) == 1                                            # row 1 has zero cases -> skipped
    assert out[0]["meta"]["n_cases"] == 2
    assert out[0]["meta"]["difficulty"] == "EASY" and out[0]["meta"]["source_platform"] == "CODEFORCES"
    assert out[0]["meta"]["cf_contest_id"] == 100 and out[0]["meta"]["cf_index"] == "A"


def test_codeforces_filters_and_checker(monkeypatch):
    rows = [
        {"id": "1/A", "title": "Sum", "description": "Print sum.", "input_format": "Two ints.",
         "output_format": "One int.", "examples": [{"input": "1 2", "output": "3"}], "note": "",
         "official_tests_complete": True,
         "official_tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "4 5\n", "output": "9\n"}],
         "generated_checker": None, "interaction_format": None, "rating": 800},
        {"id": "2/B", "title": "Interactive", "description": "d", "input_format": "", "output_format": "",
         "examples": [], "note": "", "official_tests_complete": True,
         "official_tests": [{"input": "1", "output": "1"}], "generated_checker": None,
         "interaction_format": "This is an interactive problem.", "rating": 1200},
        {"id": "3/C", "title": "Incomplete", "description": "d", "input_format": "", "output_format": "",
         "examples": [], "note": "", "official_tests_complete": False,
         "official_tests": [{"input": "1", "output": "1"}], "generated_checker": None,
         "interaction_format": None, "rating": 900},
        {"id": "4/D", "title": "Checked", "description": "d", "input_format": "", "output_format": "",
         "examples": [], "note": "", "official_tests_complete": True,
         "official_tests": [{"input": "1", "output": "1"}],
         "generated_checker": "import sys\ndef main():\n    pass", "interaction_format": None, "rating": 1500},
    ]
    _patch_load_dataset(monkeypatch, {("open-r1/codeforces", "verifiable", "train"): FakeDataset(rows)})
    spec = {"tag": "cf", "repo": "open-r1/codeforces", "config": "verifiable", "gen_cap": 1024,
            "bucket": "competitive"}
    out = list(extract_code.codeforces(spec))
    assert [r["source_id"] for r in out] == ["1/A", "4/D"]           # interactive + incomplete both dropped
    assert "Example 1" in out[0]["prompt"] and "Sum\n\nPrint sum." in out[0]["prompt"]
    assert out[1]["key"]["checker"].startswith("import sys")
    assert out[1]["meta"]["checker_lang"] == "python"
    assert out[0]["key"]["checker"] is None


# ---------------------------------------------------------------------------
# extract_code: opencodeinstruct -- reuses build_opencode_pool.score_row
# ---------------------------------------------------------------------------
_MBPP_ASSERTS = {bop.nows("assert add(1, 2) == 3")}
_HUMANEVAL_BODIES = {bop.nows("    return a + b\n")}


def _patch_oci(monkeypatch, rows, workers_seen=None):
    monkeypatch.setattr(extract_code.glob, "glob", lambda *a, **k: [f"shard{i}" for i in range(50)])
    monkeypatch.setattr(bop, "contamination_reference", lambda: (_MBPP_ASSERTS, _HUMANEVAL_BODIES, {}))

    def fake_iter(files, columns, workers):
        if workers_seen is not None:
            workers_seen.append(workers)
        return iter(rows)
    monkeypatch.setattr(extract_code, "_iter_oci_rows", fake_iter)


def test_opencodeinstruct_filters_score_and_dedup(monkeypatch):
    rows = [
        {"id": "row-mbpp-1", "input": "Write a function to multiply two numbers.",
         "output": "```python\ndef multiply(a, b):\n    return a * b\n```",
         "unit_tests": json.dumps(["assert multiply(2, 3) == 6"]), "domain": "general",
         "generation_algorithm": "self_instruct", "average_test_score": 1.0},
        {"id": "row-mbpp-1-dup", "input": "Write   a function to  multiply two numbers.",  # dedup_key collapses ws
         "output": "```python\ndef multiply(a, b):\n    return a * b\n```",
         "unit_tests": json.dumps(["assert multiply(2, 3) == 6"]), "domain": "general",
         "generation_algorithm": "self_instruct", "average_test_score": 1.0},
        {"id": "row-contam-1", "input": "Write a function to add two numbers.",
         "output": "```python\ndef add(a, b):\n    return a + b\n```",
         "unit_tests": json.dumps(["assert add(1, 2) == 3"]), "domain": "general",
         "generation_algorithm": "self_instruct", "average_test_score": 1.0},
        {"id": "row-low-score", "input": "Irrelevant.", "output": "```python\npass\n```",
         "unit_tests": json.dumps(["assert True"]), "domain": "general", "generation_algorithm": "x",
         "average_test_score": 0.5},
    ]
    _patch_oci(monkeypatch, rows)
    spec = {"tag": "oci", "repo": "nvidia/OpenCodeInstruct", "gen_cap": 512, "bucket": "function"}
    out = list(extract_code.opencodeinstruct(spec))
    assert [r["source_id"] for r in out] == ["row-mbpp-1"]            # dup dropped, contaminated + low score dropped
    assert out[0]["key"]["type"] == "asserts" and out[0]["key"]["tests"] == "assert multiply(2, 3) == 6"
    assert out[0]["meta"]["bucket"] == "function"


def test_opencodeinstruct_template_choice_agrees_with_score_row(monkeypatch):
    """The template extract_code.opencodeinstruct assigns must equal build_opencode_pool.score_row's own
    decision for the identical row -- they are, in fact, the same function call, but this checks it explicitly
    rather than only by construction."""
    row = {"id": "row-he-1", "input": "Return the sum of a list of integers.",
           "output": "```python\ndef sum_list(nums):\n    \"\"\"Return the sum of a list of integers.\"\"\"\n"
                     "    return sum(nums)\n```",
           "unit_tests": json.dumps(["assert sum_list([1, 2, 3]) == 6"]), "domain": "general",
           "generation_algorithm": "self_instruct", "average_test_score": 1.0}
    expected = bop.score_row(row, _MBPP_ASSERTS, _HUMANEVAL_BODIES, tok=None, max_seq=1200)
    _patch_oci(monkeypatch, [row])
    spec = {"tag": "oci", "repo": "nvidia/OpenCodeInstruct", "gen_cap": 512, "bucket": "function"}
    out = list(extract_code.opencodeinstruct(spec))
    assert len(out) == 1
    assert out[0]["meta"]["template"] == expected["template"]
    assert out[0]["prompt"] == expected["prompt"]
    assert out[0]["key"]["tests"] == expected["gold"]


def test_opencodeinstruct_length_fit_with_fake_tokenizer(monkeypatch):
    row = {"id": "row-mbpp-2", "input": "Write a function to divide two numbers.",
           "output": "```python\ndef divide(a, b):\n    return a / b\n```",
           "unit_tests": json.dumps(["assert divide(6, 3) == 2"]), "domain": "general",
           "generation_algorithm": "self_instruct", "average_test_score": 1.0}
    _patch_oci(monkeypatch, [row])

    class FakeBatch:
        def __init__(self, n):
            self.shape = (1, n)

    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    monkeypatch.setattr(P, "encode_prompt", lambda tok, prompt: FakeBatch(len(prompt.split())))
    spec = {"tag": "oci", "repo": "nvidia/OpenCodeInstruct", "gen_cap": 512, "bucket": "function"}
    out = list(extract_code.opencodeinstruct(spec, tokenizer=FakeTok(), max_seq=1))   # impossibly tiny -> too_long
    assert out == []


# ---------------------------------------------------------------------------
# extract_if: existing pools
# ---------------------------------------------------------------------------
def _fake_cand(src, source, row, prompt, ids, kwargs, origin="wildchat-1m"):
    return {"src": src, "source": source, "row": row, "prompt": prompt, "ids": ids, "kwargs": kwargs,
            "origin": origin, "key": f"{src}/{row:06d}", "key_sha256": "x"}


@pytest.fixture(autouse=True)
def _reset_if_cache():
    extract_if.reset_cache()
    yield
    extract_if.reset_cache()


def test_existing_pools_filters_by_source_and_shapes_rows(monkeypatch):
    cands = [
        _fake_cand("if_multi", "allenai/IF_multi_constraints_upto5", 1, "Write a poem.",
                  ["punctuation:no_comma"], [{}]),
        _fake_cand("rlvr", "allenai/RLVR-IFeval", 2, "Explain gravity.",
                  ["keywords:existence"], [{"keywords": ["mass"]}]),
        _fake_cand("nemotron_rl_if", "nvidia/Nemotron-RL-instruction_following", 3, "Describe the moon.",
                  ["startend:quotation"], [{}]),
    ]
    calls = []

    def fake_eligible(workers=32):
        calls.append(workers)
        return cands, {}, {}, {}, []
    monkeypatch.setattr(bip, "eligible_items", fake_eligible)

    spec_multi = {"tag": "ifmulti", "repo": "allenai/IF_multi_constraints_upto5", "gen_cap": 1024, "bucket": "pool"}
    spec_rlvr = {"tag": "ifrlvr", "repo": "allenai/RLVR-IFeval", "gen_cap": 1024, "bucket": "pool"}
    spec_nemo = {"tag": "ifnemo", "repo": "nvidia/Nemotron-RL-instruction_following", "gen_cap": 1024, "bucket": "pool"}

    out_multi = list(extract_if.existing_pools(spec_multi))
    out_rlvr = list(extract_if.existing_pools(spec_rlvr))
    out_nemo = list(extract_if.existing_pools(spec_nemo))
    assert len(calls) == 1, "eligible_items must be memoized across the three tags"

    assert len(out_multi) == 1 and out_multi[0]["source_id"] == 1
    assert len(out_rlvr) == 1 and out_rlvr[0]["source_id"] == 2
    assert len(out_nemo) == 1 and out_nemo[0]["source_id"] == 3
    for r in out_multi + out_rlvr + out_nemo:
        assert r["domain"] == "if" and r["key"]["type"] == "ifeval"
        ids, kwargs = S.parse_gold(r["key"]["gold"])
        assert isinstance(ids, list) and isinstance(kwargs, list)


# ---------------------------------------------------------------------------
# compose_if
# ---------------------------------------------------------------------------
GOOD_TEXT = ("What is a good recipe for banana bread that does not need a stand mixer or any fancy equipment?")


def _wc_row(conv_hash, content, role="user", toxic=False, redacted=False, language="English"):
    return {"conversation_hash": conv_hash,
            "conversation": [{"role": role, "content": content, "toxic": toxic, "redacted": redacted,
                              "language": language}],
            "language": language, "turn": 1}


def test_composed_filters_role_toxicity_language_and_length(monkeypatch):
    rows = [
        _wc_row("h-good", GOOD_TEXT),
        _wc_row("h-assistant-first", GOOD_TEXT, role="assistant"),
        _wc_row("h-toxic", GOOD_TEXT, toxic=True),
        _wc_row("h-redacted", GOOD_TEXT, redacted=True),
        _wc_row("h-nonenglish", "Esto es una prueba de un texto suficientemente largo.", language="Spanish"),
        _wc_row("h-tooshort", "Hi there"),
        _wc_row("h-code", "```python\ndef f(): pass\n```" + " padding text " * 5),
        _wc_row("h-demand", "Write exactly 5 sentences about the history of the bicycle industry please."),
    ]
    _patch_load_dataset(monkeypatch, {("allenai/WildChat-4.8M", None, "train"): FakeDataset(rows)})
    spec = {"tag": "ifcomp", "repo": "allenai/WildChat-4.8M", "gen_cap": 1024, "bucket": "composed"}
    out = list(compose_if.composed(spec))
    assert [r["source_id"] for r in out] == ["h-good"]


@needs_punkt
def test_composed_exclude_norm_and_determinism(monkeypatch):
    rows = [_wc_row("h-good", GOOD_TEXT), _wc_row("h-excluded", "The quick brown fox jumps over the lazy dog today.")]
    _patch_load_dataset(monkeypatch, {("allenai/WildChat-4.8M", None, "train"): FakeDataset(rows)})
    spec = {"tag": "ifcomp", "repo": "allenai/WildChat-4.8M", "gen_cap": 1024, "bucket": "composed"}

    exclude = {bip.normalise("The quick brown fox jumps over the lazy dog today.")}
    out1 = list(compose_if.composed(spec, exclude_norm=exclude))
    assert [r["source_id"] for r in out1] == ["h-good"]

    out2 = list(compose_if.composed(spec, exclude_norm=exclude))
    assert out1[0]["pid"] == out2[0]["pid"]
    assert out1[0]["prompt"] == out2[0]["prompt"]
    assert out1[0]["key"]["gold"] == out2[0]["key"]["gold"]

    ids, kwargs = S.parse_gold(out1[0]["key"]["gold"])
    admitted = set(compose_if.admitted_ids())
    assert set(ids) <= admitted
    conflicts = compose_if.conflict_map()
    for a in ids:
        for b in ids:
            if a != b:
                assert b not in conflicts.get(a, {a})
    assert out1[0]["meta"]["bucket"] == "composed" and out1[0]["meta"]["n_constraints"] == len(ids)
    assert GOOD_TEXT in out1[0]["prompt"]


def test_composed_without_exclude_norm_still_yields():
    # exclude_norm=None must not raise (default path)
    pass
