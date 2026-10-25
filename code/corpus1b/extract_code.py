"""Code-domain extractors for the 1B-token corpus (docs/CORPUS_1B_BUILD.md; registry: corpus1b/sources.py).

Every function here takes one `corpus1b.sources.SOURCES` spec dict and yields PROMPT ROWS (corpus1b/common.py
docstring): no `wave` (assigned later by common.wave_of), `key["type"]` one of "asserts" / "stdio".

  opencodeinstruct  nvidia/OpenCodeInstruct, function-level, key "asserts" -- reuses
                    draftalign.build_opencode_pool.score_row (the exact round-3 filter + template logic, factored
                    out of that file's main() without changing its behaviour: see build_opencode_pool.py's
                    `score_row` / `contamination_reference` docstrings) over the FULL 5,000,000-row snapshot,
                    not just the round-3 12,000-row prefix.
  mbpp_train        google-research-datasets/mbpp "full" train+validation+prompt, key "asserts"
  taco, apps, code_contests, codeforces
                    competitive programming, stdin/stdout, key "stdio" (corpus1b.sources.STDIO_TMPL)

Field names were verified against the live HF datasets-server before writing this file (see the task's final
report for the exact schemas); nothing here guesses a column name.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))              # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))                # code/
for _p in (HERE, os.path.join(CODE, "compat"), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import research_protocol as P                                   # noqa: E402  (sets HF offline env; unused directly
                                                                  # here but every entry point imports it first)
from common import make_pid                                     # noqa: E402
from sources import STDIO_TMPL                                  # noqa: E402

CASE_CAP = 20                                                    # at most this many stdio cases kept per problem

# "the statement says several outputs are accepted": TACO ships no checker field at all (unlike open-r1/codeforces),
# so a match here always drops the problem -- the "unless a checker exists" clause in sources.py's taco filters
# can never rescue one. Best-effort text heuristic, not a parser; documented as a known limitation.
MULTI_ANSWER_HINT = re.compile(
    r"(?i)\bif there (?:is|are) (?:multiple|more than one|several)\b|\bin case of multiple\b|"
    r"\bmultiple (?:answers|solutions|outputs|correct answers)\b|"
    r"\bany (?:valid|correct)\s+(?:answer|solution|arrangement|order|sequence|way)\b|"
    r"\bprint any\b|\byou can print any\b|\boutput any\b"
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _stdio_row(tag: str, repo: str, source_id, statement: str, cases: List[dict], gen_cap: int, bucket: str,
               checker: Optional[str], **extra_meta) -> dict:
    """One PROMPT ROW of key type "stdio"."""
    meta = {"bucket": bucket, "template": "stdio", "n_cases": len(cases)}
    meta.update(extra_meta)
    return {
        "pid": make_pid("code", tag, repo, source_id), "domain": "code", "source": repo, "source_id": source_id,
        "prompt": STDIO_TMPL.format(q=statement), "gen_cap": gen_cap,
        "key": {"type": "stdio", "cases": cases, "checker": checker},
        "meta": meta,
    }


def _stdio_cases_from_apps_io(raw: str, cap: int = CASE_CAP) -> Optional[List[dict]]:
    """Parse an APPS-shaped `input_output` JSON blob (TACO and APPS share this format) into up to `cap` stdio
    cases. None for a function-call problem (a "fn_name" key -- caller must skip the row), for a blob that does
    not parse, or when zero well-formed string input/output pairs survive ("problems without tests" -> skip)."""
    try:
        io = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(io, dict) or "fn_name" in io:
        return None
    ins, outs = io.get("inputs"), io.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list):
        return None
    cases = []
    for i, o in zip(ins, outs):
        if isinstance(i, str) and isinstance(o, str):
            cases.append({"input": i, "output": o})
            if len(cases) == cap:
                break
    return cases or None


def _code_contests_cases(row: dict, cap: int = CASE_CAP) -> List[dict]:
    """public_tests + private_tests + generated_tests merged, in that order, capped at `cap`."""
    cases = []
    for group in (row.get("public_tests"), row.get("private_tests"), row.get("generated_tests")):
        if not group:
            continue
        ins, outs = group.get("input") or [], group.get("output") or []
        for i, o in zip(ins, outs):
            cases.append({"input": i, "output": o})
            if len(cases) == cap:
                return cases
    return cases


def _codeforces_cases(row: dict, cap: int = CASE_CAP) -> List[dict]:
    tests = row.get("official_tests") or []
    return [{"input": t["input"], "output": t["output"]} for t in tests[:cap]]


def _codeforces_statement(row: dict) -> str:
    """title + description + input/output format + worked examples + note, in that order -- the fields the task
    names; `examples` become the statement's illustrative "Example" blocks, distinct from `official_tests`
    (the fuller, judged case set that becomes key["cases"])."""
    parts = []
    if row.get("title"):
        parts.append(row["title"].strip())
    if row.get("description"):
        parts.append(row["description"].strip())
    if row.get("input_format"):
        parts.append("Input\n" + row["input_format"].strip())
    if row.get("output_format"):
        parts.append("Output\n" + row["output_format"].strip())
    for i, ex in enumerate(row.get("examples") or [], 1):
        parts.append(f"Example {i}\nInput\n{ex['input']}\nOutput\n{ex['output']}")
    if row.get("note"):
        parts.append("Note\n" + row["note"].strip())
    return "\n\n".join(parts)


def _checker_lang(checker: str) -> str:
    """Best-effort language guess for a codeforces `generated_checker` source (every sample seen while writing
    this file was Python; C++ is handled defensively, documented as a known limitation: this extractor does not
    compile anything, it only records the language so a later stage knows a C++ checker needs compiling)."""
    head = checker.lstrip()[:200]
    if head.startswith("#include") or re.search(r"\bint\s+main\s*\(", checker[:2000]):
        return "cpp"
    if head.startswith(("import ", "from ", "def ", "#!")) or "def main(" in checker[:500]:
        return "python"
    return "unknown"


# ---------------------------------------------------------------------------
# nvidia/OpenCodeInstruct (function-level; key "asserts")
# ---------------------------------------------------------------------------
def _read_shard_columns(path: str, columns: List[str]) -> List[dict]:
    """One parquet shard's rows (only `columns`), as plain dicts. Top-level so it is picklable for
    ProcessPoolExecutor (see `_iter_oci_rows`)."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pylist()


def _iter_oci_rows(files: List[str], columns: List[str], workers: int) -> Iterator[dict]:
    """Every row of every shard in `files`, shard order then row order, read `workers`-wide in parallel (the
    process pool: this is the I/O-bound part of the 5,000,000-row scan). A thin, separately monkeypatchable
    layer so tests can feed opencodeinstruct() synthetic rows without touching the filesystem."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        for rows in ex.map(_read_shard_columns, files, [columns] * len(files)):
            yield from rows


def opencodeinstruct(spec: dict, tokenizer=None, max_seq: int = 1200, workers: int = 16) -> Iterator[dict]:
    """Every nvidia/OpenCodeInstruct row (all 5,000,000 rows of the 50-shard local snapshot,
    draftalign.build_opencode_pool.DATA_DIR) that passes every round-3 filter build_opencode_pool.py applies to
    its frozen 12,000-row pool: average_test_score == 1.0, not contaminated against MBPP-sanitized-test-257 /
    HumanEval-164 (build_opencode_pool.contaminated, via contamination_reference() -- refuses if its positive
    controls are not 100%), and -- when `tokenizer` is given -- prompt + reference answer <= max_seq tokens
    (round 3 used 1,200). Template choice (MBPP vs HumanEval shape, humaneval_stub fallback) is exactly
    build_opencode_pool.score_row's, the same function build_opencode_pool.main() now calls, so a row's shape
    here agrees with the frozen pool for any id the two share.

    Scan order: shard file order (train-00000-of-00050 .. train-00049-of-00050), then row order within a shard --
    NOT build_opencode_pool's sha256(id) order. That order exists only so the frozen builder can stop early at a
    fixed N with every shard sampled uniformly; this function keeps every eligible row, so an id-hash scan buys
    nothing but a slower start. Dedup ("first row in scan order wins", on the same whitespace-collapsed input
    text as build_opencode_pool.dedup_key) therefore keeps a possibly different representative shard/row than the
    frozen 12K pool for a duplicated input -- the SET of surviving unique inputs is identical either way, only
    which single row represents one can differ.

    `tokenizer=None` (the default) skips the length-fit filter entirely; pass the b32 tokenizer to apply it.
    """
    from draftalign import build_opencode_pool as bop

    data_dir = bop.DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "data", "train-*-of-*.parquet")))
    if not files:            # no round-3 --local-dir copy: the HF-cache snapshot download.py fetched ($HF_HOME)
        for snap in sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", ""), "hub",
                                                  "datasets--nvidia--OpenCodeInstruct", "snapshots", "*"))):
            got = sorted(glob.glob(os.path.join(snap, "data", "train-*-of-*.parquet")))
            if len(got) == 50:
                data_dir, files = snap, got
    assert len(files) == 50, f"expected 50 OpenCodeInstruct shards under {data_dir}/data, found {len(files)}"
    mbpp_asserts, humaneval_bodies, _cref = bop.contamination_reference()

    repo, tag, bucket, gen_cap = spec["repo"], spec["tag"], spec["bucket"], spec["gen_cap"]
    seen = set()
    for row in _iter_oci_rows(files, bop.COLUMNS, workers):
        if row["average_test_score"] != 1.0:
            continue
        res = bop.score_row(row, mbpp_asserts, humaneval_bodies, tok=tokenizer, max_seq=max_seq)
        if not res["admitted"]:
            continue
        dkey = bop.dedup_key(row["input"])
        if dkey in seen:
            continue
        seen.add(dkey)
        yield {
            "pid": make_pid("code", tag, repo, row["id"]), "domain": "code", "source": repo,
            "source_id": row["id"], "prompt": res["prompt"], "gen_cap": gen_cap,
            "key": {"type": "asserts", "tests": res["gold"]},
            "meta": {"bucket": bucket, "template": res["template"], "n_cases": res["gold"].count("\n") + 1,
                     "oci_domain": row["domain"], "generation_algorithm": row["generation_algorithm"]},
        }


# ---------------------------------------------------------------------------
# google-research-datasets/mbpp "full" (function-level; key "asserts")
# ---------------------------------------------------------------------------
def mbpp_train(spec: dict) -> Iterator[dict]:
    """Every row of google-research-datasets/mbpp "full" train + validation + prompt (474 rows: 374 + 90 + 10 --
    never `test`, the eval split every result in this repository is scored against). Same shape as the eval-set
    decode (run_desvote.MBPP_TMPL); `test_setup_code`, when non-empty, is kept as key["setup"] (a handful of rows
    need it executed before the asserts)."""
    from datasets import load_dataset
    from run_desvote import MBPP_TMPL

    repo, config = spec["repo"], spec["config"]
    splits = spec["split"] if isinstance(spec["split"], (list, tuple)) else [spec["split"]]
    assert "test" not in splits, "mbpp_train must never read the held-out eval split"
    for split in splits:
        ds = load_dataset(repo, config, split=split)
        for row in ds:
            test_list = list(row["test_list"])
            gold = "\n".join(test_list)
            key = {"type": "asserts", "tests": gold}
            if row.get("test_setup_code"):
                key["setup"] = row["test_setup_code"]
            yield {
                "pid": make_pid("code", spec["tag"], repo, row["task_id"]), "domain": "code", "source": repo,
                "source_id": row["task_id"], "prompt": MBPP_TMPL.format(text=row["text"], test_list=gold),
                "gen_cap": spec["gen_cap"], "key": key,
                "meta": {"bucket": spec["bucket"], "template": "mbpp", "n_cases": len(test_list), "split": split},
            }


# ---------------------------------------------------------------------------
# competitive programming (key "stdio")
# ---------------------------------------------------------------------------
def taco(spec: dict) -> Iterator[dict]:
    """likaixin/TACO-verified: `input_output` parses to stdio cases (a "fn_name" key -> function-call problem,
    skipped), at most CASE_CAP cases kept, problems whose statement says several outputs are accepted dropped
    (MULTI_ANSWER_HINT; TACO has no checker field to rescue them -- see the constant's docstring)."""
    from datasets import load_dataset

    repo = spec["repo"]
    ds = load_dataset(repo, split=spec.get("split", "train"))
    for row in ds:
        cases = _stdio_cases_from_apps_io(row["input_output"])
        if not cases:
            continue
        if MULTI_ANSWER_HINT.search(row["question"]):
            continue
        yield _stdio_row(spec["tag"], repo, row["id"], row["question"], cases, spec["gen_cap"], spec["bucket"],
                         None, difficulty=row.get("difficulty"))


def apps(spec: dict) -> Iterator[dict]:
    """codeparrot/apps: the repo is a loading script (arbitrary code), so its raw `train.jsonl` is read directly
    (huggingface_hub.hf_hub_download), never `datasets.load_dataset`. difficulty in {introductory, interview};
    stdin/stdout problems only (same fn_name / no-tests skip as TACO, via `_stdio_cases_from_apps_io`)."""
    from huggingface_hub import hf_hub_download

    repo = spec["repo"]
    path = hf_hub_download(repo, "train.jsonl", repo_type="dataset")
    keep_difficulty = {"introductory", "interview"}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("difficulty") not in keep_difficulty:
                continue
            cases = _stdio_cases_from_apps_io(row.get("input_output") or "")
            if not cases:
                continue
            pid_src = row["problem_id"] if "problem_id" in row else row["id"]     # train.jsonl names it `id`
            yield _stdio_row(spec["tag"], repo, pid_src, row["question"], cases, spec["gen_cap"],
                             spec["bucket"], None, difficulty=row["difficulty"])


def code_contests(spec: dict) -> Iterator[dict]:
    """deepmind/code_contests: public_tests + private_tests + generated_tests merged (<= CASE_CAP). No `task_id`
    column exists, so source_id is the split row index (stable for a pinned revision). `cf_contest_id`/`cf_index`
    are kept in meta, when present, for exact Codeforces-id decontamination against the eval sets (sources.py's
    note); `difficulty`/`source` are HF ClassLabel columns, decoded to their string names."""
    from datasets import load_dataset

    repo = spec["repo"]
    ds = load_dataset(repo, split=spec.get("split", "train"))
    diff_feat, src_feat = ds.features["difficulty"], ds.features["source"]
    for idx, row in enumerate(ds):
        cases = _code_contests_cases(row)
        if not cases:
            continue
        meta = {"difficulty": diff_feat.int2str(int(row["difficulty"])),
                "source_platform": src_feat.int2str(int(row["source"]))}
        if row.get("cf_contest_id", -1) not in (None, -1):
            meta["cf_contest_id"], meta["cf_index"] = row["cf_contest_id"], row["cf_index"]
        yield _stdio_row(spec["tag"], repo, idx, row["description"], cases, spec["gen_cap"], spec["bucket"], None,
                         **meta)


def codeforces(spec: dict) -> Iterator[dict]:
    """open-r1/codeforces, config "verifiable": `official_tests_complete` must be True; the statement is composed
    from title/description/input_format/output_format/examples/note (`_codeforces_statement`); a
    `generated_checker`, when present, is kept verbatim as key["checker"] (meta["checker_lang"] records a
    best-effort python/cpp/unknown guess -- see `_checker_lang`). ADDED beyond sources.py's own filter list: rows
    with a non-empty `interaction_format` (interactive problems) are skipped -- they cannot be posed as a single
    read-stdin/write-stdout program at all, which is what STDIO_TMPL asks the model to submit."""
    from datasets import load_dataset

    repo = spec["repo"]
    ds = load_dataset(repo, spec.get("config"), split=spec.get("split", "train"))
    for row in ds:
        if not row.get("official_tests_complete"):
            continue
        if row.get("interaction_format"):
            continue
        cases = _codeforces_cases(row)
        if not cases:
            continue
        checker = row.get("generated_checker") or None
        meta = {"rating": row.get("rating")}
        if checker:
            meta["checker_lang"] = _checker_lang(checker)
        yield _stdio_row(spec["tag"], repo, row["id"], _codeforces_statement(row), cases, spec["gen_cap"],
                         spec["bucket"], checker, **meta)
