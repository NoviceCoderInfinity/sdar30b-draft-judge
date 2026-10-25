#!/usr/bin/env python
"""Freeze the round-3 code training pool: code/configs/opencode_pool.json (+ .sha256).

Spec: docs/TRAINING_RECIPE_2026-09-09.md sections 3.2-3.5. Source: nvidia/OpenCodeInstruct @ REVISION, the 50
parquet shards (5,000,000 rows) of a local snapshot (--data-dir); each shard's sha256 is checked against the Hugging
Face download metadata before anything is read. research_protocol.opencode_pool_items() loads the result.

Scan order -- deterministic, no seed: all rows sorted by sha256(id) ascending (ids are unique; asserted) and walked
from the smallest hash, so every shard is sampled uniformly (the pool is not shard 0). The walk stops at the
N_SELECT-th row that passes every filter; `selection.rows_scanned` records how far it went.

Filters, in order (recipe 3.3):
  1. average_test_score == 1.0
  2. contamination -- normalised exact match (all whitespace removed, lowercased):
       each `assert` line of the row's unit_tests   vs  the MBPP-sanitized-test (257) assert lines
       each function body of the reference output  vs  the HumanEval-164 canonical_solution bodies
     A body is taken with and without its docstring (from the first statement, and from after every string-literal
     statement). The positive controls must flag 257/257 MBPP and 164/164 HumanEval items or the build refuses;
     matching the WHOLE output instead flags 0/164, which is why bodies are extracted.
  3. length fit -- len(encode_prompt(b32 tokenizer, the templated prompt)) + len(reference output tokens) <= MAX_SEQ
  4. dedup on the input with whitespace runs collapsed, lowercased; the first row in scan order wins

Prompt shape (recipe 3.5): sha256("template:" + id) mod 100 < 40 -> HUMANEVAL_TMPL, else MBPP_TMPL (both imported from
run_desvote). The HumanEval stub is the reference's top-level imports + the signature of the one top-level function
the unit tests call + a docstring made from the input (code-fence lines dropped); a row whose stub cannot be derived
cleanly gets the MBPP shape instead and is counted by reason.
gold, both shapes (recipe 3.2 fix 2): the unit tests joined by newlines, each entry stripped of the newlines it is
stored with (so the test list renders like MBPP-257's), scored by analyze_s1b.per_item_correct("mbpp").

Usage (CPU only):
  HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_opencode_pool.py
"""
from __future__ import annotations

import argparse, ast, glob, hashlib, inspect, json, os, re, statistics, sys, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
for p in (os.path.join(HERE, "compat"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")                 # the parquet passes fork after the tokenizer loads
import research_protocol as P                                            # noqa: E402  (sets HF offline env)
from research_protocol import safe_out                                   # noqa: E402

import numpy as np                                                       # noqa: E402

REPO, REVISION = "nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
DATA_DIR = "/dev/shm/dlm/data/OpenCodeInstruct"                          # hf download --local-dir snapshot
OUT = os.path.join(HERE, "configs", "opencode_pool.json")
N_SELECT, MAX_SEQ, HUMANEVAL_PCT = 12000, 1200, 40
GEN_BUDGET = 512                                                         # GEN_LEN of mbpp / opencode_pool
CHUNK = 100_000                                                          # scan ranks fetched per parquet pass
COLUMNS = ["id", "input", "output", "domain", "generation_algorithm", "unit_tests", "average_test_score"]
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)                         # the reference output's first fenced block
FENCE_LINE = re.compile(r"^\s*```")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def nows(s: str) -> str:
    """Contamination key: every whitespace character removed, lowercased."""
    return re.sub(r"\s+", "", s).lower()


def dedup_key(s: str) -> str:
    """Dedup key: whitespace runs collapsed to one space, lowercased."""
    return re.sub(r"\s+", " ", s).strip().lower()


def solution_code(output: str) -> str:
    m = FENCE.search(output)
    return m.group(1) if m else output


def parse(src: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                  # invalid-escape SyntaxWarnings in the data
        try:
            return ast.parse(src)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None


def _lines(src: str):
    return [l + "\n" for l in src.split("\n")]                           # ast numbers lines on "\n" only


def _is_str_stmt(s) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def function_bodies(code: str):
    """Every function's body text, from its first statement and from after each string-literal statement (a docstring
    wherever it sits: HumanEval/115 puts `import math` before its docstring)."""
    mod = parse(code)
    if mod is None:
        return []
    lines, out = _lines(code), []
    for fn in ast.walk(mod):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for lo in [fn.body[0].lineno] + [s.end_lineno + 1 for s in fn.body if _is_str_stmt(s)]:
                text = "".join(lines[lo - 1:fn.end_lineno])
                if text.strip():
                    out.append(text)
    return out


def contaminated(tests, code: str, mbpp_asserts: set, humaneval_bodies: set):
    """-> (an assert line hits MBPP-257, a function body reproduces HumanEval-164)"""
    m = any(nows(l) in mbpp_asserts for t in tests for l in t.split("\n") if l.strip().startswith("assert"))
    return m, any(nows(b) in humaneval_bodies for b in function_bodies(code))


def humaneval_stub(text: str, code: str, tests):
    """The HumanEval-shape prompt of one row -> (stub, None), or (None, why) when it cannot be derived cleanly."""
    mod, tmod = parse(code), parse("\n".join(tests))
    if mod is None or tmod is None:
        return None, "does_not_parse"
    defined = {}
    for n in mod.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                defined.update({x.id: n for x in ast.walk(t) if isinstance(x, ast.Name)})
    used = {x.id for x in ast.walk(tmod) if isinstance(x, ast.Name)}
    called = sorted({x.func.id for x in ast.walk(tmod) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                     and isinstance(defined.get(x.func.id), ast.FunctionDef)})
    if len(called) != 1:
        if called:
            return None, "tests_call_several_functions"
        return None, ("tests_call_a_class" if any(isinstance(defined.get(u), ast.ClassDef) for u in used)
                      else "tests_call_no_defined_function")
    fn = defined[called[0]]
    if (used & set(defined)) - {fn.name}:
        return None, "tests_use_other_definitions"
    if fn.decorator_list or fn.body[0].lineno == fn.lineno:
        return None, "decorated_or_one_line_def"
    head = _lines(code)[fn.lineno - 1:fn.body[0].lineno - 1]           # the signature, however many lines it spans
    while head and (not head[-1].strip() or head[-1].lstrip().startswith("#")):
        head.pop()
    doc = "\n".join(l.rstrip() for l in text.strip().split("\n") if not FENCE_LINE.match(l)).strip()
    if not doc or '"""' in doc or "```" in doc or "\\" in doc:
        return None, "input_unfit_for_docstring"
    imports = [ast.get_source_segment(code, n) for n in mod.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    stub = (("\n".join(imports) + "\n\n\n") if imports else "") + "".join(head).rstrip() + '\n    """\n' \
        + "\n".join(("    " + l) if l.strip() else "" for l in doc.split("\n")) + '\n    """\n'
    chk = parse(stub + "    pass\n")
    got = [n for n in (chk.body if chk else []) if isinstance(n, ast.FunctionDef)]
    if len(got) != 1 or got[0].name != fn.name or ast.dump(got[0].args) != ast.dump(fn.args) \
            or ast.get_docstring(got[0]) != inspect.cleandoc(doc):
        return None, "stub_does_not_round_trip"
    return stub, None


def cached_revision(ds):
    """The dataset revision encoded in its HF datasets cache path (…/0.0.0/<sha>/…), or None."""
    m = re.search(r"/([0-9a-f]{40})/", ds.cache_files[0]["filename"]) if ds.cache_files else None
    return m.group(1) if m else None


def contamination_reference():
    """(mbpp_asserts, humaneval_bodies, meta) for `contaminated()` / `score_row()`: MBPP-sanitized-test (257)
    assert lines and HumanEval-164 canonical_solution bodies, both normalised (`nows`); the MBPP test revision is
    checked against the frozen code/configs/prompts.json record. `meta["positive_controls"]` must show both sets
    100% flagged: every caller -- this file's main() and corpus1b/extract_code.opencodeinstruct -- refuses to
    proceed otherwise (an assert below), since a failing positive control means `contaminated()` itself is
    broken, not that the data is clean.
    """
    from datasets import load_dataset
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    humaneval = load_dataset("openai_humaneval", split="test")
    mbpp_rev = json.load(open(P.PROMPTS_PATH))["test_sets"]["mbpp"]["revision"]
    assert cached_revision(mbpp) == mbpp_rev, f"MBPP test revision {cached_revision(mbpp)} != frozen {mbpp_rev}"
    mbpp_lines = [l for r in mbpp for t in r["test_list"] for l in t.split("\n") if l.strip().startswith("assert")]
    mbpp_asserts = {nows(l) for l in mbpp_lines}
    humaneval_bodies = {nows(r["canonical_solution"]) for r in humaneval}
    pc_mbpp = sum(contaminated(["\n" + t + "\n" for t in r["test_list"]], "", mbpp_asserts, humaneval_bodies)[0]
                  for r in mbpp)
    he_outputs = ["```python\n" + r["prompt"] + r["canonical_solution"] + "```" for r in humaneval]
    pc_he = sum(contaminated([], solution_code(o), mbpp_asserts, humaneval_bodies)[1] for o in he_outputs)
    pc_he_whole = sum(nows(o) in humaneval_bodies or nows(solution_code(o)) in humaneval_bodies for o in he_outputs)
    assert pc_mbpp == len(mbpp) and pc_he == len(humaneval), f"positive controls failed: {pc_mbpp}, {pc_he}"
    meta = {
        "mbpp_revision": cached_revision(mbpp), "mbpp_n_items": len(mbpp), "mbpp_n_assert_lines": len(mbpp_lines),
        "humaneval_revision": cached_revision(humaneval), "humaneval_n_items": len(humaneval),
        "positive_controls": {"mbpp_items_flagged": f"{pc_mbpp}/{len(mbpp)}",
                              "humaneval_items_flagged": f"{pc_he}/{len(humaneval)}",
                              "humaneval_items_flagged_matching_the_whole_output": f"{pc_he_whole}/{len(humaneval)}"},
    }
    return mbpp_asserts, humaneval_bodies, meta


def score_row(row: dict, mbpp_asserts: set, humaneval_bodies: set, tok=None, max_seq: int = MAX_SEQ) -> dict:
    """Apply every round-3 OpenCodeInstruct filter (recipe 3.3-3.5: contamination, MBPP/HumanEval template choice
    + stub derivation, length fit) to one raw row -- `average_test_score == 1.0` is the caller's job (both callers
    pre-filter on it before fetching the row's other columns). Shared by this file's main() (the frozen 12,000-row
    pool) and corpus1b/extract_code.opencodeinstruct (every eligible row of the full 5,000,000-row scan), so a
    row's template/prompt/gold agree between the two for any id they share.

    Returns a dict with everything the two callers need for their own bookkeeping:
      hit_mbpp, hit_humaneval  bool        contaminated() outcome (both False == not contaminated)
      template_drawn           str|None    the hash draw ("mbpp"/"humaneval"), before any stub fallback; None when
                                            contaminated (the row is rejected before a template is drawn)
      template                 str|None    the FINAL shape: a humaneval draw that could not derive a stub falls
                                            back to "mbpp"; None when contaminated
      fallback_reason          str|None    humaneval_stub()'s reason, set only when template_drawn == "humaneval"
                                            and it fell back
      prompt, gold              str|None   the templated prompt and the newline-joined unit tests; None when
                                            contaminated
      n_prompt, n_answer        int|None   token counts; both None when `tok` is None (length fit skipped then)
      too_long                  bool        n_prompt + n_answer > max_seq (always False when tok is None)
      admitted                  bool        not contaminated and (tok is None or not too_long)
    """
    from run_desvote import MBPP_TMPL, HUMANEVAL_TMPL
    tests = [t.strip() for t in json.loads(row["unit_tests"])]
    code = solution_code(row["output"])
    hit_m, hit_h = contaminated(tests, code, mbpp_asserts, humaneval_bodies)
    out = {"hit_mbpp": hit_m, "hit_humaneval": hit_h, "template_drawn": None, "template": None,
           "fallback_reason": None, "prompt": None, "gold": None, "n_prompt": None, "n_answer": None,
           "too_long": False, "admitted": False}
    if hit_m or hit_h:
        return out
    gold = "\n".join(tests)
    template = "humaneval" if int(sha("template:" + row["id"]), 16) % 100 < HUMANEVAL_PCT else "mbpp"
    out["template_drawn"] = template
    stub = None
    if template == "humaneval":
        stub, why = humaneval_stub(row["input"], code, tests)
        if stub is None:
            template = "mbpp"
            out["fallback_reason"] = why
    prompt = (HUMANEVAL_TMPL.format(prompt=stub) if template == "humaneval"
              else MBPP_TMPL.format(text=row["input"], test_list=gold))
    out.update(template=template, prompt=prompt, gold=gold)
    if tok is not None:
        n_prompt = int(P.encode_prompt(tok, prompt).shape[1])
        n_answer = len(tok(row["output"], add_special_tokens=False)["input_ids"])
        out.update(n_prompt=n_prompt, n_answer=n_answer)
        if n_prompt + n_answer > max_seq:
            out["too_long"] = True
            return out
    out["admitted"] = True
    return out


# ---------------------------------------------------------------------------
# parquet workers (pyarrow only)
# ---------------------------------------------------------------------------
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_ids(path: str):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["id", "average_test_score"])
    return (b"".join(hashlib.sha256(i.encode()).digest() for i in t.column("id").to_pylist()),
            np.asarray(t.column("average_test_score").to_numpy(), float))


def _read_rows(job):
    path, rows = job
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=COLUMNS).take(rows).to_pylist()


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR, help="local snapshot of nvidia/OpenCodeInstruct @ REVISION")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-select", type=int, default=N_SELECT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    out_path = safe_out(args.out)
    from transformers import AutoTokenizer

    # -- the snapshot: 50 shards at REVISION, each file's sha256 == its LFS oid in the download metadata ----------
    files = sorted(glob.glob(os.path.join(args.data_dir, "data", "train-*-of-*.parquet")))
    assert len(files) == 50, f"{len(files)} shards under {args.data_dir}/data"
    with ProcessPoolExecutor(args.workers) as ex:
        file_sha = list(ex.map(_file_sha256, files))
        hashed = list(ex.map(_hash_ids, files))
    shards = []
    for f, d, (_, sc) in zip(files, file_sha, hashed):
        rel = os.path.relpath(f, args.data_dir)
        commit, etag = open(os.path.join(args.data_dir, ".cache", "huggingface", "download", rel + ".metadata")).read().split()[:2]
        assert commit == REVISION and etag == d, f"{rel}: {commit} sha256 {d} is not {REVISION} {etag}"
        shards.append({"file": rel, "rows": len(sc), "sha256": d})
    print(f"[source] {len(files)} shards verified at {REVISION}", flush=True)

    # -- scan order: sha256(id) ascending over every row ----------------------------------------------------------
    offsets = np.cumsum([0] + [len(sc) for _, sc in hashed])
    words = np.frombuffer(b"".join(h for h, _ in hashed), dtype=">u8").reshape(-1, 4)
    scores = np.concatenate([sc for _, sc in hashed])
    order = np.lexsort(words[:, ::-1].T)                                 # word 0 is the primary key
    srt = words[order]
    assert np.any(srt[1:] != srt[:-1], axis=1).all(), "duplicate sha256(id): ids are not unique"
    del srt
    print(f"[order] {len(order):,} rows, {int((scores == 1.0).sum()):,} with average_test_score == 1.0", flush=True)

    # -- contamination references + positive controls ---------------------------------------------------------------
    mbpp_asserts, humaneval_bodies, cref = contamination_reference()

    m = P.MODELS["b32"]
    tok = AutoTokenizer.from_pretrained(m["repo"], trust_remote_code=True, local_files_only=True,
                                        revision=m["revision"], cache_dir=m["cache_dir"])

    # -- the walk -------------------------------------------------------------------------------------------------
    counts = Counter({k: 0 for k in ("rows_scanned", "pass_score_1", "contaminated_mbpp", "contaminated_humaneval",
                                     "contaminated_both", "pass_contamination", "too_long", "pass_length",
                                     "duplicate_input", "selected")})
    assigned, fallback, items, seen, rank = Counter(), Counter(), [], set(), 0
    while len(items) < args.n_select and rank < len(order):
        chunk = order[rank:rank + CHUNK]
        want = {}
        for flat in chunk[scores[chunk] == 1.0]:
            s = int(np.searchsorted(offsets, flat, side="right")) - 1
            want.setdefault(s, []).append(int(flat - offsets[s]))
        jobs = [(s, sorted(rows)) for s, rows in sorted(want.items())]
        with ProcessPoolExecutor(args.workers) as ex:
            got = ex.map(_read_rows, [(files[s], rows) for s, rows in jobs])
            fetched = {(s, r): d for (s, rows), res in zip(jobs, got) for r, d in zip(rows, res)}
        for flat in chunk:
            rank += 1
            counts["rows_scanned"] += 1
            if scores[flat] != 1.0:
                continue
            counts["pass_score_1"] += 1
            s = int(np.searchsorted(offsets, flat, side="right")) - 1
            r = int(flat - offsets[s])
            row = fetched[(s, r)]
            assert hashlib.sha256(row["id"].encode()).digest() == words[flat].tobytes(), (s, r)
            res = score_row(row, mbpp_asserts, humaneval_bodies, tok=tok, max_seq=MAX_SEQ)
            counts["contaminated_mbpp"] += res["hit_mbpp"]
            counts["contaminated_humaneval"] += res["hit_humaneval"]
            counts["contaminated_both"] += res["hit_mbpp"] and res["hit_humaneval"]
            if res["hit_mbpp"] or res["hit_humaneval"]:
                continue
            counts["pass_contamination"] += 1
            assigned[res["template_drawn"]] += 1
            if res["fallback_reason"] is not None:
                fallback[res["fallback_reason"]] += 1
            if res["too_long"]:
                counts["too_long"] += 1
                continue
            counts["pass_length"] += 1
            key = dedup_key(row["input"])
            if key in seen:
                counts["duplicate_input"] += 1
                continue
            seen.add(key)
            counts["selected"] += 1
            items.append({"idx": len(items), "id": row["id"], "template": res["template"], "prompt": res["prompt"],
                          "gold": res["gold"], "domain": row["domain"],
                          "generation_algorithm": row["generation_algorithm"],
                          "text_sha256_16": sha(row["input"])[:16], "shard": s, "row": r,
                          "prompt_tokens": res["n_prompt"], "answer_tokens": res["n_answer"]})
            if len(items) == args.n_select:
                break
        print(f"[scan] {rank:,} rows scanned, {len(items):,} selected", flush=True)
    assert len(items) == args.n_select, f"only {len(items)} rows pass every filter"

    pt, at = [it["prompt_tokens"] for it in items], [it["answer_tokens"] for it in items]
    spec = {
        "source": {"repo": REPO, "revision": REVISION, "split": "train", "n_rows": int(len(order)),
                   "n_average_test_score_1": int((scores == 1.0).sum()), "shards": shards},
        "tokenizer": {"model_key": "b32", "repo": m["repo"], "revision": m["revision"]},
        "selection": {
            "n_select": args.n_select, "max_seq": MAX_SEQ, "rows_scanned": counts["rows_scanned"],
            "order": "every row sorted by sha256(id) ascending (ids unique), walked from the smallest hash; "
                     "stops at the n_select-th row that passes every filter",
            "filters": ["average_test_score == 1.0",
                        "contamination: see decontamination",
                        "length fit: len(encode_prompt(b32 tokenizer, templated prompt)) "
                        "+ len(tokenizer(output, add_special_tokens=False)) <= max_seq",
                        "dedup: re.sub(r'\\s+', ' ', input).strip().lower(); the first row in scan order wins"],
            "template": f"int(sha256('template:' + id), 16) % 100 < {HUMANEVAL_PCT} -> humaneval "
                        "(run_desvote.HUMANEVAL_TMPL, stub = imports + signature of the function the tests call + "
                        "input as docstring), else mbpp (run_desvote.MBPP_TMPL, text = input, test_list = gold); "
                        "a humaneval row whose stub cannot be derived gets the mbpp shape",
            "gold": "'\\n'.join(t.strip() for t in json.loads(unit_tests)), scored by analyze_s1b.per_item_correct('mbpp')",
        },
        "counts": dict(counts),
        "templates": {"assigned": dict(assigned), "humaneval_fallback_to_mbpp": dict(fallback),
                      "selected": dict(Counter(it["template"] for it in items))},
        "lengths": {"prompt_tokens_median": statistics.median(pt), "answer_tokens_median": statistics.median(at),
                    "total_tokens_median": statistics.median([a + b for a, b in zip(pt, at)]),
                    f"answer_tokens_over_{GEN_BUDGET}": sum(a > GEN_BUDGET for a in at)},
        "decontamination": {
            "method": "normalised exact match (every whitespace character removed, lowercased) of each unit-test "
                      "assert line against the MBPP test assert lines, and of each function body of the reference "
                      "output (with and without its docstring) against the HumanEval canonical_solution bodies",
            "mbpp_test": {"repo": "google-research-datasets/mbpp", "revision": cref["mbpp_revision"],
                          "config": "sanitized", "split": "test", "n_items": cref["mbpp_n_items"],
                          "n_assert_lines": cref["mbpp_n_assert_lines"]},
            "humaneval_test": {"repo": "openai_humaneval", "revision": cref["humaneval_revision"], "split": "test",
                               "n_items": cref["humaneval_n_items"]},
            "rows_checked": counts["pass_score_1"],
            "positive_controls": cref["positive_controls"],
            "note": "Positive controls feed each MBPP item's asserts as unit tests and each HumanEval prompt + "
                    "canonical_solution as a fenced reference output; both must be 100%, so the hit counts are "
                    "genuine. Whole-output matching cannot flag a body, hence the per-function bodies.",
        },
        "items": items,
    }
    raw = (json.dumps(spec, indent=2) + "\n").encode()
    with open(out_path, "wb") as fh:
        fh.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with open(out_path + ".sha256", "w") as fh:
        fh.write(f"{digest}  {os.path.basename(out_path)}\n")
    print(json.dumps({k: v for k, v in spec.items() if k not in ("source", "items")}, indent=2))
    print(f"wrote {out_path} ({len(raw) / 2**20:.1f} MiB) sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
