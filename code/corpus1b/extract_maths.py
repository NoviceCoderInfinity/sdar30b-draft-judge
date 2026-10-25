"""Maths extractors for the 1B corpus (docs/CORPUS_1B_BUILD.md): one function per sources.py tag.

Each function takes a source spec dict (sources.source(tag)) and yields PROMPT ROWs (common.py schema,
without "wave" -- build_prompt_lists.py assigns that). Nothing here downloads: every `datasets.load_dataset`
/ `huggingface_hub.snapshot_download` call is offline (research_protocol sets HF_HUB_OFFLINE=1 on import),
so `download.py` must have already cached the source. A missing cache entry raises the ordinary
`datasets`/`huggingface_hub` "not found locally" error -- there is no separate check here.

Field names below were verified against the HF datasets-server API on 2026-09-28 (not assumed from
dataset cards); see the build report for the exact commands. Two corrections to sources.py surfaced by
that check (both fixed there, kept here for the record):
  - nvidia/OpenScienceReasoning-2 has 307,862 rows, not "1.6M" (sources.py notes said 1.6M).
  - zwhe99/DeepMath-103K's question field is `question`, not `problem` (sources.py never named it).

Verified per-source facts that shape the code below:
  gsm8k            openai/gsm8k, config=main: fields `question`, `answer` ("...#### N").
  hendrycks_math   EleutherAI/hendrycks_math: 7 real configs (algebra, counting_and_probability, geometry,
                   intermediate_algebra, number_theory, prealgebra, precalculus); sources.py's config
                   "ALL_7_SUBJECTS" is a sentinel this module expands, not a real HF config name. Fields
                   `problem`, `level` ("Level N"), `type` (the subject name; not used -- the config name
                   already gives the subject), `solution`.
  openmathinstruct2 nvidia/OpenMathInstruct-2, config=default. split="train" is the FULL 14M-row split
                   (distinct from train_1M/2M/5M, which are separate parquet shards this module never
                   reads). Fields `problem`, `generated_solution` (unused), `expected_answer`,
                   `problem_source` (seen: "augmented_gsm8k"; sources.py's other 3 values are the dataset
                   card's documented set). Streamed with pyarrow (14M rows), not `datasets.load_dataset`.
  numina           AI-MO/NuminaMath-1.5, config=default, split=train (only split). Fields `problem`,
                   `solution` (unused), `answer`, `problem_type` (kept as meta "subject"), `question_type`,
                   `problem_is_valid`, `solution_is_valid`, `source`, `synthetic` (unused).
  deepmath         zwhe99/DeepMath-103K, config=default, split=train (only split). Fields `question`,
                   `final_answer`, `difficulty` (float, e.g. 4.5), `topic`.
  openmathreasoning nvidia/OpenMathReasoning, config=default, split=cot (also has tir/genselect/
                   additional_problems, unused). Fields `expected_answer`, `problem_type`
                   ("has_answer_extracted" is the filter value), `problem_source`, `problem`.

Dataset loading is factored into small `_load_*` module-level functions so tests can monkeypatch them
instead of touching the network or the real HF cache (see code/corpus1b/tests/test_data_builders.py).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                             # noqa: E402
import sources                                             # noqa: E402
import research_protocol as P                              # noqa: E402  (sets HF_*_OFFLINE=1 on import)

DOMAIN = "maths"
HENDRYCKS_MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                           "number_theory", "prealgebra", "precalculus")
# sources.py tag -> the OpenMathInstruct-2 problem_source values that tag draws from
_OMI2_PROBLEM_SOURCES = {"omi2gsm": {"gsm8k", "augmented_gsm8k"}, "omi2math": {"math", "augmented_math"}}
_OMI2_COLUMNS = ["problem", "expected_answer", "problem_source"]
_OMI2_SHARD_GLOB = os.path.join("data", "train-*-of-00032.parquet")   # the full 14M-row "train" split only


# ---------------------------------------------------------------------------
# mockable loaders (tests monkeypatch these, never `datasets`/`huggingface_hub` directly)
# ---------------------------------------------------------------------------
def _load_hf_split(repo: str, config: Optional[str], split: str):
    """`datasets.load_dataset(repo, config, split=split)`, offline. The one seam every non-OMI2 source uses."""
    from datasets import load_dataset
    return load_dataset(repo, config, split=split)


def _omi2_shard_files(repo: str) -> List[str]:
    """Local paths of the 32 "train" parquet shards of `repo` (already cached by download.py)."""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[_OMI2_SHARD_GLOB])
    files = sorted(glob.glob(os.path.join(local_dir, _OMI2_SHARD_GLOB)))
    assert files, f"no {repo} train-*-of-00032 shards under {local_dir} (run download.py --only omi2gsm,omi2math)"
    return files


def _omi2_rows(repo: str) -> Iterator[Dict]:
    """Stream {problem, expected_answer, problem_source} dicts over every row of the 14M-row train split."""
    import pyarrow.parquet as pq
    for f in _omi2_shard_files(repo):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=_OMI2_COLUMNS, batch_size=50_000):
            cols = batch.to_pydict()
            for problem, ans, psrc in zip(cols["problem"], cols["expected_answer"], cols["problem_source"]):
                yield {"problem": problem, "expected_answer": ans, "problem_source": psrc}


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _row(tag: str, repo: str, source_id, prompt: str, gen_cap: int, key: dict, meta: dict) -> dict:
    meta = dict(meta)
    assert "bucket" in meta, meta
    return {"pid": common.make_pid(DOMAIN, tag, repo, source_id), "domain": DOMAIN, "source": repo,
            "source_id": str(source_id), "prompt": prompt, "gen_cap": gen_cap, "key": key, "meta": meta}


def _level_n(level: Optional[str]) -> Optional[int]:
    m = re.match(r"Level\s*(\d)", level or "")
    return int(m.group(1)) if m else None


def _level_bucket(level: Optional[str]) -> str:
    """Level 1-3 and unknown -> math_easy; Level 4-5 -> math_hard (docs/CORPUS_1B_PLAN_2026-09-28.md)."""
    return "math_hard" if _level_n(level) in (4, 5) else "math_easy"


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------
def gsm8k(spec: dict) -> Iterator[dict]:
    """openai/gsm8k train, minus the 200 frozen calib/valid items. gold = the '#### N' tail, commas stripped."""
    frozen = {int(it["split_index"]) for it in P._frozen_set("gsm8k_train")["items"]}
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for i, r in enumerate(ds):
        if i in frozen:
            continue
        ans = r["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        yield _row(spec["tag"], spec["repo"], i, sources.GSM8K_TMPL.format(q=r["question"]), spec["gen_cap"],
                   {"type": "math", "answer": ans}, {"bucket": spec["bucket"]})


def hendrycks_math(spec: dict) -> Iterator[dict]:
    """EleutherAI/hendrycks_math train, all 7 subject configs. gold = last \\boxed{} of `solution`."""
    subjects = HENDRYCKS_MATH_SUBJECTS if spec["config"] == "ALL_7_SUBJECTS" else (spec["config"],)
    for subj in subjects:
        ds = _load_hf_split(spec["repo"], subj, "train")
        for i, r in enumerate(ds):
            gold = P.last_boxed(r["solution"])
            if not gold:
                continue
            yield _row(spec["tag"], spec["repo"], f"{subj}:{i}", sources.GSM8K_TMPL.format(q=r["problem"]),
                       spec["gen_cap"], {"type": "math", "answer": gold},
                       {"bucket": _level_bucket(r.get("level")), "level": r.get("level"), "subject": subj})


def openmathinstruct2(spec: dict) -> Iterator[dict]:
    """nvidia/OpenMathInstruct-2 train (14M rows, pyarrow-streamed), filtered by problem_source, deduped on
    the normalised problem text (many generated solutions share one problem). Serves omi2gsm and omi2math."""
    wanted = _OMI2_PROBLEM_SOURCES[spec["tag"]]
    seen: Set[str] = set()
    for r in _omi2_rows(spec["repo"]):
        if r["problem_source"] not in wanted:
            continue
        norm = common.normalize_text(r["problem"])
        if norm in seen:
            continue
        seen.add(norm)
        sid = common.sha256_hex(r["problem"])[:16]
        yield _row(spec["tag"], spec["repo"], sid, sources.GSM8K_TMPL.format(q=r["problem"]), spec["gen_cap"],
                   {"type": "math", "answer": str(r["expected_answer"])},
                   {"bucket": spec["bucket"], "problem_source": r["problem_source"]})


def numina(spec: dict) -> Iterator[dict]:
    """AI-MO/NuminaMath-1.5 train. Serves both `numina` (source not olympiad -> math_hard) and `numinaolym`
    (source in NUMINA_OLYMPIAD_SOURCES -> olympiad); each call yields only the rows for its own tag."""
    want_olympiad = spec["tag"] == "numinaolym"
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for i, r in enumerate(ds):
        is_olympiad_src = r["source"] in sources.NUMINA_OLYMPIAD_SOURCES
        if is_olympiad_src != want_olympiad:
            continue
        ans = (r.get("answer") or "").strip()
        if ans.lower() in ("proof", "notfound", "") or r.get("question_type") == "proof":
            continue
        if str(r.get("problem_is_valid", "")).strip().lower() == "no":
            continue
        if str(r.get("solution_is_valid", "")).strip().lower() == "no":
            continue
        bucket = "olympiad" if is_olympiad_src else "math_hard"
        yield _row(spec["tag"], spec["repo"], i, sources.GSM8K_TMPL.format(q=r["problem"]), spec["gen_cap"],
                   {"type": "math", "answer": ans},
                   {"bucket": bucket, "subject": r.get("problem_type"), "source": r["source"]})


def deepmath(spec: dict) -> Iterator[dict]:
    """zwhe99/DeepMath-103K train. difficulty >= 8 -> olympiad, else math_hard. gold = `final_answer`."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for i, r in enumerate(ds):
        difficulty = r.get("difficulty")
        bucket = "olympiad" if (difficulty is not None and difficulty >= 8) else "math_hard"
        yield _row(spec["tag"], spec["repo"], i, sources.GSM8K_TMPL.format(q=r["question"]), spec["gen_cap"],
                   {"type": "math", "answer": str(r["final_answer"])},
                   {"bucket": bucket, "difficulty": difficulty, "topic": r.get("topic")})


def openmathreasoning(spec: dict) -> Iterator[dict]:
    """nvidia/OpenMathReasoning split=cot. problem_type == has_answer_extracted, deduped on `problem`."""
    seen: Set[str] = set()
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for i, r in enumerate(ds):
        if r.get("problem_type") != "has_answer_extracted":
            continue
        norm = common.normalize_text(r["problem"])
        if norm in seen:
            continue
        seen.add(norm)
        yield _row(spec["tag"], spec["repo"], i, sources.GSM8K_TMPL.format(q=r["problem"]), spec["gen_cap"],
                   {"type": "math", "answer": str(r["expected_answer"])},
                   {"bucket": spec["bucket"], "problem_source": r.get("problem_source")})
