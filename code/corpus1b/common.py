"""Shared conventions for the 1B-token judge corpus (docs/CORPUS_1B_BUILD.md).

Everything here is CPU-only and import-safe (no model, no network).

Row schemas
-----------
PROMPT ROW  (data/c1b/prompt_lists/<domain>.jsonl, one JSON object per line; built by build_prompt_lists.py)
    pid       str   "<domain>-<source_tag>-<h16>", h16 = sha256(source + "|" + source_id)[:16]; unique corpus-wide
    domain    str   one of DOMAINS
    source    str   HF repo id of the source dataset ("openai/gsm8k", ...)
    source_id str   the row's id / index inside that source (stable across rebuilds)
    prompt    str   the RAW user message. The chat template is applied at generation time with
                    research_protocol.encode_prompt(tok, prompt) -- never here.
    gen_cap   int   answer token cap (512 for function-level code, 1,024 otherwise)
    key       dict  the answer key; its "type" selects the checker (KEY_TYPES below)
    meta      dict  free-form provenance (level, subject, template, ...)
    wave      str   "A" or "B" = wave_of(pid)

KEY TYPES (checkers.check dispatches on key["type"])
    math      {"type": "math", "answer": str}                       last \\boxed{} vs answer (MATH evaluator)
    asserts   {"type": "asserts", "tests": str}                     MBPP-style: newline-joined assert lines
    humaneval {"type": "humaneval", "test": str, "entry_point": str}
    stdio     {"type": "stdio", "cases": [{"input": str, "output": str}, ...], "checker": str|None}
    mc        {"type": "mc", "answer": "A".."J", "n_options": int}   letter in the last \\boxed{}
    free      {"type": "free", "answer": str, "answer_type": str}    science free-form (numeric tolerance / equiv)
    ifeval    {"type": "ifeval", "gold": str}                        JSON {"instruction_id_list", "kwargs"}
    chat      {"type": "chat"}                                       entry check only (ends, no loop, English)

GENERATION RECORD (logs/c1b/gen/<wave>/..., written by gen_sglang.py; one per decoded prompt)
    pid, domain, wave, prompt_len, block_start0 (= (prompt_len // 32) * 32),
    blocks: [{"ids": [32 ints], "steps": [32 ints]}...]   steps[j] = stock pass that committed slot j
                                                         (-1 = not generated: the prompt tail of block 0)
    eos (bool: an EOS id inside the gen_cap), n_gen (generated tokens up to and incl. the first EOS),
    kept (bool, checker verdict), reason (str, why dropped; "" when kept)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import unicodedata
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))              # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))                # code/
ROOT = os.path.abspath(os.path.join(CODE, ".."))                # repository root

BLOCK = 32
MASK_ID = 151669                     # == research_protocol.MASK_ID
EOS_IDS = (151645, 151643)           # == research_protocol.EOS_IDS

DOMAINS = ("maths", "code", "science", "if", "chat")
KEY_TYPES = ("math", "asserts", "humaneval", "stdio", "mc", "free", "ifeval", "chat")

# plan targets (docs/CORPUS_1B_PLAN_2026-09-28.md): prompts to decode per domain, both waves together
TARGET_PROMPTS = {"maths": 894_000, "code": 710_000, "science": 455_000, "if": 940_000, "chat": 342_000}
WAVE_A_FRACTION = 0.128              # 128M of 1B tokens; a hash of the pid decides, so wave A is uniform

# on-disk layout (all under the repository root, so research_protocol.safe_out accepts every path)
DATA_DIR = os.path.join(ROOT, "data", "c1b")
PROMPT_DIR = os.path.join(DATA_DIR, "prompt_lists")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
LOG_DIR = os.path.join(ROOT, "logs", "c1b")


# ---------------------------------------------------------------------------
# ids and waves
# ---------------------------------------------------------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    """NFKC, lower-case, every whitespace character removed -- the exact-match / dedup key used across
    extract_*.py, decontam.py and dedupe.py (same convention as draftalign/build_if_pool.py::normalise)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()


def make_pid(domain: str, source_tag: str, source: str, source_id) -> str:
    assert domain in DOMAINS, domain
    assert "-" not in source_tag and ":" not in source_tag, source_tag
    return f"{domain}-{source_tag}-{sha256_hex(f'{source}|{source_id}')[:16]}"


def domain_of(pid: str) -> str:
    d = pid.split("-", 1)[0]
    assert d in DOMAINS, pid
    return d


def source_tag_of(pid: str) -> str:
    """"maths-gsm8k-<h16>" -> "gsm8k". Safe because make_pid forbids "-"/":" in source_tag."""
    parts = pid.split("-", 2)
    assert len(parts) == 3, pid
    return parts[1]


def wave_of(pid: str, frac_a: float = WAVE_A_FRACTION) -> str:
    """'A' for a uniform `frac_a` share of pids (a hash, not an order), else 'B'."""
    u = int(sha256_hex("c1b-wave|" + pid)[:12], 16) / float(16 ** 12)
    return "A" if u < frac_a else "B"


def order_key(pid: str, salt: str = "c1b-order") -> str:
    """Deterministic order for sampling / sharding that does not depend on file order."""
    return sha256_hex(f"{salt}|{pid}")


# ---------------------------------------------------------------------------
# the loop rule and other entry checks on token ids
# ---------------------------------------------------------------------------
LOOP_SPAN, LOOP_REPEATS = 32, 3


def has_loop(ids: Sequence[int], span: int = LOOP_SPAN, repeats: int = LOOP_REPEATS) -> bool:
    """True if some window of `span` consecutive tokens occurs at `repeats` or more start positions.

    Every start position counts (overlapping windows), so a one-token run ("hours hours hours ...") of
    length >= span + repeats - 1 is caught, as are a repeated sentence and a repeated paragraph.
    """
    n = len(ids)
    if n < span + repeats - 1:
        return False
    seen: Dict[tuple, int] = {}
    t = tuple(int(x) for x in ids)
    for i in range(n - span + 1):
        w = t[i:i + span]
        c = seen.get(w, 0) + 1
        if c >= repeats:
            return True
        seen[w] = c
    return False


def first_eos(ids: Sequence[int], eos_ids: Iterable[int] = EOS_IDS) -> int:
    """Index of the first EOS id, or -1."""
    eos = set(int(e) for e in eos_ids)
    for i, t in enumerate(ids):
        if int(t) in eos:
            return i
    return -1


# ---------------------------------------------------------------------------
# JSONL helpers (plain or .gz), append-safe
# ---------------------------------------------------------------------------
def open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8") if "t" in mode else gzip.open(path, mode)
    return open(path, mode.replace("t", ""), encoding="utf-8")


def read_jsonl(path: str) -> Iterator[dict]:
    """Yield objects; a truncated last line (a killed writer) is skipped, not fatal."""
    with open_text(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def write_jsonl(path: str, rows: Iterable[dict], mode: str = "wt") -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open_text(path, mode) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_path(*parts: str) -> str:
    """Absolute path under the repository root (research_protocol.safe_out accepts it)."""
    p = os.path.abspath(os.path.join(ROOT, *parts))
    assert p.startswith(ROOT + os.sep), p
    return p
