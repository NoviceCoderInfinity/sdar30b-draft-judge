"""Science extractors for the 1B corpus (docs/CORPUS_1B_BUILD.md): one function per sources.py tag.

Same conventions as extract_maths.py: each function takes a source spec dict and yields PROMPT ROWs
(no "wave"); loading is offline (research_protocol sets HF_*_OFFLINE=1 on import) through the mockable
`_load_hf_split` seam tests monkeypatch.

Field names verified against the HF datasets-server API on 2026-09-28:
  webinst          TIGER-Lab/WebInstruct-verified, config=default, split=train. Fields `id`, `question`,
                   `answer`, `answer_type` (Expression/Float/Integer/String/List/Percentage/Fraction/
                   Boolean/Other/Matrix/"Multiple Choice"), `category` (Physics/Chemistry/Biology/
                   Mathematics/Finance/Business/...). "Multiple Choice" rows write the options straight
                   into `question` in inconsistent, freeform lettering (lower-case "a." in one row,
                   upper-case "A." in the next; `answer` is similarly inconsistent case) -- not worth a
                   bespoke parser given nemosci_mcq/osr2/arc already cover 'mc'. This extractor takes
                   sources.py's primary filter (`answer_type != 'Multiple Choice'`) and drops them, rather
                   than its parenthetical "(or map to key mc when the options are in the question)".
  nemosci_mcq/rqa  nvidia/Nemotron-Science-v1, config=default, split=MCQ or split=RQA. Fields `uuid`,
                   `messages` (list of {role, content, reasoning_content}; one user + one assistant turn).
                   gold = the LAST \\boxed{} of the assistant turn's `content`. Verified on a 20-row sample
                   of each split: RQA is reliable (20/20 boxed); MCQ is not -- only ~9/20 assistant turns
                   used \\boxed{LETTER} (the rest wrote "Answer: E", "The Best Answer is B.", "**Final
                   Answer: B**", "((E))", none of which this extractor parses). Rows without a boxed
                   answer are dropped and counted (reason "no_boxed_answer") -- a real yield cut on nemosci_
                   mcq, not a bug; 174K source rows comfortably covers what the 'mc' bucket needs even so.
  osr2             nvidia/OpenScienceReasoning-2, config=default, split=train. **307,862 rows, not "1.6M"**
                   as sources.py's notes claimed (verified via the datasets-server /info endpoint; the notes
                   field is corrected in sources.py). Fields `expected_answer`, `input`, `output` only (no
                   id/domain/subject field: source_id is sha256(input)[:16]). Lettered options in `input`
                   are colon-separated ("A: ...\\nB: ...", not "A)"). On a 250-row sample spread across the
                   full offset range, EVERY row had a single-letter A-J `expected_answer` with matching
                   lettered options in `input` -- i.e. in practice this source is close to 100% MCQ-shaped,
                   so it contributes almost nothing to the 'free' half of the science mix. Implemented
                   exactly as specified (per-row: single-letter answer + lettered options -> mc, else free)
                   so a future snapshot with more free-form rows is still handled correctly.
  arc              allenai/ai2_arc, config=["ARC-Challenge","ARC-Easy"] (sources.py's `config` is a list;
                   both are read), split=train. Fields `id`, `question`, `choices` ({text:[...], label:
                   [...]}), `answerKey`. Confirmed both alphabetic ("A".."D") and numeric ("1".."4")
                   answerKey styles occur (ARC-Easy train has both); `choices["label"]` is written in the
                   same style as `answerKey`, so `choices["label"].index(answerKey)` maps either style to a
                   0-based option index without a hardcoded 1->A table.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, Iterator, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402
import sources                                              # noqa: E402
import research_protocol as P                               # noqa: E402  (sets HF_*_OFFLINE=1 on import)

DOMAIN = "science"
_OPT_PAREN = re.compile(r"(?m)^\(?([A-J])\)\s")             # "A) text" (nemosci_mcq user turns)
_OPT_COLON = re.compile(r"(?m)^\(?([A-J]):\s")              # "A: text" (osr2 `input`)
_LATEX_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# mockable loader (tests monkeypatch this, never `datasets` directly)
# ---------------------------------------------------------------------------
def _load_hf_split(repo: str, config: Optional[str], split: str):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split)


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _row(tag: str, repo: str, source_id, prompt: str, gen_cap: int, key: dict, meta: dict) -> dict:
    meta = dict(meta)
    assert "bucket" in meta, meta
    return {"pid": common.make_pid(DOMAIN, tag, repo, source_id), "domain": DOMAIN, "source": repo,
            "source_id": str(source_id), "prompt": prompt, "gen_cap": gen_cap, "key": key, "meta": meta}


def _clean_letter(boxed: Optional[str]) -> Optional[str]:
    """A \\boxed{} payload -> a bare "A".."J", or None if it is not (cleanly) a single letter."""
    if boxed is None:
        return None
    s = boxed.strip()
    m = _LATEX_WRAP.fullmatch(s)
    if m:
        s = m.group(1).strip()
    s = s.strip("$ .")
    return s if re.fullmatch(r"[A-J]", s) else None


def _letters_phrase(n: int) -> str:
    """4 -> "A, B, C or D" (sources.MC_TMPL's `letters` slot), generalised to any option count >= 1."""
    ls = [chr(ord("A") + i) for i in range(n)]
    return ls[0] if n <= 1 else ", ".join(ls[:-1]) + f" or {ls[-1]}"


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------
def webinstruct(spec: dict) -> Iterator[dict]:
    """TIGER-Lab/WebInstruct-verified train: category in {Physics, Chemistry, Biology}, answer_type not MC."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        if r.get("category") not in ("Physics", "Chemistry", "Biology"):
            continue
        if r.get("answer_type") == "Multiple Choice":
            continue
        answer = (r.get("answer") or "").strip()
        if not answer:
            continue
        yield _row(spec["tag"], spec["repo"], r["id"], sources.GSM8K_TMPL.format(q=r["question"]),
                   spec["gen_cap"], {"type": "free", "answer": answer, "answer_type": r.get("answer_type")},
                   {"bucket": spec["bucket"], "category": r["category"], "answer_type": r.get("answer_type")})


def nemotron_science(spec: dict) -> Iterator[dict]:
    """nvidia/Nemotron-Science-v1, split MCQ or RQA. gold = last \\boxed{} of the assistant turn; dropped
    when absent (see module docstring: reliable on RQA, ~45% miss rate on MCQ in the verified sample)."""
    is_mcq = spec["split"] == "MCQ"
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        msgs = r.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if not user or not assistant:
            continue
        boxed = P.last_boxed(assistant)
        if boxed is None:
            continue
        if is_mcq:
            letter = _clean_letter(boxed)
            if letter is None:
                continue
            n_options = len(set(_OPT_PAREN.findall(user)))
            if n_options == 0:
                continue
            key = {"type": "mc", "answer": letter, "n_options": n_options}
        else:
            answer = boxed.strip()
            if not answer:
                continue
            key = {"type": "free", "answer": answer, "answer_type": "auto"}
        yield _row(spec["tag"], spec["repo"], r["uuid"], user, spec["gen_cap"], key, {"bucket": spec["bucket"]})


def openscience2(spec: dict) -> Iterator[dict]:
    """nvidia/OpenScienceReasoning-2 train. A single-letter `expected_answer` with matching lettered options
    in `input` -> key mc (bucket mc); everything else -> key free (bucket free). See module docstring: in
    the verified sample this source is almost entirely the mc case."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        answer = (r.get("expected_answer") or "").strip()
        inp = r.get("input") or ""
        if not answer:
            continue
        sid = common.sha256_hex(inp)[:16]
        letters = sorted(set(_OPT_COLON.findall(inp)))
        if re.fullmatch(r"[A-J]", answer) and letters and answer in letters:
            key = {"type": "mc", "answer": answer, "n_options": len(letters)}
            bucket = "mc"
        else:
            key = {"type": "free", "answer": answer, "answer_type": "auto"}
            bucket = "free"
        yield _row(spec["tag"], spec["repo"], sid, inp, spec["gen_cap"], key, {"bucket": bucket})


def arc(spec: dict) -> Iterator[dict]:
    """allenai/ai2_arc train, both ARC-Challenge and ARC-Easy. answerKey mapped to a letter via its position
    in `choices["label"]` (handles both the "A".."D" and the "1".."4" answerKey styles); kept iff that
    letter is in A-E (sources.py filter)."""
    configs = spec["config"] if isinstance(spec["config"], (list, tuple)) else [spec["config"]]
    for cfg in configs:
        ds = _load_hf_split(spec["repo"], cfg, spec["split"])
        for r in ds:
            labels: List[str] = list(r["choices"]["label"])
            texts: List[str] = list(r["choices"]["text"])
            key_raw = str(r["answerKey"])
            if key_raw not in labels:
                continue
            idx = labels.index(key_raw)
            letter = chr(ord("A") + idx)
            if letter not in "ABCDE" or idx >= len(texts):
                continue
            opts = "\n".join(f"{chr(ord('A') + i)}) {t}" for i, t in enumerate(texts))
            q = f"{r['question']}\n\n{opts}"
            prompt = sources.MC_TMPL.format(q=q, letters=_letters_phrase(len(texts)))
            yield _row(spec["tag"], spec["repo"], f"{cfg}:{r['id']}", prompt, spec["gen_cap"],
                       {"type": "mc", "answer": letter, "n_options": len(texts)},
                       {"bucket": spec["bucket"], "config": cfg})
