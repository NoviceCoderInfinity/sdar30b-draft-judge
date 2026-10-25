#!/usr/bin/env python
"""Strict, deterministic scorer for the verifiable instruction-following pool (code/configs/if_pool.json).

  score_ifeval(pred, gold)        -> 1 iff `pred` passes EVERY constraint in `gold`, else 0
  score_ifeval_record(rec)        -> score_ifeval on a run_da_decode record, fail-closed on unfinished output
  parse_gold(gold)                -> (instruction_ids, kwargs), validated against the vendored registry
  check_constraints(resp, ids, kw)-> [bool] per constraint, strict
  build_instruction(iid, kwargs)  -> a built checker instance (fixed RNG seed)

Checker: open-instruct's IFEvalG -- IFEval's 25 Google verifiers plus the 29 IFBench-train verifiers -- the
verifier the allenai/IF_multi_constraints_upto5 authors point to (IFBench README). Vendored byte-identical in
code/vendor/ifeval/; provenance, commit and hashes in code/vendor/ifeval/PROVENANCE.md.

`gold` is a JSON string {"instruction_id_list": [...], "kwargs": [{...}, ...]}: the google/IFEval input format,
one kwargs dict per id, no null values (code/draftalign/build_if_pool.py writes it). A malformed gold or an id the
registry does not implement RAISES -- that is a pipeline bug, not a wrong answer.

Strict mode is IFEval's `evaluation_lib.test_instruction_following_strict` (identical in allenai/IFBench):
for each constraint, `build_description(**kwargs)`, then `response.strip() and check_following(response)`; the
item is correct iff every constraint passes. No loose variant (no markdown stripping, no first/last-line
removal). Deviations from that loop, all fail-closed:

  * reasoning: if `</think>` occurs, only the text after the LAST `</think>` is scored, with its surrounding
    whitespace stripped (the template's separator); a `<think>` without a closing tag means the response
    never reached an answer -> 0. Text without think tags is scored unchanged. SDAR's chat_template.jinja
    does know <think> blocks, so this is not hypothetical.
  * a checker that raises counts as not followed (the reference loop would crash).
  * score_ifeval_record only: no EOS inside the generation budget, or any non-EOS token after the first EOS,
    -> 0. `pred` is tok.decode of ALL non-mask generated ids, so text after EOS would otherwise be scored as
    part of the answer, and a truncated response can satisfy "at least N" constraints without finishing.

Determinism: `build_description` draws from the global `random` module whenever a kwarg is absent; every build
here runs under random.seed(BUILD_SEED) and restores the caller's RNG state. langdetect (the language and
English-case checkers) is seeded at import (DetectorFactory.seed = 0). The pool builder also drops any row whose
checker args differ between two build seeds, so no admitted item depends on BUILD_SEED.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.abspath(os.path.join(HERE, "..", "vendor", "ifeval"))
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

import langdetect                                                   # noqa: E402

langdetect.DetectorFactory.seed = 0                                 # langdetect is randomised; pin it

from open_instruct.IFEvalG import instructions_registry            # noqa: E402  (vendored, see PROVENANCE.md)

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT
BUILD_SEED = 0
EOS_IDS = (151645, 151643)            # == research_protocol.EOS_IDS (pinned by code/tests/test_if_scorer.py)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def strip_reasoning(pred: str):
    """Answer text of a response: after the last </think> (stripped); None if a <think> never closes."""
    if THINK_CLOSE in pred:
        return pred.rsplit(THINK_CLOSE, 1)[1].strip()
    if THINK_OPEN in pred:
        return None
    return pred


def parse_gold(gold) -> Tuple[List[str], List[dict]]:
    g = json.loads(gold) if isinstance(gold, str) else gold
    ids, kwargs = g["instruction_id_list"], g["kwargs"]
    if not (isinstance(ids, list) and isinstance(kwargs, list) and ids and len(ids) == len(kwargs)):
        raise ValueError(f"malformed IF gold: {gold!r}")
    for iid, kw in zip(ids, kwargs):
        if iid not in INSTRUCTION_DICT:
            raise KeyError(f"instruction id {iid!r} is not implemented by the vendored checker")
        if not isinstance(kw, dict) or any(v is None for v in kw.values()):
            raise ValueError(f"kwargs of {iid!r} must be a dict without nulls: {kw!r}")
    return list(ids), [dict(kw) for kw in kwargs]


def build_instruction(iid: str, kwargs: dict, seed: int = BUILD_SEED):
    inst = INSTRUCTION_DICT[iid](iid)
    state = random.getstate()
    random.seed(seed)
    try:
        inst.build_description(**kwargs)
    finally:
        random.setstate(state)
    args = inst.get_instruction_args()
    # the reference loop re-builds with the prompt when an instruction has an argument literally named
    # "prompt"; no vendored instruction has one, so that branch cannot change any result here.
    assert not (args and "prompt" in args), f"{iid} takes a `prompt` argument; the scorer does not pass one"
    return inst


def check_constraints(response: str, ids: Sequence[str], kwargs: Sequence[dict]) -> List[bool]:
    out = []
    for iid, kw in zip(ids, kwargs):
        inst = build_instruction(iid, kw)
        try:
            ok = bool(response.strip()) and bool(inst.check_following(response))
        except Exception:                                           # fail closed
            ok = False
        out.append(ok)
    return out


def score_ifeval(pred: str, gold: str) -> int:
    """1 iff the response (reasoning stripped) passes every constraint of `gold` under the strict checker."""
    ids, kwargs = parse_gold(gold)
    response = strip_reasoning(pred) if isinstance(pred, str) else None
    if response is None or not response.strip():
        return 0
    return int(all(check_constraints(response, ids, kwargs)))


def finished_at_eos(rec: dict) -> bool:
    """The record's generation reached an EOS inside its budget (`n_gen_to_eos` points at an EOS id).
    Block diffusion always fills the whole final 32-slot block, so tokens AFTER the first EOS are normal
    filler, not part of the answer (da_decode.make_record: n_gen_to_eos = first EOS + 1) -- they are ignored."""
    gen, n = rec.get("gen_ids"), rec.get("n_gen_to_eos")
    if gen is None or n is None:
        return True                                                 # nothing to check against (pred-only rows)
    n = int(n)
    return 1 <= n <= len(gen) and int(gen[n - 1]) in EOS_IDS


_TOK = None


def response_text(rec: dict) -> str:
    """The answer as the model produced it: gen_ids up to (excluding) the first EOS, decoded with the b32
    tokenizer. `rec["pred"]` decodes ALL non-mask ids, i.e. it also carries the post-EOS filler of the last
    block, which would corrupt end-with / no-comma / count checks. Falls back to pred for pred-only rows."""
    global _TOK
    gen, n = rec.get("gen_ids"), rec.get("n_gen_to_eos")
    if gen is None or n is None:
        return rec["pred"]
    if _TOK is None:
        from transformers import AutoTokenizer
        import research_protocol as P
        m = P.MODELS["b32"]
        _TOK = AutoTokenizer.from_pretrained(m["repo"], trust_remote_code=True, local_files_only=True,
                                             revision=m["revision"], cache_dir=m["cache_dir"])
    return _TOK.decode([int(t) for t in gen[:int(n) - 1]], skip_special_tokens=True)


def response_ids(text: str) -> List[int]:
    """b32 token ids of `text` (test helper: build gen_ids the way the decoder would)."""
    response_text({"gen_ids": [], "n_gen_to_eos": 1, "pred": ""})   # loads the tokenizer
    return [int(t) for t in _TOK(text, add_special_tokens=False)["input_ids"]]


def score_ifeval_record(rec: dict) -> int:
    """score_ifeval(response up to EOS, rec["gold"]); 0 when the response never reached EOS in budget."""
    if not finished_at_eos(rec):
        return 0
    return score_ifeval(response_text(rec), rec["gold"])
