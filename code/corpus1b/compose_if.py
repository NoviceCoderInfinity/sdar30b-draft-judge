"""Composed instruction-following extractor for the 1B-token corpus (docs/CORPUS_1B_BUILD.md; registry:
corpus1b/sources.py, tag "ifcomp"). Fills the IF domain's "composed" bucket (~85% of it) by taking open-ended
first user turns from allenai/WildChat-4.8M and appending 1-3 freshly drawn, non-conflicting IFEvalG
constraints, reusing draftalign/build_if_pool.py's own prompt screens (task screen, unlabelled demand, English)
and draftalign/if_scorer.py's checker registry, so a composed prompt is screened and its gold is scored exactly
like an existing-pool row's.

  composed(spec, exclude_norm=None)  yields PROMPT ROWS lazily, in dataset row order; the orchestrator samples
                                      by pid hash order up to the IF domain target (common.TARGET_PROMPTS["if"])
  pick_constraints(pid, ids_pool, conflicts, base)
                                      the constraint-selection step, standalone and separately testable
  admitted_ids() / conflict_map()    the id pool and its (symmetrised) conflict graph
"""
from __future__ import annotations

import hashlib
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))              # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))                # code/
VENDOR = os.path.abspath(os.path.join(CODE, "vendor", "ifeval"))
for _p in (HERE, os.path.join(CODE, "compat"), CODE, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import research_protocol as P                                   # noqa: E402  (sets HF offline env)
from common import make_pid                                     # noqa: E402

MIN_CHARS, MAX_CHARS = 20, 2000               # base-prompt character-length filter
N_TARGET_CUM = (0.50, 0.85, 1.00)             # cumulative thresholds: 1 constraint / 2 / 3


def _sha16_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def admitted_ids() -> List[str]:
    """sorted(if_scorer.INSTRUCTION_DICT ids not in build_if_pool.EXCLUDED_IDS) -- the same admitted set an
    existing-pool row's ids are held to (corpus1b/extract_if.py)."""
    from draftalign import if_scorer as S
    from draftalign import build_if_pool as bip
    return sorted(set(S.INSTRUCTION_DICT) - set(bip.EXCLUDED_IDS))


def conflict_map() -> Dict[str, Set[str]]:
    """id -> set of ids it must not be paired with (itself always included), from the vendored IFEvalG
    registry's own `instructions_registry.INSTRUCTION_CONFLICTS` -- checked in BOTH directions here, since the
    registry defines a `conflict_make` symmetriser but never calls it at import, so a few pairs are listed only
    one-way in the raw dict."""
    from open_instruct.IFEvalG import instructions_registry as reg
    sym: Dict[str, Set[str]] = {}
    for a, bs in reg.INSTRUCTION_CONFLICTS.items():
        sym.setdefault(a, {a}).update(bs)
        for b in bs:
            sym.setdefault(b, {b}).add(a)
    return sym


def pick_constraints(pid: str, ids_pool: List[str], conflicts: Dict[str, Set[str]],
                     base: str) -> Optional[Tuple[List[str], List[dict], List[str]]]:
    """1-3 deterministic, non-conflicting, feasible constraints for one composed prompt. Everything is derived
    from `random.Random(int(sha256(pid)[:16], 16))`: the target count (1 constraint 50% of the time, 2 35%, 3
    15%), the try order over `ids_pool`, and each candidate's build seed.

    Each candidate id is built with `build_if_pool._build(iid, kw, seed)` (build_if_pool's own builder --
    `if_scorer.build_instruction` discards the description text this needs): kw = {} for every id except
    "combination:repeat_prompt", pinned to `{"prompt_to_repeat": base}` (its checker requires the response to
    repeat exactly the prompt it names; left to the checker's own random fill under an empty kw it would name an
    unrelated canned phrase, which would be a well-formed but meaningless constraint here).
    `inst.get_instruction_args()` returns None, not {}, for a genuinely parameterless instruction (e.g.
    punctuation:no_comma) -- normalised to {} here. A build is kept only
    when `build_if_pool.determinism(iid, kw) == ""` (this also catches a kwarg literally named "prompt", which
    `if_scorer.build_instruction` refuses to score -- see its assert) and `not build_if_pool.infeasible(iid,
    kw)`. An id already banned by an earlier pick (itself and everything `conflicts` pairs it with) is skipped
    without spending a build.

    Returns (ids, kwargs, descriptions) with 1 <= len <= 3, or None when not even one constraint could be
    assembled (rare, given ~30 admitted ids and a sparse conflict graph); the caller then drops the prompt.
    """
    from draftalign import build_if_pool as bip

    rng = random.Random(_sha16_int(pid))
    u = rng.random()
    n_target = 1 if u < N_TARGET_CUM[0] else (2 if u < N_TARGET_CUM[1] else 3)
    pool = list(ids_pool)
    rng.shuffle(pool)

    chosen_ids: List[str] = []
    chosen_kwargs: List[dict] = []
    chosen_desc: List[str] = []
    banned: Set[str] = set()
    for iid in pool:
        if len(chosen_ids) == n_target:
            break
        if iid in banned:
            continue
        seed = rng.getrandbits(63)
        kw0 = {"prompt_to_repeat": base} if iid == "combination:repeat_prompt" else {}
        try:
            inst, desc = bip._build(iid, kw0, seed)
        except Exception:
            continue
        kw = inst.get_instruction_args() or {}    # a parameterless instruction (e.g. punctuation:no_comma) -> None
        if "prompt" in kw:                                         # if_scorer.build_instruction refuses this id
            continue
        if bip.determinism(iid, kw) or bip.infeasible(iid, kw):
            continue
        chosen_ids.append(iid)
        chosen_kwargs.append(kw)
        chosen_desc.append(desc)
        banned |= conflicts.get(iid, {iid})
    if not chosen_ids:
        return None
    return chosen_ids, chosen_kwargs, chosen_desc


def _eligible_base(text: str, exclude_norm: Optional[Set[str]]) -> bool:
    """Every screen that depends only on the candidate base-prompt text (WildChat's own per-row flags -- role,
    language label, toxic, redacted -- are checked in `composed`, cheaper and first): 20-2,000 characters, not a
    normalised duplicate of a chat-domain prompt, build_if_pool's task-screen and unlabelled-demand regexes, and
    build_if_pool's own English check (langdetect, deterministic, checked last: the priciest of the five) -- the
    same bar an existing IF-pool row's base prompt is held to."""
    from draftalign import build_if_pool as bip
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if exclude_norm and bip.normalise(text) in exclude_norm:
        return False
    if bip.task_screen_reason(text):
        return False
    if bip.unlabelled_demand_reason(text):
        return False
    if bip.english_reason(text):
        return False
    return True


def composed(spec: dict, exclude_norm: Optional[Set[str]] = None) -> Iterator[dict]:
    """PROMPT ROWS composed from allenai/WildChat-4.8M first user turns.

    Filters, cheapest first: `conversation[0].role == "user"`; WildChat's own per-turn `language == "English"`
    (a cheap pre-filter -- `_eligible_base` still runs the authoritative langdetect check below); non-toxic, not
    redacted (both per-turn fields in this dataset); then `_eligible_base` (length, task screen, unlabelled
    demand, `exclude_norm`, English). A kept prompt draws 1-3 constraints (`pick_constraints`); a prompt for
    which not even one could be assembled contributes nothing.

    `exclude_norm`: normalised (`build_if_pool.normalise`) chat-domain prompts, passed by the orchestrator, so a
    composed IF prompt never duplicates a chat-domain training prompt.

    Yields lazily in dataset row order -- this function does not sample or cap; the orchestrator samples by pid
    hash order up to the IF domain target (common.TARGET_PROMPTS["if"]).
    """
    from datasets import load_dataset
    from draftalign import build_if_pool as bip

    repo, tag, gen_cap, bucket = spec["repo"], spec["tag"], spec["gen_cap"], spec["bucket"]
    ids_pool, conflicts = admitted_ids(), conflict_map()
    ds = load_dataset(repo, split=spec.get("split", "train"))
    for row in ds:
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        turn = conv[0]
        if turn.get("language") != "English":
            continue
        if turn.get("toxic") or turn.get("redacted"):
            continue
        text = turn.get("content") or ""
        if not _eligible_base(text, exclude_norm):
            continue
        pid = make_pid("if", tag, repo, row["conversation_hash"])
        picked = pick_constraints(pid, ids_pool, conflicts, text)
        if picked is None:
            continue
        ids, kwargs, descriptions = picked
        prompt = text + "\n\n" + " ".join(descriptions)
        yield {
            "pid": pid, "domain": "if", "source": repo, "source_id": row["conversation_hash"],
            "prompt": prompt, "gen_cap": gen_cap,
            "key": {"type": "ifeval", "gold": bip.dump_gold(ids, kwargs)},
            "meta": {"bucket": bucket, "n_constraints": len(ids),
                     "base_prompt_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()},
        }
