"""Existing-pool instruction-following extractor for the 1B-token corpus (docs/CORPUS_1B_BUILD.md; registry:
corpus1b/sources.py). Serves the "pool" bucket tags ifnemo / ifmulti / ifrlvr -- the three sources
draftalign/build_if_pool.py already screens for its frozen, program-verifiable round-3 IF pool.

  existing_pools(spec)  every row of `spec["repo"]` that survives EVERY round-3 build_if_pool.py filter (single
                        turn, gold parses, implemented ids, id policy, <=3 constraints, determinism, feasibility,
                        text consistency, origin, task screen, unlabelled demand, English, dedup, length,
                        decontamination -- see build_if_pool.py's module docstring) EXCEPT the final
                        sha256(key) sort + --n cut, i.e. draftalign.build_if_pool.eligible_items()'s full
                        eligible pool, filtered to this one source. prompt = the raw user message, exactly as
                        round 3 (build_if_pool never edits it).

Dedup is cross-source (a duplicate prompt's survivor can come from a different source than the one asking), so
this module runs the full three-source build_if_pool pipeline once per process and reuses it for all three tags
(`_eligible`, memoized -- see its docstring).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))              # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))                # code/
for _p in (HERE, os.path.join(CODE, "compat"), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import research_protocol as P                                   # noqa: E402  (sets HF offline env)
from common import make_pid                                     # noqa: E402

# spec["repo"] -> the build_if_pool.SOURCES key that candidate carries as c["src"]
_SRC_BY_REPO = {
    "nvidia/Nemotron-RL-instruction_following": "nemotron_rl_if",
    "allenai/IF_multi_constraints_upto5": "if_multi",
    "allenai/RLVR-IFeval": "rlvr",
}

_cache: Optional[Tuple[List[dict], dict]] = None  # (cands, stages); memoized across the three tags in one process


def _eligible(workers: int = 32) -> Tuple[List[dict], dict]:
    """draftalign.build_if_pool.eligible_items(workers), memoized for the life of the process. existing_pools()
    is called once per tag (ifnemo / ifmulti / ifrlvr) by the orchestrator; re-running the full three-source
    pipeline for each (it loads and screens ~156K candidates plus the 326K-row tulu-v2-sft-mixture origin join
    and the eval-set decontamination texts) would triple the cost of a single, pure computation. Call
    `reset_cache()` to force a rerun -- tests do this after monkeypatching build_if_pool.eligible_items."""
    global _cache
    if _cache is None:
        from draftalign import build_if_pool as bip
        cands, stages, _controls, _eval_texts, _length_before = bip.eligible_items(workers)
        _cache = (cands, stages)
    return _cache


def reset_cache() -> None:
    """Drop the memoized eligible pool (tests only; see `_eligible`)."""
    global _cache
    _cache = None


def existing_pools(spec: dict, workers: int = 32) -> Iterator[dict]:
    """PROMPT ROWS for one of the three existing IF pools (spec["tag"] in ifnemo / ifmulti / ifrlvr)."""
    from draftalign import build_if_pool as bip

    src = _SRC_BY_REPO[spec["repo"]]
    cands, _stages = _eligible(workers)
    tag, gen_cap, bucket = spec["tag"], spec["gen_cap"], spec["bucket"]
    for c in cands:
        if c["src"] != src:
            continue
        yield {
            "pid": make_pid("if", tag, c["source"], c["row"]), "domain": "if", "source": c["source"],
            "source_id": c["row"], "prompt": c["prompt"], "gen_cap": gen_cap,
            "key": {"type": "ifeval", "gold": bip.dump_gold(c["ids"], c["kwargs"])},
            "meta": {"bucket": bucket, "n_constraints": len(c["ids"]), "origin": c["origin"]},
        }
