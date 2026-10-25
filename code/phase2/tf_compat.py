"""transformers compatibility for Phase 2's venv (/venv/p2, transformers 4.54.1).

Why: no single unpatched transformers loads both the SDAR remote code (written for 4.52.4) and the new drafters
(Fast-dLLM v2 needs >= 4.53, the dLLM BD3LM checkpoints >= 4.54). On >= 4.54 the SDAR code's
`from transformers.utils import LossKwargs` fails because LossKwargs was removed; it is only a TypedDict base for
the kwargs annotation of `SDARForCausalLM.forward` (modeling_sdar.py:736), so an equivalent TypedDict restores it
without touching any computation. On 4.52.4 LossKwargs exists and `apply()` changes nothing.

`research_protocol.load_sdar` calls `apply()` before every SDAR load, and `phase2.drafters.load` before every
drafter load. setup_p2.sh then checks on the real 30B model that --stock still reproduces the recorded stock cell
token for token under this venv (the stop rule stock_integrity).
"""
from __future__ import annotations


def apply() -> list:
    import transformers.utils as tu
    done = []
    if not hasattr(tu, "LossKwargs"):
        from typing import Optional, TypedDict

        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: Optional[int]

        tu.LossKwargs = LossKwargs
        done.append("LossKwargs")
    return done
