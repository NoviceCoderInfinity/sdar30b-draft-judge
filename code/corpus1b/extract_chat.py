"""Chat extractors for the 1B corpus (docs/CORPUS_1B_BUILD.md): one function per sources.py tag.

Same conventions as extract_maths.py: each function takes a source spec dict and yields PROMPT ROWs (key
always {"type": "chat"}; no "wave"); loading is offline through the mockable `_load_hf_split` seam.

Field names verified against the HF datasets-server API on 2026-09-28:
  wildchat   allenai/WildChat-1M, config=default, split=train (only split). Row-level fields `conversation`
             (list of turns, each {role, content, language, toxic, redacted, ...}), and the SAME field names
             (`language`, `toxic`, `redacted`) also exist at the top level of the row, describing the whole
             conversation -- sources.py's filter reads those, and so does this extractor. First turn
             confirmed role == "user". source_id = `conversation_hash` (unique, stable).
  ultrachat  HuggingFaceH4/ultrachat_200k, config=default, split=train_sft (also has test_sft, train_gen,
             test_gen). Fields `prompt`, `prompt_id` (a hex digest; used as source_id), `messages`.
  oasst2     OpenAssistant/oasst2, config=default, split=train. Fields `message_id`, `parent_id`, `role`
             ("prompter"/"assistant"), `lang`, `text`, `deleted` (bool). sources.py's filter is role ==
             'prompter' and parent_id is None and lang == 'en'; this extractor additionally drops
             deleted == True rows (not in sources.py's filter list, but a retracted message is obviously
             not a training prompt -- noted here since it is a real addition, not just a restatement).
"""
from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402

DOMAIN = "chat"
CHAT_KEY = {"type": "chat"}


# ---------------------------------------------------------------------------
# mockable loader (tests monkeypatch this, never `datasets` directly)
# ---------------------------------------------------------------------------
def _load_hf_split(repo: str, config: Optional[str], split: str):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split)


def _row(tag: str, repo: str, source_id, prompt: str, gen_cap: int, meta: dict) -> dict:
    meta = dict(meta)
    assert "bucket" in meta, meta
    return {"pid": common.make_pid(DOMAIN, tag, repo, source_id), "domain": DOMAIN, "source": repo,
            "source_id": str(source_id), "prompt": prompt, "gen_cap": gen_cap, "key": dict(CHAT_KEY), "meta": meta}


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------
def wildchat(spec: dict) -> Iterator[dict]:
    """allenai/WildChat-1M train: English, non-toxic, non-redacted, first user turn, 20-4,000 chars."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        if r.get("language") != "English" or r.get("toxic") or r.get("redacted"):
            continue
        conv = r.get("conversation") or []
        first = next((t for t in conv if t.get("role") == "user"), None)
        if first is None:
            continue
        text = (first.get("content") or "").strip()
        if not (20 <= len(text) <= 4000):
            continue
        yield _row(spec["tag"], spec["repo"], r["conversation_hash"], text, spec["gen_cap"],
                   {"bucket": spec["bucket"]})


def ultrachat(spec: dict) -> Iterator[dict]:
    """HuggingFaceH4/ultrachat_200k train_sft: the `prompt` field, as-is."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        yield _row(spec["tag"], spec["repo"], r["prompt_id"], text, spec["gen_cap"], {"bucket": spec["bucket"]})


def oasst2(spec: dict) -> Iterator[dict]:
    """OpenAssistant/oasst2 train: root prompter messages (role prompter, parent_id null, lang en, not deleted)."""
    ds = _load_hf_split(spec["repo"], spec["config"], spec["split"])
    for r in ds:
        if r.get("role") != "prompter" or r.get("parent_id") is not None or r.get("lang") != "en":
            continue
        if r.get("deleted"):
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        yield _row(spec["tag"], spec["repo"], r["message_id"], text, spec["gen_cap"], {"bucket": spec["bucket"]})
