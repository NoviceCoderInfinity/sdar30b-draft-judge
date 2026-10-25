#!/usr/bin/env python
"""Late decontamination of finished prompt lists against eval sets their build could not load.

On the 2026-09-28 box, decontam._configs did not expand the ALL_7_SUBJECTS sentinel until 15:16 UTC, so the chat,
science, maths and code lists were built without `math_test` (MATH test, 5,000 problems; MATH-500 itself was
covered through HuggingFaceH4/MATH-500). Rather than rebuild them, every list is screened here with the same three
checks (exact, containment, 13-gram; positive control first) against LATE_TAGS, the flags are cached per domain in
data/c1b/prompt_lists/postdecontam_<domain>.json (keyed by the list's sha256), and to_trainset.py drops every
flagged pid, so no flagged prompt reaches a training item.

    /venv/train/bin/python code/corpus1b/postdecontam.py            # screen every built list, print counts
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402

LATE_TAGS = ("math_test",)


def _index():
    import decontam as D
    import sources as S
    texts, warnings = {}, []
    for spec in S.EVAL_SETS:
        if spec["tag"] in LATE_TAGS:
            got = D._load_tagged(spec["tag"], spec["repo"], spec["config"], D._split_names(spec), spec["field"], warnings)
            assert got, f"late decontam: {spec['tag']} loaded 0 texts ({warnings})"
            texts[spec["tag"]] = got
    idx = D.DecontamIndex(texts)
    idx.positive_control()
    return idx


def flagged(domain: str, _cache={}) -> Dict[str, str]:
    """pid -> reason for every prompt of `domain`'s list that matches a LATE_TAGS eval text (cached on disk)."""
    lst = os.path.join(C.PROMPT_DIR, f"{domain}.jsonl")
    if not os.path.exists(lst):
        return {}
    out = os.path.join(C.PROMPT_DIR, f"postdecontam_{domain}.json")
    sha = C.sha256_file(lst)
    if os.path.exists(out):
        got = json.load(open(out))
        if got.get("list_sha256") == sha and tuple(got.get("tags", ())) == LATE_TAGS:
            return got["flagged"]
    if "idx" not in _cache:
        _cache["idx"] = _index()
    rows = list(C.read_jsonl(lst))
    verdicts = _cache["idx"].flag_batch([r["prompt"] for r in rows])
    fl = {r["pid"]: v for r, v in zip(rows, verdicts) if v}
    rep = {"domain": domain, "list_sha256": sha, "tags": list(LATE_TAGS), "n_rows": len(rows), "n_flagged": len(fl),
           "flagged": fl}
    json.dump(rep, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print(f"[postdecontam] {domain}: {len(fl)} of {len(rows)} prompts flagged against {LATE_TAGS}", flush=True)
    return fl


def all_flagged() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in C.DOMAINS:
        out.update(flagged(d))
    return out


if __name__ == "__main__":
    fl = all_flagged()
    print(json.dumps({"flagged_total": len(fl)}))
