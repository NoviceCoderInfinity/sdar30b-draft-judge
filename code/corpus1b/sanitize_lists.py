#!/usr/bin/env python
"""Drop prompts that carry credential-like strings (third-party access tokens users pasted into public chat logs,
e.g. `hf_...` in WildChat) from built prompt lists, BEFORE their domain is generated, so the public corpus
backup never republishes them and the judge never trains on them. The same patterns the backup's secret scan
refuses (hf_sync.py); the dropped pids are recorded in data/c1b/prompt_lists/manifest.json under the domain
("dropped_credential_like"). A few rows out of hundreds of thousands: the mix is unchanged.

    /venv/train/bin/python code/corpus1b/sanitize_lists.py chat [if code ...]
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402

RX = re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")


def sanitize(domain: str) -> list:
    path = os.path.join(C.PROMPT_DIR, f"{domain}.jsonl")
    kept, dropped = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            (dropped if RX.search(line) else kept).append(line)
    if dropped:
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.writelines(kept)
        os.replace(path + ".tmp", path)
    pids = [json.loads(l)["pid"] for l in dropped]
    man = os.path.join(C.PROMPT_DIR, "manifest.json")
    with open(man + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        m = json.load(open(man))
        dm = m["domains"][domain]
        dm["dropped_credential_like"] = sorted(set(dm.get("dropped_credential_like", [])) | set(pids))
        dm["output"].update({"sha256": C.sha256_file(path), "n_rows": len(kept)})
        with open(man + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(man + ".tmp", man)
    print(f"[sanitize] {domain}: dropped {len(pids)} of {len(kept) + len(pids)} prompts {pids}", flush=True)
    return pids


if __name__ == "__main__":
    for d in sys.argv[1:]:
        sanitize(d)
