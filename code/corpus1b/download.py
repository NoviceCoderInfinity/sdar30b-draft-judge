#!/usr/bin/env python
"""Download every dataset the 1B corpus reads (sources.SOURCES, EVAL_SETS, HELD_OUT) into the local HF
cache, and record what was actually resolved in data/c1b/sources_manifest.json.

Run this on the training box only. This Mac (the repo's dev machine) must never run it in any mode but
--list: extract_*.py and decontam.py then read the cache offline (research_protocol sets HF_HUB_OFFLINE=1
/ HF_DATASETS_OFFLINE=1 on import, so nothing they do can accidentally download). Because of that, THIS
script must flip both back to "0" before anything it imports has a chance to import research_protocol --
done at the top of the file, ahead of every other import.

Usage (training box):
    python code/corpus1b/download.py --list                    # metadata only: sizes + shas, no download
    python code/corpus1b/download.py                            # every SOURCES + EVAL_SETS + HELD_OUT entry
    python code/corpus1b/download.py --only gsm8k,mathtrain      # just these tags (comma-separated)
    python code/corpus1b/download.py --include-optional          # also the EVAL_SETS entries marked optional

`--only` matches a tag from any of the three registries (sources.py asserts tags are unique within each
list, and by inspection they don't collide across lists either). A tag whose repo needs authorisation
(sources.py "gated": True) is attempted anyway when HF_TOKEN is set; on failure the run continues and the
manifest records the error against that tag rather than aborting the whole download.

Three repos are large enough to need `allow_patterns` (7.5-16 GB apiece for just the split this corpus
reads): nvidia/OpenMathInstruct-2 (restricted to the full 14M-row "train" split's 32 parquet shards, not
the separate train_1M/2M/5M shard sets), allenai/WildChat-4.8M and nvidia/OpenScienceReasoning-2 (both
already only have the one split, but are still restricted so a future repo change can't silently pull in
extra files). codeparrot/apps ships a *loading script* (`apps.py`) rather than plain data files, which the
datasets-server API and plain `datasets.load_dataset` both refuse to serve from the cache in a sandboxed
read; sources.py's own fix is the one this script follows -- fetch its raw `train.jsonl` (107 MB) directly
and skip the unused `test.jsonl` (1.3 GB), so extract_code.py can read it with plain `json.loads` per line.
"""
from __future__ import annotations

# Must happen before anything below (transitively) imports research_protocol, which sets these to "1".
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse                                            # noqa: E402
import datetime                                             # noqa: E402
import json                                                  # noqa: E402
import re                                                     # noqa: E402
import sys                                                      # noqa: E402
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402  (does NOT import research_protocol)
import sources                                               # noqa: E402

MANIFEST_PATH = os.path.join(common.DATA_DIR, "sources_manifest.json")

# repo -> snapshot_download allow_patterns. Anything not listed here is downloaded in full (every other
# repo in SOURCES/EVAL_SETS/HELD_OUT is small enough -- under a few hundred MB -- that restricting patterns
# would only add fragility for no real saving).
ALLOW_PATTERNS: Dict[str, List[str]] = {
    "nvidia/OpenMathInstruct-2": ["data/train-*-of-00032.parquet"],
    "allenai/WildChat-4.8M": ["data/*"],
    "nvidia/OpenScienceReasoning-2": ["train/*"],
    "codeparrot/apps": ["train.jsonl"],
}


def _hf_token() -> Optional[str]:
    """HF_TOKEN from the environment, else ~/Conscious_Engines/.env as a fallback. Never logged or written."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    envp = os.path.expanduser("~/Conscious_Engines/.env")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8"):
            if line.startswith("HF_TOKEN="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _resolved_revision(local_dir: str) -> Optional[str]:
    """The commit sha encoded in a snapshot_download path (.../snapshots/<sha>/...), if present."""
    for part in local_dir.replace("\\", "/").split("/"):
        if re.fullmatch(r"[0-9a-f]{40}", part):
            return part
    return None


# ---------------------------------------------------------------------------
# registry traversal
# ---------------------------------------------------------------------------
def iter_all_specs(only: Optional[Set[str]], include_optional: bool) -> Iterator[Tuple[str, dict]]:
    """Yield (kind, spec) for every SOURCES / EVAL_SETS / HELD_OUT entry `--only` / `--include-optional` admit."""
    for s in sources.SOURCES:
        if only is None or s["tag"] in only:
            yield "source", s
    for s in sources.EVAL_SETS:
        if only is not None and s["tag"] not in only:
            continue
        if s.get("optional") and not include_optional:
            continue
        yield "eval", s
    for s in sources.HELD_OUT:
        if only is None or s["tag"] in only:
            yield "held_out", s


# ---------------------------------------------------------------------------
# the two modes
# ---------------------------------------------------------------------------
def list_one(kind: str, spec: dict, token: Optional[str]) -> dict:
    """Metadata only (HfApi().dataset_info(files_metadata=True)) -- no bytes of the dataset are fetched."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        info = api.dataset_info(spec["repo"], files_metadata=True)
        total = sum((f.size or 0) for f in (info.siblings or []) if getattr(f, "size", None))
        return {"tag": spec["tag"], "kind": kind, "repo": spec["repo"], "status": "ok", "sha": info.sha,
                "n_files": len(info.siblings or []), "total_bytes": total, "total_gib": round(total / 2**30, 3)}
    except Exception as e:                                  # noqa: BLE001 -- one bad repo must not stop the listing
        return {"tag": spec["tag"], "kind": kind, "repo": spec["repo"], "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def download_one(kind: str, spec: dict, token: Optional[str]) -> dict:
    """snapshot_download one spec into the HF cache; -> the manifest entry for it."""
    from huggingface_hub import snapshot_download
    repo = spec["repo"]
    patterns = ALLOW_PATTERNS.get(repo)
    entry = {"tag": spec["tag"], "kind": kind, "repo": repo, "allow_patterns": patterns}
    try:
        local_dir = snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, token=token)
        entry.update(status="ok", local_dir=local_dir, revision=_resolved_revision(local_dir))
    except Exception as e:                                  # noqa: BLE001 -- one gated/missing repo must not abort the run
        entry.update(status="error", error=f"{type(e).__name__}: {e}")
    return entry


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"created_utc": None, "sources": {}}


def write_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated tags; default: every SOURCES/EVAL_SETS/HELD_OUT entry")
    ap.add_argument("--list", action="store_true",
                     help="metadata only (sizes + shas), no download -- the only mode safe to run off the training box")
    ap.add_argument("--include-optional", action="store_true", help="also fetch EVAL_SETS entries marked optional=True")
    ap.add_argument("--out", default=MANIFEST_PATH, help="manifest path (default data/c1b/sources_manifest.json)")
    args = ap.parse_args(argv)

    only = {t.strip() for t in args.only.split(",") if t.strip()} if args.only else None
    token = _hf_token()
    specs = list(iter_all_specs(only, args.include_optional))
    if not specs:
        print(f"[download] --only {args.only!r} matched nothing", file=sys.stderr)
        return 1
    print(f"[download] {len(specs)} spec(s){' (--list, metadata only)' if args.list else ''}"
          f"{'  [HF_TOKEN set]' if token else '  [no HF_TOKEN]'}", flush=True)

    if args.list:
        total_gib = 0.0
        for kind, s in specs:
            r = list_one(kind, s, token)
            if r["status"] == "ok":
                total_gib += r["total_gib"]
                print(f"[list] {r['tag']:16s} {kind:9s} {r['repo']:42s} {r['total_gib']:9.3f} GiB "
                      f"{r['n_files']:5d} files  sha {r['sha'][:10]}", flush=True)
            else:
                print(f"[list] {r['tag']:16s} {kind:9s} {r['repo']:42s} ERROR {r['error']}", flush=True)
        print(f"[list] total (of the entries listed above): {total_gib:.2f} GiB", flush=True)
        return 0

    entries = []
    for kind, s in specs:
        print(f"[download] {s['tag']} ({kind}) <- {s['repo']} ...", flush=True)
        e = download_one(kind, s, token)
        entries.append(e)
        tail = f" rev={e['revision']}" if e.get("revision") else ""
        tail += f" ERROR {e['error']}" if e.get("error") else ""
        print(f"[download]   {e['status']}{tail}", flush=True)

    manifest = load_manifest(args.out)
    manifest.setdefault("sources", {}).update({e["tag"]: e for e in entries})
    manifest["updated_utc"] = _utc_now()
    manifest.setdefault("created_utc", manifest["updated_utc"])
    write_manifest(args.out, manifest)

    n_ok = sum(1 for e in entries if e["status"] == "ok")
    n_err = len(entries) - n_ok
    print(f"[download] wrote {args.out}: {n_ok} ok, {n_err} error (this run); "
          f"{len(manifest['sources'])} tags total in the manifest", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
