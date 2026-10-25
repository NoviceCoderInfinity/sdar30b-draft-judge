#!/usr/bin/env python
"""Orchestrator CLI for the 1B corpus prompt lists (docs/CORPUS_1B_BUILD.md).

Per domain, in the fixed order chat, maths, science, code, if (IF's composer draws on the chat domain's
final prompt set to stay disjoint from it -- chat must finish first):
  1. run every sources.py source's extractor ("module:function", imported dynamically -- a domain whose
     extractor module does not exist yet (code -> extract_code.py, if -> extract_if.py / compose_if.py,
     both owned by other agents) fails with a message naming the missing file, not a bare ImportError)
  2. token-length filter: len(research_protocol.encode_prompt(tok, prompt)) <= --max-prompt-tokens (800),
     tokenizer = the b32 tokenizer loaded offline via research_protocol.MODELS; `--no-tokenizer` skips this
     (for tests -- there is no cached 30B tokenizer in a lightweight test environment)
  3. decontam.DecontamIndex.flag_batch over the WHOLE domain's surviving rows at once (one containment pass
     per domain, not per source -- see decontam.py's docstring for why that matters at scale)
  4. dedupe.exact_dedupe then dedupe.minhash_dedupe (--workers parallelises the MinHash-construction step)
  5. fill bucket quotas: target = sources.QUOTAS[domain][bucket] * common.TARGET_PROMPTS[domain] rows,
     each source capped at its own sources.py `quota` (applied across ALL of that source's rows, by
     ascending common.order_key(pid), before bucketing -- a source whose rows span two buckets, e.g. osr2's
     mc/free split, must not have its cap silently applied to only one of them); a bucket left short after
     that draws surplus rows from sources.FILL_ORDER[domain], in the listed order
  6. wave = common.wave_of(pid); write data/c1b/prompt_lists/<domain>.jsonl and .../manifest.json (merged
     across runs, like download.py's sources_manifest.json, so `--domains chat` does not erase other
     domains' entries)

TODO (left for the GPU box, per the 2026-09-28 build session -- not runnable/testable on the dev Mac):
  - This has been unit-tested with tiny monkeypatched registries and --no-tokenizer only; it has never run
    against the real HF cache, the real b32 tokenizer, or a full-size domain. Run smoke tests with
    --limit-per-source small first.
  - --workers only parallelises dedupe.minhash_dedupe (extraction is sequential across sources). A
    ProcessPoolExecutor over sources.sources_for(domain), one process per source doing extraction + the
    token-length filter, is the natural next step if extraction throughput matters -- deferred for time,
    not attempted-and-broken.
  - The quota/FILL_ORDER fill algorithm below is implemented exactly as specified and unit-tested on
    synthetic data, but its behaviour on the real skewed yields (e.g. nemosci_mcq's ~45% boxed-answer drop,
    osr2 being almost all 'mc') has not been observed.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402
import sources                                               # noqa: E402
import decontam                                                # noqa: E402
import dedupe                                                    # noqa: E402
import research_protocol as P                                     # noqa: E402  (sets HF_*_OFFLINE=1 on import)

DOMAIN_ORDER = ["chat", "maths", "science", "code", "if"]
SOURCES_MANIFEST_PATH = os.path.join(common.DATA_DIR, "sources_manifest.json")

# module -> which agent owns it, for the "module missing" error (see module docstring).
_AGENT_OWNED = {"extract_code": "extract_code.py", "extract_if": "extract_if.py", "compose_if": "compose_if.py",
                "checkers": "checkers.py", "eval_sets": "eval_sets.py"}

_TOK = None


def _get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        m = P.MODELS["b32"]
        kw = {"trust_remote_code": True, "local_files_only": True, "revision": m["revision"]}
        if m["cache_dir"]:
            kw["cache_dir"] = m["cache_dir"]
        _TOK = AutoTokenizer.from_pretrained(m["repo"], **kw)
    return _TOK


def _import_extractor(extractor: str, tag: str, domain: str):
    module_name, func_name = extractor.split(":")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        owner = _AGENT_OWNED.get(module_name, f"{module_name}.py")
        raise SystemExit(f"[build_prompt_lists] domain '{domain}' source '{tag}' needs {owner}, which is not "
                          f"importable yet ({type(e).__name__}: {e}). That file is owned by another agent; "
                          f"this run cannot proceed past it.")
    try:
        return getattr(mod, func_name)
    except AttributeError as e:
        raise SystemExit(f"[build_prompt_lists] {module_name}.py has no function '{func_name}' (needed by "
                          f"domain '{domain}' source '{tag}'): {e}")


def extract_source(spec: dict, domain: str, limit: Optional[int], max_prompt_tokens: int, use_tokenizer: bool,
                    exclude_norm: Optional[Set[str]] = None) -> Tuple[List[dict], dict]:
    """Run one source's extractor to completion, apply the token-length filter. -> (rows, stats)."""
    module_name, func_name = spec["extractor"].split(":")
    fn = _import_extractor(spec["extractor"], spec["tag"], domain)
    gen = fn(spec, exclude_norm=exclude_norm) if module_name == "compose_if" else fn(spec)
    if limit is not None:
        gen = itertools.islice(gen, limit)
    tok = _get_tokenizer() if use_tokenizer else None
    kept, yielded, dropped_len = [], 0, 0
    for row in gen:
        yielded += 1
        if tok is not None:
            n = int(P.encode_prompt(tok, row["prompt"]).shape[1])
            if n > max_prompt_tokens:
                dropped_len += 1
                continue
        assert "bucket" in row["meta"], row
        kept.append(row)
    return kept, {"yielded": yielded, "dropped_token_length": dropped_len, "kept_after_length": len(kept)}


# ---------------------------------------------------------------------------
# quota filling
# ---------------------------------------------------------------------------
def fill_quotas(domain: str, rows: List[dict]) -> Tuple[List[dict], dict]:
    quotas = sources.QUOTAS[domain]
    fill_order = sources.FILL_ORDER.get(domain, [])
    target_total = common.TARGET_PROMPTS[domain]
    source_quota = {s["tag"]: s.get("quota") for s in sources.sources_for(domain)}

    # 1) cap each SOURCE's total contribution (across every bucket it feeds) by ascending order_key(pid)
    by_source: Dict[str, List[dict]] = {}
    for r in rows:
        by_source.setdefault(common.source_tag_of(r["pid"]), []).append(r)
    capped: List[dict] = []
    for tag, rs in by_source.items():
        rs.sort(key=lambda r: common.order_key(r["pid"]))
        cap = source_quota.get(tag)
        capped.extend(rs if cap is None else rs[:cap])

    # 2) bucket the capped rows by each row's own meta["bucket"], sorted by order_key(pid)
    pools: Dict[str, List[dict]] = {b: [] for b in quotas}
    for r in capped:
        pools.setdefault(r["meta"]["bucket"], []).append(r)
    for b in pools:
        pools[b].sort(key=lambda r: common.order_key(r["pid"]))

    targets = {b: round(q * target_total) for b, q in quotas.items()}
    assigned: Dict[str, List[dict]] = {}
    for b in quotas:
        take = pools.get(b, [])[:targets[b]]
        assigned[b] = take
        pools[b] = pools.get(b, [])[len(take):]

    # 3) a short bucket draws surplus (already-qualified) rows from FILL_ORDER, in order
    for b in quotas:
        short = targets[b] - len(assigned[b])
        for lender in fill_order:
            if short <= 0:
                break
            if lender == b or lender not in pools:
                continue
            lend = pools[lender][:short]
            if not lend:
                continue
            assigned[b].extend(lend)
            pools[lender] = pools[lender][len(lend):]
            short -= len(lend)

    final_rows = [r for b in quotas for r in assigned[b]]
    stats = {"targets": targets, "taken": {b: len(assigned[b]) for b in quotas},
             "shortfall": {b: max(0, targets[b] - len(assigned[b])) for b in quotas}}
    return final_rows, stats


# ---------------------------------------------------------------------------
# per-domain pipeline
# ---------------------------------------------------------------------------
def build_domain(domain: str, limit_per_source: Optional[int], max_prompt_tokens: int, use_tokenizer: bool,
                 workers: int, check_containment: bool, exclude_norm: Optional[Set[str]]) -> Tuple[List[dict], dict]:
    manifest: dict = {"sources": {}, "buckets": {}, "wave_counts": {}}
    all_rows: List[dict] = []
    for spec in sources.sources_for(domain):
        rows, stats = extract_source(spec, domain, limit_per_source, max_prompt_tokens, use_tokenizer, exclude_norm)
        manifest["sources"][spec["tag"]] = stats
        all_rows.extend(rows)

    prompts = [r["prompt"] for r in all_rows]
    di = decontam.DecontamIndex.build()
    di.positive_control()
    verdicts = di.flag_batch(prompts, check_containment=check_containment)
    survivors, decontam_reasons = [], {}
    for r, v in zip(all_rows, verdicts):
        if v is None:
            survivors.append(r)
        else:
            reason = v.split(":", 1)[0]
            tag = common.source_tag_of(r["pid"])
            decontam_reasons.setdefault(tag, {}).setdefault(reason, 0)
            decontam_reasons[tag][reason] += 1
    for tag, d in decontam_reasons.items():
        manifest["sources"].setdefault(tag, {})["dropped_decontam"] = d

    exact_kept, _ = dedupe.exact_dedupe(survivors)
    exact_dropped_pids = {r["pid"] for r in survivors} - {r["pid"] for r in exact_kept}
    final_kept, _ = dedupe.minhash_dedupe(exact_kept, workers=workers)
    minhash_dropped_pids = {r["pid"] for r in exact_kept} - {r["pid"] for r in final_kept}
    for pid in exact_dropped_pids:
        tag = common.source_tag_of(pid)
        manifest["sources"].setdefault(tag, {})["dropped_exact_dup"] = \
            manifest["sources"].setdefault(tag, {}).get("dropped_exact_dup", 0) + 1
    for pid in minhash_dropped_pids:
        tag = common.source_tag_of(pid)
        manifest["sources"].setdefault(tag, {})["dropped_minhash_dup"] = \
            manifest["sources"].setdefault(tag, {}).get("dropped_minhash_dup", 0) + 1

    final_rows, bucket_stats = fill_quotas(domain, final_kept)
    for r in final_rows:
        r["wave"] = common.wave_of(r["pid"])
    kept_pids = {r["pid"] for r in final_rows}
    for tag in manifest["sources"]:
        manifest["sources"][tag]["kept"] = sum(1 for r in final_rows if common.source_tag_of(r["pid"]) == tag)
    manifest["buckets"] = bucket_stats
    manifest["wave_counts"] = {"A": sum(1 for r in final_rows if r["wave"] == "A"),
                               "B": sum(1 for r in final_rows if r["wave"] == "B")}
    manifest["n_candidates_after_extraction"] = len(all_rows)
    manifest["n_after_decontam"] = len(survivors)
    manifest["n_after_dedupe"] = len(final_kept)
    manifest["n_final"] = len(final_rows)
    return final_rows, manifest


# ---------------------------------------------------------------------------
def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default=",".join(DOMAIN_ORDER), help="comma-separated, kept in the fixed order")
    ap.add_argument("--max-prompt-tokens", type=int, default=800)
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the token-length filter (tests: no cached tokenizer)")
    ap.add_argument("--limit-per-source", type=int, default=None, help="cap rows read per source (smoke tests)")
    ap.add_argument("--workers", type=int, default=1, help="dedupe.minhash_dedupe's MinHash-construction pool size")
    ap.add_argument("--no-decontam-containment", action="store_true",
                     help="skip decontam.py's containment pass (exact + 13-gram only) -- see decontam.py's docstring")
    ap.add_argument("--out", default=common.PROMPT_DIR, help="output directory (default data/c1b/prompt_lists)")
    args = ap.parse_args(argv)

    wanted = [d.strip() for d in args.domains.split(",") if d.strip()]
    domains = [d for d in DOMAIN_ORDER if d in wanted]
    unknown = set(wanted) - set(DOMAIN_ORDER)
    if unknown:
        print(f"[build_prompt_lists] unknown domain(s) {sorted(unknown)}; valid: {DOMAIN_ORDER}", file=sys.stderr)
        return 1

    src_manifest = _load_json(SOURCES_MANIFEST_PATH).get("sources", {})
    out_manifest_path = os.path.join(args.out, "manifest.json")
    built: Dict[str, dict] = {}

    exclude_norm: Optional[Set[str]] = None
    for domain in domains:
        print(f"[build_prompt_lists] domain={domain} ...", flush=True)
        if domain == "if" and exclude_norm is None:
            # `--domains if` in a later run than chat: IF must still be disjoint from the chat prompt list on disk
            chat_path = os.path.join(args.out, "chat.jsonl")
            if not os.path.exists(chat_path):
                print(f"[build_prompt_lists] if needs {chat_path} (build chat first)", file=sys.stderr)
                return 1
            exclude_norm = {common.normalize_text(r["prompt"]) for r in common.read_jsonl(chat_path)}
        rows, dmanifest = build_domain(domain, args.limit_per_source, args.max_prompt_tokens,
                                       not args.no_tokenizer, args.workers, not args.no_decontam_containment,
                                       exclude_norm if domain == "if" else None)
        # third-party credential-like strings (tokens users pasted into public chat logs) never enter a list: the
        # public backup would republish them (sanitize_lists.py applies the same rule to lists built before this)
        from sanitize_lists import RX as _CRED
        cred = [r["pid"] for r in rows if _CRED.search(json.dumps(r, ensure_ascii=False))]
        if cred:
            rows = [r for r in rows if r["pid"] not in set(cred)]
        dmanifest["dropped_credential_like"] = cred
        out_path = os.path.join(args.out, f"{domain}.jsonl")
        common.write_jsonl(out_path + ".tmp", rows)                  # readers never see a half-written list
        os.replace(out_path + ".tmp", out_path)
        dmanifest["output"] = {"path": out_path, "sha256": common.sha256_file(out_path), "n_rows": len(rows)}
        dmanifest["revisions"] = {tag: src_manifest.get(tag, {}).get("revision")
                                  for tag in dmanifest["sources"]}
        built[domain] = dmanifest
        print(f"[build_prompt_lists]   wrote {out_path} ({len(rows)} rows)", flush=True)

        if domain == "chat":
            exclude_norm = {common.normalize_text(r["prompt"]) for r in rows}

    # domain groups may build in parallel processes (setup_box.sh): merge into the manifest on disk under a
    # lock, re-read at write time, so one process never erases another's domains
    import fcntl
    os.makedirs(args.out, exist_ok=True)
    with open(out_manifest_path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load_json(out_manifest_path)
        manifest.setdefault("domains", {}).update(built)
        manifest["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest.setdefault("created_utc", manifest["updated_utc"])
        manifest["params"] = {"max_prompt_tokens": args.max_prompt_tokens, "tokenizer": not args.no_tokenizer}
        with open(out_manifest_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(out_manifest_path + ".tmp", out_manifest_path)
    print(f"[build_prompt_lists] wrote {out_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
