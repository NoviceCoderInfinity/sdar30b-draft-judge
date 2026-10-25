#!/usr/bin/env python
"""Fresh prompt lists for wave C: prompts that Phase 1 (waves A and B) never decoded.

  maths, code, chat  rows of the existing lists (--lists, data/c1b/prompt_lists/<domain>.jsonl) whose pid is in
                     no --done file (a decoded pid = finished, kept or dropped), in common.order_key(pid) order; the
                     first --targets[domain] of them, unchanged except wave = "C". The lists were filtered for
                     length, decontaminated and deduped when they were built, so none of that runs again.
  if                 NEW composed instruction-following prompts (the IF list was fully decoded, and compose_if's
                     WildChat pool is exhausted). Bases = the undecoded chat rows AFTER the chat target in the same
                     order, so fresh chat prompts and IF bases are disjoint (`--domains if` alone splits at the same
                     chat target). Walking the bases in order: compose_if._eligible_base(text, None), then 1-3
                     constraints from compose_if.pick_constraints; the row is assembled exactly as
                     compose_if.composed does (source tag "ifchat", pid = make_pid("if", "ifchat", source,
                     source_id), meta.base_pid = the chat row's pid); then, cheapest first, the credential screen
                     (sanitize_lists.RX), exact dedupe on common.normalize_text(prompt), and the token-length filter
                     (research_protocol.encode_prompt with the b32 tokenizer <= --max-prompt-tokens). Stops at the
                     IF target or when the bases run out (the shortfall goes in the manifest). No decontam or
                     near-dedupe pass: the chat rows had both when the chat list was built, and the appended
                     constraint text adds nothing to match.

Every domain also drops credential-like rows (sanitize_lists.RX) while streaming; the lists were sanitized when
built, so this is a guard and normally 0. Lists are streamed, and maths/code keep only their target rows in
memory (a bounded heap); chat keeps all its undecoded rows when IF is built, since the IF bases come from them.

Writes <out>/<domain>.jsonl (.tmp then os.replace) and <out>/manifest.json (domain entries merged across runs).

    /venv/train/bin/python code/corpus1b/fresh_lists.py                     # maths, code, chat, if; default targets
    /venv/train/bin/python code/corpus1b/fresh_lists.py --domains if --targets if=90000
then wave C decodes them:
    gen_sglang.py --wave C --prompt-dir data/c1b/prompt_lists_fresh --extra-done 'logs/c1b/gen/wave[AB]/gpu*/done.txt'
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import heapq
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402
import sources                                              # noqa: E402
import build_prompt_lists as B                              # noqa: E402  (tokenizer loader, manifest reader)
import research_protocol as P                               # noqa: E402  (sets HF_*_OFFLINE=1 on import)
from sanitize_lists import RX as CRED                       # noqa: E402

DOMAINS = ("maths", "code", "chat", "if")                  # chat before if: the IF bases are chat rows
DEFAULT_DONE = "logs/c1b/gen/wave[AB]/gpu*/done.txt"   # never wave C's own done.txt: reruns stay repeatable
DEFAULT_OUT = os.path.join(common.DATA_DIR, "prompt_lists_fresh")
DEFAULT_TARGETS = "maths=68000,code=65000,chat=39000,if=79000"
WAVE = "C"
IF_TAG = "ifchat"
IFCOMP = sources.source("ifcomp")                           # composed IF rows: same gen_cap and bucket


def parse_targets(s: str) -> Dict[str, int]:
    """"maths=68000,if=79000" -> {"maths": 68000, "if": 79000}."""
    out = {}
    for part in s.split(","):
        if part.strip():
            d, n = part.split("=")
            out[d.strip()] = int(n)
    return out


def load_done(specs: Sequence[str]) -> Tuple[set, List[str]]:
    """Union of the pids (one per line) in every file matched by `specs` (paths or globs; relative = under the
    repo root, as gen_sglang.load_extra_done) -> (pids, files)."""
    pids, files = set(), []
    for spec in specs:
        for f in sorted(glob.glob(spec if os.path.isabs(spec) else os.path.join(common.ROOT, spec))):
            with open(f) as fh:
                pids.update(fh.read().split())
            files.append(f)
    return pids, files


def _stream_undecoded(path: str, done: set, stats: dict) -> Iterator[dict]:
    """Rows of one list whose pid is not in `done`, minus credential-like rows; counts into `stats`."""
    with common.open_text(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:                  # a truncated last line, as common.read_jsonl
                continue
            if r["pid"] in done:
                continue
            if CRED.search(line):
                stats["n_dropped_credential"] += 1
                continue
            stats["n_undecoded_available"] += 1
            yield r


def undecoded_sorted(path: str, done: set, keep: Optional[int]) -> Tuple[List[dict], dict]:
    """The undecoded rows of one list in order_key order: the first `keep` of them (a bounded heap, so a 3 GB list
    is never held whole), or all of them when keep is None -> (rows, stats)."""
    stats = {"n_undecoded_available": 0, "n_dropped_credential": 0}
    rows = _stream_undecoded(path, done, stats)
    key = lambda r: common.order_key(r["pid"])                 # noqa: E731
    if keep is None:
        return sorted(rows, key=key), stats
    out = heapq.nsmallest(keep, rows, key=key)
    for _ in rows:                              # nsmallest(0, ...) reads nothing; the counts cover the whole list
        pass
    return out, stats


# ---------------------------------------------------------------------------
# IF: new composed prompts on undecoded chat bases
# ---------------------------------------------------------------------------
def _if_tools():
    """compose_if's base screen and constraint picker, its id pool and conflict map, and build_if_pool's gold
    writer -> (eligible, pick, ids_pool, conflicts, dump_gold). The one seam the CPU tests replace where langdetect
    and the vendored IFEvalG deps are not installed."""
    import compose_if
    from draftalign import build_if_pool as bip
    return (compose_if._eligible_base, compose_if.pick_constraints, compose_if.admitted_ids(),
            compose_if.conflict_map(), bip.dump_gold)


def compose_fresh_if(bases: Sequence[dict], target: int, tok, max_prompt_tokens: int) -> Tuple[List[dict], dict]:
    """Walk the chat `bases` in order until `target` IF rows are built or the bases run out -> (rows, stats)."""
    eligible, pick, ids_pool, conflicts, dump_gold = _if_tools()
    stats = dict.fromkeys(("n_bases_walked", "n_eligible", "n_with_constraints", "n_dropped_credential",
                           "n_dropped_dup", "n_dropped_token_length"), 0)
    rows, seen = [], set()
    for base in bases:
        if len(rows) >= target:
            break
        stats["n_bases_walked"] += 1
        text = base["prompt"]
        if not eligible(text, None):
            continue
        stats["n_eligible"] += 1
        pid = common.make_pid("if", IF_TAG, base["source"], base["source_id"])
        picked = pick(pid, ids_pool, conflicts, text)
        if picked is None:
            continue
        stats["n_with_constraints"] += 1
        ids, kwargs, descriptions = picked
        prompt = text + "\n\n" + " ".join(descriptions)
        row = {"pid": pid, "domain": "if", "source": base["source"], "source_id": base["source_id"],
               "prompt": prompt, "gen_cap": IFCOMP["gen_cap"],
               "key": {"type": "ifeval", "gold": dump_gold(ids, kwargs)},
               "meta": {"bucket": IFCOMP["bucket"], "n_constraints": len(ids),
                        "base_prompt_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "base_pid": base["pid"]},
               "wave": WAVE}
        if CRED.search(json.dumps(row, ensure_ascii=False)):
            stats["n_dropped_credential"] += 1
            continue
        norm = common.normalize_text(prompt)
        if norm in seen:
            stats["n_dropped_dup"] += 1
            continue
        if tok is not None and int(P.encode_prompt(tok, prompt).shape[1]) > max_prompt_tokens:
            stats["n_dropped_token_length"] += 1
            continue
        seen.add(norm)
        rows.append(row)
    stats["pass_rate"] = round(len(rows) / stats["n_bases_walked"], 4) if stats["n_bases_walked"] else 0.0
    return rows, stats


# ---------------------------------------------------------------------------
def _write_list(out_dir: str, domain: str, rows: List[dict]) -> dict:
    path = os.path.join(out_dir, f"{domain}.jsonl")
    common.write_jsonl(path + ".tmp", rows)                  # readers never see a half-written list
    os.replace(path + ".tmp", path)
    return {"path": _rel(path), "sha256": common.sha256_file(path), "n_rows": len(rows)}


def _rel(path: str) -> str:
    p = os.path.abspath(path)
    return os.path.relpath(p, common.ROOT) if p.startswith(common.ROOT + os.sep) else p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", default=common.PROMPT_DIR, help="existing prompt lists (default data/c1b/prompt_lists)")
    ap.add_argument("--done", action="append", default=None, metavar="GLOB",
                    help=f"done.txt files of decoded pids, path or glob, repeatable (default '{DEFAULT_DONE}')")
    ap.add_argument("--allow-empty-done", action="store_true", help="run even if the done files hold no pid")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default data/c1b/prompt_lists_fresh)")
    ap.add_argument("--targets", default=DEFAULT_TARGETS,
                    help=f"prompts per domain (default {DEFAULT_TARGETS}); a domain left out keeps its default")
    ap.add_argument("--max-prompt-tokens", type=int, default=800)
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the IF token-length filter (tests)")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args(argv)

    wanted = [d.strip() for d in args.domains.split(",") if d.strip()]
    if set(wanted) - set(DOMAINS):
        print(f"[fresh] unknown domain(s) {sorted(set(wanted) - set(DOMAINS))}; valid: {list(DOMAINS)}", file=sys.stderr)
        return 1
    targets = parse_targets(DEFAULT_TARGETS)
    targets.update(parse_targets(args.targets))
    done_specs = args.done or [DEFAULT_DONE]
    done, done_files = load_done(done_specs)
    if not done and not args.allow_empty_done:
        print(f"[fresh] no decoded pid in {done_specs}: every prompt would count as fresh "
              f"(--allow-empty-done to run anyway)", file=sys.stderr)
        return 1
    print(f"[fresh] decoded: {len(done)} pids from {len(done_files)} done files", flush=True)
    os.makedirs(args.out, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    built: Dict[str, dict] = {}

    chat_sorted = chat_stats = None
    if "if" in wanted:                          # IF bases = the undecoded chat rows after the chat target
        chat_sorted, chat_stats = undecoded_sorted(os.path.join(args.lists, "chat.jsonl"), done, None)

    for domain in ("maths", "code", "chat"):
        if domain not in wanted:
            continue
        target = targets[domain]
        if domain == "chat" and chat_sorted is not None:
            rows, st = chat_sorted[:target], dict(chat_stats)
        else:
            rows, st = undecoded_sorted(os.path.join(args.lists, f"{domain}.jsonl"), done, target)
        rows = [dict(r, wave=WAVE) for r in rows]
        st.update(target=target, n_taken=len(rows), shortfall=max(0, target - len(rows)), created_utc=now)
        st["output"] = _write_list(args.out, domain, rows)
        built[domain] = st
        print(f"[fresh] {domain}: {st['n_undecoded_available']} undecoded, took {len(rows)}/{target} "
              f"(shortfall {st['shortfall']}) -> {st['output']['path']}", flush=True)

    if "if" in wanted:
        target = targets["if"]
        bases = chat_sorted[targets["chat"]:]
        tok = None if args.no_tokenizer else B._get_tokenizer()
        rows, ist = compose_fresh_if(bases, target, tok, args.max_prompt_tokens)
        st = {"target": target, "n_undecoded_available": len(bases), "n_taken": len(rows),
              "shortfall": max(0, target - len(rows)), "chat_target_skipped": targets["chat"], **ist,
              "created_utc": now}
        st["output"] = _write_list(args.out, "if", rows)
        built["if"] = st
        print(f"[fresh] if: {len(bases)} chat bases, walked {st['n_bases_walked']} (eligible {st['n_eligible']}, "
              f"with constraints {st['n_with_constraints']}; dropped credential {st['n_dropped_credential']}, "
              f"dup {st['n_dropped_dup']}, token length {st['n_dropped_token_length']}), took {len(rows)}/{target} "
              f"(shortfall {st['shortfall']}, pass rate {st['pass_rate']}) -> {st['output']['path']}", flush=True)

    man_path = os.path.join(args.out, "manifest.json")
    manifest = B._load_json(man_path)
    manifest.setdefault("domains", {}).update(built)
    manifest.update(created_utc=now, wave=WAVE, lists_dir=_rel(args.lists), n_decoded_pids=len(done),
                    done_specs=list(done_specs), done_files=[_rel(f) for f in done_files],
                    params={"targets": targets, "max_prompt_tokens": args.max_prompt_tokens,
                            "tokenizer": not args.no_tokenizer})
    with open(man_path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(man_path + ".tmp", man_path)
    print(f"[fresh] wrote {_rel(man_path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
