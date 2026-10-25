#!/usr/bin/env python
"""Compact training set from DA-decode / stock trajectories (TRAINER_SPEC_v0 section 4).

Two source formats are read:

  DA-decode   `code/draftalign/run_da_decode.py`
              <dir>/<cell>.<split>.shard*of*.jsonl        (make_record, da_decode.py:454)
              <dir>/traces/<split>/<idx>.npz              (save_trace,  da_decode.py:526)
              -> tokens[b,32] valid[b,32] committed[b,p,32] draft_tokens[b,32] accepted[b,32]

  stock v1    `code/run_research_baselines.py`
              <dir>/<cell>.<split>.shard*of*.jsonl
              <traces>/<cell>/<idx>.npz                  (save_trace, run_research_baselines.py:100)
              -> the same arrays WITHOUT draft_tokens. Drafts may be supplied from a
                 separate NPZ (`--stock-drafts DIR`, one <idx>.npz per item with a
                 `draft_tokens[b,32]` array); where they are absent the block is
                 usable only in the all-mask (stock) regime.

Output: one npz per item under <out>/items/<split>__<idx>.npz plus <out>/index.jsonl.
The npz carries everything `dataset.StateSampler` needs and nothing else:

    prompt_ids[P] tokens[nb,32] valid[nb,32] committed[nb,P_max,32] n_pass[nb]
    draft[nb,32] has_draft[nb] accepted[nb,32] usable[nb] block_start[nb]
    prompt_length block_length correct idx

Filters: --only-correct, --max-seq (a BLOCK is usable iff its hi <= max-seq, so a long
item still contributes its early blocks), the 100 frozen train prompts
(research_protocol.calibration_items()/validation_items()), and every test-split record.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402  (sets HF offline env)
from research_protocol import safe_out                                   # noqa: E402

import numpy as np                                                       # noqa: E402

BLOCK = 32
TEST_SPLITS = ("test",)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def read_records(rec_dir: str, pattern: str = "*.jsonl") -> List[dict]:
    """Every jsonl record under `rec_dir` (shards concatenated, later duplicates dropped)."""
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(rec_dir, pattern))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:                                            # truncated last line
                continue
            key = (r.get("split"), int(r["idx"]))
            if key in seen:
                continue
            seen.add(key)
            r["_source_file"] = os.path.basename(f)
            out.append(r)
    return out


def trace_path(trace_root: str, split_dir: str, idx: int, layout: str) -> str:
    """DA-decode namespaces traces by the RUN's split name (run_da_decode.py:239 uses
    `--items`, e.g. "calibvalid", which is not the record's own `split` field, "train");
    the older stock runner writes them flat under <traces>/<cell>/."""
    name = f"{idx:05d}.npz"
    return os.path.join(trace_root, split_dir, name) if layout == "da" else os.path.join(trace_root, name)


def frozen_train_idxs() -> set:
    """The 100 frozen GSM8K TRAIN prompts (calibration + validation), by split index."""
    return {int(it["idx"]) for it in (P.calibration_items() + P.validation_items())}


# ---------------------------------------------------------------------------
# one item
# ---------------------------------------------------------------------------
def build_item(rec: dict, trace: dict, prompt_ids: Sequence[int], drafts: Optional[np.ndarray] = None,
               max_seq: Optional[int] = None, block: int = BLOCK) -> dict:
    """Compact arrays for one (record, trace) pair.  Raises AssertionError on inconsistency."""
    tokens = np.asarray(trace["tokens"], np.int32)
    valid = np.asarray(trace["valid"], bool)
    committed = np.asarray(trace["committed"], bool)
    n_pass = np.asarray(trace["n_pass"], np.int32)
    start = np.asarray(trace["block_start"], np.int32)
    pl = int(trace["prompt_length"])
    bl = int(trace["block_length"])
    assert bl == block, f"trace block_length {bl} != {block}"
    nb = tokens.shape[0]
    assert valid.shape == (nb, bl) and committed.shape[0] == nb and committed.shape[2] == bl

    prompt_ids = np.asarray(prompt_ids, np.int32)
    assert prompt_ids.shape[0] == pl, f"prompt has {prompt_ids.shape[0]} ids, trace says {pl}"
    lo0 = int(start[0])
    assert lo0 == (pl // bl) * bl, f"first block starts at {lo0}, prompt_length {pl}"
    # the prompt tail lives inside block 0 and must be the prompt's own ids
    tail = pl - lo0
    if tail:
        assert np.array_equal(tokens[0, :tail], prompt_ids[lo0:pl]), "block 0 prompt tail != prompt ids"
    assert np.array_equal(valid[0][:tail], np.zeros(tail, bool)), "prompt-tail slots must be invalid"
    assert np.array_equal(start, lo0 + bl * np.arange(nb, dtype=np.int32)), "blocks are not contiguous"

    # committed slots must hold the final token (they are never rewritten)
    for b in range(nb):
        cum = committed[b, :int(n_pass[b])].any(0)
        assert not bool((cum & ~valid[b]).any()), f"block {b}: a non-valid slot was committed"

    if drafts is None:
        draft = np.asarray(trace.get("draft_tokens", np.full((nb, bl), -1, np.int32)), np.int32)
    else:
        draft = np.asarray(drafts, np.int32)
        assert draft.shape == (nb, bl), f"drafts {draft.shape} != {(nb, bl)}"
    # a block has a usable draft only if EVERY generated slot carries one
    has_draft = np.array([bool(valid[b].any() and (draft[b][valid[b]] >= 0).all()) for b in range(nb)], bool)

    hi = start + bl
    usable = np.ones(nb, bool) if max_seq is None else (hi <= int(max_seq))
    usable &= valid.any(1)                                     # a block with no generated slot is useless

    accepted = np.asarray(trace.get("accepted", np.zeros((nb, bl), bool)), bool)
    return {
        "prompt_ids": prompt_ids, "tokens": tokens, "valid": valid, "committed": committed,
        "n_pass": n_pass, "block_start": start, "draft": draft, "has_draft": has_draft,
        "accepted": accepted, "usable": usable,
        "prompt_length": np.int32(pl), "block_length": np.int32(bl),
        "correct": np.int8(-1 if rec.get("correct") is None else int(rec["correct"])),
        "idx": np.int32(int(rec["idx"])),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build(records: List[dict], trace_root: str, out_dir: str, *, layout: str = "da",
          prompt_ids_fn: Callable[[dict], Sequence[int]], source: str = "da",
          only_correct: bool = False, max_seq: Optional[int] = 1200,
          exclude_idxs: Optional[set] = None, drafts_dir: Optional[str] = None,
          trace_split: Optional[str] = None, block: int = BLOCK, verbose: bool = True) -> dict:
    """Write one npz per kept item + append to <out_dir>/index.jsonl. Returns a summary."""
    os.makedirs(os.path.join(out_dir, "items"), exist_ok=True)
    index_path = os.path.join(out_dir, "index.jsonl")
    kept, drop = [], {"test_split": 0, "frozen": 0, "incorrect": 0, "no_trace": 0,
                      "bad": 0, "no_usable_block": 0}
    exclude_idxs = exclude_idxs or set()
    with open(index_path, "a") as fh:
        for rec in records:
            split = rec.get("split", "?")
            idx = int(rec["idx"])
            if split in TEST_SPLITS:
                drop["test_split"] += 1; continue
            if split in ("train", "train_pool") and idx in exclude_idxs:
                drop["frozen"] += 1; continue
            if only_correct and not rec.get("correct"):
                drop["incorrect"] += 1; continue
            tp = trace_path(trace_root, trace_split or split, idx, layout)
            if not os.path.exists(tp):
                drop["no_trace"] += 1; continue
            trace = {k: v for k, v in np.load(tp).items()}
            drafts = None
            if drafts_dir:
                dp = os.path.join(drafts_dir, f"{idx:05d}.npz")
                if os.path.exists(dp):
                    drafts = np.load(dp)["draft_tokens"]
            try:
                item = build_item(rec, trace, prompt_ids_fn(rec), drafts, max_seq, block)
            except AssertionError as e:
                drop["bad"] += 1
                if verbose:
                    print(f"[skip] {split}/{idx}: {e}", flush=True)
                continue
            if not bool(item["usable"].any()):
                drop["no_usable_block"] += 1; continue
            name = f"{split}__{idx:05d}.npz"
            np.savez_compressed(os.path.join(out_dir, "items", name), **item)
            entry = {"file": os.path.join("items", name), "idx": idx, "split": split, "source": source,
                     "cell": rec.get("cell"), "config_hash": rec.get("config_hash"),
                     "correct": None if rec.get("correct") is None else int(rec["correct"]),
                     "n_blocks": int(item["tokens"].shape[0]),
                     "usable_blocks": int(item["usable"].sum()),
                     "draft_blocks": int((item["has_draft"] & item["usable"]).sum()),
                     # per-block metadata so StateSampler can enumerate examples without
                     # opening every npz
                     "usable": [bool(v) for v in item["usable"]],
                     "has_draft": [bool(v) for v in item["has_draft"]],
                     "block_start": [int(v) for v in item["block_start"]],
                     "seq_len": int(item["block_start"][-1]) + block,
                     "prompt_length": int(item["prompt_length"])}
            fh.write(json.dumps(entry) + "\n")
            kept.append(entry)
    summary = {"out": out_dir, "kept_items": len(kept), "dropped": drop,
               "usable_blocks": sum(k["usable_blocks"] for k in kept),
               "draft_blocks": sum(k["draft_blocks"] for k in kept)}
    if verbose:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def tokenizer_prompt_ids(split_name: str, model_key: str = "b32") -> Callable[[dict], Sequence[int]]:
    """idx -> `research_protocol.encode_prompt` ids, for the split that produced these records."""
    from transformers import AutoTokenizer
    m = P.MODELS[model_key]
    kw = {"trust_remote_code": True, "local_files_only": True, "revision": m["revision"]}
    if m["cache_dir"]:
        kw["cache_dir"] = m["cache_dir"]
    tok = AutoTokenizer.from_pretrained(m["repo"], **kw)
    items = {int(it["idx"]): it for it in P.items_for_split(split_name)}
    cache: Dict[int, List[int]] = {}

    def fn(rec: dict) -> Sequence[int]:
        idx = int(rec["idx"])
        if idx not in cache:
            cache[idx] = P.encode_prompt(tok, items[idx]["prompt"])[0].tolist()
        return cache[idx]
    return fn


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True,
                    help="directory of <cell>.<split>.shard*.jsonl records")
    ap.add_argument("--records-glob", default="*.jsonl",
                    help="glob inside --records; NEEDED when one directory holds several cells "
                         "(e.g. 'gsm8k_B32_t95_stock.*.jsonl')")
    ap.add_argument("--traces", required=True, help="trace root (DA: contains <split>/<idx>.npz)")
    ap.add_argument("--layout", choices=("da", "stock"), default="da",
                    help="da = traces/<split>/<idx>.npz (run_da_decode); stock = traces/<idx>.npz")
    ap.add_argument("--split-name", required=True,
                    help="research_protocol split the records came from (calibvalid, gsm8k_pool:0:2000, ...)")
    ap.add_argument("--trace-split", default=None,
                    help="trace subdirectory for --layout da (default: --split-name)")
    ap.add_argument("--stock-drafts", default=None,
                    help="optional directory of <idx>.npz with draft_tokens[nb,32] for --layout stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=None, help="label written into the index (default: --layout)")
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--max-seq", type=int, default=1200,
                    help="a block is usable only if its end position is <= this (0 disables)")
    ap.add_argument("--exclude-frozen", dest="exclude_frozen", action="store_true", default=True,
                    help="drop the 100 frozen train prompts (default)")
    ap.add_argument("--no-exclude-frozen", dest="exclude_frozen", action="store_false")
    ap.add_argument("--model", default="b32")
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    out = safe_out(args.out)
    records = read_records(safe_out(args.records), args.records_glob)
    excl = frozen_train_idxs() if args.exclude_frozen else set()
    print(f"[plan] {len(records)} records, {len(excl)} frozen idxs excluded, layout {args.layout}", flush=True)
    summary = build(records, safe_out(args.traces), out, layout=args.layout,
                    prompt_ids_fn=tokenizer_prompt_ids(args.split_name, args.model),
                    source=args.source or args.layout, only_correct=args.only_correct,
                    max_seq=args.max_seq or None, exclude_idxs=excl,
                    drafts_dir=safe_out(args.stock_drafts) if args.stock_drafts else None,
                    trace_split=args.trace_split or args.split_name)
    with open(os.path.join(out, "build_summary.json"), "w") as fh:
        json.dump({"args": vars(args), "summary": summary}, fh, indent=2)
    return summary


if __name__ == "__main__":
    main()
