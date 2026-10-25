#!/usr/bin/env python
"""Drafter training set: the MoE's own generations, one npz per item.

Round 3 of the draft-aligned plan (EXPLAINER section 19: "train the drafter on the MoE's
outputs; raises the 55% agreement cap").  The targets are the BIG model's tokens, so this is
knowledge distillation on outputs -- the 1.7B dense drafter learns to write what
SDAR-30B-A3B-Chat-b32 would write.

Source: the DV-blind decode records of `code/draftalign/run_da_decode.py`

    logs/draftalign/datagen/<cell>/<cell>.<split>.shard*of*.jsonl      (da_decode.py:826-836)

Only the record's own fields are used -- `idx`, `split`, `correct`, `prompt_tokens`,
`gen_ids`, `n_gen_to_eos`.  The per-block NPZ traces `build_train_set.py` needs are NOT
required here: the drafter is trained on the finished text, not on the judge's states.

    gen_ids       the non-[MASK] generated tokens in absolute order, i.e. the tokens at
                  absolute positions [prompt_tokens, prompt_tokens + len(gen_ids))
                  (da_decode.py:815 `generated_ids()` = x[:, prompt_length:], masks dropped;
                  masks appear only in blocks that were never reached after an EOS stop)
    n_gen_to_eos  index of the first EOS + 1, so gen_ids[:n_gen_to_eos] ENDS with the EOS
                  (da_decode.py:816-817)

Filters mirror `build_train_set.build` exactly: `--only-correct`, the 100 frozen GSM8K train
prompts (`build_train_set.frozen_train_idxs`), every `split == "test"` record, and `--max-seq`
(a BLOCK is kept only if its end is <= max-seq, so a long item still contributes its early
blocks).  In addition the record's `prompt_tokens` is checked against
`research_protocol.encode_prompt` -- a mismatch means the record was made with a different
prompt and the item is dropped.

The absolute 32-grid is the decoder's (seqcache.py docstring, build_train_set.py:104-110):
the prompt's FULL blocks are prefilled, the ragged prompt tail shares the first response
block, so the first response block starts at `lo0 = (prompt_tokens // 32) * 32`.

Output
    <out>/index.jsonl                one line per item
    <out>/items/<source>__<split>__<idx>.npz   prompt_ids[P] int32, gen_ids[G] int32, n_blocks int32,
                                     block_start(lo0) int32, prompt_length, correct, idx
    <out>/build_summary.json

Usage
    python code/draftalign/build_drafter_set.py \
        --source logs/draftalign/datagen/gsm8k_pool_0_3000_dv-s8_blind_q50=gsm8k_pool:0:3000 \
        --source logs/draftalign/datagen/mbpp_pool_dv-s8_blind_q50=mbpp_pool \
        --out logs/draftalign/drafterset_union --only-correct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402  (sets HF offline env)
from research_protocol import safe_out                                   # noqa: E402

import numpy as np                                                       # noqa: E402

from draftalign.build_train_set import (TEST_SPLITS, frozen_train_idxs,  # noqa: E402
                                        read_records, tokenizer_prompt_ids)

BLOCK = 32
MAX_SEQ = 1200


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def grid(prompt_length: int, n_gen: int, max_seq: Optional[int] = MAX_SEQ,
         block: int = BLOCK) -> Dict[str, int]:
    """Block layout of one item on the ABSOLUTE `block`-grid.

    Returns lo0 (first response block start), n_blocks (blocks kept under `max_seq`),
    n_gen_kept (response tokens inside those blocks) and n_pad (slots of the last kept
    block that lie beyond the response -- filled with [MASK] and never a loss row).

    A block is kept iff its END is <= max_seq, the same rule as
    `build_train_set.build_item` (`usable = hi <= max_seq`).  When the cut removes the
    last block the response is truncated on the block boundary, so a truncated item has
    no padding at all.
    """
    assert prompt_length >= 0 and n_gen >= 0
    lo0 = (prompt_length // block) * block
    span = prompt_length - lo0 + n_gen                       # slots from lo0 to the end of the response
    n_all = (span + block - 1) // block if span > 0 else 0
    n_blocks = n_all
    if max_seq:
        n_blocks = min(n_all, max(0, (int(max_seq) - lo0) // block))
    if n_blocks <= 0:
        return {"lo0": lo0, "n_blocks": 0, "n_gen_kept": 0, "n_pad": 0, "seq_len": lo0}
    seq_len = lo0 + block * n_blocks
    n_gen_kept = min(n_gen, seq_len - prompt_length)
    return {"lo0": lo0, "n_blocks": n_blocks, "n_gen_kept": n_gen_kept,
            "n_pad": seq_len - prompt_length - n_gen_kept, "seq_len": seq_len}


# ---------------------------------------------------------------------------
# one item
# ---------------------------------------------------------------------------
def build_item(rec: dict, prompt_ids: Sequence[int], max_seq: Optional[int] = MAX_SEQ,
               block: int = BLOCK) -> dict:
    """Arrays for one record.  Raises AssertionError on any inconsistency."""
    prompt_ids = np.asarray(prompt_ids, np.int32)
    pl = int(rec["prompt_tokens"])
    assert prompt_ids.shape[0] == pl, f"prompt has {prompt_ids.shape[0]} ids, record says {pl}"

    gen = np.asarray(rec["gen_ids"], np.int32)
    n_eos = int(rec.get("n_gen_to_eos", gen.shape[0]))
    assert 0 <= n_eos <= gen.shape[0], f"n_gen_to_eos {n_eos} outside gen_ids[{gen.shape[0]}]"
    gen = gen[:n_eos]                                        # trimmed AT the EOS (inclusive)

    g = grid(pl, int(gen.shape[0]), max_seq, block)
    assert g["n_blocks"] > 0, f"no block fits under max_seq={max_seq} (prompt {pl})"
    gen = gen[:g["n_gen_kept"]]
    assert gen.shape[0] > 0, "no generated token left"
    return {
        "prompt_ids": prompt_ids, "gen_ids": gen,
        "n_blocks": np.int32(g["n_blocks"]), "block_start": np.int32(g["lo0"]),
        "prompt_length": np.int32(pl), "block_length": np.int32(block),
        "n_pad": np.int32(g["n_pad"]), "seq_len": np.int32(g["seq_len"]),
        "correct": np.int8(-1 if rec.get("correct") is None else int(rec["correct"])),
        "idx": np.int32(int(rec["idx"])),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build(records: List[dict], out_dir: str, *, prompt_ids_fn, source: str = "dv",
          only_correct: bool = True, max_seq: Optional[int] = MAX_SEQ,
          exclude_idxs: Optional[set] = None, require_eos: bool = False,
          block: int = BLOCK, verbose: bool = True) -> dict:
    """Write one npz per kept item + append to <out_dir>/index.jsonl.  Returns a summary."""
    os.makedirs(os.path.join(out_dir, "items"), exist_ok=True)
    index_path = os.path.join(out_dir, "index.jsonl")
    kept: List[dict] = []
    drop = {"test_split": 0, "frozen": 0, "incorrect": 0, "no_eos": 0, "empty": 0, "bad": 0}
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
            if require_eos and not rec.get("stopped_eos"):
                drop["no_eos"] += 1; continue
            if not rec.get("gen_ids"):
                drop["empty"] += 1; continue
            try:
                item = build_item(rec, prompt_ids_fn(rec), max_seq, block)
            except AssertionError as e:
                drop["bad"] += 1
                if verbose:
                    print(f"[skip] {split}/{idx}: {e}", flush=True)
                continue
            name = f"{source}__{split}__{idx:05d}.npz"      # source prefix: every source names its problems 0.. and all splits are train_pool
            np.savez_compressed(os.path.join(out_dir, "items", name), **item)
            entry = {"file": os.path.join("items", name), "idx": idx, "split": split,
                     "source": source, "cell": rec.get("cell"),
                     "config_hash": rec.get("config_hash"),
                     "correct": None if rec.get("correct") is None else int(rec["correct"]),
                     "prompt_length": int(item["prompt_length"]),
                     "n_gen": int(item["gen_ids"].shape[0]),
                     "n_blocks": int(item["n_blocks"]),
                     "block_start": int(item["block_start"]),
                     "n_pad": int(item["n_pad"]),
                     "seq_len": int(item["seq_len"]),
                     "truncated_by_max_seq": bool(int(item["n_pad"]) == 0
                                                  and int(rec.get("n_gen_to_eos", 0)) > int(item["gen_ids"].shape[0]))}
            fh.write(json.dumps(entry) + "\n")
            kept.append(entry)
    summary = {"out": out_dir, "kept_items": len(kept), "dropped": drop,
               "blocks": sum(k["n_blocks"] for k in kept),
               "gen_tokens": sum(k["n_gen"] for k in kept),
               "mean_blocks": round(sum(k["n_blocks"] for k in kept) / max(1, len(kept)), 2)}
    if verbose:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_source(spec: str):
    """'<dir>=<split-name>[=<glob>]' -> (dir, split_name, glob)."""
    parts = spec.split("=")
    assert len(parts) in (2, 3), f"--source wants <dir>=<split-name>[=<glob>], got {spec!r}"
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else "*.jsonl")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="DIR=SPLIT[=GLOB]",
                    help="record directory = research_protocol split name (repeatable), e.g. "
                         "logs/draftalign/datagen/mbpp_pool_dv-s8_blind_q50=mbpp_pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="b32",
                    help="tokenizer used to re-encode the prompts; MUST be the model that produced the "
                         "records (b32). The drafter d32 shares the chat template and MASK id 151669, "
                         "only the MASK token's SURFACE string differs (<|MASK|> vs <MASK>).")
    ap.add_argument("--only-correct", dest="only_correct", action="store_true", default=True)
    ap.add_argument("--no-only-correct", dest="only_correct", action="store_false")
    ap.add_argument("--require-eos", action="store_true", default=False,
                    help="additionally drop items that hit the generation cap without an EOS")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ,
                    help="a block is kept only if its end position is <= this (0 disables)")
    ap.add_argument("--exclude-frozen", dest="exclude_frozen", action="store_true", default=True)
    ap.add_argument("--no-exclude-frozen", dest="exclude_frozen", action="store_false")
    ap.add_argument("--block", type=int, default=BLOCK)
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    out = safe_out(args.out)
    excl = frozen_train_idxs() if args.exclude_frozen else set()
    summaries = []
    for spec in args.source:
        d, split_name, pat = parse_source(spec)
        records = read_records(safe_out(d), pat)
        print(f"[plan] {os.path.basename(d)}: {len(records)} records, split {split_name}", flush=True)
        summaries.append(build(records, out, prompt_ids_fn=tokenizer_prompt_ids(split_name, args.model),
                               source=os.path.basename(d.rstrip("/")), only_correct=args.only_correct,
                               max_seq=args.max_seq or None, exclude_idxs=excl,
                               require_eos=args.require_eos, block=args.block))
    total = {"kept_items": sum(s["kept_items"] for s in summaries),
             "blocks": sum(s["blocks"] for s in summaries),
             "gen_tokens": sum(s["gen_tokens"] for s in summaries),
             "per_source": summaries}
    with open(os.path.join(out, "build_summary.json"), "w") as fh:
        json.dump({"args": vars(args), "summary": total}, fh, indent=2)
    print(json.dumps({k: v for k, v in total.items() if k != "per_source"}, indent=2), flush=True)
    return total


if __name__ == "__main__":
    main()
