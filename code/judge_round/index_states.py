#!/usr/bin/env python
"""Judge round, piece A0: one validated, memory-mappable index of the judge's RECORDED per-pass states.

`judge_sglang.py --mode record` writes, per GPU, `<root>/gpu<k>/states.NNNNN.npz`: one compressed npz per chunk of
requests (layout: judge_sglang.py's docstring, built by its pack_states).  Training reads those states in a random
order over ~860k blocks, which per-chunk compressed files cannot serve (np.load decompresses a whole member per
access).  This script reads every COMPLETE chunk of one or more record roots once, validates it, and writes flat
.npy arrays that every trainer rank memory-maps through one page cache:

    /venv/p2/bin/python code/judge_round/index_states.py logs/phase5/judge_sglang/record \
        --index logs/c1b/trainset/fresh/index.jsonl --out logs/phase5/judge_states/fresh_rd1

REFUSES (StatesError, nothing written): a chunk whose layout or recorder invariants fail (list below); roots whose
run.json differ (another judge config, adapter or drafter: rd1 and rd2 records are separate indexes until the
operator fixes a mix); records made without the rejected-token ban unless --allow-no-ban (the decode under test
bans); a block size other than 32; with --index, a recorded block whose teacher tokens, start or count differ from
the trainset item's, or a trainset-valid slot the recorder did not treat as valid.

DROPS AND COUNTS: a rid already taken from an earlier chunk (a crash between judge_sglang's npz rename and its
done.txt append records the rid again in a later chunk; the first copy wins, sources in command-line order, gpu
dirs by number, chunks by number); with --index, a rid in no index row (--strict refuses instead); the crash
leftovers states.NNNNN.tmp.npz are never read.

CHECKS, per chunk, vectorised: item_block / blk_pass offsets monotone and complete; rid == the index items; blk_index
0..n-1 per item, blk_lo on the 32-grid and consecutive, blk_start the prompt tail in block 0 and 0 after; passes per
block == the n_judge counter; pass_no 0..n-1 per block; per pass: commit, tentative and remask partition open;
flag == open & (input != [MASK]); argmax >= 0, p in [0, 1] and q finite exactly at open slots (-1 / NaN elsewhere);
a forced pass commits exactly one slot; pass 0 is open exactly at the recorder-valid slots (slot >= blk_start) and
holds the first draft there, and its draft row is the first draft; between consecutive passes of a block,
open' = open & ~commit, input' = the decision applied (commit -> argmax, tentative -> draft, remask -> [MASK],
the rest unchanged) with the re-draft's tokens at its holes, draft' = draft with the re-draft at its holes; a
re-draft's holes are exactly the pass's remask slots and its input holds [MASK] exactly at holes + tentative slots;
rd_* are empty (-1 / NaN / 0 / False) on passes without a re-draft.  Counted, an error only with --strict: a
re-draft token equal to the banned (rejected) draft at its hole under redraft_ban.

OUTPUT (--out through research_protocol.safe_out; written to <out>.tmp, then renamed; --force replaces an old one):
    meta.json            version, the judge cfg, counter names, mask id, every source (root, run.json, chunks read)
    summary.json         items / blocks / passes (by domain and fresh / repeated with --index), passes and
                         re-drafts per block, forced passes, post-EOS passes, ban-unreachable holes, duplicates
    item_rid.npy         [I] str      item i's rid (== the trainset pid)
    item_block.npy       [I+1] int64  item i's blocks are rows item_block[i]:item_block[i+1]
    blk_*.npy            [NB, ...]    the chunk keys of the same names, dtypes unchanged
    blk_pass.npy         [NB+1] int64 block j's passes are rows blk_pass[j]:blk_pass[j+1]
    <pass key>.npy       [NP, ...]    pass_no, forced, input, draft, argmax, flag, open, commit, tentative, remask, p,
                                      q, rd_on, rd_holes, rd_tokens, rd_input, rd_p_commit, rd_n_fwd
Within an item, block rows are in block order (row item_block[i] + b holds trainset block b).

READ SIDE: StatesIndex(path) memory-maps the arrays; .block(pid, b) returns block b's record (RecBlock), or None.
CPU only, no model.  Tests: code/tests/test_index_states.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(CODE, "compat"), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VERSION = 1
BLOCK = 32
DEFAULT_MASK_ID = 151669          # research_protocol.MASK_ID (not imported here: the reader stays import-light)
BLK_KEYS = ("blk_index", "blk_lo", "blk_start", "blk_tokens", "blk_teacher", "blk_draft", "blk_p_draft",
            "blk_counters")
PASS_KEYS = ("pass_no", "forced", "input", "draft", "argmax", "flag", "open", "commit", "tentative", "remask", "p",
             "q", "rd_on", "rd_holes", "rd_tokens", "rd_input", "rd_p_commit", "rd_n_fwd")
OFFSET_KEYS = ("item_rid", "item_block", "blk_pass")
ALL_KEYS = OFFSET_KEYS + BLK_KEYS + PASS_KEYS
CHUNK_RE = re.compile(r"states\.(\d+)\.npz")
GPU_RE = re.compile(r"gpu(\d+)")


class StatesError(Exception):
    """A record that must not be trained on (the message names the file and the check)."""


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def find_chunks(root: str) -> List[Tuple[str, str]]:
    """[(gpu dir, chunk path)] of one record root: gpu<k> dirs by k, complete chunks by number (no .tmp.npz)."""
    if not os.path.isdir(root):
        raise StatesError(f"{root}: not a directory")
    gdirs = sorted((d for d in os.listdir(root) if GPU_RE.fullmatch(d) and os.path.isdir(os.path.join(root, d))),
                   key=lambda d: int(GPU_RE.fullmatch(d).group(1)))
    out = []
    for g in gdirs:
        gd = os.path.join(root, g)
        names = sorted((f for f in os.listdir(gd) if CHUNK_RE.fullmatch(f)),
                       key=lambda f: int(CHUNK_RE.fullmatch(f).group(1)))
        out += [(gd, os.path.join(gd, f)) for f in names]
    return out


def read_run_json(gpu_dir: str) -> dict:
    path = os.path.join(gpu_dir, "run.json")
    if not os.path.isfile(path):
        raise StatesError(f"{gpu_dir}: no run.json (judge_sglang.py writes it before the first chunk)")
    return json.load(open(path))


def chunk_index(z) -> dict:
    return json.loads(str(z["index"]))


# ---------------------------------------------------------------------------
# validation of one chunk (raises StatesError)
# ---------------------------------------------------------------------------
def check_chunk(a: Dict[str, np.ndarray], index: dict, mask_id: int, where: str) -> Dict[str, int]:
    """Every layout and recorder invariant of one states chunk (module docstring); returns soft counts."""
    def bad(msg: str):
        raise StatesError(f"{where}: {msg}")

    if index.get("version") != 1:
        bad(f"index version {index.get('version')!r}, expected 1")
    counters = list(index.get("counters") or [])
    if "n_judge" not in counters:
        bad("no n_judge counter in the index")
    items = index.get("items") or []
    rid = [str(x) for x in np.asarray(a["rid"]).tolist()]
    if [str(x["rid"]) for x in items] != rid:
        bad("the rid array differs from index['items']")
    ib, bp = np.asarray(a["item_block"], np.int64), np.asarray(a["blk_pass"], np.int64)
    I, NB, NP = len(rid), int(a["blk_index"].shape[0]), int(a["pass_no"].shape[0])
    if ib.shape != (I + 1,) or ib[0] != 0 or ib[-1] != NB or np.any(np.diff(ib) < 0):
        bad("item_block is not a complete monotone offset array")
    if [int(x["n_blocks"]) for x in items] != np.diff(ib).tolist():
        bad("index n_blocks != item_block counts")
    if bp.shape != (NB + 1,) or bp[0] != 0 or bp[-1] != NP or np.any(np.diff(bp) < 0):
        bad("blk_pass is not a complete monotone offset array")
    for k in ("blk_index", "blk_lo", "blk_start"):
        if a[k].shape != (NB,):
            bad(f"{k} shape {a[k].shape}")
    for k in ("blk_tokens", "blk_teacher", "blk_draft", "blk_p_draft"):
        if a[k].shape != (NB, BLOCK):
            bad(f"{k} shape {a[k].shape}")
    if a["blk_counters"].shape != (NB, len(counters)):
        bad(f"blk_counters shape {a['blk_counters'].shape}")
    for k in ("pass_no", "forced", "rd_on", "rd_n_fwd"):
        if a[k].shape != (NP,):
            bad(f"{k} shape {a[k].shape}")
    for k in ("input", "draft", "argmax", "flag", "open", "commit", "tentative", "remask", "p", "q", "rd_holes",
              "rd_tokens", "rd_input", "rd_p_commit"):
        if a[k].shape != (NP, BLOCK):
            bad(f"{k} shape {a[k].shape}")

    # -- blocks --------------------------------------------------------------
    nbi = np.diff(ib)
    within = np.arange(NB) - np.repeat(ib[:-1], nbi)
    if not np.array_equal(np.asarray(a["blk_index"], np.int64), within):
        bad("blk_index is not 0..n-1 within each item")
    lo = np.asarray(a["blk_lo"], np.int64)
    start = np.asarray(a["blk_start"], np.int64)
    if np.any(lo % BLOCK):
        bad("blk_lo off the 32-grid")
    if NB and not np.array_equal(lo, np.repeat(lo[ib[:-1][nbi > 0]], nbi[nbi > 0]) + BLOCK * within):
        bad("blk_lo is not consecutive within an item")
    if np.any(start[within > 0] != 0) or np.any((start[within == 0] < 0) | (start[within == 0] >= BLOCK)):
        bad("blk_start is not the prompt tail in block 0 and 0 after")
    npb = np.diff(bp)
    if not np.array_equal(npb, np.asarray(a["blk_counters"], np.int64)[:, counters.index("n_judge")]):
        bad("passes per block != the n_judge counter")

    # -- passes --------------------------------------------------------------
    blk_of = np.repeat(np.arange(NB), npb)
    if not np.array_equal(np.asarray(a["pass_no"], np.int64), np.arange(NP) - np.repeat(bp[:-1], npb)):
        bad("pass_no is not 0..n-1 within each block")
    opn, com, ten, rem = (np.asarray(a[k], bool) for k in ("open", "commit", "tentative", "remask"))
    inp, dr, am = (np.asarray(a[k], np.int64) for k in ("input", "draft", "argmax"))
    if np.any(com & ten) or np.any(com & rem) or np.any(ten & rem) or not np.array_equal(com | ten | rem, opn):
        bad("commit / tentative / remask do not partition open")
    if not np.array_equal(np.asarray(a["flag"], bool), opn & (inp != mask_id)):
        bad("flag != open & (input != [MASK]) (wrong --mask-id?)")
    if not np.array_equal(am >= 0, opn):
        bad("argmax is not >= 0 exactly at the open slots")
    p, q = np.asarray(a["p"], np.float64), np.asarray(a["q"], np.float64)
    if not np.array_equal(np.isnan(p), ~opn) or not np.array_equal(np.isnan(q), ~opn):
        bad("p / q are not finite exactly at the open slots")
    if np.any((p[opn] < 0) | (p[opn] > 1.0 + 1e-6)) or np.any((q[opn] < 0) | (q[opn] > 1.0 + 1e-6)):
        bad("p / q outside [0, 1]")
    forced = np.asarray(a["forced"], bool)
    if np.any(com[forced].sum(1) != 1):
        bad("a forced pass does not commit exactly one slot")

    # pass 0 of every block: open = recorder-valid slots, holding the first draft
    vrec = np.arange(BLOCK)[None, :] >= start[:, None]                       # [NB, 32]
    hb = np.flatnonzero(npb > 0)
    j0 = bp[:-1][hb]
    bdraft = np.asarray(a["blk_draft"], np.int64)
    if not np.array_equal(opn[j0], vrec[hb]):
        bad("pass 0 is not open exactly at the recorder-valid slots")
    if not np.array_equal(np.where(vrec[hb], inp[j0], 0), np.where(vrec[hb], bdraft[hb], 0)):
        bad("pass 0's input is not the first draft at the valid slots")
    if not np.array_equal(dr[j0], bdraft[hb]):
        bad("pass 0's draft row is not the first draft")

    # re-drafts
    on = np.asarray(a["rd_on"], bool)
    rh = np.asarray(a["rd_holes"], bool)
    rt, ri = np.asarray(a["rd_tokens"], np.int64), np.asarray(a["rd_input"], np.int64)
    rpc = np.asarray(a["rd_p_commit"], np.float64)
    off = ~on
    if (np.any(rh[off]) or np.any(rt[off] != -1) or np.any(ri[off] != -1) or not np.all(np.isnan(rpc[off]))
            or np.any(np.asarray(a["rd_n_fwd"])[off] != 0)):
        bad("rd_* fields set on a pass without a re-draft")
    if not np.array_equal(rh[on], rem[on]):
        bad("a re-draft's holes are not the pass's remask slots")
    if not np.array_equal(ri[on] == mask_id, rh[on] | ten[on]):
        bad("a re-draft's input is not [MASK] exactly at its holes and the tentative slots")
    if np.any(rt[on][rh[on]] == mask_id):
        bad("a re-draft filled a hole with [MASK]")

    # consecutive passes of one block
    j = np.flatnonzero(blk_of[:-1] == blk_of[1:]) if NP > 1 else np.zeros(0, np.int64)
    j1 = j + 1
    applied = np.where(com[j], am[j], np.where(ten[j], dr[j], np.where(rem[j], mask_id, inp[j])))
    hole = on[j][:, None] & rh[j]
    if not np.array_equal(opn[j1], opn[j] & ~com[j]):
        bad("open of pass p+1 != open & ~commit of pass p")
    if not np.array_equal(inp[j1], np.where(hole, rt[j], applied)):
        bad("input of pass p+1 != pass p's decision with its re-draft applied")
    if not np.array_equal(dr[j1], np.where(hole, rt[j], dr[j])):
        bad("draft of pass p+1 != pass p's draft with its re-draft applied")

    ban_viol = int((on[:, None] & rh & (rt == dr)).sum())
    return {"items": I, "blocks": NB, "passes": NP, "redrafts": int(on.sum()), "ban_violations": ban_viol}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _ranges(starts: Sequence[int], stops: Sequence[int]) -> np.ndarray:
    parts = [np.arange(a, b, dtype=np.int64) for a, b in zip(starts, stops)]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def _load_entries(index_paths: Sequence[str]) -> Dict[str, dict]:
    from drafter_lora import data as D
    return {str(e["pid"]): e for e in D.load_entries(list(index_paths))}


def build_index(roots: Sequence[str], out: str, *, index_paths: Sequence[str] = (), mask_id: int = DEFAULT_MASK_ID,
                allow_no_ban: bool = False, strict: bool = False, force: bool = False, item_checks: bool = True,
                log=print) -> dict:
    """Index `roots` into `out` (a plain path; the CLI passes it through safe_out). Returns the summary."""
    t0 = time.time()
    if os.path.exists(out) and not force:
        raise StatesError(f"{out} exists (pass --force to replace it)")
    entries = _load_entries(index_paths) if index_paths else None

    # -- pass 1: sources, configs, the plan (small members only) ----------------
    run_meta = None
    cfg = counters = None
    sources, plan = [], []          # plan: (chunk path, kept item ordinals)
    seen: Dict[str, str] = {}
    dups, no_row = [], []
    n_items = n_blocks = n_passes = 0
    for root in roots:
        chunks = find_chunks(root)
        gdirs = sorted({g for g, _ in chunks})
        for g in gdirs:
            rj = read_run_json(g)
            if run_meta is None:
                run_meta = rj
            elif rj != run_meta:
                raise StatesError(f"{g}/run.json differs from the first source's: records of another judge config, "
                                  f"adapter or drafter are a separate index ({rj} vs {run_meta})")
        sources.append({"root": os.path.abspath(root), "gpu_dirs": [os.path.basename(g) for g in gdirs],
                        "chunks": [os.path.relpath(c, root) for _, c in chunks]})
        for g, path in chunks:
            with np.load(path, allow_pickle=False) as z:
                idx = chunk_index(z)
                rid = [str(x) for x in z["rid"].tolist()]
                ib = np.asarray(z["item_block"], np.int64)
                bp = np.asarray(z["blk_pass"], np.int64)
            if cfg is None:
                cfg, counters = dict(idx.get("cfg") or {}), list(idx.get("counters") or [])
            if dict(idx.get("cfg") or {}) != cfg or list(idx.get("counters") or []) != counters:
                raise StatesError(f"{path}: judge cfg or counters differ from the first chunk's")
            if dict(run_meta.get("cfg") or {}) != cfg:
                raise StatesError(f"{path}: the chunk's cfg differs from run.json's")
            if len(ib) != len(rid) + 1:
                raise StatesError(f"{path}: item_block has {len(ib)} entries for {len(rid)} rids")
            keep = []
            for i, r in enumerate(rid):
                if r in seen:
                    dups.append({"rid": r, "kept": seen[r], "dropped": path})
                    continue
                if entries is not None and r not in entries:
                    no_row.append(r)
                    continue
                seen[r] = path
                keep.append(i)
                n_items += 1
                n_blocks += int(ib[i + 1] - ib[i])
                n_passes += int(bp[ib[i + 1]] - bp[ib[i]])
            plan.append((path, keep))
    if cfg is None:
        raise StatesError(f"no states.NNNNN.npz under {list(roots)}")
    if int(cfg.get("block_size", -1)) != BLOCK:
        raise StatesError(f"block_size {cfg.get('block_size')} != {BLOCK}")
    if not cfg.get("redraft_ban") and not allow_no_ban:
        raise StatesError("the records were made WITHOUT --redraft-ban, unlike the decode under test "
                          "(pass --allow-no-ban to index them anyway)")
    if no_row and strict:
        raise StatesError(f"{len(no_row)} recorded rids are in no --index row (first: {no_row[:3]})")
    if n_blocks == 0 or n_passes == 0:
        raise StatesError(f"nothing to index: {n_items} items, {n_blocks} blocks, {n_passes} passes kept")
    log(f"[index_states] plan: {n_items} items, {n_blocks} blocks, {n_passes} passes from "
        f"{sum(len(s['chunks']) for s in sources)} chunks; {len(dups)} duplicate rids dropped, "
        f"{len(no_row)} rids without an index row dropped")

    # -- pass 2: validate every chunk, copy the kept items --------------------------
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        arrs: Dict[str, np.ndarray] = {}
        rid_out: List[str] = []
        item_block = np.zeros(n_items + 1, np.int64)
        blk_pass = np.zeros(n_blocks + 1, np.int64)
        oi = ob = op = 0
        soft = Counter()
        for path, keep in plan:
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            idx = json.loads(str(a["index"]))
            soft.update(check_chunk(a, idx, mask_id, path))
            if not arrs:                                         # dtypes / widths from the first chunk
                for k in BLK_KEYS:
                    arrs[k] = np.lib.format.open_memmap(os.path.join(tmp, f"{k}.npy"), mode="w+", dtype=a[k].dtype,
                                                        shape=(n_blocks,) + a[k].shape[1:])
                for k in PASS_KEYS:
                    arrs[k] = np.lib.format.open_memmap(os.path.join(tmp, f"{k}.npy"), mode="w+", dtype=a[k].dtype,
                                                        shape=(n_passes,) + a[k].shape[1:])
            if not keep:
                continue
            ib, bp = a["item_block"].astype(np.int64), a["blk_pass"].astype(np.int64)
            brows = _ranges([ib[i] for i in keep], [ib[i + 1] for i in keep])
            prows = _ranges([bp[ib[i]] for i in keep], [bp[ib[i + 1]] for i in keep])
            for k in BLK_KEYS:
                arrs[k][ob:ob + len(brows)] = a[k][brows]
            for k in PASS_KEYS:
                arrs[k][op:op + len(prows)] = a[k][prows]
            npb = (bp[brows + 1] - bp[brows])
            blk_pass[ob + 1:ob + 1 + len(brows)] = op + np.cumsum(npb)
            rid_list = [str(x) for x in a["rid"].tolist()]
            for i in keep:
                rid_out.append(rid_list[i])
                item_block[oi + 1] = item_block[oi] + int(ib[i + 1] - ib[i])
                oi += 1
            ob += len(brows)
            op += len(prows)
        assert (oi, ob, op) == (n_items, n_blocks, n_passes), ((oi, ob, op), (n_items, n_blocks, n_passes))
        if soft["ban_violations"] and cfg.get("redraft_ban") and strict:
            raise StatesError(f"{soft['ban_violations']} re-drafted holes hold the banned (rejected) draft token")
        for k in list(arrs):
            arrs[k].flush()
        np.save(os.path.join(tmp, "item_rid.npy"), np.array(rid_out, dtype=str))
        np.save(os.path.join(tmp, "item_block.npy"), item_block)
        np.save(os.path.join(tmp, "blk_pass.npy"), blk_pass)
        del arrs

        meta = {"version": VERSION, "cfg": cfg, "counters": counters, "mask_id": int(mask_id), "block": BLOCK,
                "run": run_meta, "sources": sources, "index": [os.path.abspath(p) for p in index_paths],
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(tmp, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        ix = StatesIndex(tmp)
        summary = summarize(ix, entries, item_checks=item_checks, log=log)
        summary.update({"duplicates": len(dups), "duplicate_examples": dups[:20],
                        "records_without_index_row": len(no_row), "without_row_examples": no_row[:20],
                        "ban_violations": int(soft["ban_violations"]), "chunks_read": len(plan),
                        "build_seconds": round(time.time() - t0, 1)})
        with open(os.path.join(tmp, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        del ix
        if os.path.exists(out):
            shutil.rmtree(out)
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"[index_states] wrote {out}: {n_items} items, {n_blocks} blocks, {n_passes} passes "
        f"in {time.time() - t0:.1f} s")
    return summary


# ---------------------------------------------------------------------------
# summary (+ the trainset checks with --index)
# ---------------------------------------------------------------------------
def summarize(ix: "StatesIndex", entries: Optional[Dict[str, dict]], item_checks: bool = True, log=print) -> dict:
    a = ix.a
    npb = np.diff(np.asarray(a["blk_pass"], np.int64))
    nj = np.asarray(a["blk_counters"])[:, ix.counters.index("n_judge")] if len(npb) else np.zeros(0)
    nrd = (np.asarray(a["blk_counters"])[:, ix.counters.index("n_redraft_rounds")]
           if "n_redraft_rounds" in ix.counters and len(npb) else np.zeros(0))
    s = {"items": ix.n_items, "blocks": int(len(npb)), "passes": int(npb.sum()) if len(npb) else 0,
         "passes_per_block_mean": float(npb.mean()) if len(npb) else float("nan"),
         "passes_per_block_hist": np.bincount(npb).tolist() if len(npb) else [],
         "redrafts_per_block_mean": float(nrd.mean()) if len(nrd) else float("nan"),
         "forced_pass_share": float(np.asarray(a["forced"]).mean()) if s_len(a) else float("nan"),
         "cfg": ix.cfg}
    assert len(nj) == 0 or np.array_equal(nj, npb)
    if entries is None:
        return s
    dom, rep = Counter(), Counter()
    for r in ix.rids:
        e = entries[r]
        dom[str(e.get("domain") or "unknown")] += 1
        rep["repeated" if e.get("repeated") else "fresh"] += 1
    s["items_by_domain"] = dict(sorted(dom.items()))
    s["items_fresh_vs_repeated"] = dict(sorted(rep.items()))
    if not item_checks:
        return s
    from drafter_lora import data as D
    ib = np.asarray(a["item_block"], np.int64)
    bp = np.asarray(a["blk_pass"], np.int64)
    post_eos_passes = post_eos_blocks = unreachable = holes = 0
    t0 = time.time()
    for i, r in enumerate(ix.rids):
        it = D.load_item(entries[r])
        tok = np.asarray(it["tokens"], np.int64)
        bs = np.asarray(it["block_start"], np.int64)
        valid = np.asarray(it["valid"], bool)
        rows = np.arange(ib[i], ib[i + 1])
        if len(rows) != tok.shape[0]:
            raise StatesError(f"{r}: {len(rows)} recorded blocks, the trainset item has {tok.shape[0]}")
        if not np.array_equal(np.asarray(a["blk_teacher"][rows], np.int64), tok):
            raise StatesError(f"{r}: recorded teacher tokens != the trainset item's tokens")
        if not np.array_equal(np.asarray(a["blk_lo"][rows], np.int64), bs):
            raise StatesError(f"{r}: recorded blk_lo != the trainset item's block_start")
        L = int(it["prompt_length"])
        if int(a["blk_start"][rows[0]]) != L - int(bs[0]):
            raise StatesError(f"{r}: recorded blk_start {int(a['blk_start'][rows[0]])} != prompt tail {L - int(bs[0])}")
        vrec = np.arange(BLOCK)[None, :] >= np.asarray(a["blk_start"][rows], np.int64)[:, None]
        if np.any(valid & ~vrec):
            raise StatesError(f"{r}: a trainset-valid slot is not recorder-valid")
        post = vrec & ~valid                                               # [nb, 32] slots after the first EOS
        for b, row in enumerate(rows):
            ps = slice(bp[row], bp[row + 1])
            if post[b].any():
                pe = (np.asarray(a["open"][ps], bool) & post[b]).any(1)
                post_eos_passes += int(pe.sum())
                post_eos_blocks += int(pe.any())
            on = np.asarray(a["rd_on"][ps], bool)
            if on.any():
                h = np.asarray(a["rd_holes"][ps], bool)[on]
                holes += int(h.sum())
                unreachable += int((h & (np.asarray(a["draft"][ps], np.int64)[on] == tok[b])).sum())
        if (i + 1) % 10000 == 0:
            log(f"[index_states] item checks: {i + 1}/{ix.n_items} items in {time.time() - t0:.0f} s")
    s.update({"post_eos_passes": post_eos_passes, "post_eos_blocks": post_eos_blocks,
              "post_eos_pass_share": post_eos_passes / max(1, s["passes"]),
              "redraft_holes": holes, "ban_unreachable_holes": unreachable,
              "ban_unreachable_share": unreachable / max(1, holes), "item_checks": "all"})
    return s


def s_len(a) -> int:
    return int(np.asarray(a["forced"]).shape[0])


# ---------------------------------------------------------------------------
# read side
# ---------------------------------------------------------------------------
@dataclass
class RecBlock:
    """One trainset block's recorded passes (small eager arrays; `row` / `pass0` are global rows of the index)."""
    rid: str
    b: int
    row: int
    pass0: int
    lo: int
    start: int
    teacher: np.ndarray            # [32] the stored (vanilla) tokens
    draft0: np.ndarray             # [32] the first draft
    passes: Dict[str, np.ndarray]  # PASS_KEYS -> [n, ...]

    @property
    def n(self) -> int:
        return int(self.passes["pass_no"].shape[0])


class StatesIndex:
    """An index_states.py output dir, memory-mapped."""

    def __init__(self, path: str, mmap: bool = True):
        self.path = os.path.abspath(path)
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        if self.meta.get("version") != VERSION:
            raise StatesError(f"{path}: index version {self.meta.get('version')}, expected {VERSION}")
        mode = "r" if mmap else None
        self.a = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode=mode, allow_pickle=False) for k in ALL_KEYS}
        self.rids = [str(x) for x in self.a["item_rid"].tolist()]
        self.pos = {r: i for i, r in enumerate(self.rids)}
        self.ib = np.asarray(self.a["item_block"], np.int64)
        self.bp = np.asarray(self.a["blk_pass"], np.int64)
        self.cfg = dict(self.meta["cfg"])
        self.counters = list(self.meta["counters"])
        self.mask_id = int(self.meta["mask_id"])
        self.block_size = int(self.meta.get("block", BLOCK))

    @property
    def n_items(self) -> int:
        return len(self.rids)

    def n_blocks(self, pid: str) -> int:
        i = self.pos.get(str(pid))
        return 0 if i is None else int(self.ib[i + 1] - self.ib[i])

    def block_row(self, pid: str, b: int) -> Optional[int]:
        i = self.pos.get(str(pid))
        if i is None or not 0 <= int(b) < int(self.ib[i + 1] - self.ib[i]):
            return None
        return int(self.ib[i] + int(b))

    def n_passes(self, row: int) -> int:
        return int(self.bp[row + 1] - self.bp[row])

    def block(self, pid: str, b: int, keys: Sequence[str] = PASS_KEYS) -> Optional[RecBlock]:
        row = self.block_row(pid, b)
        if row is None:
            return None
        s = slice(int(self.bp[row]), int(self.bp[row + 1]))
        return RecBlock(rid=str(pid), b=int(b), row=row, pass0=int(self.bp[row]), lo=int(self.a["blk_lo"][row]),
                        start=int(self.a["blk_start"][row]), teacher=np.asarray(self.a["blk_teacher"][row]),
                        draft0=np.asarray(self.a["blk_draft"][row]),
                        passes={k: np.asarray(self.a[k][s]) for k in keys})

    def summary(self) -> dict:
        return {"path": self.path, "items": self.n_items, "blocks": int(len(self.bp) - 1),
                "passes": int(self.bp[-1]), "redraft_every": self.cfg.get("redraft_every"),
                "redraft_ban": self.cfg.get("redraft_ban"), "loop_guard": self.cfg.get("loop_guard")}


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="record roots: judge_sglang.py --mode record --out dirs (gpu<k>/ inside)")
    ap.add_argument("--index", action="append", default=[], help="the trainset index.jsonl the records were made "
                    "from (repeatable): enables the teacher / start checks, domains, fresh vs repeated, post-EOS")
    ap.add_argument("--out", required=True, help="output dir (repo-relative ok; written through safe_out)")
    ap.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    ap.add_argument("--allow-no-ban", action="store_true", help="index records made without --redraft-ban")
    ap.add_argument("--strict", action="store_true", help="refuse rids without an index row and ban violations")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    ap.add_argument("--no-item-checks", dest="item_checks", action="store_false", default=True,
                    help="skip loading every trainset item (teacher / start / post-EOS checks)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import research_protocol as P                     # noqa: E402  (safe_out; sets HF offline, harmless here)
    out = P.safe_out(args.out)
    try:
        s = build_index(args.roots, out, index_paths=args.index, mask_id=args.mask_id, allow_no_ban=args.allow_no_ban,
                        strict=args.strict, force=args.force, item_checks=args.item_checks)
    except StatesError as e:
        print(f"[index_states] REFUSED: {e}", flush=True)
        return 2
    keys = ("items", "blocks", "passes", "passes_per_block_mean", "redrafts_per_block_mean", "forced_pass_share",
            "duplicates", "records_without_index_row", "post_eos_pass_share", "ban_unreachable_share",
            "ban_violations", "items_by_domain", "items_fresh_vs_repeated")
    for k in keys:
        if k in s:
            print(f"  {k:28s} {s[k]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
