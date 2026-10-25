#!/usr/bin/env python
"""Judge round, step 2b: the drafter's RE-drafts D1 of every block's holes, from a vanilla pass-p context.

    # one shard per GPU (run_jr.sh redrafts runs k = 0..7)
    CUDA_VISIBLE_DEVICES=k /venv/p2/bin/python code/judge_round/redraft_items.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --pids logs/phase5/judge/subset_pids.json --drafts logs/phase5/judge/drafts \
        --trained logs/phase4/train/main/merged --shard k --nshards 8 --device cuda:0 --seed 0 \
        --out logs/phase5/judge/redrafts

Per item, per block b (T = the 30B's tokens, D0 = the NEW first draft from --drafts, the item's stored draft when the
pid is missing there):
  p        ~ U{1, ..., n_pass[b] - 1}, numpy default_rng([seed, sha256(pid)[:8] as int, b]); 0 when n_pass[b] <= 1
  committed = the valid slots vanilla committed at passes < p: VISIBLE to the drafter, holding T
  tentative = valid, uncommitted, D0 == T: [MASK] in the drafter's input, NOT filled
  holes     = valid, uncommitted, D0 != T (D0 >= 0): [MASK], filled by ONE re-draft call
  invalid slots (prompt tail of block 0, post-EOS) hold T, as in drafter_lora/replay.py's probe.
  (A valid uncommitted slot with D0 < 0 -- not expected -- is hidden like a tentative one and counted as `undrafted`.)
The re-draft is replay.py's probe with the ban on (decode's --redraft-ban-rejected): draft_block(..., "s8",
static_steps=min(8, ceil(holes / rate)), kind="redraft", ban=D0 at the holes).  Then the block's TRUE tokens are
flushed, so the prefix is always the clean 30B output (SeqCache, as in replay.py).

Output, one pair per shard (shard k of n processes sorted(pids)[k::n]):
  <out>/redrafts.shard{k}of{n}.npz   pids [n_items] str, nb [n_items] int32, pass_p [sum nb] int16,
                                     redraft [sum nb, 32] int32 (D1 at the holes, -1 elsewhere),
                                     rconf [sum nb, 32] uint8 (round(255 p_commit) at the holes, 0 elsewhere),
                                     n_holes / n_agree / n_repeat [sum nb] int16 (per-block counts)
  <out>/redrafts.shard{k}of{n}.json  totals (items, blocks, blocks_with_holes, holes, agree, repeat, forwards,
                                     undrafted, seconds) and the same counts per domain
`load_redrafts(dir)` reads every shard back as {pid: (pass_p [nb], redraft [nb, 32], rconf [nb, 32])}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

from draftalign.drafter import draft_block                               # noqa: E402
from draftalign.seqcache import SeqCache                                 # noqa: E402
from drafter_lora import data as D                                       # noqa: E402
from drafter_lora.replay import BLOCK, MASK_ID, eval_mode, restore_mode  # noqa: E402

COUNT_KEYS = ("items", "blocks", "blocks_with_holes", "holes", "agree", "repeat", "forwards", "undrafted")


# ---------------------------------------------------------------------------
# one block, one item
# ---------------------------------------------------------------------------
def sample_pass(seed: int, pid: str, b: int, n_pass: int) -> int:
    """p ~ U{1, ..., n_pass - 1}, seeded by (seed, pid, block); 0 when the block has a single vanilla pass."""
    if int(n_pass) <= 1:
        return 0
    h = int(hashlib.sha256(str(pid).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([int(seed), h, int(b)])
    return int(rng.integers(1, int(n_pass)))


def block_masks(valid: np.ndarray, committed_hist: np.ndarray, T: np.ndarray, D0: np.ndarray,
                p: int, accepted: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The re-draft state's slot sets of one block ([32] bool each); `committed_hist` is item["committed"][b].
    `accepted` is item["accepted"][b] when the item has it: at a pass p > 0 those slots count as committed too, the
    rule draftalign/dataset.py's StateSampler (committed_includes_accepted, the trainer's default) and
    judge_round/sampler_k3.py apply, so the state re-drafted here is the state the judge is trained on."""
    valid = np.asarray(valid, bool)
    committed = np.zeros(BLOCK, bool)
    if p > 0:
        committed = np.asarray(committed_hist[:p], bool).any(0)
        if accepted is not None:
            committed |= np.asarray(accepted, bool)
        committed &= valid
    open_ = valid & ~committed
    tentative = open_ & (D0 == T)
    holes = open_ & (D0 != T) & (D0 >= 0)
    undrafted = open_ & (D0 < 0)
    return {"committed": committed, "tentative": tentative, "holes": holes, "undrafted": undrafted,
            "hidden": open_}                                   # hidden = tentative | holes | undrafted


@torch.no_grad()
def redraft_item(model, item: dict, d0, device, *, seed: int = 0, rate: int = 4,
                 mask_id: int = MASK_ID) -> Dict[str, object]:
    """Re-drafts of every block of one item (see the module docstring).  `d0` [nb, 32] is the first draft."""
    pid = str(item["pid"])
    prompt = torch.as_tensor(np.asarray(item["prompt_ids"], np.int64)).unsqueeze(0)
    tokens = np.asarray(item["tokens"], np.int64)
    valid = np.asarray(item["valid"], bool)
    starts = np.asarray(item["block_start"], np.int64)
    n_pass = np.asarray(item["n_pass"], np.int64)
    hist = np.asarray(item["committed"], bool)
    accepted = np.asarray(item["accepted"], bool) if "accepted" in item else None
    d0 = np.asarray(d0, np.int64)
    nb = tokens.shape[0]
    assert d0.shape == tokens.shape, (pid, d0.shape, tokens.shape)
    seq = SeqCache(model, BLOCK, "redraft")
    seq.prefill(prompt)
    pass_p = np.zeros(nb, np.int16)
    redraft = np.full((nb, BLOCK), -1, np.int32)
    rconf = np.zeros((nb, BLOCK), np.uint8)
    n_holes = np.zeros(nb, np.int16)
    n_agree = np.zeros(nb, np.int16)
    n_repeat = np.zeros(nb, np.int16)
    masks = {k: np.zeros((nb, BLOCK), bool) for k in ("committed", "tentative", "holes")}
    c = {k: 0 for k in COUNT_KEYS}
    c.update(items=1, blocks=nb)
    for b in range(nb):
        lo = int(starts[b])
        T = tokens[b]
        p = sample_pass(seed, pid, b, int(n_pass[b]))
        m = block_masks(valid[b], hist[b], T, d0[b], p, accepted[b] if accepted is not None else None)
        pass_p[b] = p
        for k in masks:
            masks[k][b] = m[k]
        c["undrafted"] += int(m["undrafted"].sum())
        final = torch.as_tensor(T, device=device)
        n_h = int(m["holes"].sum())
        if n_h:
            inp = torch.as_tensor(np.where(m["hidden"], mask_id, T), device=device)
            holes = torch.as_tensor(m["holes"], device=device)
            ban = torch.as_tensor(np.where(m["holes"], d0[b], -1), device=device)
            steps = min(8, -(-n_h // int(rate)))
            r = draft_block(seq, inp, holes, lo, "s8", mask_id, static_steps=steps, kind="redraft", ban=ban)
            new = r["tokens"].cpu().numpy().astype(np.int64)
            pc = torch.nan_to_num(r["p_commit"].float(), nan=0.0).mul(255.0).round().clamp(0, 255).to(torch.uint8)
            hb = m["holes"]
            redraft[b] = np.where(hb, new, -1)
            rconf[b] = np.where(hb, pc.cpu().numpy(), 0)
            n_holes[b] = n_h
            n_agree[b] = int(((new == T) & hb).sum())
            n_repeat[b] = int(((new == d0[b]) & hb).sum())
            c["blocks_with_holes"] += 1
            c["holes"] += n_h
            c["agree"] += int(n_agree[b])
            c["repeat"] += int(n_repeat[b])
            c["forwards"] += int(r["n_fwd"])
        seq.flush(final, lo)
    return {"pid": pid, "domain": str(item.get("domain", "unknown")), "nb": nb, "pass_p": pass_p,
            "redraft": redraft, "rconf": rconf, "n_holes": n_holes, "n_agree": n_agree, "n_repeat": n_repeat,
            "counts": c, **masks}


# ---------------------------------------------------------------------------
# a shard
# ---------------------------------------------------------------------------
def load_first_drafts(d: str) -> Dict[str, np.ndarray]:
    """{pid: draft [nb, 32] int32} from `drafts.shard*.npz` (judge_probe.load_drafts' format; each array is read
    once per file here, where load_drafts re-reads the whole array from the zip for every pid)."""
    out: Dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(os.path.join(d, "drafts.shard*.npz"))):
        z = np.load(f)
        pids, nbs, draft = z["pids"].tolist(), z["nb"].tolist(), z["draft"]
        off = 0
        for pid, nb in zip(pids, nbs):
            out[str(pid)] = draft[off:off + nb].astype(np.int32)
            off += nb
        assert off == draft.shape[0], f"{f}: nb sums to {off}, draft has {draft.shape[0]} rows"
    return out


def _add(acc: Dict[str, Dict[str, int]], dom: str, row: Dict[str, int]) -> None:
    for d in (dom, "all"):
        a = acc.setdefault(d, {k: 0 for k in COUNT_KEYS})
        for k in COUNT_KEYS:
            a[k] += int(row.get(k, 0))


def shard_paths(out_dir: str, shard: int, nshards: int) -> Tuple[str, str]:
    stem = os.path.join(out_dir, f"redrafts.shard{shard}of{nshards}")
    return stem + ".npz", stem + ".json"


def run_shard(model, pids: List[str], load: Callable[[str], dict], drafts: Dict[str, np.ndarray], *, shard: int,
              nshards: int, out_dir: str, device, seed: int = 0, rate: int = 4, mask_id: int = MASK_ID,
              info=None, log_every: int = 100) -> dict:
    """Re-draft sorted(pids)[shard::nshards] and write the shard's npz + json into `out_dir` (already resolved)."""
    assert 0 <= shard < nshards, (shard, nshards)
    mine = sorted(str(p) for p in pids)[shard::nshards]
    was = eval_mode(model)
    acc: Dict[str, Dict[str, int]] = {"all": {k: 0 for k in COUNT_KEYS}}
    rows: Dict[str, list] = {k: [] for k in ("pids", "nb", "pass_p", "redraft", "rconf", "n_holes", "n_agree",
                                             "n_repeat")}
    n_fallback = 0
    t0 = time.time()
    for n, pid in enumerate(mine):
        it = dict(load(pid))
        it["pid"] = pid
        if pid in drafts:
            d0 = drafts[pid]
        else:
            d0 = it["draft"]
            n_fallback += 1
        r = redraft_item(model, it, d0, device, seed=seed, rate=rate, mask_id=mask_id)
        _add(acc, r["domain"], r["counts"])
        rows["pids"].append(pid)
        rows["nb"].append(r["nb"])
        for k in ("pass_p", "redraft", "rconf", "n_holes", "n_agree", "n_repeat"):
            rows[k].append(r[k])
        if (n + 1) % log_every == 0:
            a, dt = acc["all"], time.time() - t0
            print(f"[shard {shard}] {n + 1}/{len(mine)} holes {a['holes']} agree "
                  f"{a['agree'] / max(a['holes'], 1):.4f} repeat {a['repeat']} "
                  f"({dt:.0f}s, {(n + 1) / max(dt, 1e-9):.2f} items/s)", flush=True)
    restore_mode(model, was)
    seconds = round(time.time() - t0, 1)

    def cat(k, dtype, shape):
        return np.concatenate(rows[k]).astype(dtype) if rows[k] else np.zeros(shape, dtype)

    os.makedirs(out_dir, exist_ok=True)
    npz, js = shard_paths(out_dir, shard, nshards)
    np.savez_compressed(npz, pids=np.array(rows["pids"], dtype=str), nb=np.array(rows["nb"], np.int32),
                        pass_p=cat("pass_p", np.int16, (0,)), redraft=cat("redraft", np.int32, (0, BLOCK)),
                        rconf=cat("rconf", np.uint8, (0, BLOCK)), n_holes=cat("n_holes", np.int16, (0,)),
                        n_agree=cat("n_agree", np.int16, (0,)), n_repeat=cat("n_repeat", np.int16, (0,)))
    a = acc["all"]
    res = {"shard": shard, "nshards": nshards, "n_pids": len(pids), **a, "seconds": seconds,
           "agree_rate": a["agree"] / a["holes"] if a["holes"] else None,
           "repeat_rate": a["repeat"] / a["holes"] if a["holes"] else None,
           "per_domain": {d: v for d, v in sorted(acc.items()) if d != "all"}, "fallback_items": n_fallback,
           "seed": seed, "rate": rate, "setting": "s8", "ban": True, "info": info, "npz": npz,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    return res


def load_redrafts(d: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{pid: (pass_p [nb] int16, redraft [nb, 32] int32, rconf [nb, 32] uint8)} from every redrafts.shard*.npz."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for f in sorted(glob.glob(os.path.join(d, "redrafts.shard*.npz"))):
        z = np.load(f)
        pids, nbs = z["pids"].tolist(), z["nb"].tolist()
        pp, rd, rc = z["pass_p"], z["redraft"], z["rconf"]
        off = 0
        for pid, nb in zip(pids, nbs):
            pid = str(pid)
            assert pid not in out, f"pid {pid} appears in two shard files (mixed runs in {d}?)"
            out[pid] = (pp[off:off + nb].astype(np.int16), rd[off:off + nb].astype(np.int32),
                        rc[off:off + nb].astype(np.uint8))
            off += nb
        assert off == rd.shape[0], f"{f}: nb sums to {off}, redraft has {rd.shape[0]} rows"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_trained(trained: str, device: str):
    """(model, info): the trained drafter only -- an adapter dir through eval_agree.load_models (adapter on), a
    merged HF dir through the same load_drafter call load_models makes (without loading the unused stock base)."""
    from drafter_lora import lora as LA
    if os.path.isfile(os.path.join(trained, LA.ADAPTER_JSON)):
        from drafter_lora.eval_agree import load_models
        _m0, m1, info = load_models(trained, device)
        return m1[0], info
    from draftalign.eval_drafter_agreement import load_drafter
    _tok, merged, info = load_drafter(trained, device)
    merged.eval()
    return merged, {"trained": os.path.abspath(trained), "kind": "merged", **info}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--pids", required=True, help='JSON with {"pids": [...]} (select_subset.py / holdout_pids.json)')
    ap.add_argument("--drafts", required=True, help="dir of drafts.shard*.npz: the new first drafts D0")
    ap.add_argument("--trained", required=True, help="merged drafter HF dir (or a drafter_lora adapter dir)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=int, default=4, help="re-draft steps = min(8, ceil(holes / rate))")
    ap.add_argument("--mask-id", type=int, default=MASK_ID, help=argparse.SUPPRESS)       # tests only
    ap.add_argument("--out", required=True, help="output dir (redrafts.shard{k}of{n}.npz/.json)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    assert 0 <= args.shard < args.nshards, (args.shard, args.nshards)
    by_pid = {str(e["pid"]): e for e in D.load_entries(args.index)}
    pids = [str(p) for p in D.load_holdout(args.pids)]
    missing = [p for p in pids if p not in by_pid]
    assert not missing, f"{len(missing)} pids are not in the index (first: {missing[:3]})"
    drafts = load_first_drafts(args.drafts)
    out = P.safe_out(args.out)
    model, info = load_trained(args.trained, args.device)
    res = run_shard(model, pids, lambda pid: D.load_item(by_pid[pid]), drafts, shard=args.shard,
                    nshards=args.nshards, out_dir=out, device=args.device, seed=args.seed, rate=args.rate,
                    mask_id=args.mask_id, info=info)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_domain", "info")}, indent=2), flush=True)
    print(f"[out] {shard_paths(out, args.shard, args.nshards)[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
