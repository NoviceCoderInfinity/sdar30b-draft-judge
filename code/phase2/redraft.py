#!/usr/bin/env python
"""Phase 2, step 3 (+ the data for steps 4 and 6): re-draft an existing training index with another drafter.

The targets stay exactly what they are -- the 30B model's own tokens -- and only the drafts change. For every
usable block of every item the new drafter writes the block from all-[MASK] with the decode's own drafting rule
(`draftalign.drafter.draft_block`, s8) against the TEACHER-FORCED prefix (prompt + the 30B model's final tokens of
the earlier blocks), exactly as inside `da_decode.py`; then the block's final tokens are flushed and the next block
starts. So the new drafts are the drafts that drafter would write at inference, pass 1.

Outputs (per shard, merged by --merge):
    <out>/items/<same relative path>.npz   the item with `draft`, `has_draft`, `draft_conf` (uint8 round(255 p))
                                           from the new drafter; the previous drafts kept as `draft_prev`
    <out>/index.shard<k>of<n>.jsonl        the source index lines, `file` rewritten, plus per-item agreement
    <out>/agreement.shard<k>of<n>.jsonl    one row per item: blocks, slots, new / previous agreement, per domain
    <out>/index.jsonl, summary.json        (--merge) in the source order; per domain and overall: agreement of the
                                           new drafter and of the previous one ON THE SAME BLOCKS, their ratio,
                                           raw-[MASK]-argmax rate, drafter seconds per block

Stop rules (rules.json, stage "agreement") are checked after EVERY item: near-chance agreement (a remap or
vocabulary bug), the drafter predicting its own [MASK], and agreement far below the previous drafter's on the same
blocks (step 3's "stop: the drafter does not fit this MoE").

    CUDA_VISIBLE_DEVICES=0 /venv/train/bin/python code/phase2/redraft.py \
        --index logs/phase2/data/trainset_r3/index.jsonl --drafter F1 --out logs/phase2/redraft/F1 --shard 0 --nshards 8
    /venv/train/bin/python code/phase2/redraft.py --index ... --drafter F1 --out logs/phase2/redraft/F1 --merge --nshards 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402

from draftalign.dataset import domain_of_entry                           # noqa: E402
from phase2.stoprules import STOP_EXIT, StopMonitor, rules_sha           # noqa: E402

BLOCK = 32


def open_forwards(n_open: int, setting: str) -> int:
    """Sum over the drafter's forwards of the slots still open at that forward (the raw-[MASK] rate denominator)."""
    from draftalign.drafter import SETTINGS
    from sdar_harness.decode import get_num_transfer_tokens
    cfg = SETTINGS[setting]
    if n_open == 0:
        return 0
    if cfg["mode"] != "static":
        return n_open                                                     # lower bound for the dynamic rule
    nt = get_num_transfer_tokens(n_open, min(int(cfg["steps"]), n_open)).tolist()
    tot, left = 0, n_open
    for k in nt:
        tot += left
        left -= int(k)
    return tot


def redraft_item(seq, it: dict, setting: str, mask_id: int, device) -> dict:
    """Teacher-forced drafts of every usable block of one training item."""
    import torch
    from draftalign.drafter import draft_block
    prompt = torch.as_tensor(np.asarray(it["prompt_ids"], np.int64)).unsqueeze(0)
    tokens = np.asarray(it["tokens"], np.int64)
    valid = np.asarray(it["valid"], bool)
    usable = np.asarray(it["usable"], bool)
    start = np.asarray(it["block_start"], np.int64)
    nb = tokens.shape[0]
    seq.reset()
    pl = seq.prefill(prompt)
    assert pl == int(start[0]), f"prefill {pl} != first block start {int(start[0])}"
    draft = np.full((nb, BLOCK), -1, np.int32)
    conf = np.zeros((nb, BLOCK), np.uint8)
    raw_mask, open_fwd, t0 = 0, 0, time.time()
    last = int(np.nonzero(usable)[0].max()) if usable.any() else -1
    for b in range(last + 1):
        lo = int(start[b])
        fin = torch.as_tensor(tokens[b], device=device)
        va = torch.as_tensor(valid[b], device=device)
        if usable[b] and bool(va.any()):
            d = draft_block(seq, fin.clone(), va, lo, setting, mask_id)
            dt = d["tokens"].cpu().numpy().astype(np.int32)
            draft[b] = np.where(valid[b], dt, -1)
            pc = d["p_commit"].float().cpu().numpy()
            conf[b] = np.where(valid[b], np.clip(np.round(pc * 255.0), 0, 255), 0).astype(np.uint8)
            raw_mask += int(d["raw_mask_argmax"])
            open_fwd += open_forwards(int(valid[b].sum()), setting)
        seq.flush(fin, lo)
    return {"draft": draft, "draft_conf": conf, "raw_mask": raw_mask, "open_fwd": open_fwd,
            "seconds": time.time() - t0}


def block_agreement(draft: np.ndarray, tokens: np.ndarray, valid: np.ndarray, mask: np.ndarray):
    """(matching slots, slots) over the blocks in `mask` whose drafts exist."""
    has = np.array([bool((draft[b][valid[b]] >= 0).all()) for b in range(len(valid))], bool)
    ok = mask & has & valid.any(1)
    m = (draft == tokens) & valid
    return int(m[ok].sum()), int(valid[ok].sum()), int(ok.sum())


def run_shard(args) -> int:
    import torch
    from phase2 import drafters as P2D
    root = os.path.dirname(os.path.abspath(P.safe_out(args.index)))
    entries = [json.loads(l) for l in open(P.safe_out(args.index)) if l.strip()]
    if args.domains:
        keep = set(args.domains.split(","))
        entries = [e for e in entries if domain_of_entry(e) in keep]
    if args.limit_items:
        entries = entries[:args.limit_items]
    mine = entries[args.shard::args.nshards]
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    tag = f"shard{args.shard}of{args.nshards}"
    done_path = os.path.join(out, f"index.{tag}.jsonl.done")
    if os.path.exists(done_path):
        print(f"[skip] {tag} done", flush=True)
        return 0
    idx_path = os.path.join(out, f"index.{tag}.jsonl")
    agr_path = os.path.join(out, f"agreement.{tag}.jsonl")
    done = set()
    if os.path.exists(idx_path):
        for l in open(idx_path):
            try:
                done.add(json.loads(l)["file"])
            except ValueError:
                pass
    device = torch.device(args.device)
    drafter = P2D.load(args.drafter, args.device)
    seq = drafter.as_seq(BLOCK)
    mon = StopMonitor("agreement", out, context={"drafter": args.drafter, "index": args.index, "shard": tag},
                      **({"rules_path": args.rules} if args.rules else {}))
    print(f"[redraft] {args.drafter} ({drafter.describe()}) on {len(mine)} items ({len(done)} done) -> {out}",
          flush=True)
    tot = {"new_ok": 0, "prev_ok": 0, "slots": 0, "blocks": 0, "raw_mask": 0, "open_fwd": 0, "sec": 0.0,
           "items": 0, "nonfinite": 0}
    with open(idx_path, "a") as fi, open(agr_path, "a") as fa:
        for k, e in enumerate(mine):
            rel = e["file"]
            if rel in done:
                continue
            src = rel if os.path.isabs(rel) else os.path.join(root, rel)
            it = dict(np.load(src).items())
            r = redraft_item(seq, it, args.setting, P.MASK_ID, device)
            tokens, valid, usable = it["tokens"], it["valid"].astype(bool), it["usable"].astype(bool)
            prev = np.asarray(it["draft"], np.int32)
            prev_has = np.asarray(it["has_draft"], bool)
            # the SAME blocks for both drafters: usable, with a previous draft
            same = usable & prev_has
            n_ok, n_sl, n_bl = block_agreement(r["draft"], tokens, valid, same)
            p_ok, _, _ = block_agreement(prev, tokens, valid, same)
            n_same, _, _ = block_agreement(r["draft"], prev, valid, same)       # slots where new draft == previous
            new_has = np.array([bool(valid[b].any() and (r["draft"][b][valid[b]] >= 0).all()) for b in range(len(valid))])
            item = dict(it)
            item.update({"draft": r["draft"], "has_draft": new_has & usable, "draft_conf": r["draft_conf"],
                         "draft_prev": prev, "has_draft_prev": prev_has})
            dst_rel = rel if not os.path.isabs(rel) else os.path.join("items", os.path.basename(rel))
            dst = os.path.join(out, dst_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **item)
            nf = int(getattr(seq, "n_nonfinite", 0))
            row = {"file": dst_rel, "domain": domain_of_entry(e), "split": e.get("split"), "idx": e.get("idx"),
                   "blocks": n_bl, "slots": n_sl, "new_ok": n_ok, "prev_ok": p_ok, "same_as_prev": n_same,
                   "raw_mask": r["raw_mask"],
                   "open_fwd": r["open_fwd"], "seconds": round(r["seconds"], 3), "nonfinite_logits": nf}
            fa.write(json.dumps(row) + "\n"); fa.flush()
            hd = (new_has & usable)
            fi.write(json.dumps({**e, "file": dst_rel, "drafter": args.drafter, "has_draft": hd.tolist(),
                                 "draft_blocks": int(hd.sum()), "draft_agreement": n_ok / max(1, n_sl),
                                 "prev_draft_agreement": p_ok / max(1, n_sl)}) + "\n"); fi.flush()
            for a, b in (("new_ok", n_ok), ("prev_ok", p_ok), ("slots", n_sl), ("blocks", n_bl),
                         ("raw_mask", r["raw_mask"]), ("open_fwd", r["open_fwd"]), ("sec", r["seconds"])):
                tot[a] += b
            tot["items"] += 1
            tot["nonfinite"] = nf
            agr = tot["new_ok"] / tot["slots"] if tot["slots"] else None
            prev_agr = tot["prev_ok"] / tot["slots"] if tot["slots"] else None
            mon.refs["s0_agreement"] = prev_agr
            if (k + 1) % args.print_every == 0 or k + 1 == len(mine):
                print(f"[redraft {tag}] {k + 1}/{len(mine)} items, {tot['blocks']} blocks: agreement {agr:.4f} "
                      f"(previous drafter {prev_agr:.4f} on the same blocks), raw-mask rate "
                      f"{tot['raw_mask'] / max(1, tot['open_fwd']):.4f}, {tot['sec'] / max(1, tot['blocks']) * 1e3:.0f} "
                      f"ms/block", flush=True)
            d = mon.check({"agreement": agr, "raw_mask_argmax_rate": tot["raw_mask"] / max(1, tot["open_fwd"]),
                           "nonfinite_logits": nf},
                          counts={"blocks": tot["blocks"], "items": tot["items"], "n": tot["items"]})
            if d.stop:
                with open(os.path.join(out, f"index.{tag}.jsonl.stopped"), "w") as fh:
                    json.dump({"totals": tot, "stop": mon.stopped}, fh, indent=2, default=str)
                return STOP_EXIT
    with open(done_path, "w") as fh:
        json.dump(tot, fh)
    print(f"[redraft {tag}] DONE {tot}", flush=True)
    return 0


def merge(args) -> int:
    out = P.safe_out(args.out)
    src = [json.loads(l) for l in open(P.safe_out(args.index)) if l.strip()]
    shards = sorted(glob.glob(os.path.join(out, f"index.shard*of{args.nshards}.jsonl")))
    missing = [s for s in shards if not os.path.exists(s + ".done")]
    if len(shards) != args.nshards or missing:
        print(f"[merge] refusing: {len(shards)}/{args.nshards} shard indexes, not done: {missing}", flush=True)
        return 2
    by_src = {}
    for s in shards:
        for l in open(s):
            e = json.loads(l)
            by_src[(e.get("split"), e.get("idx"))] = e
    merged = [by_src[(e.get("split"), e.get("idx"))] for e in src if (e.get("split"), e.get("idx")) in by_src]
    with open(os.path.join(out, "index.jsonl"), "w") as fh:
        for e in merged:
            fh.write(json.dumps(e) + "\n")
    agg = {}
    for s in glob.glob(os.path.join(out, f"agreement.shard*of{args.nshards}.jsonl")):
        for l in open(s):
            r = json.loads(l)
            for dom in (r["domain"], "all"):
                a = agg.setdefault(dom, {"items": 0, "blocks": 0, "slots": 0, "new_ok": 0, "prev_ok": 0,
                                         "raw_mask": 0, "open_fwd": 0, "seconds": 0.0})
                a["items"] += 1
                for k in ("blocks", "slots", "new_ok", "prev_ok", "raw_mask", "open_fwd", "seconds"):
                    a[k] += r[k]
    summ = {"drafter": args.drafter, "source_index": args.index, "items": len(merged), "rules_sha256": rules_sha(),
            "per_domain": {}}
    for dom, a in sorted(agg.items()):
        new = a["new_ok"] / a["slots"] if a["slots"] else None
        prev = a["prev_ok"] / a["slots"] if a["slots"] else None
        summ["per_domain"][dom] = {**a, "agreement": new, "prev_agreement": prev,
                                   "ratio_to_prev": (new / prev) if (new is not None and prev) else None,
                                   "raw_mask_rate": a["raw_mask"] / max(1, a["open_fwd"]),
                                   "ms_per_block": 1e3 * a["seconds"] / max(1, a["blocks"])}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ["per_domain"], indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="source training index (build_train_set / to_trainset format)")
    ap.add_argument("--drafter", required=True, help="phase2.drafters key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="s8")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--domains", default=None, help="comma list of domains to keep (default all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)
    return merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
