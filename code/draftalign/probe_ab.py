#!/usr/bin/env python
"""AB probe driver (scratchpad ab_spec_v0.md): replay the b32 MoE's recorded stock 32/.95
trajectories block by block, teacher-forced from the trace, and per block run
  drafter  : s4 / s8 / dyn90 block generation against a persistent drafter cache;
  MoE      : P0 mask-init, P1/P2 draft(s8) full/halves, P3/P4 self-draft full/halves,
             P5/P6 oracle full/halves -- all store_kv=False against the MoE's persistent cache,
             router top-8 + full softmax captured per layer via gate hooks;
then flush the FINAL tokens into both caches (store_kv=True) and move on.

Items: the 100 frozen calib+valid train records (research_protocol.items_for_split) whose
stock trace in logs/research_v1/baselines_b32_traces/<trace-cell>/<idx>.npz matches the
calibvalid jsonl record (prompt_length AND generated ids).  13 of the 100 traces were
overwritten by gsm8k500 test items sharing the idx (found 2026-09-06); those are skipped
with status "skipped_trace_overwritten".

Usage (one process per GPU; items are sharded items[shard::nshards]):
  CUDA_VISIBLE_DEVICES=4 /venv/main/bin/python code/draftalign/probe_ab.py \
      --items calibvalid --shard 0 --nshards 4 --gpu 4
Outputs: logs/draftalign/ab_probe/<cell>/shard{k}of{n}.jsonl (+ .done), traces/<idx>.npz
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                   # noqa: E402  (HF offline env + modules cache)
from research_protocol import MASK_ID, MODELS, load_sdar, config_hash, safe_out, items_for_split, encode_prompt, LaunchLog  # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from draftalign.seqcache import SeqCache                        # noqa: E402
from draftalign.drafter import SETTINGS, draft_all_settings     # noqa: E402
from draftalign.moe_probe import (KINDS, MS, RouterCapture, build_block_state, slot_metrics,  # noqa: E402
                                  coverage_all_kinds, summary_to_lists)

BLOCK = 32
TRACE_CELL = "gsm8k_B32_t95_stock"
BASELINES = os.path.join("logs", "research_v1", "baselines_b32")
OUT_ROOT = os.path.join("logs", "draftalign", "ab_probe")


def _r(t, nd=5):
    return [round(float(x), nd) for x in t.tolist()]


def _i(t):
    return [int(x) for x in t.tolist()]


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------
def load_baseline_records(baselines_dir: str, trace_cell: str, split: str) -> dict:
    recs = {}
    for f in sorted(glob.glob(os.path.join(baselines_dir, f"{trace_cell}.{split}.shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[int(r["idx"])] = r
    return recs


def load_trace(traces_dir: str, idx: int) -> dict:
    d = np.load(os.path.join(traces_dir, f"{idx:05d}.npz"))
    return {k: d[k] for k in d.files}


def trace_matches_record(tr: dict, rec: dict) -> bool:
    """The trace belongs to this record iff prompt_length AND the generated ids agree."""
    if int(tr["prompt_length"]) != int(rec["prompt_tokens"]):
        return False
    gen = np.concatenate([tr["tokens"][b][tr["valid"][b]] for b in range(tr["tokens"].shape[0])])
    return gen.tolist() == list(rec["gen_ids"])


# ---------------------------------------------------------------------------
# one block
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_block(moe_seq: SeqCache, dr_seq: SeqCache, rc: RouterCapture, final: torch.Tensor,
                valid: torch.Tensor, self_draft: torch.Tensor, lo: int, mask_id: int = MASK_ID):
    """final / self_draft [bl] long, valid [bl] bool, all on the models' device.
    Returns (record dict of python lists/scalars, {kind: int16 top-8 ids [L, bl, K]})."""
    dev = final.device
    # -- drafter (three settings, all store_kv=False) ------------------------
    drafts = draft_all_settings(dr_seq, final, valid, lo, mask_id)
    draft8 = drafts["s8"]["tokens"]
    fills = {"draft": draft8, "self": self_draft, "final": final}
    # -- MoE probes --------------------------------------------------------
    caps, per_kind = {}, {}
    for kind, mode, src in KINDS:
        fill = fills[src] if src else None
        ids, masked = build_block_state(final, valid, fill, mode, mask_id)
        rc.arm()
        logits = moe_seq.block_forward(ids, lo, store_kv=False, kind=kind)
        caps[kind] = rc.disarm()
        m = slot_metrics(logits, final, draft8, fill if fill is not None else draft8, mask_id, valid)
        rec = {"masked": _i(masked), "argmax": _i(m["argmax"]), "top1": _r(m["top1"]),
               "p_final": _r(m["p_final"]), "p_draft": _r(m["p_draft"]), "p_fill": _r(m["p_fill"]),
               "raw_mask_argmax": int(m["raw_mask_argmax"]), "ms": round(moe_seq.timings[-1][1], 3),
               "fill_is_mask": int(((fill == mask_id) & valid).sum()) if fill is not None else 0}
        if kind == "P0":
            rec["argmax_bf16"] = _i(m["argmax_bf16"])
        per_kind[kind] = rec
        del logits
    experts = {k: summary_to_lists(s) for k, s in coverage_all_kinds(caps, valid, MS).items()}
    # -- integrity: P0 (bf16 greedy path) must reproduce the recorded pass-1 proposals on valid slots
    p0 = torch.tensor(per_kind["P0"]["argmax_bf16"], device=dev)
    mism = int(((p0 != self_draft) & valid).sum())
    # -- flush the FINAL tokens into both caches (teacher forcing) -----------
    moe_seq.flush(final, lo)
    dr_seq.flush(final, lo)
    rec = {
        "lo": int(lo), "n_valid": int(valid.sum()), "valid": _i(valid), "final": _i(final), "self_draft": _i(self_draft),
        "drafter": {name: {"tokens": _i(d["tokens"]), "p_commit": _r(d["p_commit"]), "p_final_last": _r(d["p_final_last"]),
                           "n_fwd": int(d["n_fwd"]), "n_steps": int(d["n_steps"]), "raw_mask_argmax": int(d["raw_mask_argmax"]),
                           "ms": [round(x, 3) for x in d["ms"]]} for name, d in drafts.items()},
        "moe": per_kind, "experts": experts,
        "integrity": {"p0_mismatch": mism},
        "moe_flush_ms": round(moe_seq.timings[-1][1], 3), "drafter_flush_ms": round(dr_seq.timings[-1][1], 3),
    }
    top8 = {k: c.ids.to(torch.int16).cpu().numpy() for k, c in caps.items()}
    return rec, top8


# ---------------------------------------------------------------------------
# one item
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_item(moe_seq: SeqCache, dr_seq: SeqCache, rc: RouterCapture, prompt_ids: torch.Tensor, tr: dict,
             mask_id: int = MASK_ID, log=None):
    """Replay every recorded block of one item. Returns (block records, item record, npz arrays)."""
    dev = moe_seq.dev
    nb = int(tr["tokens"].shape[0])
    plen = int(tr["prompt_length"])
    assert prompt_ids.shape == (1, plen), f"encoded prompt {tuple(prompt_ids.shape)} != trace prompt_length {plen}"
    moe_seq.reset(); dr_seq.reset()
    pl = moe_seq.prefill(prompt_ids)
    assert dr_seq.prefill(prompt_ids) == pl
    assert int(tr["block_start"][0]) == pl, f"trace block_start[0]={int(tr['block_start'][0])} != prefill_length {pl}"
    tail = prompt_ids[0, pl:].cpu()
    b0 = torch.as_tensor(tr["tokens"][0]).long()
    v0 = torch.as_tensor(tr["valid"][0]).bool()
    assert torch.equal(b0[~v0], tail), "trace block-0 prompt tail != encoded prompt"
    blocks, top8 = [], {k: [] for k, _, _ in KINDS}
    t_item = time.time()
    for b in range(nb):
        lo = int(tr["block_start"][b])
        assert lo == pl + b * moe_seq.bl
        final = torch.as_tensor(tr["tokens"][b]).long().to(dev)
        valid = torch.as_tensor(tr["valid"][b]).bool().to(dev)
        self_draft = torch.as_tensor(tr["proposed"][b, 0]).long().to(dev)
        assert bool((final[valid] != mask_id).all()), "final tokens hold a MASK"
        rec, t8 = probe_block(moe_seq, dr_seq, rc, final, valid, self_draft, lo, mask_id)
        rec.update({"rec": "block", "b": b, "n_pass_rec": int(tr["n_pass"][b])})
        blocks.append(rec)
        for k, v in t8.items():
            top8[k].append(v)
        if log:
            log(b, nb, rec)
    item = {"rec": "item", "prompt_length": plen, "prefill_length": pl, "n_blocks": nb,
            "moe_n_by_kind": dict(moe_seq.n_by_kind), "moe_ms_by_kind": {k: round(v, 2) for k, v in moe_seq.ms_by_kind().items()},
            "drafter_n_by_kind": dict(dr_seq.n_by_kind), "drafter_ms_by_kind": {k: round(v, 2) for k, v in dr_seq.ms_by_kind().items()},
            "p0_mismatch_total": sum(r["integrity"]["p0_mismatch"] for r in blocks),
            "seconds": round(time.time() - t_item, 2), "status": "ok"}
    arrays = {f"top8_{k}": np.stack(v) for k, v in top8.items()}          # [nb, L, bl, K] int16
    arrays.update({"valid": tr["valid"], "block_start": tr["block_start"], "prompt_length": np.int32(plen)})
    return blocks, item, arrays


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_models(device: str):
    from sdar_harness.moe_instr import enable_instrumented_moe, SINK
    tok, moe = load_sdar("b32", device)
    n = enable_instrumented_moe(moe)                     # bit-exact hit-experts-only dispatch (moe_instr.py)
    assert n == moe.config.num_hidden_layers
    SINK.disarm(); SINK.verify = False
    tok_d, drafter = load_sdar("d32", device)
    assert tok_d(tok_d.mask_token)["input_ids"][0] == MASK_ID
    return tok, moe, tok_d, drafter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="calibvalid", help="calib | valid | calibvalid (train records only)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU label for the launch log (set CUDA_VISIBLE_DEVICES outside)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trace-cell", default=TRACE_CELL)
    ap.add_argument("--baselines", default=BASELINES)
    ap.add_argument("--out", default=None, help=f"default {OUT_ROOT}/<trace-cell>.<items>")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert args.items in ("calib", "valid", "calibvalid"), "train records only; never the test items"

    cell = f"{args.trace_cell}.{args.items}"
    out = safe_out(args.out or os.path.join(OUT_ROOT, cell))
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    traces_dir = safe_out(os.path.join(args.baselines + "_traces", args.trace_cell))
    base_recs = load_baseline_records(safe_out(args.baselines), args.trace_cell, "calibvalid")
    items = items_for_split(args.items)
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.nshards]

    cfg = {"probe": "ab_v0", "cell": cell, "trace_cell": args.trace_cell, "items": args.items, "block": BLOCK,
           "moe": MODELS["b32"], "drafter": MODELS["d32"], "drafter_settings": SETTINGS, "Ms": list(MS),
           "kinds": [k for k, _, _ in KINDS], "shard": args.shard, "nshards": args.nshards, "gpu": args.gpu}
    chash = config_hash(cfg)
    dest = os.path.join(out, f"shard{args.shard}of{args.nshards}.jsonl")
    if os.path.exists(dest + ".done"):
        print(f"[skip] {dest} already done", flush=True); return
    done = set()
    if os.path.exists(dest):
        for line in open(dest):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("rec") == "item":
                assert r["config_hash"] == chash, f"{dest} holds records from another config; refusing to mix"
                done.add(int(r["idx"]))
    todo = [it for it in items if it["idx"] not in done]
    print(f"[plan] cell {cell} shard {args.shard}/{args.nshards}: {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)

    torch.manual_seed(0)
    tok, moe, tok_d, drafter = load_models(args.device)
    import transformers
    meta = {"torch": torch.__version__, "transformers": transformers.__version__, "gpu_name": torch.cuda.get_device_name(moe.device),
            "config_hash": chash}
    launch = LaunchLog(f"draftalign/ab_probe/{cell}/shard{args.shard}of{args.nshards}", {**cfg, **meta})
    moe_seq = SeqCache(moe, BLOCK, "moe"); dr_seq = SeqCache(drafter, BLOCK, "drafter")
    rc = RouterCapture(moe, BLOCK)

    with open(dest, "a") as fh:
        for k, it in enumerate(todo):
            idx = int(it["idx"])
            common = {"idx": idx, "rank": it.get("rank"), "split": it["split"], "config_hash": chash,
                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid()}
            rec = base_recs.get(idx)
            tr = load_trace(traces_dir, idx) if rec is not None else None
            if rec is None or not trace_matches_record(tr, rec):
                status = "skipped_no_record" if rec is None else "skipped_trace_overwritten"
                fh.write(json.dumps({**common, "rec": "item", "status": status, "n_blocks": 0}) + "\n"); fh.flush()
                print(f"[sh{args.shard}] item {idx}: {status}", flush=True)
                continue
            prompt_ids = encode_prompt(tok, it["prompt"])
            assert torch.equal(prompt_ids, encode_prompt(tok_d, it["prompt"])), "drafter tokenizer differs"
            prompt_ids = prompt_ids.to(moe.device)
            blocks, item, arrays = run_item(moe_seq, dr_seq, rc, prompt_ids, tr)
            np.savez_compressed(os.path.join(out, "traces", f"{idx:05d}.npz"), **arrays)
            for b in blocks:
                fh.write(json.dumps({**common, **b}) + "\n")
            fh.write(json.dumps({**common, **item, "peak_mem_gib": round(torch.cuda.max_memory_allocated(moe.device) / 2**30, 2)}) + "\n")
            fh.flush()
            n_probe = sum(item["moe_n_by_kind"].get(kk, 0) for kk, _, _ in KINDS)
            ms_probe = sum(item["moe_ms_by_kind"].get(kk, 0.0) for kk, _, _ in KINDS)
            n_dr = sum(item["drafter_n_by_kind"].get(f"draft_{s}", 0) for s in SETTINGS)
            ms_dr = sum(item["drafter_ms_by_kind"].get(f"draft_{s}", 0.0) for s in SETTINGS)
            print(f"[sh{args.shard}] item {idx} ({k + 1}/{len(todo)}) blocks {item['n_blocks']} {item['seconds']:.0f}s "
                  f"moe {ms_probe / max(n_probe, 1):.0f} ms/fwd drafter {ms_dr / max(n_dr, 1):.0f} ms/fwd "
                  f"p0-mismatch {item['p0_mismatch_total']}", flush=True)
    open(dest + ".done", "w").write("ok\n")
    rc.close()
    launch.exit_code = 0
    print("SHARD DONE", dest, flush=True)


if __name__ == "__main__":
    main()
