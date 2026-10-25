#!/usr/bin/env python
"""Slot-level agreement of a drafter with the MoE's recorded blocks (GPU).

This is the metric the whole round-3 lever is about: 0.556 for the stock 1.7B drafter at 8
static steps (EXPLAINER section 18 -- "the draft is right at 55.5%"), and the ceiling on the
judge's confirmations (section 19).  It replays the big model's OWN decode:

    prefill the prompt blocks into a `SeqCache`;
    for every complete 32-block of the recorded generation
        the drafter writes the block from all-[MASK] with `drafter.draft_block(..., "s8")`,
        against the TEACHER-FORCED prefix (the MoE's final tokens of every earlier block);
        agreement = the share of the block's generated slots where the drafter's token equals
                    the MoE's token;
        then the MoE's final tokens are flushed into the cache and the next block starts.

The prefix is the big model's text, never the drafter's, so a block's score does not depend on
the drafter's earlier mistakes -- exactly how the drafter is used inside `da_decode.py`.

The last, incomplete block of an item is skipped: its slots beyond the EOS were never part of
the trimmed target, and a partly-defined block would need a convention the decode does not have.

Items come from a `build_drafter_set.py` index, restricted to the HELD-OUT split of
`train_drafter.DrafterSet.split_holdout`, so the same `--seed` and `--holdout-items` as the
training run give exactly the items the drafter never saw.

    /venv/main/bin/python code/draftalign/eval_drafter_agreement.py \
        --index logs/draftalign/drafterset_union/index.jsonl \
        --ckpt logs/draftalign/train_drafter/d1 --baseline d32 \
        --holdout-items 100 --n-items 100 \
        --out results/metrics/draftalign/drafter_agreement_d1.json

`--ckpt` is a checkpoint DIRECTORY written by `train_drafter.save_checkpoint` (or `<out>`, whose
`latest` marker is followed), or one of the `research_protocol.MODELS` keys ("d32" = stock).
With `--baseline` both models are run over the same items and the paired per-item deltas and a
paired bootstrap interval are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402
from research_protocol import MASK_ID, LaunchLog, config_hash, load_sdar, safe_out   # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

from draftalign.drafter import SETTINGS, draft_block                     # noqa: E402
from draftalign.seqcache import SeqCache                                 # noqa: E402
from draftalign.train_drafter import DrafterSet, DrafterSetConfig, resolve_checkpoint  # noqa: E402

BLOCK = 32


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def localize_auto_map(path: str) -> int:
    """`save_pretrained` records the remote-code classes as `<hub repo>--module.Class`; offline, from a
    plain directory, transformers then tries to fetch the module from the hub and fails even though
    the .py files sit right there.  Strip the repo prefix so the local files are used.  Idempotent;
    returns the number of entries rewritten."""
    n = 0
    for name in ("config.json", "tokenizer_config.json"):
        fp = os.path.join(path, name)
        if not os.path.isfile(fp):
            continue
        d = json.load(open(fp))
        am = d.get("auto_map")
        if not isinstance(am, dict):
            continue
        for k, v in list(am.items()):
            if isinstance(v, str) and "--" in v:
                am[k] = v.split("--", 1)[1]; n += 1
            elif isinstance(v, list):
                new = [(x.split("--", 1)[1] if isinstance(x, str) and "--" in x else x) for x in v]
                n += sum(1 for a, b in zip(v, new) if a != b); am[k] = new
        if n:
            json.dump(d, open(fp, "w"), indent=2, sort_keys=True)
    return n


def load_drafter(spec: str, device: str, model_key: str = "d32"):
    """`spec` = a MODELS key ("d32" = stock) or a checkpoint directory.  Returns (tok, model, info)."""
    if spec in P.MODELS:
        tok, model = load_sdar(spec, device)
        return tok, model, {"kind": "stock", "spec": spec,
                            "revision": P.MODELS[spec]["revision"]}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = resolve_checkpoint(safe_out(spec))
    assert os.path.isdir(path), f"not a checkpoint directory: {path}"
    localize_auto_map(path)
    kw = {"trust_remote_code": True, "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(path, **kw)
    assert tok(tok.mask_token)["input_ids"][0] == MASK_ID, "checkpoint tokenizer lost the MASK id"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map=device,
                                                 **kw).eval()
    step = None
    tp = os.path.join(path, "trainer_state.pt")
    if os.path.isfile(tp):
        step = int(torch.load(tp, map_location="cpu", weights_only=False).get("step", -1))
    return tok, model, {"kind": "checkpoint", "spec": path, "step": step}


# ---------------------------------------------------------------------------
# one item
# ---------------------------------------------------------------------------
@torch.no_grad()
def item_agreement(model, item: dict, device, setting: str = "s8", mask_id: int = MASK_ID,
                   block: int = BLOCK) -> dict:
    """Replay every COMPLETE block of one recorded item.  Returns per-block agreement."""
    prompt = np.asarray(item["prompt_ids"], np.int64)
    gen = np.asarray(item["gen_ids"], np.int64)
    lo0, pl = int(item["block_start"]), int(item["prompt_length"])
    assert prompt.shape[0] == pl and lo0 == (pl // block) * block
    n_complete = (pl - lo0 + int(gen.shape[0])) // block
    full = torch.from_numpy(np.concatenate([prompt, gen])).to(device)

    seq = SeqCache(model, block, "eval")
    seq.prefill(torch.from_numpy(prompt).unsqueeze(0))
    per_block, slots, agree, n_fwd = [], 0, 0, 0
    for b in range(n_complete):
        lo = lo0 + b * block
        final = full[lo:lo + block].clone()
        valid = torch.ones(block, dtype=torch.bool, device=device)
        if lo < pl:                                     # block 0 carries the ragged prompt tail
            valid[: pl - lo] = False
        out = draft_block(seq, final, valid, lo, setting, mask_id)
        eq = int(((out["tokens"] == final) & valid).sum())
        nv = int(valid.sum())
        per_block.append(eq / nv if nv else float("nan"))
        slots += nv; agree += eq; n_fwd += int(out["n_fwd"])
        seq.flush(final, lo)                            # teacher forcing: the MoE's own tokens
    return {"idx": int(item.get("idx", -1)), "n_blocks": n_complete, "slots": slots,
            "agree": agree, "agreement": (agree / slots) if slots else float("nan"),
            "per_block": per_block, "drafter_forwards": n_fwd}


@torch.no_grad()
def run_model(model, ds: DrafterSet, n_items: int, device, setting: str) -> dict:
    rows = []
    t0 = time.time()
    for k in range(min(int(n_items), len(ds))):
        it = ds.arrays(k)
        it = {**{key: it[key] for key in ("prompt_ids", "gen_ids", "block_start", "prompt_length")},
              "idx": ds.entries[k]["idx"], "split": ds.entries[k]["split"]}
        r = item_agreement(model, it, device, setting)
        r["split"] = ds.entries[k]["split"]
        rows.append(r)
        if (k + 1) % 10 == 0:
            done = sum(x["agree"] for x in rows) / max(1, sum(x["slots"] for x in rows))
            print(f"[{k + 1}/{min(int(n_items), len(ds))}] running agreement {done:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    slots = sum(r["slots"] for r in rows)
    return {"items": rows, "n_items": len(rows), "slots": slots,
            "agreement": sum(r["agree"] for r in rows) / max(1, slots),
            "item_mean_agreement": float(np.mean([r["agreement"] for r in rows])) if rows else float("nan"),
            "blocks": sum(r["n_blocks"] for r in rows),
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def paired(a: dict, b: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Per-item paired delta a - b (item-level agreement) with a bootstrap interval."""
    ia = {r["idx"]: r for r in a["items"]}
    ib = {r["idx"]: r for r in b["items"]}
    keys = [k for k in ia if k in ib]
    d = np.array([ia[k]["agreement"] - ib[k]["agreement"] for k in keys], dtype=np.float64)
    if d.size == 0:
        return {"n_paired": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(int(n_boot))])
    # slot-pooled delta, the number quoted as "agreement"
    sa = sum(ia[k]["agree"] for k in keys) / max(1, sum(ia[k]["slots"] for k in keys))
    sb = sum(ib[k]["agree"] for k in keys) / max(1, sum(ib[k]["slots"] for k in keys))
    return {"n_paired": int(d.size), "mean_item_delta": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "items_better": int((d > 0).sum()), "items_worse": int((d < 0).sum()),
            "items_equal": int((d == 0).sum()),
            "slot_agreement_ckpt": float(sa), "slot_agreement_baseline": float(sb),
            "slot_delta": float(sa - sb)}


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="build_drafter_set index.jsonl")
    ap.add_argument("--ckpt", required=True, help="checkpoint directory, or a MODELS key (d32 = stock)")
    ap.add_argument("--baseline", default=None,
                    help="second drafter for the paired comparison (usually d32)")
    ap.add_argument("--out", default=None, help="json report path")
    ap.add_argument("--setting", choices=tuple(SETTINGS), default="s8")
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--holdout-items", type=int, default=100,
                    help="MUST equal the training run's --holdout-items")
    ap.add_argument("--seed", type=int, default=0, help="MUST equal the training run's --seed")
    ap.add_argument("--max-seq", type=int, default=1200)
    ap.add_argument("--min-blocks", type=int, default=2)
    ap.add_argument("--only-correct", dest="only_correct", action="store_true", default=True)
    ap.add_argument("--no-only-correct", dest="only_correct", action="store_false")
    ap.add_argument("--all-items", action="store_true",
                    help="score the WHOLE index instead of the held-out split (training-set diagnostic)")
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    cfg = dict(vars(args))
    launch = LaunchLog("draftalign/eval_drafter_agreement", cfg)          # noqa: F841
    device = torch.device(args.device)

    scfg = DrafterSetConfig(index_path=args.index, seed=args.seed, mask_id=MASK_ID,
                            max_seq=args.max_seq, min_blocks=args.min_blocks,
                            only_correct=args.only_correct)
    full = DrafterSet(scfg)
    ds = full if args.all_items else full.split_holdout(args.holdout_items)[1]
    print(f"[data] {full.summary()} -> scoring {min(args.n_items, len(ds))} of {len(ds)} "
          f"{'index' if args.all_items else 'held-out'} items", flush=True)

    report: Dict[str, object] = {"config": cfg, "config_hash": config_hash(cfg),
                                 "setting": args.setting, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                               time.gmtime())}
    tok, model, info = load_drafter(args.ckpt, str(device))
    report["ckpt_info"] = info
    report["ckpt"] = run_model(model, ds, args.n_items, device, args.setting)
    print(f"[ckpt] agreement {report['ckpt']['agreement']:.4f} over "
          f"{report['ckpt']['slots']} slots", flush=True)
    del model
    torch.cuda.empty_cache()

    if args.baseline:
        _, base, binfo = load_drafter(args.baseline, str(device))
        report["baseline_info"] = binfo
        report["baseline"] = run_model(base, ds, args.n_items, device, args.setting)
        print(f"[baseline] agreement {report['baseline']['agreement']:.4f}", flush=True)
        report["paired"] = paired(report["ckpt"], report["baseline"])
        print("[paired] " + json.dumps({k: v for k, v in report["paired"].items()
                                        if k != "items"}, indent=2), flush=True)
        del base
        torch.cuda.empty_cache()

    if args.out:
        out = safe_out(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[out] {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
