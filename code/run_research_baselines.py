#!/usr/bin/env python
"""Matched width x threshold baseline cells for research_v1 (handoff section 9).

Each cell decodes with the UNCHANGED reference routine
`sdar_harness.decode.block_diffusion_generate` (steps = block length, greedy,
bf16 confidence, strict > threshold, EOS at block boundaries, full-block kv
flush) and the unrestricted vote-MoE forward (CTRL.M = None: the same
hit-experts-only dispatch as fastmoe, used here only for its expert counters).

Per item we record the decoded text and ids, forwards by kind (prefill /
denoise / flush), device-synchronised milliseconds per forward, unique
(layer, expert) loads per forward, passes and commits per block, peak memory,
and a per-pass trace NPZ (proposed token, confidence, commit mask per pass) so
that CPU replays (state-speculation acceptance, schedule replay) can be run on
every width/threshold without another GPU pass.

Usage (one process per GPU; items are sharded items[shard::nshards]):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/workspace/.hf_home /venv/main/bin/python \
      code/run_research_baselines.py --split calibvalid \
      --cells B32:0.95,B32:0.90 \
      --shard 0 --nshards 3 --out logs/research_v1/baselines
"""
from __future__ import annotations

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                       # sets HF offline env
from research_protocol import (MOE, MODEL_REVISION, MASK_ID, EOS_IDS, GEN_LEN, MODELS, load_sdar,
                               config_hash, safe_out, items_for_split, encode_prompt, LaunchLog)

import numpy as np
import torch

KIND_CODE = {"prefill": 0, "denoise": 1, "kvflush": 2}


def parse_cells(spec: str):
    cells = []
    for tok in [s for s in spec.split(",") if s]:
        b, t = tok.split(":")
        assert b.startswith("B")
        cells.append({"block": int(b[1:]), "threshold": float(t)})
    return cells


def cell_name(ds: str, c: dict, numerics: str):
    return f"{ds}_B{c['block']}_t{int(round(100 * c['threshold']))}_{numerics}"


def load_model(device: str, numerics: str, model_key: str = "b32"):
    from sdar_harness.desvote import CTRL, enable_vote_moe
    tok, model = load_sdar(model_key, device)
    n = enable_vote_moe(model)
    assert n == model.config.num_hidden_layers
    CTRL.M = None; CTRL.force_restrict = False; CTRL.valid = None; CTRL.masked = None
    CTRL.recorded = None; CTRL.use_recorded = False; CTRL.recording = False; CTRL.log = None
    CTRL.reset_cost()
    if numerics == "rowinv":
        from sdar_harness.rowinv import enable_rowinv
        enable_rowinv(model)
    return tok, model


class ForwardMeter:
    """Times every model forward (device-synchronised) and attributes the vote
    controller's expert counters to it. `kind` is set by the decode's on_forward."""

    def __init__(self, model, ctrl):
        self.model, self.ctrl = model, ctrl
        self.kind = "unknown"
        self.records = []              # [kind_code, ms, unique_expert_layer_loads, layer_forwards, n_positions]
        self._orig = model.forward
        model.forward = self._timed

    def on_forward(self, kind, bt, step):
        self.kind = kind

    def _timed(self, *a, **kw):
        dev = self.model.device
        u0, l0 = self.ctrl.unique_sum, self.ctrl.layer_forwards
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        out = self._orig(*a, **kw)
        torch.cuda.synchronize(dev); ms = (time.perf_counter() - t0) * 1e3
        npos = int(kw["input_ids"].shape[1]) if "input_ids" in kw else int(a[0].shape[1])
        self.records.append([KIND_CODE.get(self.kind, -1), round(ms, 3),
                             self.ctrl.unique_sum - u0, self.ctrl.layer_forwards - l0, npos])
        return out

    def reset(self):
        self.records = []
        self.kind = "unknown"

    def restore(self):
        self.model.forward = self._orig


def save_trace(path, res, block):
    """Per-pass proposed / confidence / committed for every decoded block."""
    blocks = res.blocks
    nb = len(blocks)
    if nb == 0:
        return
    Pn = max(b.n_steps for b in blocks)
    proposed = np.full((nb, Pn, block), -1, np.int32)
    conf = np.full((nb, Pn, block), np.nan, np.float32)
    committed = np.zeros((nb, Pn, block), bool)
    valid = np.zeros((nb, block), bool)
    n_pass = np.zeros(nb, np.int32); start = np.zeros(nb, np.int32)
    tokens = np.full((nb, block), -1, np.int32)
    for i, b in enumerate(blocks):
        n_pass[i] = b.n_steps; start[i] = b.start; valid[i] = b.valid.numpy()
        tokens[i] = b.tokens.numpy()
        for s in range(b.n_steps):
            proposed[i, s] = b.proposed[s].numpy(); conf[i, s] = b.confidence[s].numpy()
            committed[i, s] = b.committed[s].numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, proposed=proposed, confidence=conf, committed=committed, valid=valid,
                        n_pass=n_pass, block_start=start, tokens=tokens,
                        prompt_length=np.int32(res.prompt_length), block_length=np.int32(block))


def decode_item(model, tok, meter, item, cell, gen_length, ctrl):
    from sdar_harness.decode import block_diffusion_generate
    ids = encode_prompt(tok, item["prompt"])
    ctrl.reset_cost(); meter.reset()
    dev = model.device
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    res = block_diffusion_generate(model, ids, mask_id=MASK_ID, gen_length=gen_length,
                                   block_length=cell["block"], denoising_steps=cell["block"],
                                   temperature=1.0, top_k=1, top_p=1.0, threshold=cell["threshold"],
                                   stopping_criteria_idx=EOS_IDS, greedy_exact=True,
                                   on_forward=meter.on_forward)
    torch.cuda.synchronize(dev); wall = time.perf_counter() - t0
    gen = res.generated_ids()[0].cpu()
    nonmask = gen[gen != MASK_ID]
    eos_pos = [int(i) for i in torch.nonzero(torch.isin(nonmask, torch.tensor(EOS_IDS))).flatten().tolist()]
    n_to_eos = (eos_pos[0] + 1) if eos_pos else int(nonmask.numel())
    text = tok.decode(nonmask.tolist(), skip_special_tokens=True)
    recs = meter.records
    by_kind = {k: {"n": 0, "ms": 0.0, "uniq": 0, "lf": 0} for k in KIND_CODE}
    for code, ms, u, lf, npos in recs:
        k = {v: k for k, v in KIND_CODE.items()}.get(code, "unknown")
        if k in by_kind:
            by_kind[k]["n"] += 1; by_kind[k]["ms"] += ms; by_kind[k]["uniq"] += u; by_kind[k]["lf"] += lf
    denoise_lf = by_kind["denoise"]["lf"]
    rec = {
        "idx": item["idx"], "rank": item.get("rank"), "split": item["split"],
        "prompt_tokens": int(ids.shape[1]),
        "pred": text, "gold": item["gold"], "gen_ids": nonmask.tolist(),
        "n_gen_nonmask": int(nonmask.numel()), "n_gen_to_eos": n_to_eos,
        "stopped_eos": bool(res.stopped_early),
        "truncated": bool(not res.stopped_early and len(res.blocks) == res.num_blocks - res.prefill_blocks),
        "n_blocks": len(res.blocks), "n_forward": res.n_forward,
        "n_prefill": by_kind["prefill"]["n"], "n_denoise": by_kind["denoise"]["n"], "n_flush": by_kind["kvflush"]["n"],
        "passes_per_block": [b.n_steps for b in res.blocks],
        "n_valid_per_block": [int(b.valid.sum()) for b in res.blocks],
        "commits_per_pass": [int(c.sum()) for b in res.blocks for c in b.committed],
        "unique_expert_layer_loads": ctrl.unique_sum, "layer_forwards": ctrl.layer_forwards,
        "uniq_per_layer_forward_denoise": (by_kind["denoise"]["uniq"] / denoise_lf) if denoise_lf else None,
        "uniq_per_layer_forward_all": (ctrl.unique_sum / ctrl.layer_forwards) if ctrl.layer_forwards else None,
        "seconds": round(wall, 3),
        "ms_by_kind": {k: round(v["ms"], 1) for k, v in by_kind.items()},
        "uniq_by_kind": {k: v["uniq"] for k, v in by_kind.items()},
        "forwards": recs,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
    }
    if "task_id" in item:
        rec["task_id"] = item["task_id"]
    return rec, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="calibvalid", help="calib | valid | calibvalid | gsm8k500 | mbpp")
    ap.add_argument("--cells", default="B32:0.95,B32:0.90")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("logs", "research_v1", "baselines"))
    ap.add_argument("--numerics", default="stock", choices=["stock", "rowinv"])
    ap.add_argument("--model", default="b32", choices=sorted(MODELS), help="checkpoint key (research_protocol.MODELS)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-trace", action="store_true", help="skip the per-pass NPZ traces")
    ap.add_argument("--score", action="store_true", help="score each item inline (CPU, vendored evaluator)")
    args = ap.parse_args()

    out = safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    proto, proto_sha = P.load_protocol()
    cells = parse_cells(args.cells)
    _b = P.split_base(args.split)
    ds = ("mbpp" if _b.startswith("mbpp") else
          "mbpp" if _b.startswith("opencode") else            # opencode_pool: gold = unit-test asserts, MBPP sandbox
          "ifeval" if _b.startswith("if") else                # if_pool AND ifeval541 (corpus-1b): draftalign/if_scorer
          "math500" if _b.startswith("math") else
          "aime" if _b.startswith("aime") else
          "gpqa" if _b.startswith("gpqa") else
          "humaneval" if _b.startswith("humaneval") else
          "finqa" if _b.startswith("finqa") else              # corpus-1b held-out eval (2026-09-28)
          "chartqa" if _b.startswith("chartqa") else
          "chat" if _b.startswith("chat") else
          "gsm8k")
    gen_length = GEN_LEN[args.split]
    items = items_for_split(args.split)
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.nshards]
    scorer = None
    if args.score:
        from analyze_s1b import per_item_correct
        scorer = lambda r: per_item_correct(ds, [r])[r["idx"]]

    torch.manual_seed(0)
    tok, model = load_model(args.device, args.numerics, args.model)
    MODEL_REV = MODELS[args.model]["revision"]
    from sdar_harness.desvote import CTRL
    meter = ForwardMeter(model, CTRL)
    import transformers
    run_meta = {"protocol": os.path.relpath(P.PROTOCOL_PATH, P.ROOT), "protocol_sha256": proto_sha,
                "model": MODELS[args.model]["repo"], "model_key": args.model, "model_revision": MODEL_REV, "numerics": args.numerics,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(model.device),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "split": args.split, "gen_length": gen_length, "shard": args.shard, "nshards": args.nshards}
    print(f"[model] {MODELS[args.model]['repo']} rev {MODEL_REV} on {run_meta['gpu']} ({args.numerics}); "
          f"{len(items)} items of split {args.split}; cells {[cell_name(ds, c, args.numerics) for c in cells]}", flush=True)
    launch = LaunchLog(f"baselines/{args.split}/shard{args.shard}of{args.nshards}",
                       {"cells": cells, **{k: v for k, v in run_meta.items() if k != 'gpu'}})

    for c in cells:
        name = cell_name(ds, c, args.numerics)
        cfg = {"cell": name, "dataset": ds, "split": args.split, "block": c["block"], "steps": c["block"],
               "threshold": c["threshold"], "gen_length": gen_length, "numerics": args.numerics,
               "model_key": args.model, "model_revision": MODEL_REV, "sampling": proto["sampling"], "protocol_sha256": proto_sha}
        chash = config_hash(cfg)
        dest = os.path.join(out, f"{name}.{args.split}.shard{args.shard}of{args.nshards}.jsonl")
        if os.path.exists(dest + ".done"):
            print(f"[skip] {name} shard {args.shard} already done", flush=True)
            continue
        done = set()
        if os.path.exists(dest):
            for line in open(dest):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                                   # truncated last line is redone
                assert r["config_hash"] == chash and r["model_revision"] == MODEL_REV, \
                    f"{dest} holds records from another config; refusing to mix"
                assert r["idx"] not in done, f"duplicate idx {r['idx']} in {dest}"
                done.add(r["idx"])
        t_cell = time.time()
        n_new = 0
        with open(dest, "a") as fh:
            for k, it in enumerate(items):
                if it["idx"] in done:
                    continue
                rec, res = decode_item(model, tok, meter, it, c, gen_length, CTRL)
                rec.update({"cell": name, "config_hash": chash, "model_revision": MODEL_REV, "model_key": args.model,
                            "numerics": args.numerics, "gpu": run_meta["gpu"], "pid": os.getpid(),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if scorer is not None:
                    try:
                        rec["correct"] = int(scorer(rec))
                    except Exception as e:                      # pragma: no cover
                        rec["correct"] = None; rec["score_error"] = repr(e)
                if not args.no_trace:
                    save_trace(os.path.join(out + "_traces", name, f"{it['idx']:05d}.npz"), res, c["block"])
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n_new += 1
                el = time.time() - t_cell
                print(f"[{name}/sh{args.shard}] item {it['idx']} ({k + 1}/{len(items)}) {rec['seconds']:.1f}s "
                      f"fwd {rec['n_forward']} blocks {rec['n_blocks']} gen {rec['n_gen_to_eos']} "
                      f"uniq/lf {rec['uniq_per_layer_forward_denoise'] or 0:.1f} eos {rec['stopped_eos']} "
                      f"correct {rec.get('correct')} | {el / n_new:.0f}s/item", flush=True)
                del res
        open(dest + ".done", "w").write("ok\n")
        print(f"CELL DONE {name} shard {args.shard}: {n_new} new items in {(time.time() - t_cell) / 60:.1f} min", flush=True)
    meter.restore()
    launch.exit_code = 0
    print("ALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
