#!/usr/bin/env python
"""GPU sanity gates for the draft-aligned trainer (TRAINER_SPEC_v0 section 6 gates).

RUN THIS BEFORE ANY TRAINING RUN. It loads the real b32 MoE on one GPU and, on one
real item from a build_train_set index, checks:

  (a) `training_forward` (LoRA at B=0, pool off, no checkpointing) reproduces
      `stock_forward_reference` on the active rows -- max |dlogit| within bf16
      tolerance and the argmax identical at every row;
  (b) the same with layer checkpointing on;
  (c) one optimiser step changes ONLY the LoRA and gate parameters, and the loss
      is finite;
  (d) peak memory and step time at S = 1200;
  (e) with pool M=32 every layer's active rows use at most 32 unique experts.

  CUDA_VISIBLE_DEVICES=4 /venv/main/bin/python code/draftalign/sanity_check.py \
      --index logs/draftalign/trainset/index.jsonl --out results/audits/draftalign/sanity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402
from research_protocol import MASK_ID, LaunchLog, load_sdar, safe_out    # noqa: E402

import torch                                                             # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler               # noqa: E402
from draftalign.losses import active_logits, masked_ce_and_certainty     # noqa: E402
from draftalign.train_forward import stock_forward_reference, training_forward   # noqa: E402


def pick_state(sampler: StateSampler, min_len: int, regime: str = "sample"):
    """The longest example that still fits, chosen from the index metadata (no npz reads)."""
    lens = [int(sampler.entries[i]["block_start"][b]) + sampler.cfg.block
            for i, b in sampler.examples]
    order = sorted(range(len(lens)), key=lambda k: -lens[k])
    k = next((k for k in order if lens[k] >= min_len), order[0])
    return sampler.state(k, regime=regime)


def parity(model, st, checkpoint_layers: bool, device) -> dict:
    ids = st.input_ids.unsqueeze(0).to(device)
    rows = st.open_mask.to(device)
    with torch.no_grad():
        ref = stock_forward_reference(model, ids, logits_to_keep=32).logits[0].float()
        with torch.enable_grad():                       # checkpointing only engages with grad on
            out = training_forward(model, ids, st.active_lo, st.valid, pool_M=None,
                                   checkpoint_layers=checkpoint_layers)
        ours = model.lm_head(out.hidden).float()
    d = (ours - ref).abs()
    return {"checkpoint": checkpoint_layers, "seq_len": int(ids.shape[1]),
            "max_abs_diff_all_rows": float(d.max()),
            "max_abs_diff_loss_rows": float(d[rows].max()) if int(rows.sum()) else 0.0,
            "argmax_equal_all_rows": bool((ours.argmax(-1) == ref.argmax(-1)).all()),
            "n_rows": int(ids.shape[1] and 32)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="results/audits/draftalign/sanity.json")
    ap.add_argument("--model", default="b32")
    ap.add_argument("--rank", dest="lora_r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--pool-M", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1200)
    ap.add_argument("--tol", type=float, default=0.25, help="bf16 logit tolerance")
    args = ap.parse_args(argv)

    device = torch.device("cuda:0")
    LaunchLog("draftalign/sanity_check", vars(args))
    sampler = StateSampler(SamplerConfig(index_path=args.index, max_seq=args.seq))
    st = pick_state(sampler, args.seq - 32)
    print(f"[state] item {st.meta['idx']} block {st.meta['block']} S={st.meta['seq_len']} "
          f"open {st.meta['n_open']} draft-filled {st.meta['n_draft_filled']}", flush=True)

    tok, model = load_sdar(args.model, str(device))                       # noqa: F841
    A.attach_lora(model, r=args.lora_r, alpha=args.alpha)
    A.freeze_base(model)
    model.eval()
    model.config.use_cache = False
    rep = {"state": st.meta, "trainable": A.count_trainable(model)}

    # (a) + (b) parity, LoRA at B = 0 and the pool off
    rep["parity_nockpt"] = parity(model, st, False, device)
    print("[parity no-ckpt]", json.dumps(rep["parity_nockpt"]), flush=True)
    rep["parity_ckpt"] = parity(model, st, True, device)
    print("[parity ckpt]", json.dumps(rep["parity_ckpt"]), flush=True)
    for key in ("parity_nockpt", "parity_ckpt"):
        assert rep[key]["argmax_equal_all_rows"], f"{key}: argmax differs from the stock forward"
        assert rep[key]["max_abs_diff_all_rows"] <= args.tol, f"{key}: {rep[key]}"

    # (e) pool restriction really restricts
    ids = st.input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = training_forward(model, ids, st.active_lo, st.valid, pool_M=args.pool_M,
                               checkpoint_layers=False, return_pools=True)
    rep["pool"] = {"M": args.pool_M,
                   "max_unique_active": int(out.unique_experts.max()),
                   "mean_unique_active": float(out.unique_experts.float().mean()),
                   "pool_sizes": sorted(set(out.pools.sum(1).tolist()))}
    assert rep["pool"]["max_unique_active"] <= args.pool_M, rep["pool"]

    with torch.no_grad():
        off = training_forward(model, ids, st.active_lo, st.valid, pool_M=None, checkpoint_layers=False)
    rep["pool"]["mean_unique_active_pool_off"] = float(off.unique_experts.float().mean())

    # (c) + (d) one optimiser step: only adapter parameters move, loss finite, cost measured
    #   the adapter tensors are small enough to clone (~150 MB); the 61 GB of base weights
    #   are guarded by a per-tensor checksum over a deterministic sample plus the
    #   requires_grad / .grad invariants.
    model.train()
    before = {n: p.detach().clone() for n, p in A.trainable_parameters(model)}
    base = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    g = torch.Generator().manual_seed(0)
    sample = [base[int(i)] for i in torch.randperm(len(base), generator=g)[:256]]
    base_sum_before = {n: float(p.detach().float().sum()) for n, p in sample}
    opt = torch.optim.AdamW(A.parameter_groups(model, 2e-5, 1e-5), betas=(0.9, 0.95))
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    rows = st.open_mask.to(device)
    targets = st.targets.to(device)[rows]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        o = training_forward(model, ids, st.active_lo, st.valid, pool_M=args.pool_M,
                             checkpoint_layers=True)
    logits = active_logits(model, o.hidden, rows)
    loss, stats = masked_ce_and_certainty(logits, targets, mask_id=MASK_ID)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_([p for _, p in A.trainable_parameters(model)], 1.0))
    opt.step()
    torch.cuda.synchronize(device)
    step_s = time.perf_counter() - t0

    moved = [n for n, p in A.trainable_parameters(model) if not torch.equal(p.detach(), before[n])]
    base_changed = [n for n, p in sample if float(p.detach().float().sum()) != base_sum_before[n]]
    base_with_grad = [n for n, p in base if p.grad is not None]
    rep["step"] = {"loss": float(loss), "finite": bool(torch.isfinite(loss)),
                   "ce": float(stats["ce"]), "cert": float(stats["cert"]),
                   "acc": float(stats["acc"]), "grad_norm": gnorm,
                   "n_trainable_tensors": len(before), "n_moved_adapter": len(moved),
                   "n_base_tensors": len(base), "n_base_sampled": len(sample),
                   "base_changed": base_changed[:8], "base_with_grad": base_with_grad[:8],
                   "seconds": round(step_s, 3),
                   "peak_mem_gib": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)}
    assert rep["step"]["finite"], "loss is not finite"
    assert not base_changed, f"a frozen base tensor moved: {base_changed[:8]}"
    assert not base_with_grad, f"a frozen base tensor got a gradient: {base_with_grad[:8]}"
    # LoRA A receives a zero gradient on the first step (B is zero-initialised), so only B and the gates must move
    must_move = [n for n, _ in A.trainable_parameters(model) if "lora_A" not in n]
    not_moved = [n for n in must_move if n not in set(moved)]
    rep["step"]["must_move_not_moved"] = not_moved[:8]
    assert not not_moved, f"adapter tensors that should have moved did not: {not_moved[:8]}"
    print("[step]", json.dumps(rep["step"]), flush=True)

    path = safe_out(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str), flush=True)
    print(f"[ok] all gates passed -> {path}", flush=True)
    return rep


if __name__ == "__main__":
    main()
