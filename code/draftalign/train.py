#!/usr/bin/env python
"""Draft-aligned LoRA training for SDAR-30B-A3B-Chat-b32 (TRAINER_SPEC_v0 section 6).

One FULL bf16 model per rank (61 GB of weights; no sharding), batch 1 state per
micro-step, gradient accumulation, and a manual all-reduce of the ~100 M adapter
gradients at the end of every optimiser step. DDP is deliberately NOT used: our
forward bypasses `model.forward` (see `train_forward.py`), so DDP's autograd hooks
would never see the graph; the explicit reduce is both simpler and correct.

    # 4 GPUs (4-7), the spec's batch 1 x accum 8 x 4 = 32 states per optimiser step
    CUDA_VISIBLE_DEVICES=4,5,6,7 /venv/main/bin/torchrun --standalone --nproc_per_node=4 \
        code/draftalign/train.py --index logs/draftalign/trainset/index.jsonl \
        --out logs/draftalign/train/t1 --steps 1250 --grad-accum 8 \
        --pool-Ms 24,32,48,128 --max-seq 1200

    # 8 GPUs: halve the accumulation to keep the same global batch of 32
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /venv/main/bin/torchrun --standalone --nproc_per_node=8 \
        code/draftalign/train.py --index logs/draftalign/trainset/index.jsonl \
        --out logs/draftalign/train/t1 --steps 1250 --grad-accum 4 \
        --pool-Ms 24,32,48,128 --max-seq 1200

Every rank consumes a disjoint slice of the shuffled example order, so the global
batch of one optimiser step is `world_size * grad_accum` distinct states.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # code/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "compat"))
import research_protocol as P                                            # noqa: E402
from research_protocol import MASK_ID, LaunchLog, config_hash, load_sdar, safe_out   # noqa: E402

import torch                                                             # noqa: E402
import numpy as np                                                       # noqa: E402
import torch.distributed as dist                                         # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler               # noqa: E402
from draftalign.losses import active_logits, eval_row_stats, masked_ce_and_certainty   # noqa: E402
from draftalign import calib_metrics as CM                                  # noqa: E402
from draftalign.train_forward import training_forward                    # noqa: E402
from draftalign import wandb_log                                        # noqa: E402


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------
def dist_init() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local)
    return {"rank": rank, "world": world, "local_rank": local}


def all_reduce_grads(params: List[torch.nn.Parameter], world: int) -> None:
    """Average the adapter gradients across ranks (flattened into one bucket)."""
    if world <= 1:
        return
    for p in params:                       # a missing grad on one rank only would desync the bucket
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)
    for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(s)


def reduce_stats(d: Dict[str, float], world: int, device) -> Dict[str, float]:
    if world <= 1:
        return d
    keys = sorted(d)
    t = torch.tensor([d[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world
    return {k: float(v) for k, v in zip(keys, t.tolist())}


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def lr_scale(step: int, warmup: int, total: int, min_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_frac` of the peak (spec section 6)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))


def pool_for_step(step: int, pool_Ms: List[Optional[int]], seed: int) -> Optional[int]:
    """The step's pool M, drawn from `pool_Ms` with a fixed seed (identical on every rank)."""
    g = torch.Generator().manual_seed((seed * 1000003 + step) % (2 ** 63 - 1))
    return pool_Ms[int(torch.randint(len(pool_Ms), (1,), generator=g))]


# ---------------------------------------------------------------------------
# one micro-step
# ---------------------------------------------------------------------------
def state_draft_flag(st, mask_id: int, block: int = 32) -> torch.Tensor:
    """[S] bool: the TENTATIVE DRAFT slots of a sampled state -- open (valid & uncommitted) and
    holding a token that is not [MASK].  Committed slots, masked slots and the whole prefix
    (including the prompt tail of the first generated block) are unflagged, so an all-mask state
    carries no flag at all and is therefore exactly the stock model."""
    S = int(st.input_ids.numel())
    flag = torch.zeros(S, dtype=torch.bool)
    lo = int(st.active_lo)
    blk = st.input_ids[lo:lo + block]
    flag[lo:lo + block] = st.open_mask & st.valid & (blk != mask_id)
    return flag


def slot_override_mask(st, mask_id: int, block: int = 32) -> torch.Tensor:
    """[n_open] bool: open slots that hold a draft token which is NOT the target (the judge must override)."""
    lo = int(st.active_lo)
    inp = st.input_ids[lo:lo + block][st.open_mask]
    return (inp != mask_id) & (inp != st.targets[st.open_mask])


def override_row_weight(st, override: torch.Tensor, override_weight: float = 1.0,
                        code_override_weight: Optional[float] = None) -> Optional[torch.Tensor]:
    """Per-row loss weight: 1, or `override_weight` on override rows (`code_override_weight` for code-pool
    items). None when every weight would be 1, so the r3_A loss path is untouched."""
    w = float(override_weight)
    if code_override_weight is not None and str(st.meta.get("split", "")).startswith("opencode"):
        w = float(code_override_weight)
    if w == 1.0:
        return None
    return torch.where(override, torch.tensor(w), torch.tensor(1.0))


def forward_state(model, st, pool_M, pool_rule, checkpoint_layers, beta, temp, mask_id,
                  device, want_eval: bool = False, block: int = 32, draft_flag: bool = False,
                  calib: float = 0.0, override_weight: float = 1.0,
                  code_override_weight: Optional[float] = None):
    """Forward + loss for one sampled state. Returns (loss, stats, logits_fp32, rows)."""
    ids = st.input_ids.unsqueeze(0).to(device, non_blocking=True)
    rows = st.open_mask.to(device)
    targets = st.targets.to(device)[rows]
    flag = state_draft_flag(st, mask_id, block) if draft_flag else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = training_forward(model, ids, st.active_lo, st.valid, pool_M=pool_M,
                               pool_rule=pool_rule, checkpoint_layers=checkpoint_layers,
                               block=block, draft_flag=flag)
    logits = active_logits(model, out.hidden, rows)                # fp32, only the loss rows
    override = slot_override_mask(st, mask_id, block)
    row_w = override_row_weight(st, override, override_weight, code_override_weight)
    loss, stats = masked_ce_and_certainty(logits, targets, beta=beta, temp=temp, mask_id=mask_id, calib=calib,
                                          row_weight=None if row_w is None else row_w.to(device))
    stats["uniq_experts"] = out.unique_experts.float().mean()
    if want_eval:
        stats.update({f"ev_{k}": v for k, v in eval_row_stats(logits, targets, override=override.to(device)).items()})
    return loss, stats, logits, rows


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, sampler: StateSampler, n: int, pool_M, pool_rule, mask_id, device,
             tau: float = 0.95, seed: int = 0, block: int = 32,
             draft_flag: bool = False) -> Dict[str, float]:
    """Fixed states: pass-1 draft-filled and all-mask; argmax accuracy and top1>tau precision."""
    model.eval()
    if len(sampler) == 0:
        return {}
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(sampler), generator=g)[:n].tolist()
    acc: Dict[str, List[float]] = {}
    for regime in ("draft_pass1", "allmask_pass1"):
        tot = {"rows": 0.0, "correct": 0.0, "conf": 0.0, "conf_correct": 0.0, "top1": 0.0, "n": 0.0,
               "ovr": 0.0, "ovr_correct": 0.0, "conf_ovr": 0.0, "conf_correct_ovr": 0.0}
        for k in pick:
            st = sampler.state(k, regime=regime)
            if regime == "draft_pass1" and not st.meta["has_draft"]:
                continue
            _, stats, logits, _ = forward_state(model, st, pool_M, pool_rule, False, 0.0, 0.5,
                                                mask_id, device, want_eval=True, block=block,
                                                draft_flag=draft_flag)
            tot["rows"] += float(stats["ev_n_rows"]); tot["n"] += 1
            tot["correct"] += float(stats["ev_acc"]) * float(stats["ev_n_rows"])
            tot["conf"] += float(stats["ev_n_conf"]); tot["conf_correct"] += float(stats["ev_conf_correct"])
            tot["top1"] += float(stats["ev_mean_top1"])
            tot["ovr"] += float(stats["ev_n_ovr"]); tot["ovr_correct"] += float(stats["ev_ovr_correct"])
            tot["conf_ovr"] += float(stats["ev_n_conf_ovr"]); tot["conf_correct_ovr"] += float(stats["ev_conf_correct_ovr"])
        if tot["rows"]:
            acc[f"{regime}_acc"] = [tot["correct"] / tot["rows"]]
            acc[f"{regime}_top1"] = [tot["top1"] / max(1.0, tot["n"])]
            acc[f"{regime}_conf_frac"] = [tot["conf"] / tot["rows"]]
            acc[f"{regime}_conf_precision"] = [tot["conf_correct"] / tot["conf"] if tot["conf"] else float("nan")]
            acc[f"{regime}_states"] = [tot["n"]]
            if regime == "draft_pass1":      # override rows: draft present and wrong (checkpoint selection)
                nan = float("nan")
                acc[f"{regime}_ovr_acc"] = [tot["ovr_correct"] / tot["ovr"] if tot["ovr"] else nan]
                acc[f"{regime}_ovr_conf_frac"] = [tot["conf_ovr"] / tot["ovr"] if tot["ovr"] else nan]
                acc[f"{regime}_ovr_conf_precision"] = [tot["conf_correct_ovr"] / tot["conf_ovr"] if tot["conf_ovr"] else nan]
    return {k: v[0] for k, v in acc.items()}


CUM_KEYS = ("loss", "ce", "cert", "calib")


@torch.no_grad()
def calib_eval(model, sampler, n_sample: int, n_extra: int, pool_M, pool_rule, mask_id, device, rank: int, world: int,
               step: int, out: str, wb=None, seed: int = 0, block: int = 32, draft_flag: bool = False,
               train_hist=None, forward=None):
    """Holdout calibration eval (calib_metrics.py), sharded over ranks: every rank decodes ITS slice of a fixed
    deterministic state list and ALWAYS joins the all_gather (errors travel in the payload, so no rank can hang).
    Rank 0 merges, analyses, writes <out>/calib/stepNNNNNN.json + PNGs and logs scalars/images to W&B."""
    t0 = time.time()
    payload = {"slots": None, "joint": [], "error": None, "n_states": 0}
    if model is not None:
        model.eval()
    try:
        g = torch.Generator().manual_seed(int(seed) + 7919)
        perm = torch.randperm(len(sampler), generator=g).tolist()
        jobs = [(k, "sample") for k in perm[:n_sample * world][rank::world]]
        extra = perm[n_sample * world:(n_sample + n_extra) * world][rank::world]
        jobs += [(k, "draft_pass1") for k in extra] + [(k, "allmask_pass1") for k in extra]
        parts, joint = [], []
        for k, regime in jobs:
            st = sampler.state(k, regime=regime)
            if regime == "draft_pass1" and not st.meta["has_draft"]:
                continue
            if forward is not None:
                logits = forward(st)
            else:
                logits = forward_state(model, st, pool_M, pool_rule, False, 0.0, 0.5, mask_id, device,
                                       block=block, draft_flag=draft_flag)[2]
            lo = int(st.active_lo)
            conf, cor, nll, kind, t = CM.slot_view(logits, st.targets[st.open_mask].to(logits.device),
                                                   st.input_ids[lo:lo + block], st.open_mask, st.valid, mask_id)
            sl, jt = CM.state_record(conf, cor, nll, kind, t, regime)
            parts.append(sl); joint.extend(jt)
        if parts:
            payload["slots"] = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        payload["joint"] = joint; payload["n_states"] = len(parts)
    except Exception as e:                                                   # noqa: BLE001
        payload["error"] = f"rank {rank}: {type(e).__name__}: {e}"
    if world > 1:
        got = [None] * world
        dist.all_gather_object(got, payload)
    else:
        got = [payload]
    if model is not None:
        model.train()
    if rank != 0:
        return None
    errs = [p["error"] for p in got if p and p.get("error")]
    try:
        slots, joint = CM.merge_payloads([p for p in got if p and p.get("slots")])
        res = CM.analyze(slots, joint, step)
        res["n_states"] = int(sum(p.get("n_states", 0) for p in got if p)); res["errors"] = errs
        res["seconds"] = round(time.time() - t0, 1)
        cdir = os.path.join(out, "calib")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, f"step{step:06d}.json"), "w") as fh:
            json.dump(res, fh)
        sc = CM.wandb_scalars(res)
        if wb is not None:
            wb.log(wandb_log.prefixed(sc, "calib"), step=step)
        try:
            paths = CM.plot_all(res, cdir, train_hist)
        except Exception as e:                                               # noqa: BLE001
            paths = {}; print(f"[calib-eval {step}] plotting failed: {type(e).__name__}: {e}", flush=True)
        perr = {k: v for k, v in paths.items() if k.startswith("error_")}
        paths = {k: v for k, v in paths.items() if not k.startswith("error_")}
        m = res["masked_by_t"]
        print(f"[calib-eval {step}] {res['n_states']} states {res['n_slots']} slots in {res['seconds']}s | all ECE "
              f"{res['all_open'].get('ece', float('nan')):.4f} gap {res['all_open'].get('gap', float('nan')):+.4f} | masked ECE by t "
              + " ".join(f"{g.get('ece', float('nan')):.3f}" for g in m) + " | joint gap (tau rule, k buckets) "
              + " ".join(f"{r['gap']:+.3f}" for r in res["joint"]["tau_rule"]) + (f" | ERRORS {errs}" if errs else "")
              + (f" | PLOT ERRORS {perr}" if perr else ""), flush=True)
        if wb is not None and paths:
            if getattr(wb, "active", False) and hasattr(wb, "run"):
                try:
                    import wandb
                    wb.run.log({f"calib_plots/{k}": wandb.Image(v) for k, v in paths.items()}, step=step)
                except Exception as e:                                       # noqa: BLE001
                    print(f"[calib-eval {step}] W&B image log failed: {e}", flush=True)
        return res
    except Exception as e:                                                   # noqa: BLE001
        print(f"[calib-eval {step}] analysis failed: {type(e).__name__}: {e} | rank errors {errs}", flush=True)
        return None


def cum_init(path: str, start_step: int) -> Dict[str, float]:
    """Running means of the logged loss terms up to `start_step` (so a resumed run continues the cumulative curves)."""
    cum = {"n": 0.0, **{k: 0.0 for k in CUM_KEYS}}
    if not start_step or not os.path.exists(path):
        return cum
    seen = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:                                                    # noqa: BLE001
            continue
        if r.get("event") or "loss" not in r or int(r.get("step", 0)) > start_step or r["step"] in seen:
            continue
        seen.add(r["step"]); cum["n"] += 1
        for k in CUM_KEYS:
            cum[k] += float(r.get(k, 0.0))
    return cum


# ---------------------------------------------------------------------------
# the --draft-flag step-0 identity check
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_flag_identity(model, sampler: StateSampler, mask_id: int, block: int = 32,
                        n_states: int = 8) -> Dict[str, object]:
    """Cheap proof that with `--draft-flag` the all-mask regime IS the stock model.

    Two facts, both checked without a second 30B forward (one such forward costs ~5 s and
    ~58 GiB, and a bit-for-bit comparison against stock would need a second full model):

      1. every `allmask_pass1` state's flag is all-False -- no slot is flagged, so no
         `draft_embed` is added anywhere (`0 * vec`, and the fp32 round trip is lossless);
      2. under an all-False flag a real `LoRALinear` of this model returns its frozen base
         output BIT-FOR-BIT (`adapter.LoRALinear.forward` skips the delta entirely), on a
         random input in the model's own dtype.

    Together: an all-mask forward touches no adapter parameter except the router gates, so it
    equals the stock model exactly when the gates are frozen (`--no-train-gates`); with trained
    gates the router differs by design and this check does not claim otherwise.
    """
    flagged = 0
    for k in range(min(int(n_states), len(sampler))):
        st = sampler.state(k, regime="allmask_pass1")
        flagged += int(state_draft_flag(st, mask_id, block).sum())
    assert flagged == 0, f"{flagged} slots flagged in an all-mask state"

    lora = next(m for m in model.modules() if isinstance(m, A.LoRALinear))
    x = torch.randn(1, 4, lora.in_features, device=lora.base.weight.device, dtype=lora.base.weight.dtype)
    off = torch.zeros(1, 4, dtype=torch.bool, device=x.device)
    with A.flag_scope(off, False):
        gated = lora(x)
    base = lora.base(x)
    assert torch.equal(gated, base), "an all-False flag changed a LoRA projection"

    de = A.get_draft_embed(model)
    return {"states_checked": min(int(n_states), len(sampler)), "flagged_slots": flagged,
            "lora_identity_at_zero_flag": True,
            "draft_embed_zero": None if de is None else bool(torch.count_nonzero(de.weight) == 0)}


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="build_train_set index.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="b32")
    ap.add_argument("--steps", type=int, default=1250, help="optimiser steps")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=2e-5)
    ap.add_argument("--lr-gate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", dest="lora_r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--no-train-gates", dest="train_gates", action="store_false", default=True)
    ap.add_argument("--draft-flag", action="store_true", default=False,
                    help="round 2: flag the tentative draft slots (a trained zero-init embedding added to "
                         "their input embeddings) and GATE the LoRA delta to those slots; off = round 1")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--calib", type=float, default=0.0,
                    help="weight of the calibration BCE(top-1 prob, argmax correct): penalises confident wrong rows (0 = off, r3_A)")
    ap.add_argument("--override-weight", type=float, default=1.0,
                    help="loss weight on OVERRIDE rows (draft token present and wrong); 1 = off (r3_A)")
    ap.add_argument("--code-override-weight", type=float, default=None,
                    help="override-row weight for code-pool (opencode) items; default = --override-weight")
    ap.add_argument("--init-adapter", default=None,
                    help="start a FRESH run (step 0, new optimizer + lr schedule) from this adapter's weights")
    ap.add_argument("--calib-log", action="store_true",
                    help="per log interval: confidence histogram + per-bin accuracy (ECE), CE/acc/conf per slot kind, "
                         "weighted loss terms (calib_metrics.py); logging only, training is unchanged")
    ap.add_argument("--calib-eval-states", type=int, default=0,
                    help="per-rank holdout states (mixed mask ratios) for the calibration eval run at every --eval-every "
                         "(ECE/NLL vs t, reliability grid, joint calibration); 0 = off")
    ap.add_argument("--calib-eval-extra", type=int, default=16,
                    help="per-rank extra holdout states, each run as draft_pass1 AND allmask_pass1, for the calibration eval")
    ap.add_argument("--calib-eval-at-start", action="store_true", help="also run the calibration eval before the first step")
    ap.add_argument("--pool-Ms", default="24,32,48,128", help="128 (or >= num_experts) means off")
    ap.add_argument("--pool-rule", choices=("full", "top8"), default="full")
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=1200)
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--p-pass1", type=float, default=0.4)
    ap.add_argument("--p-allmask", type=float, default=0.2)
    ap.add_argument("--p-draftfull", type=float, default=0.0, help="share of draft states with EVERY open slot draft-filled (the pass-1 inference state)")
    ap.add_argument("--holdout-items", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-states", type=int, default=64)
    ap.add_argument("--eval-pool-M", type=int, default=0, help="0 = pool off during eval")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--keep-checkpoints", type=int, default=2,
                    help="keep only the N newest step directories in --out (0 = keep every one); the "
                         "checkpoint named in `latest` is never deleted")
    ap.add_argument("--resume", default=None, help="checkpoint directory to resume from")
    ap.add_argument("--dry-run", action="store_true", help="build everything, load no model")
    return ap


def parse_pool_Ms(spec: str, num_experts: int) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for tok in [s for s in spec.split(",") if s.strip()]:
        m = int(tok)
        out.append(None if m <= 0 or m >= num_experts else m)
    return out or [None]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    info = dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = safe_out(args.out)
    os.makedirs(out, exist_ok=True)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"world_size": world, "model_revision": P.MODELS[args.model]["revision"],
                "decoder": "draftalign_train_v0"})
    chash = config_hash(cfg)
    if rank == 0:
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump({"config": cfg, "config_hash": chash}, fh, indent=2, sort_keys=True)
    if not args.dry_run:
        launch = LaunchLog(f"draftalign/train/{os.path.basename(out)}/rank{rank}", cfg)   # noqa: F841

    # -- data ------------------------------------------------------------
    scfg = SamplerConfig(index_path=args.index, seed=args.seed, p_pass1=args.p_pass1,
                         p_allmask=args.p_allmask, p_draftfull=args.p_draftfull, mask_id=MASK_ID, max_seq=args.max_seq,
                         only_correct=args.only_correct)
    full = StateSampler(scfg)
    train_s, eval_s = full.split_holdout(args.holdout_items)
    if rank == 0:
        print(f"[data] {full.summary()} -> train {len(train_s)} / eval {len(eval_s)} examples", flush=True)
    if args.dry_run:
        return

    # -- model -----------------------------------------------------------
    wb = wandb_log.init(os.path.basename(out), cfg, out, group="draftalign-round2b", job_type="judge-lora",
                        tags=("judge", args.model), rank=rank,
                        notes=(f"resume from {args.resume}" if args.resume else None))
    torch.manual_seed(args.seed)
    tok, model = load_sdar(args.model, str(device))                       # noqa: F841
    counts = A.attach_lora(model, r=args.lora_r, alpha=args.alpha, dropout=args.lora_dropout,
                           train_gates=args.train_gates, draft_flag=args.draft_flag)
    A.freeze_base(model)
    model.train()
    model.config.use_cache = False
    params = [p for _, p in A.trainable_parameters(model)]
    if rank == 0:
        print(f"[model] wrapped {counts}, trainable {A.count_trainable(model)}", flush=True)

    opt = torch.optim.AdamW(A.parameter_groups(model, args.lr_lora, args.lr_gate, args.weight_decay),
                            betas=(0.9, 0.95), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    pool_Ms = parse_pool_Ms(args.pool_Ms, int(model.config.num_experts))
    eval_M = args.eval_pool_M if args.eval_pool_M > 0 else None

    if args.init_adapter:
        assert not args.resume, "--init-adapter starts a fresh run; use --resume to continue one"
        got = A.load_adapter(model, args.init_adapter)
        if rank == 0:
            print(f"[init-adapter] {args.init_adapter}: {got['loaded']} tensors; fresh optimizer + schedule from step 0",
                  flush=True)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, opt, safe_out(args.resume), device)
        if rank == 0:
            print(f"[resume] from step {start_step}", flush=True)

    if args.draft_flag:
        chk = check_flag_identity(model, eval_s if len(eval_s) else train_s, MASK_ID, args.max_seq)
        if rank == 0:
            print(f"[draft-flag] step-0 identity check {chk}", flush=True)

    tacc = CM.train_accum_zeros(device) if args.calib_log else None
    hist_since_eval = None
    cum = cum_init(os.path.join(out, f"train.rank{rank}.jsonl"), start_step)
    if args.calib_eval_at_start and args.calib_eval_states > 0:
        calib_eval(model, eval_s, args.calib_eval_states, args.calib_eval_extra, eval_M, args.pool_rule, MASK_ID,
                   device, rank, world, start_step, out, wb=wb, seed=args.seed, draft_flag=args.draft_flag)

    log_path = os.path.join(out, f"train.rank{rank}.jsonl")
    logf = open(log_path, "a")
    t_last, n_tok, agg = time.time(), 0, {}

    for step in range(start_step, args.steps):
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_scale(step, args.warmup, args.steps, args.min_lr_frac)
        pool_M = pool_for_step(step, pool_Ms, args.seed)
        opt.zero_grad(set_to_none=True)
        step_stats: Dict[str, float] = {}
        for j in range(args.grad_accum):
            gi = (step * args.grad_accum + j) * world + rank
            epoch = gi // max(1, len(train_s))
            st = train_s.global_state(gi, epoch=epoch)
            loss, stats, logits_, _ = forward_state(model, st, pool_M, args.pool_rule, args.checkpoint,
                                              args.beta, args.temp, MASK_ID, device,
                                              draft_flag=args.draft_flag, calib=args.calib,
                                              override_weight=args.override_weight,
                                              code_override_weight=args.code_override_weight)
            (loss / args.grad_accum).backward()
            if tacc is not None:
                try:
                    lo_ = int(st.active_lo)
                    cf_, cr_, nl_, kd_, _ = CM.slot_view(logits_.detach(), st.targets[st.open_mask].to(device),
                                                         st.input_ids[lo_:lo_ + 32], st.open_mask, st.valid, MASK_ID)
                    CM.train_accum_add(tacc, cf_, cr_, nl_, kd_)
                except Exception as e:                                       # noqa: BLE001
                    print(f"[calib-log] rank {rank} step {step + 1}: {type(e).__name__}: {e}", flush=True)
            del logits_
            n_tok += int(st.meta["seq_len"])
            for k, v in stats.items():
                step_stats[k] = step_stats.get(k, 0.0) + float(v) / args.grad_accum
            step_stats["loss"] = step_stats.get("loss", 0.0) + float(loss.detach()) / args.grad_accum

        all_reduce_grads(params, world)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        step_stats["grad_norm"] = gnorm
        for k, v in step_stats.items():
            agg[k] = agg.get(k, 0.0) + v

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            row = {k: v / n for k, v in agg.items()}
            row = reduce_stats(row, world, device)
            row["loss_cert_term"] = args.beta * row.get("cert", 0.0)
            row["loss_calib_term"] = args.calib * row.get("calib", 0.0)
            cum["n"] += 1
            for k_ in CUM_KEYS:
                cum[k_] += float(row.get(k_, 0.0)); row[f"cum_{k_}"] = cum[k_] / cum["n"]
            if tacc is not None:
                if world > 1:
                    dist.all_reduce(tacc, op=dist.ReduceOp.SUM)
                tsum = CM.train_accum_summary(tacc)
                tacc.zero_()
                hist_since_eval = hist_since_eval or {}
                for k_ in ("conf_hist_n", "conf_hist_correct", "conf_hist_conf"):
                    prev_ = hist_since_eval.get(k_, [0.0] * len(tsum[k_]))
                    hist_since_eval[k_] = [a_ + b_ for a_, b_ in zip(prev_, tsum[k_])]
                row.update(tsum)
                if rank == 0 and getattr(wb, "active", False) and hasattr(wb, "run"):
                    try:
                        import wandb
                        wb.run.log({"train/conf_hist": wandb.Histogram(np_histogram=(
                            np.array(tsum["conf_hist_n"]), np.linspace(0, 1, CM.N_TRAIN_BINS + 1)))}, step=step + 1)
                    except Exception as e:                                   # noqa: BLE001
                        print(f"[calib-log] W&B histogram failed: {e}", flush=True)
            dt = time.time() - t_last
            row.update({"step": step + 1, "lr": opt.param_groups[0]["lr"], "pool_M": pool_M,
                        "tokens_per_s": n_tok / max(dt, 1e-9), "sec_per_step": dt / n,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            if rank == 0:
                print(f"[step {step + 1}/{args.steps}] loss {row['loss']:.4f} ce {row['ce']:.4f} "
                      f"cert {row['cert']:.4f} cal {row.get('calib', 0.0):.4f} acc {row['acc']:.3f} top1 {row['top1']:.3f} "
                      f"uniq {row['uniq_experts']:.1f} M {pool_M} lr {row['lr']:.2e} "
                      f"{row['tokens_per_s']:.0f} tok/s | cum {row['cum_loss']:.4f}"
                      + (f" ece {row.get('ece_train', float('nan')):.4f} ce[mask/conf/ovr] {row.get('ce_mask', float('nan')):.3f}/"
                         f"{row.get('ce_confirm', float('nan')):.3f}/{row.get('ce_override', float('nan')):.3f}" if tacc is not None else ""),
                      flush=True)
            logf.write(json.dumps(row) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(row, "train"), step=step + 1)
            agg, n_tok, t_last = {}, 0, time.time()

        if args.eval_every and (step + 1) % args.eval_every == 0:
            ev = evaluate(model, eval_s, args.eval_states, eval_M, args.pool_rule, MASK_ID, device,
                          seed=args.seed, draft_flag=args.draft_flag)
            model.train()
            ev = reduce_stats(ev, world, device)
            ev.update({"step": step + 1, "event": "eval"})
            if rank == 0:
                print(f"[eval {step + 1}] " + " ".join(f"{k}={v:.4f}" for k, v in sorted(ev.items())
                                                       if isinstance(v, float)), flush=True)
            logf.write(json.dumps(ev) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(ev, "eval"), step=step + 1)
            if args.calib_eval_states > 0:
                calib_eval(model, eval_s, args.calib_eval_states, args.calib_eval_extra, eval_M, args.pool_rule, MASK_ID,
                           device, rank, world, step + 1, out, wb=wb, seed=args.seed, draft_flag=args.draft_flag,
                           train_hist=hist_since_eval)
                hist_since_eval = None

        if args.save_every and (step + 1) % args.save_every == 0 and rank == 0:
            save_checkpoint(model, opt, out, step + 1, keep=args.keep_checkpoints)
            wb.log({"checkpoint/step": step + 1}, step=step + 1)

    if rank == 0:
        save_checkpoint(model, opt, out, args.steps, keep=args.keep_checkpoints)
    wb.finish()
    logf.close()
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
CKPT_RE = re.compile(r"^step(\d{6})$")


def checkpoint_dirs(out: str) -> List[str]:
    """The `stepNNNNNN` directories of `out`, oldest first."""
    if not os.path.isdir(out):
        return []
    got = [(int(m.group(1)), n) for n in os.listdir(out)
           for m in [CKPT_RE.match(n)] if m and os.path.isdir(os.path.join(out, n))]
    return [n for _, n in sorted(got)]


def prune_checkpoints(out: str, keep: int) -> List[str]:
    """Delete all but the `keep` newest step directories. Never deletes the one named in `latest`.
    `keep <= 0` keeps everything. Returns the names removed."""
    if keep is None or int(keep) <= 0:
        return []
    names = checkpoint_dirs(out)
    latest = None
    lp = os.path.join(out, "latest")
    if os.path.isfile(lp):
        latest = open(lp).read().strip() or None
    removed = []
    for n in names[:max(0, len(names) - int(keep))]:
        if n == latest:
            continue
        shutil.rmtree(os.path.join(out, n), ignore_errors=True)
        removed.append(n)
    return removed


def save_checkpoint(model, opt, out: str, step: int, keep: int = 0) -> str:
    d = os.path.join(out, f"step{step:06d}")
    A.save_adapter(model, d)
    torch.save({"step": step, "optimizer": opt.state_dict()}, os.path.join(d, "trainer_state.pt"))
    with open(os.path.join(out, "latest"), "w") as fh:
        fh.write(os.path.basename(d))
    gone = prune_checkpoints(out, keep)                 # only AFTER this save succeeded
    print(f"[save] {d}" + (f" (pruned {', '.join(gone)})" if gone else ""), flush=True)
    return d


def load_checkpoint(model, opt, path: str, device) -> int:
    if os.path.isfile(os.path.join(path, "latest")):
        path = os.path.join(path, open(os.path.join(path, "latest")).read().strip())
    A.load_adapter(model, path)
    state = torch.load(os.path.join(path, "trainer_state.pt"), map_location=device, weights_only=False)
    opt.load_state_dict(state["optimizer"])
    return int(state["step"])


if __name__ == "__main__":
    main()
