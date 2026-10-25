#!/usr/bin/env python
"""Phase 4: train a LoRA adapter on the SDAR-1.7B drafter on the 30B's own answers (docs/PHASE4_DRAFTER_LORA.md).

    # the full pass, 8 GPUs (from the repository root; code/drafter_lora/p4.sh train wraps this)
    torchrun --standalone --nproc_per_node=8 code/drafter_lora/train_drafter_lora.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --out logs/phase4/train/main

    # the smoke test: 2,000 items, 200 steps (p4.sh smoke)
    torchrun ... train_drafter_lora.py --index ... --out logs/phase4/train/smoke --limit-items 2000 --items-per-step 10

What is trained: LoRA r 32 / alpha 64 / dropout 0 on q, k, v, o, gate, up, down of all 28 layers of
JetLM/SDAR-1.7B-Chat-b32 @ research_protocol.MODELS["d32"] (34,865,152 parameters, asserted); embeddings, LM head
and every base weight frozen.  LoRA weights fp32, base bf16, compute under bf16 autocast.  The 30B is not loaded.

Data (drafter_lora/data.py): waves A + B of the 1B corpus; 3,000 items held out (stratified by domain, numpy
default_rng(0), saved to <out>/holdout_pids.json); EXACTLY ONE PASS over the rest, ceil(n / --items-per-step)
optimiser steps (the last one smaller).  Every usable block of an item is a training state (40% first-draft,
60% re-draft), all of an item's blocks in one two-stream forward.

Loss (drafter_lora/forward.py): CE toward the 30B's token + 0.1 x KL(q_0 || q_theta) on the hidden slots the
original drafter already gets right; one denominator |H| per micro-batch; micro-batch losses averaged over the
step's micro-batches on all ranks.  AdamW (0.9, 0.95), eps 1e-8, weight decay 0, peak lr 8e-5, linear warmup over
2% of the steps, cosine to 10%, clip 1.0.  Gradients all-reduced by hand (the forward bypasses model.forward).

Stops (each writes <out>/STOP.json, saves a checkpoint and exits with code 3; nothing is cut silently):
  * --max-hours: the measured speed at step --project-at projects the pass beyond it (default 6 h);
  * --check-at (0.25): held-out first-draft agreement (s8 replay, drafter_lora/replay.py) is not at least
    --check-min-delta (+1.0 point) above the same items' agreement at step 0 (the identical model: lora_B = 0).
Saves: the adapter every 10% of the pass and at the end (<out>/ckpt/step_NNNNNN, `latest` marker, optimizer state
for --resume), and at the end a MERGED full HF directory <out>/merged that run_da_decode.py --drafter loads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.distributed as dist                                         # noqa: E402

from draftalign import wandb_log                                         # noqa: E402
from draftalign.train_drafter import all_reduce_grads, dist_init, lr_scale   # noqa: E402
from drafter_lora import data as D                                       # noqa: E402
from drafter_lora import lora as LA                                      # noqa: E402
from drafter_lora.forward import drafter_loss, summarize                 # noqa: E402
from drafter_lora.replay import eval_mode, replay_item, restore_mode     # noqa: E402

STOP_CODE = 3


# ---------------------------------------------------------------------------
# held-out agreement (the 25% check), sharded over ranks
# ---------------------------------------------------------------------------
@torch.no_grad()
def held_out_agreement(model, entries_by_pid: Dict[str, dict], pids: List[str], device, rank: int, world: int,
                       adapter: bool = True) -> Dict[str, float]:
    was = eval_mode(model)
    agree = slots = 0
    t0 = time.time()
    try:
        for p in pids[rank::world]:
            it = D.load_item(entries_by_pid[p])
            if adapter:
                r = replay_item(model, it, device)
            else:
                with LA.adapter_off((1, 1), device):
                    r = replay_item(model, it, device)
            agree += int(r["right"].sum()); slots += int(r["valid"].sum())
    finally:
        restore_mode(model, was)
    t = torch.tensor([agree, slots], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(t)
    return {"agree": float(t[0]), "slots": float(t[1]), "agreement": float(t[0] / t[1]) if t[1] else float("nan"),
            "items": float(len(pids)), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def ckpt_dir(out: str, step: int) -> str:
    return os.path.join(out, "ckpt", f"step_{step:06d}")


def save_ckpt(model, opt, out: str, step: int, state: dict) -> str:
    d = ckpt_dir(out, step)
    LA.save(model, d, extra={"step": int(step), "total_steps": int(state.get("total_steps", 0))})
    torch.save({"optimizer": opt.state_dict(), **state}, os.path.join(d, "trainer_state.pt"))
    with open(os.path.join(d, "trainer_state.json"), "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    with open(os.path.join(out, "ckpt", "latest"), "w") as fh:
        fh.write(os.path.basename(d))
    print(f"[save] {d}", flush=True)
    return d


def latest_ckpt(out: str) -> Optional[str]:
    fp = os.path.join(out, "ckpt", "latest")
    if not os.path.isfile(fp):
        return None
    d = os.path.join(out, "ckpt", open(fp).read().strip())
    return d if os.path.isdir(d) else None


def save_points(total: int, frac: float) -> List[int]:
    """Steps (1-based counts) at every `frac` of the pass, the last step included."""
    k = max(1, int(round(1.0 / frac)))
    return sorted({max(1, int(math.ceil(total * i / k))) for i in range(1, k + 1)} | {total})


def write_stop(out: str, reason: str, payload: dict) -> None:
    with open(os.path.join(out, "STOP.json"), "w") as fh:
        json.dump({"reason": reason, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload},
                  fh, indent=2, sort_keys=True)
    print(f"[STOP] {reason}: {json.dumps(payload, sort_keys=True)}", flush=True)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True, help="a to_trainset index.jsonl (repeat for waves)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="d32", help="research_protocol.MODELS key of the drafter")
    # LoRA
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--targets", default=",".join(LA.TARGETS))
    ap.add_argument("--expect-trainable", type=int, default=None,
                    help=f"assert this trainable count (default {LA.EXPECTED_TRAINABLE_D32:,} for d32 at the "
                         f"runbook's r / targets; 0 = no check)")
    # loss / states
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--p-first", type=float, default=0.4, help="share of first-draft states (the rest re-draft)")
    ap.add_argument("--post-eos", choices=("truth", "mask"), default="truth",
                    help="post-EOS slots of the last block in a state: the truth (StateSampler, default) or [MASK]")
    ap.add_argument("--max-seq", type=int, default=None, help="drop blocks ending past this (default: the index's)")
    # optimisation
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--betas", default="0.9,0.95")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--items-per-step", type=int, default=256, help="items per optimiser step over all ranks")
    ap.add_argument("--micro-tokens", type=int, default=16384,
                    help="padded two-stream positions per micro-batch (B x Lmax); bounds activation memory")
    ap.add_argument("--no-grad-checkpointing", dest="grad_ckpt", action="store_false", default=True)
    ap.add_argument("--no-autocast", dest="autocast", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    # data scope
    ap.add_argument("--holdout", type=int, default=3000)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--limit-items", type=int, default=0, help="smoke test: train on the first N items of the order")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps (0 = the whole pass)")
    # checks / saves / logs
    ap.add_argument("--save-every-frac", type=float, default=0.1)
    ap.add_argument("--check-at", type=float, default=0.25, help="fraction of the pass for the agreement check (0 = off)")
    ap.add_argument("--check-min-delta", type=float, default=0.01, help="agreement gain required at --check-at")
    ap.add_argument("--check-items", type=int, default=0, help="held-out items replayed by the check (0 = all)")
    ap.add_argument("--project-at", type=int, default=20, help="step at which the pass time is projected")
    ap.add_argument("--max-hours", type=float, default=6.0, help="stop if the projected pass exceeds this (0 = off)")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--no-merge", dest="merge", action="store_false", default=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt/latest")
    ap.add_argument("--wandb-project", default=None, help="default: $DA_WANDB_PROJECT, else $WANDB_PROJECT")
    ap.add_argument("--run-name", default=None, help="W&B run name (default: phase4_<basename of --out>)")
    ap.add_argument("--load-workers", type=int, default=8, help="threads that read item npz files ahead")
    ap.add_argument("--dry-run", action="store_true", help="data, split and plan only; no model")
    ap.add_argument("--mask-id", type=int, default=P.MASK_ID, help=argparse.SUPPRESS)      # tests: a tiny vocabulary
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    info = dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = P.safe_out(args.out)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)

    # -- data, split, plan (deterministic, identical on every rank) ----------
    entries = D.load_entries(args.index)
    by_pid = {str(e["pid"]): e for e in entries}
    hold, per_dom = D.stratified_holdout(entries, args.holdout, args.holdout_seed)
    hold_set = set(hold)
    hp = os.path.join(out, "holdout_pids.json")
    if rank == 0:
        D.save_holdout(hp, hold, per_dom, {"n": len(hold), "seed": args.holdout_seed, "indexes": args.index,
                                           "rule": "stratified by domain, largest remainder, numpy default_rng"})
    lengths = {str(e["pid"]): D.two_stream_length(e, args.max_seq) for e in entries}
    train_pids = [p for p in by_pid if p not in hold_set and lengths[p] > 0]
    order = D.pass_order(train_pids, args.seed)
    if args.limit_items:
        order = order[:int(args.limit_items)]
    plan = D.plan_pass(order, lengths, args.items_per_step, world, args.micro_tokens)
    total = len(plan) if not args.max_steps else min(len(plan), int(args.max_steps))
    warmup = max(1, int(round(args.warmup_frac * total)))
    corpus_tok = {str(e["pid"]): int(e.get("corpus_tokens") or 0) for e in entries}
    check_pids = list(hold)
    if args.check_items and args.check_items < len(check_pids):
        rng = np.random.default_rng(args.holdout_seed + 1)
        check_pids = sorted(check_pids[int(i)] for i in rng.choice(len(check_pids), args.check_items, replace=False))
    check_step = int(math.ceil(args.check_at * total)) if args.check_at else 0
    saves = save_points(total, args.save_every_frac)
    cfg = dict(vars(args), world_size=world, model_revision=P.MODELS[args.model]["revision"],
               trainer="drafter_lora/train_drafter_lora v1", n_items=len(entries), n_holdout=len(hold),
               holdout_per_domain=per_dom, n_train_items=len(order), total_steps=total, warmup_steps=warmup,
               save_steps=saves, check_step=check_step,
               train_corpus_tokens=int(sum(corpus_tok[p] for p in order)))
    if rank == 0:
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True)
        print(f"[data] {len(entries)} items ({per_dom} held out = {len(hold)}); one pass over {len(order)} items = "
              f"{cfg['train_corpus_tokens'] / 1e6:.1f}M corpus tokens in {total} steps of <= {args.items_per_step} "
              f"items; warmup {warmup}; saves at {saves}; check at step {check_step}", flush=True)
    if args.dry_run:
        n_micro = [sum(len(m) for m in s.per_rank[rank]) for s in plan[:total]]
        print(f"[plan] rank {rank}: {sum(n_micro)} micro-batches over {total} steps", flush=True)
        return 0

    # -- model + adapter -------------------------------------------------------
    if not os.environ.get("DA_WANDB_PROJECT"):
        proj = args.wandb_project or os.environ.get("WANDB_PROJECT")
        if proj:
            os.environ["DA_WANDB_PROJECT"] = proj
    torch.manual_seed(args.seed)                                    # identical A init on every rank
    # GPU: bf16 base under autocast (the runbook's layout).  CPU (tests / dry checks only): fp32, bf16 matmuls are slow
    _tok, model = P.load_sdar(args.model, str(device), dtype=None if device.type == "cuda" else torch.float32)
    targets = tuple(t for t in args.targets.split(",") if t)
    LA.attach(model, r=args.lora_r, alpha=args.alpha, dropout=args.dropout, targets=targets)
    model.train()
    cnt = LA.count(model)
    expect = args.expect_trainable
    if expect is None and args.model == "d32" and args.lora_r == 32 and set(targets) == set(LA.TARGETS):
        expect = LA.EXPECTED_TRAINABLE_D32
    if rank == 0:
        print(f"[model] {args.model} rev {P.MODELS[args.model]['revision'][:8]}: trainable {cnt['total']:,} "
              f"(attention {cnt['attention']:,} + MLP {cnt['mlp']:,})", flush=True)
    if rank == 0:
        with open(os.path.join(out, "model_info.json"), "w") as fh:
            json.dump({"trainable": cnt, "expected": expect, "lora": getattr(model, "_drafter_lora", {}),
                       "model": args.model, "revision": P.MODELS[args.model]["revision"]}, fh, indent=2)
    if expect:
        assert cnt["total"] == int(expect), f"trainable {cnt['total']:,} != expected {int(expect):,}"
    params = [p for _, p in LA.trainable(model)]
    b1, b2 = (float(x) for x in args.betas.split(","))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(b1, b2), eps=args.eps, weight_decay=args.weight_decay)

    state = {"step": 0, "items": 0, "corpus_tokens": 0, "total_steps": total, "agree_step0": None}
    if args.resume:
        d = latest_ckpt(out)
        if d:
            LA.load(model, d)
            st = torch.load(os.path.join(d, "trainer_state.pt"), map_location=device, weights_only=False)
            opt.load_state_dict(st["optimizer"])
            state.update({k: v for k, v in st.items() if k != "optimizer"})
            assert int(state["total_steps"]) == total, "resume with a different plan"
            if rank == 0:
                print(f"[resume] {d}: step {state['step']}", flush=True)

    wb = wandb_log.init(args.run_name or f"phase4_{os.path.basename(out)}", cfg, out, group="phase4-drafter-lora",
                        job_type="drafter-lora", tags=("drafter", "lora", args.model), rank=rank)
    logf = open(os.path.join(out, f"train.rank{rank}.jsonl"), "a")

    def log_row(row: dict, prefix: str):
        if rank == 0:
            logf.write(json.dumps(row) + "\n"); logf.flush()
            wb.log(wandb_log.prefixed(row, prefix), step=int(row.get("step", state["step"])))

    def stop(reason: str, payload: dict) -> int:
        if rank == 0:
            save_ckpt(model, opt, out, state["step"], dict(state))
            write_stop(out, reason, payload)
            log_row({"step": state["step"], "event": "stop", "reason": reason, **payload}, "event")
        if world > 1:
            dist.barrier()
        wb.finish()
        return STOP_CODE

    # -- the untrained drafter's held-out agreement (same model, lora_B = 0) ---
    if check_step and state["agree_step0"] is None:
        ag = held_out_agreement(model, by_pid, check_pids, device, rank, world, adapter=False)
        state["agree_step0"] = ag["agreement"]
        if rank == 0:
            print(f"[agree step0] {ag['agreement']:.4f} over {ag['slots']:.0f} slots / {len(check_pids)} held-out "
                  f"items ({ag['seconds']:.0f}s)", flush=True)
            log_row({"step": 0, "event": "agree", **ag}, "check")

    pool = ThreadPoolExecutor(max_workers=max(1, args.load_workers))

    def fetch(step: int):
        pids = [p for m in plan[step].per_rank[rank] for p in m] if step < total else []
        return {p: pool.submit(D.load_item, by_pid[p]) for p in pids}

    nxt = fetch(state["step"])
    t_start, steps_timed = time.time(), 0
    t_log, agg = time.time(), {}
    rc = 0
    for step in range(int(state["step"]), total):
        cur, nxt = nxt, fetch(step + 1)
        sp = plan[step]
        lr = args.lr * lr_scale(step, warmup, total, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot: Dict[str, float] = {}
        scale = world / max(1, sp.n_micro_global)          # mean over the step's micro-batches on ALL ranks
        t_step = time.time()
        for micro in sp.per_rank[rank]:
            exs = []
            for p in micro:
                it = cur[p].result()
                ex = D.item_example(it, seed=args.seed, p_first=args.p_first, max_seq=args.max_seq,
                                    mask_id=args.mask_id, post_eos=args.post_eos)
                if ex is not None:
                    exs.append(ex)
            if not exs:
                continue
            batch = D.collate(exs, mask_id=args.mask_id)
            loss, st = drafter_loss(model, batch, kl_weight=args.kl_weight, mask_id=args.mask_id,
                                    checkpoint_layers=args.grad_ckpt, autocast=args.autocast)
            (loss * scale).backward()
            st["corpus_tokens"] = float(sum(corpus_tok[p] for p in micro))
            for k, v in st.items():
                tot[k] = tot.get(k, 0.0) + float(v)
        all_reduce_grads(params, world)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        if not math.isfinite(gnorm):
            rc = stop("non-finite gradient norm", {"step": step + 1, "grad_norm": gnorm})
            break
        opt.step()
        state["step"] = step + 1
        keys = sorted(set(tot) | {"n_rows", "ce_sum", "kl_sum", "n_kl", "n_right", "n_right0", "n_kept", "n_items",
                                  "n_pos", "corpus_tokens", "n_first", "n_right_first", "n_right0_first",
                                  "n_redraft", "n_right_redraft", "n_right0_redraft"})
        tvec = torch.tensor([tot.get(k, 0.0) for k in keys], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(tvec)
        g_tot = dict(zip(keys, tvec.tolist()))
        state["items"] += int(g_tot["n_items"]); state["corpus_tokens"] += int(g_tot["corpus_tokens"])
        if step > 0:                                        # step 0 carries allocator / autotune warm-up
            steps_timed += 1
        if step == 0:
            t_start = time.time()
        for k, v in g_tot.items():
            agg[k] = agg.get(k, 0.0) + v
        agg["grad_norm"] = agg.get("grad_norm", 0.0) + gnorm
        agg["_n"] = agg.get("_n", 0) + 1
        if step == 0 and rank == 0:
            print(f"[step 1] KL sum {g_tot['kl_sum']:.3e} (must be exactly 0: lora_B starts at zero)", flush=True)
            state["kl_sum_step1"] = g_tot["kl_sum"]
        if args.log_every and (state["step"] % args.log_every == 0 or state["step"] == total):
            dt = time.time() - t_log
            s = summarize({**agg, "kl_weight": args.kl_weight})
            row = {"step": state["step"], "lr": lr, "grad_norm": agg["grad_norm"] / agg["_n"],
                   "items_per_s": agg["n_items"] / max(dt, 1e-9), "positions_per_s": agg["n_pos"] / max(dt, 1e-9),
                   "corpus_tokens_per_s": agg["corpus_tokens"] / max(dt, 1e-9), "sec_per_step": dt / agg["_n"],
                   "items_seen": state["items"], "corpus_tokens_seen": state["corpus_tokens"],
                   "step_seconds_this_rank": round(time.time() - t_step, 2),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **s}
            if rank == 0:
                print(f"[step {state['step']}/{total}] loss {s['loss']:.4f} ce {s['ce']:.4f} kl {s['kl']:.2e} "
                      f"kl_share {s['kl_share']:.3f} acc {s['acc']:.3f} (orig {s['acc0']:.3f}) "
                      f"ret {s['retention']:.4f} gnorm {row['grad_norm']:.2f} lr {lr:.2e} "
                      f"{row['items_per_s']:.1f} it/s {row['corpus_tokens_per_s']:.0f} tok/s", flush=True)
            log_row(row, "train")
            agg, t_log = {}, time.time()

        # speed projection (measured on this box; the pass must fit in --max-hours)
        if state["step"] == args.project_at and steps_timed > 0:
            sps_t = torch.tensor([(time.time() - t_start) / steps_timed], dtype=torch.float64, device=device)
            if world > 1:
                dist.broadcast(sps_t, src=0)               # one decision for all ranks
            sps = float(sps_t)
            proj_h = sps * len(plan) / 3600.0                # the WHOLE pass, also when --max-steps caps this run
            sp_row = {"step": state["step"], "event": "projection", "sec_per_step": sps, "pass_steps": len(plan),
                      "projected_hours": proj_h, "measured_steps": steps_timed, "world_size": world,
                      "items_per_step": args.items_per_step, "micro_tokens": args.micro_tokens}
            if rank == 0:
                with open(os.path.join(out, "speed.json"), "w") as fh:
                    json.dump(sp_row, fh, indent=2)
                print(f"[projection] {sps:.2f} s/step x {len(plan)} steps = {proj_h:.2f} h (measured over "
                      f"{steps_timed} steps)", flush=True)
                log_row(sp_row, "event")
            if args.max_hours and proj_h > args.max_hours and not args.max_steps:
                rc = stop("projected pass over --max-hours", sp_row)
                break

        if state["step"] in saves and rank == 0:
            save_ckpt(model, opt, out, state["step"], dict(state))
        if state["step"] in saves and world > 1:
            dist.barrier()

        if check_step and state["step"] == check_step:
            ag = held_out_agreement(model, by_pid, check_pids, device, rank, world, adapter=True)
            delta = ag["agreement"] - float(state["agree_step0"])
            row = {"step": state["step"], "event": "agree", "agree_step0": state["agree_step0"], "delta": delta, **ag}
            if rank == 0:
                print(f"[check {args.check_at:.0%}] agreement {ag['agreement']:.4f} vs step 0 "
                      f"{state['agree_step0']:.4f}: {100 * delta:+.2f} points (need >= "
                      f"{100 * args.check_min_delta:+.2f})", flush=True)
                log_row(row, "check")
            if delta < args.check_min_delta:
                rc = stop(f"agreement gain below {args.check_min_delta} at {args.check_at:.0%} of the pass", row)
                break
            if rank == 0:
                with open(os.path.join(out, "check_25pct.json"), "w") as fh:
                    json.dump(row, fh, indent=2)
    pool.shutdown(wait=False, cancel_futures=True)

    if rc == 0:
        if rank == 0 and state["step"] not in saves:
            save_ckpt(model, opt, out, state["step"], dict(state))
        if rank == 0:
            LA.save(model, os.path.join(out, "final_adapter"), extra={"step": state["step"], "total_steps": total})
            if args.merge:
                del opt
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                m = LA.save_merged(args.model, os.path.join(out, "final_adapter"), os.path.join(out, "merged"),
                                   device=str(device))
                print(f"[merged] {m}  (run_da_decode.py --drafter {m})", flush=True)
            with open(os.path.join(out, "DONE.json"), "w") as fh:
                json.dump({"steps": state["step"], "items": state["items"], "corpus_tokens": state["corpus_tokens"],
                           "seconds": round(time.time() - t_start, 1)}, fh, indent=2)
        if world > 1:
            dist.barrier()
        wb.finish()
    logf.close()
    if world > 1:
        dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
