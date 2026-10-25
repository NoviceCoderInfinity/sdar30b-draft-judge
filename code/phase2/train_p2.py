#!/usr/bin/env python
"""Phase 2, step 6: retrain the LoRA judge adapter on RE-DRAFTED targets (the drafter and the 30B MoE stay frozen).

Only ever trains the judge adapter (LoRA r32 / alpha 64 on q, k, v, o in all 48 layers + the draft flag, about
27M parameters) on the frozen pinned stock JetLM/SDAR-30B-A3B-Chat-b32. The targets are the existing ones (the
30B model's own generations); only the drafts in the index come from the new drafter (`redraft.py`).

The per-state loss and the state sampler are the round-3 ones through the two-stream forward of
`draftalign/train_fast.py` (equal to `train.forward_state`, pinned by code/corpus1b/tests/test_train_fast.py),
with the loop of `corpus1b/train_c1b.py`. Phase 2 adds, at EVERY optimizer step:

  * stop rules (code/phase2/rules.json, stage "train"): non-finite loss / gradient, gradient explosion, loss
    divergence, held-out commit precision under 0.99 at two probes in a row, held-out CE rising at three probes in a
    row, plus the STOP-file kill switch. A stop saves `stopped/`, writes STOPPED.json and exits 3.
  * one JSON row per step (`steps.jsonl`): loss, CE, certainty, calibration, accuracy, grad norm, lr, tokens,
    states, seconds, GPU memory -- all-reduced over ranks;
  * every --log-every steps: per-domain means (`train.rank0.jsonl`, W&B `train/*`) and the expert-activation window:
    per MoE layer the experts the prefix pass and the trained block pass routed to (`experts/step_<n>.npz`,
    W&B `experts/*`: unique experts per layer, load entropy, max-load share);
  * the extended probe (`p2probe.py`: keep / fix rate, commit precision, ECE, draft accuracy, per domain) every
    --probe-every-steps and at the end (`probe.jsonl`, W&B `probe/*`); the checkpoint with the lowest held-out CE
    whose commit precision is >= 0.99 is kept as `best/`.

Schedule: `--schedule cosine` (default) is round 3's (train.lr_scale: linear warmup over --warmup-steps, cosine to
--min-lr-frac over --steps) at a fixed --states-per-step; `--schedule wsd` is train_c1b's token schedule.

    torchrun --standalone --nproc_per_node=8 code/phase2/train_p2.py \
        --index logs/phase2/redraft/F1/trainset/index.jsonl --out logs/phase2/train/F1 \
        --drafter-key F1 --steps 1400 --states-per-step 0:32 --lr 2e-5 --min-lr-frac 0.5 --warmup-steps 50
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.distributed as dist                                         # noqa: E402

import train_c1b as C                                                    # noqa: E402  (corpus1b trainer)
from draftalign import adapter as A                                      # noqa: E402
from draftalign import train as T                                        # noqa: E402
from draftalign import train_fast as F                                   # noqa: E402
from draftalign import wandb_log                                         # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry   # noqa: E402
from phase2 import p2probe                                               # noqa: E402
from phase2.explog import TrainExpertCapture                             # noqa: E402
from phase2.stoprules import STOP_EXIT, RunningMedian, StopMonitor, rules_sha   # noqa: E402

STAT_KEYS = ("loss", "ce", "cert", "calib", "acc", "top1")


def build_parser():
    ap = C.build_parser()
    ap.description = __doc__
    ap.add_argument("--drafter-key", required=True, help="phase2.drafters key whose drafts the index holds (label)")
    ap.add_argument("--init-adapter", default=None,
                    help="warm-start the LoRA judge adapter from this directory (default: fresh, as round 3)")
    ap.add_argument("--schedule", choices=("cosine", "wsd"), default="cosine")
    ap.add_argument("--steps", type=int, default=1400, help="cosine: optimizer steps (round 3: 700 per epoch)")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--probe-every-steps", type=int, default=100)
    ap.add_argument("--probe-tau", type=float, default=0.95)
    ap.add_argument("--save-every-steps", type=int, default=100)
    ap.add_argument("--rules", default=None, help="rules.json (default code/phase2/rules.json)")
    ap.add_argument("--no-expert-log", dest="expert_log", action="store_false", default=True)
    ap.add_argument("--holdout-mode", choices=("r3", "domain"), default="r3")
    ap.add_argument("--holdout-items", type=int, default=150, help="--holdout-mode r3: round 3's value")
    return ap


def gpu_mem_frac(device) -> float:
    if device.type != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return float(1.0 - free / total)


def main(argv=None, model_builder=None) -> int:
    args = build_parser().parse_args(argv)
    info = T.dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    rules_kw = {"rules_path": args.rules} if args.rules else {}
    cfg_dump = dict(vars(args), world_size=world, model_revision=P.MODELS[args.model]["revision"],
                    trainer="phase2/train_p2 (two-stream, round-3 loss)",
                    rules_sha256=rules_sha(args.rules) if args.rules else rules_sha())
    if rank == 0:
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump(cfg_dump, fh, indent=2, sort_keys=True)

    scfg = SamplerConfig(index_path=args.index, seed=args.seed, p_pass1=args.p_pass1, p_allmask=args.p_allmask,
                         p_draftfull=args.p_draftfull, mask_id=args.mask_id, block=args.block, max_seq=args.max_seq)
    full = StateSampler(scfg)
    if args.holdout_mode == "r3":
        # round 3's item-level holdout (train.py --holdout-items 150 --seed 1): the items the S0 judge (r3_B) never
        # trained on, so every Phase 2 judge is probed on the same held-out items as the S0 judge
        train_s, eval_s = full.split_holdout(args.holdout_items)
    else:
        train_s, eval_s = full.split_holdout_by_domain(args.holdout_per_domain)
    shares = [train_s.tokens_share(i) for i, _b in train_s.examples]
    mean_share = float(np.mean(shares)) if shares else 1.0
    doms = sorted({domain_of_entry(e) for e in full.entries})
    stat_keys = [f"{d}/{k}" for d in doms + ["all"] for k in STAT_KEYS] + ["grad_norm"]
    if rank == 0:
        print(f"[data] {full.summary()} -> train {len(train_s)} examples, holdout {len(eval_s.entries)} items; "
              f"domains {doms}", flush=True)
    ramp = C.parse_ramp(args.states_per_step)
    if args.dry_run:
        plan = C.plan_epoch(train_s, 0, rank, world, ramp, 0.0, mean_share, args.max_states_micro)
        print(f"[plan] rank {rank}: {len(plan)} steps per epoch", flush=True)
        return 0

    # -- the pinned stock model + the adapter --------------------------------------------------------------------
    torch.manual_seed(args.seed)
    if model_builder is None:
        _tok, model = P.load_sdar(args.model, str(device))
    else:
        model = model_builder().to(device)
    A.attach_lora(model, r=args.lora_r, alpha=args.alpha, train_gates=False, draft_flag=True,
                  conf_embed=args.conf_embed)
    A.freeze_base(model)
    if args.init_adapter:
        got = A.load_adapter(model, P.safe_out(args.init_adapter))
        if rank == 0:
            print(f"[init] adapter {args.init_adapter}: {got['loaded']} tensors", flush=True)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    params = [p for _, p in A.trainable_parameters(model)]
    opt = torch.optim.AdamW(A.parameter_groups(model, args.lr, args.lr, args.weight_decay), betas=(0.9, 0.95), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lcfg = F.LossCfg(beta=args.beta, chat_beta=args.chat_beta, temp=args.temp, calib=args.calib,
                     override_weight=args.override_weight, code_override_weight=args.code_override_weight,
                     chat_override_weight=args.chat_override_weight, mask_id=args.mask_id)

    state = {"step": 0, "tokens": 0.0, "epoch": 0, "step_in_epoch": 0, "milestones_done": [],
             "best_ce": None, "probe_ce_hist": []}
    if args.resume:
        best = None
        for d in sorted(os.listdir(out)):
            f = os.path.join(out, d, "trainer_state.json")
            if d.startswith(("roll_", "step_")) and os.path.exists(f):
                t = json.load(open(f))
                if best is None or t["step"] > best[1]["step"]:
                    best = (d, t)
        if best:
            state.update(C.load_ckpt(model, opt, os.path.join(out, best[0]), device))
            if rank == 0:
                print(f"[resume] {best[0]}: step {state['step']}", flush=True)

    cap = TrainExpertCapture(model, device) if args.expert_log else None
    if cap is not None:
        cap.install()
        os.makedirs(os.path.join(out, "experts"), exist_ok=True)
    mon = StopMonitor("train", out, rank=rank, context={"drafter": args.drafter_key, "index": args.index}, **rules_kw)
    wb = wandb_log.init(os.path.basename(out), cfg_dump, out, group=f"phase2-train", job_type="judge-lora",
                        tags=("phase2", "judge", args.drafter_key), rank=rank)
    logf = open(os.path.join(out, f"train.rank{rank}.jsonl"), "a")
    stepf = open(os.path.join(out, "steps.jsonl"), "a") if rank == 0 else None
    gmed, smed = RunningMedian(200), RunningMedian(200)
    loss_ema, loss_ema_min, ema_a = None, None, 0.05
    agg: Dict[str, float] = {}
    aggn: Dict[str, float] = {}
    t_last, n_states_log, n_tok_log = time.time(), 0, 0.0
    total_steps = int(args.steps)

    def lr_scale() -> float:
        if args.schedule == "cosine":
            return T.lr_scale(state["step"], args.warmup_steps, total_steps, args.min_lr_frac)
        return C.wsd_lr_scale(state["tokens"], args.warmup_tokens, args.total_tokens, args.decay_frac, args.min_lr_frac)

    def save(name: str):
        if rank == 0:
            C.save_ckpt(model, opt, os.path.join(out, name), dict(state))
            print(f"[save] {name} at step {state['step']}", flush=True)

    def agree_stop(local_stop: bool) -> bool:
        """Every rank must leave the loop at the same step."""
        t = torch.tensor([1.0 if local_stop else 0.0], device=device)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def do_probe(tag: str) -> bool:
        pr = p2probe.probe(model, eval_s, lcfg, args.probe_states_per_domain, args.block, args.seed, rank, world,
                           device, args.conf_embed, tau=args.probe_tau)
        ce = pr.get("all/ce")
        hist = state["probe_ce_hist"]
        rises = 0
        if ce is not None and math.isfinite(ce):
            hist.append(ce)
            for a_, b_ in zip(hist[::-1][:-1], hist[::-1][1:]):
                if a_ > b_:
                    rises += 1
                else:
                    break
        prec = pr.get("all/conf_precision", float("nan"))
        improved = (ce is not None and math.isfinite(ce) and prec == prec and prec >= 0.99
                    and (state["best_ce"] is None or ce < state["best_ce"]))
        if improved:
            state["best_ce"] = ce
            save("best")
        row = {"event": "probe", "tag": tag, "step": state["step"], "tokens": state["tokens"], "best": improved,
               "ce_rises": rises, **pr}
        if rank == 0:
            with open(os.path.join(out, "probe.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[probe {tag} @ step {state['step']}] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(pr.items()) if k.startswith("all/") and isinstance(v, float)),
                flush=True)
            wb.log(wandb_log.prefixed({k: v for k, v in pr.items() if isinstance(v, float)}, "probe"), step=state["step"])
        d = mon.check({**{f"probe/{k}": v for k, v in pr.items() if isinstance(v, float)}, "probe_ce_rises": rises},
                      counts={"tokens": state["tokens"], "steps": state["step"]})
        return agree_stop(d.stop)

    if state["step"] == 0 and do_probe("start"):
        return finish(model, opt, out, state, rank, world, wb, logf, stepf, cap, stopped=True)

    stop, stopped = False, False
    for epoch in range(state["epoch"], max(args.epochs, 1000)):
        plan = C.plan_epoch(train_s, epoch, rank, world, ramp, state["tokens"] if epoch == state["epoch"] else 0.0,
                            mean_share, args.max_states_micro)
        n = torch.tensor([len(plan)], device=device)
        if world > 1:
            dist.all_reduce(n, op=dist.ReduceOp.MIN)
        plan = plan[:int(n.item())]
        start = state["step_in_epoch"] if epoch == state["epoch"] else 0
        for si in range(start, len(plan)):
            t_step = time.time()
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_scale()
            opt.zero_grad(set_to_none=True)
            n_states = sum(len(m) for m in plan[si])
            tok_local, loss_local, nf_local = 0.0, 0.0, 0.0
            dom_stats: Dict[str, List[float]] = {}
            for micro in plan[si]:
                sts = [train_s.state(k, epoch=epoch) for k in micro]
                if cap is not None:
                    with cap.armed_scope():
                        loss, stats, _ = F.micro_batch_loss(model, sts, lcfg, block=args.block,
                                                            checkpoint_layers=args.checkpoint_layers,
                                                            conf_embed=args.conf_embed)
                else:
                    loss, stats, _ = F.micro_batch_loss(model, sts, lcfg, block=args.block,
                                                        checkpoint_layers=args.checkpoint_layers,
                                                        conf_embed=args.conf_embed)
                (loss / max(1, n_states)).backward()
                lv = float(loss.detach())
                loss_local += lv
                nf_local += 0.0 if math.isfinite(lv) else 1.0
                for st_, s in zip(sts, stats):
                    tok_local += float(st_.meta["tokens_share"])
                    for k_ in STAT_KEYS:
                        dom_stats.setdefault(f"{s['domain']}/{k_}", []).append(s[k_])
                        dom_stats.setdefault(f"all/{k_}", []).append(s[k_])
            T.all_reduce_grads(params, world)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            nf_local += 0.0 if math.isfinite(gnorm) else 1.0
            tt = torch.tensor([tok_local, float(n_states), loss_local, nf_local], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(tt)
            if float(tt[3]) == 0:                                  # every rank skips a non-finite step together
                opt.step()
            state["tokens"] += float(tt[0])
            state["step"] += 1
            state["epoch"], state["step_in_epoch"] = epoch, si + 1
            n_states_log += int(tt[1]); n_tok_log += float(tt[0])
            step_loss = float(tt[2]) / max(1.0, float(tt[1]))
            for k_, v in dom_stats.items():
                agg[k_] = agg.get(k_, 0.0) + float(np.mean(v)); aggn[k_] = aggn.get(k_, 0.0) + 1.0
            agg["grad_norm"] = agg.get("grad_norm", 0.0) + gnorm; aggn["grad_norm"] = aggn.get("grad_norm", 0.0) + 1.0

            # -- every step: the per-step row and the stop rules ------------------------------------------------
            dt_step = time.time() - t_step
            sps = float(tt[1]) / max(dt_step, 1e-9)
            gmed.add(gnorm); smed.add(sps)
            if math.isfinite(step_loss):
                loss_ema = step_loss if loss_ema is None else (1 - ema_a) * loss_ema + ema_a * step_loss
                if state["step"] > args.warmup_steps:
                    loss_ema_min = loss_ema if loss_ema_min is None else min(loss_ema_min, loss_ema)
            memf = gpu_mem_frac(device)
            metrics = {"nonfinite": float(tt[3]),
                       "grad_norm_over_median": gnorm / gmed.median() if gmed.median() else None,
                       "loss_ema_over_min": (loss_ema / loss_ema_min) if (loss_ema and loss_ema_min) else None,
                       "states_per_s_over_median": sps / smed.median() if smed.median() else None,
                       "gpu_mem_frac": memf}
            if stepf is not None:
                stepf.write(json.dumps({"step": state["step"], "tokens": state["tokens"], "loss": step_loss,
                                        "loss_ema": loss_ema, "grad_norm": gnorm, "lr": opt.param_groups[0]["lr"],
                                        "states": int(tt[1]), "sec": round(dt_step, 3), "states_per_s": round(sps, 2),
                                        "gpu_mem_frac": round(memf, 4), "skipped_update": bool(tt[3] > 0),
                                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                stepf.flush()
            d = mon.check(metrics, counts={"steps": state["step"], "tokens": state["tokens"]})
            if agree_stop(d.stop):
                stop = stopped = True
                break

            # -- every --log-every steps: per-domain means + the expert window ---------------------------------
            if state["step"] % args.log_every == 0:
                dt = time.time() - t_last
                st_t = torch.tensor([agg.get(k_, 0.0) for k_ in stat_keys] + [aggn.get(k_, 0.0) for k_ in stat_keys],
                                    dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(st_t)
                sums, cnts = st_t[:len(stat_keys)].tolist(), st_t[len(stat_keys):].tolist()
                row = {k_: sm / cn for k_, sm, cn in zip(stat_keys, sums, cnts) if cn > 0}
                row.update({"step": state["step"], "tokens": state["tokens"], "lr": opt.param_groups[0]["lr"],
                            "states_per_s": n_states_log / max(dt, 1e-9), "tokens_per_s": n_tok_log / max(dt, 1e-9),
                            "gpu_mem_frac": memf, "gpu_max_alloc_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                                        if device.type == "cuda" else 0.0),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if cap is not None:
                    win = cap.window(world)
                    row.update(TrainExpertCapture.metrics(win))
                    if rank == 0:
                        np.savez_compressed(os.path.join(out, "experts", f"step_{state['step']:07d}.npz"),
                                            prefix=win["prefix"].astype(np.int64), block=win["block"].astype(np.int64))
                if rank == 0:
                    print(f"[step {state['step']}/{total_steps}] loss {row.get('all/loss', 0):.4f} "
                          f"ce {row.get('all/ce', 0):.4f} cal {row.get('all/calib', 0):.4f} acc {row.get('all/acc', 0):.3f} "
                          f"gn {row.get('grad_norm', 0):.3f} lr {row['lr']:.2e} {row['states_per_s']:.1f} st/s "
                          f"mem {memf:.2f} exp/L(block) {row.get('experts/block_uniq_per_layer', float('nan')):.1f}",
                          flush=True)
                    wb.log(wandb_log.prefixed(row, "train"), step=state["step"])
                logf.write(json.dumps(row) + "\n"); logf.flush()
                agg, aggn, t_last, n_states_log, n_tok_log = {}, {}, time.time(), 0, 0.0

            if state["step"] % args.save_every_steps == 0:
                save(f"step_{state['step']:06d}")
            if state["step"] % args.probe_every_steps == 0 and do_probe("periodic"):
                stop = stopped = True
                break
            if args.schedule == "cosine" and state["step"] >= total_steps:
                stop = True
                break
            if args.max_tokens and state["tokens"] >= float(args.max_tokens):
                stop = True
                break
            if args.stop_at and agree_stop(time.time() >= args.stop_at):
                if rank == 0:
                    print(f"[stop-at] step {state['step']}", flush=True)
                stop = True
                break
        if stop:
            break
        state["step_in_epoch"] = 0
        state["epoch"] = epoch + 1
        if args.schedule == "wsd" and epoch + 1 >= args.epochs:
            break
    return finish(model, opt, out, state, rank, world, wb, logf, stepf, cap, stopped=stopped, probe_fn=do_probe)


def finish(model, opt, out, state, rank, world, wb, logf, stepf, cap, stopped: bool, probe_fn=None) -> int:
    name = "stopped" if stopped else "final"
    if rank == 0:
        C.save_ckpt(model, opt, os.path.join(out, name), dict(state))
        print(f"[{'STOPPED' if stopped else 'done'}] step {state['step']}, {state['tokens'] / 1e6:.2f}M tokens "
              f"-> {out}/{name}", flush=True)
    if probe_fn is not None and not stopped:
        probe_fn("final")
    if cap is not None:
        cap.uninstall()
    wb.finish()
    logf.close()
    if stepf is not None:
        stepf.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return STOP_EXIT if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
