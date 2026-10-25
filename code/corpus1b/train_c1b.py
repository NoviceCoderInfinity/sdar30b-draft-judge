#!/usr/bin/env python
"""Train the judge on the 1B corpus: the round-3 recipe on the two-stream forward (draftalign/train_fast.py).

    # wave A: fresh LoRA on the STOCK SDAR-30B-A3B-Chat-b32, one epoch of wave A = 128M corpus tokens
    torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
        --index logs/c1b/trainset/waveA/index.jsonl --out logs/c1b/train/main --max-tokens 128e6

    # wave B: continue the same run (adapter + optimizer + token count) on the wave-B index
    torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
        --index logs/c1b/trainset/waveB/index.jsonl --out logs/c1b/train/main_b \
        --continue-from logs/c1b/train/main/tok_000128000000 --max-tokens 1e9

What is identical to round 3 (r3_B, code/draftalign/restart_r3B_calib.sh): the model (pinned stock revision,
experts and routers frozen), LoRA r32 / alpha 64 on q, k, v, o in all 48 layers + the draft flag, the state
sampler (p_pass1 0.4, p_draftfull 0.3, p_allmask 0.0), the loss (CE with override weight 2 / code 3, certainty
beta 0.5 at temperature 0.5, calibration BCE 1.0), AdamW (0.9, 0.95), grad clip 1.0, manual all-reduce.
Per state, the loss is exactly train.forward_state's (code/corpus1b/tests/test_train_fast.py pins it).

What is new, all in the plan (docs/CORPUS_1B_PLAN_2026-09-28.md):
  * chat states: certainty weight --chat-beta (default 0) and override weight --chat-override-weight (default 1);
  * the step is built from ITEMS: all usable blocks of an item share one no-grad prefix pass; a step holds
    --states-per-step states over all ranks, ramped by tokens seen (default 32 -> 128 at 8M -> 512 at 32M);
  * warmup-stable-decay learning rate by corpus tokens (--total-tokens 1e9, decay over the last 15%), so the
    wave-A checkpoints are stable-phase checkpoints of the one 1B run;
  * checkpoints at the scaling milestones (--save-at-tokens, kept forever as tok_<12-digit tokens>/) plus rolling
    ones for resume; the held-out probe (200 items per domain) runs at every milestone and every --eval-every-tokens;
  * optional --conf-embed: the 32M drafter-confidence test (zero-initialised, so step 0 equals the main run).

A "token" is a corpus token (prompt + kept answer) -- the plan's unit; each state carries its item's corpus tokens
divided by the item's usable blocks, so one epoch counts exactly the index's corpus tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.distributed as dist                                         # noqa: E402

from draftalign import adapter as A                                      # noqa: E402
from draftalign import train as T                                        # noqa: E402
from draftalign import train_fast as F                                   # noqa: E402
from draftalign import wandb_log                                         # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry   # noqa: E402


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def parse_ramp(spec: str) -> List[Tuple[float, int]]:
    """"0:32,8e6:128,32e6:512" -> [(0, 32), (8e6, 128), (32e6, 512)] (tokens threshold, states per step)."""
    out = sorted((float(a), int(b)) for a, b in (tok.split(":") for tok in spec.split(",") if tok.strip()))
    assert out and out[0][0] == 0.0, f"the ramp must start at 0 tokens: {spec}"
    return out


def states_for_tokens(ramp: List[Tuple[float, int]], tokens: float) -> int:
    s = ramp[0][1]
    for t, n in ramp:
        if tokens >= t:
            s = n
    return s


def wsd_lr_scale(tokens: float, warmup: float, total: float, decay_frac: float, min_frac: float) -> float:
    """Warmup (linear) -> stable (1.0) -> linear decay to `min_frac` over the last `decay_frac` of `total`."""
    if tokens < warmup:
        return max(1e-3, tokens / max(1.0, warmup))
    d0 = total * (1.0 - decay_frac)
    if tokens <= d0:
        return 1.0
    t = min(1.0, (tokens - d0) / max(1.0, total - d0))
    return 1.0 - (1.0 - min_frac) * t


def parse_tokens_list(spec: str) -> List[float]:
    return sorted(float(x) for x in spec.split(",") if x.strip())


# ---------------------------------------------------------------------------
# the epoch plan: identical on every rank's view of the step structure
# ---------------------------------------------------------------------------
def plan_epoch(sampler: StateSampler, epoch: int, rank: int, world: int, ramp, tokens_start: float,
               mean_share: float, max_states_micro: int) -> List[List[List[int]]]:
    """This rank's steps for one epoch: step -> micro-batches -> example indices.

    The ramp is driven by an ESTIMATE of global tokens (step states x world x the index-wide mean tokens per
    state), identical on every rank, so every rank agrees on each step's size; the caller then truncates all
    ranks to the smallest step count (all_reduce MIN) so the collective all-reduces stay in lockstep.
    """
    by_item = sampler.item_examples()
    items = sampler.item_order(epoch)[rank::world]
    steps: List[List[List[int]]] = []
    cur_step: List[List[int]] = []
    cur_micro: List[int] = []
    n_in_step = 0
    est = tokens_start
    target = max(1, math.ceil(states_for_tokens(ramp, est) / world))

    def close_micro():
        nonlocal cur_micro
        if cur_micro:
            cur_step.append(cur_micro)
            cur_micro = []

    for i in items:
        ks = list(by_item[i])
        while ks:
            room = min(max_states_micro - len(cur_micro), target - n_in_step)
            take, ks = ks[:room], ks[room:]
            cur_micro.extend(take)
            n_in_step += len(take)
            if len(cur_micro) >= max_states_micro:
                close_micro()
            if n_in_step >= target:
                close_micro()
                steps.append(cur_step)
                est += n_in_step * world * mean_share
                cur_step, n_in_step = [], 0
                target = max(1, math.ceil(states_for_tokens(ramp, est) / world))
    # micro-batches pack several items (one no-grad prefix pass each, one shared grad pass); a trailing
    # partial step is dropped
    return steps


def plan_units(sampler: StateSampler, units: List[Tuple[int, int]], ramp, tokens_start: float, mean_share: float,
               world: int, max_states_micro: int) -> List[List[List[Tuple[int, int]]]]:
    """plan_epoch over an explicit, already rank-sliced list of (item, pass) units: step -> micro -> (example, pass).
    Same packing and ramp rule as plan_epoch; `pass` offsets the sampler's epoch so a repeated item draws fresh states."""
    by_item = sampler.item_examples()
    steps, cur_step, cur_micro, n_in_step = [], [], [], 0
    est = tokens_start
    target = max(1, math.ceil(states_for_tokens(ramp, est) / world))
    for i, p in units:
        ks = [(k, p) for k in by_item[i]]
        while ks:
            room = min(max_states_micro - len(cur_micro), target - n_in_step)
            take, ks = ks[:room], ks[room:]
            cur_micro.extend(take)
            n_in_step += len(take)
            if len(cur_micro) >= max_states_micro:
                cur_step.append(cur_micro); cur_micro = []
            if n_in_step >= target:
                if cur_micro:
                    cur_step.append(cur_micro); cur_micro = []
                steps.append(cur_step)
                est += n_in_step * world * mean_share
                cur_step, n_in_step = [], 0
                target = max(1, math.ceil(states_for_tokens(ramp, est) / world))
    return steps


def domain_weighted_units(sampler: StateSampler, epoch: int, world: int, ramp, epoch_tok0: float, mean_share: float,
                          max_states_micro: int, trained_steps: int, weights: Dict[str, float],
                          repeat_domains: Sequence[str], max_passes: int, cap_tokens: float = 400e6) -> List[Tuple[int, int]]:
    """The global (item, pass) sequence for the rest of the run: items not trained in the first `trained_steps` steps of
    the epoch's original plan keep their order within each domain; domains are interleaved by corpus tokens toward
    `weights` (stride scheduling: next unit from the domain furthest below its share); a domain in `repeat_domains` starts
    another pass over its items (original order) when it runs out, each item at most `max_passes` times in all (its pass
    in the original plan counts); other domains never repeat and drop out when exhausted."""
    plans = [plan_epoch(sampler, epoch, r, world, ramp, epoch_tok0, mean_share, max_states_micro) for r in range(world)]
    n = min(len(p) for p in plans)
    trained = {sampler.examples[k][0] for p in plans for step in p[:min(trained_steps, n)] for m in step for k in m}
    by_item = sampler.item_examples()
    order = sampler.item_order(epoch)
    dom = {i: domain_of_entry(sampler.entries[i]) for i in order}
    tok = {i: sampler.tokens_share(i) * len(by_item[i]) for i in order}      # the item's corpus tokens
    passes = {i: (1 if i in trained else 0) for i in order}
    w = {d: float(v) for d, v in weights.items() if v > 0}
    dom_order = {d: [i for i in order if dom[i] == d] for d in w}
    queues: Dict[str, List[int]] = {d: [i for i in dom_order[d] if i not in trained] for d in w}
    heads = {d: 0 for d in w}
    taken = {d: 0.0 for d in w}
    dead: set = set()                     # exhausted for good: never rescanned (a per-unit rescan made this O(N^2))
    units: List[Tuple[int, int]] = []
    total = 0.0
    while total < cap_tokens:
        live = []
        for d in w:
            if d in dead:
                continue
            if heads[d] >= len(queues[d]) and d in repeat_domains:     # a new pass over the domain's items
                queues[d] = [i for i in dom_order[d] if passes[i] < max_passes]
                heads[d] = 0
            if heads[d] < len(queues[d]):
                live.append(d)
            else:
                dead.add(d)
        if not live:
            break
        d = min(live, key=lambda x: taken[x] / w[x])
        i = queues[d][heads[d]]; heads[d] += 1
        units.append((i, passes[i]))
        passes[i] += 1
        taken[d] += tok[i]; total += tok[i]
    return units


def parse_weights(spec: str) -> Dict[str, float]:
    out = {}
    for tok_ in spec.split(","):
        if tok_.strip():
            k, v = tok_.split("=")
            out[k.strip()] = float(v)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_ckpt(model, opt, path: str, state: dict) -> None:
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    A.save_adapter(model, tmp)
    torch.save({"optimizer": opt.state_dict(), **state}, os.path.join(tmp, "trainer_state.pt"))
    with open(os.path.join(tmp, "trainer_state.json"), "w") as fh:
        json.dump(state, fh, indent=2)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


def load_ckpt(model, opt, path: str, device) -> dict:
    A.load_adapter(model, path)
    st = torch.load(os.path.join(path, "trainer_state.pt"), map_location=device, weights_only=False)
    if opt is not None and "optimizer" in st:
        opt.load_state_dict(st["optimizer"])
    return {k: v for k, v in st.items() if k != "optimizer"}


def prune_rolling(out: str, keep: int) -> None:
    rolls = sorted(d for d in os.listdir(out) if d.startswith("roll_") and os.path.isdir(os.path.join(out, d)))
    for d in rolls[:max(0, len(rolls) - keep)]:
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)


# ---------------------------------------------------------------------------
# the held-out probe, per domain
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe(model, sampler: StateSampler, cfg: F.LossCfg, n_per_domain: int, block: int, seed: int,
          rank: int, world: int, device, conf_embed: bool) -> Dict[str, float]:
    """Draft-filled pass-1 states (the state the judge meets first at inference), n per domain, sharded over
    ranks: argmax accuracy; share / precision of rows with top-1 > 0.95; the same on OVERRIDE rows (draft wrong)."""
    model.eval()
    by_dom: Dict[str, List[int]] = {}
    for k, (i, _b) in enumerate(sampler.examples):
        by_dom.setdefault(domain_of_entry(sampler.entries[i]), []).append(k)
    keys = ("rows", "correct", "conf", "conf_correct", "ovr", "ovr_correct", "conf_ovr", "conf_correct_ovr", "ce_sum")
    doms = sorted(by_dom)
    acc = torch.zeros(len(doms), len(keys), dtype=torch.float64, device=device)
    for di, d in enumerate(doms):
        g = torch.Generator().manual_seed(seed + 7 * di)
        ks = [by_dom[d][j] for j in torch.randperm(len(by_dom[d]), generator=g)[:n_per_domain].tolist()]
        mine = [sampler.state(k, regime="draft_pass1") for k in ks[rank::world]]
        mine = [s for s in mine if s.meta.get("has_draft")]
        for j in range(0, len(mine), 32):
            _, stats, _ = F.micro_batch_loss(model, mine[j:j + 32], cfg, block=block, checkpoint_layers=False,
                                             conf_embed=conf_embed, want_eval=True)
            for s in stats:
                n = s["ev_n_rows"]
                acc[di] += torch.tensor([n, s["ev_acc"] * n, s["ev_n_conf"], s["ev_conf_correct"], s["ev_n_ovr"],
                                         s["ev_ovr_correct"], s["ev_n_conf_ovr"], s["ev_conf_correct_ovr"],
                                         s["ce"] * n], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(acc)
    model.train()
    out: Dict[str, float] = {}
    tot = acc.sum(0)
    for name, row in [(d, acc[i]) for i, d in enumerate(doms)] + [("all", tot)]:
        r = dict(zip(keys, row.tolist()))
        nan = float("nan")
        out[f"{name}/acc"] = r["correct"] / r["rows"] if r["rows"] else nan
        out[f"{name}/ce"] = r["ce_sum"] / r["rows"] if r["rows"] else nan
        out[f"{name}/conf_frac"] = r["conf"] / r["rows"] if r["rows"] else nan
        out[f"{name}/conf_precision"] = r["conf_correct"] / r["conf"] if r["conf"] else nan
        out[f"{name}/ovr_conf_frac"] = r["conf_ovr"] / r["ovr"] if r["ovr"] else nan
        out[f"{name}/ovr_conf_precision"] = r["conf_correct_ovr"] / r["conf_ovr"] if r["conf_ovr"] else nan
        out[f"{name}/rows"] = r["rows"]
    return out


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="b32", help="research_protocol.MODELS key (the pinned stock 30B)")
    ap.add_argument("--resume", action="store_true", help="continue this --out from its newest checkpoint")
    ap.add_argument("--continue-from", default=None,
                    help="a checkpoint dir of an EARLIER run (adapter + optimizer + token count), e.g. wave A's "
                         "tok_000128000000, to continue the same run on a new --index (wave B)")
    # adapter (round 3)
    ap.add_argument("--rank", dest="lora_r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--conf-embed", action="store_true", help="the 32M drafter-confidence test")
    # loss (round 3 r3_B, plus the chat rule)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--chat-beta", type=float, default=0.0, help="0.5 = round-3 loss on chat too")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--calib", type=float, default=1.0)
    ap.add_argument("--override-weight", type=float, default=2.0)
    ap.add_argument("--code-override-weight", type=float, default=3.0)
    ap.add_argument("--chat-override-weight", type=float, default=1.0)
    # sampler (round 3)
    ap.add_argument("--p-pass1", type=float, default=0.4)
    ap.add_argument("--p-allmask", type=float, default=0.0)
    ap.add_argument("--p-draftfull", type=float, default=0.3)
    ap.add_argument("--max-seq", type=int, default=1856)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--mask-id", type=int, default=P.MASK_ID)
    ap.add_argument("--seed", type=int, default=1)
    # optimisation
    ap.add_argument("--lr", type=float, default=8e-5, help="peak LoRA learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--states-per-step", default="0:32,8e6:128,32e6:512", help="tokens:states ramp (global)")
    ap.add_argument("--max-states-micro", type=int, default=48, help="states per micro-batch per rank")
    ap.add_argument("--warmup-tokens", type=float, default=2e6)
    ap.add_argument("--total-tokens", type=float, default=1e9, help="the schedule's horizon (the whole 1B run)")
    ap.add_argument("--decay-frac", type=float, default=0.15)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=float, default=None, help="stop once this many tokens are seen")
    ap.add_argument("--stop-file", default=None, help="stop cleanly (save final + probe) once this file exists")
    ap.add_argument("--fast-moe", default="", choices=("", "auto", "grouped", "bmm"),
                    help="grouped-GEMM expert forward (corpus1b/fast_moe.py); auto = _grouped_mm if it works here")
    ap.add_argument("--domain-weights", default="",
                    help="e.g. maths=25,code=25,science=15,if=15,chat=20: token shares for the REST of the epoch "
                         "(untrained items keep their order; see domain_weighted_units)")
    ap.add_argument("--repeat-domains", default="science,if", help="domains whose items may repeat")
    ap.add_argument("--max-passes", type=int, default=4, help="most passes per item of a repeat domain")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="unix time: stop at the first step boundary after it (then save `final` and probe, as at "
                         "--max-tokens); all ranks decide together")
    ap.add_argument("--stop-at-file", default=None,
                    help="file holding a unix time; when present it replaces --stop-at (read every step)")
    ap.add_argument("--epochs", type=int, default=1)
    # checkpoints / probe / logs
    ap.add_argument("--save-at-tokens", default="1.84e6,8e6,32e6,128e6,512e6,1e9")
    ap.add_argument("--save-every-tokens", type=float, default=10e6, help="rolling resume checkpoints")
    ap.add_argument("--keep-rolling", type=int, default=2)
    ap.add_argument("--holdout-per-domain", type=int, default=200)
    ap.add_argument("--holdout-pids", default=None,
                    help="file with one pid per line: these items are the held-out probe set (never trained) instead of the per-domain random pick; keeps the probe identical when --index changes (e.g. a merged wave A + B index)")
    ap.add_argument("--probe-states-per-domain", type=int, default=256)
    ap.add_argument("--eval-every-tokens", type=float, default=20e6)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-checkpoint-layers", dest="checkpoint_layers", action="store_false", default=True)
    ap.add_argument("--dry-run", action="store_true", help="plan the data only")
    return ap


def main(argv=None, model_builder=None) -> dict:
    args = build_parser().parse_args(argv)
    info = T.dist_init()
    rank, world, local = info["rank"], info["world"], info["local_rank"]
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = P.safe_out(args.out)
    os.makedirs(out, exist_ok=True)
    cfg_dump = dict(vars(args), world_size=world, model_revision=P.MODELS[args.model]["revision"],
                    trainer="corpus1b/train_c1b (two-stream)")
    if rank == 0:
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump(cfg_dump, fh, indent=2, sort_keys=True)

    scfg = SamplerConfig(index_path=args.index, seed=args.seed, p_pass1=args.p_pass1, p_allmask=args.p_allmask,
                         p_draftfull=args.p_draftfull, mask_id=args.mask_id, block=args.block, max_seq=args.max_seq)
    full = StateSampler(scfg)
    if args.holdout_pids:                                   # pinned probe set (same items as an earlier index)
        hold = set(open(args.holdout_pids).read().split())
        train_s = StateSampler(scfg, [e for e in full.entries if e["pid"] not in hold], full.root)
        eval_s = StateSampler(scfg, [e for e in full.entries if e["pid"] in hold], full.root)
    else:
        train_s, eval_s = full.split_holdout_by_domain(args.holdout_per_domain)
    shares = [train_s.tokens_share(i) for i, _b in train_s.examples]
    mean_share = float(np.mean(shares)) if shares else 1.0
    if rank == 0:
        print(f"[data] {full.summary()} -> train {len(train_s)} examples / {len(train_s.item_examples())} items, "
              f"holdout {len(eval_s.entries)} items; epoch = {sum(shares) / 1e6:.2f}M corpus tokens", flush=True)
    ramp = parse_ramp(args.states_per_step)
    if args.dry_run:
        plan = plan_epoch(train_s, 0, rank, world, ramp, 0.0, mean_share, args.max_states_micro)
        print(f"[plan] rank {rank}: {len(plan)} steps, {sum(len(m) for s in plan for m in s)} states", flush=True)
        return {"steps": len(plan)}

    # -- the pinned stock model + a fresh round-3 adapter ------------------
    torch.manual_seed(args.seed)
    if model_builder is None:
        _tok, model = P.load_sdar(args.model, str(device))
        if args.fast_moe:
            import fast_moe
            moe_mode = fast_moe.patch_moe(model, mode=args.fast_moe)
            if rank == 0:
                print(f"[fast-moe] expert forward: {moe_mode}", flush=True)
    else:
        model = model_builder().to(device)
    A.attach_lora(model, r=args.lora_r, alpha=args.alpha, train_gates=False, draft_flag=True,
                  conf_embed=args.conf_embed)
    A.freeze_base(model)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    params = [p for _, p in A.trainable_parameters(model)]
    opt = torch.optim.AdamW(A.parameter_groups(model, args.lr, args.lr, args.weight_decay), betas=(0.9, 0.95), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lcfg = F.LossCfg(beta=args.beta, chat_beta=args.chat_beta, temp=args.temp, calib=args.calib,
                     override_weight=args.override_weight, code_override_weight=args.code_override_weight,
                     chat_override_weight=args.chat_override_weight, mask_id=args.mask_id)

    state = {"step": 0, "tokens": 0.0, "epoch": 0, "step_in_epoch": 0, "milestones_done": []}
    if args.continue_from:
        got = load_ckpt(model, opt, P.safe_out(args.continue_from), device)
        state["tokens"] = float(got.get("tokens", 0.0))
        state["step"] = int(got.get("step", 0))
        state["milestones_done"] = list(got.get("milestones_done", []))
        if rank == 0:
            print(f"[continue] from {args.continue_from} at {state['tokens'] / 1e6:.2f}M tokens", flush=True)
    if args.resume:
        rolls = sorted(d for d in os.listdir(out) if d.startswith(("roll_", "tok_")) or d == "final")   # newest wins
        best = None
        for d in rolls:
            f = os.path.join(out, d, "trainer_state.json")
            if os.path.exists(f):
                t = json.load(open(f))
                if best is None or t["tokens"] > best[1]["tokens"]:
                    best = (d, t)
        if best:
            state.update(load_ckpt(model, opt, os.path.join(out, best[0]), device))
            if rank == 0:
                print(f"[resume] {best[0]}: step {state['step']}, {state['tokens'] / 1e6:.2f}M tokens", flush=True)

    milestones = [m for m in parse_tokens_list(args.save_at_tokens)
                  if m not in state["milestones_done"] and m > state["tokens"]]
    wb = wandb_log.init(os.path.basename(out), cfg_dump, out, group="c1b", job_type="judge-lora",
                        tags=("judge", "c1b"), rank=rank)
    logf = open(os.path.join(out, f"train.rank{rank}.jsonl"), "a")
    # the logged stats are all-reduced over ONE fixed key list (every domain of the index + "all"), as (sum, count)
    # pairs: a rank that saw no state of some domain in a log window must still send the same tensor shape
    stat_doms = sorted({domain_of_entry(e) for e in full.entries})
    stat_keys = [f"{d}/{k}" for d in stat_doms + ["all"]
                 for k in ("loss", "ce", "cert", "calib", "acc", "top1")] + ["grad_norm"]
    aggn: Dict[str, float] = {}
    last_eval = state["tokens"]
    last_roll = state["tokens"]
    agg: Dict[str, float] = {}
    t_last, n_states_log, n_tok_log = time.time(), 0, 0.0

    def do_probe(tag: str):
        pr = probe(model, eval_s, lcfg, args.probe_states_per_domain, args.block, args.seed, rank, world, device,
                   args.conf_embed)
        pr.update({"event": "probe", "tag": tag, "tokens": state["tokens"], "step": state["step"]})
        if rank == 0:
            with open(os.path.join(out, "probe.jsonl"), "a") as fh:
                fh.write(json.dumps(pr) + "\n")
            print(f"[probe {tag} @ {state['tokens'] / 1e6:.2f}M] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(pr.items()) if isinstance(v, float) and k.endswith(
                    ("acc", "conf_precision", "conf_frac"))), flush=True)
            wb.log(wandb_log.prefixed({k: v for k, v in pr.items() if isinstance(v, float)}, "probe"),
                   step=state["step"])

    stop = False
    for epoch in range(state["epoch"], args.epochs):
        # the epoch's plan must be rebuilt from the token count at the START of the epoch: step sizes follow the
        # ramp, so a resume that planned from the current count would skip `step_in_epoch` steps of a different plan
        if epoch == state["epoch"] and "epoch_tokens_start" in state:
            tok0 = float(state["epoch_tokens_start"])
        elif epoch == state["epoch"] and state["step_in_epoch"] > 0:      # a checkpoint from before this field existed
            tok0 = float(got.get("tokens", 0.0)) if args.continue_from and not args.resume else 0.0
        else:
            tok0 = state["tokens"] if epoch == state["epoch"] else 0.0
        state["epoch_tokens_start"] = tok0
        dw = bool(args.domain_weights) and epoch == state["epoch"]
        if dw:
            if "dw_from_step" not in state:                  # the switch: the rest of this epoch follows the weights
                state.update({"dw_from_step": state["step_in_epoch"], "dw_tokens_start": state["tokens"], "dw_step": 0})
            units = domain_weighted_units(train_s, epoch, world, ramp, tok0, mean_share, args.max_states_micro,
                                          state["dw_from_step"], parse_weights(args.domain_weights),
                                          [d for d in args.repeat_domains.split(",") if d], args.max_passes)
            plan = plan_units(train_s, units[rank::world], ramp, state["dw_tokens_start"], mean_share, world,
                              args.max_states_micro)
            if rank == 0:
                print(f"[domain-weights] {args.domain_weights}: {len(units)} units from step {state['dw_from_step']}, "
                      f"resuming at weighted step {state['dw_step']}", flush=True)
        else:
            plan = plan_epoch(train_s, epoch, rank, world, ramp, tok0, mean_share, args.max_states_micro)
        n = torch.tensor([len(plan)], device=device)
        if world > 1:
            dist.all_reduce(n, op=dist.ReduceOp.MIN)
        plan = plan[:int(n.item())]
        start = (state["dw_step"] if dw else state["step_in_epoch"]) if epoch == state["epoch"] else 0
        for si in range(start, len(plan)):
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * wsd_lr_scale(state["tokens"], args.warmup_tokens, args.total_tokens,
                                              args.decay_frac, args.min_lr_frac)
            opt.zero_grad(set_to_none=True)
            n_states = sum(len(m) for m in plan[si])
            tok_local = 0.0
            dom_tok_local: Dict[str, float] = {}
            dom_stats: Dict[str, List[float]] = {}
            for micro in plan[si]:
                sts = [train_s.state(e[0], epoch=epoch + e[1]) if isinstance(e, tuple) else train_s.state(e, epoch=epoch)
                       for e in micro]
                loss, stats, _ = F.micro_batch_loss(model, sts, lcfg, block=args.block,
                                                    checkpoint_layers=args.checkpoint_layers,
                                                    conf_embed=args.conf_embed)
                (loss / max(1, n_states)).backward()
                for st_, s in zip(sts, stats):
                    tok_local += float(st_.meta["tokens_share"])
                    dom_tok_local[s["domain"]] = dom_tok_local.get(s["domain"], 0.0) + float(st_.meta["tokens_share"])
                    for k_ in ("loss", "ce", "cert", "calib", "acc", "top1"):
                        dom_stats.setdefault(f"{s['domain']}/{k_}", []).append(s[k_])
                        dom_stats.setdefault(f"all/{k_}", []).append(s[k_])
            T.all_reduce_grads(params, world)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            opt.step()
            tt = torch.tensor([tok_local, float(n_states)], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(tt)
            state["tokens"] += float(tt[0])
            state["step"] += 1
            if dw:
                state["epoch"], state["dw_step"] = epoch, si + 1
            else:
                state["epoch"], state["step_in_epoch"] = epoch, si + 1
            dtv = torch.tensor([dom_tok_local.get(d_, 0.0) for d_ in stat_doms], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(dtv)
            dom_tokens = state.setdefault("dom_tokens", {})
            for d_, v_ in zip(stat_doms, dtv.tolist()):
                dom_tokens[d_] = dom_tokens.get(d_, 0.0) + v_
            n_states_log += int(tt[1]); n_tok_log += float(tt[0])
            for k_, v in dom_stats.items():
                agg[k_] = agg.get(k_, 0.0) + float(np.mean(v))
                aggn[k_] = aggn.get(k_, 0.0) + 1.0
            agg["grad_norm"] = agg.get("grad_norm", 0.0) + gnorm
            aggn["grad_norm"] = aggn.get("grad_norm", 0.0) + 1.0

            if state["step"] % args.log_every == 0:
                dt = time.time() - t_last
                st_t = torch.tensor([agg.get(k_, 0.0) for k_ in stat_keys] + [aggn.get(k_, 0.0) for k_ in stat_keys],
                                    dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(st_t)
                sums, cnts = st_t[:len(stat_keys)].tolist(), st_t[len(stat_keys):].tolist()
                row = {k_: sm / cn for k_, sm, cn in zip(stat_keys, sums, cnts) if cn > 0}
                row.update({"step": state["step"], "tokens": state["tokens"], "lr": opt.param_groups[0]["lr"],
                            "dom_tokens": dict(state.get("dom_tokens", {})),
                            "states_per_s": n_states_log / max(dt, 1e-9), "tokens_per_s": n_tok_log / max(dt, 1e-9),
                            "states_per_step": int(tt[1]), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if rank == 0:
                    print(f"[step {state['step']}] {state['tokens'] / 1e6:.2f}M tok | loss {row.get('all/loss', 0):.4f} "
                          f"ce {row.get('all/ce', 0):.4f} cert {row.get('all/cert', 0):.4f} "
                          f"cal {row.get('all/calib', 0):.4f} acc {row.get('all/acc', 0):.3f} lr {row['lr']:.2e} "
                          f"{row['states_per_s']:.1f} states/s {row['tokens_per_s']:.0f} tok/s", flush=True)
                    wb.log(wandb_log.prefixed(row, "train"), step=state["step"])
                logf.write(json.dumps(row) + "\n"); logf.flush()
                agg, aggn, t_last, n_states_log, n_tok_log = {}, {}, time.time(), 0, 0.0

            hit = [m for m in milestones if state["tokens"] >= m]
            for m in hit:
                milestones.remove(m)
                state["milestones_done"].append(m)
                if rank == 0:
                    save_ckpt(model, opt, os.path.join(out, f"tok_{int(m):012d}"), dict(state))
                    print(f"[save] milestone {m / 1e6:.2f}M -> tok_{int(m):012d}", flush=True)
                do_probe(f"tok_{int(m):012d}")
                last_eval = state["tokens"]
            if state["tokens"] - last_eval >= args.eval_every_tokens:
                do_probe("periodic")
                last_eval = state["tokens"]
            if rank == 0 and state["tokens"] - last_roll >= args.save_every_tokens:
                save_ckpt(model, opt, os.path.join(out, f"roll_{state['step']:08d}"), dict(state))
                prune_rolling(out, args.keep_rolling)
                last_roll = state["tokens"]
            if args.max_tokens and state["tokens"] >= float(args.max_tokens):
                stop = True
                break
            if args.stop_at or args.stop_file or args.stop_at_file:   # wall clock / stop file, agreed by all ranks
                t_stop = args.stop_at
                if args.stop_at_file and os.path.exists(args.stop_at_file):     # moves the stop without a restart
                    try:
                        t_stop = float(open(args.stop_at_file).read().split()[0])
                    except (OSError, ValueError, IndexError):
                        pass
                late = torch.tensor([float((bool(t_stop) and time.time() >= t_stop)
                                           or (bool(args.stop_file) and os.path.exists(args.stop_file)))], device=device)
                if world > 1:
                    dist.all_reduce(late, op=dist.ReduceOp.MAX)
                if late.item() > 0:
                    if rank == 0:
                        print(f"[stop-at] {state['tokens'] / 1e6:.2f}M tokens at step {state['step']}", flush=True)
                    stop = True
                    break
        if stop:
            break
        state["step_in_epoch"] = 0
        state["epoch"] = epoch + 1

    if rank == 0:
        save_ckpt(model, opt, os.path.join(out, "final"), dict(state))
        print(f"[done] {state['tokens'] / 1e6:.2f}M tokens in {state['step']} steps -> {out}/final", flush=True)
    do_probe("final")
    wb.finish()
    logf.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return dict(state)


if __name__ == "__main__":
    main()
