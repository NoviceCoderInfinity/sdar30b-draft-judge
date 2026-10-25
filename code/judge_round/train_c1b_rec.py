#!/usr/bin/env python
"""Judge round, step 5 on RECORDED states: corpus1b/train_c1b.py with judge_round/sampler_rec.py's RecStateSampler.

    torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_rec.py \
        --states logs/phase5/judge_states/fresh_rd1 --index logs/c1b/trainset/fresh/index.jsonl \
        --out logs/phase5/judge/rec_main --continue-from logs/c1b/train/main_128m/tok_000128000000 \
        --max-tokens 178e6 --save-at-tokens 153e6,178e6

Training stays train_c1b's in every other respect, on the HF/torch trainer: the pinned model, the round-3 adapter,
the loss, AdamW, the schedule, plan_epoch, the held-out probe, checkpoints, --continue-from, --resume.  It changes no
recipe or numerics, so under the operator's rule of 23 Oct (training speed-ups are allowed, each recipe or numerics
change is a deviation logged in STATUS.md) it adds no deviation.  This wrapper adds five arguments to train_c1b's
parser (so they land in the run's config.json):

  --states DIR            an index_states.py (A0) output dir: the records of THIS --index (pids join the two)
  --pass-pick uniform     which recorded pass a block's state is (one state per block per epoch)
  --post-eos truth|exact  the post-EOS slots of a last block (sampler_rec.py)
  --step0-states N        the step-0 check below on N recorded states (0 = off; default 256)
  --step0-min-agree F     its bar (default 0.95)

Refused: --conf-embed (the 128M adapter has no confidence embedding and recorded states carry no confidence).

STEP-0 CHECK.  The run continues from the very adapter that recorded the states (--continue-from).  Right after that
checkpoint is loaded, before the first optimiser step, the trainer's own forward (train_fast.micro_batch_loss) on N
recorded states must reproduce the recorded judge argmax at the loss rows on at least --step0-min-agree of them
(default 0.95, the parity bar of docs/PHASE5_SGLANG_JUDGE.md section 8).  Far below means a sampler bug (prefix,
flag, positions) or a kernel gap; the run stops (SystemExit) before it trains on anything.  On CPU in fp32 the
agreement is exact (code/tests/test_sampler_rec.py).  Every rank checks the same N states; without --continue-from
(a fresh adapter did not record these states) the check is skipped and says so.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

import train_c1b as TC                                                     # noqa: E402
from draftalign import train_fast as F                                     # noqa: E402
from judge_round import sampler_rec as SR                                  # noqa: E402
from judge_round.index_states import StatesIndex                           # noqa: E402


def _add_rec(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--states", required=True, help="an index_states.py output dir (the records of --index)")
    ap.add_argument("--pass-pick", choices=SR.PASS_PICKS, default="uniform",
                    help="which recorded pass a block's state is, once per block per epoch")
    ap.add_argument("--post-eos", choices=SR.POST_EOS, default="truth",
                    help="truth: the trainset's valid slots (default); exact: the recorder's (adds post-EOS rows)")
    ap.add_argument("--step0-states", type=int, default=256, help="recorded states for the step-0 check (0 = off)")
    ap.add_argument("--step0-min-agree", type=float, default=0.95, help="the step-0 check's argmax agreement bar")
    return ap


@torch.no_grad()
def step0_check(model, sampler, store: StatesIndex, n: int, *, micro: int = 32, seed: int = 0) -> dict:
    """The trainer's forward on `n` recorded states (pass drawn as in epoch 0) vs the recorded judge read-out at the
    loss rows: argmax agreement and the largest |top-1 p - recorded p|."""
    block, mask_id = int(sampler.cfg.block), int(sampler.cfg.mask_id)
    was_training = model.training
    model.eval()
    try:
        g = torch.Generator().manual_seed(seed)
        ks = torch.randperm(len(sampler), generator=g)[:max(0, int(n))].tolist()
        lcfg = F.LossCfg(mask_id=mask_id)
        agree = rows = 0
        max_dp = 0.0
        for j in range(0, len(ks), micro):
            sts = [sampler.state(k, 0, "sample") for k in ks[j:j + micro]]
            _, _, logits = F.micro_batch_loss(model, sts, lcfg, block=block, checkpoint_layers=False, want_eval=True)
            for st, lg in zip(sts, logits):
                lg = lg.float().cpu().clone()
                lg[:, mask_id] = float("-inf")
                top, am = torch.softmax(lg, dim=-1).max(dim=-1)
                r = int(st.meta["rec_pass"])
                ra = torch.from_numpy(np.array(store.a["argmax"][r], np.int64))[st.open_mask]      # copies: the index is read-only
                rp = torch.from_numpy(np.array(store.a["p"][r], np.float32))[st.open_mask]
                agree += int((am == ra).sum())
                rows += int(st.open_mask.sum())
                if len(rp):
                    max_dp = max(max_dp, float((top - rp).abs().max()))
        return {"states": len(ks), "rows": rows, "agree": agree / max(1, rows), "max_abs_dp": max_dp}
    finally:
        if was_training:
            model.train()


def main(argv=None, model_builder=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--conf-embed" in argv:
        raise SystemExit("[rec] --conf-embed is refused: the 128M judge adapter has no confidence embedding and "
                         "recorded states carry no drafter confidence")
    pre = _add_rec(argparse.ArgumentParser(add_help=False, allow_abbrev=False))
    pre.add_argument("--continue-from", default=None)
    known, _rest = pre.parse_known_args(argv)
    rank0 = os.environ.get("RANK", "0") == "0"
    store = StatesIndex(known.states)
    if rank0:
        print(f"[rec] states {store.summary()} | pass_pick {known.pass_pick} | post_eos {known.post_eos}", flush=True)

    made = []

    def make_sampler(cfg, entries=None, **kw):
        s = SR.RecStateSampler(cfg, entries, store=store, pass_pick=known.pass_pick, post_eos=known.post_eos, **kw)
        made.append(s)
        return s

    orig_sampler, orig_parser, orig_load = TC.StateSampler, TC.build_parser, TC.load_ckpt

    def load_and_check(model, opt, path, device):
        got = orig_load(model, opt, path, device)
        if known.continue_from is None or os.path.abspath(path) != os.path.abspath(TC.P.safe_out(known.continue_from)):
            return got                                       # --resume of this run: it already trained
        if known.step0_states <= 0 or not made:
            if rank0:
                print("[step0] check off (--step0-states 0)", flush=True)
            return got
        res = step0_check(model, made[0], store, known.step0_states)
        if rank0:
            print(f"[step0] {res['states']} recorded states, {res['rows']} loss rows: argmax agreement "
                  f"{res['agree']:.4f} (bar {known.step0_min_agree}), max |dp| {res['max_abs_dp']:.2e}", flush=True)
        if res["agree"] < known.step0_min_agree:
            raise SystemExit(f"[step0] argmax agreement {res['agree']:.4f} < {known.step0_min_agree}: the trainer does "
                             f"not reproduce the recording judge on its own states (sampler or kernel gap); not "
                             f"training")
        return got

    TC.StateSampler = make_sampler
    TC.build_parser = lambda: _add_rec(orig_parser())
    TC.load_ckpt = load_and_check
    try:
        if known.continue_from is None and known.step0_states > 0 and rank0:
            print("[step0] skipped: no --continue-from (a fresh adapter did not record these states)", flush=True)
        return TC.main(argv, model_builder=model_builder)
    finally:
        TC.StateSampler, TC.build_parser, TC.load_ckpt = orig_sampler, orig_parser, orig_load


if __name__ == "__main__":
    main()
