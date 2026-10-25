#!/usr/bin/env python
"""Judge round, step 5: corpus1b/train_c1b.py with the K1 / K2 / K3 state sampler (judge_round/sampler_k3.py).

    torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_k3.py --k-mix 0.15,0.25,0.60 \
        --index logs/c1b/trainset/p4drafts/index.jsonl --out logs/phase5/judge/main \
        --continue-from logs/c1b/train/main_128m/tok_000128000000 --max-tokens 178e6 --save-at-tokens 153e6,178e6

--k-mix = P(K1 pass-1 draft state), P(K2 rd2 masked-hole state), P(K3 rd1 re-draft state) for blocks whose item carries
a re-draft record (swap_drafts.py --redrafts); every other block is sampled exactly as train_c1b samples it.  Every
other argument is train_c1b's, passed through untouched; --k-mix is also added to train_c1b's parser so it lands in
the run's config.json.  The held-out probe still builds regime "draft_pass1" states, which K3Sampler hands to the
parent unchanged.
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
import train_c1b as TC                                                    # noqa: E402
from judge_round.sampler_k3 import DEFAULT_K_MIX, K3Sampler, parse_k_mix   # noqa: E402

DEFAULT_SPEC = ",".join(f"{v:.2f}" for v in DEFAULT_K_MIX)


def _add_k_mix(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--k-mix", default=DEFAULT_SPEC,
                    help="P(K1),P(K2),P(K3) in regime 'sample' for blocks with a re-draft record (sums to 1)")
    return ap


def main(argv=None, model_builder=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    known, _rest = _add_k_mix(argparse.ArgumentParser(add_help=False, allow_abbrev=False)).parse_known_args(argv)
    mix = parse_k_mix(known.k_mix)
    if os.environ.get("RANK", "0") == "0":
        print(f"[k3] state mix K1 (pass-1 drafts) {mix[0]:.3f} | K2 (rd2 masked holes) {mix[1]:.3f} | "
              f"K3 (rd1 re-drafts) {mix[2]:.3f}", flush=True)
    orig_sampler, orig_parser = TC.StateSampler, TC.build_parser
    TC.StateSampler = lambda cfg, entries=None, **kw: K3Sampler(cfg, entries, k_mix=mix, **kw)
    TC.build_parser = lambda: _add_k_mix(orig_parser())
    try:
        return TC.main(argv, model_builder=model_builder)
    finally:
        TC.StateSampler, TC.build_parser = orig_sampler, orig_parser


if __name__ == "__main__":
    main()
