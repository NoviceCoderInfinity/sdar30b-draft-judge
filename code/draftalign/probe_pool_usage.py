#!/usr/bin/env python
"""Step 1 of the expert-utilisation probe: WHICH experts actually run, captured from a real decode.

The decode records on disk carry only counts (`unique_expert_layer_loads`, `uniq_by_kind`), never
identities, so "which experts fire together" is unanswerable from them.  This dumps the identities.

It captures `MoECapture.ids` -- the `selected_experts` tensor the dispatch loop actually indexes.
That matters: `moe_instr`'s docstring records that re-deriving the top-8 from `router_logits`
afterwards disagreed with an fp32 recompute ~10 % of the time on strict set identity, so capture is
the only trustworthy route.

Nothing here modifies da_decode.py or run_da_decode.py -- `decode_item` takes the meter as an
argument, so a subclass is enough.  Per item it writes <out>/routing/<idx>.npz:

    ids   [F, L, 32, 8] uint8   selected experts, padded with 255 where a forward had < 32 rows
    npos  [F] int16            true row count of each forward
    kind  [F] int8             MOE_KIND_CODE (1 denoise, 3 judge, 4 rejudge)
    pool  [F] int16            pool size in force for that forward (0 = unrestricted)

Usage mirrors run_da_decode:
  probe_pool_usage.py --model b32 --drafter d32 --items calibvalid --adapter <dir> --judge full \
      --draft s8 --pool 48 --pool-rule des --pool-flush --limit 24 --shard 0 --nshards 4 --gpu 0
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # code/
for p in (HERE, os.path.join(HERE, "compat")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import research_protocol as P                                               # noqa: E402
from research_protocol import (MASK_ID, EOS_IDS, GEN_LEN, safe_out, items_for_split,  # noqa: E402
                               encode_prompt)
from draftalign.da_decode import (MoEMeter, SeqCache, POOL, decode_item,            # noqa: E402
                                  MOE_KIND_CODE, POOLED_KINDS)
BLOCK = 32                        # run_da_decode.py:65 (the block size this whole line uses)
from draftalign import run_da_decode as R                                    # noqa: E402

KEEP = ("denoise", "judge", "rejudge")          # block forwards; prefill/flush are not the question


class RoutingMeter(MoEMeter):
    """MoEMeter + a per-forward dump of the expert ids the dispatch used."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.routing = []                        # (kind_code, npos, pool_size, ids [L, N, 8] uint8)

    def reset(self):
        super().reset()
        self.routing = []

    def _record(self, cap, kind: str, npos: int, pool_size: int = 0):
        super()._record(cap, kind, npos, pool_size)
        if cap is None or not cap.ids or kind not in KEEP:
            return
        n = min(int(cap.ids[0].shape[0]), BLOCK)
        buf = np.full((len(cap.ids), BLOCK, 8), 255, np.uint8)
        for li, ids in enumerate(cap.ids):
            buf[li, :n] = ids[:n].to(torch.uint8).cpu().numpy()
        self.routing.append((MOE_KIND_CODE[kind], int(npos), int(pool_size), buf))

    def dump(self, path: str):
        if not self.routing:
            return 0
        np.savez_compressed(
            path,
            ids=np.stack([r[3] for r in self.routing]),
            kind=np.array([r[0] for r in self.routing], np.int8),
            npos=np.array([r[1] for r in self.routing], np.int16),
            pool=np.array([r[2] for r in self.routing], np.int16),
        )
        return len(self.routing)


def main(argv=None):
    ap = R.build_parser()
    ap.add_argument("--routing-out", default="logs/draftalign/probes/routing")
    args = R.resolve_args(ap.parse_args(argv))

    base = P.split_base(args.items)
    gen_length = GEN_LEN[base]
    ad_hash = R.adapter_weights_hash(R.adapter_dir(args.adapter)) if args.adapter else None
    out = safe_out(args.routing_out)
    os.makedirs(out, exist_ok=True)

    items = items_for_split(args.items)
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.nshards]
    print(f"[plan] {len(items)} items -> {out}", flush=True)

    torch.manual_seed(0)
    tok, moe_model, tok_d, drafter, sink, ad = R.load_models(args)
    POOL.configure(int(args.pool) or None, args.pool_rule)
    meter = RoutingMeter(SeqCache(moe_model, BLOCK, "moe"), sink,
                         pool=POOL if args.pool else None,
                         pool_flush=bool(args.pool_flush),
                         draft_flag=bool((ad or {}).get("draft_flag")))
    dr_seq = None if args.stock else SeqCache(drafter, BLOCK, "drafter")
    gen_kw = dict(block_length=BLOCK, denoising_steps=BLOCK, threshold=args.threshold,
                  draft=None if args.stock else args.draft, tau_commit=args.tau_commit,
                  tau_acc=args.tau_acc, tau_acc_q=args.tau_acc_q, judge=args.judge,
                  tau_draft=args.tau_draft, tau_override=args.tau_override,
                  full_max_passes=args.full_max_passes,
                  draft_after_pass=0 if args.stock else int(args.draft_after_pass),
                  judge_passes=0 if args.stock else int(args.judge_passes))

    t0, nf = time.time(), 0
    for k, it in enumerate(items):
        prompt_ids = encode_prompt(tok, it["prompt"])
        meter.reset()
        decode_item(meter, dr_seq, tok, it, prompt_ids, mask_id=MASK_ID, eos_ids=EOS_IDS,
                    gen_length=gen_length, adapter=args.adapter, adapter_hash=ad_hash, **gen_kw)
        n = meter.dump(os.path.join(out, f"{int(it['idx']):05d}.npz"))
        nf += n
        print(f"[{k+1}/{len(items)}] idx {it['idx']}: {n} block forwards captured "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"[done] {nf} forwards from {len(items)} items in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
