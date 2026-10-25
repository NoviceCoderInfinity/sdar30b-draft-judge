#!/usr/bin/env python
"""Held-out precision analysis of a trained judge (GPU, one card, ~5 min).

For N held-out states in two regimes (pass-1 all-draft; mixed), run the flagged forward and collect, at every open
slot: top-1 prob, argmax, target, the input token (the draft token at flagged slots). Then report, for several commit
thresholds tau: the share of open slots that would commit, the precision of those commits, and the split into
"confirm" (argmax == draft token) vs "override" (argmax != draft) with their precisions -- so we know which kind of
confident commit is wrong and which threshold buys ~99% precision.

  CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/analyze_holdout_precision.py \
      --index logs/draftalign/trainset_gsm8k_r1/index.jsonl --adapter logs/draftalign/train/t2a_gsm8k_flag/step001057 \
      --n 300 --out results/metrics/draftalign/holdout_precision_t2a.json
"""
from __future__ import annotations
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(ROOT, "compat"), ROOT):          # NOT HERE: import everything through the `draftalign` package so
    if p not in sys.path: sys.path.insert(0, p)         # there is exactly ONE adapter module (one gate context CTX)
import research_protocol as P                                                     # noqa: E402
from research_protocol import MASK_ID, load_sdar, safe_out                        # noqa: E402
import torch                                                                       # noqa: E402
import torch.nn.functional as F                                                    # noqa: E402
from draftalign import adapter as A                                                # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                         # noqa: E402
from draftalign.train import forward_state                                         # noqa: E402
from draftalign import train_forward as _tf                                        # noqa: E402
assert sys.modules["draftalign.adapter"] is A and _tf.flag_scope is A.flag_scope, "duplicate adapter module: gate context would be split"

TAUS = (0.90, 0.95, 0.98, 0.99, 0.995)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--holdout-items", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--p-draftfull", type=float, default=0.3); ap.add_argument("--p-allmask", type=float, default=0.0)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    tok, model = load_sdar("b32", str(device)); del tok
    meta = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    flagged = bool(meta.get("draft_embed", False))
    A.load_adapter(model, args.adapter); model.eval()          # attaches LoRA/gates/draft_embed from the metadata
    scfg = SamplerConfig(index_path=args.index, seed=args.seed, p_pass1=0.4, p_allmask=args.p_allmask,
                         p_draftfull=args.p_draftfull, mask_id=MASK_ID, max_seq=1200, only_correct=False)
    full = StateSampler(scfg); _, eval_s = full.split_holdout(args.holdout_items)
    g = torch.Generator().manual_seed(args.seed); pick = torch.randperm(len(eval_s), generator=g)[:args.n].tolist()
    rows = {"draft_pass1": [], "sample": []}
    with torch.no_grad():
        for regime in rows:
            for k in pick:
                st = eval_s.state(k, regime=regime)
                if not st.meta["has_draft"]: continue
                _, _, logits, open_rows = forward_state(model, st, None, "full", False, 0.0, 0.5, MASK_ID, device,
                                                        want_eval=False, block=32, draft_flag=flagged)
                p = F.softmax(logits, dim=-1); top, am = p.max(-1)
                blk = st.input_ids[st.active_lo:st.active_lo + 32]
                inp = blk[st.open_mask].to(device); tgt = st.targets.to(device)[st.open_mask]
                is_draft = inp != MASK_ID
                for i in range(int(tgt.numel())):
                    rows[regime].append((float(top[i]), int(am[i] == tgt[i]), int(is_draft[i]),
                                         int(am[i] == inp[i]), int(inp[i] == tgt[i])))
    rep = {"adapter": args.adapter, "n_states": args.n, "regimes": {}}
    for regime, R in rows.items():
        if not R: continue
        t = torch.tensor(R)  # top, correct, is_draft, agree, draft_correct
        top, corr, isd, agree, dc = t.T
        d = isd.bool()
        out = {"open_slots": int(t.shape[0]), "draft_slots": int(d.sum()), "draft_correct_rate": float(dc[d].float().mean()) if d.any() else None,
               "argmax_acc_all": float(corr.float().mean()), "argmax_acc_draft_slots": float(corr[d].float().mean()) if d.any() else None,
               "argmax_acc_masked_slots": float(corr[~d].float().mean()) if (~d).any() else None, "by_tau": {}}
        for tau in TAUS:
            c = top > tau
            e = {"commit_share": float(c.float().mean()), "precision": float(corr[c].float().mean()) if c.any() else None}
            for name, m in (("confirm", c & d & agree.bool()), ("override", c & d & ~agree.bool()), ("masked", c & ~d)):
                e[name] = {"share_of_open": float(m.float().mean()), "precision": float(corr[m].float().mean()) if m.any() else None}
            out["by_tau"][str(tau)] = e
        rep["regimes"][regime] = out
    path = safe_out(args.out); os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
