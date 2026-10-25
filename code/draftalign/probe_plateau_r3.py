#!/usr/bin/env python
"""Why did r3_A's curves flatten? Per-pool / per-slot-kind probe of ONE checkpoint (GPU, one card, ~6-10 min).

Same holdout split as training (--holdout-items 150, seed 1) and the same deterministic states for every checkpoint.
Three state sets: holdout pass-1 all-draft (the inference state), holdout mixed ("sample", the training regime), and
TRAIN-split pass-1 all-draft (memorisation check). At every open slot, grouped by pool (opencode / gsm8k / if) and slot
kind -- draft_ok (draft token == target: the judge should CONFIRM), draft_bad (draft != target: it should OVERRIDE),
mask -- report: n, argmax accuracy, CE, mean top-1, keeps-draft share, commit share and precision at tau 0.95.
  CUDA_VISIBLE_DEVICES=0 /venv/r3/bin/python code/draftalign/probe_plateau_r3.py --adapter <step dir | stock> --n 160 --out x.json
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(ROOT, "compat"), ROOT):          # import through the `draftalign` package (one adapter module)
    if p not in sys.path: sys.path.insert(0, p)
from research_protocol import MASK_ID, load_sdar, safe_out                        # noqa: E402
import torch                                                                       # noqa: E402
import torch.nn.functional as F                                                    # noqa: E402
from draftalign import adapter as A                                                # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                         # noqa: E402
from draftalign.train import forward_state                                         # noqa: E402

TAU = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="logs/draftalign/trainset_r3/index.jsonl")
    ap.add_argument("--adapter", required=True, help="checkpoint dir, or 'stock'")
    ap.add_argument("--n", type=int, default=160); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    tok, model = load_sdar("b32", str(device)); del tok
    flagged = False
    if args.adapter != "stock":
        meta = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
        flagged = bool(meta.get("draft_embed", False))
        A.load_adapter(model, args.adapter)
    model.eval()
    scfg = SamplerConfig(index_path=args.index, seed=1, p_pass1=0.4, p_allmask=0.0, p_draftfull=0.3,
                         mask_id=MASK_ID, max_seq=1200, only_correct=False)
    train_s, eval_s = StateSampler(scfg).split_holdout(150)
    sets = {"holdout_pass1": (eval_s, "draft_pass1"), "holdout_mixed": (eval_s, "sample"), "train_pass1": (train_s, "draft_pass1")}
    acc = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for name, (smp, regime) in sets.items():
            g = torch.Generator().manual_seed(0)
            for k in torch.randperm(len(smp), generator=g)[:args.n].tolist():
                st = smp.state(k, regime=regime)
                if not st.meta["has_draft"]: continue
                _, _, logits, _ = forward_state(model, st, None, "full", False, 0.0, 0.5, MASK_ID, device,
                                                want_eval=False, block=32, draft_flag=flagged)
                p = F.softmax(logits.float(), dim=-1); top, am = p.max(-1)
                blk = st.input_ids[st.active_lo:st.active_lo + 32]
                inp = blk[st.open_mask].to(device); tgt = st.targets.to(device)[st.open_mask]
                ce = -torch.log(p.gather(-1, tgt[:, None]).squeeze(-1).clamp_min(1e-12))
                pool = str(st.meta.get("split", "?")).replace("_pool", "")
                kind = torch.where(inp == MASK_ID, 0, torch.where(inp == tgt, 1, 2))
                for kid, kname in ((0, "mask"), (1, "draft_ok"), (2, "draft_bad")):
                    m = kind == kid
                    if not m.any(): continue
                    c = (am == tgt) & m; conf = (top > TAU) & m
                    for key in (f"{name}|{pool}|{kname}", f"{name}|all|{kname}", f"{name}|{pool}|all", f"{name}|all|all"):
                        a = acc[key]
                        a["n"] += float(m.sum()); a["correct"] += float(c.sum()); a["ce"] += float(ce[m].sum())
                        a["top1"] += float(top[m].sum()); a["keep_draft"] += float(((am == inp) & m).sum())
                        a["commit"] += float(conf.sum()); a["commit_ok"] += float((conf & c).sum())
    rep = {"adapter": args.adapter, "n_states_per_set": args.n, "tau": TAU, "groups": {}}
    for key, a in sorted(acc.items()):
        n = a["n"] or 1.0
        rep["groups"][key] = {"n": int(a["n"]), "acc": a["correct"] / n, "ce": a["ce"] / n, "top1": a["top1"] / n,
                              "keep_draft": a["keep_draft"] / n, "commit_share": a["commit"] / n,
                              "precision": (a["commit_ok"] / a["commit"]) if a["commit"] else None}
    path = safe_out(args.out); os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep["groups"].items() if k.endswith("|all|all")}, indent=1))


if __name__ == "__main__":
    main()
