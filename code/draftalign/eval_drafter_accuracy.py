#!/usr/bin/env python
"""Standalone accuracy of the DRAFTER (stock d32 or a fine-tuned checkpoint) on the 100 frozen train prompts
(calibvalid), decoding alone at block 32 with the stock rule (threshold .95, >= 1 commit per pass) -- the safeguard
against the drafter losing its own competence while gaining agreement with the big model.

  CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python code/draftalign/eval_drafter_accuracy.py --ckpt d32 \
      --out results/metrics/draftalign/drafter_accuracy_stock.json
  ... --ckpt logs/draftalign/train_drafter/d1 --out results/metrics/draftalign/drafter_accuracy_d1.json
Scoring: the vendored opencompass GSM8K evaluator through analyze_s1b.per_item_correct, as every accuracy on this branch.
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(ROOT, "compat"), ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import research_protocol as P                                                     # noqa: E402
from research_protocol import MASK_ID, EOS_IDS, GEN_LEN, encode_prompt, items_for_split, safe_out, LaunchLog  # noqa: E402
import torch                                                                       # noqa: E402
from sdar_harness.decode import block_diffusion_generate                          # noqa: E402
from draftalign.eval_drafter_agreement import load_drafter                        # noqa: E402
from analyze_s1b import per_item_correct                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="'d32' (stock drafter) or a checkpoint directory")
    ap.add_argument("--items", default="calibvalid"); ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda:0"
    tok, model, info = load_drafter(args.ckpt, device)
    items = items_for_split(args.items)
    if args.limit: items = items[:args.limit]
    gen_length = GEN_LEN[P.split_base(args.items)]
    launch = LaunchLog(f"draftalign/drafter_accuracy/{os.path.basename(args.ckpt.rstrip('/'))}.{args.items}",
                       {"ckpt": args.ckpt, "items": args.items, "threshold": args.threshold, "info": info})  # noqa: F841
    rows = []; t0 = time.perf_counter()
    for it in items:
        ids = encode_prompt(tok, it["prompt"]).to(device)
        with torch.no_grad():
            res = block_diffusion_generate(model, ids, mask_id=MASK_ID, gen_length=gen_length, block_length=32,
                                           denoising_steps=32, temperature=1.0, top_k=1, top_p=1.0,
                                           threshold=args.threshold, stopping_criteria_idx=EOS_IDS, greedy_exact=True)
        gen = res.generated_ids()[0].cpu(); nonmask = gen[gen != MASK_ID]
        text = tok.decode(nonmask.tolist(), skip_special_tokens=True)
        rows.append({"idx": it["idx"], "pred": text, "gold": it["gold"], "n_blocks": len(res.blocks),
                     "passes_per_block": [b.n_steps for b in res.blocks], "stopped_eos": bool(res.stopped_early)})
        print(f"[{len(rows)}/{len(items)}] idx {it['idx']} blocks {len(res.blocks)} passes/blk "
              f"{sum(b.n_steps for b in res.blocks)/max(1,len(res.blocks)):.1f}", flush=True)
    correct = per_item_correct("gsm8k" if P.split_base(args.items) in ("calib", "valid", "calibvalid", "gsm8k500", "gsm8k_pool") else "math500", rows)
    for r in rows: r["correct"] = int(correct.get(r["idx"], 0))
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    rep = {"ckpt": args.ckpt, "info": info, "items": args.items, "n": len(rows), "accuracy": acc,
           "passes_per_block_mean": sum(sum(r["passes_per_block"]) for r in rows) / max(1, sum(len(r["passes_per_block"]) for r in rows)),
           "seconds": time.perf_counter() - t0, "rows": rows}
    path = safe_out(args.out); os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rep, open(path, "w"), indent=1)
    print(f"[done] {args.ckpt}: accuracy {acc:.3f} on {len(rows)} items, passes/block {rep['passes_per_block_mean']:.2f} -> {path}")


if __name__ == "__main__":
    main()
