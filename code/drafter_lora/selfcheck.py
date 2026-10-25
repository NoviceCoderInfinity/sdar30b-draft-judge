#!/usr/bin/env python
"""Phase 4 self-check on the REAL drafter and REAL corpus items (CPU or one GPU; minutes).  Exit 0 = all pass.

    /venv/p2/bin/python code/drafter_lora/selfcheck.py --index logs/c1b/trainset/waveA/index.jsonl --device cuda:0
    (p4.sh setup runs it; the CPU tests use a tiny stand-in model, this uses SDAR-1.7B @ MODELS["d32"])

  1. count       LoRA r 32 on q,k,v,o,gate,up,down of the real model = 34,865,152 trainable parameters
  2. forward     the two-stream forward == the model's OWN remote-code forward (model.model, block-tril mask) on
                 [clean prefix + the block in its training state], for every block of --n-items items, adapter
                 perturbed so it is not the identity (argmax agreement and max |diff| of the hidden states)
  3. kl0         with a fresh adapter (lora_B = 0) the KL term is exactly 0 and the loss is the CE
  4. step        one AdamW step at lr 1e-3 lowers the loss on the same micro-batch
  5. parity      the untrained drafter's s8 replay vs the stored SGLang drafts (the smoke test asks >= 95%; on CPU
                 in fp32 the number is indicative only -- the stored drafts were made in bf16 on GPU)
  6. merge       save adapter -> save_merged -> reload with eval_drafter_agreement.load_drafter (what
                 run_da_decode.py --drafter uses): logits equal the un-merged model's (bf16 merge: argmax agreement)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

from draftalign.train_forward import block_tril_bool                     # noqa: E402
from drafter_lora import data as D                                       # noqa: E402
from drafter_lora import lora as LA                                      # noqa: E402
from drafter_lora.forward import drafter_loss, two_stream_hidden         # noqa: E402
from drafter_lora.replay import parity_with_stored, replay_item, stored_agreement   # noqa: E402

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def perturb(model, scale: float, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in LA.trainable(model):
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * scale)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=tuple(DT), default=None, help="default: fp32 on CPU, bf16 on GPU")
    ap.add_argument("--n-items", type=int, default=3)
    ap.add_argument("--parity-items", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=900, help="skip items whose two-stream length exceeds this")
    ap.add_argument("--out", default="logs/phase4/selfcheck.json")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    dtype = DT[args.dtype or ("fp32" if dev.type == "cpu" else "bf16")]
    res, t0 = {}, time.time()

    entries = D.load_entries(args.index)
    entries = [e for e in entries if os.path.isfile(e["_path"]) and 0 < D.two_stream_length(e) <= args.max_len
               and int(e["block_start"][0]) > 0]
    assert entries, "no usable item found (are the npz files extracted?)"
    pick = entries[:args.n_items]

    _tok, model = P.load_sdar("d32", str(dev), dtype=dtype)
    LA.attach(model)                                                      # the runbook's defaults
    c = LA.count(model)
    res["count"] = {"value": c, "pass": c["total"] == LA.EXPECTED_TRAINABLE_D32}
    print(f"[1 count] {c} -> {'PASS' if res['count']['pass'] else 'FAIL'}", flush=True)

    # 3. KL exactly 0 with the fresh adapter (before any perturbation)
    model.train()
    exs = [D.item_example(D.load_item(e), seed=0) for e in pick]
    batch = D.collate(exs)
    loss, st = drafter_loss(model, batch, checkpoint_layers=False, autocast=(dev.type == "cuda"))
    res["kl0"] = {"kl_sum": st["kl_sum"], "loss": float(loss.detach()), "ce": st["ce_sum"] / st["n_rows"],
                  "rows": st["n_rows"], "acc0": st["n_right0"] / st["n_rows"],
                  "pass": st["kl_sum"] == 0.0 and abs(float(loss.detach()) - st["ce_sum"] / st["n_rows"]) < 1e-4}
    print(f"[3 kl0] KL sum {st['kl_sum']:.3e}, loss {float(loss.detach()):.4f} = CE {res['kl0']['ce']:.4f} over "
          f"{st['n_rows']:.0f} rows (original drafter right on {res['kl0']['acc0']:.3f}) -> "
          f"{'PASS' if res['kl0']['pass'] else 'FAIL'}", flush=True)

    # 4. one optimiser step lowers the loss on the same micro-batch
    opt = torch.optim.AdamW([p for _, p in LA.trainable(model)], lr=1e-3, betas=(0.9, 0.95))
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        loss2, _ = drafter_loss(model, batch, checkpoint_layers=False, autocast=(dev.type == "cuda"))
    res["step"] = {"before": float(loss.detach()), "after": float(loss2), "pass": float(loss2) < float(loss.detach())}
    print(f"[4 step] loss {float(loss.detach()):.4f} -> {float(loss2):.4f} -> {'PASS' if res['step']['pass'] else 'FAIL'}",
          flush=True)

    # 2. two-stream == the remote-code forward, block by block (adapter perturbed: not the identity)
    perturb(model, 0.02 if dtype == torch.float32 else 0.05)
    model.eval()
    worst, agree, n = 0.0, 0, 0
    with torch.no_grad():
        for e, ex in zip(pick, exs):
            it = D.load_item(e)
            b1 = D.collate([ex])
            ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda"))
            with ac:
                h2 = two_stream_hidden(model, b1.ids.to(dev), b1.pos.to(dev), b1.attn.to(dev), False)[0].float()
            clean = D.clean_sequence(it)
            for j, blk in enumerate(ex.blocks):
                lo = int(it["block_start"][blk]); S = lo + 32
                seq = torch.as_tensor(clean[:S]).clone()
                seq[lo:S] = ex.ids[ex.n_clean + 32 * j: ex.n_clean + 32 * (j + 1)]
                with ac:
                    ref = model.model(input_ids=seq[None].to(dev), attention_mask=block_tril_bool(S, dev, 32),
                                      position_ids=torch.arange(S, device=dev)[None], use_cache=False,
                                      store_kv=False).last_hidden_state[0, lo:S].float()
                got = h2[ex.n_clean + 32 * j: ex.n_clean + 32 * (j + 1)]
                worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
                la, lb = model.lm_head(got.to(dtype)).argmax(-1), model.lm_head(ref.to(dtype)).argmax(-1)
                agree += int((la == lb).sum()); n += 32
    tol = 1e-3 if dtype == torch.float32 else 3e-2
    res["forward"] = {"max_rel_diff": worst, "argmax_agree": agree / n, "positions": n,
                      "pass": worst < tol and agree / n >= (1.0 if dtype == torch.float32 else 0.99)}
    print(f"[2 forward] max |diff| / max |h| = {worst:.2e} (tol {tol:.0e}), argmax agreement {agree / n:.4f} over "
          f"{n} positions -> {'PASS' if res['forward']['pass'] else 'FAIL'}", flush=True)

    # 6. merge round trip through the decode runner's loader
    out_dir = P.safe_out(os.path.join("logs", "phase4", "_selfcheck"))
    shutil.rmtree(out_dir, ignore_errors=True)
    LA.save(model, os.path.join(out_dir, "adapter"))
    probe = torch.as_tensor(D.clean_sequence(D.load_item(pick[0]))[:96])[None].to(dev)
    with torch.no_grad():
        want = model(input_ids=probe, attention_mask=block_tril_bool(96, dev, 32),
                     position_ids=torch.arange(96, device=dev)[None]).logits.float()
    del model, opt
    merged = LA.save_merged("d32", os.path.join(out_dir, "adapter"), os.path.join(out_dir, "merged"), device=str(dev))
    from draftalign.eval_drafter_agreement import load_drafter
    _t, m2, info = load_drafter(merged, str(dev))
    with torch.no_grad():
        got = m2(input_ids=probe, attention_mask=block_tril_bool(96, dev, 32),
                 position_ids=torch.arange(96, device=dev)[None]).logits.float()
    am = float((got.argmax(-1) == want.argmax(-1)).float().mean())
    res["merge"] = {"argmax_agree": am, "loaded_dtype": str(next(m2.parameters()).dtype), "info": info,
                    "pass": am >= 0.97}
    print(f"[6 merge] merged dir loads via load_drafter ({res['merge']['loaded_dtype']}); argmax agreement with the "
          f"un-merged adapter {am:.4f} over 96 positions -> {'PASS' if res['merge']['pass'] else 'FAIL'}", flush=True)

    # 5. replay parity of the UNTRAINED drafter with the stored SGLang drafts
    del m2
    _tok, base = P.load_sdar("d32", str(dev), dtype=dtype)
    base.eval()
    eq = sl = agree = slots = s_agree = s_slots = 0
    for e in pick[:args.parity_items]:
        it = D.load_item(e)
        r = replay_item(base, it, dev)
        pr = parity_with_stored(it, r["drafts"])
        sa = stored_agreement(it)
        eq += pr["equal"]; sl += pr["slots"]; agree += int(r["right"].sum()); slots += int(r["valid"].sum())
        s_agree += sa["agree"]; s_slots += sa["slots"]
    par, ra, sa_ = eq / max(sl, 1), agree / max(slots, 1), s_agree / max(s_slots, 1)
    # bf16 on GPU: token parity with the stored (bf16, GPU) drafts must be >= 95%.  fp32 / CPU: one early near-tie
    # flips a whole block's s8 order, so token parity is not comparable; the procedure is checked instead -- the
    # replay's agreement with the 30B must match the stored drafts' within 3 points.
    strict = dtype == torch.bfloat16 and dev.type == "cuda"
    ok = par >= 0.95 if strict else abs(ra - sa_) <= 0.03
    res["parity"] = {"parity": par, "slots": sl, "replay_agreement": ra, "stored_agreement": sa_,
                     "criterion": "parity >= 0.95" if strict else "|replay - stored agreement| <= 0.03",
                     "dtype": str(dtype), "device": str(dev), "pass": ok}
    print(f"[5 parity] untrained replay == stored drafts on {par:.4f} of {sl} slots ({dtype}, {dev}); agreement with "
          f"the 30B: replay {ra:.4f} vs stored {sa_:.4f}; criterion {res['parity']['criterion']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    shutil.rmtree(out_dir, ignore_errors=True)
    ok = all(v["pass"] for v in res.values())
    res["all_pass"] = ok
    res["seconds"] = round(time.time() - t0, 1)
    out = P.safe_out(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"[selfcheck] {'ALL PASS' if ok else 'NOT ALL PASS'} in {res['seconds']:.0f}s -> {out}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
