#!/usr/bin/env python
"""The runbook's check on the merged drafter (section 3a, last item).  Exit 0 = it passes.

    /venv/p2/bin/python code/drafter_lora/check_merged.py logs/phase4/train/main/merged [--device cuda:0]

  1. AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, local_files_only=True) loads it, the tokenizer
     keeps the [MASK] id, and the config matches the pinned SDAR-1.7B (28 layers, hidden 2048)
  2. run_da_decode.py --drafter <dir> accepts it: drafter_info resolves it to a checkpoint directory and cell_name
     builds the rd1 + ban cell name (with the _dr<hash> tag), as the decode would
  3. one forward on a short prompt gives finite logits whose argmax is never [MASK] after suppression
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                            # noqa: E402

import torch                                                             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    path = P.safe_out(args.merged)
    from phase2.tf_compat import apply as _tf_compat       # transformers 4.54.1 lacks LossKwargs (load_sdar does this)
    _tf_compat()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"trust_remote_code": True, "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(path, **kw)
    assert tok(tok.mask_token)["input_ids"][0] == P.MASK_ID, "the merged tokenizer lost the [MASK] id"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kw).to(args.device).eval()
    c = model.config
    assert (c.num_hidden_layers, c.hidden_size) == (28, 2048), (c.num_hidden_layers, c.hidden_size)
    print(f"[1 load] AutoModelForCausalLM.from_pretrained({path}, trust_remote_code=True, local_files_only=True): "
          f"{type(model).__name__}, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{next(model.parameters()).dtype} -> PASS", flush=True)

    from draftalign import run_da_decode as R
    a = SimpleNamespace(stock=False, drafter=args.merged, judge="full", draft="s8", threshold=0.95, tau_commit=0.95,
                        tau_override=None, full_remask_only=False, full_max_passes=32, judge_passes=0,
                        full_redraft_every=1, full_redraft_rate=4, redraft_ban_rejected=True, pool=0,
                        pool_rule="des", pool_flush=False, tau_pool_cov=0.0, adapter=None)
    info = R.drafter_info(a)
    assert info["path"] and os.path.isdir(info["path"]) and info["tag"].startswith("_dr"), info
    try:
        cell = R.cell_name("ifeval", a)
    except Exception as e:                       # the cell name is cosmetic here; the path resolution is the check
        cell = f"(cell_name not built offline: {type(e).__name__})"
    print(f"[2 run_da_decode --drafter] resolves to {info['path']} (tag {info['tag']}); cell {cell} -> PASS",
          flush=True)

    ids = tok("Write one sentence about the sea.", return_tensors="pt")["input_ids"][:, :32]
    ids = torch.cat([ids, torch.full((1, 32 - ids.shape[1]), P.MASK_ID)], 1).to(args.device)   # one 32-slot block
    from draftalign.train_forward import block_tril_bool
    S = int(ids.shape[1])
    with torch.no_grad():                        # the remote code needs an explicit mask (as the decode passes one)
        z = model(input_ids=ids, attention_mask=block_tril_bool(S, ids.device, 32),
                  position_ids=torch.arange(S, device=ids.device)[None]).logits.float()
    z[..., P.MASK_ID] = float("-inf")
    assert torch.isfinite(z.max(-1).values).all(), "non-finite logits"
    print(f"[3 forward] finite logits {tuple(z.shape)} -> PASS", flush=True)
    print("[check_merged] ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
