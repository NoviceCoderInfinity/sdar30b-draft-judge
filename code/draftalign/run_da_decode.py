#!/usr/bin/env python
"""Runner for the training-free Draft-Verify decode (draftalign.da_decode) and its stock integrity twin.

MoE = research_protocol.MODELS[--model] with the bit-exact instrumented hit-experts-only MoE forward
(sdar_harness.moe_instr) for the expert counters; drafter = MODELS[--drafter] behind a persistent
SeqCache.  Records carry the SAME fields as run_research_baselines.decode_item (so
analyze_research_baselines.py scores them) plus the DV extras; a per-block trace NPZ per item.

  --stock   identical loop with NO drafter and NO judge; must reproduce the recorded stock cell
            (logs/research_v1/baselines_b32/gsm8k_B32_t95_stock.<items>.shard*.jsonl, field gen_ids)
            item for item -- the runner compares each item against that record when it exists
            (`ref_match` in the record, summary at the end of the shard).

  --judge full  the TRAINED regime (da_decode's full judge): draft-filled judge forwards, read at every
            open slot, commit > --tau-commit, keep a_i == D_i as tentative context else re-mask, until
            every valid slot is committed; --full-max-passes caps it (the rest falls back to the stock
            rule), --full-remask-only is the no-tentative ablation.
  --draft-after-pass N   judge blind, ANCHORED DRAFTING (default 0 = draft before pass 1): run N STOCK
            passes on the all-mask block first, then draft only the still-open slots with those commits
            held fixed as anchors, then the usual blind acceptance on the next stock pass.  Cell `_da{N}`.
  --judge-passes K   judge full, ONE-PASS HYBRID (default 0 = the unbounded judge loop): K judge passes,
            then every uncommitted slot is re-masked and the STOCK loop finishes the block.  Cell `_jp{K}`.
  --adapter DIR   load_adapter(DIR) + merge_lora before the MoE is instrumented; the adapter's
            directory and its adapter_config.json `config_hash` go into the cell name and every record.
            An adapter trained with `train.py --draft-flag` (metadata `draft_embed: true`) is loaded
            UN-merged instead -- its LoRA delta is gated to the tentative draft slots, which a merged
            weight could not express -- and every forward then runs with the flag armed: the block's
            open non-[MASK] slots for a `--judge full` pass, all-False (= the stock model) everywhere
            else.  No extra flag: the adapter's own metadata decides.
  --pool M --pool-rule {full,top8,des}   per-forward top-M expert pool on every BLOCK forward
            (judge / denoise; NEVER the prefill, and the flush only with --pool-flush).

Usage (one process per GPU; items are sharded items[shard::nshards]; set CUDA_VISIBLE_DEVICES outside):
  CUDA_VISIBLE_DEVICES=4 /venv/main/bin/python code/draftalign/run_da_decode.py \
      --model b32 --drafter d32 --items calibvalid --draft s8 --shard 0 --nshards 4 --gpu 4 --score
  # B2, faithful DES-Vote M=32 on the b32 model (no drafter, no judge):
  CUDA_VISIBLE_DEVICES=4 ... run_da_decode.py --stock --items calibvalid --pool 32 --pool-rule des \
      --shard 0 --nshards 4 --gpu 4 --score
  # the trained full judge, with and without the pool:
  CUDA_VISIBLE_DEVICES=4 ... run_da_decode.py --adapter <dir> --judge full --draft s8 [--pool 32] ...
Outputs: <out>/<cell>.<items>.shard{k}of{n}.jsonl (+ .done), <out>/traces/<items>/<idx>.npz
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
import research_protocol as P                                   # noqa: E402  (HF offline env + modules cache)
from research_protocol import (MASK_ID, EOS_IDS, GEN_LEN, MODELS, load_sdar, config_hash, safe_out,  # noqa: E402
                               items_for_split, encode_prompt, LaunchLog)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from draftalign.seqcache import SeqCache                        # noqa: E402
from draftalign import wandb_log                                # noqa: E402
from draftalign.drafter import SETTINGS                         # noqa: E402
from draftalign.da_decode import (MoEMeter, decode_item, save_trace, POOL, POOL_RULES,   # noqa: E402
                                  FULL_MAX_PASSES, enable_pool_moe)

BLOCK = 32
OUT_ROOT = os.path.join("logs", "draftalign", "da_decode")
REF_DIR = os.path.join("logs", "research_v1", "baselines_b32")


def pct(x: float) -> int:
    return int(round(100 * x))


def pool_suffix(args) -> str:
    """`_p<M><rule>` when a pool is active, `+f` when the flush is restricted too (default: it is not)."""
    M = getattr(args, "pool", 0) or 0
    if not M:
        return ""
    cv = float(getattr(args, "tau_pool_cov", 0.0) or 0.0)
    return (f"_p{int(M)}{getattr(args, 'pool_rule', 'full')}" + ("f" if getattr(args, "pool_flush", False) else "")
            + (f"cv{pct(cv)}" if cv > 0 else ""))


def adapter_weights_hash(path: str) -> str:
    """sha256 of the adapter WEIGHTS (adapter.safetensors), 16 hex chars.  The metadata config_hash is the same
    for every checkpoint of one recipe (T2a step 1,057 and T2b step 2,600 both hash to b3106a4df420f375), so
    naming cells by it made different checkpoints collide; the weights hash is unique per checkpoint."""
    import hashlib
    fp = os.path.join(path, "adapter.safetensors")
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def adapter_suffix(args, adapter_hash=None) -> str:
    """`_ad<hash8>` -- the adapter's weights hash (adapter_weights_hash), so a cell can never mix checkpoints."""
    if not getattr(args, "adapter", None):
        return ""
    h = adapter_hash or getattr(args, "_adapter_hash", None) or "unknown"
    return f"_ad{str(h)[:8]}"


def drafter_info(args) -> dict:
    """MODELS key -> repo/revision; otherwise a fine-tuned checkpoint directory (train_drafter.py output, `latest` honoured).
    `tag` is appended to the cell name so records of different drafters never mix."""
    if args.stock:
        return {"key": None, "repo": None, "revision": None, "tag": "", "path": None}
    if args.drafter in MODELS:
        return {"key": args.drafter, "repo": MODELS[args.drafter]["repo"], "revision": MODELS[args.drafter]["revision"],
                "tag": "" if args.drafter == "d32" else f"_dr{args.drafter}", "path": None}
    from phase2 import drafters as P2D                           # Phase 2: a released drafter from the registry
    if P2D.is_key(args.drafter):
        m = P2D.spec(args.drafter)
        return {"key": args.drafter, "repo": m["repo"], "revision": m["revision"], "tag": P2D.cell_tag(args.drafter),
                "path": None}
    import hashlib
    from draftalign.train_drafter import resolve_checkpoint
    path = resolve_checkpoint(safe_out(args.drafter))
    assert os.path.isdir(path), f"--drafter: not a MODELS key nor a checkpoint directory: {args.drafter}"
    step = None
    tp = os.path.join(path, "trainer_state.pt")
    if os.path.isfile(tp):
        step = int(torch.load(tp, map_location="cpu", weights_only=False).get("step", -1))
    h = hashlib.sha256(path.encode()).hexdigest()[:8]
    return {"key": args.drafter, "repo": path, "revision": f"ckpt:{path}:step{step}", "tag": f"_dr{h}", "path": path, "step": step}


def cell_name(ds: str, args, adapter_hash=None) -> str:
    t = pct(args.threshold)
    tail = pool_suffix(args) + adapter_suffix(args, adapter_hash)
    if args.stock:
        return f"{ds}_B{BLOCK}_t{t}_stock" + tail
    judge = getattr(args, "judge", "halves")
    if judge == "blind":
        s = f"{ds}_B{BLOCK}_t{t}_dv-{args.draft}_blind_q{pct(args.tau_acc_q)}"
        if getattr(args, "tau_draft", 0.0) > 0:
            s += f"_d{pct(args.tau_draft)}"
        if args.rejudge:
            s += "_rj"
        if int(getattr(args, "draft_after_pass", 0) or 0):
            s += f"_da{int(args.draft_after_pass)}"          # anchored drafting: draft after N stock passes
        return s + drafter_info(args)["tag"] + tail
    if judge == "full":
        s = f"{ds}_B{BLOCK}_t{t}_dv-{args.draft}_full_c{pct(args.tau_commit)}"
        if getattr(args, "tau_override", None) is not None and float(args.tau_override) != float(args.tau_commit):
            s += f"_o{pct(args.tau_override)}"
        if getattr(args, "full_remask_only", False):
            s += "_ro"
        if int(getattr(args, "full_max_passes", FULL_MAX_PASSES)) != FULL_MAX_PASSES:
            s += f"_mp{int(args.full_max_passes)}"
        if int(getattr(args, "judge_passes", 0) or 0):
            s += f"_jp{int(args.judge_passes)}"              # one-pass hybrid: K judge passes, then STOCK
        return s + drafter_info(args)["tag"] + tail
    s = f"{ds}_B{BLOCK}_t{t}_dv-{args.draft}_c{pct(args.tau_commit)}_a{pct(args.tau_acc)}_q{pct(args.tau_acc_q)}"
    if args.rejudge:
        s += f"_rj{args.rejudge_k}"
    return s + drafter_info(args)["tag"] + tail


def load_ref_records(ref_dir: str, ref_cell: str, split: str) -> dict:
    recs = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, f"{ref_cell}.{split}.shard*of*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs[int(r["idx"])] = r
    return recs


def ref_compare(rec: dict, ref: dict) -> dict:
    return {"gen_ids": rec["gen_ids"] == ref["gen_ids"],
            "passes_per_block": rec["passes_per_block"] == ref["passes_per_block"],
            "commits_per_pass": rec["commits_per_pass"] == ref["commits_per_pass"],
            "n_forward": rec["n_forward"] == ref["n_forward"],
            "prompt_tokens": rec["prompt_tokens"] == ref["prompt_tokens"]}


def adapter_dir(path: str) -> str:
    return os.path.abspath(path if os.path.isabs(path) else safe_out(path))


def read_adapter_meta(path: str) -> dict:
    """`adapter_config.json` of an adapter dir (read before the model loads, so the cell name can carry
    its `config_hash`; `load_adapter_into` re-reads it and the two are asserted equal)."""
    from draftalign.adapter import ADAPTER_JSON
    with open(os.path.join(adapter_dir(path), ADAPTER_JSON)) as fh:
        return json.load(fh)


def load_adapter_into(model, path: str) -> dict:
    """`load_adapter` then `merge_lora`: the inference model is a STOCK module tree again.

    merge_lora turns every `GateFP32` back into a plain bf16 `nn.Linear` whose weight is the fp32
    master cast to bf16 (adapter.py:106-111).  That is exactly the arithmetic training saw --
    `GateFP32.forward` is `F.linear(x, weight.to(bf16))` (adapter.py:103-104), i.e. the cast happens
    before the matmul either way -- so inference numerics are bit-identical to training, and the gate
    stays an `nn.Linear` that `moe_instr` / `pool_moe_forward` (`self.gate(hidden_states)`) and
    `moe_probe.RouterCapture` (a hook on `blk.gate`) keep working with.

    A FLAG-GATED adapter (metadata `draft_embed: true`) is loaded UN-MERGED: its delta exists only
    at flagged draft slots, and a merged weight would apply it everywhere (`merge_lora` refuses).
    `LoRALinear` / `GateFP32` stay in the tree; both are plain `nn.Module`s that the MoE patches and
    the router hooks keep working with, and at an all-False flag the projections are bit-identical
    to the frozen base, so every unflagged forward is the stock model.
    """
    from draftalign.adapter import (load_adapter, merge_lora, adapter_metadata, is_flag_gated,
                                    get_draft_embed, can_apply_draft_flag)
    info = load_adapter(model, path)
    meta = info["meta"]
    flagged = bool(is_flag_gated(model))
    if flagged:
        assert get_draft_embed(model) is not None, "a flag-gated adapter without its draft_embed vector"
        assert can_apply_draft_flag(model), \
            ("this adapter was trained with --draft-flag but the model can take neither inputs_embeds "
             "nor an embed_tokens hook: --judge full would silently run without the flag")
        merged = {"lora": 0, "gates": 0}
    else:
        merged = merge_lora(model)
    return {"path": os.path.abspath(path), "hash": adapter_weights_hash(path), "config_hash": meta.get("config_hash"), "meta": meta,
            "loaded": info["loaded"], "merged": merged, "draft_flag": flagged}


def load_models(args):
    from sdar_harness.moe_instr import enable_instrumented_moe, SINK
    tok, moe = load_sdar(args.model, args.device)
    ad = None
    if args.adapter:
        ad = load_adapter_into(moe, adapter_dir(args.adapter))
        print(f"[adapter] {ad['path']} hash {ad['hash']} ({ad['loaded']} tensors, "
              + ("FLAG-GATED: kept un-merged, the draft flag is applied per forward"
                 if ad["draft_flag"] else
                 f"merged {ad['merged']['lora']} lora + {ad['merged']['gates']} gates)"), flush=True)
        if ad["draft_flag"] and args.judge != "full":
            print(f"[adapter] warning: this adapter was trained with --draft-flag but --judge {args.judge} "
                  f"flags no slot, so it runs as the stock model", flush=True)
    n = enable_pool_moe(moe) if args.pool else enable_instrumented_moe(moe)
    assert n == moe.config.num_hidden_layers
    SINK.disarm(); SINK.verify = False
    drafter, tok_d = None, None
    if not args.stock:
        from phase2 import drafters as P2D
        if args.drafter in MODELS:
            tok_d, drafter = load_sdar(args.drafter, args.device)
        elif P2D.is_key(args.drafter):
            # a released drafter behind the Phase 2 wrapper: it reads the MoE's own token ids (vocabulary map and
            # mask remap inside), so there is no drafter tokenizer to compare against
            drafter = P2D.load(args.drafter, args.device, moe_tok=tok)
            print(f"[drafter] phase2 {args.drafter}: {drafter.describe()}", flush=True)
            return tok, moe, None, drafter, SINK, ad
        else:
            from draftalign.eval_drafter_agreement import load_drafter
            tok_d, drafter, dinfo = load_drafter(args.drafter, args.device)
            print(f"[drafter] fine-tuned checkpoint {dinfo['spec']} (step {dinfo.get('step')})", flush=True)
        assert tok_d(tok_d.mask_token)["input_ids"][0] == MASK_ID
    return tok, moe, tok_d, drafter, SINK, ad


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="b32", choices=sorted(MODELS))
    ap.add_argument("--drafter", default="d32", help="MODELS key (d32 = stock 1.7B) or a fine-tuned drafter checkpoint dir")
    ap.add_argument("--items", default="calibvalid", help="calib | valid | calibvalid | gsm8k500 | mbpp")
    ap.add_argument("--draft", default="s8", choices=sorted(SETTINGS))
    ap.add_argument("--threshold", type=float, default=0.95, help="stock commit threshold (strict >, bf16)")
    ap.add_argument("--tau-commit", type=float, default=0.95)
    ap.add_argument("--tau-override", type=float, default=None,
                    help="full judge only: stricter threshold for commits that OVERRIDE the draft token (default = --tau-commit)")
    ap.add_argument("--tau-acc", type=float, default=0.70)
    ap.add_argument("--tau-acc-q", type=float, default=None, help="halves default .50, blind default .30; 1.0 disables the q rule")
    ap.add_argument("--judge", default="halves", choices=["halves", "blind", "full"],
                    help="halves: 2 judge forwards per block; blind: q read from the first stock pass (0 extra "
                         "forwards); full: draft-filled judge passes until every slot is committed (trained regime)")
    ap.add_argument("--full-max-passes", type=int, default=FULL_MAX_PASSES,
                    help="judge full: cap on judge passes per block; the rest falls back to the STOCK rule")
    ap.add_argument("--full-remask-only", action="store_true",
                    help="judge full ablation: never keep a tentative draft token (every uncommitted slot -> [MASK])")
    ap.add_argument("--draft-after-pass", type=int, default=0,
                    help="judge blind: ANCHORED DRAFTING -- run N stock passes first, then draft only the still-open "
                         "slots with those commits fixed as anchors (0 = today's rule: draft before pass 1)")
    ap.add_argument("--judge-passes", type=int, default=0,
                    help="judge full: ONE-PASS HYBRID -- K judge passes, then every uncommitted slot is re-masked and "
                         "the STOCK loop finishes the block (0 = today's unbounded judge loop)")
    ap.add_argument("--adapter", default=None, help="adapter dir (load_adapter + merge_lora before instrumenting)")
    ap.add_argument("--pool", type=int, default=0,
                    help="per-forward expert pool size M (0 = off); restricts every BLOCK forward (judge/denoise), never the prefill")
    ap.add_argument("--pool-rule", default="full", choices=sorted(POOL_RULES),
                    help="full: sum of the fp32 softmax (dMoE); top8: sum of the RENORMALISED top-8 weights; "
                         "des: sum of the UNNORMALISED top-8 weights (faithful DES-Vote, desvote.py:96-99)")
    ap.add_argument("--pool-flush", action="store_true", help="also restrict the kv-flush forward (default: off)")
    ap.add_argument("--tau-pool-cov", type=float, default=0.0,
                    help="full judge + pool: also require the slot's ORIGINAL top-k router mass surviving "
                         "the pool to exceed this, so commits are held back on rows whose routing the pool "
                         "pruned (0 = off, the unchanged rule)")
    ap.add_argument("--tau-draft", type=float, default=0.0, help="blind: also require the drafter's confidence > tau_draft (0 = off)")
    ap.add_argument("--rejudge", action="store_true", help="halves: redraft+rejudge open slots after passes with >= k open; blind: re-read q every later pass")
    ap.add_argument("--rejudge-k", type=int, default=8)
    ap.add_argument("--rejudge-redraft", action="store_true", help="blind rejudge with a redraft (NOT implemented; raises)")
    ap.add_argument("--stock", action="store_true", help="no drafter, no judge: the stock decode (integrity twin)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU label for the launch log (set CUDA_VISIBLE_DEVICES outside)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help=f"default {OUT_ROOT}/<cell>")
    ap.add_argument("--ref-dir", default=REF_DIR, help="stock records for the per-item integrity comparison")
    ap.add_argument("--ref-cell", default=None, help="default <ds>_B32_t<threshold>_stock")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--score", action="store_true", help="score each item inline (CPU, vendored evaluator)")
    # Phase 2 (code/phase2/decode_hooks.py); all off by default, none of them enters the config hash
    ap.add_argument("--expert-log", choices=("off", "bits", "full"), default="off",
                    help="per forward, per layer: the experts that ran (bits) or every row's top-8 ids + probs (full)")
    ap.add_argument("--stop-rules", default=None,
                    help="check code/phase2/rules.json (or this path) after every item; 'default' = the frozen file")
    ap.add_argument("--p2-s0-dir", default=None, help="S0 (SDAR-1.7B) reference cell dir: paired delta + refs")
    ap.add_argument("--p2-stock-dir", default=None, help="stock reference cell dir: speedup vs stock")
    ap.add_argument("--p2-integrity", action="store_true",
                    help="warn when records differ from --ref-dir/--ref-cell (the S0 rerun against round 3)")
    return ap


def resolve_args(args):
    if args.tau_acc_q is None:
        args.tau_acc_q = 0.30 if args.judge == "blind" else 0.50
    if args.judge == "blind" and args.rejudge and args.rejudge_redraft:
        raise NotImplementedError("--rejudge-redraft: blind rejudge with redraft is not implemented (no-redraft variant only)")
    if args.judge == "full" and args.rejudge and not args.stock:
        raise NotImplementedError("--rejudge is halves/blind only; the full judge re-judges every pass")
    assert int(getattr(args, "full_max_passes", FULL_MAX_PASSES)) >= 1, "--full-max-passes must be >= 1"
    da = int(getattr(args, "draft_after_pass", 0) or 0)
    jp = int(getattr(args, "judge_passes", 0) or 0)
    assert da >= 0, "--draft-after-pass must be >= 0"
    assert jp >= 0, "--judge-passes must be >= 0"
    assert not (da and args.stock) and not (jp and args.stock), \
        "--draft-after-pass / --judge-passes are decode-rule variants; they do not apply to --stock"
    if da and args.judge != "blind":
        raise NotImplementedError("--draft-after-pass (anchored drafting) is the blind rule's variant: use --judge blind")
    if jp and args.judge != "full":
        raise NotImplementedError("--judge-passes (the one-pass hybrid) is the full judge's variant: use --judge full")
    M = int(getattr(args, "pool", 0) or 0)
    assert M == 0 or M >= 8, f"--pool {M} is below the model's top-8"
    return args


def main(argv=None):
    args = resolve_args(build_parser().parse_args(argv))

    base = P.split_base(args.items)
    ds = ("mbpp" if base.startswith("mbpp") else
          "mbpp" if base.startswith("opencode") else          # opencode_pool: gold = unit-test asserts, MBPP sandbox
          "ifeval" if base.startswith("if") else              # if_pool AND ifeval541 (corpus-1b): draftalign/if_scorer
          "math500" if base.startswith("math") else
          "aime" if base.startswith("aime") else
          "gpqa" if base.startswith("gpqa") else
          "humaneval" if base.startswith("humaneval") else
          "finqa" if base.startswith("finqa") else            # corpus-1b held-out eval (2026-09-28)
          "chartqa" if base.startswith("chartqa") else
          "chat" if base.startswith("chat") else
          "gsm8k")
    gen_length = GEN_LEN[base]
    ad_meta = read_adapter_meta(args.adapter) if args.adapter else None
    ad_hash = adapter_weights_hash(adapter_dir(args.adapter)) if args.adapter else None
    cell = cell_name(ds, args, ad_hash)
    out = safe_out(args.out or os.path.join(OUT_ROOT, cell))
    os.makedirs(out, exist_ok=True)
    proto, proto_sha = P.load_protocol()
    # --p2-integrity (Phase 2's S0 rerun) compares against the SAME cell recorded in round 3
    ref_cell = args.ref_cell or (cell if args.p2_integrity else f"{ds}_B{BLOCK}_t{pct(args.threshold)}_stock")
    ref_recs = load_ref_records(safe_out(args.ref_dir), ref_cell, args.items) if os.path.isdir(safe_out(args.ref_dir)) else {}

    items = items_for_split(args.items)
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.nshards]
    scorer = None
    if args.score:
        from analyze_s1b import per_item_correct
        scorer = lambda r: per_item_correct(ds, [r])[r["idx"]]

    cfg = {"cell": cell, "dataset": ds, "split": args.items, "block": BLOCK, "steps": BLOCK, "threshold": args.threshold,
           "gen_length": gen_length, "numerics": "stock", "model_key": args.model, "model_revision": MODELS[args.model]["revision"],
           "sampling": proto["sampling"], "protocol_sha256": proto_sha, "decoder": "da_decode_v1",
           "draft": None if args.stock else args.draft,
           "drafter_key": drafter_info(args)["key"],
           "drafter_revision": drafter_info(args)["revision"],
           "tau_commit": None if args.stock else args.tau_commit, "tau_acc": None if args.stock else args.tau_acc,
           "tau_acc_q": None if args.stock else args.tau_acc_q,
           "judge": None if args.stock else args.judge,
           "tau_draft": None if (args.stock or args.judge != "blind") else args.tau_draft,
           "rejudge": bool(args.rejudge and not args.stock),
           "rejudge_k": args.rejudge_k if (args.rejudge and not args.stock and args.judge == "halves") else None,
           "rejudge_redraft": bool(args.rejudge_redraft and not args.stock),
           "full_max_passes": args.full_max_passes if (args.judge == "full" and not args.stock) else None,
           "full_remask_only": bool(args.full_remask_only and args.judge == "full" and not args.stock),
           # only present when it changes the decode; keeps the config hash of every earlier cell unchanged
           **({"tau_override": args.tau_override} if (not args.stock and args.judge == "full" and args.tau_override is not None) else {}),
           **({"draft_after_pass": int(args.draft_after_pass)} if (not args.stock and args.judge == "blind" and args.draft_after_pass) else {}),
           **({"judge_passes": int(args.judge_passes)} if (not args.stock and args.judge == "full" and args.judge_passes) else {}),
           "pool_M": int(args.pool) or None, "pool_rule": args.pool_rule if args.pool else None,
           **({"tau_pool_cov": float(args.tau_pool_cov)} if args.pool and args.tau_pool_cov else {}),
           "pool_flush": bool(args.pool_flush and args.pool),
           "adapter": adapter_dir(args.adapter) if args.adapter else None, "adapter_hash": ad_hash}
    chash = config_hash(cfg)
    MODEL_REV = MODELS[args.model]["revision"]
    dest = os.path.join(out, f"{cell}.{args.items}.shard{args.shard}of{args.nshards}.jsonl")
    if os.path.exists(dest + ".done"):
        print(f"[skip] {dest} already done", flush=True); return
    done = set()
    if os.path.exists(dest):
        for line in open(dest):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                                   # truncated last line is redone
            assert r["config_hash"] == chash and r["model_revision"] == MODEL_REV, \
                f"{dest} holds records from another config; refusing to mix"
            assert r["idx"] not in done, f"duplicate idx {r['idx']} in {dest}"
            done.add(r["idx"])
    todo = [it for it in items if it["idx"] not in done]
    wb = wandb_log.init(f"{cell}.{args.items}.sh{args.shard}of{args.nshards}", cfg, out, group=cell, job_type="decode",
                        tags=("decode", ds, "stock" if args.stock else f"judge-{args.judge}"), rank=0,
                        notes=f"{len(todo)} items to decode in this shard ({len(done)} already on disk)")
    _wb_n = _wb_correct = _wb_scored = 0
    print(f"[plan] cell {cell} split {args.items} shard {args.shard}/{args.nshards}: {len(items)} items, "
          f"{len(done)} done, {len(todo)} to do; {len(ref_recs)} reference records ({ref_cell})", flush=True)

    torch.manual_seed(0)
    tok, moe_model, tok_d, drafter, sink, ad = load_models(args)
    assert (ad or {}).get("hash") == ad_hash, f"adapter hash changed under us: {(ad or {}).get('hash')} != {ad_hash}"
    import transformers
    gpu_name = torch.cuda.get_device_name(moe_model.device) if moe_model.device.type == "cuda" else str(moe_model.device)
    # NOT part of `cfg` (and so not of config_hash): the adapter's own config_hash already separates
    # cells, and adding a key here would invalidate every existing cell's records.
    run_meta = {"adapter_draft_flag": bool((ad or {}).get("draft_flag")),
                "torch": torch.__version__, "transformers": transformers.__version__, "gpu_name": gpu_name,
                "config_hash": chash, "shard": args.shard, "nshards": args.nshards, "gpu": args.gpu,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    print(f"[model] {MODELS[args.model]['repo']} rev {MODEL_REV}" +
          ("" if args.stock else f" + drafter {drafter_info(args)['repo']} rev {drafter_info(args)['revision']} ({args.draft}, judge {args.judge})")
          + (f" + adapter {ad['path']} ({ad_hash})" if ad else "")
          + (f" + pool M={args.pool} rule={args.pool_rule}{' (flush too)' if args.pool_flush else ''}" if args.pool else "")
          + f" on {gpu_name}", flush=True)
    launch = LaunchLog(f"draftalign/da_decode/{cell}.{args.items}/shard{args.shard}of{args.nshards}", {**cfg, **run_meta})

    POOL.configure(int(args.pool) or None, args.pool_rule)
    POOL.track_cov = bool(args.pool) and float(args.tau_pool_cov) > 0.0
    hooks = None
    if args.expert_log != "off" or args.stop_rules is not None:
        from phase2.decode_hooks import DecodeHooks
        from phase2 import drafters as P2D
        dkey = None if args.stock else args.drafter
        hooks = DecodeHooks(args, out, dest, args.items, is_stock=bool(args.stock), drafter_key=dkey,
                            drafter_eq=0.0 if args.stock else P2D.cost_per_forward(dkey),
                            s0_dir=safe_out(args.p2_s0_dir) if args.p2_s0_dir else None,
                            stock_dir=safe_out(args.p2_stock_dir) if args.p2_stock_dir else None,
                            expert_level=args.expert_log, rules_path=args.stop_rules,
                            integrity=bool(args.p2_integrity), cell=cell,
                            stock_cell=f"{ds}_B{BLOCK}_t{pct(args.threshold)}_stock")
    meter_cls = hooks.meter_class(MoEMeter) if hooks else MoEMeter
    moe = meter_cls(SeqCache(moe_model, BLOCK, "moe"), sink, pool=POOL if args.pool else None,
                    pool_flush=bool(args.pool_flush), draft_flag=bool((ad or {}).get("draft_flag")),
                    **(hooks.meter_kwargs() if hooks else {}))
    if hooks:
        hooks.attach(moe)                                           # the judge's per-pass probabilities
    dr_seq = (None if args.stock else
              drafter.as_seq(BLOCK) if hasattr(drafter, "as_seq") else SeqCache(drafter, BLOCK, "drafter"))
    gen_kw = dict(block_length=BLOCK, denoising_steps=BLOCK, threshold=args.threshold,
                  draft=None if args.stock else args.draft, tau_commit=args.tau_commit, tau_acc=args.tau_acc,
                  tau_acc_q=args.tau_acc_q, rejudge=bool(args.rejudge and not args.stock), rejudge_k=args.rejudge_k,
                  judge=args.judge, tau_draft=args.tau_draft, rejudge_redraft=bool(args.rejudge_redraft),
                  full_max_passes=args.full_max_passes, full_remask_only=bool(args.full_remask_only),
                  tau_override=args.tau_override, tau_pool_cov=float(args.tau_pool_cov),
                  draft_after_pass=0 if args.stock else int(args.draft_after_pass),
                  judge_passes=0 if args.stock else int(args.judge_passes))
    t_cell = time.time()
    n_new, n_ref, n_ref_ok = 0, 0, 0
    with open(dest, "a") as fh:
        for k, it in enumerate(todo):
            prompt_ids = encode_prompt(tok, it["prompt"])
            if tok_d is not None:
                assert torch.equal(prompt_ids, encode_prompt(tok_d, it["prompt"])), "drafter tokenizer differs"
            rec, res = decode_item(moe, dr_seq, tok, it, prompt_ids, mask_id=MASK_ID, eos_ids=EOS_IDS,
                                   gen_length=gen_length, adapter=(ad["path"] if ad else None),
                                   adapter_hash=ad_hash, **gen_kw)
            rec.update({"cell": cell, "config_hash": chash, "model_revision": MODEL_REV, "model_key": args.model,
                        "numerics": "stock", "gpu": gpu_name, "pid": os.getpid(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            ref = ref_recs.get(int(it["idx"]))
            if ref is not None:
                rec["ref_cell"] = ref_cell
                rec["ref_match"] = ref_compare(rec, ref)
                rec["ref_correct"] = ref.get("correct")
                if args.stock:
                    n_ref += 1; n_ref_ok += int(rec["ref_match"]["gen_ids"])
            if scorer is not None:
                try:
                    rec["correct"] = int(scorer(rec))
                except Exception as e:                      # pragma: no cover
                    rec["correct"] = None; rec["score_error"] = repr(e)
            if not args.no_trace:
                save_trace(os.path.join(out, "traces", args.items, f"{it['idx']:05d}.npz"), res, BLOCK)
            p2_stop = hooks.after_item(rec, moe) if hooks else False
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n_new += 1
            el = time.time() - t_cell
            acc = sum(rec["accepted_per_block"]); cj = sum(rec["committed_judge_per_block"]); nv = sum(rec["n_valid_per_block"])
            if args.judge == "blind":
                cj = sum(rec["blind_accept_per_block"])          # printed in the cj slot: total blind accepts
            if args.judge == "full":
                cj = sum(rec["committed_judge_per_block"])       # printed in the cj slot: judge commits
            agree = [a for a in rec["draft_agreement_per_block"] if a is not None]
            print(f"[{cell}/sh{args.shard}] item {it['idx']} ({k + 1}/{len(todo)}) {rec['seconds']:.1f}s "
                  f"moe {rec['n_forward']} (judge {rec['n_judge']} den {rec['n_denoise']}) drafter {sum(rec['drafter_n_by_kind'].values())} "
                  f"blocks {rec['n_blocks']} pass/blk {np.mean(rec['passes_per_block']):.2f} gen {rec['n_gen_to_eos']} "
                  f"acc {acc}/{nv} cj {cj} agree {np.mean(agree) if agree else float('nan'):.3f} "
                  f"uniq/lf {rec['uniq_per_layer_forward_all'] or 0:.1f} "
                  + (f"tent {sum(rec['tentative_per_block'])} remask {sum(rec['remask_per_block'])} " if args.judge == "full" else "")
                  + f"eos {rec['stopped_eos']} correct {rec.get('correct')}"
                  + (f" ref_gen_ids {rec['ref_match']['gen_ids']}" if ref is not None else "")
                  + f" | {el / n_new:.0f}s/item", flush=True)
            _wb_n += 1
            if rec.get("correct") is not None:
                _wb_scored += 1; _wb_correct += int(rec["correct"])
            wb.log({"item/correct": rec.get("correct"), "item/seconds": rec["seconds"], "item/n_forward": rec["n_forward"],
                    "item/n_judge": rec["n_judge"], "item/passes_per_block": float(np.mean(rec["passes_per_block"])),
                    "item/n_blocks": rec["n_blocks"], "item/gen_tokens": rec["n_gen_to_eos"],
                    "item/ms_per_token": 1000.0 * rec["seconds"] / max(1, rec["n_gen_to_eos"]),
                    "item/uniq_experts_per_layer_forward": rec["uniq_per_layer_forward_all"] or 0.0,
                    "item/draft_agreement": float(np.mean(agree)) if agree else float("nan"),
                    "item/accepted_slots": acc, "item/judge_commits": cj,
                    "run/accuracy_so_far": (_wb_correct / _wb_scored) if _wb_scored else float("nan"),
                    "run/items_done": _wb_n, "run/sec_per_item": el / n_new,
                    **({k: v for k, v in rec.items() if k.startswith("exp_") and isinstance(v, (int, float))} if hooks else {}),
                    **(hooks.wandb_row() if hooks else {})}, step=_wb_n)
            del res
            if p2_stop:
                wb.finish()
                code = hooks.stopped(dest + ".stopped")
                print(f"SHARD STOPPED BY RULE {cell} shard {args.shard}: {n_new} new items; see {dest}.stopped", flush=True)
                launch.exit_code = code
                sys.exit(code)
    wb.log({"run/final_accuracy": (_wb_correct / _wb_scored) if _wb_scored else float("nan"), "run/items_done": _wb_n}, step=max(_wb_n, 1))
    wb.finish()
    if hooks:
        hooks.close()
    open(dest + ".done", "w").write("ok\n")
    msg = f"SHARD DONE {cell} shard {args.shard}: {n_new} new items in {(time.time() - t_cell) / 60:.1f} min"
    if args.stock:
        msg += f"; stock integrity: gen_ids match {n_ref_ok}/{n_ref} referenced items"
    print(msg, flush=True)
    launch.exit_code = 0


if __name__ == "__main__":
    main()
