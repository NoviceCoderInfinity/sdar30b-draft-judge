#!/usr/bin/env python
"""Drafter-informed DES-Vote runner. Specification: docs/DESVOTE_SPEC.md.

Arms (the decode is identical in all of them: from 32 [MASK], block 32,
threshold 0.95, greedy, bd3 confidence semantics, KV cache across blocks; only
which experts are reachable differs):

  none   unrestricted reference.
  des    DES-Vote as published, per forward and per layer, after the warm-up.
         Positions still [MASK] vote with their own (blind) weights.
  blend  ours. Warm-up = the stock decode until >= warm_frac of the block's
         valid slots are committed. Then the drafter fills ONLY the open slots
         with the committed tokens held fixed, one readout forward records the
         router weights those draft tokens produce, and from then on positions
         still [MASK] vote with their recorded weights while committed
         positions vote with their own. Draft tokens never enter the decode.

One process per GPU walks the cell list; every cell is sharded over the
workers (items[shard::nshards]). Item-level resume; a .done marker per shard
closes the cell. All paths live under this repository; nothing is written to
the collaborator tree.
"""
from __future__ import annotations
import json

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "compat"))   # flash_attn import shim

import numpy as np
import torch

MOE = "JetLM/SDAR-30B-A3B-Chat-b32"
DRAFTER = "JetLM/SDAR-1.7B-Chat"
MASK_ID = 151669
EOS_IDS = [151645, 151643]
BLOCK = 32

GSM8K_TMPL = ("{question}\n"
              "Please reason step by step, and put your final answer within \\boxed{{}}.")
MBPP_TMPL = ("You are an expert Python programmer, and here is your task:\n{text}\n"
             "Your code should pass these tests:\n\n{test_list}\n"
             " You should submit your final solution in the following format: "
             "```python\n\n```")
MATH_TMPL = GSM8K_TMPL
GPQA_TMPL = ("{question}\n\nPlease reason step by step, and put the letter of your final answer "
             "(A, B, C or D) within \\boxed{{}}.")
HUMANEVAL_TMPL = ("You are an expert Python programmer. Complete the following function.\n"
                  "```python\n{prompt}\n```\n"
                  "Return the complete function including its signature. You should submit your "
                  "final solution in the following format: ```python\n\n```")
GEN_LEN = {"gsm8k": 1024, "gsm8k500": 1024, "mbpp": 512, "math500": 1024, "aime": 1024, "gpqa": 1024, "humaneval": 512}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_items(dataset: str):
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")          # full 1319
        return [{"idx": i, "prompt": GSM8K_TMPL.format(question=r["question"]),
                 "gold": r["answer"]} for i, r in enumerate(ds)]
    if dataset == "gsm8k500":
        # the frozen 500-question subset the friend's unrestricted cell ran on;
        # idx is the index into the full test split, so items pair exactly
        keep = set(json.load(open(os.path.join(HERE, "configs", "gsm8k_test_500.json")))["split_indices"])
        return [it for it in build_items("gsm8k") if it["idx"] in keep]
    if dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        out = []
        for i, r in enumerate(ds):
            tl = "\n".join(r["test_list"])
            out.append({"idx": i, "prompt": MBPP_TMPL.format(text=r["prompt"], test_list=tl),
                        "gold": tl, "task_id": r["task_id"]})
        return out
    if dataset == "gpqa":
        ds = load_dataset("fingertap/GPQA-Diamond", split="test")   # ungated Diamond mirror, 198 MCQ items
        return [{"idx": i, "prompt": GPQA_TMPL.format(question=r["question"]),
                 "gold": str(r["answer"]).strip().upper()} for i, r in enumerate(ds)]
    if dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        return [{"idx": i, "prompt": HUMANEVAL_TMPL.format(prompt=r["prompt"]),
                 "gold": json.dumps({"test": r["test"], "entry_point": r["entry_point"]}),
                 "task_id": r["task_id"]} for i, r in enumerate(ds)]
    if dataset == "aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        return [{"idx": i, "prompt": MATH_TMPL.format(question=r["problem"]),
                 "gold": str(int(str(r["answer"]).strip()))} for i, r in enumerate(ds)]   # AIME golds are 0-padded to 3 digits; canonicalise
    if dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"idx": i, "prompt": MATH_TMPL.format(question=r["problem"]),
                 "gold": r["answer"]} for i, r in enumerate(ds)]
    raise ValueError(dataset)


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def cell(ds, arm, M=0, f=0.0, d=8, p=0):
    return {"ds": ds, "arm": arm, "M": M, "f": f, "d": d, "p": p}


def cell_name(c):
    s = f"{c['ds']}_{c['arm']}_M{c['M']}_f{int(round(100 * c['f']))}_d{c['d']}"
    return s + ("_probe" if c.get("p") else "")


def plan_trace():
    """Prefetch analysis: DES M=32 as published, with per-pass expert-set logs
    and the draft readout recorded (not used) for comparison."""
    return [cell("gsm8k500", "des", 32, 0.0, 8, p=1)]


def plan_first():
    """The first readable result: one setting of ours and its two baselines."""
    return [cell("gsm8k", "none"), cell("gsm8k", "des", 32), cell("gsm8k", "blend", 32, 0.25)]


def plan_first500():
    """User decision 2026-09-04: reuse the friend's unrestricted cell on the
    frozen 500-question subset (imported by import_friend_cell.py) and run only
    the two new arms on exactly those questions."""
    return [cell("gsm8k500", "des", 32), cell("gsm8k500", "blend", 32, 0.25)]


def plan_sweep():
    cells = []
    for ds in ("gsm8k", "mbpp"):
        cells += [cell(ds, "none"), cell(ds, "des", 32)]
        for f in (0.25, 0.30, 0.20, 0.15, 0.10, 0.0):
            cells.append(cell(ds, "blend", 32, f))
        cells.append(cell(ds, "des", 32, 0.25))          # warm-up alone, no draft
        for M in (24, 48, 64):
            cells += [cell(ds, "des", M), cell(ds, "blend", M, 0.25)]
    return cells


PLANS = {"first": plan_first, "first500": plan_first500, "sweep": plan_sweep, "trace": plan_trace}


# ---------------------------------------------------------------------------
# drafting the open slots (spec section 4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def draft_open_slots(drafter, x_ctx, lo, hi, passes=8):
    """Fill the [MASK] slots of block [lo, hi) with the drafter; every other
    token of x_ctx[:, :hi] (prompt, earlier blocks, committed slots) is fixed
    context. Greedy, bd3 confidence, no threshold, fixed schedule over the OPEN
    slots. Returns (block tokens [bl], drafter forwards)."""
    from transformers.cache_utils import DynamicCache
    from sdar_harness.decode import sample_with_confidence, get_num_transfer_tokens

    dev = x_ctx.device
    bl = hi - lo
    nb = hi // bl
    bm = torch.tril(torch.ones(nb, nb, device=dev))
    attn = bm.repeat_interleave(bl, 0).repeat_interleave(bl, 1).unsqueeze(0)
    pos = torch.arange(hi, device=dev).unsqueeze(0)
    cache = DynamicCache()
    n_fwd = 0
    if lo > 0:
        drafter(input_ids=x_ctx[:, :lo], attention_mask=attn[:, :lo, :lo],
                position_ids=pos[:, :lo], past_key_values=cache, use_cache=True, store_kv=True)
        n_fwd += 1
    cur = x_ctx[:, lo:hi].clone()
    assert not (x_ctx[:, :lo] == MASK_ID).any(), "context before the block holds a MASK"
    n_open = int((cur == MASK_ID).sum())
    if n_open == 0:
        return cur[0].clone(), n_fwd
    passes = min(passes, n_open)
    nt = get_num_transfer_tokens(n_open, passes)
    ninf = torch.tensor(float("-inf"), device=dev)
    for s in range(passes):
        mi = (cur == MASK_ID)
        if mi.sum() == 0:
            break
        lg = drafter(input_ids=cur, attention_mask=attn[:, lo:hi, :hi], position_ids=pos[:, lo:hi],
                     past_key_values=cache, use_cache=True, store_kv=False).logits
        n_fwd += 1
        x0, x0p = sample_with_confidence(lg, temperature=1.0, top_k=1, top_p=1.0)
        x0 = torch.where(mi, x0, cur)
        conf = torch.where(mi, x0p, ninf.to(x0p.dtype))
        k = min(int(nt[s]), int(mi.sum()))
        _, idx = torch.topk(conf[0], k)
        cur[0, idx] = x0[0, idx]
    return cur[0].clone(), n_fwd


# ---------------------------------------------------------------------------
# one item (spec section 3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(moe, drafter, input_ids, *, arm, M, warm_frac, draft_passes, gen_length,
           threshold=0.95, denoising_steps=32, trace=None, log_experts=False,
           probe_draft=False):
    """`trace`, if a list, receives (block_index, pass, committed_tokens[32] with -1
    where nothing was committed) for every denoising pass -- used by the gates.
    `log_experts`: return per-block expert-set logs (see _collect_log) for the
    prefetch analysis. `probe_draft`: under arm des, draft + readout at the end
    of warm-up and LOG the readout's expert set, without using it for routing."""
    from transformers.cache_utils import DynamicCache
    from sdar_harness.decode import sample_with_confidence, get_num_transfer_tokens
    from sdar_harness.desvote import CTRL

    assert arm in ("none", "des", "blend")
    assert input_ids.shape[0] == 1, "single-sequence decoding only (CTRL.masked is [N])"
    dev = moe.device
    input_ids = input_ids.to(dev)
    plen = input_ids.shape[1]
    bl = BLOCK
    cache = DynamicCache()
    num_blocks = (plen + gen_length + bl - 1) // bl
    total = num_blocks * bl
    bm = torch.tril(torch.ones(num_blocks, num_blocks, device=dev))
    attn = bm.repeat_interleave(bl, 0).repeat_interleave(bl, 1).unsqueeze(0)
    pos = torch.arange(total, device=dev).unsqueeze(0)
    x = torch.full((1, total), MASK_ID, dtype=torch.long, device=dev)
    x[:, :plen] = input_ids
    pfb = plen // bl

    CTRL.reset_cost()
    CTRL.M = None; CTRL.use_recorded = False; CTRL.recorded = None
    CTRL.valid = None; CTRL.masked = None; CTRL.recording = False
    CTRL.log = [] if log_experts else None
    n_moe = n_readout = n_draft = n_draft_mask = 0
    elog = {"prefill": None, "blocks": []} if log_experts else None

    def _collect_log():
        """CTRL.log -> (core [L,E] bool or None, hits [L,E] bool) for ONE forward."""
        entries = CTRL.log; CTRL.log = []
        L = CTRL.n_layers; E = CTRL.n_experts
        hits = np.zeros((L, E), bool); core = np.zeros((L, E), bool); has_core = False
        for l, c, h in entries:
            hits[l, h.astype(np.int64)] = True
            if c is not None:
                core[l] = c; has_core = True
        return (core if has_core else None), hits

    if pfb:                                            # prefill: never restricted
        moe(input_ids=x[:, :pfb * bl], attention_mask=attn[:, :pfb * bl, :pfb * bl],
            position_ids=pos[:, :pfb * bl], past_key_values=cache, use_cache=True, store_kv=True)
        n_moe += 1
        if log_experts:
            elog["prefill"] = _collect_log()[1]

    nt = get_num_transfer_tokens(bl, denoising_steps)
    steps_per_block, committed_per_step, warmup_end, nblocks = [], [], [], 0
    agree, agree_n = 0, 0
    ninf = torch.tensor(float("-inf"), device=dev)

    for nb in range(pfb, num_blocks):
        lo, hi = nb * bl, (nb + 1) * bl
        vt = (torch.arange(lo, hi, device=dev) >= plen)
        n_valid = int(vt.sum())
        ca, cp = attn[:, lo:hi, :hi], pos[:, lo:hi]
        nblocks += 1
        CTRL.valid = vt
        cur = x[:, lo:hi].clone()
        warm = True
        wend = -1
        dtok = None
        nstep = 0
        blog = {"valid": vt.cpu().numpy(), "pass_hits": [], "pass_core": [], "pass_masked": [],
                "flush_hits": None, "readout_hits": None, "readout_pass": -1} if log_experts else None
        for step in range(denoising_steps + 1):
            mi = (cur == MASK_ID) & vt.unsqueeze(0)
            n_open = int(mi.sum())
            if warm and (n_valid - n_open) >= warm_frac * n_valid:
                warm = False
                wend = step
                if (arm == "blend" or (arm == "des" and probe_draft)) and n_open > 0:
                    xx = x.clone(); xx[:, lo:hi] = cur
                    dtok, nf = draft_open_slots(drafter, xx[:, :hi], lo, hi, draft_passes)
                    n_draft += nf
                    mi_readout = mi[0].clone()          # slots the draft filled
                    blk = cur.clone(); blk[0, mi[0]] = dtok[mi[0]]
                    n_draft_mask += int((blk[0, mi[0]] == MASK_ID).sum())   # drafter emitted MASK (counted, not fixed)
                    before = cache.get_seq_length()
                    CTRL.recording = True
                    moe(input_ids=blk, attention_mask=ca, position_ids=cp,
                        past_key_values=cache, use_cache=True, store_kv=False)
                    rec = CTRL.take_recording()
                    CTRL.recorded = rec if arm == "blend" else None     # des+probe: logged only
                    n_readout += 1
                    assert cache.get_seq_length() == before, "readout touched the cache"
                    if log_experts:
                        blog["readout_hits"] = _collect_log()[1]; blog["readout_pass"] = step
            restricted = (arm in ("des", "blend")) and not warm
            CTRL.M = M if restricted else None
            CTRL.use_recorded = (arm == "blend") and restricted
            CTRL.masked = mi[0]

            if n_open == 0:                            # flush: restricted under des/blend
                moe(input_ids=cur, attention_mask=ca, position_ids=cp,
                    past_key_values=cache, use_cache=True, store_kv=True)
                n_moe += 1
                if log_experts:
                    blog["flush_hits"] = _collect_log()[1]
                break
            lg = moe(input_ids=cur, attention_mask=ca, position_ids=cp,
                     past_key_values=cache, use_cache=True, store_kv=False).logits
            n_moe += 1; nstep += 1
            if log_experts:
                c_, h_ = _collect_log()
                blog["pass_hits"].append(h_); blog["pass_core"].append(c_)
                blog["pass_masked"].append(mi[0].cpu().numpy())

            x0, x0p = sample_with_confidence(lg, temperature=1.0, top_k=1, top_p=1.0)
            x0 = torch.where(mi, x0, cur)
            conf = torch.where(mi, x0p, ninf.to(x0p.dtype))
            ti = torch.zeros_like(x0, dtype=torch.bool)
            hc = conf[0] > threshold
            need = int(nt[min(step, len(nt) - 1)])
            if int(hc.sum()) >= need:
                ti[0] = hc
            else:
                k = min(need, n_open)
                _, idx = torch.topk(conf[0], k)
                ti[0, idx] = True
            cur = torch.where(ti, x0, cur)
            committed_per_step.append(int(ti.sum()))
            if trace is not None:
                trace.append((nb - pfb, step, torch.where(ti[0], x0[0], torch.full_like(x0[0], -1)).cpu().numpy()))

        steps_per_block.append(nstep)
        warmup_end.append(wend)
        if log_experts:
            elog["blocks"].append(blog)
        if dtok is not None:
            # draft_agreement: over the slots the draft filled, does the draft
            # token equal the token the decode finally committed there?
            agree += int((dtok[mi_readout] == cur[0, mi_readout]).sum())
            agree_n += int(mi_readout.sum())
        x[:, lo:hi] = cur
        CTRL.M = None; CTRL.use_recorded = False; CTRL.recorded = None
        if any((x[:, plen:] == s).any() for s in EOS_IDS):
            break

    cost = CTRL.cost()
    CTRL.M = None; CTRL.valid = None; CTRL.masked = None; CTRL.log = None
    return {"ids": x[0, plen:].cpu().numpy(), "elog": elog,
            "n_moe": n_moe, "n_readout": n_readout,
            "n_draft": n_draft, "n_blocks": nblocks, "steps_per_block": steps_per_block,
            "committed_per_step": committed_per_step, "warmup_end_pass": warmup_end,
            "draft_agreement": (agree / agree_n) if agree_n else None,
            "draft_mask_tokens": n_draft_mask, **cost}


def save_elog(dirname, idx, elog):
    """One compressed npz per item. Arrays are padded to the max pass count."""
    os.makedirs(dirname, exist_ok=True)
    blocks = elog["blocks"]
    B = len(blocks)
    if B == 0:
        return
    S = max(len(b["pass_hits"]) for b in blocks)
    L, E = blocks[0]["flush_hits"].shape if blocks[0]["flush_hits"] is not None else blocks[0]["pass_hits"][0].shape
    hits = np.zeros((B, S, L, E), bool); core = np.zeros((B, S, L, E), bool); has_core = np.zeros((B, S), bool)
    masked = np.zeros((B, S, 32), bool); flush = np.zeros((B, L, E), bool)
    readout = np.zeros((B, L, E), bool); has_readout = np.zeros(B, bool); rpass = np.full(B, -1, np.int32)
    valid = np.zeros((B, 32), bool); npass = np.zeros(B, np.int32)
    for i, b in enumerate(blocks):
        n = len(b["pass_hits"]); npass[i] = n; valid[i] = b["valid"]
        for s in range(n):
            hits[i, s] = b["pass_hits"][s]; masked[i, s] = b["pass_masked"][s]
            if b["pass_core"][s] is not None:
                core[i, s] = b["pass_core"][s]; has_core[i, s] = True
        if b["flush_hits"] is not None:
            flush[i] = b["flush_hits"]
        if b["readout_hits"] is not None:
            readout[i] = b["readout_hits"]; has_readout[i] = True; rpass[i] = b["readout_pass"]
    np.savez_compressed(os.path.join(dirname, f"{idx:05d}.npz"),
                        pass_hits=hits, pass_core=core, has_core=has_core, pass_masked=masked,
                        flush_hits=flush, readout_hits=readout, has_readout=has_readout,
                        readout_pass=rpass, valid=valid, n_pass=npass,
                        prefill_hits=(elog["prefill"] if elog["prefill"] is not None else np.zeros((L, E), bool)),
                        has_prefill=np.bool_(elog["prefill"] is not None))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "desvote"))
    ap.add_argument("--plan", default="first", choices=sorted(PLANS))
    ap.add_argument("--only", default="", help="comma-separated cell-name substrings")
    ap.add_argument("--limit", type=int, default=0, help="first N items of each dataset")
    ap.add_argument("--log-experts", action="store_true",
                    help="write per-item expert-set traces to {out}_traces/{cell}/")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    assert out.startswith(ROOT + os.sep), f"refusing to write outside {ROOT}: {out}"
    assert not out.startswith("/workspace/stage0"), "never write into a colleague's tree"
    os.makedirs(out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sdar_harness.desvote import enable_vote_moe

    tok = AutoTokenizer.from_pretrained(MOE, trust_remote_code=True)
    assert tok(tok.mask_token)["input_ids"][0] == MASK_ID
    moe = AutoModelForCausalLM.from_pretrained(
        MOE, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0").eval()
    n = enable_vote_moe(moe)
    drafter = AutoModelForCausalLM.from_pretrained(
        DRAFTER, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0").eval()
    print(f"[model] vote-instrumented {n} MoE blocks; drafter loaded", flush=True)

    only = [s for s in args.only.split(",") if s]
    items_cache = {}
    for c in PLANS[args.plan]():
        name = cell_name(c)
        if only and not any(s in name for s in only):
            continue
        dest = os.path.join(out, f"{name}.shard{args.shard}.jsonl")
        if os.path.exists(dest + ".done"):
            continue
        if c["ds"] not in items_cache:
            items_cache[c["ds"]] = build_items(c["ds"])
        items = items_cache[c["ds"]]
        if args.limit:
            items = items[:args.limit]
        items = items[args.shard::args.nshards]
        done = set()
        if os.path.exists(dest):
            for l in open(dest):
                try:
                    done.add(json.loads(l)["idx"])
                except (ValueError, KeyError):
                    pass                                   # a truncated last line is simply redone
        t0 = time.time()
        with open(dest, "a") as fh:
            for k, it in enumerate(items):
                if it["idx"] in done:
                    continue
                s = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                            add_generation_prompt=True, tokenize=False)
                ids = tok([s], return_tensors="pt", add_special_tokens=False)["input_ids"]
                torch.cuda.synchronize(); t1 = time.time()
                r = decode(moe, drafter, ids, arm=c["arm"], M=c["M"], warm_frac=c["f"],
                           draft_passes=c["d"], gen_length=GEN_LEN[c["ds"]],
                           log_experts=args.log_experts, probe_draft=bool(c.get("p")))
                torch.cuda.synchronize(); dt = time.time() - t1
                if args.log_experts and r["elog"] is not None:
                    save_elog(os.path.join(out + "_traces", name), it["idx"], r["elog"])
                rec = {"idx": it["idx"], "cell": name,
                       "pred": tok.decode(r["ids"], skip_special_tokens=True),
                       "gold": it["gold"], "seconds": round(dt, 2)}
                if "task_id" in it:
                    rec["task_id"] = it["task_id"]
                rec.update({k2: v for k2, v in r.items() if k2 not in ("ids", "elog")})
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                if k % 20 == 0:
                    el = time.time() - t0
                    print(f"[{name}/sh{args.shard}] {k+1}/{len(items)} {dt:.1f}s "
                          f"({el/(k+1):.1f}s avg) eta {(len(items)-k-1)*el/(k+1)/60:.0f}m",
                          flush=True)
        open(dest + ".done", "w").write("ok\n")
        print(f"CELL DONE {name} sh{args.shard} {(time.time()-t0)/60:.1f} min", flush=True)
    print("ALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
