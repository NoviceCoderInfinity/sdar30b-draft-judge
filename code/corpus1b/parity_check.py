#!/usr/bin/env python
"""Stage-1 checks on the GPU box, BEFORE any wave is generated. Each subcommand prints a JSON report, appends it
to logs/c1b/stage1/report.jsonl, and exits 0 = PASS, 1 = FAIL.

  gen    SGLang (gen_sglang.py --items ...) vs the reference stock decoder (run_da_decode.py --stock) on the same
         items: accuracy within +-3 points and block-weighted passes/block within +-5% per split. Token agreement is
         reported (bit-identity is not expected: SGLang batches and uses its own kernels).
  draft  SGLang drafts (draft_sglang.py) vs draftalign/drafter.py::draft_block (the round-3 drafter, HF, one
         sequence at a time) on the same teacher-forced blocks: >= 95% token agreement.
  train  the two-stream trainer (draftalign/train_fast.py) vs the round-3 per-state path (train.forward_state) on
         the REAL 30B in bf16: loss within 1% and adapter-gradient cosine >= 0.99; also times both paths.
  speed  the SGLang throughput of a short wave-A sample (gen_sglang.py --limit) turned into projected hours for
         waves A and B with the plan's formula; FAIL if generation of both waves would exceed --max-gen-hours.

Code items are scored in a throwaway sandbox cwd (AGENTS.md 9(p)); every output path is absolute.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402
import research_protocol as P                                            # noqa: E402

REPORT = os.path.join(C.LOG_DIR, "stage1", "report.jsonl")


def emit(name: str, rep: dict, ok: bool) -> int:
    rep = {"check": name, "pass": bool(ok), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rep}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "a") as fh:
        fh.write(json.dumps(rep) + "\n")
    print(json.dumps(rep, indent=2), flush=True)
    return 0 if ok else 1


def family(base: str) -> str:
    """The scorer family, as the runners map it (run_da_decode.py ~l.333)."""
    if base.startswith(("mbpp", "opencode")):
        return "mbpp"
    if base.startswith("if"):
        return "ifeval"
    if base.startswith("math"):
        return "math500"
    return "gsm8k"


def score(fam: str, rows: List[dict]) -> Dict[int, int]:
    from analyze_s1b import per_item_correct
    cwd = os.getcwd()
    box = tempfile.mkdtemp(prefix="c1b_parity_sandbox_")
    try:
        os.chdir(box)                                          # generated programs run here, never in the repo
        return per_item_correct(fam, rows)
    finally:
        os.chdir(cwd)


def ppb_weighted(ppb_lists: List[List[int]]) -> float:
    n = sum(len(x) for x in ppb_lists)
    return sum(sum(x) for x in ppb_lists) / max(1, n)


# ---------------------------------------------------------------------------
def cmd_gen(a) -> int:
    """Reference records live in <ref-dir>/<split base>/ (setup_box.sh passes run_da_decode --out that dir)."""
    sgl = defaultdict(dict)
    for path in glob.glob(os.path.join(P.safe_out(a.sgl_dir), "gpu*", "records.*.jsonl.gz")):
        for r in C.read_jsonl(path):
            base = r["pid"].split("#")[0]
            sgl[base][int(r["idx"])] = r
    out, ok = {}, True
    for split in a.splits.split(","):
        base = P.split_base(split)
        refs = {}
        for path in glob.glob(os.path.join(P.safe_out(a.ref_dir), base, "**", "*.jsonl"), recursive=True):
            for r in C.read_jsonl(path):
                refs[int(r["idx"])] = r
        refs = {i: r for i, r in refs.items() if i in sgl[base]}
        pairs = [(refs[i], sgl[base][i]) for i in sorted(refs)]
        if not pairs:
            out[base] = {"error": "no paired items"}
            ok = False
            continue
        fam = family(base)
        rc = score(fam, [{"idx": r["idx"], "pred": r["pred"], "gold": r["gold"]} for r, _ in pairs])
        sc = score(fam, [{"idx": s["idx"], "pred": s["pred"], "gold": s["gold"]} for _, s in pairs])
        acc_r = 100.0 * np.mean([rc.get(r["idx"], 0) for r, _ in pairs])
        acc_s = 100.0 * np.mean([sc.get(s["idx"], 0) for _, s in pairs])
        ppb_r = ppb_weighted([r.get("stock_passes_per_block") or r.get("passes_per_block") or [] for r, _ in pairs])
        ppb_s = ppb_weighted([s.get("ppb") or [] for _, s in pairs])
        same = np.mean([list(r["gen_ids"]) == list(s["gen_ids"]) for r, s in pairs])
        agree = []
        for r, s in pairs:
            g1, g2 = list(r["gen_ids"]), list(s["gen_ids"])
            n = min(len(g1), len(g2))
            k = next((j for j in range(n) if g1[j] != g2[j]), n)
            agree.append(k / max(1, max(len(g1), len(g2))))
        rel_ppb = abs(ppb_s - ppb_r) / max(1e-9, ppb_r)
        good = abs(acc_s - acc_r) <= a.acc_tol and rel_ppb <= a.ppb_tol
        ok &= bool(good)
        out[base] = {"n": len(pairs), "acc_ref": round(acc_r, 2), "acc_sglang": round(acc_s, 2),
                     "ppb_ref": round(ppb_r, 3), "ppb_sglang": round(ppb_s, 3), "ppb_rel_diff": round(rel_ppb, 4),
                     "identical_answers": round(float(same), 3), "mean_prefix_agreement": round(float(np.mean(agree)), 3),
                     "pass": bool(good)}
    return emit("gen_parity", {"splits": out, "acc_tol_points": a.acc_tol, "ppb_tol": a.ppb_tol}, ok)


# ---------------------------------------------------------------------------
def cmd_draft(a) -> int:
    import torch
    from draftalign.drafter import draft_block
    from draftalign.seqcache import SeqCache
    gen = {}
    for path in glob.glob(os.path.join(P.safe_out(a.parity_gen), "gpu*", "records.*.jsonl.gz")):
        for r in C.read_jsonl(path):
            if r.get("kept") and r.get("prompt_ids"):
                gen[r["pid"]] = r
    drafts = {}
    for path in glob.glob(os.path.join(P.safe_out(a.parity_draft), "gpu*", "drafts.*.npz")):
        z = np.load(path, allow_pickle=True)
        off = 0
        for pid, nb in zip(z["pids"].tolist(), z["nb"].tolist()):
            drafts[str(pid)] = z["draft"][off:off + nb]
            off += nb
    _tok, model = P.load_sdar("d32", "cuda:0")
    seq = SeqCache(model, C.BLOCK, "d32")
    match = total = blocks = 0
    for pid in sorted(set(gen) & set(drafts))[:a.max_items]:
        r = gen[pid]
        seq.reset()
        ids = torch.tensor([r["prompt_ids"]], dtype=torch.long)
        seq.prefill(ids)
        L, lo0 = len(r["prompt_ids"]), int(r["lo0"])
        for b, blk in enumerate(r["blocks"]):
            lo = lo0 + C.BLOCK * b
            final = torch.tensor(blk["ids"], dtype=torch.long, device=seq.dev)
            valid = torch.ones(C.BLOCK, dtype=torch.bool, device=seq.dev)
            if b == 0:
                valid[:L - lo0] = False
            d = draft_block(seq, final, valid, lo, "s8", P.MASK_ID)
            ref = d["tokens"].cpu().numpy()
            got = drafts[pid][b]
            v = valid.cpu().numpy()
            match += int((ref[v] == got[v]).sum()); total += int(v.sum()); blocks += 1
            seq.flush(final, lo)                                   # teacher forcing: the 30B's final block
            if blocks >= a.max_blocks:
                break
        if blocks >= a.max_blocks:
            break
    agree = match / max(1, total)
    return emit("draft_parity", {"blocks": blocks, "slots": total, "token_agreement": round(agree, 4),
                                 "threshold": a.min_agree}, agree >= a.min_agree)


# ---------------------------------------------------------------------------
def cmd_train(a) -> int:
    import torch
    from draftalign import adapter as A
    from draftalign import train as T
    from draftalign import train_fast as F
    from draftalign.dataset import SamplerConfig, StateSampler
    dev = torch.device("cuda:0")
    _tok, model = P.load_sdar("b32", str(dev))
    A.attach_lora(model, r=32, alpha=64.0, train_gates=False, draft_flag=True)
    A.freeze_base(model)
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():                                        # a non-identity adapter
        for _, p in A.trainable_parameters(model):
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * 0.01)
    model.train()
    s = StateSampler(SamplerConfig(index_path=P.safe_out(a.index), seed=0, p_draftfull=0.3, max_seq=1856))
    states = [s.state(k) for k in range(min(a.n_states, len(s)))]
    params = [p for _, p in A.trainable_parameters(model)]

    def grads():
        return torch.cat([p.grad.detach().float().flatten() for p in params if p.grad is not None])

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.time()
    ref_loss = 0.0
    for st in states:
        loss, _, _, _ = T.forward_state(model, st, None, "full", True, 0.5, 0.5, P.MASK_ID, dev, draft_flag=True,
                                        calib=1.0, override_weight=2.0, code_override_weight=3.0)
        (loss / len(states)).backward()
        ref_loss += float(loss.detach()) / len(states)
    torch.cuda.synchronize(); t_ref = time.time() - t0
    g_ref = grads()
    model.zero_grad(set_to_none=True)
    cfg = F.LossCfg(beta=0.5, chat_beta=0.5, calib=1.0, override_weight=2.0, code_override_weight=3.0,
                    chat_override_weight=2.0)
    torch.cuda.synchronize(); t0 = time.time()
    tot, _, _ = F.micro_batch_loss(model, states, cfg, checkpoint_layers=True)
    (tot / len(states)).backward()
    torch.cuda.synchronize(); t_fast = time.time() - t0
    g_fast = grads()
    rel = abs(float(tot.detach()) / len(states) - ref_loss) / max(1e-9, abs(ref_loss))
    cos = float(torch.nn.functional.cosine_similarity(g_ref, g_fast, dim=0))
    ok = rel <= 0.01 and cos >= 0.99
    return emit("train_parity", {"states": len(states), "loss_round3": ref_loss, "loss_fast": float(tot) / len(states),
                                 "loss_rel_diff": rel, "grad_cosine": cos, "sec_per_state_round3": t_ref / len(states),
                                 "sec_per_state_fast": t_fast / len(states), "speedup": t_ref / max(1e-9, t_fast)}, ok)


# ---------------------------------------------------------------------------
def cmd_speed(a) -> int:
    """Mixed prompts per GPU-hour from a short run on the wave-A list in its natural mix
    (gen_sglang.py --limit-total N: the first N of the GPU's shard in hash order) -> projected generation hours.
    Domains interleave inside one engine, so only the MIXED rate is measured; it holds for a wave with the same mix."""
    tot_prompts, gpu_h, by_dom = 0, 0.0, defaultdict(int)
    for path in glob.glob(os.path.join(P.safe_out(a.gen_dir), "gpu*", "stats.jsonl")):
        last = None
        for line in open(path):
            last = json.loads(line)
        if not last:
            continue
        for d, v in last["by_domain"].items():
            tot_prompts += v["prompts"]
            by_dom[d] += v["prompts"]
        gpu_h += last["elapsed_h"]
    counts = {}
    for d in C.DOMAINS:
        path = os.path.join(C.PROMPT_DIR, f"{d}.jsonl")
        if os.path.exists(path):
            n = defaultdict(int)
            for r in C.read_jsonl(path):
                n[r["wave"]] += 1
            counts[d] = dict(n)
    mix_rate = tot_prompts / max(1e-9, gpu_h)                       # mixed prompts per GPU-hour, as sampled
    proj = {}
    for w in ("A", "B"):
        n = sum(c.get(w, 0) for c in counts.values())
        proj[w] = {"prompts": n, "decode_gpu_h": n / max(1e-9, mix_rate),
                   "draft_gpu_h_est": a.draft_gpu_h * (0.128 if w == "A" else 0.872)}
        proj[w]["hours_on_8"] = (proj[w]["decode_gpu_h"] + proj[w]["draft_gpu_h_est"]) / 8
    both = proj["A"]["hours_on_8"] + proj["B"]["hours_on_8"]
    return emit("speed", {"sampled_prompts": tot_prompts, "sampled_by_domain": dict(by_dom),
                          "gpu_hours_sampled": round(gpu_h, 3),
                          "mixed_prompts_per_gpu_h": round(mix_rate), "projection": proj,
                          "both_waves_hours_on_8": round(both, 1), "max_gen_hours": a.max_gen_hours},
                both <= a.max_gen_hours)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--ref-dir", required=True, help="run_da_decode --stock --out dir(s) (searched recursively)")
    g.add_argument("--sgl-dir", default="logs/c1b/parity/sglang")
    g.add_argument("--splits", default="gsm8k_pool:0:200,opencode_pool:0:100")
    g.add_argument("--acc-tol", type=float, default=3.0)
    g.add_argument("--ppb-tol", type=float, default=0.05)
    d = sub.add_parser("draft")
    d.add_argument("--parity-gen", default="logs/c1b/parity/sglang")
    d.add_argument("--parity-draft", default="logs/c1b/parity/draft")
    d.add_argument("--max-items", type=int, default=100)
    d.add_argument("--max-blocks", type=int, default=400)
    d.add_argument("--min-agree", type=float, default=0.95)
    t = sub.add_parser("train")
    t.add_argument("--index", default="logs/c1b/parity/trainset/index.jsonl")
    t.add_argument("--n-states", type=int, default=24)
    s = sub.add_parser("speed")
    s.add_argument("--gen-dir", default="logs/c1b/speed")
    s.add_argument("--draft-gpu-h", type=float, default=74.0, help="plan estimate for both waves' drafts")
    s.add_argument("--max-gen-hours", type=float, default=73.0, help="plan go/no-go: both waves on 8 GPUs")
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "draft": cmd_draft, "train": cmd_train, "speed": cmd_speed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
