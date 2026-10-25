#!/usr/bin/env python
"""Teacher-forced drafts: SDAR-1.7B-Chat-b32 drafts every block of every KEPT answer, via SGLang, one process per GPU.

    CUDA_VISIBLE_DEVICES=3 /venv/sgl/bin/python code/corpus1b/draft_sglang.py --wave A --gpu 3 --num-gpus 8

For each kept generation record (gen_sglang.py) and each of its blocks b, one request:
    input_ids      = prompt ids + the 30B's FINAL tokens before block b   (teacher forcing: the correct prefix)
    max_new_tokens = the block's generated slots (32, or 32 - prompt tail for block 0)
so SGLang runs exactly one decode block, with the algorithm StaticDraftTraced (sglang_plugin/c1b_static_draft.py):
the round-3 drafter setting s8 (min(8, n_open) static steps, top-k by fp32 confidence, [MASK] suppressed) from the
PINNED base checkpoint JetLM/SDAR-1.7B-Chat-b32 @ research_protocol.MODELS["d32"]["revision"]. The block's
drafted tokens and each token's committing confidence come back through the plugin's trace. Requests are sent
item by item, so the radix cache reuses each item's prefix across its blocks.

Output per GPU: logs/c1b/draft/wave<W>/gpu<k>/drafts.<n>.npz with
    pids [n] (str), nb [n] int32, draft [sum nb, 32] int32 (-1 = slot not drafted: the prompt tail),
    conf [sum nb, 32] uint8 (round(255 * p); 0 where not drafted)
and done.txt (pids finished). The work split is by pid hash, independent of which GPU generated the item.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402
import research_protocol as P                                            # noqa: E402
from gen_sglang import TraceReader, shard_of, snapshot_dir               # noqa: E402

BLOCK = C.BLOCK


def kept_records(gen_root: str, gpu: int, ngpu: int, done: set, limit_items: int = 0):
    """Kept generation records of this GPU's share (by pid hash), streamed from every gen shard."""
    n = 0
    for path in sorted(glob.glob(os.path.join(gen_root, "gpu*", "records.*.jsonl.gz"))):
        for r in C.read_jsonl(path):
            if not r.get("kept") or r["pid"] in done or shard_of("draft|" + r["pid"], ngpu) != gpu:
                continue
            yield r
            n += 1
            if limit_items and n >= limit_items:
                return


def block_requests(rec: dict) -> List[dict]:
    """One request per block of a kept record (teacher-forced prefix)."""
    prompt_ids = rec["prompt_ids"]
    L = len(prompt_ids)
    lo0 = int(rec["lo0"])
    gen = [t for b in rec["blocks"] for t in b["ids"]][L - lo0:]      # generated tokens, block-aligned from lo0
    out = []
    for b in range(len(rec["blocks"])):
        lo = lo0 + BLOCK * b
        n_prev = lo - L if b else 0                                    # generated tokens before this block
        ids = list(prompt_ids) + (gen[:n_prev] if b else [])
        assert len(ids) == (L if b == 0 else lo), (rec["pid"], b)
        out.append({"rid": f"{rec['pid']}#{b}", "input_ids": ids, "max_new": BLOCK - (L - lo0 if b == 0 else 0),
                    "lo": lo, "b": b})
    return out


def pack(pids: List[str], per_item: Dict[str, dict]) -> dict:
    nb = np.array([per_item[p]["draft"].shape[0] for p in pids], np.int32)
    return {"pids": np.array(pids, dtype=object), "nb": nb,
            "draft": np.concatenate([per_item[p]["draft"] for p in pids]).astype(np.int32),
            "conf": np.concatenate([per_item[p]["conf"] for p in pids]).astype(np.uint8)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", choices=("A", "B"), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--gen", default=None, help="default logs/c1b/gen/wave<W>")
    ap.add_argument("--out", default=None, help="default logs/c1b/draft/wave<W>")
    ap.add_argument("--items-per-chunk", type=int, default=2000)
    ap.add_argument("--limit-items", type=int, default=0, help="smoke tests")
    ap.add_argument("--max-running", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    args = ap.parse_args(argv)

    gen_root = P.safe_out(args.gen or os.path.join("logs", "c1b", "gen", f"wave{args.wave}"))
    out = os.path.join(P.safe_out(args.out or os.path.join("logs", "c1b", "draft", f"wave{args.wave}")),
                       f"gpu{args.gpu}")
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join(out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    trace_dir = os.path.join(out, "trace")
    os.environ["C1B_TRACE_DIR"] = trace_dir
    if next(kept_records(gen_root, args.gpu, args.num_gpus, done, 1), None) is None:
        print(f"[draft] gpu {args.gpu}: nothing to draft", flush=True)        # a rerun: never load the engine for nothing
        return

    import sglang as sgl
    engine = sgl.Engine(
        model_path=snapshot_dir("d32"), tokenizer_path=snapshot_dir("d32"), trust_remote_code=True,
        skip_tokenizer_init=True, dtype="bfloat16", random_seed=0, log_level="warning",
        dllm_algorithm="StaticDraftTraced",
        dllm_algorithm_config=os.path.join(HERE, "sglang_plugin", "draft_s8.yaml"),
        max_running_requests=args.max_running, mem_fraction_static=args.mem_fraction,
        context_length=args.context_length,
        disable_cuda_graph=args.disable_cuda_graph,
        **({"attention_backend": args.attention_backend} if args.attention_backend else {}))
    reader = TraceReader(trace_dir)
    t0 = time.time()
    n_items = n_blocks = n_bad = 0
    ci = len(glob.glob(os.path.join(out, "drafts.*.npz")))
    try:
        batch: List[dict] = []
        stream = kept_records(gen_root, args.gpu, args.num_gpus, done, args.limit_items)
        while True:
            batch = []
            for rec in stream:
                batch.append(rec)
                if len(batch) >= args.items_per_chunk:
                    break
            if not batch:
                break
            reqs = [q for rec in batch for q in block_requests(rec)]
            engine.generate(input_ids=[q["input_ids"] for q in reqs],
                            sampling_params=[{"temperature": 0.0, "max_new_tokens": q["max_new"],
                                              "ignore_eos": True} for q in reqs],
                            rid=[q["rid"] for q in reqs])
            reader.poll()
            per_item: Dict[str, dict] = {}
            good = []
            for rec in batch:
                nb = len(rec["blocks"])
                draft = np.full((nb, BLOCK), -1, np.int32)
                conf = np.zeros((nb, BLOCK), np.uint8)
                ok = True
                for q in block_requests(rec):
                    got = reader.take(q["rid"]).get(q["lo"])
                    if got is None or len(got["ids"]) != BLOCK:
                        ok = False
                        break
                    start = int(got["start"])
                    ids = np.asarray(got["ids"], np.int64)
                    cf = np.asarray(got["conf"], np.float64)
                    draft[q["b"], start:] = ids[start:]
                    conf[q["b"], start:] = np.clip(np.rint(cf[start:] * 255.0), 0, 255).astype(np.uint8)
                if ok:
                    per_item[rec["pid"]] = {"draft": draft, "conf": conf}
                    good.append(rec["pid"])
                    n_blocks += nb
                else:
                    n_bad += 1
            if good:
                path = os.path.join(out, f"drafts.{ci:05d}.npz")
                np.savez_compressed(path + ".tmp.npz", **pack(good, per_item))
                os.replace(path + ".tmp.npz", path)
                ci += 1
                with open(done_path, "a") as fh:
                    fh.write("\n".join(good) + "\n")
            n_items += len(good)
            el = time.time() - t0
            print(f"[draft] gpu {args.gpu}: {n_items} items / {n_blocks} blocks drafted ({n_bad} incomplete) "
                  f"in {el / 3600:.2f} h = {n_blocks / max(el, 1e-9):.0f} blocks/s", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
