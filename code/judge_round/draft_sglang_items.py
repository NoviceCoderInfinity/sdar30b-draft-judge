#!/usr/bin/env python
"""Judge round, step 2 via SGLang: the TRAINED drafter's s8 first drafts of corpus ITEMS, teacher-forced, one process per GPU.

    CUDA_VISIBLE_DEVICES=3 /venv/sgl/bin/python code/judge_round/draft_sglang_items.py \
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \
        --pids logs/phase5/judge/subset_pids.json --drafter-dir logs/phase4/train/main/merged \
        --out logs/phase5/judge/drafts --gpu 3 --num-gpus 8

The flow of corpus1b/draft_sglang.py (same engine arguments, same plugin StaticDraftTraced with
sglang_plugin/draft_s8.yaml, same trace parsing, same per-GPU chunk format), with three changes:
  * the input is the 1B corpus' trainset items (corpus1b/to_trainset.py npz, read through drafter_lora/data.py),
    not the gen records (those are not on the box). This GPU's items are sorted(--pids)[gpu::num_gpus];
  * the requests are rebuilt from the item: block b gets input_ids = prompt_ids + the 30B's tokens before block b
    (tokens[b'] for b' < b, the prompt tail of block 0 skipped) and max_new_tokens = 32, or 32 - prompt tail for
    block 0, so SGLang runs exactly one decode block per request;
  * the model is --drafter-dir (the merged Phase 4 drafter, an HF dir with the SDAR remote code), not the pinned base.

Output: <out>/gpu<k>/drafts.<n>.npz with
    pids [n] (str), nb [n] int32, draft [sum nb, 32] int32 (-1 = slot not drafted: the prompt tail),
    conf [sum nb, 32] uint8 (round(255 * p); 0 where not drafted)
and done.txt (pids finished; a rerun skips them), then, once this GPU's work is done, every chunk merged into
<out>/drafts.shard<k>of<n>.npz (same keys, pids as a plain str array), which drafter_lora/judge_probe.load_drafts
and judge_round/swap_drafts.py read.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CORPUS = os.path.join(CODE, "corpus1b")
for p in (os.path.join(CODE, "compat"), CODE, CORPUS):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402
import research_protocol as P                                            # noqa: E402
from gen_sglang import TraceReader                                       # noqa: E402
from drafter_lora import data as D                                       # noqa: E402

BLOCK = C.BLOCK
_CHUNK = re.compile(r"drafts\.(\d+)\.npz$")


# ---------------------------------------------------------------------------
# work list
# ---------------------------------------------------------------------------
def shard_pids(pids: Sequence[str], gpu: int, num_gpus: int) -> List[str]:
    """This GPU's items: the sorted pid list, sliced [gpu::num_gpus]."""
    return sorted({str(p) for p in pids})[gpu::max(1, num_gpus)]


def pending(mine: Sequence[str], done: set, limit_items: int = 0) -> List[str]:
    """The shard's pids not finished yet (resume), at most `limit_items` of them (0 = all)."""
    out = [p for p in mine if p not in done]
    return out[:limit_items] if limit_items else out


def read_done(gpu_dir: str) -> set:
    path = os.path.join(gpu_dir, "done.txt")
    return set(open(path).read().split()) if os.path.exists(path) else set()


# ---------------------------------------------------------------------------
# requests and their trace
# ---------------------------------------------------------------------------
def item_requests(item: dict) -> List[dict]:
    """One request per block of a trainset item (teacher-forced prefix), as draft_sglang.block_requests."""
    pid = str(item["pid"])
    prompt_ids = [int(x) for x in np.asarray(item["prompt_ids"]).tolist()]
    L = int(item["prompt_length"])
    assert len(prompt_ids) == L, (pid, len(prompt_ids), L)
    tokens = np.asarray(item["tokens"])
    starts = [int(x) for x in np.asarray(item["block_start"]).tolist()]
    nb = tokens.shape[0]
    assert tokens.shape == (nb, BLOCK) and len(starts) == nb, (pid, tokens.shape, len(starts))
    lo0 = starts[0]
    assert lo0 == (L // BLOCK) * BLOCK, (pid, lo0, L)
    tail = L - lo0
    gen = [int(t) for t in tokens.reshape(-1)[tail:].tolist()]   # generated tokens, from absolute position L
    out = []
    for b in range(nb):
        lo = starts[b]
        n_prev = lo - L if b else 0                                    # generated tokens before this block
        ids = prompt_ids + (gen[:n_prev] if b else [])
        assert len(ids) == (L if b == 0 else lo), (pid, b)
        out.append({"rid": f"{pid}#{b}", "input_ids": ids, "max_new": BLOCK - (tail if b == 0 else 0),
                    "lo": lo, "b": b})
    return out


def pack_results(items: Sequence[dict], reader, per_reqs: Optional[Sequence[List[dict]]] = None
                 ) -> Tuple[Dict[str, dict], List[str], int, int]:
    """The plugin's trace of each item's blocks -> (per_item {pid: {draft, conf}}, good pids, blocks, n incomplete).

    `reader.take(rid)` -> {lo: {"start", "ids"[32], "conf"[32]}} (gen_sglang.TraceReader); an item with any block
    missing from the trace is incomplete and left out (not marked done, so a rerun drafts it again).
    """
    per_item: Dict[str, dict] = {}
    good: List[str] = []
    n_blocks = n_bad = 0
    for i, it in enumerate(items):
        reqs = per_reqs[i] if per_reqs is not None else item_requests(it)
        nb = len(reqs)
        draft = np.full((nb, BLOCK), -1, np.int32)
        conf = np.zeros((nb, BLOCK), np.uint8)
        ok = True
        for q in reqs:
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
            pid = str(it["pid"])
            per_item[pid] = {"draft": draft, "conf": conf}
            good.append(pid)
            n_blocks += nb
        else:
            n_bad += 1
    return per_item, good, n_blocks, n_bad


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def pack(pids: List[str], per_item: Dict[str, dict]) -> dict:
    nb = np.array([per_item[p]["draft"].shape[0] for p in pids], np.int32)
    return {"pids": np.array(pids, dtype=object), "nb": nb,
            "draft": np.concatenate([per_item[p]["draft"] for p in pids]).astype(np.int32),
            "conf": np.concatenate([per_item[p]["conf"] for p in pids]).astype(np.uint8)}


def chunk_paths(gpu_dir: str) -> List[str]:
    """drafts.<n>.npz of one GPU, in chunk order (a crash's leftover .tmp.npz is not a chunk)."""
    if not os.path.isdir(gpu_dir):
        return []
    got = sorted((int(m.group(1)), f) for f in os.listdir(gpu_dir) for m in [_CHUNK.match(f)] if m)
    return [os.path.join(gpu_dir, f) for _, f in got]


def next_chunk_index(gpu_dir: str) -> int:
    got = [int(_CHUNK.match(os.path.basename(f)).group(1)) for f in chunk_paths(gpu_dir)]
    return max(got) + 1 if got else 0


def write_chunk(gpu_dir: str, ci: int, good: List[str], per_item: Dict[str, dict]) -> str:
    """drafts.<ci>.npz (atomic), then the chunk's pids appended to done.txt (only after the file is complete)."""
    path = os.path.join(gpu_dir, f"drafts.{ci:05d}.npz")
    np.savez_compressed(path + ".tmp.npz", **pack(good, per_item))
    os.replace(path + ".tmp.npz", path)
    with open(os.path.join(gpu_dir, "done.txt"), "a") as fh:
        fh.write("\n".join(good) + "\n")
    return path


def merge_shard(out_root: str, gpu: int, num_gpus: int, keep: Optional[set] = None) -> Optional[dict]:
    """Every chunk of <out_root>/gpu<gpu> -> <out_root>/drafts.shard<gpu>of<num_gpus>.npz (judge_probe.load_drafts).

    pids are sorted and stored as a plain str array (load_drafts opens the file without allow_pickle); a pid found in
    two chunks (a crash between a chunk and its done.txt line) keeps its later draft. `keep` restricts to these pids.
    """
    rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in chunk_paths(os.path.join(out_root, f"gpu{gpu}")):
        z = np.load(path, allow_pickle=True)
        draft, conf = z["draft"], z["conf"]
        off = 0
        for pid, nb in zip(z["pids"].tolist(), z["nb"].tolist()):
            if keep is None or str(pid) in keep:
                rows[str(pid)] = (draft[off:off + nb], conf[off:off + nb])
            off += int(nb)
    if not rows:
        return None
    pids = sorted(rows)
    dst = os.path.join(out_root, f"drafts.shard{gpu}of{num_gpus}.npz")
    tmp = os.path.join(out_root, f".tmp.drafts.shard{gpu}of{num_gpus}.npz")   # never matches drafts.shard*.npz
    np.savez_compressed(tmp, pids=np.array(pids, dtype=str),
                        nb=np.array([rows[p][0].shape[0] for p in pids], np.int32),
                        draft=np.concatenate([rows[p][0] for p in pids]).astype(np.int32),
                        conf=np.concatenate([rows[p][1] for p in pids]).astype(np.uint8))
    os.replace(tmp, dst)
    return {"path": dst, "items": len(pids), "blocks": int(sum(rows[p][0].shape[0] for p in pids))}


def report_merge(info: Optional[dict], gpu: int, n_mine: int) -> None:
    if info is None:
        print(f"[draft-items] gpu {gpu}: no drafts to merge", flush=True)
    else:
        print(f"[draft-items] gpu {gpu}: merged {info['items']} / {n_mine} shard items ({info['blocks']} blocks) "
              f"-> {info['path']}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True, help="trainset index.jsonl (repeat per wave)")
    ap.add_argument("--pids", required=True, help='JSON {"pids": [...]} (select_subset.py / holdout_pids.json)')
    ap.add_argument("--drafter-dir", required=True, help="merged Phase 4 drafter (HF dir, SDAR remote code)")
    ap.add_argument("--out", required=True, help="e.g. logs/phase5/judge/drafts")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--items-per-chunk", type=int, default=2000)
    ap.add_argument("--limit-items", type=int, default=0, help="smoke tests")
    ap.add_argument("--max-running", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    return ap


def main(argv=None) -> Optional[dict]:
    args = build_parser().parse_args(argv)
    out_root = P.safe_out(args.out)
    out = os.path.join(out_root, f"gpu{args.gpu}")
    os.makedirs(out, exist_ok=True)
    by_pid = {str(e["pid"]): e for e in D.load_entries(args.index)}
    mine = shard_pids(D.load_holdout(args.pids), args.gpu, args.num_gpus)
    missing = [p for p in mine if p not in by_pid]
    assert not missing, f"{len(missing)} pids are in no --index (first: {missing[:3]})"
    done = read_done(out)
    todo = pending(mine, done, args.limit_items)
    print(f"[draft-items] gpu {args.gpu}/{args.num_gpus}: {len(mine)} shard items, {len(todo)} to draft "
          f"({len(done)} already done) -> {out}", flush=True)
    if not todo:
        print(f"[draft-items] gpu {args.gpu}: nothing to draft", flush=True)   # a rerun: never load the engine for nothing
        info = merge_shard(out_root, args.gpu, args.num_gpus, set(mine))
        report_merge(info, args.gpu, len(mine))
        return info

    drafter = os.path.abspath(args.drafter_dir)
    assert os.path.isfile(os.path.join(drafter, "config.json")), f"no config.json in --drafter-dir {drafter}"
    trace_dir = os.path.join(out, "trace")
    # request ids repeat across runs (pid#b): a crashed run's leftover trace must not mix into this run's drafts
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.environ["C1B_TRACE_DIR"] = trace_dir                   # inherited by SGLang's scheduler process
    import sglang as sgl
    engine = sgl.Engine(
        model_path=drafter, tokenizer_path=drafter, trust_remote_code=True,
        skip_tokenizer_init=True, dtype="bfloat16", random_seed=0, log_level="warning",
        dllm_algorithm="StaticDraftTraced",
        dllm_algorithm_config=os.path.join(CORPUS, "sglang_plugin", "draft_s8.yaml"),
        max_running_requests=args.max_running, mem_fraction_static=args.mem_fraction,
        context_length=args.context_length,
        disable_cuda_graph=args.disable_cuda_graph,
        **({"attention_backend": args.attention_backend} if args.attention_backend else {}))
    reader = TraceReader(trace_dir)
    t0 = time.time()
    n_items = n_blocks = n_bad = 0
    ci = next_chunk_index(out)
    try:
        for c0 in range(0, len(todo), args.items_per_chunk):
            batch = [D.load_item(by_pid[p]) for p in todo[c0:c0 + args.items_per_chunk]]
            per_reqs = [item_requests(it) for it in batch]
            reqs = [q for r in per_reqs for q in r]
            engine.generate(input_ids=[q["input_ids"] for q in reqs],
                            sampling_params=[{"temperature": 0.0, "max_new_tokens": q["max_new"],
                                              "ignore_eos": True} for q in reqs],
                            rid=[q["rid"] for q in reqs])
            reader.poll()
            per_item, good, nbk, bad = pack_results(batch, reader, per_reqs)
            n_blocks += nbk
            n_bad += bad
            if good:
                write_chunk(out, ci, good, per_item)
                ci += 1
            n_items += len(good)
            el = time.time() - t0
            print(f"[draft-items] gpu {args.gpu}: {n_items} items / {n_blocks} blocks drafted ({n_bad} incomplete) "
                  f"in {el / 3600:.2f} h = {n_blocks / max(el, 1e-9):.0f} blocks/s", flush=True)
    finally:
        engine.shutdown()
    info = merge_shard(out_root, args.gpu, args.num_gpus, set(mine))
    report_merge(info, args.gpu, len(mine))
    return info


if __name__ == "__main__":
    main()
