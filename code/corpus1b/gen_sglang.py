#!/usr/bin/env python
"""Wave generation: SDAR-30B-A3B-Chat-b32's OWN answers to the corpus prompts, via SGLang, one process per GPU.

    # one GPU (the launcher code/corpus1b/run_gen.sh starts one per GPU)
    CUDA_VISIBLE_DEVICES=3 /venv/sgl/bin/python code/corpus1b/gen_sglang.py --wave A --gpu 3 --num-gpus 8

What it does, per GPU:
  1. takes the prompt rows of the wave whose pid hashes to this GPU (data/c1b/prompt_lists/<domain>.jsonl),
     skipping pids already finished (resume);
  2. encodes each prompt exactly as every runner does (research_protocol.encode_prompt, b32 tokenizer);
  3. decodes with SGLang's dLLM engine, algorithm LowConfidenceTraced (sglang_plugin/c1b_traced.py) at block 32,
     threshold 0.95 -- the stock rule of sdar_harness/decode.py -- from the PINNED base checkpoint
     JetLM/SDAR-30B-A3B-Chat-b32 @ research_protocol.MODELS["b32"]["revision"] (the snapshot directory of that
     revision is passed as the model path, so no other weights can be loaded);
  4. rebuilds every answer block by block from the plugin's trace (tokens + the pass that committed each slot),
     cross-checks it against the tokens SGLang returned, and runs the domain's checker (checkers.py) in a
     process pool while the next chunk decodes;
  5. writes logs/c1b/gen/wave<W>/gpu<k>/records.<n>.jsonl.gz (kept rows carry prompt_ids + blocks; dropped rows
     carry the reason) and appends the chunk's pids to done.txt only after the file is complete.

`--items SPLIT` decodes research_protocol items instead (e.g. gsm8k_pool:0:200) for the stage-1 parity check
against run_da_decode.py --stock; those records carry `pred`/`gold` and are scored by parity_check.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                       # noqa: E402
import research_protocol as P                                            # noqa: E402  (HF offline: local files only)

BLOCK = C.BLOCK


# ---------------------------------------------------------------------------
# the pinned base checkpoints
# ---------------------------------------------------------------------------
def snapshot_dir(model_key: str) -> str:
    """Local snapshot directory of MODELS[model_key] at its PINNED revision (downloaded in stage 1)."""
    m = P.MODELS[model_key]
    d = os.path.join(m["cache_dir"], "models--" + m["repo"].replace("/", "--"), "snapshots", m["revision"])
    assert os.path.isfile(os.path.join(d, "config.json")), (
        f"{m['repo']} @ {m['revision']} is not in {m['cache_dir']} -- run stage 1 (setup_box.sh) first")
    return d


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(model_key), trust_remote_code=True, local_files_only=True)
    assert tok(tok.mask_token)["input_ids"][0] == C.MASK_ID, "tokenizer mask id != 151669"
    return tok


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def shard_of(pid: str, n: int) -> int:
    return int(C.order_key(pid, "c1b-shard")[:12], 16) % max(1, n)


def load_rows(wave: str, domains: Sequence[str], gpu: int, ngpu: int, done: set,
              limit: Optional[int] = None) -> List[dict]:
    rows = []
    for d in domains:
        path = os.path.join(C.PROMPT_DIR, f"{d}.jsonl")
        assert os.path.exists(path), f"missing prompt list {path} -- run build_prompt_lists.py"
        n = 0
        for r in C.read_jsonl(path):
            if r.get("wave") != wave or shard_of(r["pid"], ngpu) != gpu or r["pid"] in done:
                continue
            rows.append(r)
            n += 1
            if limit and n >= limit:
                break
    rows.sort(key=lambda r: C.order_key(r["pid"]))            # domains interleave: a slow one shows up early
    return rows


def rows_from_items(split: str, gpu: int, ngpu: int) -> List[dict]:
    """research_protocol items (parity mode); gen_cap = GEN_LEN of the split's family."""
    base = P.split_base(split)
    cap = int(P.GEN_LEN[base])
    out = []
    for it in P.items_for_split(split):
        pid = f"{base}#{int(it['idx'])}"
        if shard_of(pid, ngpu) == gpu:
            out.append({"pid": pid, "domain": base, "source": base, "prompt": it["prompt"], "gen_cap": cap,
                        "key": None, "gold": it.get("gold"), "idx": int(it["idx"]), "split": it.get("split")})
    return out


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------
class TraceReader:
    """Reads the plugin's JSONL trace files incrementally (the scheduler process appends to them)."""

    def __init__(self, trace_dir: str):
        self.dir = trace_dir
        self.offsets: Dict[str, int] = {}
        self.pending: Dict[str, Dict[int, dict]] = defaultdict(dict)   # rid -> lo -> row (a re-run block overwrites)

    def poll(self) -> None:
        if not os.path.isdir(self.dir):
            return
        for name in sorted(os.listdir(self.dir)):
            if not name.startswith("trace."):
                continue
            path = os.path.join(self.dir, name)
            off = self.offsets.get(path, 0)
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(off)
                while True:
                    line = fh.readline()
                    if not line or not line.endswith("\n"):          # a line being written: read it next time
                        break
                    off = fh.tell()
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self.pending[str(r["rid"])][int(r["lo"])] = r
            self.offsets[path] = off

    def take(self, rid: str) -> Dict[int, dict]:
        return self.pending.pop(rid, {})


class TraceError(ValueError):
    pass


def assemble(blocks_by_lo: Dict[int, dict], prompt_ids: Sequence[int], cap: int) -> dict:
    """Blocks of one answer, in order, up to the block holding the first EOS (or the cap).

    Returns {"blocks": [{"ids","steps"}], "gen_ids", "n_gen", "eos", "ppb", "lo0"}. Raises TraceError when the
    trace is not a contiguous run of blocks from the prompt's block boundary, or block 0's fixed slots are not
    the prompt tail.
    """
    L = len(prompt_ids)
    lo0 = (L // BLOCK) * BLOCK
    tail = L - lo0
    if lo0 not in blocks_by_lo:
        raise TraceError(f"no block at lo0={lo0}")
    blocks, gen = [], []
    lo = lo0
    while lo in blocks_by_lo:
        b = blocks_by_lo[lo]
        ids, steps = [int(x) for x in b["ids"]], [int(x) for x in b["steps"]]
        if len(ids) != BLOCK or len(steps) != BLOCK:
            raise TraceError(f"block at {lo} has {len(ids)} ids / {len(steps)} steps")
        fixed = tail if lo == lo0 else 0
        if int(b.get("start", fixed)) != fixed:
            raise TraceError(f"block at {lo}: start {b.get('start')} != {fixed}")
        if fixed and ids[:fixed] != list(prompt_ids[lo0:L]):
            raise TraceError("block 0 fixed slots are not the prompt tail")
        if any(s < 0 for s in steps[fixed:]) or any(s != -1 for s in steps[:fixed]):
            raise TraceError(f"block at {lo}: bad commit steps")
        blocks.append({"ids": ids, "steps": steps})
        gen.extend(ids[fixed:])
        if C.first_eos(ids[fixed:]) >= 0 or len(gen) >= cap:
            break
        lo += BLOCK
    e = C.first_eos(gen)
    eos = 0 <= e < cap
    n_gen = (e + 1) if eos else min(len(gen), cap)
    return {"blocks": blocks, "gen_ids": gen[:n_gen], "n_gen": n_gen, "eos": bool(eos),
            "ppb": [max(b["steps"]) + 1 for b in blocks], "lo0": lo0}


# ---------------------------------------------------------------------------
# checking (process pool; code execution happens in subprocesses inside checkers.py)
# ---------------------------------------------------------------------------
_TOK = None
_SANDBOX = None


def _init_worker(model_key: str, sandbox: str) -> None:
    global _TOK, _SANDBOX
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TOK = load_tokenizer(model_key)
    _SANDBOX = sandbox
    os.makedirs(sandbox, exist_ok=True)


def _check_one(job: dict) -> dict:
    import checkers                                              # code/corpus1b/checkers.py
    try:
        return checkers.check(job["key"], job["gen_ids"],
                              lambda ids: _TOK.decode(ids, skip_special_tokens=True),
                              sandbox_dir=_SANDBOX, gen_cap=int(job["gen_cap"]))
    except Exception as e:                                           # noqa: BLE001 - a checker bug drops, never kills
        return {"ok": False, "reason": f"checker_error:{type(e).__name__}", "loop": False,
                "eos": job.get("eos", False), "n_gen": len(job["gen_ids"])}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", choices=("A", "B"))
    ap.add_argument("--items", default=None, help="parity mode: a research_protocol split instead of the prompt lists")
    ap.add_argument("--domains", default=",".join(C.DOMAINS))
    ap.add_argument("--gpu", type=int, required=True, help="this process's shard (0..num_gpus-1)")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--out", default=None, help="default logs/c1b/gen/wave<W> (parity: logs/c1b/parity/sglang)")
    ap.add_argument("--limit", type=int, default=None, help="max prompts per domain (smoke tests)")
    ap.add_argument("--limit-total", type=int, default=None,
                    help="the first N of this GPU's shard in hash order = the wave's natural domain mix (speed check)")
    ap.add_argument("--chunk", type=int, default=4096, help="prompts per engine.generate call")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="unix time: after it, submit no new chunk, finish the one in flight, exit 0. Chunks are "
                         "then sized from the measured decode rate so the last one ends near this time")
    ap.add_argument("--first-chunk", type=int, default=1024, help="with --stop-at: size of the first chunk (rate probe)")
    ap.add_argument("--min-chunk", type=int, default=256, help="with --stop-at: smallest chunk")
    ap.add_argument("--max-running", type=int, default=128, help="SGLang max_running_requests")
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--attention-backend", default=None, help="flashinfer | triton | torch_native (stage 1 picks)")
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--check-workers", type=int, default=6)
    ap.add_argument("--sandbox", default="/dev/shm/c1b_sandbox", help="cwd root for executing generated code")
    ap.add_argument("--no-check", action="store_true", help="skip checkers (parity mode sets this)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    parity = args.items is not None
    assert parity or args.wave, "--wave or --items"
    out_root = P.safe_out(args.out or (os.path.join("logs", "c1b", "parity", "sglang") if parity
                                       else os.path.join("logs", "c1b", "gen", f"wave{args.wave}")))
    out = os.path.join(out_root, f"gpu{args.gpu}")
    os.makedirs(out, exist_ok=True)
    trace_dir = os.path.join(out, "trace")
    # request ids are the prompt ids: a crashed run's leftover trace would mix into this run's answers
    shutil.rmtree(trace_dir, ignore_errors=True)
    done_path = os.path.join(out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    tok = load_tokenizer("b32")
    if parity:
        rows = [r for r in rows_from_items(args.items, args.gpu, args.num_gpus) if r["pid"] not in done]
    else:
        rows = load_rows(args.wave, [d for d in args.domains.split(",") if d], args.gpu, args.num_gpus, done, args.limit)
        if args.limit_total:
            rows = rows[:args.limit_total]
    print(f"[gen] gpu {args.gpu}/{args.num_gpus}: {len(rows)} prompts to decode ({len(done)} already done) -> {out}",
          flush=True)
    if not rows:
        return
    if args.stop_at and time.time() >= args.stop_at:
        print(f"[gen] gpu {args.gpu}: --stop-at already passed; nothing submitted", flush=True)
        return
    for r in rows:
        r["prompt_ids"] = P.encode_prompt(tok, r["prompt"])[0].tolist()

    # the engine: pinned base checkpoint, traced stock rule, block 32
    os.environ["C1B_TRACE_DIR"] = trace_dir                   # inherited by SGLang's scheduler process
    import sglang as sgl
    engine = sgl.Engine(
        model_path=snapshot_dir("b32"), tokenizer_path=snapshot_dir("b32"), trust_remote_code=True,
        skip_tokenizer_init=True, dtype="bfloat16", random_seed=0, log_level="warning",
        dllm_algorithm="LowConfidenceTraced",
        dllm_algorithm_config=os.path.join(HERE, "sglang_plugin", "gen_b32.yaml"),
        max_running_requests=args.max_running, mem_fraction_static=args.mem_fraction,
        context_length=args.context_length, disable_radix_cache=True,
        disable_cuda_graph=args.disable_cuda_graph,
        **({"attention_backend": args.attention_backend} if args.attention_backend else {}))
    reader = TraceReader(trace_dir)
    pool = None if (parity or args.no_check) else ProcessPoolExecutor(
        max_workers=args.check_workers, initializer=_init_worker, initargs=("b32", args.sandbox))

    stats = Counter()
    by_dom = defaultdict(Counter)
    pending = None                                           # (chunk index, rows, results, futures, t_decode)
    t_start = time.time()
    ci0 = next_record_index(out)

    def finish(job) -> None:
        ci, chunk_rows, results, futs, t_dec = job
        path = os.path.join(out, f"records.{ci:05d}.jsonl.gz")
        recs = []
        for r, res, fut in zip(chunk_rows, results, futs):
            rec = {"pid": r["pid"], "domain": r["domain"], "source": r.get("source"), "wave": r.get("wave"),
                   "prompt_len": len(r["prompt_ids"]), "gen_cap": int(r["gen_cap"])}
            if isinstance(res, str):                                  # trace problem
                rec.update({"kept": False, "reason": res, "n_gen": 0, "eos": False, "loop": False, "ppb": []})
            else:
                rec.update({"n_gen": res["n_gen"], "eos": res["eos"], "ppb": res["ppb"]})
                if parity:
                    rec.update({"idx": r["idx"], "split": r.get("split"), "gold": r.get("gold"),
                                "prompt_ids": r["prompt_ids"],
                                "gen_ids": res["gen_ids"], "blocks": res["blocks"], "lo0": res["lo0"],
                                "pred": tok.decode(res["gen_ids"], skip_special_tokens=True), "kept": True,
                                "reason": ""})
                else:
                    v = fut.result()
                    rec.update({"kept": bool(v["ok"]), "reason": v.get("reason", ""), "loop": bool(v.get("loop"))})
                    if rec["kept"]:
                        rec.update({"prompt_ids": r["prompt_ids"], "blocks": res["blocks"], "lo0": res["lo0"]})
            recs.append(rec)
            d = by_dom[rec["domain"]]
            d["prompts"] += 1; d["kept"] += int(rec["kept"]); d["gen_tokens"] += rec["n_gen"]
            d["blocks"] += len(rec["ppb"]); d["passes"] += sum(rec["ppb"])
            if not rec["kept"]:
                d["drop:" + (rec["reason"] or "?")] += 1
        C.write_jsonl(path + ".tmp.gz", recs, mode="wt")          # the name must end in .gz: open_text gzips by suffix
        os.replace(path + ".tmp.gz", path)
        with open(done_path, "a") as fh:
            fh.write("\n".join(r["pid"] for r in chunk_rows) + "\n")
        el = time.time() - t_start
        line = {"chunk": ci, "n_prompts": len(chunk_rows), "left": len(rows) - n_submitted,
                "elapsed_h": round(el / 3600, 3), "decode_s": round(t_dec, 1),
                "by_domain": {k: dict(v) for k, v in by_dom.items()}}
        with open(os.path.join(out, "stats.jsonl"), "a") as fh:
            fh.write(json.dumps(line) + "\n")
        summ = " ".join(f"{k}:{v['kept']}/{v['prompts']} ppb={v['passes'] / max(1, v['blocks']):.1f} "
                        f"{v['prompts'] / max(el / 3600, 1e-9):.0f}/h" for k, v in sorted(by_dom.items()))
        print(f"[gen] chunk {ci} written ({len(chunk_rows)} prompts, {t_dec:.0f}s decode, "
              f"{len(rows) - n_submitted} left) | {summ}", flush=True)

    n_submitted, dec_prompts, dec_s = 0, 0, 0.0
    try:
        ci = ci0
        while n_submitted < len(rows):
            now = time.time()
            if args.stop_at and now >= args.stop_at:
                print(f"[gen] gpu {args.gpu}: --stop-at reached, {len(rows) - n_submitted} prompts not submitted",
                      flush=True)
                break
            n = args.chunk
            if args.stop_at:                                    # size the chunk to end near --stop-at
                n = (min(n, args.first_chunk) if not dec_prompts else
                     min(n, max(args.min_chunk, int(dec_prompts / max(dec_s, 1e-9) * (args.stop_at - now)))))
            chunk_rows = rows[n_submitted:n_submitted + n]
            n_submitted += len(chunk_rows)
            t0 = time.time()
            outs = engine.generate(
                input_ids=[r["prompt_ids"] for r in chunk_rows],
                sampling_params=[{"temperature": 0.0, "max_new_tokens": int(r["gen_cap"]),
                                  "stop_token_ids": list(C.EOS_IDS)} for r in chunk_rows],
                rid=[r["pid"] for r in chunk_rows])
            t_dec = time.time() - t0
            dec_prompts += len(chunk_rows); dec_s += t_dec
            reader.poll()
            results, futs = [], []
            for r, o in zip(chunk_rows, outs):
                try:
                    res = assemble(reader.take(r["pid"]), r["prompt_ids"], int(r["gen_cap"]))
                    got = [int(x) for x in (o.get("output_ids") or [])]
                    n = min(len(got), res["n_gen"])
                    if n and got[:n] != res["gen_ids"][:n]:
                        raise TraceError("trace tokens != returned tokens")
                except TraceError as e:
                    results.append(f"trace:{e}"[:120])
                    futs.append(None)
                    continue
                results.append(res)
                if parity:
                    futs.append(None)
                elif not res["eos"]:                            # no EOS within the cap: dropped, no checker needed
                    futs.append(_Immediate({"ok": False, "reason": "no_eos", "loop": C.has_loop(res["gen_ids"])}))
                elif pool is None:                              # --no-check: keep every answer that ended
                    futs.append(_Immediate({"ok": True, "reason": "", "loop": C.has_loop(res["gen_ids"])}))
                else:
                    futs.append(pool.submit(_check_one, {"key": r["key"], "gen_ids": res["gen_ids"],
                                                         "gen_cap": r["gen_cap"], "eos": res["eos"]}))
            if pending is not None:
                finish(pending)
            pending = (ci, chunk_rows, results, futs, t_dec)
            ci += 1
        if pending is not None:
            finish(pending)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        engine.shutdown()
    print(f"[gen] gpu {args.gpu}: done in {(time.time() - t_start) / 3600:.2f} h", flush=True)


def next_record_index(out: str) -> int:
    """1 + the highest records.<n>.jsonl.gz index already in `out` (0 if none)."""
    got = [int(m.group(1)) for m in (re.match(r"records\.(\d+)\.jsonl\.gz$", f) for f in os.listdir(out)) if m]
    return max(got) + 1 if got else 0


class _Immediate:
    """A resolved future (rows that need no checker)."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


if __name__ == "__main__":
    main()
