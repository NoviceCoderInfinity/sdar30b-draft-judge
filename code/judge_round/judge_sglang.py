#!/usr/bin/env python
"""Phase-5 judge round on SGLang, one process per GPU: the 30B JUDGE (flag-gated judge LoRA; judge_core's FULL judge with
rd re-drafts, the rejected-token ban and the loop guard) as an in-process SGLang engine running JudgeLoopTraced
(corpus1b/sglang_plugin/c1b_judge_loop.py), and the trained 1.7B DRAFTER as an SGLang server subprocess on the same
GPU running DraftTracedV2 (corpus1b/sglang_plugin/c1b_judge_draft.py), which the judge plugin calls over HTTP.

    # decode: research_protocol items, the twin of run_da_decode.py --judge full (records.jsonl per GPU)
    CUDA_VISIBLE_DEVICES=2 /venv/sgl/bin/python code/judge_round/judge_sglang.py --mode decode --items gsm8k500 \\
        --shard 2 --nshards 8 --gpu 2 --redraft-ban --attention-backend triton --out logs/phase5/judge_sglang/gsm8k500

    # record: trainset items teacher-forced on the vanilla 30B's tokens, every judge pass's state (states.*.npz)
    CUDA_VISIBLE_DEVICES=2 /venv/sgl/bin/python code/judge_round/judge_sglang.py --mode record \\
        --index logs/c1b/trainset/waveA/index.jsonl --index logs/c1b/trainset/waveB/index.jsonl \\
        --pids logs/phase5/judge/subset_pids.json --shard 2 --nshards 8 --gpu 2 --redraft-ban \\
        --attention-backend triton --out logs/phase5/judge_sglang/record

THE MEASUREMENT RULE APPLIES: this is data generation and harness parity only.  Its timings (wall_s, blocks/s) are
progress indicators of a two-engine HTTP setup and are never a reported speed.

Steps:
  1. <out>/judge_<gpu>.yaml = the judge config (block_size 32, tau_commit, tau_override null, max_passes, redraft_every,
     redraft_rate, redraft_ban, loop_guard, threshold 0.95, draft_steps 8); fresh <out>/gpu<k>/{judge_side, judge_trace,
     drafter_side, drafter_trace} (a crashed run's files are removed: request ids repeat across runs);
  2. the drafter server: `python -m sglang.launch_server` on --drafter-dir with DraftTracedV2 + draft_s8.yaml, on
     127.0.0.1:--drafter-port (default 30100 + gpu), radix cache and CUDA graphs off, its own process group, log in
     gpu<k>/drafter_server.log; GET /health must answer within 600 s;
  3. the judge engine in-process: the pinned b32 snapshot (gen_sglang.snapshot_dir), JudgeLoopTraced + the yaml, CUDA
     graphs and radix cache off; the C1B_* environment is set before it starts (its scheduler inherits it);
  4. per chunk of --chunk requests: the judge side files, one engine.generate, the judge trace -> per-item output, then
     the chunk's rids appended to done.txt (a rerun skips them; gpu<k>/run.json refuses a resume with another config).
Both engines are shut down in `finally` (the drafter's whole process group is terminated).

Judge side file per request (read once by the plugin):  {"prompt_ids", "mode", "max_new_tokens", "stop_ids", "teacher"}
  decode   max_new_tokens = --max-new-tokens or research_protocol.GEN_LEN[split] (as run_da_decode); SGLang stops on
           EOS_IDS (stop_token_ids, also the plugin's stop_ids) or at max_new_tokens: the blocks of da_generate
  record   max_new_tokens = 32 * nb - prompt tail with ignore_eos: the request runs exactly the item's nb blocks;
           teacher = the item's tokens[b] (block 0 holds the prompt tail), stored by every closing forward

Output, decode: <out>/gpu<k>/records.jsonl, one JSON per item:
    idx, split, rid, gold, prompt_tokens, gen_ids (the closed blocks' generated slots, cut after the first EOS),
    pred (b32 tokenizer, special tokens skipped), n_blocks, n_judge, n_draft_fwd, n_redraft_rounds, n_redraft_slots,
    n_redraft_fwd, n_loopguard_slots, n_loopguard_changed, full_fallback, n_stock (sums over the blocks),
    passes_per_block (n_judge per block), stopped_eos, wall_s (chunk wall / items: approximate), engine, cfg
Output, record: <out>/gpu<k>/states.NNNNN.npz per chunk (np.load(path, allow_pickle=False) works):
    index        0-d str: JSON {"version": 1, "cfg", "counters": [names], "items": [{"rid", "n_blocks"}]}
    rid          [I] str
    item_block   [I+1] int64     item i's blocks are rows item_block[i]:item_block[i+1] of the blk_* arrays
    blk_index, blk_lo, blk_start   [NB] int32
    blk_tokens   [NB, 32] int32  the judge pipeline's own final tokens
    blk_teacher  [NB, 32] int32  the stored (teacher) tokens = the next block's prefix
    blk_draft    [NB, 32] int32  the first draft; blk_p_draft [NB, 32] float32 its commit confidences
    blk_counters [NB, C] int32   judge_core.block_summary counters, columns = index["counters"]
    blk_pass     [NB+1] int64    block j's judge passes are rows blk_pass[j]:blk_pass[j+1] of the pass arrays
    pass_no [NP] int16; forced [NP] bool
    input, draft, argmax                    [NP, 32] int32    the judge's input, the current draft, its argmax (-1)
    flag, open, commit, tentative, remask   [NP, 32] bool
    p, q                                    [NP, 32] float32  (NaN where not read)
    rd_on [NP] bool; rd_holes [NP, 32] bool; rd_tokens, rd_input [NP, 32] int32 (-1 without a re-draft);
    rd_p_commit [NP, 32] float32 (NaN without); rd_n_fwd [NP] int16       the re-draft that followed the pass

Memory, both engines on one 96 GB RTX PRO 6000 (95.6 GiB): the 30B in bf16 ~61 GB (56.8 GiB) + the 1.7B ~3.4 GB.
--mem-fraction and --drafter-mem-fraction are fractions of the WHOLE GPU.  SGLang's mem_fraction_static is a fraction
of the memory FREE when that engine starts (model_runner.init_torch_distributed's pre_model_load_memory feeds
init_memory_pool), and the drafter starts first, so the judge is passed mem_fraction / (1 - drafter_mem_fraction -
0.02) (0.02 = the drafter's CUDA context and warmup).  Arithmetic, not measured: at the defaults the judge's static pool
is 0.62 x 95.6 = 59.3 GiB = weights + ~2.5 GiB of KV (96 KiB/token: 48 layers x 4 KV heads x 128 x 2 x bf16), i.e.
~26k tokens or ~20 live 1.3k-token requests (SGLang then admits fewer than --max-running); the drafter's 0.18 = 17.2
GiB = weights + ~14 GiB of KV (112 KiB/token) = ~128k tokens.  What is left (~19 GiB) holds both engines' activations.

CPU tests (no SGLang): python3 -m pytest -q code/tests/test_judge_sglang_cli.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CORPUS = os.path.join(CODE, "corpus1b")
PLUGIN = os.path.join(CORPUS, "sglang_plugin")
for p in (os.path.join(CODE, "compat"), CODE, CORPUS):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as C                                                                     # noqa: E402
import research_protocol as P                                                          # noqa: E402
from gen_sglang import TraceReader, TraceError, snapshot_dir, load_tokenizer           # noqa: E402
from drafter_lora import data as D                                                     # noqa: E402

BLOCK = C.BLOCK
EOS_IDS = [int(t) for t in P.EOS_IDS]
CONTEXT_LENGTH = 4096
DRAFTER_OVERHEAD = 0.02          # of the GPU: the drafter's CUDA context + warmup, outside its static pool
DEFAULT_ADAPTER = os.path.join("logs", "c1b", "train", "main_128m", "tok_000128000000")
DEFAULT_DRAFTER = os.path.join("logs", "phase4", "train", "main", "merged")
SUM_KEYS = ("n_judge", "n_draft_fwd", "n_redraft_rounds", "n_redraft_slots", "n_redraft_fwd", "n_loopguard_slots",
            "n_loopguard_changed", "full_fallback", "n_stock")
COUNTERS = ("n_judge", "n_draft_fwd", "n_tentative", "n_remask", "full_commit_eq_draft", "n_redraft_rounds",
            "n_redraft_slots", "n_redraft_fwd", "n_redraft_eq_judge", "n_redraft_eq_old", "n_loopguard_slots",
            "n_loopguard_changed", "full_fallback", "n_stock", "judge_raw_mask_argmax")
SIDE_DIRS = ("judge_side", "judge_trace", "drafter_side", "drafter_trace")
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1 never through an http_proxy


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("decode", "record"), required=True)
    ap.add_argument("--items", default=None, help="decode: a research_protocol split (gsm8k500, ifeval541, "
                                                  "gsm8k_pool:0:200, ...), sharded items[shard::nshards]")
    ap.add_argument("--index", action="append", default=[], help="record: trainset index.jsonl (repeat per wave)")
    ap.add_argument("--pids", default=None, help='record: JSON {"pids": [...]}; sharded sorted(pids)[shard::nshards]')
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu", type=int, required=True,
                    help="this GPU: CUDA_VISIBLE_DEVICES when unset, the drafter port, the gpu<k> output dir")
    ap.add_argument("--judge-adapter", default=DEFAULT_ADAPTER, help="flag-gated judge adapter dir (repo-relative ok)")
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER, help="merged drafter (HF dir, SDAR remote code)")
    ap.add_argument("--redraft-every", type=int, default=1, help="0 = no re-draft, 1 = rd1, 2 = rd2")
    ap.add_argument("--redraft-rate", type=int, default=4)
    ap.add_argument("--redraft-ban", action="store_true", help="the drafter may not re-propose the rejected token")
    ap.add_argument("--loop-guard", type=int, default=12, help="n of the loop guard (0 = off)")
    ap.add_argument("--tau-commit", type=float, default=0.95)
    ap.add_argument("--max-passes", type=int, default=32)
    ap.add_argument("--max-running", type=int, default=64, help="judge engine max_running_requests")
    ap.add_argument("--mem-fraction", type=float, default=0.62, help="judge static memory, fraction of the whole GPU")
    ap.add_argument("--drafter-mem-fraction", type=float, default=0.18,
                    help="drafter static memory, fraction of the whole GPU")
    ap.add_argument("--drafter-port", type=int, default=None, help="default 30100 + gpu")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="decode: default GEN_LEN of the split's family")
    ap.add_argument("--limit", type=int, default=0, help="decode: the first N items before sharding; record: N pids")
    ap.add_argument("--out", default=None, help="default logs/phase5/judge_sglang/<split | record>")
    ap.add_argument("--attention-backend", default=None, help="both engines (triton on sm_120 + CUDA 12.8)")
    ap.add_argument("--chunk", type=int, default=256, help="requests per engine.generate call")
    return ap


def check_args(args) -> None:
    if args.mode == "decode":
        assert args.items and not args.index and not args.pids, "--mode decode takes --items (no --index / --pids)"
    else:
        assert args.index and args.pids and not args.items, "--mode record takes --index and --pids (no --items)"
        assert args.max_new_tokens is None, "--max-new-tokens is decode only (record runs the item's blocks)"
    assert args.redraft_every >= 0 and args.redraft_rate >= 1 and args.max_passes >= 1
    assert args.loop_guard == 0 or args.loop_guard >= 2, "--loop-guard needs n >= 2 (0 = off)"
    if args.redraft_ban:
        assert args.redraft_every > 0, "--redraft-ban needs --redraft-every > 0 (it acts at re-draft time)"
    assert 0 <= args.shard < args.nshards


def judge_yaml_config(args) -> dict:
    """The algorithm config JudgeLoopTraced reads (c1b_judge_loop.judge_config)."""
    return {"block_size": BLOCK, "tau_commit": float(args.tau_commit), "tau_override": None,
            "max_passes": int(args.max_passes), "redraft_every": int(args.redraft_every),
            "redraft_rate": int(args.redraft_rate), "redraft_ban": bool(args.redraft_ban),
            "loop_guard": int(args.loop_guard), "threshold": 0.95, "draft_steps": 8}


def _yaml_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    raise TypeError(f"no yaml for {v!r}")


def write_yaml(path: str, cfg: dict) -> str:
    lines = ["# JudgeLoopTraced config, written by code/judge_round/judge_sglang.py"]
    lines += [f"{k}: {_yaml_value(v)}" for k, v in cfg.items()]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def drafter_port(args) -> int:
    return int(args.drafter_port) if args.drafter_port is not None else 30100 + int(args.gpu)


def judge_sglang_fraction(mem_fraction: float, drafter_mem_fraction: float) -> float:
    """The judge's whole-GPU fraction as SGLang's mem_fraction_static of the memory left free by the drafter."""
    f = float(mem_fraction) / (1.0 - float(drafter_mem_fraction) - DRAFTER_OVERHEAD)
    assert 0.0 < f < 0.97, f"--mem-fraction {mem_fraction} + --drafter-mem-fraction {drafter_mem_fraction}: " \
                           f"the judge would need {f:.3f} of the memory the drafter leaves"
    return f


def repo_path(path: str) -> str:
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(P.ROOT, path))


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def block_end(prompt_length: int, max_new: int) -> int:
    """The end of the last block SGLang runs for this request (blocks are 32-aligned)."""
    return -(-(int(prompt_length) + int(max_new)) // BLOCK) * BLOCK


def decode_side_spec(prompt_ids: Sequence[int], max_new: int, stop_ids: Sequence[int] = EOS_IDS) -> dict:
    return {"prompt_ids": [int(t) for t in prompt_ids], "mode": "decode", "max_new_tokens": int(max_new),
            "stop_ids": [int(t) for t in stop_ids]}


def record_request(item: dict) -> dict:
    """A trainset item (drafter_lora.data.load_item) -> the teacher-forced judge request."""
    pid = str(item["pid"])
    prompt_ids = [int(x) for x in np.asarray(item["prompt_ids"]).tolist()]
    L = int(item["prompt_length"])
    assert len(prompt_ids) == L, (pid, len(prompt_ids), L)
    tokens = np.asarray(item["tokens"])
    nb = int(tokens.shape[0])
    starts = [int(x) for x in np.asarray(item["block_start"]).tolist()]
    assert tokens.shape == (nb, BLOCK) and nb >= 1 and len(starts) == nb, (pid, tokens.shape, len(starts))
    lo0 = (L // BLOCK) * BLOCK
    assert starts == [lo0 + BLOCK * b for b in range(nb)], (pid, starts[:3], lo0)
    tail = L - lo0
    teacher = [[int(t) for t in row] for row in tokens.tolist()]
    assert teacher[0][:tail] == prompt_ids[lo0:L], f"{pid}: block 0 does not start with the prompt tail"
    max_new = BLOCK * nb - tail
    spec = {"prompt_ids": prompt_ids, "mode": "record", "max_new_tokens": max_new, "stop_ids": [],
            "teacher": teacher}
    return {"rid": pid, "prompt_ids": prompt_ids, "spec": spec, "max_new": max_new, "teacher": teacher,
            "meta": {"pid": pid, "domain": str(item.get("domain", ""))}}


def write_side(side_dir: str, rid: str, spec: dict) -> str:
    assert "/" not in rid and not rid.startswith("."), f"rid {rid!r} is not a file name"
    path = os.path.join(side_dir, f"{rid}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(spec, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def sampling_params(mode: str, max_new: int) -> dict:
    if mode == "decode":
        return {"temperature": 0.0, "max_new_tokens": int(max_new), "ignore_eos": False,
                "stop_token_ids": list(EOS_IDS)}
    return {"temperature": 0.0, "max_new_tokens": int(max_new), "ignore_eos": True}


# ---------------------------------------------------------------------------
# the judge trace -> output
# ---------------------------------------------------------------------------
def trim_eos(ids: Sequence[int], eos_ids: Sequence[int] = EOS_IDS) -> List[int]:
    """ids up to and including the first EOS (all of them when there is none)."""
    e = C.first_eos(ids, eos_ids)
    return [int(t) for t in (ids[:e + 1] if e >= 0 else ids)]


def assemble_blocks(rows_by_lo: Dict[int, dict], prompt_length: int) -> List[dict]:
    """One request's trace rows in block order; TraceError unless they are the contiguous blocks from lo0."""
    L = int(prompt_length)
    lo0 = (L // BLOCK) * BLOCK
    tail = L - lo0
    if lo0 not in rows_by_lo:
        raise TraceError(f"no block at lo0={lo0}")
    out, lo = [], lo0
    while lo in rows_by_lo:
        r = rows_by_lo[lo]
        k = len(out)
        if int(r["block"]) != k or int(r["start"]) != (tail if k == 0 else 0):
            raise TraceError(f"block at {lo}: index {r['block']} start {r['start']} (expected {k}, "
                             f"{tail if k == 0 else 0})")
        if len(r["summary"]["tokens_out"]) != BLOCK:
            raise TraceError(f"block at {lo}: {len(r['summary']['tokens_out'])} tokens")
        out.append(r)
        lo += BLOCK
    if len(out) != len(rows_by_lo):
        raise TraceError(f"blocks at {sorted(rows_by_lo)} are not contiguous from {lo0}")
    return out


def generated(blocks: Sequence[dict]) -> List[int]:
    """The closed blocks' generated slots (what SGLang appended to output_ids), in order."""
    return [int(t) for r in blocks for t in r["summary"]["tokens_out"][int(r["start"]):]]


def decode_record(meta: dict, prompt_ids: Sequence[int], blocks: Sequence[dict], decode_fn, cfg: dict,
                  wall_s: float, output_ids: Optional[Sequence[int]] = None, eos_ids: Sequence[int] = EOS_IDS) -> dict:
    gen = generated(blocks)
    if output_ids:
        got = [int(t) for t in output_ids]
        n = min(len(got), len(gen))
        if got[:n] != gen[:n]:
            raise TraceError("trace tokens != the tokens SGLang returned")
    gen_ids = trim_eos(gen, eos_ids)
    rec = {"idx": meta.get("idx"), "split": meta.get("split"), "rid": meta["rid"], "gold": meta.get("gold"),
           "prompt_tokens": len(prompt_ids), "gen_ids": gen_ids, "pred": decode_fn(gen_ids),
           "n_blocks": len(blocks)}
    for k in SUM_KEYS:
        rec[k] = int(sum(int(r["summary"][k]) for r in blocks))
    rec.update({"passes_per_block": [int(r["summary"]["n_judge"]) for r in blocks],
                "stopped_eos": C.first_eos(gen_ids, eos_ids) >= 0, "wall_s": round(float(wall_s), 3),
                "engine": "sglang", "cfg": dict(cfg)})
    return rec


def check_record_blocks(blocks: Sequence[dict], req: dict, output_ids: Optional[Sequence[int]] = None) -> None:
    """Record mode: the request ran exactly the teacher's blocks and stored the teacher's tokens."""
    if len(blocks) != len(req["teacher"]):
        raise TraceError(f"{len(blocks)} blocks traced, the item has {len(req['teacher'])}")
    for b, (r, t) in enumerate(zip(blocks, req["teacher"])):
        if [int(x) for x in r["summary"]["tokens_out"]] != t:
            raise TraceError(f"block {b}: stored tokens != teacher")
        if len(r.get("states") or []) != int(r["summary"]["n_judge"]):
            raise TraceError(f"block {b}: {len(r.get('states') or [])} states for {r['summary']['n_judge']} passes")
    if output_ids:
        gen = generated(blocks)
        got = [int(t) for t in output_ids]
        n = min(len(got), len(gen))
        if got[:n] != gen[:n]:
            raise TraceError("trace tokens != the tokens SGLang returned")


def _nan(xs) -> List[float]:
    return [np.nan if x is None else float(x) for x in xs]


def _arr(rows: list, dtype, width: Optional[int] = None) -> np.ndarray:
    a = np.asarray(rows, dtype=dtype)
    if width is not None and a.size == 0:
        a = a.reshape(0, width)
    return a


def pack_states(entries: Sequence[Tuple[str, Sequence[dict]]], cfg: dict) -> Dict[str, np.ndarray]:
    """[(rid, its trace rows in block order)] -> the flat arrays of states.NNNNN.npz (layout in the docstring)."""
    rid, item_block, blk_pass = [], [0], [0]
    bk = {k: [] for k in ("index", "lo", "start", "tokens", "teacher", "draft", "p_draft", "counters")}
    ps = {k: [] for k in ("pass_no", "forced", "input", "draft", "argmax", "flag", "open", "commit", "tentative",
                          "remask", "p", "q", "rd_on", "rd_holes", "rd_tokens", "rd_input", "rd_p_commit", "rd_n_fwd")}
    items = []
    for r_id, blocks in entries:
        rid.append(str(r_id))
        items.append({"rid": str(r_id), "n_blocks": len(blocks)})
        for row in blocks:
            s = row["summary"]
            bk["index"].append(int(row["block"])); bk["lo"].append(int(row["lo"])); bk["start"].append(int(row["start"]))
            bk["tokens"].append(s["tokens"]); bk["teacher"].append(s["tokens_out"])
            bk["draft"].append(s["draft_tokens"] if s.get("draft_tokens") is not None else [-1] * BLOCK)
            bk["p_draft"].append(_nan(s["p_draft"]) if s.get("p_draft") is not None else [np.nan] * BLOCK)
            bk["counters"].append([int(s[k]) for k in COUNTERS])
            for st in row.get("states") or []:
                rd = st.get("redraft")
                ps["pass_no"].append(int(st["pass"])); ps["forced"].append(bool(st["forced"]))
                for k in ("input", "draft", "argmax", "flag", "open", "commit", "tentative", "remask"):
                    ps[k].append(st[k])
                ps["p"].append(_nan(st["p"])); ps["q"].append(_nan(st["q"]))
                ps["rd_on"].append(rd is not None)
                ps["rd_holes"].append(rd["holes"] if rd else [False] * BLOCK)
                ps["rd_tokens"].append(rd["tokens"] if rd else [-1] * BLOCK)
                ps["rd_input"].append(rd["input"] if rd else [-1] * BLOCK)
                ps["rd_p_commit"].append(_nan(rd["p_commit"]) if rd else [np.nan] * BLOCK)
                ps["rd_n_fwd"].append(int(rd["n_fwd"]) if rd else 0)
            blk_pass.append(len(ps["pass_no"]))
        item_block.append(len(bk["index"]))
    index = {"version": 1, "cfg": dict(cfg), "counters": list(COUNTERS), "items": items}
    W = BLOCK
    return {
        "index": np.array(json.dumps(index)), "rid": np.array(rid, dtype=str),
        "item_block": _arr(item_block, np.int64), "blk_pass": _arr(blk_pass, np.int64),
        "blk_index": _arr(bk["index"], np.int32), "blk_lo": _arr(bk["lo"], np.int32),
        "blk_start": _arr(bk["start"], np.int32), "blk_tokens": _arr(bk["tokens"], np.int32, W),
        "blk_teacher": _arr(bk["teacher"], np.int32, W), "blk_draft": _arr(bk["draft"], np.int32, W),
        "blk_p_draft": _arr(bk["p_draft"], np.float32, W), "blk_counters": _arr(bk["counters"], np.int32, len(COUNTERS)),
        "pass_no": _arr(ps["pass_no"], np.int16), "forced": _arr(ps["forced"], bool),
        "input": _arr(ps["input"], np.int32, W), "draft": _arr(ps["draft"], np.int32, W),
        "argmax": _arr(ps["argmax"], np.int32, W), "flag": _arr(ps["flag"], bool, W),
        "open": _arr(ps["open"], bool, W), "commit": _arr(ps["commit"], bool, W),
        "tentative": _arr(ps["tentative"], bool, W), "remask": _arr(ps["remask"], bool, W),
        "p": _arr(ps["p"], np.float32, W), "q": _arr(ps["q"], np.float32, W),
        "rd_on": _arr(ps["rd_on"], bool), "rd_holes": _arr(ps["rd_holes"], bool, W),
        "rd_tokens": _arr(ps["rd_tokens"], np.int32, W), "rd_input": _arr(ps["rd_input"], np.int32, W),
        "rd_p_commit": _arr(ps["rd_p_commit"], np.float32, W), "rd_n_fwd": _arr(ps["rd_n_fwd"], np.int16)}


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def read_done(gpu_dir: str) -> set:
    """done.txt rids, plus every rid already in records.jsonl (a crash between the two never duplicates a record)."""
    done = set()
    path = os.path.join(gpu_dir, "done.txt")
    if os.path.exists(path):
        done |= set(open(path).read().split())
    rec = os.path.join(gpu_dir, "records.jsonl")
    if os.path.exists(rec):
        for line in open(rec):
            try:
                done.add(str(json.loads(line)["rid"]))
            except (ValueError, KeyError):
                continue                                    # a truncated last line is redone
    return done


def append_lines(path: str, lines: Sequence[str]) -> None:
    if lines:
        with open(path, "a") as fh:
            fh.write("".join(f"{x}\n" for x in lines))
            fh.flush()
            os.fsync(fh.fileno())


def next_states_index(gpu_dir: str) -> int:
    """1 + the highest states.<n>.npz index in gpu_dir (a crash's .tmp.npz is not a chunk)."""
    got = [int(m.group(1)) for m in (re.match(r"states\.(\d+)\.npz$", f) for f in os.listdir(gpu_dir)) if m]
    return max(got) + 1 if got else 0


def prepare_dirs(gpu_dir: str) -> Dict[str, str]:
    """Fresh side / trace dirs (request ids repeat across runs: a crashed run's files must not leak in)."""
    dirs = {}
    for name in SIDE_DIRS:
        d = os.path.join(gpu_dir, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        dirs[name] = d
    return dirs


def check_run_meta(gpu_dir: str, meta: dict) -> None:
    """gpu<k>/run.json: a resume must use the same mode, judge config, adapter and drafter."""
    path = os.path.join(gpu_dir, "run.json")
    if os.path.exists(path):
        old = json.load(open(path))
        assert old == meta, f"{gpu_dir} was written with another configuration:\n  {old}\n  vs {meta}"
    else:
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=1)


# ---------------------------------------------------------------------------
# the two engines
# ---------------------------------------------------------------------------
def drafter_cmd(drafter_dir: str, port: int, mem_fraction: float, attention_backend: Optional[str] = None) -> List[str]:
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", drafter_dir, "--tokenizer-path", drafter_dir,
           "--trust-remote-code", "--skip-tokenizer-init", "--dtype", "bfloat16", "--random-seed", "0",
           "--dllm-algorithm", "DraftTracedV2", "--dllm-algorithm-config", os.path.join(PLUGIN, "draft_s8.yaml"),
           "--disable-radix-cache", "--disable-cuda-graph", "--port", str(int(port)), "--host", "127.0.0.1",
           "--mem-fraction-static", f"{float(mem_fraction):.4f}", "--max-running-requests", "512",
           "--context-length", str(CONTEXT_LENGTH), "--log-level", "warning"]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    return cmd


def drafter_env(dirs: Dict[str, str], gpu: int) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({"C1B_SIDE_DIR": dirs["drafter_side"], "C1B_TRACE_DIR": dirs["drafter_trace"], "C1B_CODE_DIR": CODE,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu))})
    return env


def judge_env(dirs: Dict[str, str], adapter: str, url: str) -> Dict[str, str]:
    return {"C1B_CODE_DIR": CODE, "C1B_JUDGE_ADAPTER": adapter, "C1B_JUDGE_SIDE_DIR": dirs["judge_side"],
            "C1B_JUDGE_TRACE_DIR": dirs["judge_trace"], "C1B_DRAFTER_URL": url,
            "C1B_DRAFTER_SIDE_DIR": dirs["drafter_side"], "C1B_DRAFTER_TRACE_DIR": dirs["drafter_trace"]}


def judge_engine_kwargs(args, yaml_path: str, mem_fraction_static: float) -> dict:
    model = snapshot_dir("b32")
    return dict(model_path=model, tokenizer_path=model, trust_remote_code=True, skip_tokenizer_init=True,
                dtype="bfloat16", random_seed=0, log_level="warning", dllm_algorithm="JudgeLoopTraced",
                dllm_algorithm_config=yaml_path, max_running_requests=int(args.max_running),
                mem_fraction_static=float(mem_fraction_static), context_length=CONTEXT_LENGTH,
                disable_cuda_graph=True, disable_radix_cache=True,
                **({"attention_backend": args.attention_backend} if args.attention_backend else {}))


def wait_health(url: str, proc: subprocess.Popen, log_path: str, timeout: float = 600.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"the drafter server exited with code {proc.returncode}; see {log_path}")
        try:
            with _NO_PROXY.open(url + "/health", timeout=5) as r:
                if getattr(r, "status", 200) == 200:
                    return time.time() - t0
        except Exception:                                   # noqa: BLE001 - not up yet
            pass
        time.sleep(2.0)
    raise RuntimeError(f"the drafter server at {url} is not healthy after {timeout:.0f} s; see {log_path}")


def stop_drafter(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)                 # launch_server's scheduler / detokenizer children too
        proc.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# work lists
# ---------------------------------------------------------------------------
def decode_work(args, done: set) -> List[dict]:
    """This shard's research_protocol items not done yet (items[:limit][shard::nshards], as run_da_decode)."""
    base = P.split_base(args.items)
    items = P.items_for_split(args.items)
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.nshards]
    out = []
    for it in items:
        rid = f"{base}#{int(it['idx'])}"
        if rid not in done:
            out.append({"rid": rid, "idx": int(it["idx"]), "split": it.get("split") or args.items,
                        "gold": it.get("gold"), "prompt": it["prompt"]})
    return out


def record_work(args, done: set) -> Tuple[List[str], Dict[str, dict]]:
    """This shard's trainset pids (sorted(pids)[shard::nshards], as draft_sglang_items) not done yet."""
    by_pid = {str(e["pid"]): e for e in D.load_entries(args.index)}
    mine = sorted({str(p) for p in D.load_holdout(args.pids)})[args.shard::args.nshards]
    missing = [p for p in mine if p not in by_pid]
    assert not missing, f"{len(missing)} pids are in no --index (first: {missing[:3]})"
    todo = [p for p in mine if p not in done]
    return (todo[:args.limit] if args.limit else todo), by_pid


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    check_args(args)
    out_root = P.safe_out(args.out or os.path.join("logs", "phase5", "judge_sglang",
                                                   P.split_base(args.items) if args.mode == "decode" else "record"))
    gpu_dir = os.path.join(out_root, f"gpu{args.gpu}")
    os.makedirs(gpu_dir, exist_ok=True)
    cfg = judge_yaml_config(args)
    adapter, drafter = repo_path(args.judge_adapter), repo_path(args.drafter_dir)
    assert os.path.isfile(os.path.join(adapter, "adapter_config.json")), f"no judge adapter in {adapter}"
    assert os.path.isfile(os.path.join(drafter, "config.json")), f"no drafter checkpoint in {drafter}"
    max_new_decode = (args.max_new_tokens or int(P.GEN_LEN[P.split_base(args.items)])) if args.mode == "decode" else None
    check_run_meta(gpu_dir, {"mode": args.mode, "cfg": cfg, "judge_adapter": adapter, "drafter_dir": drafter,
                             "max_new_tokens": max_new_decode})

    done = read_done(gpu_dir)
    if args.mode == "decode":
        todo, by_pid = decode_work(args, done), None
    else:
        todo, by_pid = record_work(args, done)
    print(f"[judge-sglang] gpu {args.gpu} {args.mode}: {len(todo)} requests to run ({len(done)} done) -> {gpu_dir}",
          flush=True)
    if not todo:
        return

    dirs = prepare_dirs(gpu_dir)
    yaml_path = write_yaml(os.path.join(out_root, f"judge_{args.gpu}.yaml"), cfg)
    tok = load_tokenizer("b32")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if os.environ["CUDA_VISIBLE_DEVICES"] != str(args.gpu):
        print(f"[judge-sglang] note: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (--gpu {args.gpu} "
              f"names the port and the output dir); both engines use it", flush=True)
    port = drafter_port(args)
    url = f"http://127.0.0.1:{port}"
    f_judge = judge_sglang_fraction(args.mem_fraction, args.drafter_mem_fraction)
    print(f"[judge-sglang] mem_fraction_static: drafter {args.drafter_mem_fraction} (empty GPU), judge {f_judge:.4f} "
          f"(= {args.mem_fraction} / (1 - {args.drafter_mem_fraction} - {DRAFTER_OVERHEAD}): SGLang counts the "
          f"memory free at its start)", flush=True)

    log_path = os.path.join(gpu_dir, "drafter_server.log")
    proc, engine = None, None
    t_start = time.time()
    n_items = n_blocks = n_bad = n_long = 0
    try:
        with open(log_path, "a") as log:
            proc = subprocess.Popen(drafter_cmd(drafter, port, args.drafter_mem_fraction, args.attention_backend),
                                    env=drafter_env(dirs, args.gpu), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        up = wait_health(url, proc, log_path)
        print(f"[judge-sglang] drafter server up at {url} after {up:.0f} s", flush=True)

        os.environ.update(judge_env(dirs, adapter, url))          # inherited by the judge's scheduler process
        import sglang as sgl
        engine = sgl.Engine(**judge_engine_kwargs(args, yaml_path, f_judge))
        reader = TraceReader(dirs["judge_trace"])
        ci = next_states_index(gpu_dir)
        for c0 in range(0, len(todo), args.chunk):
            reqs = []
            for w in todo[c0:c0 + args.chunk]:
                if args.mode == "decode":
                    ids = P.encode_prompt(tok, w["prompt"])[0].tolist()
                    q = {"rid": w["rid"], "prompt_ids": ids, "max_new": max_new_decode,
                         "spec": decode_side_spec(ids, max_new_decode), "meta": w}
                else:
                    q = record_request(D.load_item(by_pid[w]))
                if block_end(len(q["prompt_ids"]), q["max_new"]) > CONTEXT_LENGTH:
                    n_long += 1
                    print(f"[judge-sglang] skip {q['rid']}: {len(q['prompt_ids'])} + {q['max_new']} tokens exceed "
                          f"the context {CONTEXT_LENGTH}", flush=True)
                    continue
                write_side(dirs["judge_side"], q["rid"], q["spec"])
                reqs.append(q)
            if not reqs:
                continue
            t0 = time.time()
            outs = engine.generate(input_ids=[q["prompt_ids"] for q in reqs],
                                   sampling_params=[sampling_params(args.mode, q["max_new"]) for q in reqs],
                                   rid=[q["rid"] for q in reqs])
            dt = time.time() - t0
            reader.poll()
            good, recs, entries, nb_chunk = [], [], [], 0
            for q, o in zip(reqs, outs):
                got = o.get("output_ids") if isinstance(o, dict) else None
                try:
                    blocks = assemble_blocks(reader.take(q["rid"]), len(q["prompt_ids"]))
                    if args.mode == "decode":
                        recs.append(decode_record(q["meta"], q["prompt_ids"], blocks,
                                                  lambda ids: tok.decode(ids, skip_special_tokens=True),
                                                  cfg, dt / len(reqs), got))
                    else:
                        check_record_blocks(blocks, q, got)
                        entries.append((q["rid"], blocks))
                except TraceError as e:
                    n_bad += 1
                    print(f"[judge-sglang] {q['rid']}: incomplete ({e}); left for a rerun", flush=True)
                    continue
                good.append(q["rid"])
                nb_chunk += len(blocks)
            stale = [f for f in os.listdir(dirs["judge_side"]) if f.endswith(".json")]
            if stale:
                print(f"[judge-sglang] warning: {len(stale)} judge side files never read (first {stale[:3]})", flush=True)
                for f in stale:
                    os.remove(os.path.join(dirs["judge_side"], f))
            if args.mode == "decode":
                append_lines(os.path.join(gpu_dir, "records.jsonl"), [json.dumps(r) for r in recs])
            elif entries:
                path = os.path.join(gpu_dir, f"states.{ci:05d}.npz")
                np.savez_compressed(path + ".tmp.npz", **pack_states(entries, cfg))
                os.replace(path + ".tmp.npz", path)
                ci += 1
            append_lines(os.path.join(gpu_dir, "done.txt"), good)
            n_items += len(good)
            n_blocks += nb_chunk
            el = time.time() - t_start
            print(f"[judge-sglang] gpu {args.gpu}: {n_items} items / {n_blocks} blocks ({n_bad} incomplete, "
                  f"{n_long} too long) in {el / 3600:.2f} h = {n_blocks / max(el, 1e-9):.2f} blocks/s "
                  f"(progress only, not a speed measurement)", flush=True)
    finally:
        if engine is not None:
            engine.shutdown()
        stop_drafter(proc)
    print(f"[judge-sglang] gpu {args.gpu}: done, {n_items} items in {(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
