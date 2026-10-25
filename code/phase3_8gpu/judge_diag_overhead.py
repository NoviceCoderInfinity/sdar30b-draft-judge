#!/usr/bin/env python
"""Cost of --judge-diag inside the timed decode, measured on this machine.

The flag's only work inside the timed span is one call of da_decode.copy_diag per judge pass (judge_full_block:
`if diag: copy_diag(bt, cur, r["argmax"], open_, flag, mask_id)`); everything else it touches is record writing,
outside the timing. This script times that exact function -- taken verbatim from code/draftalign/da_decode.py via
the AST, so it needs no transformers -- on block-sized CUDA tensors, the way the loop calls it (int() reads included).

    CUDA_VISIBLE_DEVICES=<k> /venv/p2/bin/python code/phase3_8gpu/judge_diag_overhead.py <out.json>

Writes microseconds per call: 7 repeats of 20,000 calls after a 2,000-call warm-up, synchronize + perf_counter around
each repeat (the same clock the decode uses). The share of decode time is then sum(n_judge) x t_call / sum(seconds).
"""
import ast
import json
import os
import statistics
import sys
import time
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "code", "draftalign", "da_decode.py")


def load_copy_diag():
    tree = ast.parse(open(SRC).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "copy_diag")
    ns = {"torch": torch, "DABlock": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC + ":copy_diag", "exec"), ns)
    return ns["copy_diag"], ast.get_source_segment(open(SRC).read(), fn)


def main(out):
    copy_diag, source = load_copy_diag()
    dev = torch.device("cuda")
    bl, mask_id = 32, 151669                            # block length 32; any id works as [MASK] for the timing
    g = torch.Generator().manual_seed(0)
    inp = torch.randint(0, 150000, (bl,), generator=g)
    inp[::5] = mask_id                                  # some [MASK] slots, as after re-masking
    argmax = inp.clone()
    argmax[1::3] = torch.randint(0, 150000, (len(argmax[1::3]),), generator=g)
    open_ = torch.ones(bl, dtype=torch.bool)
    open_[:8] = False                                   # a few committed slots
    inp, argmax, open_ = inp.to(dev), argmax.to(dev), open_.to(dev)
    flag = open_ & (inp != mask_id)
    bt = types.SimpleNamespace(copy_flag_n=0, copy_flag_hit=0, copy_mask_n=0, copy_mask_hit=0)

    def run(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            copy_diag(bt, inp, argmax, open_, flag, mask_id)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    def run_empty(n):                                   # the bare Python loop + call, for reference
        f = lambda *a: None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            f(bt, inp, argmax, open_, flag, mask_id)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    run(2000)
    reps = [run(20000) for _ in range(7)]
    empty = statistics.median(run_empty(20000) for _ in range(3))
    res = {"gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch": torch.__version__, "calls_per_repeat": 20000, "repeats_us": [1e6 * r for r in reps],
           "median_us": 1e6 * statistics.median(reps), "max_us": 1e6 * max(reps), "empty_call_us": 1e6 * empty,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "function_source": source}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[diag-cost] GPU {res['cuda_visible_devices']}: median {res['median_us']:.1f} us/call "
          f"(max {res['max_us']:.1f}, repeats {', '.join(f'{x:.1f}' for x in res['repeats_us'])}); empty call {res['empty_call_us']:.2f} us")


if __name__ == "__main__":
    main(sys.argv[1])
