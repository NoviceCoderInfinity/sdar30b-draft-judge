#!/usr/bin/env python
"""Install the corpus's two SGLang dLLM algorithms into the SGLang of THIS interpreter.

    /venv/sgl/bin/python code/corpus1b/sglang_plugin/install.py            # install + verify
    /venv/sgl/bin/python code/corpus1b/sglang_plugin/install.py --check    # verify only

Copies c1b_traced.py (LowConfidenceTraced, the 30B's stock decode + commit-pass trace) and
c1b_static_draft.py (StaticDraftTraced, the 1.7B's s8 teacher-forced drafts + confidence) into
<site-packages>/sglang/srt/dllm/algorithm/, where SGLang discovers any module that defines `Algorithm`.
No SGLang file is edited. The algorithms copy SGLang 0.5.10.post1's LowConfidence loop, so any other
version is refused (pass --force only after re-diffing low_confidence.py against c1b_traced.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

PINNED = "0.5.10.post1"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("c1b_traced.py", "c1b_static_draft.py")
NAMES = ("LowConfidenceTraced", "StaticDraftTraced")


ROPE_OLD = """    module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
    module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)"""
ROPE_NEW = """    if out_loc.dtype != positions.dtype:  # c1b patch (user-approved): dLLM+triton pass int64/int32 index tensors
        out_loc = out_loc.to(positions.dtype)
""" + ROPE_OLD


def patch_fused_rope(pkg: str) -> None:
    """The one SGLang edit (approved 2026-09-28 on the Blackwell box): with the triton attention backend (needed on
    sm_120 with a CUDA 12.8 toolkit: FlashInfer's JIT wants >= 12.9 there, and dLLM + CUDA graphs forces FlashInfer),
    the fused RoPE + KV-store kernel receives int64 positions with int32 cache locations and refuses them. Casting
    the cache locations (indices < 2^31) to the positions' dtype is value-preserving. A backup is kept as .orig."""
    p = os.path.join(pkg, "jit_kernel", "rope.py")
    s = open(p).read()
    if "c1b patch" in s:
        print("fused rope patch: already applied")
        return
    assert s.count(ROPE_OLD) == 1, "sglang/jit_kernel/rope.py changed: re-check the fused rope patch"
    if not os.path.exists(p + ".orig"):
        shutil.copyfile(p, p + ".orig")
    open(p, "w").write(s.replace(ROPE_OLD, ROPE_NEW))
    print(f"fused rope patch applied to {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only verify that both algorithms are discoverable")
    ap.add_argument("--force", action="store_true", help="install into a different sglang version (see docstring)")
    args = ap.parse_args(argv)

    import sglang
    ver = getattr(sglang, "__version__", "?")
    if ver != PINNED and not args.force:
        print(f"refusing: sglang {ver} installed, the plugins copy the loop of {PINNED}", file=sys.stderr)
        return 2
    dst = os.path.join(os.path.dirname(sglang.__file__), "srt", "dllm", "algorithm")
    assert os.path.isfile(os.path.join(dst, "low_confidence.py")), f"no dLLM algorithm package at {dst}"
    if not args.check:
        for f in FILES:
            shutil.copyfile(os.path.join(HERE, f), os.path.join(dst, f))
            print(f"installed {f} -> {dst}")
        patch_fused_rope(os.path.dirname(sglang.__file__))

    from sglang.srt.dllm.algorithm import import_algorithms
    found = import_algorithms()
    missing = [n for n in NAMES if n not in found]
    if missing:
        print(f"NOT discoverable: {missing} (import errors are logged by sglang as warnings)", file=sys.stderr)
        return 1
    print(f"ok: sglang {ver}; algorithms {sorted(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
