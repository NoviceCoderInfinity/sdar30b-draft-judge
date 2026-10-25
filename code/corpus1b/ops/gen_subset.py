#!/usr/bin/env python
"""gen_sglang.py on a RANDOM subset of a domain's remaining wave-B prompts, without editing shared code: points
common.PROMPT_DIR at a folder holding the subset list (drawn by sha256('c1b-sample-<d>-2|'+pid) order, unbiased by
source, unlike a --stop-at time box), then calls the unmodified gen_sglang.main(). Records/done markers go to the
normal logs/c1b/gen/waveB/gpu<k>/ folders, so drafting/conversion treat them like every other wave-B record."""
import os, sys
ROOT = "/workspace/dmge"
for p in (os.path.join(ROOT, "code", "corpus1b"), os.path.join(ROOT, "code")):
    if p not in sys.path: sys.path.insert(0, p)
import common as C
C.PROMPT_DIR = os.environ["SUBSET_PROMPT_DIR"]
import gen_sglang
gen_sglang.main(sys.argv[1:])
