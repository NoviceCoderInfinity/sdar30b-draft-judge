#!/usr/bin/env python
"""AIME with a longer generation budget, WITHOUT editing shared code: sets research_protocol.GEN_LEN['aime'] (the same
dict run_da_decode imports) to $AIME_GEN_LEN (default 4096), then calls the unmodified run_da_decode.main() with
the same arguments. User request 2026-09-29 ~18:40 IST: 4,096 tokens, stock + judge 28.5M. AIME only."""
import os, sys
ROOT = "/workspace/dmge"
for p in (os.path.join(ROOT, "code"), os.path.join(ROOT, "code", "draftalign")):
    if p not in sys.path: sys.path.insert(0, p)
import research_protocol as P
assert "--items" in sys.argv and sys.argv[sys.argv.index("--items") + 1] == "aime", "AIME only"
P.GEN_LEN["aime"] = int(os.environ.get("AIME_GEN_LEN", "4096"))
print(f"[aime-long] GEN_LEN['aime'] = {P.GEN_LEN['aime']}", flush=True)
import run_da_decode
run_da_decode.main(sys.argv[1:])
