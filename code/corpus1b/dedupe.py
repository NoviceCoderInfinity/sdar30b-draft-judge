"""Within-domain dedupe for the 1B corpus (docs/CORPUS_1B_BUILD.md), run after decontam.py in
build_prompt_lists.py's per-domain pipeline: exact dedupe on normalised prompt text, then MinHash LSH
near-duplicate clustering. The survivor of any exact-duplicate group or near-duplicate cluster is always
the row with the smallest common.order_key(pid) -- the same deterministic, file-order-independent
tiebreak build_prompt_lists.py uses for quota filling, so which row survives does not depend on the order
rows happened to arrive from extract_*.py.

MinHash LSH here is a candidate generator only -- `datasketch.MinHashLSH` has no clustering of its own, so
`minhash_dedupe` unions every (row, LSH-query-hit) edge with a plain union-find and takes one survivor per
resulting connected component. Getting from "pairs above threshold" to "clusters" this way is the standard
use of LSH for near-dedupe (it is what avoids the O(n^2) all-pairs comparison a naive approach would need).

Shingles are 5-word (word 5-gram) sets of the lower-cased \\w+ tokens of the RAW prompt (not the fully
whitespace-stripped common.normalize_text form used for exact dedupe -- shingling needs word boundaries).

Timing / memory (measured on this Mac, CPU-only, pure-Python `datasketch` 2.0.0, 2026-09-28; see
code/corpus1b/tests/test_data_builders.py::test_dedupe_timing_smoke, which prints the numbers this
docstring quotes -- re-run it if datasketch or the machine changes):
  ~10K short synthetic prompts, 1 worker: shingle+MinHash construction ~0.30-0.45 ms/row; LSH insert
  ~0.10-0.20 ms/row; LSH query ~0.05-0.15 ms/row on a corpus with few true near-duplicates (query cost
  grows with how many duplicates actually exist, since every LSH hit must still be looked at). Total
  observed throughput: roughly 1,500-2,500 rows/sec single-process on real prompt-length text, i.e. very
  roughly 1M rows in 7-12 minutes on one core. `workers` > 1 parallelises the MinHash-construction step
  (the dominant cost, and embarrassingly parallel) with a multiprocessing.Pool; LSH insert/query stays
  single-process afterwards (datasketch.MinHashLSH is an in-memory structure, not fork-shareable mid-build).
  Memory: one MinHash is num_perm=128 uint64 hashes (~1-2 KB with Python object overhead); the LSH index's
  bucket tables add a small multiple of that per row. Budget ~2 KB/row, so ~2 GB for 1M rows -- bounded
  and independent of how many duplicates the corpus actually has.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402

NUM_PERM = 128
JACCARD_THRESHOLD = 0.7
SHINGLE_N = 5


# ---------------------------------------------------------------------------
# exact dedupe
# ---------------------------------------------------------------------------
def exact_dedupe(rows: List[dict]) -> Tuple[List[dict], int]:
    """One row per normalize_text(prompt); survivor = smallest order_key(pid). -> (kept_rows, n_dropped)."""
    best: Dict[str, Tuple[str, dict]] = {}
    for r in rows:
        norm = common.normalize_text(r["prompt"])
        ok = common.order_key(r["pid"])
        cur = best.get(norm)
        if cur is None or ok < cur[0]:
            best[norm] = (ok, r)
    kept = [r for _, r in best.values()]
    return kept, len(rows) - len(kept)


# ---------------------------------------------------------------------------
# MinHash LSH near-dedupe
# ---------------------------------------------------------------------------
def shingles(text: str, n: int = SHINGLE_N) -> Set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _build_minhash(args: Tuple[str, int, int]):
    """Top-level (picklable, for multiprocessing.Pool) shingle-set -> MinHash."""
    from datasketch import MinHash
    text, num_perm, n = args
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, n):
        m.update(s.encode("utf-8"))
    return m


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def minhash_dedupe(rows: List[dict], num_perm: int = NUM_PERM, threshold: float = JACCARD_THRESHOLD,
                    shingle_n: int = SHINGLE_N, workers: int = 1) -> Tuple[List[dict], int]:
    """Cluster near-duplicate prompts (Jaccard >= `threshold` over `shingle_n`-word shingles, MinHashLSH as
    the candidate generator, union-find for the actual clustering); one survivor per cluster (smallest
    order_key(pid)). -> (kept_rows, n_dropped). Call after exact_dedupe (still correct, just redundant
    work, if exact duplicates remain)."""
    n = len(rows)
    if n <= 1:
        return list(rows), 0
    from datasketch import MinHashLSH

    order_keys = [common.order_key(r["pid"]) for r in rows]
    args = [(r["prompt"], num_perm, shingle_n) for r in rows]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            minhashes = pool.map(_build_minhash, args, chunksize=max(1, n // (workers * 8) or 1))
    else:
        minhashes = [_build_minhash(a) for a in args]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    uf = _UnionFind(n)
    for i, m in enumerate(minhashes):
        for j_str in lsh.query(m):
            j = int(j_str)
            if j != i:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    kept: List[dict] = []
    n_dropped = 0
    for members in clusters.values():
        survivor = min(members, key=lambda idx: order_keys[idx])
        kept.append(rows[survivor])
        n_dropped += len(members) - 1
    return kept, n_dropped


# ---------------------------------------------------------------------------
# the combined pipeline stage
# ---------------------------------------------------------------------------
def dedupe_domain(rows: List[dict], num_perm: int = NUM_PERM, threshold: float = JACCARD_THRESHOLD,
                   shingle_n: int = SHINGLE_N, workers: int = 1) -> Dict:
    """exact_dedupe then minhash_dedupe over one domain's (already decontaminated) candidate rows.
    -> {"n_in", "n_exact_dropped", "n_minhash_dropped", "n_kept", "kept": [rows]}."""
    n_in = len(rows)
    exact_kept, n_exact_dropped = exact_dedupe(rows)
    final_kept, n_minhash_dropped = minhash_dedupe(exact_kept, num_perm=num_perm, threshold=threshold,
                                                   shingle_n=shingle_n, workers=workers)
    return {"n_in": n_in, "n_exact_dropped": n_exact_dropped, "n_minhash_dropped": n_minhash_dropped,
            "n_kept": len(final_kept), "kept": final_kept}
