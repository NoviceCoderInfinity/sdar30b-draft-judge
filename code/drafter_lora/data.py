"""The 1B corpus (waves A + B) as drafter training data: items, the held-out split, training states, two-stream batches.

ITEMS.  `code/corpus1b/to_trainset.py` writes one npz per kept item (format of draftalign/build_train_set.py):
    prompt_ids [L]; tokens [nb, 32] = the 30B's committed tokens T (block 0 starts with the ragged prompt tail);
    valid [nb, 32] (False on the prompt tail and after the first EOS); committed [nb, P, 32] (vanilla pass p committed
    slot j); n_pass [nb]; block_start [nb]; draft [nb, 32] + draft_conf [nb, 32] (the UNTRAINED drafter's s8 first
    draft from the clean prefix, conf = round(255 p); -1 / 0 on the prompt tail); has_draft [nb]; usable [nb]; domain.
Both waves number their items from idx 0, so (split, idx) is NOT unique across waves; everything here keys on `pid`.

HELD-OUT.  3,000 items stratified by domain (largest-remainder quotas proportional to each domain's size), drawn with
numpy default_rng(seed) over each domain's pids in sorted order.  The pid list is saved next to the run and every
evaluation script reads it back, so no held-out item is ever trained on.

TRAINING STATE, one per usable block, drawn fresh per (seed, pid, block, epoch):
    p_first (0.4)  FIRST-DRAFT state: k ~ U{0, 4, ..., 28} (capped at n_valid - 1; 0 when the block has no stored
                   draft); the k valid slots with the highest stored draft_conf show the STORED draft tokens (they may
                   be wrong); every other valid slot is [MASK].  Copies the drafter's own s8 first draft mid-way.
    1 - p_first    RE-DRAFT state: p ~ U{0, ..., K_b - 1} (K_b = the block's vanilla passes); the valid slots the
                   vanilla 30B committed at passes < p show the TRUE tokens; every other valid slot is [MASK].
                   Copies an rd1 / rd2 re-draft, where most drafter calls go.
    Invalid slots are never loss rows and hold the true tokens, exactly as draftalign/dataset.py's StateSampler does
    (`--post-eos mask` instead shows [MASK] after the EOS, as an un-guided decode would; off by default).
    Loss rows H = the valid [MASK] slots.  The prefix (prompt + earlier blocks) is always the 30B's clean tokens.

TWO-STREAM LAYOUT (the official SDAR / BD3-LM training layout; one forward per item covers all its blocks):
    [ clean stream: prompt + every usable answer block, clean | noisy stream: every usable block in its state ]
    clean row (absolute block c)  -> clean keys of blocks <= c                          (block-causal)
    noisy row (absolute block c)  -> clean keys of blocks <  c  +  its own noisy block  (bidirectional inside)
    position ids: a noisy block takes its clean positions.
A noisy block therefore sees exactly what the inference drafter sees: the KV of the committed prefix (computed by the
same, adapted, drafter) plus its own partially masked block.  Gradients flow through both streams, because the LoRA
also changes the prefix's keys and values at inference.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from draftalign.dataset import _seed

MASK_ID = 151669
BLOCK = 32
IGNORE = -100
KIND_FIRST, KIND_REDRAFT = 0, 1
FIRST_KS = tuple(range(0, 29, 4))                      # 0, 4, ..., 28


# ---------------------------------------------------------------------------
# index + held-out split
# ---------------------------------------------------------------------------
def load_entries(index_paths: Sequence[str]) -> List[dict]:
    """Every index row of every index, `_path` resolved to the item's npz; pids must be unique across indexes."""
    out: List[dict] = []
    seen: Dict[str, str] = {}
    for ip in index_paths:
        root = os.path.dirname(os.path.abspath(ip))
        with open(ip) as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                pid = str(e["pid"])
                assert pid not in seen, f"pid {pid} appears in {seen[pid]} and {ip}"
                seen[pid] = ip
                e["_path"] = e["file"] if os.path.isabs(e["file"]) else os.path.join(root, e["file"])
                e["domain"] = str(e.get("domain") or "unknown")
                out.append(e)
    return out


def quotas(sizes: Dict[str, int], n: int) -> Dict[str, int]:
    """Largest-remainder allocation of `n` proportional to `sizes` (ties broken by domain name)."""
    total = sum(sizes.values())
    assert 0 <= n <= total, (n, total)
    raw = {d: n * s / total for d, s in sizes.items()}
    q = {d: int(math.floor(v)) for d, v in raw.items()}
    rest = n - sum(q.values())
    for d in sorted(sizes, key=lambda d: (-(raw[d] - q[d]), d))[:rest]:
        q[d] += 1
    return q


def stratified_holdout(entries: Sequence[dict], n: int = 3000, seed: int = 0) -> Tuple[List[str], Dict[str, int]]:
    """`n` held-out pids, stratified by domain, numpy default_rng(seed).  Returns (sorted pids, per-domain counts)."""
    by: Dict[str, List[str]] = {}
    for e in entries:
        by.setdefault(e["domain"], []).append(str(e["pid"]))
    q = quotas({d: len(v) for d, v in by.items()}, n)
    rng = np.random.default_rng(seed)
    hold: List[str] = []
    for d in sorted(by):
        pids = sorted(by[d])
        pick = rng.choice(len(pids), size=q[d], replace=False) if q[d] else []
        hold.extend(pids[int(i)] for i in pick)
    return sorted(hold), {d: q[d] for d in sorted(q)}


def save_holdout(path: str, pids: Sequence[str], counts: Dict[str, int], meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"pids": list(pids), "per_domain": counts, **meta}, fh, indent=1)


def load_holdout(path: str) -> List[str]:
    return list(json.load(open(path))["pids"])


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------
def load_item(entry: dict) -> dict:
    d = dict(np.load(entry["_path"], allow_pickle=False).items())
    d["pid"] = str(entry["pid"])
    d["domain"] = str(entry.get("domain") or d.get("domain", "unknown"))
    return d


def usable_blocks(entry_or_item: dict, max_seq: Optional[int] = None) -> List[int]:
    """Usable blocks (build_train_set's `usable`), optionally also within `max_seq` (block end <= max_seq)."""
    usable = [bool(v) for v in np.asarray(entry_or_item["usable"]).tolist()]
    starts = [int(v) for v in np.asarray(entry_or_item["block_start"]).tolist()]
    return [b for b, u in enumerate(usable) if u and (max_seq is None or starts[b] + BLOCK <= int(max_seq))]


def two_stream_length(entry: dict, max_seq: Optional[int] = None) -> int:
    """Clean length (through the last usable block) + 32 per usable block: the forward's sequence length."""
    bs = usable_blocks(entry, max_seq)
    if not bs:
        return 0
    starts = entry["block_start"]
    return int(starts[bs[-1]]) + BLOCK + BLOCK * len(bs)


def clean_sequence(item: dict) -> np.ndarray:
    """Prompt (whole blocks) + every block's 30B tokens: the item on its absolute 32-grid."""
    lo0 = int(np.asarray(item["block_start"])[0])
    return np.concatenate([np.asarray(item["prompt_ids"], np.int64)[:lo0],
                           np.asarray(item["tokens"], np.int64).reshape(-1)])


def post_eos_slots(item: dict, b: int) -> np.ndarray:
    """Invalid slots of block b that are NOT the prompt tail (i.e. after the answer's first EOS)."""
    valid = np.asarray(item["valid"][b], bool)
    tail = np.zeros(BLOCK, bool)
    if b == 0:
        pl = int(np.asarray(item["prompt_ids"]).shape[0])
        tail[: pl - int(np.asarray(item["block_start"])[0])] = True
    return ~valid & ~tail


@dataclass
class BlockState:
    ids: np.ndarray        # [32] int64, the block as the drafter sees it
    hidden: np.ndarray     # [32] bool, loss rows (valid [MASK] slots)
    kind: int              # KIND_FIRST | KIND_REDRAFT
    k_or_p: int            # revealed draft slots (first) or the vanilla pass index (re-draft)


def block_state(item: dict, b: int, rng: np.random.Generator, *, p_first: float = 0.4,
                mask_id: int = MASK_ID, post_eos: str = "truth") -> BlockState:
    """One training state of block `b` (see the module docstring)."""
    assert post_eos in ("truth", "mask"), post_eos
    T = np.asarray(item["tokens"][b], np.int64)
    valid = np.asarray(item["valid"][b], bool)
    n_valid = int(valid.sum())
    assert n_valid > 0, f"{item.get('pid')} block {b} has no valid slot"
    reveal = np.zeros(BLOCK, bool)
    ids = T.copy()                                     # invalid slots hold the truth (StateSampler)
    if rng.random() < p_first:
        kind = KIND_FIRST
        k = int(rng.choice(FIRST_KS))
        draft = np.asarray(item["draft"][b], np.int64)
        has = bool(np.asarray(item["has_draft"])[b])
        k = min(k, n_valid - 1) if has else 0
        if k > 0:
            conf = np.asarray(item["draft_conf"][b], np.int64)
            cand = np.flatnonzero(valid & (draft >= 0))
            order = cand[np.lexsort((cand, -conf[cand]))]       # highest confidence first, ties by slot
            top = order[:k]
            reveal[top] = True
            ids[top] = draft[top]
        val = k
    else:
        kind = KIND_REDRAFT
        K = max(1, int(np.asarray(item["n_pass"])[b]))
        p = int(rng.integers(0, K))
        if p > 0:
            reveal = np.asarray(item["committed"][b, :p], bool).any(0) & valid
        if not bool((valid & ~reveal).any()):          # never a finished block (cannot happen with p < K_b)
            reveal[:] = False
            p = 0
        val = p
    hidden = valid & ~reveal
    ids[hidden] = mask_id
    if post_eos == "mask":
        ids[post_eos_slots(item, b)] = mask_id
    return BlockState(ids=ids, hidden=hidden, kind=kind, k_or_p=val)


@dataclass
class ItemExample:
    pid: str
    domain: str
    ids: torch.Tensor        # [L] long: clean stream then the noisy blocks
    pos: torch.Tensor        # [L] long: position ids
    stream: torch.Tensor     # [L] int8: 0 clean, 1 noisy
    absblk: torch.Tensor     # [L] long: absolute block index of each position
    nblk: torch.Tensor       # [L] long: noisy block ordinal (-1 on the clean stream)
    rows: torch.Tensor       # [L] bool: loss rows (hidden noisy slots)
    targets: torch.Tensor    # [L] long: T at loss rows, IGNORE elsewhere
    kinds: torch.Tensor      # [L] int8: the state kind at loss rows (-1 elsewhere)
    n_clean: int
    blocks: List[int]


def item_example(item: dict, *, seed: int, epoch: int = 0, p_first: float = 0.4, max_seq: Optional[int] = None,
                 mask_id: int = MASK_ID, post_eos: str = "truth") -> Optional[ItemExample]:
    """The two-stream example of one item: every usable block in a fresh training state."""
    bs = usable_blocks(item, max_seq)
    if not bs:
        return None
    starts = np.asarray(item["block_start"], np.int64)
    n_clean = int(starts[bs[-1]]) + BLOCK
    clean = clean_sequence(item)[:n_clean]
    L = n_clean + BLOCK * len(bs)
    ids = np.empty(L, np.int64); pos = np.empty(L, np.int64)
    stream = np.zeros(L, np.int8); absblk = np.empty(L, np.int64); nblk = np.full(L, -1, np.int64)
    rows = np.zeros(L, bool); targets = np.full(L, IGNORE, np.int64); kinds = np.full(L, -1, np.int8)
    ids[:n_clean] = clean
    pos[:n_clean] = np.arange(n_clean)
    absblk[:n_clean] = np.arange(n_clean) // BLOCK
    for j, b in enumerate(bs):
        rng = np.random.default_rng(_seed(seed, "drafter_lora", str(item["pid"]), b, epoch))
        st = block_state(item, b, rng, p_first=p_first, mask_id=mask_id, post_eos=post_eos)
        sl = slice(n_clean + BLOCK * j, n_clean + BLOCK * (j + 1))
        lo = int(starts[b])
        ids[sl] = st.ids
        pos[sl] = lo + np.arange(BLOCK)
        stream[sl] = 1
        absblk[sl] = lo // BLOCK
        nblk[sl] = j
        rows[sl] = st.hidden
        T = np.asarray(item["tokens"][b], np.int64)
        targets[sl] = np.where(st.hidden, T, IGNORE)
        kinds[sl] = np.where(st.hidden, st.kind, -1)
    assert not np.any(targets[rows] == mask_id), "a target is the [MASK] token"
    t = torch.from_numpy
    return ItemExample(pid=str(item["pid"]), domain=str(item.get("domain", "")), ids=t(ids), pos=t(pos),
                       stream=t(stream), absblk=t(absblk), nblk=t(nblk), rows=t(rows), targets=t(targets),
                       kinds=t(kinds), n_clean=n_clean, blocks=list(bs))


def attention_mask(ex: ItemExample) -> torch.Tensor:
    """[L, L] bool, True = attend (query rows x key columns); see the module docstring."""
    qs, ks = ex.stream[:, None], ex.stream[None, :]
    qb, kb = ex.absblk[:, None], ex.absblk[None, :]
    clean_key = ks == 0
    m = clean_key & (((qs == 0) & (kb <= qb)) | ((qs == 1) & (kb < qb)))
    m |= (qs == 1) & (ks == 1) & (ex.nblk[:, None] == ex.nblk[None, :])
    return m


@dataclass
class Batch:
    ids: torch.Tensor        # [B, L]
    pos: torch.Tensor        # [B, L]
    attn: torch.Tensor       # [B, 1, L, L] bool
    rows: torch.Tensor       # [B, L] bool
    targets: torch.Tensor    # [B, L]
    kinds: torch.Tensor      # [B, L]
    pids: List[str]
    domains: List[str]
    n_pos: int               # real (unpadded) positions


def collate(examples: Sequence[ItemExample], mask_id: int = MASK_ID) -> Batch:
    """Right-pad to the longest example (a multiple of 32).  Pad rows attend only to themselves (no all-masked
    softmax row, hence no NaN); pad keys are invisible to every real row."""
    B = len(examples)
    Lmax = max(int(e.ids.numel()) for e in examples)
    Lmax = -(-Lmax // BLOCK) * BLOCK
    ids = torch.full((B, Lmax), int(mask_id), dtype=torch.long)
    pos = torch.zeros(B, Lmax, dtype=torch.long)
    attn = torch.zeros(B, 1, Lmax, Lmax, dtype=torch.bool)
    rows = torch.zeros(B, Lmax, dtype=torch.bool)
    targets = torch.full((B, Lmax), IGNORE, dtype=torch.long)
    kinds = torch.full((B, Lmax), -1, dtype=torch.int8)
    eye = torch.eye(Lmax, dtype=torch.bool)
    for i, e in enumerate(examples):
        L = int(e.ids.numel())
        ids[i, :L] = e.ids; pos[i, :L] = e.pos; rows[i, :L] = e.rows; targets[i, :L] = e.targets
        kinds[i, :L] = e.kinds
        a = eye.clone()
        a[:L, :L] = attention_mask(e)
        attn[i, 0] = a
        if L < Lmax:
            pos[i, L:] = torch.arange(L, Lmax)
    return Batch(ids=ids, pos=pos, attn=attn, rows=rows, targets=targets, kinds=kinds,
                 pids=[e.pid for e in examples], domains=[e.domain for e in examples],
                 n_pos=int(sum(int(e.ids.numel()) for e in examples)))


# ---------------------------------------------------------------------------
# the one-pass plan: steps -> ranks -> micro-batches (deterministic, identical on every rank)
# ---------------------------------------------------------------------------
def pass_order(train_pids: Sequence[str], seed: int, epoch: int = 0) -> List[str]:
    rng = np.random.default_rng(_seed(seed, "drafter_lora_order", epoch) % (2 ** 32))
    pids = sorted(train_pids)
    return [pids[int(i)] for i in rng.permutation(len(pids))]


def split_step(pids: Sequence[str], lengths: Dict[str, int], world: int) -> List[List[str]]:
    """Deal one step's items to ranks in snake order of length, so every rank gets a similar token load."""
    order = sorted(pids, key=lambda p: (-lengths[p], p))
    per: List[List[str]] = [[] for _ in range(world)]
    for i, p in enumerate(order):
        r = i % (2 * world)
        per[r if r < world else 2 * world - 1 - r].append(p)
    return per


def micro_batches(pids: Sequence[str], lengths: Dict[str, int], micro_tokens: int) -> List[List[str]]:
    """Greedy packing by padded size: B * Lmax <= micro_tokens (an item longer than the budget goes alone)."""
    order = sorted(pids, key=lambda p: (-lengths[p], p))
    out: List[List[str]] = []
    cur: List[str] = []
    for p in order:
        L = -(-lengths[p] // BLOCK) * BLOCK
        lmax = max([L] + [-(-lengths[q] // BLOCK) * BLOCK for q in cur])
        if cur and (len(cur) + 1) * lmax > micro_tokens:
            out.append(cur); cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return out


@dataclass
class StepPlan:
    step: int
    pids: List[str]                      # the global step's items
    per_rank: List[List[List[str]]]      # rank -> micro-batches -> pids
    n_micro_global: int


def plan_pass(order: Sequence[str], lengths: Dict[str, int], items_per_step: int, world: int,
              micro_tokens: int) -> List[StepPlan]:
    """Exactly one pass: ceil(n / items_per_step) steps; the last one may be smaller."""
    steps: List[StepPlan] = []
    for s in range(0, len(order), items_per_step):
        chunk = list(order[s:s + items_per_step])
        per = [micro_batches(r, lengths, micro_tokens) for r in split_step(chunk, lengths, world)]
        steps.append(StepPlan(step=len(steps), pids=chunk, per_rank=per,
                              n_micro_global=sum(len(m) for m in per)))
    return steps


def iter_index_items(entries: Iterable[dict], pids: Iterable[str]) -> Iterable[dict]:
    by = {str(e["pid"]): e for e in entries}
    for p in pids:
        yield by[p]
