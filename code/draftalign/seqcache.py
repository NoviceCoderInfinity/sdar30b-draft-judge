"""One model + ONE persistent DynamicCache per item, on the block grid of decode.py.

Mirrors `sdar_harness.decode.block_diffusion_generate` (decode.py:165-227) exactly:
  * grid: block b spans absolute positions [lo, lo+bl); position_ids are absolute;
  * prefill: the `prompt_length // bl` full prompt blocks are pushed with the block-tril
    0/1 mask and store_kv=True (decode.py:185-196); the ragged prompt tail is NOT prefilled --
    it sits in the first decode block as fixed (non-valid) tokens;
  * block forwards: all-ones 0/1 mask [1, bl, hi] (== attn[:, lo:hi, :hi] of the block-tril,
    which is all ones for the current block) -> the remote code takes its unmasked
    bidirectional path over [cache + block]; store_kv=False reads the cache without writing,
    store_kv=True appends the block (modeling_sdar_moe.py:204-216, modeling_sdar.py:263-273).

Every forward is timed with torch.cuda.synchronize (ms) and counted by `kind`.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


def block_tril_mask(n_blocks: int, bl: int, device) -> torch.Tensor:
    """[1, n_blocks*bl, n_blocks*bl] 0/1 float block-causal mask (decode.py:176-179)."""
    bm = torch.tril(torch.ones(n_blocks, n_blocks, device=device))
    return bm.repeat_interleave(bl, dim=0).repeat_interleave(bl, dim=1).unsqueeze(0)


class SeqCache:
    """Persistent cache for one item; `reset()` between items."""

    def __init__(self, model, block_length: int, name: str):
        self.model = model
        self.bl = int(block_length)
        self.name = name
        self.dev = model.device
        self.reset()

    # ---- bookkeeping ----------------------------------------------------
    def reset(self):
        self.cache = DynamicCache()
        self.prefill_length = 0
        self.timings: List[Tuple[str, float]] = []      # (kind, ms) in call order
        self.n_by_kind: Counter = Counter()

    def cached_length(self) -> int:
        return int(self.cache.get_seq_length())

    def ms_by_kind(self) -> dict:
        out = {}
        for k, ms in self.timings:
            out[k] = out.get(k, 0.0) + ms
        return out

    def _sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    @torch.no_grad()
    def _timed(self, kind: str, **kw):
        self._sync(); t0 = time.perf_counter()
        out = self.model(**kw)
        self._sync(); ms = (time.perf_counter() - t0) * 1e3
        self.timings.append((kind, ms))
        self.n_by_kind[kind] += 1
        return out

    # ---- forwards ---------------------------------------------------------
    @staticmethod
    def _ids_or_embeds(ids: torch.Tensor, extra) -> dict:
        """`{"input_ids": ids}` -- or `{"inputs_embeds": ...}` when a caller (the draft-flag path of
        `da_decode`) has pre-built the embeddings; the remote forward takes exactly one of the two
        (modeling_sdar_moe.py:577-579) and treats them identically from there on (:592-593)."""
        if extra and extra.get("inputs_embeds") is not None:
            return {"inputs_embeds": extra["inputs_embeds"]}
        return {"input_ids": ids}

    def prefill(self, prompt_ids: torch.Tensor, extra: Optional[dict] = None) -> int:
        """Push the full prompt blocks (decode.py:185-196). Returns prefill_length."""
        assert prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1
        assert self.cached_length() == 0, "prefill on a non-empty cache"
        L = int(prompt_ids.shape[1])
        prefill_blocks = L // self.bl
        pl = prefill_blocks * self.bl
        self.prefill_length = pl
        if pl > 0:
            attn = block_tril_mask(prefill_blocks, self.bl, self.dev)
            pos = torch.arange(pl, dtype=torch.long, device=self.dev).unsqueeze(0)
            self._timed("prefill", **self._ids_or_embeds(prompt_ids[:, :pl].to(self.dev), extra),
                        attention_mask=attn,
                        position_ids=pos, past_key_values=self.cache, use_cache=True, store_kv=True)
            assert self.cached_length() == pl, f"{self.name}: prefill stored {self.cached_length()} != {pl}"
        return pl

    def block_forward(self, block_ids: torch.Tensor, lo: int, store_kv: bool, kind: str,
                      extra: Optional[dict] = None) -> torch.Tensor:
        """One forward of block [lo, lo+bl) against the prefix cache. Returns logits [1, bl, V]."""
        if block_ids.dim() == 1:
            block_ids = block_ids.unsqueeze(0)
        assert block_ids.shape == (1, self.bl), block_ids.shape
        assert self.cached_length() == lo, f"{self.name}: cache holds {self.cached_length()} positions, block starts at {lo}"
        hi = lo + self.bl
        attn = torch.ones(1, self.bl, hi, device=self.dev)
        pos = torch.arange(lo, hi, dtype=torch.long, device=self.dev).unsqueeze(0)
        out = self._timed(kind, **self._ids_or_embeds(block_ids.to(self.dev), extra),
                          attention_mask=attn, position_ids=pos,
                          past_key_values=self.cache, use_cache=True, store_kv=bool(store_kv))
        if store_kv:
            assert self.cached_length() == hi, f"{self.name}: flush stored {self.cached_length()} != {hi}"
        return out.logits

    def flush(self, final_ids: torch.Tensor, lo: int) -> None:
        """Teacher-force the block's FINAL tokens into the cache (decode.py:219-226 kvflush)."""
        self.block_forward(final_ids, lo, store_kv=True, kind="flush")
