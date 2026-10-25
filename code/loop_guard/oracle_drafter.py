"""Stage 2 probe B: an ORACLE drafter for the full-judge decode: at every slot it proposes the vanilla 30B's own
token for that position (read from the vanilla decode records of the same question), with confidence ~1.

It has the SeqCache surface da_generate / draft_block / make_record use (bl, dev, timings, reset, prefill,
block_forward, flush, cached_length, ms_by_kind) but no model: block_forward returns one-hot logits, so the s8 first
draft and every re-draft are exactly the vanilla tokens.  Beyond the vanilla answer's end it proposes the first EOS id.
The judge, its commit rule, the KV flushes and the record format are untouched, so the records answer: how many judge
passes per block does the 0.95 rule still need when every draft is right?  Use it WITHOUT the rejected-token ban and
without the loop guard (both would mask the oracle's only token).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


class OracleSeq:
    def __init__(self, block_length: int, device, vocab_size: int, eos_ids: Sequence[int], name: str = "drafter",
                 hi: float = 30.0, bg: float = -8.0):
        self.bl = int(block_length)
        self.dev = torch.device(device)
        self.name = name
        self.V = int(vocab_size)
        self.eos = int(list(eos_ids)[0])
        self.hi, self.bg = float(hi), float(bg)
        self.timings: List[Tuple[str, float]] = []
        self._len = 0
        self._tokens: Optional[torch.Tensor] = None       # [total] long: prompt + vanilla gen ids (+ eos padding)
        self._item_idx = None

    # -- per item ------------------------------------------------------------------------------------------------
    def set_item(self, record: dict, prompt_ids: torch.Tensor) -> None:
        """`record` = the vanilla decode record of this question (keys gen_ids, prompt_tokens); prompt_ids [1, L]."""
        assert int(record["prompt_tokens"]) == int(prompt_ids.shape[1]), \
            f"prompt length differs from the vanilla record: {record['prompt_tokens']} vs {prompt_ids.shape[1]}"
        gen = torch.tensor([int(t) for t in record["gen_ids"]], dtype=torch.long)
        self._tokens = torch.cat([prompt_ids[0].to("cpu", torch.long), gen]).to(self.dev)
        self._item_idx = record.get("idx")

    # -- SeqCache surface ----------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.timings = []
        self._len = 0

    def cached_length(self) -> int:
        return self._len

    def ms_by_kind(self) -> dict:
        out: Dict[str, float] = {}
        for k, ms in self.timings:
            out[k] = out.get(k, 0.0) + ms
        return out

    def prefill(self, prompt_ids: torch.Tensor, extra=None) -> int:
        assert self._tokens is not None, "set_item() before prefill()"
        pl = (int(prompt_ids.shape[1]) // self.bl) * self.bl
        self._len = pl
        self.timings.append(("prefill", 0.0))
        return pl

    def block_forward(self, block_ids: torch.Tensor, lo: int, store_kv: bool, kind: str, extra=None) -> torch.Tensor:
        assert block_ids.shape == (1, self.bl), block_ids.shape
        hi = lo + self.bl
        tok = torch.full((self.bl,), self.eos, dtype=torch.long, device=self.dev)
        n = min(hi, int(self._tokens.numel())) - lo
        if n > 0:
            tok[:n] = self._tokens[lo:lo + n]
        logits = torch.full((1, self.bl, self.V), self.bg, dtype=torch.float32, device=self.dev)
        logits[0, torch.arange(self.bl, device=self.dev), tok] = self.hi
        self.timings.append((kind, 0.0))
        if store_kv:
            self._len = hi
        return logits

    def flush(self, final_ids: torch.Tensor, lo: int) -> None:
        self._len = lo + self.bl
