"""Phase 2 drafter registry and the wrapper that lets a released diffusion LM draft blocks for the 30B MoE.

Every drafter is used AS RELEASED (Notion, Phase 2): pinned repo + revision, no conversion, no training. The
wrapper gives each one the interface `draftalign.drafter.draft_block` and `da_decode.da_generate` already use for
SDAR-1.7B (`SeqCache`): `bl`, `dev`, `timings`, `n_by_kind`, `reset()`, `prefill(prompt_ids)`,
`block_forward(block_ids, lo, store_kv, kind)` -> logits [1, bl, V] in the MoE's token space, `flush(final, lo)`.
So the decode rule, the judge and the pool are untouched; only the drafter's forward differs.

What differs between drafters, and how the wrapper handles it:

  vocabulary   The MoE speaks Qwen3 ids (mask 151669). Qwen3-vocabulary drafters (S0, F2, S1, S6) share every id.
               Qwen2.5-vocabulary drafters (F1, S2) share ids 0..151664 (checked token by token against the MoE's
               tokenizer at load), have their own mask at 151665, and lack Qwen3's <tool_response>,
               </tool_response>, <think>, </think> (151665..151668): the MoE's [MASK] is remapped to the drafter's
               mask on input, an MoE-only id on input becomes a placeholder (counted in `n_unmapped`), and the
               drafter's logits for ids the MoE does not share are -inf on output (so it can never draft them).
  attention    block-causal with a KV cache (SDAR, Fast-dLLM v2, dLLM BD3LM): the prompt's full blocks are
               prefilled, each block attends to the cache + itself, the flush stores the final block -- the same
               grid as the MoE. Full-sequence bidirectional (CoDA): no cache is valid, so every forward recomputes
               [prompt + committed blocks + current block] (its wall clock is higher; its charged cost is not).
  logit shift  SDAR and BD3LM predict slot j from position j. Fast-dLLM v2 and CoDA keep the AR shift: slot j is
               predicted from position j-1, and slot 0 of a block from the previous block's last position (for
               Fast-dLLM v2 that logit comes from the flush of the previous block, exactly as its own generate()).
  cost         charged per forward at the parameters a forward multiplies by -- the body plus the lm_head
               matmul, not the input embedding lookup -- / 4,718,592 (one 30B expert): the repository's convention
               (S0 = 1,720,574,976 / 4,718,592 = 364.64). Counted from the safetensors: a tied model stores the
               embedding once, so its head is added back (F1); the dLLM checkpoints store both (S1, S2).

Keys: S0 (baseline), F1, F2, S1, S2, S6 -- the Notion tables' recommended and runnable picks. F3, S3, S4, S5, S7
are listed with why they are not wired (tokenizer bridge, block 4, full-sequence code-only), never silently run.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

EXPERT_PARAMS = 3 * 2048 * 768                      # 4,718,592: one expert of the 30B MoE
S0_COST = 364.64                                    # compare_table.py:23, the published constant

# repo, revision (pinned 2026-09-29 from the Hub API), family, compute_params = body + lm_head (no input embedding),
# mask_token = the drafter tokenizer's mask string (F1's tokenizer has no `mask_token` attribute)
REGISTRY: Dict[str, dict] = {
    "S0": {"name": "SDAR-1.7B-Chat-b32", "repo": "JetLM/SDAR-1.7B-Chat-b32",
           "revision": "2c83c05d84be90a01c72443b2123ebf0376f6fd4", "family": "sdar", "models_key": "d32",
           "vocab": "qwen3", "mask_id": 151669, "mask_token": "<|MASK|>", "shift": False, "compute_params": 1720574976,
           "role": "baseline"},
    "F1": {"name": "Fast_dLLM_v2_1.5B", "repo": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
           "revision": "25093b6f63300adfd57f72145083c8a528fe4f16", "family": "fastdllm_v2",
           "vocab": "qwen2.5", "mask_id": 151665, "mask_token": "|<MASK>|", "shift": True, "compute_params": 1543714304,
           "role": "family: Qwen2.5 base, SFT on synthetic Nemotron data"},
    "F2": {"name": "CoDA-v0-Instruct", "repo": "Salesforce/CoDA-v0-Instruct",
           "revision": "f2d40ce1282766e2c9fbf5ca62c611817164a246", "family": "coda",
           "vocab": "qwen3", "mask_id": 151669, "mask_token": "<|mask|>", "shift": True, "compute_params": 1720574976,
           "role": "family (positive control): Qwen3-1.7B base, public corpus, full-sequence diffusion"},
    "S1": {"name": "Qwen3-0.6B-diffusion-bd3lm-v0.1", "repo": "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
           "revision": "60f77bba67ba94231c16f5fb29bc66c500b038f1", "family": "a2d",
           "vocab": "qwen3", "mask_id": 151669, "mask_token": "<|mask|>", "shift": False, "compute_params": 596049920,
           "role": "size: 0.6B, same tokenizer"},
    "S2": {"name": "Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1",
           "repo": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1",
           "revision": "4bcf3a00ba6ec28d9cf9c3ebc4a5a5dc01c6023c", "family": "a2d",
           "vocab": "qwen2.5", "mask_id": 151665, "mask_token": "<|mask|>", "shift": False, "compute_params": 494032768,
           "role": "size: 0.5B code drafter (HumanEval + MBPP only)", "sets": ["humaneval", "mbpp"]},
    "S6": {"name": "SDAR-4B-Chat-b32", "repo": "JetLM/SDAR-4B-Chat-b32",
           "revision": "d4dbd3439d089b843094f9b66e76088f0c2545dd", "family": "sdar",
           "vocab": "qwen3", "mask_id": 151669, "mask_token": "<|MASK|>", "shift": False, "compute_params": 4022468096,
           "role": "size: 4B, the one same-corpus size point (over the 3B limit: ceiling only)"},
}
NOT_WIRED = {
    "F3": "JetLM/MDM-1.7B: 128k InternLM2 vocabulary, needs a tokenizer bridge (several days)",
    "S3": "fredzzp/open-dcoder-0.5B: full-sequence code-only pretraining; not wired yet (same path as F2 if wanted)",
    "S4": "nieshen/SMDM: Llama-2 vocabulary, needs a tokenizer bridge and the SMDM code",
    "S5": "OpenGVLab/SDLM-3B-D4: block 4, not the MoE's block-32 grid",
    "S7": "nvidia/Nemotron-Labs-Diffusion-3B: Mistral vocabulary, needs a tokenizer bridge",
}


def is_key(key: Optional[str]) -> bool:
    return key in REGISTRY


def spec(key: str) -> dict:
    if key in NOT_WIRED:
        raise KeyError(f"{key} is not wired: {NOT_WIRED[key]}")
    return REGISTRY[key]


def cost_per_forward(key: Optional[str]) -> float:
    """Expert-equivalents charged per drafter forward: (body + lm_head parameters) / one expert."""
    if key is None:
        return 0.0
    if key in ("S0", "d32"):
        return S0_COST
    return round(spec(key)["compute_params"] / EXPERT_PARAMS, 2)


def cell_tag(key: str) -> str:
    """Suffix on the cell name; S0 adds nothing (it IS the d32 cells), every other drafter `_dr<key><rev6>`."""
    if key == "S0":
        return ""
    return f"_dr{key}{spec(key)['revision'][:6]}"


def cache_dir() -> str:
    import research_protocol as P
    return P.MODELS["b32"]["cache_dir"]


# ---------------------------------------------------------------------------------------------------------------
# vocabulary map
# ---------------------------------------------------------------------------------------------------------------
class VocabMap:
    """MoE ids <-> drafter ids. Built by comparing the two tokenizers token by token."""

    PLACEHOLDER = "<|fim_pad|>"

    def __init__(self, moe_tok, dr_tok, moe_mask: int, dr_mask: int, v_out: int, device):
        import torch
        n_moe = len(moe_tok)
        moe_tokens = moe_tok.convert_ids_to_tokens(list(range(n_moe)))
        dr_vocab = dr_tok.get_vocab()
        in_map = torch.full((n_moe,), -1, dtype=torch.long)
        shared_moe, shared_dr = [], []
        for i, t in enumerate(moe_tokens):
            if i == moe_mask:
                continue
            j = dr_vocab.get(t)
            if j is not None and j != dr_mask:
                in_map[i] = j
                shared_moe.append(i); shared_dr.append(j)
        in_map[moe_mask] = dr_mask
        ph = dr_vocab.get(self.PLACEHOLDER)
        assert ph is not None, f"drafter tokenizer has no {self.PLACEHOLDER}"
        self.unmapped_ids = [i for i in range(n_moe) if int(in_map[i]) < 0]
        in_map[in_map < 0] = ph
        self.identity = bool(len(shared_moe) == n_moe - 1 and all(a == b for a, b in zip(shared_moe, shared_dr))
                             and moe_mask == dr_mask)
        self.in_map = in_map.to(device)
        self.out_moe = torch.tensor(shared_moe, dtype=torch.long, device=device)
        self.out_dr = torch.tensor(shared_dr, dtype=torch.long, device=device)
        self.v_out = int(v_out)
        self.n_shared = len(shared_moe)
        self.first_diff = next((i for i, (a, b) in enumerate(zip(shared_moe, shared_dr)) if a != b), None)

    def to_drafter(self, ids):
        return self.in_map[ids.clamp(max=self.in_map.numel() - 1)]

    def unmapped(self, ids) -> int:
        if not self.unmapped_ids:
            return 0
        import torch
        return int(torch.isin(ids, torch.tensor(self.unmapped_ids, device=ids.device)).sum())

    def to_moe_logits(self, lg):
        """[.., V_dr] drafter logits -> [.., v_out] MoE-space logits (-inf where the MoE id is not shared)."""
        import torch
        if self.identity and lg.shape[-1] == self.v_out:
            return lg
        out = torch.full((*lg.shape[:-1], self.v_out), float("-inf"), dtype=lg.dtype, device=lg.device)
        out[..., self.out_moe] = lg[..., self.out_dr]
        return out


# ---------------------------------------------------------------------------------------------------------------
# the wrappers
# ---------------------------------------------------------------------------------------------------------------
class _Base:
    """Bookkeeping shared with SeqCache: timings per forward, counts per kind, non-finite logits."""

    def __init__(self, model, block_length: int, vmap: Optional[VocabMap], name: str = "drafter"):
        self.model = model
        self.bl = int(block_length)
        self.name = name
        self.dev = next(model.parameters()).device
        self.vmap = vmap
        self.n_nonfinite = 0
        self.n_unmapped = 0
        self.reset()

    def reset(self):
        self.timings: List[Tuple[str, float]] = []
        self.n_by_kind: Counter = Counter()
        self._reset_state()

    def _reset_state(self):
        raise NotImplementedError

    def _sync(self):
        import torch
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def _timed(self, kind: str, fn):
        self._sync(); t0 = time.perf_counter()
        out = fn()
        self._sync()
        self.timings.append((kind, (time.perf_counter() - t0) * 1e3))
        self.n_by_kind[kind] += 1
        return out

    def ms_by_kind(self) -> dict:
        out = {}
        for k, ms in self.timings:
            out[k] = out.get(k, 0.0) + ms
        return out

    def _in(self, ids):
        if self.vmap is None:
            return ids
        self.n_unmapped += self.vmap.unmapped(ids)
        return self.vmap.to_drafter(ids)

    def _out(self, lg):
        import torch
        self.n_nonfinite += int((~torch.isfinite(lg)).any(-1).sum())
        return lg if self.vmap is None else self.vmap.to_moe_logits(lg)

    def flush(self, final_ids, lo: int) -> None:
        self.block_forward(final_ids, lo, store_kv=True, kind="flush")


class CachedBlockCausal(_Base):
    """HF DynamicCache + 4D bool block mask (dLLM A2D / BD3LM models: no logit shift). Non-storing forwards are
    cropped back to the prefix, so the cache always holds exactly the committed positions."""

    def _reset_state(self):
        from transformers.cache_utils import DynamicCache
        try:
            self.cache = DynamicCache(config=self.model.config)
        except TypeError:
            self.cache = DynamicCache()
        self.prefill_length = 0

    def cached_length(self) -> int:
        return int(self.cache.get_seq_length())

    def prefill(self, prompt_ids, extra=None) -> int:
        import torch
        L = int(prompt_ids.shape[1])
        pl = (L // self.bl) * self.bl
        self.prefill_length = pl
        if pl > 0:
            nb = pl // self.bl
            bm = torch.tril(torch.ones(nb, nb, dtype=torch.bool, device=self.dev))
            attn = bm.repeat_interleave(self.bl, 0).repeat_interleave(self.bl, 1)[None, None]
            pos = torch.arange(pl, device=self.dev)[None]
            ids = self._in(prompt_ids[:, :pl].to(self.dev))
            self._timed("prefill", lambda: self.model(input_ids=ids, attention_mask=attn, position_ids=pos,
                                                      past_key_values=self.cache, use_cache=True))
            assert self.cached_length() == pl
        return pl

    def block_forward(self, block_ids, lo: int, store_kv: bool, kind: str, extra=None):
        import torch
        if block_ids.dim() == 1:
            block_ids = block_ids.unsqueeze(0)
        assert self.cached_length() == lo, f"{self.name}: cache {self.cached_length()} != block start {lo}"
        hi = lo + self.bl
        attn = torch.ones(1, 1, self.bl, hi, dtype=torch.bool, device=self.dev)
        pos = torch.arange(lo, hi, device=self.dev)[None]
        ids = self._in(block_ids.to(self.dev))
        out = self._timed(kind, lambda: self.model(input_ids=ids, attention_mask=attn, position_ids=pos,
                                                   past_key_values=self.cache, use_cache=True))
        if store_kv:
            assert self.cached_length() == hi
        else:
            self.cache.crop(lo)
        return self._out(out.logits)


class FastDLLMv2Seq(_Base):
    """Fast-dLLM v2: its own cached block-causal forward (`update_past_key_values`), AR-shifted logits. The logit
    for slot 0 of a block is the previous block's last position, read at the flush (or the prefill)."""

    def _reset_state(self):
        from transformers.cache_utils import DynamicCache
        self.cache = DynamicCache()
        self.prefill_length = 0
        self.next0 = None

    def cached_length(self) -> int:
        return int(self.cache.get_seq_length())

    def prefill(self, prompt_ids, extra=None) -> int:
        L = int(prompt_ids.shape[1])
        pl = (L // self.bl) * self.bl
        self.prefill_length = pl
        self.next0 = None
        if pl > 0:
            ids = self._in(prompt_ids[:, :pl].to(self.dev))
            out = self._timed("prefill", lambda: self.model(input_ids=ids, use_cache=True, past_key_values=self.cache,
                                                            update_past_key_values=True, block_size=self.bl))
            self.next0 = out.logits[:, -1:, :]
            assert self.cached_length() == pl
        return pl

    def block_forward(self, block_ids, lo: int, store_kv: bool, kind: str, extra=None):
        import torch
        if block_ids.dim() == 1:
            block_ids = block_ids.unsqueeze(0)
        assert self.cached_length() == lo, f"{self.name}: cache {self.cached_length()} != block start {lo}"
        ids = self._in(block_ids.to(self.dev))
        out = self._timed(kind, lambda: self.model(input_ids=ids, use_cache=True, past_key_values=self.cache,
                                                   update_past_key_values=bool(store_kv), block_size=self.bl))
        L = out.logits
        first = self.next0 if self.next0 is not None else L[:, :1, :]
        shifted = torch.cat([first, L[:, :-1, :]], dim=1)
        if store_kv:
            assert self.cached_length() == lo + self.bl
            self.next0 = L[:, -1:, :]
        return self._out(shifted)


class FullRecomputeShifted(_Base):
    """Full-sequence bidirectional drafters (CoDA): no valid KV cache, so every forward runs [committed prefix +
    current block]; AR-shifted logits (slot j from position j-1). `trailing_masks` appends that many [MASK] after
    the block (0 = the block is the end of the sequence)."""

    def __init__(self, *a, trailing_masks: int = 0, mask_id: int = 151669, **kw):
        self.trailing, self.mask_id = int(trailing_masks), int(mask_id)
        super().__init__(*a, **kw)

    def _reset_state(self):
        self.prefix = None
        self.prefill_length = 0

    def cached_length(self) -> int:
        return 0 if self.prefix is None else int(self.prefix.shape[1])

    def _logits(self, ids):
        out = self.model(input_ids=ids)
        return out[0] if isinstance(out, tuple) else out.logits

    def prefill(self, prompt_ids, extra=None) -> int:
        L = int(prompt_ids.shape[1])
        pl = (L // self.bl) * self.bl
        self.prefill_length = pl
        self.prefix = self._in(prompt_ids[:, :pl].to(self.dev))
        return pl

    def block_forward(self, block_ids, lo: int, store_kv: bool, kind: str, extra=None):
        import torch
        if block_ids.dim() == 1:
            block_ids = block_ids.unsqueeze(0)
        assert self.cached_length() == lo, f"{self.name}: prefix {self.cached_length()} != block start {lo}"
        blk = self._in(block_ids.to(self.dev))
        if store_kv:
            # the flush: no forward (there is no cache to fill), the block just becomes prefix -- so no timing row
            # either, and the charged drafter forwards are the draft forwards only
            self.prefix = torch.cat([self.prefix, blk], dim=1)
            return None
        parts = [self.prefix, blk]
        if self.trailing:
            parts.append(torch.full((1, self.trailing), self._in(torch.tensor([[self.mask_id]], device=self.dev))[0, 0].item(),
                                    dtype=torch.long, device=self.dev))
        ids = torch.cat(parts, dim=1)
        L = self._timed(kind, lambda: self._logits(ids))
        s = max(lo - 1, 0)
        sh = L[:, s:s + self.bl, :] if lo > 0 else torch.cat([L[:, :1, :], L[:, :self.bl - 1, :]], dim=1)
        return self._out(sh)


class SdarSeq:
    """The SDAR drafters run behind the repository's own SeqCache (identical to the round-3 path)."""

    def __init__(self, model, desc: str):
        self.model, self._desc = model, desc

    def as_seq(self, block: int):
        from draftalign.seqcache import SeqCache
        s = SeqCache(self.model, block, "drafter")
        s.n_nonfinite = 0
        return s

    def describe(self) -> str:
        return self._desc


class Drafter:
    """What `load` returns: `as_seq(block)` builds the per-decode sequence object."""

    def __init__(self, key: str, model, vmap: Optional[VocabMap], seq_cls, desc: str, **seq_kw):
        self.key, self.model, self.vmap, self.seq_cls, self._desc, self.seq_kw = key, model, vmap, seq_cls, desc, seq_kw

    def as_seq(self, block: int):
        return self.seq_cls(self.model, block, self.vmap, **self.seq_kw)

    def describe(self) -> str:
        return self._desc


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def _moe_tokenizer():
    import research_protocol as P
    from transformers import AutoTokenizer
    m = P.MODELS["b32"]
    return AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], cache_dir=m["cache_dir"],
                                         trust_remote_code=True, local_files_only=True)


def download(key: str) -> str:
    """Network step for setup_p2.sh: the pinned snapshot into the project cache."""
    from huggingface_hub import snapshot_download
    s = spec(key)
    return snapshot_download(s["repo"], revision=s["revision"], cache_dir=cache_dir())


def load(key: str, device: str, moe_tok=None) -> "Drafter | SdarSeq":
    import torch
    import research_protocol as P
    from phase2 import tf_compat
    tf_compat.apply()
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    s = spec(key)
    kw = {"revision": s["revision"], "cache_dir": cache_dir(), "trust_remote_code": True, "local_files_only": True}
    if s["family"] == "sdar":
        if s.get("models_key"):
            tok, model = P.load_sdar(s["models_key"], device)
        else:
            tok = AutoTokenizer.from_pretrained(s["repo"], **kw)
            model = AutoModelForCausalLM.from_pretrained(s["repo"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                         device_map=device, **kw).eval()
        assert tok(tok.mask_token)["input_ids"][0] == P.MASK_ID, f"{key}: mask id differs from the MoE's"
        return SdarSeq(model, f"{s['name']} @ {s['revision'][:8]} (SeqCache, cost {cost_per_forward(key)}/fwd)")
    moe_tok = moe_tok or _moe_tokenizer()
    tok = AutoTokenizer.from_pretrained(s["repo"], **kw)
    if s["family"] == "fastdllm_v2":
        model = AutoModelForCausalLM.from_pretrained(s["repo"], torch_dtype=torch.bfloat16, device_map=device, **kw).eval()
        seq_cls, extra = FastDLLMv2Seq, {}
    elif s["family"] == "a2d":
        model = AutoModelForMaskedLM.from_pretrained(s["repo"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map=device, **kw).eval()
        # the dLLM remote code (written for transformers >= 4.56) passes `past_key_values=` to each decoder layer;
        # 4.54's Qwen3DecoderLayer reads `past_key_value=` and drops the plural into **kwargs, so the cache would
        # never be read or written (no attention to the prefix at all). Rename it on the way in.
        def _kv_name(_mod, args, kwargs):
            if "past_key_values" in kwargs and "past_key_value" not in kwargs:
                kwargs["past_key_value"] = kwargs.pop("past_key_values")
            return args, kwargs
        for layer in model.model.layers:
            layer.register_forward_pre_hook(_kv_name, with_kwargs=True)
        seq_cls, extra = CachedBlockCausal, {}
    elif s["family"] == "coda":
        cfg = AutoConfig.from_pretrained(s["repo"], **kw)
        cfg.attention_kernel = os.environ.get("P2_CODA_ATTN", "flash_attention")   # SDPA flash: needs bf16
        # the checkpoint stores its own lm_head, but config.json has no tie_word_embeddings and CoDAConfig defaults
        # it to True: tied, the trained head is replaced by the input embedding and the drafts are noise (~1% agreement)
        cfg.tie_word_embeddings = False
        model = AutoModel.from_pretrained(s["repo"], config=cfg, torch_dtype=torch.bfloat16, device_map=device,
                                          attn_implementation="eager", **kw).eval()
        seq_cls, extra = FullRecomputeShifted, {"trailing_masks": int(os.environ.get("P2_CODA_TRAILING", "0")),
                                                "mask_id": P.MASK_ID}
    else:
        raise KeyError(s["family"])
    got_mask = tok.convert_tokens_to_ids(s["mask_token"])
    assert got_mask == s["mask_id"], f"{key}: tokenizer id of {s['mask_token']} is {got_mask}, registry {s['mask_id']}"
    v_out = int(model.get_output_embeddings().weight.shape[0]) if model.get_output_embeddings() is not None \
        else int(model.config.vocab_size)
    vmap = VocabMap(moe_tok, tok, P.MASK_ID, s["mask_id"], v_out, next(model.parameters()).device)
    assert vmap.first_diff is None, f"{key}: shared tokens sit at different ids (first at MoE id {vmap.first_diff})"
    desc = (f"{s['name']} @ {s['revision'][:8]} ({s['family']}, shift={s['shift']}, vocab shared {vmap.n_shared}, "
            f"MoE-only ids {len(vmap.unmapped_ids)}, identity={vmap.identity}, cost {cost_per_forward(key)}/fwd)")
    return Drafter(key, model, None if vmap.identity else vmap, seq_cls, desc, name=f"drafter-{key}", **extra)
